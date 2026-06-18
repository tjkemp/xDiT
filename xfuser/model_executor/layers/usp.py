# This file implements USP with torch version >= '2.5.0'
import os
import torch
import torch.distributed as dist
import functools

import torch.distributed._functional_collectives as ft_c

from torch.distributed.tensor.experimental._attention import _templated_ring_attention
import xfuser.envs as envs

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_rank,
    get_runtime_state,
)

from packaging.version import parse
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.attention_backend import ATTENTION_FUNCTION_REGISTRY

_FP8_LOG_SCALES = bool(os.environ.get("XFUSER_FP8_LOG_SCALES"))
_FP8_NCCL_NEEDS_VIEW = parse(torch.__version__).release < parse("2.11.0").release
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)


def ring_attn(attention_function, query, key, value, dropout_p=0.0, is_causal=False, joint_attn_kwargs=None, attention_kwargs=None):
    kwargs = {
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "joint_attn_kwargs": joint_attn_kwargs,
        "attention_kwargs": attention_kwargs,
    }
    if parse(torch.__version__).release >= parse("2.6.0").release:
        from torch.distributed.tensor.experimental._attention import _cp_options
        _cp_options.enable_load_balance = False
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            1,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    return out


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    x_dtype = x.dtype
    x = x.flatten()
    # NCCL does not support FP8 collectives before PyTorch 2.11, view as uint8 (same width) for the transfer.
    if _FP8_NCCL_NEEDS_VIEW and x_dtype in _FP8_DTYPES:
        x = x.view(torch.uint8)
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    x = x.view(x_dtype).reshape(x_shape)
    return x


def _ft_c_input_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)
    return x


def _per_tensor_quant(x: torch.Tensor, scale_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x to FP8 using a fixed pre-allocated scale tensor. Returns (x_fp8, descale)."""
    import aiter
    fp8_dtype = aiter.dtypes.fp8
    return aiter.per_tensor_quant(x, scale=scale_t, quant_dtype=fp8_dtype, dtypeMax=torch.finfo(fp8_dtype).max)


_FP8_COMMS_SAFETY_FACTOR = 0.85  # leave 15% headroom above observed amax


def _fp8_comms_input_all_to_all(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple:
    """Quantize Q/K/V to FP8 using per-layer per-tensor scales and run interleaved input all-to-alls.

    Returns (query, key, value, attn_kwargs_update, (q_scale_t, k_scale_t, v_scale_t), qkv_amaxes).
    qkv_amaxes is (q_amax, k_amax, v_amax) when XFUSER_FP8_LOG_SCALES is set, else None.
    """
    fp8_comms = get_runtime_state().fp8_comms
    layer_idx = fp8_comms.call_counter
    fp8_comms.call_counter += 1

    if hasattr(fp8_comms, '_fixed_scale'):
        # fixed scale: create tensors once on first call, reuse for all layers
        if fp8_comms.q_scales is None:
            s = torch.tensor(fp8_comms._fixed_scale, dtype=torch.float32, device=query.device)
            fp8_comms.q_scales = [s]
            fp8_comms.k_scales = [s]
            fp8_comms.v_scales = [s]
        q_scale_t = fp8_comms.q_scales[0]
        k_scale_t = fp8_comms.k_scales[0]
        v_scale_t = fp8_comms.v_scales[0]
    else:
        q_scale_t = fp8_comms.q_scales[layer_idx]
        k_scale_t = fp8_comms.k_scales[layer_idx]
        v_scale_t = fp8_comms.v_scales[layer_idx]

    qkv_amaxes = (
        (query.abs().amax().item(), key.abs().amax().item(), value.abs().amax().item())
        if _FP8_LOG_SCALES else None
    )

    q_fp8, q_descale = _per_tensor_quant(query, q_scale_t)
    query = _ft_c_input_all_to_all(q_fp8)
    k_fp8, k_descale = _per_tensor_quant(key, k_scale_t)
    key = _ft_c_input_all_to_all(k_fp8)
    v_fp8, v_descale = _per_tensor_quant(value, v_scale_t)
    value = _ft_c_input_all_to_all(v_fp8)

    attn_kwargs_update = {
        "pre_quantized": True,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
    }
    return query, key, value, attn_kwargs_update, (q_scale_t, k_scale_t, v_scale_t), qkv_amaxes


def _fp8_comms_output_all_to_all(out: torch.Tensor, v_scale_t: torch.Tensor) -> torch.Tensor:
    """Quantize attention output to FP8, run output all-to-all, dequantize back."""
    out_dtype = out.dtype
    if out.dtype not in _FP8_DTYPES:
        out_fp8, out_descale = _per_tensor_quant(out, v_scale_t)
    else:
        out_fp8, out_descale = out, v_scale_t
    return (_ft_c_output_all_to_all(out_fp8).float() * out_descale).to(out_dtype)


def _fp8_comms_finalize_calibration(device: torch.device):
    """All-reduce per-layer per-tensor amaxes and compute separate q/k/v scales per layer."""
    fp8_comms = get_runtime_state().fp8_comms
    if not fp8_comms or not fp8_comms.layer_amaxes:
        return
    from xfuser.core.distributed.attention_backend import AITER_FP8_DTYPE
    dtype_max = torch.finfo(AITER_FP8_DTYPE).max
    n_layers = len(fp8_comms.layer_amaxes)
    # stack into (3, n_layers): rows are q, k, v; all_reduce for global max across Ulysses ranks
    local = torch.tensor(fp8_comms.layer_amaxes, dtype=torch.float32, device=device).T  # (3, n_layers)
    dist.all_reduce(local, op=dist.ReduceOp.MAX, group=PROCESS_GROUP.ULYSSES_PG)
    scales = local / (dtype_max * _FP8_COMMS_SAFETY_FACTOR)  # (3, n_layers)
    fp8_comms.q_scales = [torch.tensor(scales[0, i].item(), dtype=torch.float32, device=device) for i in range(n_layers)]
    fp8_comms.k_scales = [torch.tensor(scales[1, i].item(), dtype=torch.float32, device=device) for i in range(n_layers)]
    fp8_comms.v_scales = [torch.tensor(scales[2, i].item(), dtype=torch.float32, device=device) for i in range(n_layers)]
    fp8_comms.static = True
    fp8_comms.layer_amaxes = None
    if dist.get_rank() == 0:
        print(f"[fp8_comms] calibrated {n_layers} layers: max_scale q={scales[0].max():.6f} k={scales[1].max():.6f} v={scales[2].max():.6f}")


def _combined_qkv_all_to_all(q, k, v):
    """Concatenate query, key, value tensors and perform a single all-to-all communication."""
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return q, k, v

    assert q.ndim == 4, f"q must have 4 dimensions, got {q.ndim}"
    b, h, s, d = q.shape
    assert h % world_size == 0, f"h must be divisible by world_size, got {h} and {world_size}"

    # [3, b, h, s, d]
    qkv = torch.stack([q, k, v], dim=0)
    # [3, b, P, h/P, s, d]
    qkv = qkv.view(3, b, world_size, h // world_size, s, d)
    # [P, 3, b, h/P, s, d]
    qkv = qkv.permute(2, 0, 1, 3, 4, 5).contiguous()

    qkv = _sdpa_all_to_all_single(qkv)

    # [3, b, h/P, P*s, d]  — reshape directly avoids the intermediate
    # contiguous copy that the separate permute+view required.
    qkv = qkv.permute(1, 2, 3, 0, 4, 5).reshape(3, b, h // world_size, -1, d)

    q, k, v = torch.unbind(qkv, dim=0)
    return q, k, v


def _ft_c_output_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x


def _preprocess_joint_tensors(joint_key, joint_value):
    """
    Preprocess the joint key and value tensors for Ulysses parallelism.
    """
    ulysses_world_size = get_ulysses_parallel_world_size()
    ulysses_rank = get_ulysses_parallel_rank()
    attn_heads_per_ulysses_rank = (
        joint_key.shape[1] // ulysses_world_size
    )
    joint_key = joint_key.transpose(1,2)
    joint_value = joint_value.transpose(1,2)
    joint_key = joint_key[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :, ].transpose(1,2)
    joint_value = joint_value[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :,
    ].transpose(1,2)
    return joint_key, joint_value

def _concat_joint_tensor(tensor, joint_tensor, joint_strategy, dim):
    """
    Concatenate the joint tensor to the main tensor based on the joint strategy.
    """
    if joint_strategy == "rear":
        tensor = torch.cat([tensor, joint_tensor], dim=dim)
    elif joint_strategy == "front":
        tensor = torch.cat([joint_tensor, tensor], dim=dim)
    else:
        raise ValueError(f"Invalid joint_strategy: {joint_strategy}")
    return tensor

def _update_and_get_kv_cache(key, value, attn_layer):
    """
    Update and get the key and value cache for pipeline parallelism.
    """
    key, value = get_cache_manager().update_and_get_kv_cache(
        new_kv=[key.transpose(1, 2), value.transpose(1, 2)],
        layer=attn_layer,
        slice_dim=1,
        layer_type="attn",
    )
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    return key, value

def _get_attention_function(backend=None):
    """
    Get the attention function based on the runtime state or from a given explicit backend.
    """
    if backend is not None:
        attention_backend = backend
    else:
        attention_backend = get_runtime_state().attention_backend
    func = ATTENTION_FUNCTION_REGISTRY.get(attention_backend, None)
    if func is None:
        raise NotImplementedError(f"Attention backend {attention_backend} not registered.")
    return concat_joint_tensors_decorator(func)

def concat_joint_tensors_decorator(func):
    """
    Decorator to handle joint tensor concatenation
    This is needed for ring attention with 'rear' joint_strategy, as it
    needs to concat the joint tensors before calling the attention function
    but only on the last step.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        query, key, value = args[0:3]
        is_causal = kwargs.get("is_causal")
        dropout_p = kwargs.get("dropout_p")
        joint_attn_kwargs = kwargs.get("joint_attn_kwargs", None)
        attention_kwargs = kwargs.get("attention_kwargs", None)

        if joint_attn_kwargs is not None:
            joint_strategy = joint_attn_kwargs.get("joint_strategy", None)
            joint_key = joint_attn_kwargs.get("joint_key", None)
            joint_value = joint_attn_kwargs.get("joint_value", None)
            step = joint_attn_kwargs.get("step", 0)
            total_steps = joint_attn_kwargs.get("total_steps", 1)
            if (joint_strategy == "front" and step == 0) or (joint_strategy == "rear" and step == total_steps - 1):
                key = _concat_joint_tensor(key, joint_key, joint_strategy, dim=2)
                value = _concat_joint_tensor(value, joint_value, joint_strategy, dim=2)
            joint_attn_kwargs["step"] = step + 1 # In place increment step

        return func(query, key, value, dropout_p=dropout_p, is_causal=is_causal, attention_kwargs=attention_kwargs)
    return wrapper


def USP(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        joint_query: torch.Tensor | None = None,
        joint_key: torch.Tensor | None = None,
        joint_value: torch.Tensor | None = None,
        joint_strategy: str | None = None,
        attn_layer=None,
        combine_qkv_a2a: bool | None = None,
        use_fp8_comms: bool = False,
        backend=None,
        attention_kwargs: dict | None = None,
    ):
    """
    Unified Sequence Parallelism (USP) attention call, supporting combinations of Ulysses and
    Ring attention. Also supports joint tensors and key-value caching for pipeline parallelism.
    Explicit backend can be provided to specify the attention backend to use.
    """
    if combine_qkv_a2a is None:
        combine_qkv_a2a = False

    attention_function = _get_attention_function(backend=backend)

    joint_attn_kwargs = None
    if joint_strategy:
        query = _concat_joint_tensor(query, joint_query, joint_strategy, dim=2)
        joint_key, joint_value = _preprocess_joint_tensors(joint_key, joint_value)
        joint_attn_kwargs = {
            "joint_value": joint_value,
            "joint_key": joint_key,
            "joint_strategy": joint_strategy,
            "step": 0,
            "total_steps": get_ring_parallel_world_size(),

        }

    qkv_scales = None  # (q_scale_t, k_scale_t, v_scale_t) after calibration
    qkv_amaxes = None
    _calibrating = False
    if get_ulysses_parallel_world_size() > 1:
        if use_fp8_comms:
            fp8_comms = get_runtime_state().fp8_comms
            if not fp8_comms.static:
                # calibration iteration: launch BF16 all-to-alls, compute amaxes while NCCL transfers
                _calibrating = True
                q_amax_t = query.abs().amax()
                query = _ft_c_input_all_to_all(query)
                k_amax_t = key.abs().amax()
                key = _ft_c_input_all_to_all(key)
                v_amax_t = value.abs().amax()
                value = _ft_c_input_all_to_all(value)
                fp8_comms.layer_amaxes.append(
                    (q_amax_t.item(), k_amax_t.item(), v_amax_t.item())
                )
                fp8_comms.call_counter += 1
            else:
                query, key, value, attn_kwargs_update, qkv_scales, qkv_amaxes = _fp8_comms_input_all_to_all(query, key, value)
                attention_kwargs = (attention_kwargs or {}) | attn_kwargs_update
        elif combine_qkv_a2a and query.shape == key.shape == value.shape:
            query, key, value = _combined_qkv_all_to_all(query, key, value)
        else:
            query = _ft_c_input_all_to_all(query)
            key = _ft_c_input_all_to_all(key)
            value = _ft_c_input_all_to_all(value)

    if attn_layer:
        key, value = _update_and_get_kv_cache(key, value, attn_layer)

    if get_sequence_parallel_world_size() == 1: # No SP
        out, _ = attention_function(query,
                                    key,
                                    value,
                                    dropout_p=dropout_p,
                                    is_causal=is_causal,
                                    joint_attn_kwargs=joint_attn_kwargs,
                                    attention_kwargs=attention_kwargs)

    elif get_ulysses_parallel_world_size() == 1: # Ring only
        out = ring_attn(attention_function,
                        query,
                        key,
                        value,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        joint_attn_kwargs=joint_attn_kwargs,
                        attention_kwargs=attention_kwargs)

    else:
        if get_ring_parallel_world_size() == 1: # Ulysses only
            out, _ = attention_function(query,
                                        key,
                                        value,
                                        dropout_p=dropout_p,
                                        is_causal=is_causal,
                                        joint_attn_kwargs=joint_attn_kwargs,
                                        attention_kwargs=attention_kwargs)
        else: # USP
            out = ring_attn(attention_function,
                            query,
                            key,
                            value,
                            dropout_p=dropout_p,
                            is_causal=is_causal,
                            joint_attn_kwargs=joint_attn_kwargs,
                            attention_kwargs=attention_kwargs)
        if use_fp8_comms and not _calibrating:
            if _FP8_LOG_SCALES and qkv_amaxes is not None:
                out_amax = out.abs().amax().item()
                rank = dist.get_rank()
                q_amax, k_amax, v_amax = qkv_amaxes
                print(f"[fp8_scales rank{rank}] q_amax={q_amax:.4f} k_amax={k_amax:.4f} v_amax={v_amax:.4f} out_amax={out_amax:.4f}")
            _, _, v_scale_t = qkv_scales
            out = _fp8_comms_output_all_to_all(out, v_scale_t)
        else:
            out = _ft_c_output_all_to_all(out)

    return out


def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        use_fp8_comms: bool = False,  # accepted for call-site uniformity with USP(), never applied
        backend=None,
        attention_kwargs=None,
    ):
    """
    Runs attention call without any parallelism.
    This can be used when the logic necessitates no Ulysses or Ring parallelism in any case.
    Explicit backend can be provided to specify the attention backend to use.
    """
    attention_function = _get_attention_function(backend=backend)
    out, _ = attention_function(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        attention_kwargs=attention_kwargs,
    )
    return out

