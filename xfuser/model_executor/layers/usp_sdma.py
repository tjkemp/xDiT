"""USP's Ulysses attention with the all-to-alls on SDMA copy engines (ROCm).

Selected with --ulysses_a2a sdma (the default is `default`: the process group's
all-to-all). --ulysses_sdma_head_groups (default 2) splits each rank's heads into groups
whose exchanges overlap the other groups' attention, and --ulysses_sdma_group_streams runs
each group's attention on a stream of its own; see xfuser/core/distributed/sdma/ulysses.py.
The options are the runtime config's `ulysses_a2a` (xfuser/config/ulysses_a2a.py); its
internal mode "packed" runs the same layout and head groups over the default all-to-all,
which tells the transport's share of a gain from the layout's.

USP hands its Ulysses calls to `run`, and the runner calls `prepare` once before
compiling. With --ulysses_sdma_overlap_qkv, `prepare` also swaps in the SDMA attention
processors that models register with `sdma_processor`: subclasses of their usual
processors that overlap the Q/K/V projection with the exchange, so the usual processors
carry no SDMA code. A call is three custom ops, opaque to torch.compile and chained by a token:
begin, one send per head group, and finish, which attends group by group with the
selected backend and returns the output. The packing of each group runs in traced code
between them, so Inductor writes it straight from the kernels producing Q/K/V; a model
can even produce each group separately (`grouped_qkv_attention`) so a group's exchange
overlaps the projection of the next.

The transport queues its copies from the host, so a CUDA graph would not replay them. The
runner compiles the transformer without CUDA graphs (`compile_mode`); its other components
keep theirs. The ops are also tagged cudagraph_unsafe, so a model compiled with graphs
some other way runs them between graph partitions.

The path covers Ulysses with joint tensors (front or rear), extra per-head inputs and
grouped-query K/V, with a dense backend. Ring attention, FP8 communication, the PipeFusion
KV cache, head balancing, sparse backends, a trailing SP pad behind rear joint tensors and
attention_kwargs it does not forward stay on the default all-to-all. A call made while a
CUDA graph is being captured anyway (by torch.cuda.graph, or a torch without graph
partitioning) exchanges the packed tensors over the default all-to-all.
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist

from xfuser.config.ulysses_a2a import UlyssesA2AConfig
from xfuser.core.attention import registry as attention_registry
from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed import (
    get_ring_parallel_world_size,
    get_runtime_state,
    get_ulysses_parallel_world_size,
    runtime_state_is_initialized,
)
from xfuser.core.distributed.sdma import ulysses
from xfuser.core.utils.runner_utils import log
from xfuser.logger import init_logger, warn_once

logger = init_logger(__name__)

# Keys of attention_kwargs that dense backends read and the ops do not carry to them
# (they carry valid_kv_len and the varlen key packing): a call that sets one stays on
# the default all-to-all. Every other key configures sparse backends, which never take this path.
_UNFORWARDED_KWARGS = ("attn_mask", "pre_quantized", "q_descale", "k_descale", "v_descale", "seqlens_k", "return_lse")


_DEFAULT = UlyssesA2AConfig()
_warned = set()
# The same compilation without CUDA graphs.
_WITHOUT_CUDAGRAPHS = {"reduce-overhead": "default", "max-autotune": "max-autotune-no-cudagraphs"}


def _config() -> UlyssesA2AConfig:
    """The runtime config's Ulysses all-to-all options (the defaults before, or without, one)."""
    if not runtime_state_is_initialized():
        return _DEFAULT
    return getattr(get_runtime_state().runtime_config, "ulysses_a2a", _DEFAULT)


def selected() -> bool:
    """Whether Ulysses calls take this path (SDMA, or its layout over the default all-to-all)."""
    return _config().mode != "default"


def uses_sdma() -> bool:
    """Whether Ulysses calls move their data with SDMA copies."""
    return _config().mode == "sdma"


def overlap_qkv() -> bool:
    return selected() and _config().overlap_qkv


def compile_mode(mode: str) -> str:
    """The torch.compile mode for a component whose Ulysses attention takes this path: without
    CUDA graphs when the transport issues its copies from the host. With graphs, Inductor runs
    the ops between graph partitions, which costs a step more than the graphs save."""
    if mode not in _WITHOUT_CUDAGRAPHS or not uses_sdma() or not ulysses.host_issued():
        return mode
    if get_ulysses_parallel_world_size() <= 1:
        return mode
    warn_once(logger, f"--ulysses_a2a sdma: compiling the transformer with mode {_WITHOUT_CUDAGRAPHS[mode]!r}.")
    return _WITHOUT_CUDAGRAPHS[mode]


def call_reason(
    *,
    backend,
    ring_world_size: int,
    joint_strategy,
    fp8_comms,
    has_kv_cache: bool,
    head_balanced: bool,
    attention_kwargs,
    is_causal: bool = False,
) -> Optional[str]:
    """Why a Ulysses call with these settings cannot take the SDMA path, or None if it can."""
    if torch.version.hip is None:
        return "the SDMA all-to-all needs ROCm"
    if ring_world_size > 1:
        return "ring attention"
    if joint_strategy not in (None, "front", "rear"):
        return f"joint strategy {joint_strategy!r}"
    if fp8_comms is not None:
        return "FP8 communication"
    if has_kv_cache:
        return "PipeFusion KV cache"
    if head_balanced:
        return "head balancing"
    spec = attention_registry.find(backend)
    if spec is None or spec.sparsity is not None:
        return f"backend {getattr(backend, 'name', backend)} is not a dense backend"
    kwargs = attention_kwargs or {}
    unforwarded = [k for k in _UNFORWARDED_KWARGS if kwargs.get(k) is not None]
    if unforwarded:
        return f"attention_kwargs {unforwarded}"
    if joint_strategy and kwargs.get("valid_kv_len") is not None:
        if joint_strategy == "rear":
            return "a trailing SP pad behind rear joint tensors"
        if is_causal or kwargs.get("indices_k") is not None:
            return "a trailing SP pad with joint tensors and causal attention or packed keys"
    return None


def unsupported_reason(query, key, value, *, extra_inputs, **settings) -> Optional[str]:
    """Why this USP call cannot take the SDMA path, or None if it can (settings: `call_reason`'s)."""
    reason = call_reason(**settings)
    if reason is not None:
        return reason
    tensors = [query, key, value, *(x for _, x in extra_inputs)]
    if len({(x.shape[0], x.shape[3], x.dtype) for x in tensors}) != 1 or key.shape != value.shape:
        return "tensors of different batch, head dim or dtype"
    if any(x.shape != query.shape for _, x in extra_inputs):
        return "extra inputs not shaped like the query"
    return None


def run(
    query,
    key,
    value,
    *,
    group: dist.ProcessGroup,
    backend,
    dropout_p,
    is_causal,
    attention_kwargs,
    extra_inputs,
    joint_key,
    joint_value,
    joint_strategy,
    kv_head_repeat: int,
    **settings,
):
    """USP's Ulysses attention on this path; returns its output, or None when the call stays on
    the default all-to-all (warning once per reason). Inputs are USP's [b, H, s, d] after its
    joint handling: Q carries the joint rows, and `joint_key` / `joint_value` are this rank's
    head slice. `settings` are `call_reason`'s other arguments."""
    if dist.get_world_size(group) <= 1:
        return None
    reason = unsupported_reason(
        query,
        key,
        value,
        backend=backend,
        joint_strategy=joint_strategy,
        extra_inputs=extra_inputs,
        attention_kwargs=attention_kwargs,
        is_causal=is_causal,
        **settings,
    )
    if reason is not None:
        warn_fallback(reason)
        return None
    return attention(
        query,
        key,
        value,
        backend=backend,
        dropout_p=dropout_p,
        is_causal=is_causal,
        attention_kwargs=attention_kwargs,
        group=group,
        extra_inputs=extra_inputs,
        joint_key=joint_key if joint_strategy else None,
        joint_value=joint_value if joint_strategy else None,
        joint_strategy=joint_strategy,
        kv_head_repeat=kv_head_repeat,
    )


def warn_fallback(reason: str) -> None:
    if reason not in _warned:
        _warned.add(reason)
        logger.warning(f"Ulysses all-to-all {_config().mode}: this attention uses the default all-to-all ({reason})")


# Dynamo drops a logging call traced inside a compiled region; registered here, the call
# runs after the graph instead, without a graph break.
_reorderable = getattr(torch._dynamo.config, "reorderable_logging_functions", None)
if _reorderable is not None:
    _reorderable.add(warn_fallback)


def _backend_name(backend) -> str:
    return backend.name if isinstance(backend, AttentionBackendType) else str(backend)


def _reserved(joint_rows: int, joint_strategy, tensors: int):
    """Inbox rows each exchanged tensor keeps for this rank's joint K/V."""
    kv = (joint_rows, 0) if joint_strategy == "front" else (0, joint_rows)
    return [(0, 0), kv, kv] + [(0, 0)] * (tensors - 3)


def begin(query, rows, heads, reserved, group: dist.ProcessGroup):
    """Open a call: returns (token, the exchange's integer spec, per group per tensor heads)."""
    world = dist.get_world_size(group)
    config = _config()
    spec = ulysses.encode(world, query.shape[0], query.shape[3], rows, heads, reserved, config.head_groups)
    token = torch.ops.xfuser.sdma_ulysses_begin(query, spec, group.group_name, config.mode)
    return token, spec, ulysses.head_split(heads, config.head_groups)


def finish(
    token,
    packed,
    spec,
    *,
    joint_key,
    joint_value,
    extra_names,
    backend,
    dropout_p,
    is_causal,
    attention_kwargs,
    kv_head_repeat,
    group: dist.ProcessGroup,
    fp8_scales=None,
):
    """Attend and exchange back; returns USP's [b, H, s, d] (a transposed view of [b, s, H, d]).

    EXPERIMENT (TASK_004): `fp8_scales` (q, k, v, o scale tensors) when the exchanged Q/K/V are
    FP8: the attention gets them pre-quantized with those descales, and the output goes back in FP8.
    """
    kwargs = attention_kwargs or {}
    q_scale, k_scale, v_scale, o_scale = fp8_scales if fp8_scales is not None else (None,) * 4
    out = torch.ops.xfuser.sdma_ulysses_finish(
        token,
        packed,
        joint_key,
        joint_value,
        spec,
        ",".join(extra_names),
        kwargs.get("indices_k"),
        kwargs.get("cu_seqlens_k"),
        _backend_name(backend),
        float(dropout_p),
        bool(is_causal),
        kwargs.get("valid_kv_len"),
        kwargs.get("max_seqlen_k"),
        int(kv_head_repeat),
        _config().group_streams,
        group.group_name,
        q_scale,
        k_scale,
        v_scale,
        o_scale,
    )
    return out.transpose(1, 2)


def attention(
    query,
    key,
    value,
    *,
    backend,
    dropout_p,
    is_causal,
    attention_kwargs,
    group: dist.ProcessGroup,
    extra_inputs=(),
    joint_key=None,
    joint_value=None,
    joint_strategy=None,
    kv_head_repeat: int = 1,
):
    """USP's Ulysses attention through the SDMA ops. Inputs are USP's [b, H, s, d] after its joint
    handling (Q carries the joint rows; `joint_key` / `joint_value` are this rank's head slice)."""
    world = dist.get_world_size(group)
    tensors = [query, key, value, *(x for _, x in extra_inputs)]
    joint_rows = joint_key.shape[2] if joint_key is not None else 0
    reserved = _reserved(joint_rows, joint_strategy, len(tensors))
    rows = [x.shape[2] for x in tensors]
    heads = [x.shape[1] // world for x in tensors]
    token, spec, sizes = begin(query, rows, heads, reserved, group)
    offsets = [0] * len(tensors)
    packed = []
    for g, group_sizes in enumerate(sizes):
        pieces = [ulysses.pieces_from_heads(x, world, offsets[t], group_sizes[t]) for t, x in enumerate(tensors)]
        offsets = [o + h for o, h in zip(offsets, group_sizes)]
        packed.append(ulysses.pack_group(pieces))
        token = torch.ops.xfuser.sdma_ulysses_send(token, packed[-1], g, [], group.group_name)
    return finish(
        token,
        packed,
        spec,
        joint_key=joint_key,
        joint_value=joint_value,
        extra_names=[name for name, _ in extra_inputs],
        backend=backend,
        dropout_p=dropout_p,
        is_causal=is_causal,
        attention_kwargs=attention_kwargs,
        kv_head_repeat=kv_head_repeat,
        group=group,
    )


def _attend_fn(backend, dropout_p, is_causal, attention_kwargs, extra_names, joint_front, joint, kv_head_repeat):
    """Run the backend on one head group as USP would after its input all-to-all."""
    from xfuser.model_executor.layers import usp

    attention_function = usp._get_attention_function(backend=backend)
    valid = attention_kwargs["valid_kv_len"]

    def attend(q, k, v, *extras):
        kwargs = dict(attention_kwargs)
        kwargs.update(zip(extra_names, extras))
        if valid is not None and not is_causal and not usp._backend_strips_sp_pad(backend, kwargs):
            if joint:
                # Front joint rows, then the main rows with their trailing pad: keep up to the pad.
                k, v = k[:, :, : joint_front + valid], v[:, :, : joint_front + valid]
            else:
                k, v, kwargs = usp._trim_trailing_kv_padding(k, v, kwargs, backend)
        if joint and valid is not None:
            # The pad is gone, and the joint rows must not be cropped by it again.
            kwargs["valid_kv_len"] = None
        if kv_head_repeat > 1:
            k, v = usp._repeat_kv_heads(k, v, kv_head_repeat)
        out, _ = attention_function(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal, joint_attn_kwargs=None, attention_kwargs=kwargs
        )
        return out

    return attend


def _group(name: str) -> dist.ProcessGroup:
    return dist.distributed_c10d._resolve_process_group(name)


# Host-issued copies: Inductor runs these ops between CUDA graph partitions instead of capturing
# them (where the tag exists; else a capture falls back to the default all-to-all in begin).
_HOST_ISSUED = (torch.Tag.cudagraph_unsafe,) if hasattr(torch.Tag, "cudagraph_unsafe") else ()


@torch.library.custom_op("xfuser::sdma_ulysses_begin", mutates_args=(), tags=_HOST_ISSUED)
def _sdma_ulysses_begin(ref: torch.Tensor, spec: List[int], group_name: str, mode: str) -> torch.Tensor:
    ulysses.begin(_group(group_name), ulysses.Exchange.decode(spec, ref.dtype), rccl=mode == "packed")
    return ref.new_empty(0)


@_sdma_ulysses_begin.register_fake
def _(ref, spec, group_name, mode):
    return ref.new_empty(0)


# `packed` is declared mutated, and handed to finish as well, although neither op writes it:
# the engines read it until finish has waited for the peers' outputs, and only an op that
# uses it keeps torch.compile from recomputing it into a second buffer or reusing its memory
# (for the next group's pack, say) once send has returned.
@torch.library.custom_op("xfuser::sdma_ulysses_send", mutates_args=("packed",), tags=_HOST_ISSUED)
def _sdma_ulysses_send(
    token: torch.Tensor, packed: torch.Tensor, group: int, tensors: List[int], group_name: str
) -> torch.Tensor:
    # `tensors` empty: the rest of the group's tensors (ulysses.send).
    ulysses.send(_group(group_name), group, packed, tensors or None)
    return token.new_empty(0)


@_sdma_ulysses_send.register_fake
def _(token, packed, group, tensors, group_name):
    return token.new_empty(0)


# A head group's Q/K/V projection after the previous group's send. Inductor would otherwise
# run every group's GEMM first (it fuses the groups' norm kernels into one) and start no send
# until the last projection is done. The GEMM is an extern kernel under Inductor either way.
@torch.library.custom_op("xfuser::sdma_ulysses_linear", mutates_args=())
def _sdma_ulysses_linear(
    token: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


@_sdma_ulysses_linear.register_fake
def _(token, x, weight, bias):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("xfuser::sdma_ulysses_finish", mutates_args=(), tags=_HOST_ISSUED)
def _sdma_ulysses_finish(
    token: torch.Tensor,
    packed: List[torch.Tensor],
    joint_key: Optional[torch.Tensor],
    joint_value: Optional[torch.Tensor],
    spec: List[int],
    extra_names: str,
    indices_k: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    backend: str,
    dropout_p: float,
    is_causal: bool,
    valid_kv_len: Optional[int],
    max_seqlen_k: Optional[int],
    kv_head_repeat: int,
    concurrent: bool,
    group_name: str,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    o_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    exchange = ulysses.Exchange.decode(spec, token.dtype)
    joint_front = exchange.reserved[1][0]
    attention_kwargs = {
        "valid_kv_len": valid_kv_len,
        "indices_k": indices_k,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_k": max_seqlen_k,
    }
    if q_descale is not None:
        # EXPERIMENT (TASK_004): Q/K/V arrive as FP8, quantized with these scales before the send.
        attention_kwargs.update(pre_quantized=True, q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
    attend = _attend_fn(
        AttentionBackendType[backend],
        dropout_p,
        is_causal,
        attention_kwargs,
        # Custom ops take no list of strings: the names travel comma-joined.
        extra_names.split(",") if extra_names else [],
        joint_front,
        joint_key is not None,
        kv_head_repeat,
    )
    local: Sequence[Optional[torch.Tensor]] = [None, joint_key, joint_value] + [None] * (exchange.tensors - 3)
    return ulysses.finish(_group(group_name), attend, local, concurrent, out_scale=o_scale)


@_sdma_ulysses_finish.register_fake
def _(
    token,
    packed,
    joint_key,
    joint_value,
    spec,
    extra_names,
    indices_k,
    cu_seqlens_k,
    backend,
    dropout_p,
    is_causal,
    valid_kv_len,
    max_seqlen_k,
    kv_head_repeat,
    concurrent,
    group_name,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    o_scale=None,
):
    world, batch, head_dim = spec[0], spec[1], spec[2]
    rows, heads = spec[5], spec[6]
    dtype = torch.bfloat16 if o_scale is not None else token.dtype
    return token.new_empty(batch, rows, world * heads, head_dim, dtype=dtype)


# EXPERIMENT (TASK_004): `x`, ordered after `token` (a copy: a custom op cannot return its input).
# Keeps a head group's quantized Q/K/V projection after the previous group's send.
@torch.library.custom_op("xfuser::sdma_ulysses_after", mutates_args=())
def _sdma_ulysses_after(token: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_sdma_ulysses_after.register_fake
def _(token, x):
    return torch.empty_like(x)


# --- SDMA attention processors -----------------------------------------------------------------
#
# A model registers, for an attention processor class of its, a subclass that overlaps the
# Q/K/V projection with the exchange and otherwise runs the base class's path. Two kinds:
# - `sdma_grouped_qkv` (an attribute or property, true for the processors that use it):
#   Q/K/V are projected one head group at a time (`grouped_qkv_attention`), for models whose
#   QK norm and RoPE act per head. `prepare` copies each module's weight rows per group.
# - `sdma_qk_first(attn)` (a static check of the module): Q/K are sent before V is projected,
#   in one head group (`qk_first_attention`), for models whose QK norm spans every head.

_PROCESSORS: Dict[type, type] = {}


def sdma_processor(base: type):
    """Class decorator registering an SDMA attention processor for `base`, its base class.

    `prepare` turns `base` instances into the subclass in place (models may share one
    processor instance across blocks and set its state per stage), so the subclass may add
    behaviour but no state of its own: no __init__."""

    def register(cls):
        if not issubclass(cls, base) or "__init__" in vars(cls):
            raise TypeError(f"{cls.__name__} must subclass {base.__name__} and add no __init__")
        _PROCESSORS[base] = cls
        return cls

    return register


def prepare(pipe, components, *, cpu_offload: bool) -> None:
    """Set up the attention modules of `pipe`'s `components` for --ulysses_sdma_overlap_qkv,
    before compilation sees them: swap in the registered SDMA processors, and copy the Q/K/V
    weight rows per head group for those that project one group at a time.

    Not the copies under CPU offload: the offload hooks, already set up, would not move them,
    and offloading is for the memory they would spend. Those modules then project Q/K/V for
    all heads at once, as without the option. Sending Q/K first copies nothing and still
    applies."""
    world = get_ulysses_parallel_world_size()
    if not overlap_qkv() or world <= 1:
        return
    for name in components:
        component = getattr(pipe, name, None)
        if not isinstance(component, torch.nn.Module):
            continue
        grouped = offloaded = qk_first = 0
        for attn in component.modules():
            processor = getattr(attn, "processor", None)
            sdma = _PROCESSORS.get(type(processor))
            if sdma is not None:
                processor.__class__ = sdma
            if getattr(processor, "sdma_grouped_qkv", False):
                if cpu_offload:
                    offloaded += 1
                else:
                    grouped += _prepare_grouped_qkv(attn, world)
            check = getattr(processor, "sdma_qk_first", None)
            qk_first += check is not None and check(attn)
        if grouped:
            log(f"{name}: {grouped} attention modules project Q/K/V one head group at a time.")
        if offloaded:
            log(f"{name}: {offloaded} attention modules project Q/K/V for all heads at once under CPU offload.")
        if qk_first:
            log(
                f"{name}: {qk_first} attention modules send Q/K before projecting V, in one head group "
                "(--ulysses_sdma_head_groups does not apply to them)."
            )


# --- Q/K/V projected one head group at a time -----------------------------------------------
#
# Each head group's Q/K/V come out of a GEMM over that group's weight rows, go through the
# model's per-head norm and RoPE, and are sent before the next group is projected, so the
# group's exchange overlaps the next projection. The weight rows are copied once per
# attention module by `prepare`, before compilation.

_GROUPED = "_xfuser_sdma_qkv"


def plain_linear(module) -> bool:
    # Quantized GEMMs replace the module; sharded (FSDP) weights are DTensors with placements;
    # torchao (--use_fp8_gemms) keeps the module but quantizes its weight.
    return type(module) is torch.nn.Linear and not hasattr(module.weight, "placements") and not _fp8_weight(module)


def _fp8_weight(module) -> bool:
    """EXPERIMENT (TASK_004): whether --use_fp8_gemms (torchao) quantized this nn.Linear's weight."""
    from xfuser.core.utils.runner_utils import _weight_is_torchao_quantized

    return type(module) is torch.nn.Linear and _weight_is_torchao_quantized(module)


def _qkv_parts(attn):
    """(weight, bias) of Q, K and V, or None unless they are plain nn.Linear layers of one shape.

    EXPERIMENT (TASK_004): a fused projection with a torchao FP8 weight is dequantized here;
    `_prepare_grouped_qkv` quantizes each head group's rows again."""
    if getattr(attn, "fused_projections", False) and hasattr(attn, "to_qkv"):
        if _fp8_weight(attn.to_qkv):
            weight, bias = attn.to_qkv.weight.dequantize(), attn.to_qkv.bias
            n = weight.shape[0] // 3
            return [
                (weight[i * n : (i + 1) * n], None if bias is None else bias[i * n : (i + 1) * n]) for i in range(3)
            ]
        if not plain_linear(attn.to_qkv):
            return None
        weight, bias = attn.to_qkv.weight, attn.to_qkv.bias
        n = weight.shape[0] // 3
        return [(weight[i * n : (i + 1) * n], None if bias is None else bias[i * n : (i + 1) * n]) for i in range(3)]
    linears = [getattr(attn, name, None) for name in ("to_q", "to_k", "to_v")]
    if not all(m is not None and plain_linear(m) for m in linears):
        return None
    if len({m.weight.shape for m in linears}) != 1:
        return None
    return [(m.weight, m.bias) for m in linears]


# --- Q/K sent before V is projected ------------------------------------------------------------
#
# A model whose QK norm spans every head (Wan) cannot project one head group at a time, but it
# can send Q and K, with all heads, as soon as they are normed and rotated, and project V while
# they move.


def qk_first_attention(
    ref, project_qk, project_v, *, heads, head_dim, backend, attention_kwargs, group: dist.ProcessGroup
):
    """Ulysses attention in one head group with Q/K sent before V is projected; returns USP's
    [b, H, s, d], or None when the call cannot take this path (the processor runs its usual one).

    `ref` is the [b, s, ...] input (batch, rows, dtype). `project_qk()` returns Q and K packed
    peer by peer: for each rank p, Q's then K's [b, s, H/P, d] slice of rank p's heads.
    `project_v(token)` returns V as [b, H, s, d]; it should project through
    `torch.ops.xfuser.sdma_ulysses_linear(token, ...)`, which keeps the projection after the Q/K
    send under torch.compile.
    """
    if not selected():
        return None
    world = dist.get_world_size(group)
    if world <= 1 or heads % world:
        return None
    runtime = get_runtime_state()
    if backend is None:
        backend = runtime.attention_backend
    reason = call_reason(
        backend=backend,
        ring_world_size=get_ring_parallel_world_size(),
        joint_strategy=None,
        fp8_comms=runtime.fp8_comms,
        has_kv_cache=False,
        head_balanced=runtime.runtime_config.use_sparseattn_head_balance,
        attention_kwargs=attention_kwargs,
    )
    if reason is not None:
        warn_fallback(reason)
        return None
    per_rank = heads // world
    spec = ulysses.encode(world, ref.shape[0], head_dim, [ref.shape[1]] * 3, [per_rank] * 3, [(0, 0)] * 3, 1)
    packed_qk = project_qk()
    # Opened on the packed Q/K: under torch.compile the call is armed once Q/K exist.
    token = torch.ops.xfuser.sdma_ulysses_begin(packed_qk, spec, group.group_name, _config().mode)
    token = torch.ops.xfuser.sdma_ulysses_send(token, packed_qk, 0, [0, 1], group.group_name)
    value = project_v(token)
    packed_v = ulysses.pack_group([ulysses.pieces_from_heads(value, world, 0, per_rank)])
    token = torch.ops.xfuser.sdma_ulysses_send(token, packed_v, 0, [2], group.group_name)
    return finish(
        token,
        [packed_qk, packed_v],
        spec,
        joint_key=None,
        joint_value=None,
        extra_names=[],
        backend=backend,
        dropout_p=0.0,
        is_causal=False,
        attention_kwargs=attention_kwargs,
        kv_head_repeat=1,
        group=group,
    )


def _prepare_grouped_qkv(attn: torch.nn.Module, world: int) -> bool:
    """Copy the attention module's Q/K/V weight rows per head group, for `world` Ulysses ranks
    and the configured head groups; False if its projections cannot be split."""
    parts = _qkv_parts(attn)
    heads = getattr(attn, "heads", None)
    if parts is None or not heads or heads % world:
        return False
    head_dim = parts[0][0].shape[0] // heads
    per_rank = heads // world
    sizes = ulysses.head_split((per_rank,), _config().head_groups)
    fp8 = getattr(attn, "fused_projections", False) and hasattr(attn, "to_qkv") and _fp8_weight(attn.to_qkv)
    offset = 0
    with torch.no_grad():
        for g, (h,) in enumerate(sizes):
            rows = torch.tensor(
                [
                    (p * per_rank + offset + j) * head_dim + e
                    for p in range(world)
                    for j in range(h)
                    for e in range(head_dim)
                ],
                device=parts[0][0].device,
            )
            weight = torch.cat([w.index_select(0, rows) for w, _ in parts])
            bias = None if parts[0][1] is None else torch.cat([b.index_select(0, rows) for _, b in parts])
            if fp8:
                # EXPERIMENT (TASK_004): this group's rows as an nn.Linear quantized like the original.
                from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp8

                linear = torch.nn.Linear(
                    weight.shape[1], weight.shape[0], bias=bias is not None, device=weight.device, dtype=weight.dtype
                )
                linear.weight.copy_(weight)
                if bias is not None:
                    linear.bias.copy_(bias)
                holder = torch.nn.ModuleDict({"linear": linear})
                quantize_linear_layers_to_fp8(holder)
                attn.add_module(f"{_GROUPED}_lin{g}", holder["linear"])
            else:
                attn.register_buffer(f"{_GROUPED}_w{g}", weight, persistent=False)
                attn.register_buffer(f"{_GROUPED}_b{g}", bias, persistent=False)
            offset += h
    attn._xfuser_sdma_qkv_layout = (world, sizes, head_dim)
    return True


def grouped_qkv_attention(attn, hidden_states, norm_rope, *, backend, attention_kwargs, group: dist.ProcessGroup):
    """Ulysses attention with Q/K/V projected one head group at a time; returns USP's [b, H, s, d],
    or None when the call cannot take this path (the processor then runs its usual one).

    `norm_rope(q, k, v)` takes one group's [b, s, h, d] projections, with every rank's heads of
    the group, and returns them normed and rotated, as the model does for all heads.
    """
    if not overlap_qkv():
        return None
    layout = getattr(attn, "_xfuser_sdma_qkv_layout", None)
    world = dist.get_world_size(group)
    if layout is None or layout[0] != world:
        return None
    runtime = get_runtime_state()
    if backend is None:
        backend = runtime.attention_backend
    # EXPERIMENT (TASK_004): FP8 Q/K/V over SDMA with --use_fp8_comms, once its scales are set
    # (--fp8_comms_scale: fixed, no calibration). The scales are the layer's fp8_comms buffers.
    fp8_scales = None
    if runtime.fp8_comms is not None:
        from xfuser.core.distributed import fp8_comms as fp8c

        call = fp8c.fp8_attention_kwargs(runtime.fp8_comms, attn, None, None, None, False, backend).get("fp8_comms")
        if call is not None:
            fp8_scales = (call.q_scale, call.k_scale, call.v_scale, call.o_scale)
    reason = call_reason(
        backend=backend,
        ring_world_size=get_ring_parallel_world_size(),
        joint_strategy=None,
        fp8_comms=None if fp8_scales is not None else runtime.fp8_comms,
        has_kv_cache=False,
        head_balanced=runtime.runtime_config.use_sparseattn_head_balance,
        attention_kwargs=attention_kwargs,
    )
    if reason is not None:
        warn_fallback(reason)
        return None
    _, sizes, head_dim = layout
    if sizes != ulysses.head_split((sum(h for (h,) in sizes),), _config().head_groups):
        return None
    per_rank = sum(h for (h,) in sizes)
    dtype = fp8c._FP8_DTYPE if fp8_scales is not None else hidden_states.dtype
    token, spec, _ = begin(
        hidden_states.new_empty(hidden_states.shape[0], 1, 1, head_dim, dtype=dtype),
        [hidden_states.shape[1]] * 3,
        [per_rank] * 3,
        [(0, 0)] * 3,
        group,
    )
    packed = []
    for g in range(len(sizes)):
        linear = getattr(attn, f"{_GROUPED}_lin{g}", None)
        if linear is not None:
            # FP8 GEMMs: the group's quantized projection, after the previous group's send.
            qkv = linear(hidden_states if g == 0 else torch.ops.xfuser.sdma_ulysses_after(token, hidden_states))
        else:
            weight, bias = getattr(attn, f"{_GROUPED}_w{g}"), getattr(attn, f"{_GROUPED}_b{g}")
            if g == 0:
                qkv = torch.nn.functional.linear(hidden_states, weight, bias)
            else:
                qkv = torch.ops.xfuser.sdma_ulysses_linear(token, hidden_states, weight, bias)
        q, k, v = (x.unflatten(-1, (-1, head_dim)) for x in qkv.chunk(3, dim=-1))
        q, k, v = norm_rope(q, k, v)
        if fp8_scales is not None:
            # As fp8_comms_input_all_to_all: rotate Q/K for the backend, quantize with the layer's
            # scales; sent as bytes (uint8 views: the pack is a cat).
            q, k = fp8c._rotate_for_backend(q, k, backend)
            q, k, v = (fp8c._per_tensor_quant(x, s)[0] for x, s in zip((q, k, v), fp8_scales))
            q, k, v = (x.view(torch.uint8) for x in (q, k, v))
        packed.append(ulysses.pack_group([ulysses.pieces_from_grouped(x, world) for x in (q, k, v)]))
        token = torch.ops.xfuser.sdma_ulysses_send(token, packed[-1], g, [], group.group_name)
    return finish(
        token,
        packed,
        spec,
        joint_key=None,
        joint_value=None,
        extra_names=[],
        backend=backend,
        dropout_p=0.0,
        is_causal=False,
        attention_kwargs=attention_kwargs,
        kv_head_repeat=1,
        group=group,
        fp8_scales=fp8_scales,
    )
