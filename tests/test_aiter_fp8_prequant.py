"""
Test whether aiter.flash_attn_fp8_pertensor_func with pre_quantized=True
produces output equivalent to BF16 attention.

Compares three paths:
  ref  — BF16 SDPA (ground truth)
  auto — AITER FP8 kernel with internal quantization (pre_quantized=False)
  pre  — AITER FP8 kernel with manually pre-quantized inputs (pre_quantized=True, our a2a path)

If ref ≈ auto but ref ≉ pre, the pre_quantized path has a bug.
If ref ≉ auto, the FP8 kernel itself is the source of quality loss.

Usage:
    python tests/test_aiter_fp8_prequant.py [--compile] [--profile]

Flags:
  --compile   wrap each attention function with torch.compile before running
  --profile   emit a Chrome trace to /tmp/aiter_fp8_{auto,pre}.json
"""

import sys
import argparse
import torch
import torch.nn.functional as F

try:
    import aiter
except ImportError:
    print("SKIP: aiter not available")
    sys.exit(0)

if not hasattr(aiter, "flash_attn_fp8_pertensor_func"):
    print("SKIP: aiter.flash_attn_fp8_pertensor_func not available in this aiter version")
    sys.exit(0)


def _sdpa_ref(q, k, v):
    """BF16 SDPA reference. Inputs/output in [B, S, H, D]."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2)


def _aiter_fp8_auto(q, k, v):
    """AITER FP8 kernel with internal quantization. Inputs/output in [B, S, H, D]."""
    fp8 = aiter.dtypes.fp8
    dtype_max = torch.finfo(fp8).max
    q_fp8, q_ds = aiter.per_tensor_quant(q, quant_dtype=fp8, dtypeMax=dtype_max)
    k_fp8, k_ds = aiter.per_tensor_quant(k, quant_dtype=fp8, dtypeMax=dtype_max)
    v_fp8, v_ds = aiter.per_tensor_quant(v, quant_dtype=fp8, dtypeMax=dtype_max)
    return aiter.flash_attn_fp8_pertensor_func(
        q_fp8, k_fp8, v_fp8,
        q_descale=q_ds, k_descale=k_ds, v_descale=v_ds,
    )


def _quant_per_tensor(x):
    """Mimic _per_tensor_quant from usp.py (without distributed all_reduce)."""
    dtype_max = torch.finfo(torch.float8_e4m3fn).max
    scale = x.float().abs().amax() / dtype_max
    return (x.float() / scale).to(torch.float8_e4m3fn), scale.reshape(1)


def _aiter_fp8_pre(q, k, v):
    """AITER FP8 kernel with manually pre-quantized inputs. Inputs/output in [B, S, H, D]."""
    q_fp8, q_ds = _quant_per_tensor(q)
    k_fp8, k_ds = _quant_per_tensor(k)
    v_fp8, v_ds = _quant_per_tensor(v)
    return aiter.flash_attn_fp8_pertensor_func(
        q_fp8, k_fp8, v_fp8,
        q_descale=q_ds, k_descale=k_ds, v_descale=v_ds,
    )


def _profile(fn, label, q, k, v, warmup=3, steps=10):
    """Run fn(q, k, v) under torch.profiler and save a Chrome trace to /tmp."""
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    trace_path = f"/tmp/aiter_fp8_{label}.json"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(steps):
            fn(q, k, v)
        torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"  [{label}] Chrome trace -> {trace_path}")

    # Print top CUDA kernels by total time
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    print(table)


def _compare(label, a, b, device):
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"  {label:45s}  max={diff.max():.5f}  mean={diff.mean():.5f}  cosine={cos:.6f}")


def test_pipeline_dtypes(B, H, S, D, device):
    """
    Trace dtype at every step of the fp8_a2a pipeline on a single GPU.

    On a single GPU there is no actual all2all collective, but we can apply
    the same permute/reshape operations that _ft_c_input_all_to_all performs
    to confirm dtype is preserved through the memory manipulation.

    Pipeline:
      BF16 [B,H,S,D]  →  FP8 [B,H,S,D]  →  (permute/reshape, no collective)
      →  FP8 [B,H,S,D]  →  aiter kernel (pre_quantized=True)  →  BF16 [B,S,H,D]
    """
    print(f"\n--- dtype trace  B={B} H={H} S={S} D={D} ---")
    torch.manual_seed(0)

    # USP receives [B, H, S, D] (BHSD)
    q_bf16 = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
    print(f"  input          dtype={q_bf16.dtype}  shape={tuple(q_bf16.shape)}")

    # Step 1: quantize to FP8
    dtype_max = torch.finfo(torch.float8_e4m3fn).max
    scale = q_bf16.float().abs().amax() / dtype_max
    q_fp8 = (q_bf16.float() / scale).to(torch.float8_e4m3fn)
    descale = scale.reshape(1)
    print(f"  after quant    dtype={q_fp8.dtype}  shape={tuple(q_fp8.shape)}  scale={scale.item():.5f}")

    # Step 2: simulate the permute/reshape from _ft_c_input_all_to_all
    # (with world_size=1 the all2all is a no-op, but we apply the memory ops)
    q_permuted = q_fp8.permute(1, 0, 2, 3).contiguous()  # [H, B, S, D]
    print(f"  after permute  dtype={q_permuted.dtype}  shape={tuple(q_permuted.shape)}")
    q_permuted2 = q_permuted.reshape(1, H, B, S, D).permute(2, 1, 0, 3, 4).reshape(B, H, -1, D)
    print(f"  after reshape  dtype={q_permuted2.dtype}  shape={tuple(q_permuted2.shape)}")

    # Step 3: kernel input — permute BHSD → BSHD as _aiter_fp8_attn_call does
    q_bshd = q_permuted2.permute(0, 2, 1, 3).contiguous()
    print(f"  kernel input   dtype={q_bshd.dtype}  shape={tuple(q_bshd.shape)}")

    # Step 4: run kernel with pre_quantized=True
    out = aiter.flash_attn_fp8_pertensor_func(
        q_bshd, q_bshd, q_bshd,
        q_descale=descale, k_descale=descale, v_descale=descale,
    )
    print(f"  kernel output  dtype={out.dtype}  shape={tuple(out.shape)}")

    # Step 5: compare kernel output against BF16 SDPA reference
    q_bshd_ref = q_bf16.permute(0, 2, 1, 3)  # BHSD → BSHD
    out_ref = F.scaled_dot_product_attention(
        q_bshd_ref, q_bshd_ref, q_bshd_ref
    )
    _compare("pipeline vs BF16 ref", out.float(), out_ref.float(), device)


def _inspect_descales(q):
    """Print descale shapes returned by aiter.per_tensor_quant vs our _quant_per_tensor."""
    fp8 = aiter.dtypes.fp8
    dtype_max = torch.finfo(fp8).max
    _, auto_ds = aiter.per_tensor_quant(q, quant_dtype=fp8, dtypeMax=dtype_max)
    _, pre_ds  = _quant_per_tensor(q)
    print(f"  aiter.per_tensor_quant descale: shape={tuple(auto_ds.shape)}  dtype={auto_ds.dtype}  value={auto_ds.flatten()[0].item():.6f}")
    print(f"  _quant_per_tensor      descale: shape={tuple(pre_ds.shape)}   dtype={pre_ds.dtype}   value={pre_ds.flatten()[0].item():.6f}")


def run(B, H, S, D, device):
    print(f"\nshape [B={B}, S={S}, H={H}, D={D}]")
    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)

    print("descale shapes:")
    _inspect_descales(q)

    out_ref  = _sdpa_ref(q, k, v)
    out_auto = _aiter_fp8_auto(q, k, v)
    out_pre  = _aiter_fp8_pre(q, k, v)

    _compare("ref  vs  auto (internal quant)", out_ref,  out_auto, device)
    _compare("ref  vs  pre  (our path)",       out_ref,  out_pre,  device)
    _compare("auto vs  pre",                   out_auto, out_pre,  device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="wrap attention functions with torch.compile")
    parser.add_argument("--profile", action="store_true", help="profile both paths and emit Chrome traces to /tmp/")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        sys.exit(0)

    device = "cuda:0"

    auto_fn = _aiter_fp8_auto
    pre_fn  = _aiter_fp8_pre
    if args.compile:
        print("torch.compile enabled")
        auto_fn = torch.compile(_aiter_fp8_auto)
        pre_fn  = torch.compile(_aiter_fp8_pre)

    # dtype trace through the full fp8_a2a pipeline
    test_pipeline_dtypes(B=1, H=40, S=512,  D=128, device=device)
    test_pipeline_dtypes(B=1, H=5,  S=4096, D=128, device=device)

    # kernel equivalence: ref vs internal quant vs pre_quantized
    # inputs in [B, S, H, D] as the kernel expects
    run(B=1, H=40, S=512,  D=128, device=device)
    run(B=1, H=5,  S=4096, D=128, device=device)

    if args.profile:
        print("\n--- profiling auto path ---")
        torch.manual_seed(42)
        q = torch.randn(1, 512, 40, 128, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, 512, 40, 128, dtype=torch.bfloat16, device=device)
        v = torch.randn(1, 512, 40, 128, dtype=torch.bfloat16, device=device)
        _profile(auto_fn, "auto", q, k, v)

        print("\n--- profiling pre path ---")
        _profile(pre_fn, "pre", q, k, v)
