"""Fused gated residual + RMSNorm + AdaLN modulation for MiniMax-H3 (Triton).

Layout / algorithm
------------------
Every MiniMax-H3 block boundary computes

    hidden = residual + gate[gate_index] * branch
    normed = RMSNorm(hidden) * (1 + scale[mod_index]) + shift[mod_index]

over ``[B, S, H]`` rows with ``H = 5376``. That is above the 1024-element
limit up to which Inductor keeps a reduction row in registers, so its fused
kernel loops over the row twice -- once for the sum of squares, once to
normalise -- and reads the inputs twice. Here one program owns one row in
registers and reads each input once. ``gate``, ``shift`` and ``scale`` are row
views into the per-block modulation table (one row per (timestep, modality)),
gathered per token; the table is small enough to stay in cache.

Without ``branch`` (the first block) the op only normalises and modulates.

Numerics
--------
Everything is computed in fp32 and rounded only where ``hidden`` and
``normed`` are stored. That is also what Inductor generates for the compiled
reference: inside a fused kernel it computes bf16 intermediates in fp32, and
even an explicit ``.to(torch.bfloat16)`` does not round unless
``emulate_precision_casts`` is set. Eager PyTorch rounds after every op, so
against eager the outputs differ by a few bf16 ULPs of the intermediates.

Envelope / fallback
-------------------
bf16, contiguous ``[B, S, H]`` activations; affine ``nn.RMSNorm(H)`` with a
bf16 weight; ``[rows, H]`` modulation views with unit column stride and
``[S]`` integer indices. Anything else, including a grad-enabled call, runs
:func:`_reference`.
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    _HAS_TRITON = True
except Exception:  # pragma: no cover - only exercised where triton is absent
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.autotune(
        configs=[triton.Config({}, num_warps=warps) for warps in (4, 8, 16)],
        key=["H", "HAS_BRANCH"],
    )
    @triton.jit
    def _residual_norm_modulate_kernel(
        residual_ptr,
        branch_ptr,
        gate_ptr,
        gate_index_ptr,
        shift_ptr,
        scale_ptr,
        mod_index_ptr,
        weight_ptr,
        hidden_ptr,
        normed_ptr,
        seq_len,
        gate_row_stride,
        shift_row_stride,
        scale_row_stride,
        eps,
        H: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_BRANCH: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        token = row % seq_len
        cols = tl.arange(0, BLOCK)
        mask = cols < H
        offsets = row * H + cols

        x = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_BRANCH:
            gate_row = tl.load(gate_index_ptr + token).to(tl.int64)
            gate = tl.load(gate_ptr + gate_row * gate_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
            branch = tl.load(branch_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            x = x + gate * branch
            tl.store(hidden_ptr + offsets, x.to(tl.bfloat16), mask=mask)

        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = x * rstd * weight

        mod_row = tl.load(mod_index_ptr + token).to(tl.int64)
        shift = tl.load(shift_ptr + mod_row * shift_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + mod_row * scale_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(normed_ptr + offsets, (normed * (1.0 + scale) + shift).to(tl.bfloat16), mask=mask)

    def _launch(residual, branch, gate, gate_index, shift, scale, mod_index, weight, eps, hidden):
        batch, seq_len, width = residual.shape
        normed = torch.empty_like(residual)
        has_branch = branch is not None
        wrap_triton(_residual_norm_modulate_kernel)[(batch * seq_len,)](
            residual,
            branch if has_branch else residual,
            gate if has_branch else shift,
            gate_index if has_branch else mod_index,
            shift,
            scale,
            mod_index,
            weight,
            hidden if has_branch else normed,
            normed,
            seq_len,
            gate.stride(0) if has_branch else 0,
            shift.stride(0),
            scale.stride(0),
            eps,
            H=width,
            BLOCK=triton.next_power_of_2(width),
            HAS_BRANCH=has_branch,
        )
        return normed

    @triton_op("xfuser::minimax_h3_residual_norm_modulate", mutates_args=())
    def _residual_norm_modulate(
        residual: torch.Tensor,
        branch: torch.Tensor,
        gate: torch.Tensor,
        gate_index: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        mod_index: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.empty_like(residual)
        normed = _launch(residual, branch, gate, gate_index, shift, scale, mod_index, weight, eps, hidden)
        return hidden, normed

    @triton_op("xfuser::minimax_h3_norm_modulate", mutates_args=())
    def _norm_modulate(
        hidden: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        mod_index: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return _launch(hidden, None, None, None, shift, scale, mod_index, weight, eps, None)


def _reference(residual, branch, gate, gate_index, shift, scale, mod_index, norm):
    hidden = residual if branch is None else residual + gate.index_select(0, gate_index) * branch
    normed = norm(hidden) * (1.0 + scale.index_select(0, mod_index)) + shift.index_select(0, mod_index)
    return hidden, normed


def _table_supported(table, width, dtype) -> bool:
    return (
        table.dim() == 2
        and table.shape[1] == width
        and table.stride(1) == 1
        and table.dtype == dtype
        and table.device.type == "cuda"
    )


def _fused_supported(residual, branch, gate, gate_index, shift, scale, mod_index, norm) -> bool:
    if not _HAS_TRITON or torch.is_grad_enabled():
        return False
    if residual.device.type != "cuda" or residual.dtype != torch.bfloat16 or residual.dim() != 3:
        return False
    if not residual.is_contiguous():
        return False
    _, seq_len, width = residual.shape
    if type(norm) is not torch.nn.RMSNorm or norm.eps is None or norm.weight is None:
        return False
    if tuple(norm.normalized_shape) != (width,) or norm.weight.dtype != residual.dtype:
        return False
    if not norm.weight.is_contiguous():
        return False
    tables = [shift, scale]
    indices = [mod_index]
    if branch is not None:
        if branch.shape != residual.shape or branch.dtype != residual.dtype or not branch.is_contiguous():
            return False
        tables.append(gate)
        indices.append(gate_index)
    for table in tables:
        if not _table_supported(table, width, residual.dtype):
            return False
    for index in indices:
        if index.dim() != 1 or index.shape[0] != seq_len or index.dtype not in (torch.int32, torch.int64):
            return False
        if index.device != residual.device or not index.is_contiguous():
            return False
    return True


def minimax_h3_residual_norm_modulate(
    residual: torch.Tensor,
    branch: Optional[torch.Tensor],
    gate: Optional[torch.Tensor],
    gate_index: Optional[torch.Tensor],
    shift: torch.Tensor,
    scale: torch.Tensor,
    mod_index: torch.Tensor,
    norm: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(hidden, normed)`` for one MiniMax-H3 block boundary.

    ``hidden = residual + gate[gate_index] * branch`` (just ``residual`` when
    ``branch`` is None) and ``normed = norm(hidden) * (1 + scale[mod_index]) +
    shift[mod_index]``. ``gate``, ``shift`` and ``scale`` hold one row per
    (timestep, modality); the indices pick a row per token.
    """
    if _fused_supported(residual, branch, gate, gate_index, shift, scale, mod_index, norm):
        eps = float(norm.eps)
        if branch is None:
            normed = torch.ops.xfuser.minimax_h3_norm_modulate(residual, shift, scale, mod_index, norm.weight, eps)
            return residual, normed
        return torch.ops.xfuser.minimax_h3_residual_norm_modulate(
            residual, branch, gate, gate_index, shift, scale, mod_index, norm.weight, eps
        )
    return _reference(residual, branch, gate, gate_index, shift, scale, mod_index, norm)
