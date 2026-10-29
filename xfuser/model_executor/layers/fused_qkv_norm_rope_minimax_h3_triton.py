"""Fused QK-RMSNorm + partial rotate-half RoPE + V pack for MiniMax-H3 (Triton).

Layout / algorithm
------------------
The unfused path reads the fused QKV projection three times (a sum-of-squares
pass per q/k, the rotate-half operand, the final rotate), writes the
``cat(-x2, x1)`` operand for q and k, and copies v out of the QKV chunk on its
own so the attention backend gets a contiguous K/V. Here one program covers
``BLOCK_ROWS`` ``(token, head)`` rows of ``D`` channels and writes contiguous
``[B, S, H, D]`` q, k and v in a single pass over QKV. The rotate-half partner
of channel ``c`` is ``c +- rotary_dim / 2`` in the same head, so it is a second
load of the same cache lines rather than a second pass over memory. The Ulysses
variant stores the same rows straight into USP's chunked send buffers instead.

Numerics
--------
The norm and the rotation are computed in fp32, with fp32 cos/sin, and q/k are
rounded once at the store; channels from ``rotary_dim`` up pass through the
norm unrotated and v is copied bit for bit. Inductor also keeps bf16
intermediates in fp32 inside a fused kernel, but rounds every intermediate it
stores between kernels, and eager PyTorch rounds after every op, so against
either one q/k differ by a few bf16 ULPs rather than bitwise.

Envelope / fallback
-------------------
bf16, contiguous ``[B, S, 3*H*D]`` QKV; a power-of-two ``D``; affine
``nn.RMSNorm(D)`` q/k norms with bf16 weights; ``[S, rotary_dim]`` cos/sin
with an even ``rotary_dim <= D``. Anything else, including a grad-enabled
call, since the op has no autograd formula, runs :func:`_reference`, the
unfused diffusers path.
"""

import torch

from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

# Triton is optional: importing this module must not fail on CPU-only boxes.
try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    _HAS_TRITON = True
except Exception:  # pragma: no cover - only exercised where triton is absent
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _norm_rope(
        x_ptrs,
        partner_ptrs,
        w_ptr,
        cols,
        partner,
        mask,
        cos,
        sin,
        sign,
        rotated,
        eps,
        D: tl.constexpr,
    ):
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_partner = tl.load(partner_ptrs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)[:, None]
        weight = tl.load(w_ptr + cols).to(tl.float32)[None, :]
        weight_partner = tl.load(w_ptr + partner).to(tl.float32)[None, :]
        y = x * rstd * weight
        y_partner = x_partner * rstd * weight_partner
        return tl.where(rotated[None, :], y * cos + sign[None, :] * y_partner * sin, y)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": rows}, num_warps=warps)
            for rows, warps in ((8, 2), (16, 4), (32, 4), (32, 8), (64, 8))
        ],
        key=["H", "D", "ROTARY_DIM", "WORLD", "CHUNK", "LARGER"],
    )
    @triton.jit
    def _qkv_norm_rope_kernel(
        qkv_ptr,
        wq_ptr,
        wk_ptr,
        cos_ptr,
        sin_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        n_rows,
        seq_len,
        qkv_token_stride,
        freq_row_stride,
        eps_q,
        eps_k,
        H: tl.constexpr,
        D: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        WORLD: tl.constexpr,
        CHUNK: tl.constexpr,
        LARGER: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < n_rows
        token = rows // H
        head = rows % H

        half = ROTARY_DIM // 2
        cols = tl.arange(0, D)
        rotated = cols < ROTARY_DIM
        partner = tl.where(cols < half, cols + half, tl.where(rotated, cols - half, cols))
        sign = tl.where(cols < half, -1.0, 1.0)

        mask = row_mask[:, None]
        src = qkv_ptr + (token.to(tl.int64) * qkv_token_stride + head * D)[:, None]

        # Head h goes to rank h // (H / WORLD); its local heads are split into LARGER chunks of
        # CHUNK + 1 then chunks of CHUNK, each stored as [rank, token, chunk head, D] after the last.
        rank = head // (H // WORLD)
        local = head % (H // WORLD)
        in_larger = local < LARGER * (CHUNK + 1)
        start = tl.where(
            in_larger,
            local // (CHUNK + 1) * (CHUNK + 1),
            LARGER * (CHUNK + 1) + (local - LARGER * (CHUNK + 1)) // CHUNK * CHUNK,
        )
        width = tl.where(in_larger, CHUNK + 1, CHUNK)
        n_tokens = (n_rows // H).to(tl.int64)
        dst_row = start * WORLD * n_tokens + (rank * n_tokens + token) * width + local - start
        dst = (dst_row * D)[:, None] + cols[None, :]

        freq = ((token % seq_len).to(tl.int64) * freq_row_stride)[:, None] + cols[None, :]
        freq_mask = mask & rotated[None, :]
        cos = tl.load(cos_ptr + freq, mask=freq_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + freq, mask=freq_mask, other=0.0).to(tl.float32)

        query = _norm_rope(
            src + cols[None, :],
            src + partner[None, :],
            wq_ptr,
            cols,
            partner,
            mask,
            cos,
            sin,
            sign,
            rotated,
            eps_q,
            D,
        )
        tl.store(q_ptr + dst, query.to(tl.bfloat16), mask=mask)

        key_src = src + H * D
        key = _norm_rope(
            key_src + cols[None, :],
            key_src + partner[None, :],
            wk_ptr,
            cols,
            partner,
            mask,
            cos,
            sin,
            sign,
            rotated,
            eps_k,
            D,
        )
        tl.store(k_ptr + dst, key.to(tl.bfloat16), mask=mask)

        value = tl.load(src + 2 * H * D + cols[None, :], mask=mask)
        tl.store(v_ptr + dst, value, mask=mask)

    def _launch(qkv, wq, wk, cos, sin, heads, eps_q, eps_k, shape, world, chunk, larger):
        batch, seq_len, width = qkv.shape
        query, key, value = qkv.new_empty(shape), qkv.new_empty(shape), qkv.new_empty(shape)
        n_rows = batch * seq_len * heads

        def grid(meta):
            return (triton.cdiv(n_rows, meta["BLOCK_ROWS"]),)

        wrap_triton(_qkv_norm_rope_kernel)[grid](
            qkv,
            wq,
            wk,
            cos,
            sin,
            query,
            key,
            value,
            n_rows,
            seq_len,
            qkv.stride(1),
            cos.stride(0),
            eps_q,
            eps_k,
            H=heads,
            D=width // (3 * heads),
            ROTARY_DIM=cos.shape[-1],
            WORLD=world,
            CHUNK=chunk,
            LARGER=larger,
        )
        return query, key, value

    @triton_op("xfuser::minimax_h3_qkv_norm_rope", mutates_args=())
    def _qkv_norm_rope(
        qkv: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        heads: int,
        eps_q: float,
        eps_k: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, width = qkv.shape
        shape = (batch, seq_len, heads, width // (3 * heads))
        return _launch(qkv, wq, wk, cos, sin, heads, eps_q, eps_k, shape, 1, heads, 0)

    @triton_op("xfuser::minimax_h3_qkv_norm_rope_ulysses", mutates_args=())
    def _qkv_norm_rope_ulysses(
        qkv: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        heads: int,
        eps_q: float,
        eps_k: float,
        world_size: int,
        chunk: int,
        larger: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat q, k and v, each holding its head chunks' send buffers one after another."""
        return _launch(qkv, wq, wk, cos, sin, heads, eps_q, eps_k, (qkv.numel() // 3,), world_size, chunk, larger)


def _reference(qkv, norm_q, norm_k, cos, sin, heads):
    query, key, value = (t.unflatten(-1, (heads, -1)) for t in qkv.chunk(3, dim=-1))
    query = _apply_rotary_emb(norm_q(query), cos, sin)
    key = _apply_rotary_emb(norm_k(key), cos, sin)
    return query, key, value


def _fused_supported(qkv, norm_q, norm_k, cos, sin, heads) -> bool:
    if not _HAS_TRITON or torch.is_grad_enabled():
        return False
    if qkv.device.type != "cuda" or qkv.dtype != torch.bfloat16 or qkv.dim() != 3:
        return False
    if not qkv.is_contiguous() or heads <= 0 or qkv.shape[-1] % (3 * heads):
        return False
    head_dim = qkv.shape[-1] // (3 * heads)
    if head_dim & (head_dim - 1):
        return False
    for norm in (norm_q, norm_k):
        if type(norm) is not torch.nn.RMSNorm or norm.eps is None:
            return False
        if tuple(norm.normalized_shape) != (head_dim,) or norm.weight is None:
            return False
        if norm.weight.dtype != qkv.dtype or not norm.weight.is_contiguous():
            return False
    rotary_dim = cos.shape[-1]
    if cos.dim() != 2 or cos.shape != sin.shape or cos.shape[0] != qkv.shape[1]:
        return False
    if cos.dtype != sin.dtype or cos.dtype not in (torch.float32, torch.bfloat16):
        return False
    if cos.stride() != sin.stride() or cos.stride(-1) != 1:
        return False
    return 0 < rotary_dim <= head_dim and rotary_dim % 2 == 0


def minimax_h3_qkv_norm_rope(
    qkv: torch.Tensor,
    norm_q: torch.nn.Module,
    norm_k: torch.nn.Module,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV projection into normed, rotated q/k and v, each ``[B, S, H, D]``.

    The fused kernel returns contiguous tensors; the reference returns
    contiguous q/k and a strided view of v.
    """
    if _fused_supported(qkv, norm_q, norm_k, cos, sin, heads):
        return torch.ops.xfuser.minimax_h3_qkv_norm_rope(
            qkv,
            norm_q.weight,
            norm_k.weight,
            cos,
            sin,
            heads,
            float(norm_q.eps),
            float(norm_k.eps),
        )
    return _reference(qkv, norm_q, norm_k, cos, sin, heads)


def _chunk_layout(chunk_sizes):
    """``(chunk, larger)`` when the sizes are ``larger`` chunks of ``chunk + 1`` heads then chunks of ``chunk``."""
    chunk = chunk_sizes[-1]
    larger = sum(size == chunk + 1 for size in chunk_sizes)
    if chunk < 1 or list(chunk_sizes) != [chunk + 1] * larger + [chunk] * (len(chunk_sizes) - larger):
        return None
    return chunk, larger


def minimax_h3_qkv_norm_rope_ulysses(
    qkv: torch.Tensor,
    norm_q: torch.nn.Module,
    norm_k: torch.nn.Module,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    world_size: int,
    chunk_sizes: list[int],
) -> list[list[torch.Tensor]]:
    """:func:`minimax_h3_qkv_norm_rope`, written as USP's chunked send buffers.

    Returns what :func:`~xfuser.model_executor.layers.usp.ulysses_send_buffers` returns for the
    same q, k and v.
    """
    from xfuser.model_executor.layers.usp import ulysses_send_buffers

    layout = _chunk_layout(chunk_sizes)
    if layout is None or not _fused_supported(qkv, norm_q, norm_k, cos, sin, heads):
        query, key, value = _reference(qkv, norm_q, norm_k, cos, sin, heads)
        return ulysses_send_buffers(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), world_size, chunk_sizes
        )

    batch, seq_len, width = qkv.shape
    head_dim = width // (3 * heads)
    flat = torch.ops.xfuser.minimax_h3_qkv_norm_rope_ulysses(
        qkv,
        norm_q.weight,
        norm_k.weight,
        cos,
        sin,
        heads,
        float(norm_q.eps),
        float(norm_k.eps),
        world_size,
        *layout,
    )
    buffers, offset = [], 0
    for size in chunk_sizes:
        numel = world_size * batch * seq_len * size * head_dim
        buffers.append([t[offset : offset + numel].view(world_size, batch, seq_len, size, head_dim) for t in flat])
        offset += numel
    return buffers
