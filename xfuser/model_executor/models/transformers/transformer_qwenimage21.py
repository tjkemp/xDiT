import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    QwenImage21FlexAttnProcessor,
    _qwenimage21_prepare_qkv,
)

from xfuser.model_executor.layers.usp import attention


def _is_unmasked_decode(attention_mask, segments) -> bool:
    """Whether this call is a cached decode step that needs no mask.

    The model hands prefill its block-causal structure as ``segments`` (the eager processor) or a flex
    ``BlockMask`` in ``attention_mask`` (the flex processor). Decode gets neither, only a key-padding mask
    when the prompt is right-padded; the pipeline drops that mask when every token is valid.
    """
    return segments is None and attention_mask is None


def _xfuser_decode_attention(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice):
    query, key, value, _ = _qwenimage21_prepare_qkv(
        attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
    )
    hidden_states = attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
    hidden_states = hidden_states.flatten(2, 3).type_as(query)
    hidden_states = attn.to_out[0](hidden_states)
    return attn.to_out[1](hidden_states)


class _xFuserQwenImage21DecodeMixin:
    """Runs unmasked decode through xDiT's attention backend; every other call goes to the diffusers processor.

    Subclassing the diffusers processors, rather than wrapping them, matters: the model picks which prefill
    metadata to build from ``isinstance`` checks against them.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask=None,
        rotary_emb=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ) -> torch.Tensor:
        if _is_unmasked_decode(attention_mask, segments):
            return _xfuser_decode_attention(
                attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
            )
        return super().__call__(
            attn,
            hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            layer_cache=layer_cache,
            kv_cache_mode=kv_cache_mode,
            cache_write_slice=cache_write_slice,
            segments=segments,
            key_valid=key_valid,
        )


class xFuserQwenImage21AttnProcessor(_xFuserQwenImage21DecodeMixin, QwenImage21AttnProcessor):
    """Eager path: exact per-segment SDPA prefill, xDiT-backend decode."""


class xFuserQwenImage21FlexAttnProcessor(_xFuserQwenImage21DecodeMixin, QwenImage21FlexAttnProcessor):
    """Compiled path: single flex_attention prefill, xDiT-backend decode."""
