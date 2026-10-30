"""Kernel selection for the two hot operations: full attention and Gated DeltaNet.

Every function here has a pure-PyTorch fallback so the runtime is correct on CPU
and on a GPU box before the optional kernel packages are installed. The fast
path prefers, in order:

* FlashAttention 3 / 2 for the state pass and the per-question fallback;
* the shared-prefix Triton kernel for the question pass (no state KV copy);
* FLA ``chunk_gated_delta_rule`` / ``fused_recurrent_gated_delta_rule`` and
  ``causal_conv1d`` for the GDN layers;
* TensorRT-LLM's own fused GDN helpers when FLA is not present.

The selection is recorded once at start-up and exposed in ``kernel_report()`` so
a benchmark run can state exactly which kernel served it.
"""

from __future__ import annotations

import functools
import importlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F

_REPORT: dict[str, str] = {}


def kernel_report() -> dict[str, str]:
    return dict(_REPORT)


def _try_import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None


def flash_attn_available() -> bool:
    return flash_attn_module() is not None


def fla_available(kind: str = "chunk") -> bool:
    chunk, recurrent, conv = fla_modules()
    if kind == "chunk":
        return chunk is not None
    if kind == "recurrent":
        return recurrent is not None
    if kind == "conv":
        return conv is not None
    raise ValueError(f"unknown FLA kernel {kind!r}")


def triton_available() -> bool:
    try:
        from ..kernels.prefix_bidir import triton_available as available

        return bool(available())
    except Exception:  # noqa: BLE001
        return False


@functools.lru_cache(maxsize=1)
def flash_attn_module() -> Any:
    mod = _try_import("flash_attn")
    if mod is not None and hasattr(mod, "flash_attn_func"):
        _REPORT["flash_attn"] = getattr(mod, "__version__", "present")
        return mod
    _REPORT["flash_attn"] = "missing"
    return None


@functools.lru_cache(maxsize=1)
def fla_modules() -> tuple[Any, Any, Any]:
    """Return ``(chunk, recurrent, conv)`` or Nones."""
    chunk = recurrent = conv = None
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as chunk  # type: ignore
    except Exception:  # noqa: BLE001
        chunk = None
    try:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as recurrent  # type: ignore
    except Exception:  # noqa: BLE001
        recurrent = None
    try:
        from causal_conv1d import causal_conv1d_fn as conv  # type: ignore
    except Exception:  # noqa: BLE001
        conv = None
    _REPORT["fla_chunk"] = "flash-linear-attention" if chunk else "missing"
    _REPORT["fla_recurrent"] = "flash-linear-attention" if recurrent else "missing"
    _REPORT["causal_conv1d"] = "causal-conv1d" if conv else "missing"
    return chunk, recurrent, conv


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    return x.repeat_interleave(groups, dim=1)


def sdpa_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Bidirectional attention over one state block, [B, H, S, D]."""
    try:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, scale=scale, is_causal=False, enable_gqa=True
        )
    except TypeError:
        groups = q.shape[1] // k.shape[1]
        if groups > 1:
            k = repeat_kv(k, groups)
            v = repeat_kv(v, groups)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, scale=scale, is_causal=False)


def sdpa_questions(
    q: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    k_q: torch.Tensor,
    v_q: torch.Tensor,
    lengths: Sequence[int],
    *,
    scale: float,
) -> torch.Tensor:
    """Per-row bidirectional attention over ``state + own question``, all rows at once.

    ``q``/``k_q``/``v_q`` are [B, H, L, D] right-padded; ``k_state``/``v_state``
    are [H_kv, S, D] (shared by every row). Returns [B, H, L, D]. Padding keys are
    masked out.

    The call uses a boolean mask, as the reference ``StateOnce._attention`` does,
    so SDPA picks the same kernel and the result is bitwise identical to the
    oracle; an unmasked GQA call picks another kernel whose bf16 rounding
    compounds to ~0.04 in answer probabilities over 64 layers. Instead of
    repeating each KV head for its query heads (6x copies of the state KV), the
    query heads of a group are folded into the sequence dimension: every query
    row sees the same keys in the same order, so the output is unchanged
    (verified bitwise) and the call is ~1.5x faster.
    """
    rows, heads, width, dim = q.shape
    kv_heads, state = k_state.shape[0], k_state.shape[1]
    groups = heads // kv_heads
    keys = torch.cat([k_state.unsqueeze(0).expand(rows, -1, -1, -1), k_q], dim=2)
    values = torch.cat([v_state.unsqueeze(0).expand(rows, -1, -1, -1), v_q], dim=2)
    present = torch.arange(width, device=q.device)[None] < torch.as_tensor(list(lengths), device=q.device)[:, None]
    mask = torch.ones(rows, 1, width, state + width, dtype=torch.bool, device=q.device)
    mask[:, 0, :, state:] = present[:, None, :]
    # Head h = kv * groups + j reads KV head kv; fold j into the query rows.
    folded = q.reshape(rows, kv_heads, groups * width, dim)
    out = F.scaled_dot_product_attention(folded, keys, values, attn_mask=mask.repeat(1, 1, groups, 1), scale=scale)
    return out.reshape(rows, heads, width, dim)


def flash_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """FlashAttention over one state block.

    ``q`` [B, H, S, D], ``k``/``v`` [B, H_kv, S, D]. FlashAttention wants
    [B, S, H, D]; GQA is native.
    """
    mod = flash_attn_module()
    if mod is None or q.dtype not in (torch.float16, torch.bfloat16):
        return sdpa_state(q, k, v, scale=scale)
    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    out = mod.flash_attn_func(qh, kh, vh, dropout_p=0.0, softmax_scale=scale, causal=False)
    return out.transpose(1, 2)


def flash_questions(
    q: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    k_q: torch.Tensor,
    v_q: torch.Tensor,
    lengths: Sequence[int],
    *,
    scale: float,
) -> torch.Tensor:
    """Per-row FlashAttention over state + own question (fallback before Triton)."""
    mod = flash_attn_module()
    if mod is None or q.dtype not in (torch.float16, torch.bfloat16):
        return sdpa_questions(q, k_state, v_state, k_q, v_q, lengths, scale=scale)
    rows, heads, width, dim = q.shape
    out = q.new_zeros(q.shape)
    for row in range(rows):
        length = int(lengths[row])
        keys = torch.cat([k_state, k_q[row, :, :length]], dim=1)
        values = torch.cat([v_state, v_q[row, :, :length]], dim=1)
        out[row, :, :length] = mod.flash_attn_func(
            q[row, :, :length].transpose(0, 1).unsqueeze(0).contiguous(),
            keys.transpose(0, 1).unsqueeze(0).contiguous(),
            values.transpose(0, 1).unsqueeze(0).contiguous(),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
        ).squeeze(0).transpose(0, 1)
    return out


def question_attention(
    q: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    k_q: torch.Tensor,
    v_q: torch.Tensor,
    lengths: Sequence[int],
    *,
    scale: float,
    prefer_triton: bool = True,
    return_kernel: bool = False,
) -> Any:
    """Best available shared-prefix question attention.

    With ``return_kernel=True`` returns ``(output, kernel_name)`` so callers can
    report the kernel that actually ran rather than the one requested.
    """
    if prefer_triton and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            from ..kernels.prefix_bidir import prefix_bidir_attention

            if triton_available():
                out = prefix_bidir_attention(
                    q, k_state, v_state, k_q, v_q, lengths, scale=scale
                )
                return (out, "triton") if return_kernel else out
        except Exception:  # noqa: BLE001 - fall back, never fail a request
            pass
    if flash_attn_available() and q.dtype in (torch.float16, torch.bfloat16):
        out = flash_questions(q, k_state, v_state, k_q, v_q, lengths, scale=scale)
        return (out, "flash") if return_kernel else out
    out = sdpa_questions(q, k_state, v_state, k_q, v_q, lengths, scale=scale)
    return (out, "sdpa") if return_kernel else out


# ---------------------------------------------------------------------------
# Gated DeltaNet
# ---------------------------------------------------------------------------


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    activation: str | None,
) -> torch.Tensor:
    """Causal depthwise convolution over [B, C, L]."""
    chunk, _, conv = fla_modules()
    if conv is not None and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        return conv(x, weight, bias, activation=activation)
    # Reference torch fallback (transformers.models.qwen3_5.modeling_qwen3_5).
    _, _, _ = chunk, _, conv
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=x.shape[1])
    out = out[..., : x.shape[-1]]
    if activation == "silu":
        out = F.silu(out)
    elif activation in ("gelu", "gelu_pytorch_tanh"):
        out = F.gelu(out, approximate="tanh")
    return out


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    chunk, _, _ = fla_modules()
    if chunk is not None and q.is_cuda:
        return chunk(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    return torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    _, recurrent, _ = fla_modules()
    if recurrent is not None and q.is_cuda:
        return recurrent(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

    return torch_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
