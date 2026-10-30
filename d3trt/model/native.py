"""Dependency-free building blocks for the d3 backbone.

These replace the pieces the fast path used to borrow from TensorRT-LLM
(``Linear``, ``Embedding``, ``RMSNorm``) and from Transformers (config loading,
gated RMSNorm, the Qwen3.5 rotary embedding). The math is copied from the
Transformers ``qwen3_5`` reference so parity is exact; the projections are plain
``nn.Linear`` and run on the same cuBLAS(-Lt) GEMMs TensorRT-LLM's ``Linear``
dispatches to at TP=1.

Nothing here imports TensorRT-LLM or Transformers, so the native backend runs in
any environment with torch (plus FLA / flash-attn when installed).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_text_config(model_dir: str | Path) -> SimpleNamespace:
    """Read ``text_config`` from ``config.json`` as an attribute namespace."""
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    text = dict(raw.get("text_config", raw))
    rope = dict(text.get("rope_parameters") or {})
    rope.setdefault("rope_type", "default")
    rope.setdefault("rope_theta", text.get("rope_theta", 10000.0))
    rope.setdefault("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))
    text["rope_parameters"] = rope
    text.setdefault("head_dim", text["hidden_size"] // text["num_attention_heads"])
    return SimpleNamespace(**text)


class Linear(nn.Linear):
    """Bias-free projection with the attributes the weight loader expects."""

    tp_size = 1

    def __init__(self, in_features: int, out_features: int, *, dtype: torch.dtype, device: Any = None):
        super().__init__(in_features, out_features, bias=False, dtype=dtype, device=device)


class RMSNorm(nn.Module):
    """Qwen3.5 (Gemma-style) RMSNorm: ``norm(x) * (1 + w)`` computed in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6, *, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class RMSNormGated(nn.Module):
    """Norm-before-gate RMSNorm used on the GDN output: ``w * norm(x) * silu(z)``."""

    def __init__(self, dim: int, eps: float = 1e-6, *, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        out = self.weight * xf.to(dtype)
        return (out * F.silu(gate.float())).to(dtype)


class RotaryEmbedding(nn.Module):
    """Qwen3.5 interleaved multimodal RoPE over a partial head dimension.

    ``forward(x, position_ids)`` takes ``position_ids`` of shape ``[3, B, L]``
    (temporal, height, width) and returns ``(cos, sin)`` of shape
    ``[B, L, rotary_dim]``. For text all three rows are equal and this reduces to
    ordinary 1-D RoPE.
    """

    def __init__(self, config: Any, device: Any = None):
        super().__init__()
        rope = config.rope_parameters
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope_type {rope['rope_type']!r} is not supported natively")
        dim = int(config.head_dim * rope.get("partial_rotary_factor", 1.0))
        base = float(rope["rope_theta"])
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq.to(device), persistent=False)
        self.mrope_section = list(rope.get("mrope_section", [11, 11, 10]))
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = position_ids.expand(3, -1, -1)
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device)
        cos = self._recompose(freqs.cos() * self.attention_scaling)
        sin = self._recompose(freqs.sin() * self.attention_scaling)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _recompose(self, freq: torch.Tensor) -> torch.Tensor:
        thw = freq[0]
        for dim, offset in enumerate((1, 2), start=1):
            thw[..., slice(offset, self.mrope_section[dim] * 3, 3)] = freq[dim, ..., slice(offset, self.mrope_section[dim] * 3, 3)]
        return torch.cat((thw, thw), dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial RoPE: rotate the first ``cos.shape[-1]`` channels, pass the rest."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rot = cos.shape[-1]
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    k_rot, k_pass = k[..., :rot], k[..., rot:]
    q_out = torch.cat([q_rot * cos + _rotate_half(q_rot) * sin, q_pass], dim=-1)
    k_out = torch.cat([k_rot * cos + _rotate_half(k_rot) * sin, k_pass], dim=-1)
    return q_out, k_out
