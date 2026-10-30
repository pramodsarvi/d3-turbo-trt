"""Weight quantization for the TensorRT-LLM stack.

The BF16 backbone is 47.7 GiB, which does not fit an L40 (48 GB) or an A100
40 GB at all. This module quantizes each projection at load time, one layer at
a time, so the BF16 copy never exists on the GPU:

* ``fp8``  - E4M3 weights with a per-output-channel scale, activations quantized
  per token at run time, ``torch._scaled_mm``. Needs FP8 tensor cores: Ada
  (sm_89, L40/L40S), Hopper, Blackwell (sm_100, sm_120 RTX Pro 6000).
* ``int8`` - the same scheme in INT8 with ``torch._int_mm``. Runs on Ampere
  (sm_80, A100), which has no FP8.

Both are W8A8 with dynamic per-token activation scales, so no calibration set
is needed. Projections whose shape the GEMMs cannot take (the tiny GDN ``a`` /
``b`` gates) stay in the activation dtype.

Every path has a pure-PyTorch emulation so the numerics are testable on CPU.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

MODES = ("auto", "none", "fp8", "int8")

_FP8 = getattr(torch, "float8_e4m3fn", None)
_FP8_MAX = 448.0
_INT8_MAX = 127.0
# Activation-amax floor; keeps an all-zero (padded) row from dividing by zero.
_EPS = 1e-12


def fp8_supported(device: Any) -> bool:
    if _FP8 is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(torch.device(device)) >= (8, 9)


def resolve_mode(mode: str, device: Any, weight_bytes: int | None = None) -> str:
    """Turn ``auto`` into a concrete mode.

    ``auto`` keeps BF16 when the weights fit with room for activations and the
    state cache, and otherwise picks the best 8-bit format the GPU supports.
    Explicit ``fp8`` on a GPU without FP8 tensor cores is an error, not a silent
    downgrade, so a benchmark always states what it ran.
    """
    if mode not in MODES:
        raise ValueError(f"unknown quant mode {mode!r}; expected one of {MODES}")
    dev = torch.device(device)
    if mode == "fp8" and dev.type == "cuda" and not fp8_supported(dev):
        major, minor = torch.cuda.get_device_capability(dev)
        raise ValueError(
            f"fp8 needs sm_89 or newer; {torch.cuda.get_device_name(dev)} is sm_{major}{minor}. "
            "Use --quant int8."
        )
    if mode != "auto":
        return mode
    if dev.type != "cuda":
        return "none"
    if weight_bytes:
        free, _ = torch.cuda.mem_get_info(dev)
        # BF16 weights plus ~20% for activations, the embedding and the state cache.
        if free >= weight_bytes * 1.2:
            return "none"
    return "fp8" if fp8_supported(dev) else "int8"


def quantizable(in_features: int, out_features: int) -> bool:
    # _scaled_mm needs both GEMM dims divisible by 16; _int_mm needs 8.
    return in_features % 16 == 0 and out_features % 16 == 0


def quantize_weight(weight: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] float -> ([N, K] fp8/int8, [N] float32 per-channel scale)."""
    w = weight.float()
    amax = w.abs().amax(dim=1).clamp_min(_EPS)
    if mode == "fp8":
        scale = amax / _FP8_MAX
        q = (w / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    elif mode == "int8":
        scale = amax / _INT8_MAX
        q = (w / scale[:, None]).round().clamp(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    else:
        raise ValueError(f"cannot quantize weights to {mode!r}")
    return q.contiguous(), scale.contiguous()


def quantize_rows(x: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """[M, K] activations -> (quantized [M, K], [M, 1] float32 per-token scale)."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(_EPS)
    if mode == "fp8":
        scale = amax / _FP8_MAX
        return (xf / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8), scale
    scale = amax / _INT8_MAX
    return (xf / scale).round().clamp(-_INT8_MAX, _INT8_MAX).to(torch.int8), scale


class QuantLinear(nn.Module):
    """A bias-free W8A8 linear that stands in for a TRT-LLM ``Linear`` (TP=1)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        mode: str,
        dtype: torch.dtype,
        device: Any,
    ):
        super().__init__()
        if mode not in ("fp8", "int8"):
            raise ValueError(f"QuantLinear needs fp8 or int8, got {mode!r}")
        if not quantizable(in_features, out_features):
            raise ValueError(f"{out_features}x{in_features} is not a quantizable GEMM shape")
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.dtype = dtype
        self.tp_size = 1
        storage = _FP8 if mode == "fp8" else torch.int8
        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=storage, device=device)
        )
        self.register_buffer("weight_scale", torch.ones(out_features, dtype=torch.float32, device=device))
        # "rowwise" / "tensorwise" for fp8 (decided on first call), "int_mm" for int8,
        # "emulated" off-GPU.
        self.kernel: str | None = None

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, mode={self.mode}"

    @torch.no_grad()
    def load_full(self, full: torch.Tensor) -> None:
        if tuple(full.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"linear shape mismatch: module {(self.out_features, self.in_features)} "
                f"vs checkpoint {tuple(full.shape)}"
            )
        q, scale = quantize_weight(full.to(self.weight.device), self.mode)
        self.weight.copy_(q)
        self.weight_scale.copy_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        if flat.is_cuda:
            out = self._fp8(flat) if self.mode == "fp8" else self._int8(flat)
        else:
            out = self._emulated(flat)
        return out.reshape(*shape, self.out_features)

    # ------------------------------------------------------------------
    def _emulated(self, flat: torch.Tensor) -> torch.Tensor:
        self.kernel = "emulated"
        q, scale = quantize_rows(flat, self.mode)
        acc = q.float() @ self.weight.float().t()
        return (acc * scale * self.weight_scale[None, :]).to(self.dtype)

    def _fp8(self, flat: torch.Tensor) -> torch.Tensor:
        rows = flat.shape[0]
        # cuBLASLt FP8 wants every dimension a multiple of 16.
        padded = _pad_rows(flat, 16)
        q, scale = quantize_rows(padded, "fp8")
        if self.kernel in (None, "rowwise"):
            try:
                out = torch._scaled_mm(
                    q,
                    self.weight.t(),
                    scale_a=scale,
                    scale_b=self.weight_scale[None, :],
                    out_dtype=self.dtype,
                )
                self.kernel = "rowwise"
                return out[:rows]
            except (RuntimeError, NotImplementedError):
                # Rowwise scaling is missing on some torch/arch pairs (sm_89 before
                # torch 2.7, some sm_120 builds). Fall back to one activation scale
                # and apply the channel scales in a separate multiply.
                self.kernel = "tensorwise"
        q, scale = _tensorwise_fp8(padded)
        one = torch.ones((), dtype=torch.float32, device=flat.device)
        out = torch._scaled_mm(q, self.weight.t(), scale_a=scale, scale_b=one, out_dtype=torch.float32)
        return (out[:rows] * self.weight_scale[None, :]).to(self.dtype)

    def _int8(self, flat: torch.Tensor) -> torch.Tensor:
        self.kernel = "int_mm"
        rows = flat.shape[0]
        # torch._int_mm needs M > 16 and M, K, N multiples of 8.
        padded = _pad_rows(flat, 8, minimum=24)
        q, scale = quantize_rows(padded, "int8")
        acc = torch._int_mm(q, self.weight.t())
        return (acc[:rows].float() * scale[:rows] * self.weight_scale[None, :]).to(self.dtype)


def _pad_rows(x: torch.Tensor, multiple: int, *, minimum: int = 0) -> torch.Tensor:
    rows = x.shape[0]
    target = max(minimum, math.ceil(rows / multiple) * multiple)
    if target == rows:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, target - rows))


def _tensorwise_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    scale = (xf.abs().amax().clamp_min(_EPS) / _FP8_MAX).reshape(())
    return (xf / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8), scale


def make_linear(
    in_features: int,
    out_features: int,
    *,
    quant: str,
    dtype: torch.dtype,
    mapping: Any,
    tensor_parallel_mode: Any,
    device: Any,
) -> nn.Module:
    """TRT-LLM ``Linear`` (native ``nn.Linear`` when ``mapping`` is None) for BF16,
    :class:`QuantLinear` for fp8/int8."""
    if quant in ("fp8", "int8") and quantizable(in_features, out_features):
        return QuantLinear(in_features, out_features, mode=quant, dtype=dtype, device=device)
    if mapping is None:
        from .native import Linear as NativeLinear

        return NativeLinear(in_features, out_features, dtype=dtype, device=device)
    from tensorrt_llm._torch.modules.linear import Linear

    return Linear(
        in_features,
        out_features,
        bias=False,
        dtype=dtype,
        mapping=mapping,
        tensor_parallel_mode=tensor_parallel_mode,
    )


def summarize(modules: Any) -> dict[str, Any]:
    """Bytes held by quantized and unquantized linear weights, and kernels used."""
    quantized = dense = 0
    kernels: dict[str, int] = {}
    for module in modules:
        if isinstance(module, QuantLinear):
            quantized += module.weight.numel() * module.weight.element_size()
            key = module.kernel or "unused"
            kernels[key] = kernels.get(key, 0) + 1
        elif hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor):
            if module.weight.dim() == 2:
                dense += module.weight.numel() * module.weight.element_size()
    return {
        "quantized_gib": round(quantized / 1024**3, 3),
        "dense_linear_gib": round(dense / 1024**3, 3),
        "kernels": kernels,
    }
