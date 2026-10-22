"""W8A8 quantized linear: numerics of the CPU emulation and mode resolution."""

from __future__ import annotations

import pytest
import torch

from d3trt.model import quant


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("mode,limit", [("fp8", 0.06), ("int8", 0.02)])
def test_quant_linear_matches_dense(mode, limit):
    torch.manual_seed(0)
    weight = torch.randn(96, 256) * 0.02
    x = torch.randn(3, 5, 256, dtype=torch.bfloat16)
    layer = quant.QuantLinear(256, 96, mode=mode, dtype=torch.bfloat16, device="cpu")
    layer.load_full(weight.to(torch.bfloat16))
    out = layer(x)
    assert out.shape == (3, 5, 96)
    assert out.dtype == torch.bfloat16
    assert layer.kernel == "emulated"
    assert _rel_err(out, x.float() @ weight.t()) < limit


def test_per_token_scale_handles_outlier_rows_and_zero_rows():
    torch.manual_seed(0)
    weight = torch.randn(32, 64)
    x = torch.randn(4, 64)
    x[1] *= 1000.0  # an outlier token must not crush the others' precision
    x[3] = 0.0      # padded rows stay exactly zero
    layer = quant.QuantLinear(64, 32, mode="int8", dtype=torch.float32, device="cpu")
    layer.load_full(weight)
    out = layer(x)
    ref = x @ weight.t()
    assert _rel_err(out[0], ref[0]) < 0.02
    assert torch.all(out[3] == 0)


def test_weight_quantization_is_per_channel():
    weight = torch.ones(16, 32)
    weight[0] *= 100.0
    q, scale = quant.quantize_weight(weight, "int8")
    assert q.dtype == torch.int8 and scale.shape == (16,)
    assert torch.all(q == 127)
    assert scale[0] == pytest.approx(100.0 / 127)
    assert scale[1] == pytest.approx(1.0 / 127)


def test_shape_checks():
    assert quant.quantizable(5120, 17408)
    assert not quant.quantizable(5120, 40)
    with pytest.raises(ValueError):
        quant.QuantLinear(64, 40, mode="fp8", dtype=torch.bfloat16, device="cpu")
    layer = quant.QuantLinear(64, 32, mode="fp8", dtype=torch.bfloat16, device="cpu")
    with pytest.raises(ValueError):
        layer.load_full(torch.zeros(32, 48))


def test_resolve_mode_off_gpu():
    assert quant.resolve_mode("auto", "cpu") == "none"
    assert quant.resolve_mode("int8", "cpu") == "int8"
    assert quant.resolve_mode("fp8", "cpu") == "fp8"
    with pytest.raises(ValueError):
        quant.resolve_mode("nvfp4", "cpu")


def test_summarize_counts_quantized_bytes():
    layers = torch.nn.ModuleList(
        [
            quant.QuantLinear(64, 32, mode="int8", dtype=torch.float32, device="cpu"),
            torch.nn.Linear(64, 8, bias=False),
        ]
    )
    report = quant.summarize(layers.modules())
    assert report["kernels"] == {"unused": 1}
    assert report["quantized_gib"] == round(64 * 32 / 1024**3, 3)
