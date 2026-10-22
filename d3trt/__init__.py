"""d3-trt: TensorRT-first serving runtime for the d3 code-readout decision model.

The package keeps the d3 frontend (prompt rendering, answer codes, readout
offsets, Box/Track composition, calibration and error handling) and replaces the
backbone forward pass with a prefill-only runtime built around TensorRT-LLM's
PyTorch modules and kernels.

Two backends ship:

``hf``
    Transformers Qwen3.5 backbone + the reference state-once algorithm. Slow but
    exact; used as the parity oracle and as the CPU fallback.
``trtllm``
    TensorRT-LLM `Linear`/`RMSNorm`/`GatedMLP`/Gated-DeltaNet modules, FlashAttention
    for the full-attention layers, a shared-prefix paged KV cache and the
    state-once scheduler. This is the performance path.

Nothing in the d3 prompt/answer contract is re-implemented: the reference
``d3_runtime`` / ``d3_format`` / ``d3_compose`` modules from the model package
are imported at runtime.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import D3Config, load_config
from .engine import DecisionEngine
from .types import DecisionRequest, DecisionResponse, PreparedState

__all__ = [
    "D3Config",
    "DecisionEngine",
    "DecisionRequest",
    "DecisionResponse",
    "PreparedState",
    "__version__",
    "load_config",
]
