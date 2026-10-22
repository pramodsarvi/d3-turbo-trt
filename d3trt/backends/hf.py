"""Transformers Qwen3.5 backend: parity oracle and CPU fallback.

The reference d3 runtime is the published implementation; this backend loads the
stock HuggingFace ``Qwen3_5Model`` and either

* runs the reference ``D3.run`` unchanged (media, composed requests, parity), or
* runs the package's :class:`D3Core` state-once path (text), which is the same
  math with swappable attention/GDN kernels and an optional cross-request cache.

The core path is what the TensorRT-LLM backend replaces; keeping it here means
every fast-path change can be compared against the published runtime on the same
machine, including on CPU.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import asdict
from typing import Any, Mapping

from ..errors import BackendUnavailable
from ..model.cache import StateCache
from ..model.hf_stack import HFStack
from .base import Backend


class HFBackend(Backend):
    name = "hf"

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | None = None,
        dtype: str | None = None,
        reference_dir: str | Path | None = None,
        load_state_once: bool = True,
        use_core: bool = True,
        state_cache_bytes: int | None = None,
    ):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.reference_dir = reference_dir
        self.device = device
        self.dtype = dtype
        self.load_state_once = load_state_once
        self.use_core = use_core
        self._cache_bytes = state_cache_bytes
        self.cache = StateCache(0)
        self.frontend: Any = None
        self.model: Any = None
        self.readout: Any = None
        self.stack: HFStack | None = None
        self.last_timings: dict[str, Any] = {}
        self._loaded = False
        self._load_seconds = 0.0

    def attach(self, frontend: Any) -> None:
        self.frontend = frontend

    def start(self) -> None:
        if self._loaded:
            return
        if self.frontend is None:
            raise RuntimeError("HFBackend.attach(frontend) must be called before start()")
        import torch
        from safetensors.torch import load_file

        rt = self.frontend.rt
        d3 = self.frontend.d3
        device = self.device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise BackendUnavailable("hf", "CUDA was requested but is not available")
        dtype_name = self.dtype or ("bfloat16" if str(device).startswith("cuda") else "float32")
        torch_dtype = getattr(torch, dtype_name)

        started = time.perf_counter()
        backbone_class = rt.backbone_class(d3.backbone_type)
        self.model = backbone_class.from_pretrained(
            str(self.model_dir),
            dtype=torch_dtype,
            attn_implementation="sdpa",
            device_map={"": str(device)},
        )
        weight = load_file(str(self.model_dir / "readout.safetensors"))["weight"]
        hidden = self.model.config.text_config.hidden_size
        if tuple(weight.shape) != (self.frontend.config.max_options, hidden):
            raise ValueError(
                f"readout weight has shape {tuple(weight.shape)}, expected "
                f"{(self.frontend.config.max_options, hidden)}"
            )
        self.readout = weight.to(device, getattr(torch, d3.readout_dtype))
        self.model.eval().requires_grad_(False)

        # Bind the loaded model into the reference object. This lets us reuse
        # the reference run for media and composed requests unchanged.
        d3.backbone = self.model
        d3.readout = self.readout
        d3.device = torch.device(device)
        d3.kernels = rt.kernel_report() if d3.backbone_type == "qwen3_5" else {}
        d3.encoder = rt.PromptEncoder(self.model.language_model) if d3.blocked else None
        d3.state_once = (
            rt.StateOnce(self.model.language_model)
            if d3.blocked and self.load_state_once and os.environ.get("D3_STATE_ONCE", "1") != "0"
            else None
        )
        d3.fast = None
        d3.state_once_fused = None
        d3.loaded_seconds = time.perf_counter() - started

        budget = (
            self._cache_bytes
            if self._cache_bytes is not None
            else self.frontend.config.state_cache_bytes
        )
        self.cache = StateCache(budget)
        self.stack = HFStack(self.model, self.readout, device=torch.device(device))
        self.stack.pad_id = d3.pad_id
        self._loaded = True
        self._load_seconds = time.perf_counter() - started

    def run(self, prepared: Any) -> tuple[dict[str, list[float]], int]:
        if not self._loaded:
            self.start()
        assert self.frontend is not None
        d3 = self.frontend.d3
        # The core path is text-only; media goes through the reference runtime
        # (which is also the parity oracle for images/videos).
        if self.use_core and not prepared.images and not prepared.videos and prepared.runnable:
            key = None
            if self.cache.capacity_bytes > 0:
                first = prepared.runnable[0]
                state_len = prepared.states[first]
                key = self.frontend.config.state_key(list(prepared.sequences[first][:state_len]))
            import torch

            with torch.inference_mode():
                result = self.stack.run_text(
                    prepared,
                    batch_size=d3.batch_size,
                    temperature=d3.temperature,
                    readout_dtype=d3.readout_dtype,
                    cache=self.cache if key is not None else None,
                    state_key=key,
                )
            self.last_timings = dict(self.stack.last_timings)
            return result
        self.last_timings = {"path": "reference"}
        return d3.run(prepared)

    def info(self) -> Mapping[str, Any]:
        payload = {
            "backend": self.name,
            "device": str(self.device),
            "dtype": self.dtype,
            "load_seconds": round(self._load_seconds, 3),
            "state_once": bool(self.frontend and self.frontend.d3.state_once),
            "core": asdict(self.stack.core.stats) if self.stack else None,
        }
        if self.cache.capacity_bytes:
            payload["state_cache"] = self.cache.stats()
        return payload

    def close(self) -> None:
        if self.frontend is not None:
            self.frontend.d3.backbone = None
            self.frontend.d3.state_once = None
            self.frontend.d3.encoder = None
        self.model = None
        self.readout = None
        self.stack = None
        self._loaded = False
