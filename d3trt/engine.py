"""Decision engine: frontend + prefill-only backend.

This is the only object the server and the benchmark harness talk to. It keeps
the reference d3 semantics for validation, prompt rendering, answer assembly and
Box/Track composition, and delegates only the backbone forward pass.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import backends
from .config import D3Config, load_config
from .errors import MediaNotSupported
from .reference import Frontend
from .types import DecisionRequest


class DecisionEngine:
    """Serve System One decisions with the selected backend.

    Parameters
    ----------
    model_dir:
        d3 checkpoint directory (``config.json``, ``decision_config.json``,
        ``readout.safetensors``, the reference ``d3_runtime.py`` and weights).
    backend:
        ``"hf"``, ``"trtllm"`` or ``"auto"``.
    device:
        torch device for the HF backend / TensorRT-LLM rank device.
    fallback:
        Load the HF backend for composed Box/Track requests and media when the
        fast backend cannot serve them. It is loaded lazily; on a GPU node set
        ``fallback_device="cpu"`` (the default) or a second GPU so the two
        weight copies do not compete for one device.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        backend: str = "auto",
        device: str | None = None,
        dtype: str | None = None,
        batch_size: int = 8,
        reference_dir: str | Path | None = None,
        state_cache_bytes: int | None = None,
        fallback: bool = False,
        fallback_device: str = "cpu",
        permutation_average: bool | None = None,
        max_length: int | None = None,
        backend_options: Mapping[str, Any] | None = None,
    ):
        self.config: D3Config = load_config(model_dir, state_cache_bytes=state_cache_bytes)
        self.frontend = Frontend(
            self.config.root,
            reference_dir=reference_dir,
            config=self.config,
            batch_size=batch_size,
            device=device or "cpu",
            permutation_average=permutation_average,
            max_length=max_length,
        )
        self.backend_name = backends.detect() if backend == "auto" else backend
        options = dict(backend_options or {})
        options.setdefault("device", device)
        options.setdefault("dtype", dtype)
        if state_cache_bytes is not None:
            options.setdefault("state_cache_bytes", state_cache_bytes)
        self.backend = backends.create(
            self.backend_name,
            self.config.root,
            reference_dir=reference_dir,
            **options,
        )
        self.backend.attach(self.frontend)
        self.batch_size = batch_size
        self._fallback: Any = None
        self._fallback_requested = fallback
        self._fallback_device = fallback_device
        self._fallback_dtype = dtype
        self._fallback_error: str | None = None
        self._started = False
        self._load_seconds = 0.0
        self._last_timings: dict[str, Any] = {}
        self.profile_sync = False

    # ------------------------------------------------------------------
    def start(self) -> "DecisionEngine":
        if self._started:
            return self
        started = time.perf_counter()
        self.backend.start()
        self._load_seconds = time.perf_counter() - started
        self._started = True
        return self

    def _ensure_fallback(self) -> Any:
        if self._fallback is not None:
            return self._fallback
        if not self._fallback_requested:
            raise MediaNotSupported(
                "this request needs the reference HF backend (images, videos, "
                "assets or Box/Track). Create the engine with fallback=True, or "
                "use backend='hf'."
            )
        if self._fallback_error is not None:
            raise MediaNotSupported(
                f"the HF fallback failed to load: {self._fallback_error}"
            )
        try:
            reference_dir = Path(self.frontend.rt.__file__).parent
            fallback = backends.create(
                "hf",
                self.config.root,
                reference_dir=reference_dir,
                device=self._fallback_device,
                dtype=self._fallback_dtype,
                use_core=False,
            )
            fallback.attach(self.frontend)
            fallback.start()
            self._fallback = fallback
            return fallback
        except Exception as exc:  # noqa: BLE001 - a missing fallback is not fatal to text
            self._fallback_error = f"{type(exc).__name__}: {exc}"
            raise MediaNotSupported(
                f"the HF fallback failed to load: {self._fallback_error}"
            ) from exc

    def close(self) -> None:
        try:
            self.backend.close()
        finally:
            if self._fallback is not None:
                self._fallback.close()
            self._started = False

    # ------------------------------------------------------------------
    def _composed(self, request: DecisionRequest) -> dict[str, Any]:
        fallback = self._ensure_fallback()
        d3 = self.frontend.d3
        d3.max_length = d3.question_cap(request.long_context)
        try:
            return d3._system_one_composed(
                request.state,
                request.questions,
                list(request.images or []),
                list(request.videos or []),
                dict(request.assets or {}),
            )
        finally:
            d3.max_length = d3.question_cap(False)

    def system_one(
        self,
        *,
        state: Any = None,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
        assets: Mapping[str, Any] | None = None,
        long_context: bool = False,
    ) -> dict[str, Any]:
        if not self._started:
            self.start()
        composed = bool(assets) or any(
            isinstance(q, Mapping) and q.get("type") in ("box", "track")
            for q in questions.values()
        )
        if composed:
            return self._composed(
                DecisionRequest(
                    state=state,
                    questions=questions,
                    images=images,
                    videos=videos,
                    assets=assets,
                    long_context=long_context,
                )
            )

        d3 = self.frontend.d3
        d3.max_length = d3.question_cap(long_context)
        try:
            t0 = time.perf_counter()
            prepared = self.frontend.prepare(state, questions, images, videos)
            t1 = time.perf_counter()
            if self.profile_sync:
                _sync(self)
            probabilities, tokens = self.run(prepared)
            if self.profile_sync:
                _sync(self)
            t2 = time.perf_counter()
            response = self.frontend.respond(prepared, probabilities, tokens)
            t3 = time.perf_counter()
        finally:
            d3.max_length = d3.question_cap(False)
        self._last_timings = {
            "prepare_ms": round((t1 - t0) * 1000, 3),
            "backend_ms": round((t2 - t1) * 1000, 3),
            "respond_ms": round((t3 - t2) * 1000, 3),
            "backend_profile": dict(getattr(self.backend, "last_timings", {}) or {}),
        }
        response.setdefault("timings", {}).update(self._last_timings)
        response.setdefault("backend", self.backend_name)
        return response

    # ------------------------------------------------------------------
    def run(self, prepared: Any) -> tuple[dict[str, list[float]], int]:
        if not self._started:
            self.start()
        try:
            return self.backend.run(prepared)
        except MediaNotSupported:
            if not self._fallback_requested:
                raise
            return self._ensure_fallback().run(prepared)

    def state_key(self, prepared: Any) -> tuple[str, int]:
        """Return ``(state_key, state_len)`` for a prepared text request."""
        first = prepared.keys[0]
        sequence = prepared.sequences[first]
        state = prepared.states[first]
        return self.config.state_key(sequence[:state]), state

    def info(self) -> dict[str, Any]:
        if not self._started:
            self.start()
        payload = {
            "model": self.config.model_name,
            "backend": self.backend_name,
            "load_seconds": round(self._load_seconds, 3),
            "batch_size": self.batch_size,
            "config": self.config.describe(),
            "frontend": self.frontend.runtime_info(),
            "backend_info": dict(self.backend.info()),
            "fallback": bool(self._fallback),
            "fallback_requested": self._fallback_requested,
        }
        if self._fallback_error:
            payload["fallback_error"] = self._fallback_error
        return payload

    def runtime_info(self) -> dict[str, Any]:
        return self.info()


def _sync(engine: Any | None = None) -> None:
    try:
        import torch

        device = None
        if engine is not None:
            stack = getattr(engine.backend, "stack", None)
            device = getattr(stack, "device", None)
            if device is None:
                device = getattr(getattr(engine.frontend, "d3", None), "device", None)
        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            torch.cuda.synchronize(device)
        elif (device is None or str(device).startswith("mps")) and getattr(
            torch.backends, "mps", None
        ) is not None and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False
