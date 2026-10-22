"""HTTP serving for the d3-trt engine.

The reference package ships ``d3_server.py``: a dependency-free HTTP/1.1 server
with the exact System One contract (``GET /health``, ``GET /v1/models``,
``POST /v1/systemone``), bearer auth, chunked bodies, media size limits and the
422 shapes for malformed questions. Rather than re-implement that contract, this
module wraps :class:`DecisionEngine` in a small adapter and reuses the reference
server unchanged.

The adapter is the only place where the backend and the reference frontend meet:
``prepare``/``respond`` are the reference methods, ``run`` is the selected
backend.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Mapping

from .engine import DecisionEngine
from .errors import MediaNotSupported
from .reference import import_server


class ServedModel:
    """The object the reference ``d3_server.Decider`` drives."""

    def __init__(self, engine: DecisionEngine):
        self.engine = engine
        self.d3 = engine.frontend.d3
        self._lock = threading.RLock()

    # ---- identity / limits -------------------------------------------
    @property
    def model_name(self) -> str:
        return self.d3.model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.d3.model_name = value

    @property
    def max_length(self) -> int | None:
        return self.d3.max_length

    @property
    def image_unavailable(self) -> str | None:
        reason = self.engine.backend.media_unavailable("image")
        return reason or self.d3.image_unavailable

    @property
    def video_unavailable(self) -> str | None:
        reason = self.engine.backend.media_unavailable("video")
        return reason or self.d3.video_unavailable

    # ---- reference frontend ------------------------------------------
    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        return self.d3.prepare(*args, **kwargs)

    def respond(self, prepared: Any, probabilities: Any, tokens: int) -> dict[str, Any]:
        return self.d3.respond(prepared, probabilities, tokens)

    def run(self, prepared: Any) -> tuple[dict[str, list[float]], int]:
        try:
            return self.engine.run(prepared)
        except MediaNotSupported as exc:
            # The reference server maps ValueError to 422 with the reason.
            raise ValueError(str(exc)) from exc

    # ---- media -------------------------------------------------------
    def load_images(self, items: Any, *, strict: bool = False) -> Any:
        reason = self.engine.backend.media_unavailable("image")
        if reason:
            raise ValueError(f"this server cannot accept images: {reason}")
        return self.d3.load_images(items, strict=strict)

    def load_videos(self, items: Any, *, strict: bool = False) -> Any:
        reason = self.engine.backend.media_unavailable("video")
        if reason:
            raise ValueError(f"this server cannot accept videos: {reason}")
        return self.d3.load_videos(items, strict=strict)

    def image_contract(self) -> Mapping[str, Any]:
        return self.d3.image_contract()

    def video_contract(self) -> Mapping[str, Any]:
        return self.d3.video_contract()

    # ---- lifecycle ---------------------------------------------------
    def warmup(self) -> Any:
        warmup = getattr(self.engine.backend, "warmup", None)
        if callable(warmup):
            return warmup()
        return self.d3.warmup()


def build_server(
    engine: DecisionEngine,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    api_key: str | None = None,
) -> Any:
    """Bind the reference System One server to a trt/hf engine."""
    if not engine._started:
        engine.start()
    server_module = import_server(engine.config.root, Path(engine.frontend.rt.__file__).parent)
    return server_module.serve(ServedModel(engine), host, port, api_key=api_key)


def serve(
    engine: DecisionEngine,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    api_key: str | None = None,
    warmup: bool = True,
) -> None:
    """Blocking serve loop (Ctrl-C stops it)."""
    model = ServedModel(engine)
    if warmup:
        model.warmup()
    server = build_server(engine, host, port, api_key=api_key)
    bound_host, bound_port = server.server_address[:2]
    access = "bearer key required" if api_key else f"open (set DECISION_API_KEY to require a key)"
    print(
        f"d3 System One server [{engine.backend_name}]: {model.model_name} "
        f"on http://{bound_host}:{bound_port}/v1/systemone; {access}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
