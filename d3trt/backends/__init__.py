"""Backend registry."""

from __future__ import annotations

from typing import Callable

from ..errors import BackendUnavailable
from .base import Backend

_REGISTRY: dict[str, Callable[..., Backend]] = {}


def register(name: str, factory: Callable[..., Backend]) -> None:
    _REGISTRY[name] = factory


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def create(name: str, *args, **kwargs) -> Backend:
    if name not in _REGISTRY:
        raise BackendUnavailable(name, f"unknown backend; choose from {', '.join(available())}")
    return _REGISTRY[name](*args, **kwargs)


def _hf(*args, **kwargs) -> Backend:
    from .hf import HFBackend

    return HFBackend(*args, **kwargs)


def _trtllm(*args, **kwargs) -> Backend:
    from .trtllm import TrtllmBackend

    return TrtllmBackend(*args, **kwargs)


register("hf", _hf)
register("trtllm", _trtllm)


def detect() -> str:
    """Pick the fastest backend that can run here, without loading weights."""
    try:
        from .trtllm import trtllm_available

        ok, _ = trtllm_available()
        if ok:
            return "trtllm"
    except Exception:  # noqa: BLE001
        pass
    return "hf"
