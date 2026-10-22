"""Errors shared by the d3-trt runtime."""

from __future__ import annotations


class D3Error(Exception):
    """Base class for every d3-trt error."""


class ConfigurationError(D3Error):
    """The checkpoint, decision config or backend configuration is invalid."""


class BackendUnavailable(D3Error):
    """The requested backend cannot run in this environment."""

    def __init__(self, backend: str, reason: str):
        super().__init__(f"backend {backend!r} is unavailable: {reason}")
        self.backend = backend
        self.reason = reason


class ParityError(D3Error):
    """A backend disagrees with the reference beyond the acceptance budget."""


class StateCacheError(D3Error):
    """A cached state is missing, evicted while in use, or inconsistent."""


class MediaNotSupported(D3Error):
    """The active backend cannot serve this modality yet."""
