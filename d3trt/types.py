"""Typed request and response objects.

The canonical frontend is the reference ``d3_runtime`` module, which works with
plain mappings. These dataclasses are a thin typed shell used by the HTTP server
and the benchmark harness; ``DecisionEngine`` accepts either form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(slots=True)
class DecisionRequest:
    """One System One request: a state plus a mapping of question id -> question."""

    state: Any = None
    questions: Mapping[str, Any] = field(default_factory=dict)
    images: Sequence[Any] | None = None
    videos: Sequence[Any] | None = None
    assets: Mapping[str, Any] | None = None
    long_context: bool = False


@dataclass(slots=True)
class DecisionResponse:
    """The reference-shaped answer mapping plus backend telemetry."""

    model: str
    answers: dict[str, Any]
    usage: dict[str, int]
    timings: dict[str, float] = field(default_factory=dict)
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "answers": self.answers,
            "usage": dict(self.usage),
        }
        if self.timings:
            payload["timings"] = dict(self.timings)
        if self.backend:
            payload["backend"] = self.backend
        return payload


@dataclass(slots=True)
class PreparedState:
    """Backend-facing view of a prepared request.

    ``prepared`` is the reference runtime's ``Prepared`` dataclass. ``state_ids``
    are the token ids of the shared state block; ``state_len`` is its length.
    For image/video requests ``media_hash`` is a stable digest of the decoded
    media so states that differ only in pixels do not alias.
    """

    prepared: Any
    state_ids: tuple[int, ...]
    state_len: int
    media_hash: str = ""
    state_key: str = ""

    @property
    def questions(self) -> tuple[str, ...]:
        return tuple(self.prepared.keys)

    @property
    def runnable(self) -> tuple[str, ...]:
        return tuple(self.prepared.runnable)
