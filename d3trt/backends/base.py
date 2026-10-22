"""Backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence


class Backend(ABC):
    """A prefill-only decision backend.

    A backend owns the weights and turns a reference ``Prepared`` request into
    per-question option probabilities. It must not re-render prompts or decode
    answers; those remain in :class:`d3trt.reference.Frontend`.
    """

    name = "base"

    def attach(self, frontend: Any) -> None:
        """Give the backend the frontend skeleton (tokenizer, processor, config)."""

    def start(self) -> None:
        """Load weights / build engines. Called once before serving."""

    def close(self) -> None:
        """Release GPU memory."""

    @abstractmethod
    def run(self, prepared: Any) -> tuple[dict[str, list[float]], int]:
        """Return ``({question_id: probabilities}, input_tokens)``."""

    def info(self) -> Mapping[str, Any]:
        return {"backend": self.name}

    def media_unavailable(self, kind: str) -> str | None:
        """Reason this backend cannot serve ``image`` or ``video``, or None."""
        return None

    def supports(self, prepared: Any) -> bool:
        return True

    # Optional composed path (Box/Track/assets). Backends that cannot serve it
    # raise a clear error; the engine falls back to the reference backend.
    def run_composed(self, prepared: Any) -> tuple[dict[str, list[float]], int]:
        raise NotImplementedError("this backend does not implement composed Box/Track requests")


def run_batches(
    keys: Sequence[str],
    sequences: Mapping[str, Sequence[int]],
    states: Mapping[str, int],
    batch_size: int,
) -> list[list[str]]:
    """Group runnable keys into ordered batches (reference ordering)."""
    batches: list[list[str]] = []
    current: list[str] = []
    for key in keys:
        current.append(key)
        if len(current) == batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches
