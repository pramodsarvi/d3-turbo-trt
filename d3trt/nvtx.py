"""Tiny NVTX helpers so nsys traces line up with d3 phases.

When no CUDA is present the context manager is a no-op; the same code runs on
CPU without importing anything NVIDIA-specific.
"""

from __future__ import annotations

import contextlib
from typing import Iterator


@contextlib.contextmanager
def range(name: str) -> Iterator[None]:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
            return
    except Exception:  # noqa: BLE001
        pass
    yield


def mark(name: str) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.nvtx.mark(name)
    except Exception:  # noqa: BLE001
        pass
