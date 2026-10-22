"""Shared fixtures.

Tests run against a tiny random Qwen3.5 checkpoint that uses the real tokenizer,
config contract and d3 reference runtime. It is not committed (130 MB); set
``D3_TEST_MODEL`` to a directory, or generate one with
``python scripts/make_tiny_model.py``.

The reference ``D3`` is loaded once per session and is the oracle for every
parity assertion.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _model_dir() -> Path | None:
    env = os.environ.get("D3_TEST_MODEL")
    if env:
        path = Path(env).expanduser().resolve()
        return path if path.is_dir() else None
    for candidate in (ROOT / ".research" / "tiny" / "model", ROOT / "tests" / "fixtures" / "tiny"):
        if (candidate / "config.json").exists():
            return candidate.resolve()
    return None


MODEL_DIR = _model_dir()


def pytest_collection_modifyitems(config, items):
    if MODEL_DIR is None:
        skip = pytest.mark.skip(
            reason=(
                "no tiny d3 checkpoint; set D3_TEST_MODEL or run "
                "python scripts/make_tiny_model.py"
            )
        )
        for item in items:
            if "no_model" not in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def model_dir() -> Path:
    assert MODEL_DIR is not None
    return MODEL_DIR


@pytest.fixture(scope="session")
def reference(model_dir: Path):
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from d3_runtime import D3

    model = D3.from_pretrained(str(model_dir), device="cpu", batch_size=4, verify="none")
    yield model
    del model


@pytest.fixture(scope="session")
def frontend(model_dir: Path):
    from d3trt.reference import Frontend

    return Frontend(model_dir, batch_size=4, device="cpu")


@pytest.fixture(scope="session")
def engine(model_dir: Path):
    from d3trt.engine import DecisionEngine

    instance = DecisionEngine(
        model_dir,
        backend="hf",
        device="cpu",
        dtype="bfloat16",
        batch_size=4,
        state_cache_bytes=1 << 30,
    )
    instance.start()
    yield instance
    instance.close()
