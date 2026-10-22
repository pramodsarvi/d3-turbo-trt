from __future__ import annotations

import pytest

from d3trt import DecisionEngine
from d3trt.errors import BackendUnavailable
from d3trt.model.trtllm_stack import trtllm_available


def test_info(engine):
    info = engine.info()
    assert info["backend"] == "hf"
    assert info["frontend"]["attention_mode"] == "prefix_bidir"
    assert info["config"]["attention_mode"] == "prefix_bidir"
    assert info["fallback"] is False


def test_invalid_question_returns_error(reference, engine):
    questions = {
        "bad": {"type": "choice", "instructions": "No criteria."},
        "ok": {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {"a": "first", "b": "second"},
        },
    }
    ours = engine.system_one(state="A state.", questions=questions)
    ref = reference.system_one(state="A state.", questions=questions)
    assert "error" in ours["answers"]["bad"]
    assert "error" in ref["answers"]["bad"]
    assert ours["answers"]["bad"]["error"] == ref["answers"]["bad"]["error"]


@pytest.mark.skipif(trtllm_available()[0], reason="TensorRT-LLM is available; auto selects it")
def test_auto_selects_hf_here(model_dir):
    engine = DecisionEngine(model_dir, backend="auto", device="cpu", dtype="bfloat16", batch_size=2)
    engine.start()
    try:
        assert engine.backend_name == "hf"
    finally:
        engine.close()


@pytest.mark.skipif(trtllm_available()[0], reason="TensorRT-LLM is installed; the guard is not exercised")
def test_trtllm_reports_unavailable(model_dir):
    engine = DecisionEngine(model_dir, backend="trtllm", device="cpu", batch_size=2)
    with pytest.raises(BackendUnavailable) as exc:
        engine.start()
    assert "trtllm" in str(exc.value)


def test_long_context_cap(model_dir):
    engine = DecisionEngine(model_dir, backend="hf", device="cpu", dtype="bfloat16", batch_size=2)
    engine.start()
    try:
        normal = engine.frontend.d3.question_cap(False)
        long = engine.frontend.d3.question_cap(True)
        assert normal is not None
        assert long is None or long >= normal
    finally:
        engine.close()
