"""Profiling harness tests: switches, cache modes and per-layer timing."""

from __future__ import annotations

import pytest

from d3trt.bench import synthetic_cases
from d3trt.engine import DecisionEngine
from d3trt.profile import (
    apply_variant,
    default_variants,
    format_table,
    profile_layers,
    profile_variants,
    select_variants,
)


@pytest.fixture(scope="module")
def prof_engine(model_dir):
    engine = DecisionEngine(
        model_dir,
        backend="hf",
        device="cpu",
        dtype="float32",
        batch_size=4,
    )
    engine.start()
    yield engine
    engine.close()


@pytest.fixture(scope="module")
def prof_cases(prof_engine):
    return synthetic_cases(
        prof_engine.frontend,
        state_tokens=48,
        questions=1,
        options=2,
    )


def test_variant_switches_reset_to_base(prof_engine):
    variants = {v.name: v for v in default_variants(prof_engine)}
    apply_variant(prof_engine, variants["core+cache+batch32"])
    assert prof_engine.batch_size == 32
    assert prof_engine.frontend.d3.batch_size == 32
    assert prof_engine.backend.use_core is True

    apply_variant(prof_engine, variants["reference"])
    assert prof_engine.batch_size == 4
    assert prof_engine.frontend.d3.batch_size == 4
    assert prof_engine.backend.use_core is False

    apply_variant(prof_engine, variants["core+cache+sdpa"])
    assert prof_engine.backend.use_core is True
    assert prof_engine.backend.stack.core.attention_question == "sdpa"


def test_select_variants_rejects_unknown(prof_engine):
    with pytest.raises(ValueError, match="unknown variants"):
        select_variants(default_variants(prof_engine), "not-a-variant")


def test_profile_variants_cold_and_warm(prof_engine, prof_cases):
    variant = [v for v in default_variants(prof_engine) if v.name == "core+cache+sdpa"]
    report = profile_variants(
        prof_engine,
        prof_cases,
        variants=variant,
        repeats=1,
        warmup=1,
        cache_modes=("cold", "warm"),
    )
    rows = report["rows"]
    assert len(rows) == 2
    cold, warm = rows
    assert cold["cache_mode"] == "cold"
    assert warm["cache_mode"] == "warm"
    assert cold["cache_hit_rate"] == 0.0
    assert warm["cache_hit_rate"] == 1.0
    assert warm["phase_ms"]["state"] < cold["phase_ms"]["state"]
    assert warm["kernel"]["state_layers"] == 0
    assert "core+cache+sdpa" in format_table(report)


def test_profile_layers_reports_state_and_questions(prof_engine, prof_cases):
    variant = [v for v in default_variants(prof_engine) if v.name == "core+cache+sdpa"][0]
    report = profile_layers(prof_engine, prof_cases[0], variant=variant)
    assert report["source"] == "core"
    assert len(report["layers"]) == 2
    assert all(layer["calls"] >= 1 for layer in report["layers"])
    assert any((layer["state_ms"] or 0) > 0 for layer in report["layers"])
    assert any((layer["question_ms"] or 0) > 0 for layer in report["layers"])
