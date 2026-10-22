from __future__ import annotations

import json
from pathlib import Path

from d3trt.parity import compare, load_cases, run_cases

ROOT = Path(__file__).resolve().parents[1]


def test_text_golden_gate(reference, engine):
    cases = load_cases(ROOT / "tests" / "golden" / "text.json")
    reference_answers = run_cases(reference, cases)
    candidate_answers = run_cases(engine, cases)
    comparisons, summary = compare(reference_answers, candidate_answers, tolerance=1e-2)
    print(json.dumps(summary, indent=2))
    assert summary["structural_problems"] == []
    assert summary["max_abs_diff"] <= 1e-2
    assert summary["choice_agreement"] == 1.0
    assert summary["passed"]
