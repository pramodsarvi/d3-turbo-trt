"""Parity harness: reference d3 runtime vs a d3-trt backend.

The acceptance gates from the serving plan:

* every answer's probability vector is within ``1e-2`` absolute difference;
* every Choice argmax agrees;
* Noul / Score values are within ``1e-2``;
* Box / Track numeric fields are within the same budget.

The Jev Decision Index is a benchmark-level number that needs the Jev dataset;
this harness reports the per-answer agreement it can measure locally and leaves
the index to the GPU-day run (see docs/VALIDATION.md).

Runs are sequential by default: load the reference, record its answers, free it,
then load the candidate. That keeps peak GPU memory to one model copy.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .engine import DecisionEngine

DEFAULT_TOLERANCE = 1e-2


@dataclass
class Case:
    id: str
    state: Any
    questions: Mapping[str, Any]
    images: Sequence[Any] | None = None
    videos: Sequence[Any] | None = None
    assets: Mapping[str, Any] | None = None
    long_context: bool = False
    expected: Mapping[str, Any] | None = None


@dataclass
class Comparison:
    case: str
    question: str
    kind: str
    max_abs_diff: float = 0.0
    choice_agrees: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)


def load_cases(path: str | Path) -> list[Case]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases") or []
    cases: list[Case] = []
    for entry in data:
        cases.append(
            Case(
                id=str(entry.get("id") or f"case-{len(cases)}"),
                state=entry.get("state"),
                questions=entry.get("questions") or {},
                images=entry.get("images"),
                videos=entry.get("videos"),
                assets=entry.get("assets"),
                long_context=bool(entry.get("long_context", False)),
                expected=entry.get("expected"),
            )
        )
    return cases


def run_cases(decider: Any, cases: Iterable[Case]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        started = time.perf_counter()
        response = decider.system_one(
            state=case.state,
            questions=case.questions,
            images=case.images,
            videos=case.videos,
            assets=case.assets,
            long_context=case.long_context,
        )
        results[case.id] = {
            "answers": response.get("answers", {}),
            "usage": response.get("usage", {}),
            "seconds": time.perf_counter() - started,
        }
    return results


def _compare_value(a: Any, b: Any, path: str = "") -> tuple[float, list[str]]:
    """Return max absolute numeric difference and a list of structural mismatches."""
    diffs: list[float] = []
    problems: list[str] = []
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                problems.append(f"{path}.{key} missing on one side")
                continue
            d, p = _compare_value(a[key], b[key], f"{path}.{key}")
            diffs.append(d)
            problems.extend(p)
        return (max(diffs) if diffs else 0.0), problems
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            problems.append(f"{path} length {len(a)} vs {len(b)}")
            return 0.0, problems
        for i, (x, y) in enumerate(zip(a, b)):
            d, p = _compare_value(x, y, f"{path}[{i}]")
            diffs.append(d)
            problems.extend(p)
        return (max(diffs) if diffs else 0.0), problems
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0, [] if a == b else [f"{path} bool {a!r} vs {b!r}"]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and not math.isfinite(a):
            return 0.0, [] if a == b else [f"{path} {a!r} vs {b!r}"]
        return abs(float(a) - float(b)), []
    if a != b:
        problems.append(f"{path} {a!r} vs {b!r}")
    return 0.0, problems


def compare(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[list[Comparison], dict[str, Any]]:
    comparisons: list[Comparison] = []
    max_diff = 0.0
    choices = 0
    choice_agrees = 0
    structural: list[str] = []

    for case_id, ref_case in reference.items():
        cand_case = candidate.get(case_id)
        if cand_case is None:
            structural.append(f"{case_id}: missing from candidate")
            continue
        ref_answers = ref_case.get("answers", {})
        cand_answers = cand_case.get("answers", {})
        for key, ref in ref_answers.items():
            cand = cand_answers.get(key)
            if cand is None:
                structural.append(f"{case_id}/{key}: missing answer")
                continue
            diff, problems = _compare_value(ref, cand, f"{case_id}/{key}")
            structural.extend(problems)
            kind = str(ref.get("type", ""))
            agrees: bool | None = None
            if kind == "choice":
                choices += 1
                agrees = ref.get("choice") == cand.get("choice")
                choice_agrees += int(agrees)
            max_diff = max(max_diff, diff)
            comparisons.append(
                Comparison(
                    case=case_id,
                    question=key,
                    kind=kind,
                    max_abs_diff=diff,
                    choice_agrees=agrees,
                    details={"problems": problems},
                )
            )

    summary = {
        "cases": len(reference),
        "answers": len(comparisons),
        "max_abs_diff": max_diff,
        "choice_questions": choices,
        "choice_agreement": (choice_agrees / choices) if choices else 1.0,
        "structural_problems": structural,
        "tolerance": tolerance,
        "passed": max_diff <= tolerance and not structural and choice_agrees == choices,
    }
    return comparisons, summary


@dataclass
class ParityReport:
    summary: dict[str, Any]
    comparisons: list[Comparison]
    reference_seconds: float = 0.0
    candidate_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "reference_seconds": round(self.reference_seconds, 3),
            "candidate_seconds": round(self.candidate_seconds, 3),
            "comparisons": [
                {
                    "case": c.case,
                    "question": c.question,
                    "kind": c.kind,
                    "max_abs_diff": c.max_abs_diff,
                    "choice_agrees": c.choice_agrees,
                    **c.details,
                }
                for c in self.comparisons
            ],
        }


def run_parity(
    model_dir: str | Path,
    cases: Sequence[Case],
    *,
    backend: str = "hf",
    device: str | None = None,
    dtype: str | None = None,
    reference_dir: str | Path | None = None,
    batch_size: int = 8,
    tolerance: float = DEFAULT_TOLERANCE,
    reference_verify: str = "none",
    backend_options: Mapping[str, Any] | None = None,
    reference_device: str | None = None,
) -> ParityReport:
    """Run the reference, free it, run the candidate, and compare.

    ``reference_device`` runs the BF16 oracle elsewhere (e.g. ``cpu``) when the
    candidate is a quantized model on a card the BF16 weights do not fit.
    """
    reference_device = reference_device or device
    from .reference import import_reference

    _, _, rt = import_reference(model_dir, reference_dir)
    started = time.perf_counter()
    reference_model = rt.D3.from_pretrained(
        str(model_dir),
        device=reference_device,
        batch_size=batch_size,
        verify=reference_verify,
    )
    reference = run_cases(reference_model, cases)
    reference_seconds = time.perf_counter() - started
    del reference_model
    _free(reference_device)

    started = time.perf_counter()
    engine = DecisionEngine(
        model_dir,
        backend=backend,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        reference_dir=reference_dir,
        fallback=False,
        backend_options=backend_options,
    )
    engine.start()
    candidate = run_cases(engine, cases)
    candidate_seconds = time.perf_counter() - started
    engine.close()
    _free(device)

    comparisons, summary = compare(reference, candidate, tolerance=tolerance)
    return ParityReport(
        summary=summary,
        comparisons=comparisons,
        reference_seconds=reference_seconds,
        candidate_seconds=candidate_seconds,
    )


def _free(device: str | None) -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
