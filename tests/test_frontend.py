from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


CASES = [
    {
        "id": "route",
        "state": "The checkout service is timing out and users cannot pay.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Pick the team that owns this.",
                "criteria": {
                    "sre": "infrastructure and reliability",
                    "billing": "payments and invoicing",
                    "frontend": "web user interface",
                },
            }
        },
    },
    {
        "id": "urgent",
        "state": "A single dashboard widget is showing a stale number.",
        "questions": {
            "urgent": {
                "type": "noul",
                "instructions": "Does this need immediate action?",
                "criteria": {"true": "needs immediate action", "false": "can wait"},
            }
        },
    },
    {
        "id": "mixed",
        "state": "The database replica is 40 minutes behind and reads are stale.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Pick the owning team.",
                "criteria": {"db": "database reliability", "web": "frontend", "billing": "payments"},
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is this?",
                "criteria": {"low": "minor", "high": "customer impacting"},
            },
            "urgent": {
                "type": "noul",
                "instructions": "Is this urgent?",
                "criteria": {"true": "yes", "false": "no"},
            },
        },
    },
]


def test_prepared_matches_reference(reference, frontend):
    for case in CASES:
        ref = reference.prepare(case["state"], case["questions"])
        ours = frontend.prepare(case["state"], case["questions"])
        assert list(ours.keys) == list(ref.keys)
        assert ours.states == ref.states
        assert ours.sequences == ref.sequences
        assert ours.errors == ref.errors
        assert ours.runnable == ref.runnable
        assert frontend.describe_tokens(ours)["state_ids"] == ref.sequences[ours.runnable[0]][: ours.states[ours.runnable[0]]]


def test_respond_matches_reference(reference, frontend):
    case = CASES[2]
    ref = reference.prepare(case["state"], case["questions"])
    ours = frontend.prepare(case["state"], case["questions"])
    probabilities = {
        key: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0] for key in ours.runnable
    }
    tokens = 17
    assert frontend.respond(ours, probabilities, tokens) == reference.respond(
        ref, probabilities, tokens
    )


def test_invalid_question_errors_match(reference, frontend):
    questions = {
        "good": {
            "type": "choice",
            "instructions": "Pick one.",
            "criteria": {"a": "first", "b": "second"},
        },
        "bad": {"type": "choice", "instructions": "Missing criteria."},
    }
    ref = reference.prepare("A state.", questions)
    ours = frontend.prepare("A state.", questions)
    assert ours.errors == ref.errors
    assert "bad" in ours.errors


def test_image_prepared_matches_reference(reference, frontend, tmp_path: Path):
    image = Image.new("RGB", (64, 64), (40, 120, 200))
    path = tmp_path / "blue.png"
    image.save(path)
    questions = {
        "describe": {
            "type": "choice",
            "instructions": "What is the dominant colour?",
            "criteria": {"blue": "a blue square", "red": "a red square"},
        }
    }
    ref = reference.prepare("Look at this image.", questions, images=[str(path)])
    ours = frontend.prepare("Look at this image.", questions, images=[str(path)])
    assert ours.sequences == ref.sequences
    assert ours.states == ref.states
    assert ours.errors == ref.errors
    assert ours.images is not None
    assert not ours.videos
