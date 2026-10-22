from __future__ import annotations

import pytest


CASES = [
    {
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
                "criteria": ["minor", "customer impacting"],
            },
        },
    },
]


def _numeric_answers(response):
    out = {}
    for key, answer in response["answers"].items():
        kind = answer["type"]
        if kind == "choice":
            out[key] = ("choice", answer["choice"], answer["probabilities"])
        elif kind in ("noul", "score"):
            out[key] = (kind, answer[kind])
        else:
            out[key] = (kind, answer)
    return out


@pytest.mark.parametrize("case", CASES)
def test_engine_matches_reference(reference, engine, case):
    ref = reference.system_one(state=case["state"], questions=case["questions"])
    ours = engine.system_one(state=case["state"], questions=case["questions"])
    ref_answers = _numeric_answers(ref)
    our_answers = _numeric_answers(ours)
    assert set(ref_answers) == set(our_answers)
    for key, ref_value in ref_answers.items():
        our_value = our_answers[key]
        if ref_value[0] == "choice":
            assert ref_value[1] == our_value[1]
            for option, probability in ref_value[2].items():
                assert abs(probability - our_value[2][option]) <= 1e-2
        elif ref_value[0] in ("noul", "score"):
            assert abs(float(ref_value[1]) - float(our_value[1])) <= 1e-2
        else:
            # Box / track are not used in this suite.
            assert ref_value == our_value


def test_state_cache_reuses_the_state(reference, engine):
    state = "The cache test state is deliberately unique: zq-9182."
    questions_a = {
        "a": {
            "type": "choice",
            "instructions": "Pick a letter.",
            "criteria": {"one": "first", "two": "second"},
        }
    }
    questions_b = {
        "b": {
            "type": "noul",
            "instructions": "Is this state present?",
            "criteria": {"true": "yes", "false": "no"},
        }
    }
    engine.system_one(state=state, questions=questions_a)
    before = engine.backend.cache.stats()
    engine.system_one(state=state, questions=questions_b)
    after = engine.backend.cache.stats()
    assert after["entries"] >= 1
    assert after["hits"] >= before["hits"] + 1
    assert after["misses"] == before["misses"]


def test_state_key_matches_prepared(engine):
    state = "State key test: abcdef."
    questions = {
        "q": {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {"a": "one", "b": "two"},
        }
    }
    prepared = engine.frontend.prepare(state, questions)
    key, state_len = engine.state_key(prepared)
    assert state_len == prepared.states["q"]
    assert key == engine.config.state_key(prepared.sequences["q"][:state_len])


def test_max_options_are_masked(engine):
    state = "Masking test."
    questions = {
        "q": {
            "type": "choice",
            "instructions": "Pick one.",
            "criteria": {"a": "first", "b": "second"},
        }
    }
    response = engine.system_one(state=state, questions=questions)
    probabilities = response["answers"]["q"]["probabilities"]
    assert set(probabilities) == {"a", "b"}
    assert abs(sum(probabilities.values()) - 1.0) < 1e-5
