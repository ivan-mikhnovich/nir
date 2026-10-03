"""Reject invalid Jev probabilities and unsafe paid-request cache reuse."""

from __future__ import annotations

import copy
import json

import pytest

from src.matchers.jev import MAX_REQUEST_COST, MODEL, parse_response
from src.runners.jev import budget_charge, load_journal


def response() -> dict:
    """Build a complete typed response with observed billing."""
    return {
        "model": MODEL + "-20260917",
        "answers": {
            "candidate_0": {"type": "noul", "noul": 0.9},
            "candidate_1": {"type": "noul", "noul": 0.1},
        },
        "usage": {"cost": 0.00002},
    }


@pytest.mark.parametrize(
    "probability", [True, None, "0.9", -0.01, 1.01, float("nan"), float("inf")]
)
def test_invalid_probabilities_cannot_become_matches(probability):
    raw = response()
    raw["answers"]["candidate_0"]["noul"] = probability
    with pytest.raises(ValueError, match="Invalid Jev probability"):
        parse_response(raw, "s", ["t0", "t1"])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "wrong_type",
        "wrong_model",
        "null_model",
        "missing_cost",
        "negative_cost",
    ],
)
def test_partial_or_unverifiable_responses_are_not_successes(mutation):
    raw = response()
    if mutation == "missing":
        del raw["answers"]["candidate_1"]
    elif mutation == "extra":
        raw["answers"]["candidate_2"] = {"type": "noul", "noul": 0.9}
    elif mutation == "wrong_type":
        raw["answers"]["candidate_0"]["type"] = "choice"
    elif mutation == "wrong_model":
        raw["model"] = "typesafe/jev-router"
    elif mutation == "null_model":
        raw["model"] = None
    elif mutation == "missing_cost":
        del raw["usage"]["cost"]
    else:
        raw["usage"]["cost"] = -1
    with pytest.raises(ValueError):
        parse_response(raw, "s", ["t0", "t1"])


@pytest.mark.parametrize("mutation", ["request", "probability", "duplicate"])
def test_resume_rejects_changed_or_duplicate_successes(tmp_path, mutation):
    payload = {"model": MODEL, "state": {"source": "author"}}
    tasks = {
        "key": {
            "source_id": "s",
            "candidate_ids": ["t0", "t1"],
            "request": payload,
        }
    }
    raw = response()
    row = {
        "key": "key",
        "request": copy.deepcopy(payload),
        "response": raw,
        "status_code": 200,
        "error": None,
        "probabilities": parse_response(raw, "s", ["t0", "t1"]),
    }
    if mutation == "request":
        row["request"]["state"]["source"] = "reviewer"
    elif mutation == "probability":
        row["probabilities"][0]["probability"] = 0.01
    rows = [row, row] if mutation == "duplicate" else [row]
    journal = tmp_path / "requests.jsonl"
    journal.write_text(
        "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_journal(journal, tasks)


def test_unbilled_interrupted_attempt_reserves_full_context_cost():
    assert (
        budget_charge({"response": None, "error": "timeout"})
        == MAX_REQUEST_COST
    )
    assert budget_charge({"response": {"usage": {"cost": 0}}}) == 0
    assert (
        budget_charge({"response": {"usage": {"cost": True}}})
        == MAX_REQUEST_COST
    )
