"""Typed equivalence probabilities from the pinned OpenRouter Jev Decisions API."""

from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime
from numbers import Real

import httpx

MODEL = "typesafe/jev-1.13"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
INPUT_USD_PER_TOKEN = 0.042 / 1_000_000
CONTEXT_TOKENS = 32_000
MAX_REQUEST_COST = INPUT_USD_PER_TOKEN * CONTEXT_TOKENS
INSTRUCTIONS = (
    "Do source and candidates[{index}] describe equivalent ontology classes "
    "(the same concept), rather than merely related concepts or a "
    "subclass/superclass relationship?"
)
CRITERIA = {
    "true": "The two classes describe the same concept.",
    "false": "The classes are different, merely related, or one is a proper subclass of the other.",
}


def build_request(
    source_description: str, candidate_descriptions: list[str]
) -> dict:
    """Ask one independent Noul question per candidate, sharing the source state."""
    if not candidate_descriptions:
        raise ValueError("A Jev request needs at least one candidate")
    return {
        "model": MODEL,
        "state": {
            "source": source_description,
            "candidates": [
                {"index": i, "description": text}
                for i, text in enumerate(candidate_descriptions)
            ],
        },
        "questions": {
            f"candidate_{i}": {
                "type": "noul",
                "instructions": INSTRUCTIONS.format(index=i),
                "criteria": CRITERIA,
            }
            for i in range(len(candidate_descriptions))
        },
    }


def parse_response(
    response: dict, source_id: str, candidate_ids: list[str]
) -> list[dict]:
    """Reject incomplete or invalid decisions instead of manufacturing matches."""
    if not isinstance(response, dict):
        raise ValueError("Jev returned a non-object response")
    model = response.get("model", "")
    if not isinstance(model, str) or (
        model != MODEL and not model.startswith(MODEL + "-")
    ):
        raise ValueError(f"Unexpected Jev model: {model!r}")
    answers = response.get("answers")
    expected = {f"candidate_{i}" for i in range(len(candidate_ids))}
    if not isinstance(answers, dict) or set(answers) != expected:
        raise ValueError(
            "Jev answer keys do not cover the requested candidates exactly"
        )
    probabilities = []
    for i, target_id in enumerate(candidate_ids):
        answer = answers[f"candidate_{i}"]
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise ValueError("Jev did not return a Noul decision")
        probability = answer.get("noul")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, Real)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"Invalid Jev probability: {probability!r}")
        probabilities.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "probability": float(probability),
            }
        )
    usage = response.get("usage", {})
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if (
        isinstance(cost, bool)
        or not isinstance(cost, Real)
        or not math.isfinite(cost)
        or cost < 0
    ):
        raise ValueError("Jev response has no valid billed usage.cost")
    return probabilities


class JevClient:
    """Make real API calls while preserving the full response and observed latency."""

    def __init__(self, timeout: float = 90.0):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not available")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the shared HTTP connection pool."""
        self.client.close()

    def request(
        self, payload: dict, source_id: str, candidate_ids: list[str]
    ) -> dict:
        """Return an auditable attempt, including failures, without embedding fallback."""
        started = time.perf_counter()
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request": payload,
            "status_code": None,
            "response": None,
            "error": None,
        }
        try:
            reply = self.client.post(ENDPOINT, json=payload)
            row["status_code"] = reply.status_code
            try:
                row["response"] = reply.json()
            except ValueError:
                row["response"] = {"unparsed_body": reply.text}
            reply.raise_for_status()
            row["probabilities"] = parse_response(
                row["response"], source_id, candidate_ids
            )
        except (httpx.HTTPError, ValueError) as error:
            row["error"] = str(error)
        finally:
            row["latency_seconds"] = time.perf_counter() - started
        return row
