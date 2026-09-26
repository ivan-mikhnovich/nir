"""Tests for the direction-aware cache layer.

A cached record must be read back in the direction it was computed in. The old
layer keyed records by a direction-insensitive pair name, so rebuilding an
alignment from the caller's loop variables flipped the ids for one pair in
twenty-one — which silently fed reversed matches into the ensemble vote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import cache

ALIGNMENTS = Path(__file__).resolve().parents[1] / "data" / "processed" / "oaei" / "alignments.json"


def record(source: str, target: str, pairs: list[tuple[str, str, float]]) -> dict:
    """Build a minimal cached record in the given direction."""
    return {
        "approach": "llm-bm25",
        "pair": cache.pair_key(source, target),
        "source": source,
        "target": target,
        "match_pairs": [
            {"source_id": s, "target_id": t, "confidence": c} for s, t, c in pairs
        ],
    }


def test_alignment_is_rebuilt_in_the_records_own_direction():
    saved = record("conference", "confOf", [("conference#Conference", "confof#Conference", 0.9)])
    built = cache.record_alignment(saved)
    assert (built.source, built.target) == ("conference", "confOf")
    assert built.as_pairs() == {("conference#Conference", "confof#Conference")}


def test_alignment_is_none_without_stored_predictions():
    assert cache.record_alignment({"source": "a", "target": "b"}) is None


def test_assert_orientation_accepts_the_saved_direction_and_rejects_the_flip():
    saved = record("conference", "confOf", [])
    cache.assert_orientation(saved, "conference", "confOf")
    with pytest.raises(ValueError):
        cache.assert_orientation(saved, "confOf", "conference")


def test_canonical_directions_come_from_the_reference_alignments():
    raw = [
        {"source": "conference", "target": "confOf", "matches": []},
        {"source": "cmt", "target": "conference", "matches": []},
    ]
    canonical = cache.canonical_directions(raw)
    assert canonical[cache.pair_key("confOf", "conference")] == ("conference", "confOf")
    assert canonical[cache.pair_key("cmt", "conference")] == ("cmt", "conference")


def test_every_shipped_cache_uses_the_canonical_direction():
    raw = json.loads(ALIGNMENTS.read_text(encoding="utf-8"))
    canonical = cache.canonical_directions(raw)
    misoriented = cache.misoriented_records(canonical)
    assert misoriented == {}, f"записи с чужим направлением: {misoriented}"


def test_demo_embeddings_do_not_enter_matcher_results(tmp_path, monkeypatch):
    """Embedding-id lists must not break result discovery or orientation audits."""
    monkeypatch.setattr(cache, "RESULTS_DIR", tmp_path)
    demo = tmp_path / "cls-violit-demo"
    demo.mkdir()
    (demo / "model.ids.json").write_text(json.dumps(["node-a", "node-b"]), encoding="utf-8")
    results = tmp_path / "llm-bm25"
    results.mkdir()
    saved = record("a", "b", [("s", "t", 0.9)])
    (results / "a_b.json").write_text(json.dumps(saved), encoding="utf-8")

    assert cache.load_all_cached() == {"llm-bm25": {"a↔b": saved}}
    assert cache.misoriented_records({"a↔b": ("a", "b")}) == {}
