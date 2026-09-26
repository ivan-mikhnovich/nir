"""Invariants of the prepared data that every number in the work rests on.

These read the processed JSON directly instead of calling the loader, so the
test keeps checking the data even while the loader is being changed. The
counts are the ones the note quotes in section 3.2.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "oaei"
ONTOLOGIES = ("cmt", "confOf", "conference", "edas", "ekaw", "iasted", "sigkdd")
NODE_COUNTS = {
    "cmt": 29,
    "confOf": 38,
    "conference": 59,
    "edas": 103,
    "ekaw": 73,
    "iasted": 140,
    "sigkdd": 49,
}
# The reference alignments of the Conference track are not strictly 1:1: two
# sources of confOf are matched to ekaw#Student. The note documents this case.
TARGET_COLLISIONS = {("confOf", "ekaw", "ekaw#Student")}


def load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def alignments() -> list[dict]:
    return json.loads((DATA / "alignments.json").read_text(encoding="utf-8"))


def test_node_counts_are_the_documented_ones():
    got = {name: len(load(name)["nodes"]) for name in ONTOLOGIES}
    assert got == NODE_COUNTS


def test_all_21_pairs_are_present():
    pairs = {(a["source"], a["target"]) for a in alignments()}
    assert len(pairs) == 21


def test_ground_truth_reaches_exactly_258_pairs():
    assert sum(len(a["matches"]) for a in alignments()) == 258


def test_every_ground_truth_match_uses_existing_nodes():
    broken = []
    for alignment in alignments():
        source_nodes = load(alignment["source"])["nodes"]
        target_nodes = load(alignment["target"])["nodes"]
        for match in alignment["matches"]:
            if match["source_id"] not in source_nodes or match["target_id"] not in target_nodes:
                broken.append((alignment["source"], alignment["target"], match))
    assert broken == []


def test_every_ground_truth_node_has_a_unique_name_within_its_taxonomy():
    # Needed by the string baseline, whose tie-break would otherwise matter.
    for name in ONTOLOGIES:
        names = [n["name"].lower() for n in load(name)["nodes"].values()]
        assert len(names) == len(set(names)), f"{name}: повторяющиеся метки"


def test_target_side_collisions_are_the_documented_one():
    counts = collections.Counter(
        (alignment["source"], alignment["target"], match["target_id"])
        for alignment in alignments()
        for match in alignment["matches"]
    )
    collided = {key for key, count in counts.items() if count > 1}
    assert collided == TARGET_COLLISIONS


def test_no_source_is_matched_twice():
    counts = collections.Counter(
        (alignment["source"], alignment["target"], match["source_id"])
        for alignment in alignments()
        for match in alignment["matches"]
    )
    assert [key for key, count in counts.items() if count > 1] == []
