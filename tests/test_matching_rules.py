"""Tests for the shared matching rules.

These rules decide which candidate pairs survive into an alignment, so a
mistake here silently changes every reported F1 while every matcher still
looks correct. The properties pinned below are exactly the ones that failed
before the repair: no source or target may be used twice, and the result must
not depend on the order in which candidates arrive (parallel classification
used to decide it).
"""

from __future__ import annotations

import random

from src.matching_rules import (
    dedupe_by_target,
    greedy_injective,
    per_source_best,
    sort_candidates,
    to_alignment,
)

RAW = [("s0", "t0", 0.9), ("s0", "t1", 0.8), ("s2", "t2", 0.95), ("s2", "t3", 0.85)]
TIED = [("s0", "t0", 0.9), ("s0", "t1", 0.9), ("s1", "t0", 0.9), ("s1", "t1", 0.9)]


def test_sort_orders_by_confidence_then_ids():
    assert sort_candidates(RAW) == [
        ("s2", "t2", 0.95),
        ("s0", "t0", 0.9),
        ("s2", "t3", 0.85),
        ("s0", "t1", 0.8),
    ]


def test_greedy_injective_uses_every_node_at_most_once():
    picked = greedy_injective(RAW)
    assert picked == [("s2", "t2", 0.95), ("s0", "t0", 0.9)]
    assert len({src for src, _, _ in picked}) == len(picked)
    assert len({tgt for _, tgt, _ in picked}) == len(picked)


def test_greedy_injective_does_not_depend_on_input_order():
    base = greedy_injective(TIED)
    assert base == [("s0", "t0", 0.9), ("s1", "t1", 0.9)]
    for seed in range(5):
        shuffled = TIED[:]
        random.Random(seed).shuffle(shuffled)
        assert greedy_injective(shuffled) == base


def test_greedy_injective_threshold_drops_weak_candidates():
    raw = [("s0", "t0", 0.4), ("s1", "t1", 0.8)]
    assert greedy_injective(raw, threshold=0.5) == [("s1", "t1", 0.8)]


def test_per_source_best_keeps_one_candidate_per_source():
    raw = [("s0", "t0", 0.9), ("s0", "t1", 0.8), ("s1", "t0", 0.7)]
    assert per_source_best(raw) == [("s0", "t0", 0.9), ("s1", "t0", 0.7)]


def test_dedupe_by_target_keeps_one_candidate_per_target():
    raw = [("s0", "t0", 0.9), ("s1", "t0", 0.8), ("s2", "t1", 0.7)]
    assert dedupe_by_target(raw) == [("s0", "t0", 0.9), ("s2", "t1", 0.7)]


def test_greedy_and_dedupe_differ_on_a_contested_target():
    # s1's best candidate (t0) is taken by s0; the nearest-target rule then
    # drops s1 entirely, while the greedy rule still matches s1 to t1.
    raw = [("s0", "t0", 0.9), ("s1", "t0", 0.85), ("s1", "t1", 0.84)]
    assert dedupe_by_target(raw) == [("s0", "t0", 0.9)]
    assert greedy_injective(raw) == [("s0", "t0", 0.9), ("s1", "t1", 0.84)]


def test_to_alignment_builds_typed_matches():
    alignment = to_alignment([("s0", "t0", 0.9)], source="a", target="b")
    assert (alignment.source, alignment.target) == ("a", "b")
    assert alignment.as_pairs() == {("s0", "t0")}
    assert alignment.matches[0].confidence == 0.9
