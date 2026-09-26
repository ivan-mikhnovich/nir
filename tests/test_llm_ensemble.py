"""Tests for the LLM ensemble and its post-processing.

Two defects from the review are pinned here. F3: the published ensemble was a
*plurality* vote — per source, the target with the most votes, ties by the
first vote seen, and no cardinality constraint at all — so two sources could
keep the same target (the note described a ">=2 of 3" majority instead). F7:
greedy post-processing kept the arrival order of the parallel classifications,
so ties were decided by thread completion order, not by the data. The code now
requires at least two of the three representations to vote for a target and
reduces the result with `greedy_injective`, whose tie-break is the source and
target id.
"""

from __future__ import annotations

import random
from itertools import permutations

import pytest

from src.matchers.llm import (
    CONFIDENCE_THRESHOLD,
    ENSEMBLE_MIN_VOTES,
    ENSEMBLE_MODES,
    LLMMatcher,
    LLMMatchResult,
    PairJudgement,
    ensemble_candidates,
)


def stub_matcher() -> LLMMatcher:
    """Build an `LLMMatcher` without the OpenAI client or any model.

    Both `_postprocess` and `match_ensemble(mode_results=...)` are pure methods
    that never touch instance state, so an uninitialised instance is enough.
    """
    return object.__new__(LLMMatcher)


def mode_results(per_mode: dict[str, list[tuple[str, str, float]]]) -> dict[str, list[LLMMatchResult]]:
    """Convert per-representation candidates into cached mode results."""
    return {
        mode: [
            LLMMatchResult(source_id=src, target_id=tgt, confidence=conf)
            for src, tgt, conf in candidates
        ]
        for mode, candidates in per_mode.items()
    }


def test_rule_is_a_two_of_three_majority():
    # The published rule: at least two of the three representations, nothing less.
    assert ENSEMBLE_MIN_VOTES == 2
    assert ENSEMBLE_MODES == ("concept", "concept-parent", "concept-children")


def test_target_survives_with_two_votes_and_drops_single_vote_items():
    per_mode = {
        "concept": [("s0", "t0", 0.9), ("s0", "t1", 0.8), ("s1", "t2", 0.95)],
        "concept-parent": [("s0", "t0", 0.7), ("s1", "t3", 0.6)],
        "concept-children": [("s0", "t0", 0.5), ("s1", "t2", 0.4)],
    }
    result = ensemble_candidates(per_mode)
    # t0 has three votes (mean 0.7), t2 has two (mean 0.675); t1 and t3,
    # carried by a single representation each, are gone.
    assert result == [
        ("s0", "t0", pytest.approx(0.7)),
        ("s1", "t2", pytest.approx(0.675)),
    ]

    # With the stricter cut the two-vote target disappears as well.
    assert ensemble_candidates(per_mode, min_votes=3) == [
        ("s0", "t0", pytest.approx(0.7)),
    ]


def test_item_supported_by_only_one_representation_does_not_survive():
    per_mode = {
        "concept": [("s0", "t0", 0.99)],
        "concept-parent": [],
        "concept-children": [],
    }
    assert ensemble_candidates(per_mode) == []


def test_assembled_alignment_is_injective_on_both_sides(taxonomy):
    # Both s0 and s1 are voted for t0 by two representations each, so the
    # plurality rule would have emitted two matches sharing t0.
    per_mode = {
        "concept": [("s0", "t0", 0.9), ("s1", "t0", 0.8)],
        "concept-parent": [("s0", "t0", 0.85), ("s1", "t0", 0.75)],
        "concept-children": [
            ("s0", "t0", 0.8),
            ("s1", "t1", 0.6),
            ("s0", "t1", 0.5),
        ],
    }
    source = taxonomy({"s0": [], "s1": []})
    target = taxonomy({"t0": [], "t1": []})

    alignment, aggregated = stub_matcher().match_ensemble(
        source, target, mode_results=mode_results(per_mode)
    )

    pairs = alignment.as_pairs()
    assert pairs == {("s0", "t0")}
    assert len({src for src, _ in pairs}) == len(pairs)
    assert len({tgt for _, tgt in pairs}) == len(pairs)
    # The losing source is reported as unmatched rather than dropped.
    assert [(r.source_id, r.target_id) for r in aggregated] == [("s0", "t0"), ("s1", "")]


def test_votes_do_not_depend_on_representation_order():
    per_mode = {
        "concept": [("s0", "t0", 0.9), ("s1", "t1", 0.6)],
        "concept-parent": [("s0", "t0", 0.7)],
        "concept-children": [("s0", "t0", 0.5), ("s1", "t1", 0.4)],
    }
    base = ensemble_candidates(per_mode)
    for order in permutations(per_mode):
        shuffled = {mode: per_mode[mode] for mode in order}
        assert ensemble_candidates(shuffled) == base


def test_postprocessing_is_order_independent_with_confidence_ties(taxonomy):
    # Every "yes" judgement has the same confidence, so only the id tie-break
    # can decide; the old arrival-order greedy would give a different mapping
    # for each permuted input list.
    source = taxonomy({f"s{i}": [] for i in range(5)})
    judgements = [
        PairJudgement("s0", "t0", True, 0.9),
        PairJudgement("s0", "t1", True, 0.9),
        PairJudgement("s1", "t0", True, 0.9),
        PairJudgement("s1", "t1", True, 0.9),
        PairJudgement("s2", "t1", True, 0.9),
        PairJudgement("s2", "t2", True, 0.9),
        PairJudgement("s3", "t2", True, 0.9),
        PairJudgement("s3", "t3", True, 0.9),
        # The confidence cut is strict (`> CONFIDENCE_THRESHOLD`).
        PairJudgement("s4", "t4", True, CONFIDENCE_THRESHOLD),
        PairJudgement("s1", "t3", True, 0.5),
        # A high-confidence "no" judgement must be ignored regardless.
        PairJudgement("s2", "t0", False, 0.99),
    ]

    matcher = stub_matcher()
    base = matcher._postprocess(judgements, source)
    assert [r.source_id for r in base] == ["s0", "s1", "s2", "s3", "s4"]
    assert {(r.source_id, r.target_id) for r in base} == {
        ("s0", "t0"),
        ("s1", "t1"),
        ("s2", "t2"),
        ("s3", "t3"),
        ("s4", ""),
    }

    outcomes = set()
    for seed in range(20):
        permuted = judgements[:]
        random.Random(seed).shuffle(permuted)
        outcomes.add(
            tuple(
                (r.source_id, r.target_id)
                for r in matcher._postprocess(permuted, source)
            )
        )
    assert outcomes == {tuple((r.source_id, r.target_id) for r in base)}
