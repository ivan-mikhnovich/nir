"""Tests for the one-to-one evaluation metric.

The metric decides every number in the work, so it is pinned on hand-computed
examples. Two properties matter most and both were broken before the repair:
a source's *best* match must win (the old code kept whichever came last in the
list), and the result must not depend on the order of the matches.
"""

from __future__ import annotations

import random
from dataclasses import asdict

import pytest

from src.metrics import evaluate_1to1
from src.runners.consistency import aggregate, check_all_alignments
from src.taxonomy import Alignment, TaxonMatch


def alignment(source: str, target: str, pairs: list[tuple[str, str, float]]) -> Alignment:
    """Build an alignment from ``(source_id, target_id, confidence)`` triples."""
    return Alignment(
        source=source,
        target=target,
        matches=[TaxonMatch(s, t, confidence=c) for s, t, c in pairs],
    )


def test_hand_computed_precision_recall_f1():
    gt = alignment("a", "b", [("s0", "t0", 1.0), ("s1", "t1", 1.0), ("s2", "t2", 1.0), ("s3", "t3", 1.0)])
    pred = alignment("a", "b", [("s0", "t0", 0.9), ("s1", "t1", 0.8), ("s2", "tX", 0.7), ("sX", "t3", 0.6)])
    metrics = evaluate_1to1(pred, gt)
    assert metrics.true_positives == 2
    assert metrics.total_predicted == 4
    assert metrics.total_ground_truth == 4
    assert (metrics.precision, metrics.recall, metrics.f1) == (0.5, 0.5, 0.5)


def test_empty_prediction_and_empty_ground_truth_yield_zero():
    gt = alignment("a", "b", [("s0", "t0", 1.0)])
    assert evaluate_1to1(alignment("a", "b", []), gt).f1 == 0.0
    assert evaluate_1to1(alignment("a", "b", []), gt).precision == 0.0
    assert evaluate_1to1(gt, alignment("a", "b", [])).f1 == 0.0
    assert evaluate_1to1(gt, alignment("a", "b", [])).recall == 0.0


def test_best_match_of_a_source_wins_regardless_of_position():
    gt = alignment("a", "b", [("s0", "t0", 1.0)])
    good_then_weak = alignment("a", "b", [("s0", "t0", 0.9), ("s0", "t1", 0.4)])
    weak_then_good = alignment("a", "b", [("s0", "t1", 0.4), ("s0", "t0", 0.9)])
    assert evaluate_1to1(good_then_weak, gt).f1 == 1.0
    assert evaluate_1to1(weak_then_good, gt).f1 == 1.0


def test_metrics_do_not_depend_on_match_order():
    gt = alignment("a", "b", [("s0", "t0", 1.0), ("s1", "t1", 1.0)])
    pairs = [("s0", "t0", 0.9), ("s1", "t1", 0.8), ("s2", "t2", 0.7)]
    base = evaluate_1to1(alignment("a", "b", pairs), gt)
    for seed in range(5):
        shuffled = pairs[:]
        random.Random(seed).shuffle(shuffled)
        metrics = evaluate_1to1(alignment("a", "b", shuffled), gt)
        assert (metrics.precision, metrics.recall, metrics.f1) == (base.precision, base.recall, base.f1)


def test_target_side_collision_is_penalised_by_precision():
    gt = alignment("a", "b", [("s0", "t0", 1.0)])
    pred = alignment("a", "b", [("s0", "t0", 0.9), ("s1", "t0", 0.8)])
    metrics = evaluate_1to1(pred, gt)
    assert metrics.true_positives == 1
    assert metrics.total_predicted == 2
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0


def test_two_sources_matched_to_one_ground_truth_pair_count_once():
    gt = alignment("a", "b", [("s0", "t0", 1.0)])
    pred = alignment("a", "b", [("s0", "t0", 0.9), ("s0", "t0", 0.8)])
    metrics = evaluate_1to1(pred, gt)
    assert metrics.true_positives == 1
    assert metrics.total_predicted == 1
    assert metrics.f1 == 1.0


@pytest.mark.parametrize(
    ("reference_sizes", "expected_precision", "expected_recall", "cut_precision", "cut_recall"),
    [
        ((1, 4), 0.75, 0.6, 1.0, 0.4),
        ((0, 0), 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_pooled_threshold_metrics_count_all_reference_pairs(
    taxonomy, reference_sizes, expected_precision, expected_recall, cut_precision, cut_recall,
):
    """Pooling weights unequal references and retains missed pairs after a cut."""
    taxonomies, cached, references = {}, {"test": {}}, []
    predictions = [
        [("s0", "t0", 0.9), ("sX", "tX", 0.6)],
        [("s0", "t0", 0.9), ("s1", "t1", 0.6)],
    ]
    for (source, target), size, pairs in zip(
        (("a", "b"), ("c", "d")), reference_sizes, predictions,
    ):
        taxonomies[source] = taxonomy({f"s{i}": [] for i in (0, 1, 2, 3, "X")})
        taxonomies[target] = taxonomy({f"t{i}": [] for i in (0, 1, 2, 3, "X")})
        reference = alignment(
            source, target, [(f"s{i}", f"t{i}", 1.0) for i in range(size)],
        )
        references.append(asdict(reference))
        prediction = alignment(source, target, pairs)
        pair = f"{source}↔{target}"
        cached["test"][pair] = {
            "pair": pair,
            "source": source,
            "target": target,
            "f1": evaluate_1to1(prediction, reference).f1,
            "match_pairs": asdict(prediction)["matches"],
        }

    reports, skipped = check_all_alignments(taxonomies, cached, references, 0.7)
    assert skipped == 0
    summary = aggregate(reports)["test"]
    precision = summary["threshold_sweep_pooled_precision"]
    recall = summary["threshold_sweep_pooled_recall"]
    assert precision["0.00"] == pytest.approx(expected_precision)
    assert recall["0.00"] == pytest.approx(expected_recall)
    assert precision["0.70"] == pytest.approx(cut_precision)
    assert recall["0.70"] == pytest.approx(cut_recall)
    assert precision["0.99"] == 0.0
    assert recall["0.99"] == 0.0
