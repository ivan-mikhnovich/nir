"""Evaluation metrics for taxonomy matching."""

from __future__ import annotations

from dataclasses import dataclass

from .matching_rules import per_source_best
from .taxonomy import Alignment


@dataclass
class MatchMetrics:
    """Standard matching evaluation metrics."""

    precision: float
    recall: float
    f1: float
    total_ground_truth: int
    total_predicted: int
    true_positives: int


def _predicted_map(predicted: Alignment) -> dict[str, str]:
    """Reduce a prediction to source_id → best target_id.

    The best-confidence match of every source node wins; ties are broken by
    target id, so the result does not depend on the order of
    ``predicted.matches``.
    """
    candidates = [
        (m.source_id, m.target_id, float(m.confidence))
        for m in predicted.matches
    ]
    return {src: tgt for src, tgt, _conf in per_source_best(candidates)}


def evaluate_1to1(
    predicted: Alignment,
    ground_truth: Alignment,
) -> MatchMetrics:
    """Evaluate one-to-one matching results.

    For each source node the predicted match with the highest confidence is
    kept (ties broken by target id); the other matches of that source are
    discarded.  The mapping is therefore injective on the source side, while
    a target claimed by several sources is counted once as a true positive and
    the remaining claims are false positives — target-side non-injectivity is
    penalised by precision, not silently removed.

    Args:
        predicted: The predicted alignment (one match per source node).
        ground_truth: The reference alignment.

    Returns:
        MatchMetrics with precision, recall and F1.
    """
    gt_pairs = ground_truth.as_pairs()

    # source_id → best-confidence target_id (order-independent).
    pred_map = _predicted_map(predicted)

    # True positives: predicted pair exists in ground truth.
    tp = sum(1 for s, t in pred_map.items() if (s, t) in gt_pairs)

    precision = tp / len(pred_map) if len(pred_map) > 0 else 0.0
    recall = tp / len(gt_pairs) if len(gt_pairs) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return MatchMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        total_ground_truth=len(gt_pairs),
        total_predicted=len(pred_map),
        true_positives=tp,
    )
