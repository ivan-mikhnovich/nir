"""Evaluation metrics for taxonomy matching."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .taxonomy import Alignment


@dataclass
class MatchMetrics:
    """Standard matching evaluation metrics."""

    precision: float
    recall: float
    f1: float
    hits_at_1: float
    hits_at_3: float
    hits_at_5: float
    mrr: float  # Mean Reciprocal Rank.
    total_ground_truth: int
    total_predicted: int
    true_positives: int


def evaluate_1to1(
    predicted: Alignment,
    ground_truth: Alignment,
) -> MatchMetrics:
    """Evaluate one-to-one matching results.

    For each source node, checks whether the predicted best match
    is in the ground-truth alignment.

    Args:
        predicted: The predicted alignment (one match per source node).
        ground_truth: The reference alignment.

    Returns:
        MatchMetrics with precision, recall, f1, hits@k, MRR.
    """
    gt_pairs = ground_truth.as_pairs()

    # Build predicted mapping: source_id -> (best target_id, confidence).
    pred_map: dict[str, str] = {}
    for m in predicted.matches:
        pred_map[m.source_id] = m.target_id

    # True positives: predicted pair exists in ground truth.
    tp = sum(1 for s, t in pred_map.items() if (s, t) in gt_pairs)

    precision = tp / len(pred_map) if len(pred_map) > 0 else 0.0
    recall = tp / len(gt_pairs) if len(gt_pairs) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Hits@k and MRR require ranked predictions; for 1-to-1 we compute
    # whether the correct target is the top choice (hits@1) and MRR.
    # Since we only have best match, hits@1 = recall for source-side eval.
    hits_at_1 = recall  # In 1-to-1: correct target is top-1 or not.
    hits_at_3 = recall  # Same: only one prediction per source.
    hits_at_5 = recall

    # MRR: 1/rank for each ground-truth source.
    mrr = recall  # Rank 1 if correct, 0 otherwise.

    return MatchMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        hits_at_1=hits_at_1,
        hits_at_3=hits_at_3,
        hits_at_5=hits_at_5,
        mrr=mrr,
        total_ground_truth=len(gt_pairs),
        total_predicted=len(pred_map),
        true_positives=tp,
    )


def evaluate_ranked(
    src_ids: list[str],
    ranked_tgt_ids: list[list[str]],  # For each source, list of targets ranked by score.
    ground_truth: Alignment,
) -> MatchMetrics:
    """Evaluate with ranked predictions (top-k for each source node).

    Args:
        src_ids: Source node IDs (ordered).
        ranked_tgt_ids: For each source node, a list of target node IDs
            ordered from best to worst.
        ground_truth: Reference alignment.

    Returns:
        MatchMetrics with full hits@k and MRR.
    """
    gt_pairs = ground_truth.as_pairs()
    # Build map: source_id -> correct target_id.
    gt_map: dict[str, str] = {}
    for s, t in gt_pairs:
        gt_map[s] = t

    tp = 0
    reciprocal_ranks: list[float] = []

    for i, src_id in enumerate(src_ids):
        correct = gt_map.get(src_id)
        if correct is None:
            continue
        try:
            rank = ranked_tgt_ids[i].index(correct) + 1  # 1-indexed.
            if rank == 1:
                tp += 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)

    total_pred = len(src_ids)
    total_gt = len(gt_pairs)

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    # Hits@k.
    def hits_at(k: int) -> float:
        correct_in_top_k = 0
        total_with_gt = 0
        for i, src_id in enumerate(src_ids):
            if src_id in gt_map:
                total_with_gt += 1
                if gt_map[src_id] in ranked_tgt_ids[i][:k]:
                    correct_in_top_k += 1
        return correct_in_top_k / total_with_gt if total_with_gt > 0 else 0.0

    return MatchMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        hits_at_1=hits_at(1),
        hits_at_3=hits_at(3),
        hits_at_5=hits_at(5),
        mrr=mrr,
        total_ground_truth=total_gt,
        total_predicted=total_pred,
        true_positives=tp,
    )


def format_metrics(m: MatchMetrics, name: str = "") -> str:
    """Format metrics as a readable string."""
    lines = [f"--- {name} ---" if name else "--- Results ---"]
    lines.append(f"  Precision:  {m.precision:.4f}")
    lines.append(f"  Recall:     {m.recall:.4f}")
    lines.append(f"  F1-score:   {m.f1:.4f}")
    lines.append(f"  Hits@1:     {m.hits_at_1:.4f}")
    lines.append(f"  Hits@3:     {m.hits_at_3:.4f}")
    lines.append(f"  Hits@5:     {m.hits_at_5:.4f}")
    lines.append(f"  MRR:        {m.mrr:.4f}")
    lines.append(f"  TP={m.true_positives}  Pred={m.total_predicted}  GT={m.total_ground_truth}")
    return "\n".join(lines)
