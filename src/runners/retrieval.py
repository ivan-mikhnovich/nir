"""Candidate-retrieval ceiling: recall@k of the ground truth in the candidate set.

The LLM only classifies the candidates a retriever hands it, so a ground-truth
pair that never enters the candidate set cannot be recovered, whatever the model
does.  Measuring recall@k of the ground truth therefore bounds the F1 that any
LLM configuration can reach, and answers the sensitivity-to-top-k question
without a single API call.

The three retrievers of the experiments are compared: MiniLM cosine similarity
over the full node description, BM25 over the class name, and their union.

Usage:
    uv run python -m src.runners.retrieval
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from src.data_loader import find_gt, load_alignments, load_taxonomies
from src.matchers.bm25_llm import BM25LLMMatcher
from src.matchers.hybrid import HybridLLMMatcher
from src.matchers.llm import LLMMatcher

DEFAULT_DATA = Path("data/processed/oaei")
DEFAULT_OUT = Path("results/consistency/retrieval.json")

# Candidate-set sizes to measure.
KS: tuple[int, ...] = (1, 2, 3, 5, 10)

# The experiments run with top-k = 5, so the ceiling is quoted for that size.
RUN_TOP_K = 5


def retriever_factories() -> dict[str, Callable[[int], LLMMatcher]]:
    """Build one factory per retriever, parameterised by the candidate count."""
    return {
        "embedding-MiniLM": lambda k: LLMMatcher(top_k=k),
        "bm25": lambda k: BM25LLMMatcher(top_k=k),
        "hybrid": lambda k: HybridLLMMatcher(top_k=k),
    }


def candidate_recall(
    candidates: dict[str, list[tuple[str, float]]],
    gt_map: dict[str, str],
    k: int,
) -> float:
    """Share of ground-truth pairs whose target is among the top-k candidates."""
    if not gt_map:
        return 0.0
    hits = sum(
        1 for src_id, tgt_id in gt_map.items()
        if tgt_id in [cid for cid, _ in candidates.get(src_id, ())[:k]]
    )
    return hits / len(gt_map)


def uncovered_targets(
    candidates: dict[str, list[tuple[str, float]]],
    gt_map: dict[str, str],
) -> float:
    """Share of ground-truth sources that got no candidate at all."""
    if not gt_map:
        return 0.0
    empty = sum(1 for src_id in gt_map if not candidates.get(src_id))
    return empty / len(gt_map)


def f1_ceiling(recall: float) -> float:
    """Best F1 reachable with the given recall when precision is perfect."""
    return 2 * recall / (1 + recall) if recall > 0 else 0.0


def print_report(results: dict[str, dict], summary: dict[str, dict]) -> None:
    """Print recall@k per retriever and the implied F1 ceiling."""
    print("Candidate-set recall of the ground truth (mean over 21 pairs)")
    header = (f"{'retriever':<20}" + "".join(f"{'k=' + str(k):>8}" for k in KS)
              + f"{'no cand':>9}{'F1 ceil @5':>12}")
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        row = "".join(f"{s['recall'][k]:>8.3f}" for k in KS)
        print(f"{name:<20}{row}{s['uncovered']:>9.2f}{s['ceiling']:>12.3f}")

    print("\nPairs with the weakest candidate coverage at top-k = 5")
    for name, per_pair in results.items():
        worst = sorted(per_pair.items(), key=lambda kv: kv[1][f"recall@{RUN_TOP_K}"])[:3]
        text = ", ".join(f"{pair}={v[f'recall@{RUN_TOP_K}']:.2f}" for pair, v in worst)
        print(f"  {name:<20}{text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the candidate-retrieval ceiling.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    taxonomies = load_taxonomies(args.data_dir)
    alignments_raw = load_alignments(args.data_dir / "alignments.json")
    pairs = [(a["source"], a["target"]) for a in alignments_raw]

    results: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for name, factory in retriever_factories().items():
        matcher = factory(max(KS))
        per_pair: dict[str, dict] = {}
        uncovered: list[float] = []
        for src, tgt in pairs:
            candidates = matcher.retrieve_candidates(taxonomies[src], taxonomies[tgt])
            gt = find_gt(src, tgt, alignments_raw)
            gt_map = {m.source_id: m.target_id for m in gt.matches}
            per_pair[f"{src}↔{tgt}"] = {
                f"recall@{k}": candidate_recall(candidates, gt_map, k) for k in KS
            }
            uncovered.append(uncovered_targets(candidates, gt_map))
        results[name] = per_pair
        recall = {k: mean(v[f"recall@{k}"] for v in per_pair.values()) for k in KS}
        summary[name] = {
            "recall": recall,
            "uncovered": mean(uncovered),
            "ceiling": f1_ceiling(recall[RUN_TOP_K]),
        }
        print(f"{name}: recall@{RUN_TOP_K} = {recall[RUN_TOP_K]:.3f}, "
              f"no candidates for {summary[name]['uncovered']:.1%} of ground-truth sources")

    print()
    print_report(results, summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "per_pair": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
