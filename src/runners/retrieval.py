"""Candidate-retrieval ceiling: recall@k of the ground truth in the candidate set.

The LLM only classifies the candidates a retriever hands it, so a ground-truth
pair that never enters the candidate set cannot be recovered, whatever the model
does.  Measuring recall@k of the ground truth therefore bounds the F1 that any
LLM configuration can reach, and answers the sensitivity-to-top-k question
without a single API call.  The bound is optimistic on purpose: it assumes the
classifier keeps every recalled candidate (precision 1.0), which is why it is
reported as an upper bound and not as a reachable score (finding 3.8).

The three retrievers of the experiments are compared — MiniLM cosine similarity
over the full node description, BM25 over the class name, and their union.  The
union of the hybrid does **not** have `top_k` candidates: it keeps the top-`top_k`
embedding candidates and adds the top-`bm25_k` lexical ones, truncated to
`total_k = min(top_k + bm25_k, 10)`, so the reported ceiling is quoted for the
number it really sends (`EXPERIMENT_K`).

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

# The experiments run with top-k = 5 for every retriever.  `HybridLLMMatcher`
# unions the top-`top_k` embedding candidates with the top-`bm25_k` lexical ones
# and truncates to `total_k = min(top_k + bm25_k, 10)`, so with the experimental
# `bm25_k = top_k = 5` it hands the LLM **10** candidates, not 5, and its ceiling
# must be quoted for k = 10 (findings 3.8).  Keep in sync with
# `src.matchers.hybrid.HybridLLMMatcher.__init__`.
EXPERIMENT_TOP_K = 5
EXPERIMENT_K: dict[str, int] = {
    "embedding-MiniLM": EXPERIMENT_TOP_K,
    "bm25": EXPERIMENT_TOP_K,
    "hybrid": min(EXPERIMENT_TOP_K + EXPERIMENT_TOP_K, 10),
}

# Label of the reconstructed ceiling: it can be reached only when every kept
# candidate is a true positive, i.e. at precision 1.0.
CEILING_LABEL = "upper bound reachable only at precision 1.0"


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
    """Upper bound on F1 at the given recall, reached only at precision 1.0.

    With recall fixed, F1 = 2·P·R/(P+R) grows monotonically in P, so the value at
    P = 1.0 is a ceiling nothing in the experiment can exceed.
    """
    return 2 * recall / (1 + recall) if recall > 0 else 0.0


def print_report(results: dict[str, dict], summary: dict[str, dict]) -> None:
    """Print recall@k per retriever and the implied F1 ceiling."""
    print("Candidate-set recall of the ground truth (mean over 21 pairs)")
    header = (f"{'retriever':<20}" + "".join(f"{'k=' + str(k):>8}" for k in KS)
              + f"{'sent k':>8}{'no cand':>9}{'F1 ceil*':>10}")
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        row = "".join(f"{s['recall'][k]:>8.3f}" for k in KS)
        print(f"{name:<20}{row}{s['sent_k']:>8}{s['uncovered']:>9.2f}{s['ceiling']:>10.3f}")
    print(f"* F1 ceiling = {CEILING_LABEL}; `sent k` is the number of candidates "
          "the retriever actually hands the LLM.")

    print("\nPairs with the weakest candidate coverage, per retriever at its own k")
    for name, per_pair in results.items():
        k = summary[name]["sent_k"]
        worst = sorted(per_pair.items(), key=lambda kv: kv[1][f"recall@{k}"])[:3]
        text = ", ".join(f"{pair}={v[f'recall@{k}']:.2f}" for pair, v in worst)
        print(f"  {name:<20}k={k:<4}{text}")


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
        sent_k = EXPERIMENT_K[name]
        summary[name] = {
            "recall": recall,
            "uncovered": mean(uncovered),
            "sent_k": sent_k,
            "ceiling": f1_ceiling(recall[sent_k]),
            "ceiling_label": CEILING_LABEL,
        }
        print(f"{name}: sends {sent_k} candidates, "
              f"recall@{sent_k} = {recall[sent_k]:.3f}, "
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
