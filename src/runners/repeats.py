"""Measure the run-to-run spread of the LLM matcher.

The cache holds a single run per configuration, so a difference of 0.02–0.05
between configurations cannot be told apart from generation noise.  This runner
repeats one configuration several times on a few pairs and reports the spread of
F1.  It never writes into `results/<approach>/`, so the published cache is left
alone; the summary lands in a single file next to it.

Each run reports its request count and retries, so the cost model of the note
(`|S|` requests per mode and pair, `O(k·|S|)` tokens) can be checked against a
real run rather than estimated.

Usage:
    uv run python -m src.runners.repeats --repeats 3
    uv run python -m src.runners.repeats --pairs cmt:confOf conference:confOf --mode concept
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, stdev

from src.data_loader import find_gt, load_alignments, load_taxonomies
from src.matchers.bm25_llm import BM25LLMMatcher
from src.matchers.hybrid import HybridLLMMatcher
from src.matchers.llm import LLMMatcher
from src.metrics import evaluate_1to1

DEFAULT_DATA = Path("data/processed/oaei")
DEFAULT_OUT = Path("results/llm-variance.json")
DEFAULT_PAIRS = ("cmt:confOf", "confOf:conference")

RETRIEVERS = {
    "hybrid": HybridLLMMatcher,
    "bm25": BM25LLMMatcher,
    "embedding": LLMMatcher,
}


def build_matcher(retriever: str, top_k: int) -> LLMMatcher:
    """Build the DeepSeek matcher of the experiments, with the given retriever."""
    factory = RETRIEVERS[retriever]
    return factory(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        top_k=top_k,
        temperature=0.0,
        max_workers=10,
        use_structured_output=False,
    )


def repeat_pair(
    matcher: LLMMatcher,
    taxonomies: dict,
    alignments_raw: list[dict],
    pair: str,
    mode: str,
    repeats: int,
) -> dict:
    """Run one configuration on one pair `repeats` times."""
    src_name, tgt_name = pair.split(":")
    src, tgt = taxonomies[src_name], taxonomies[tgt_name]
    gt = find_gt(src_name, tgt_name, alignments_raw)
    runs: list[dict] = []
    for i in range(repeats):
        prediction, _ = matcher.match(src, tgt, mode=mode)
        metrics = evaluate_1to1(prediction, gt)
        runs.append({
            "run": i + 1,
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "matches": prediction.match_count,
            "api_time": matcher.last_timing.get("api_time", 0.0),
            "requests": matcher.last_timing.get("requests", 0),
            "retries": matcher.last_timing.get("retries", 0),
        })
        print(f"  {pair} {mode} run {i + 1}: F1={metrics.f1:.4f} "
              f"(P={metrics.precision:.3f}, R={metrics.recall:.3f}, "
              f"{prediction.match_count} matches, "
              f"requests={matcher.last_timing.get('requests', 0)})")
    f1s = [r["f1"] for r in runs]
    return {
        "pair": pair,
        "mode": mode,
        "runs": runs,
        "f1_mean": mean(f1s),
        "f1_stdev": stdev(f1s) if len(f1s) > 1 else 0.0,
        "f1_min": min(f1s),
        "f1_max": max(f1s),
        "ranges": max(f1s) - min(f1s),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the LLM run-to-run spread.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS),
                        help="Pairs as source:target.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", default="concept",
                        choices=["concept", "concept-parent", "concept-children"])
    parser.add_argument("--retriever", default="hybrid", choices=sorted(RETRIEVERS))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    taxonomies = load_taxonomies(args.data_dir)
    alignments_raw = load_alignments(args.data_dir / "alignments.json")
    matcher = build_matcher(args.retriever, args.top_k)
    print(f"{args.retriever} + {matcher.model}, mode {args.mode}, "
          f"{args.repeats} runs per pair\n")

    per_pair = [
        repeat_pair(matcher, taxonomies, alignments_raw, pair, args.mode, args.repeats)
        for pair in args.pairs
    ]

    print("\nSpread over runs")
    header = f"{'pair':<22}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}{'range':>8}"
    print(header)
    print("-" * len(header))
    for entry in per_pair:
        print(f"{entry['pair']:<22}{entry['f1_mean']:>8.3f}{entry['f1_stdev']:>8.3f}"
              f"{entry['f1_min']:>8.3f}{entry['f1_max']:>8.3f}{entry['ranges']:>8.3f}")
    overall = [e["f1_mean"] for e in per_pair]
    print(f"\nMean F1 over the pairs: {mean(overall):.3f}, "
          f"largest range within a pair: {max(e['ranges'] for e in per_pair):.3f}")

    args.out.write_text(json.dumps({
        "model": matcher.model,
        "retriever": args.retriever,
        "mode": args.mode,
        "top_k": args.top_k,
        "repeats": args.repeats,
        "pairs": per_pair,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
