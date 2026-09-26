"""Paired significance tests between matcher configurations.

Reads the per-record detail produced by `src.runners.consistency` and compares
configurations pair by pair, so the report can say not only "0.688 against
0.663" but whether that gap survives the spread over the 21 OAEI pairs.

Two configurations are built for every approach: the raw prediction, and the
prediction cut at the confidence threshold chosen leave-one-pair-out.  Each
comparison reports the mean per-pair difference, a paired bootstrap confidence
interval, a Wilcoxon signed-rank test and the win/tie/loss count.

Usage:
    uv run python -m src.runners.stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import wilcoxon

DEFAULT_IN = Path("results/consistency")
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260601

# Comparisons to report, as (label, first configuration, second configuration).
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("LLM vs StringEquiv", "llm-bm25 raw", "string-equiv raw"),
    ("hybrid vs StringEquiv", "llm-hybrid raw", "string-equiv raw"),
    ("LaBSE+threshold vs StringEquiv", "embedding-LaBSE thr", "string-equiv raw"),
    ("LaBSE+threshold vs hybrid", "embedding-LaBSE thr", "llm-hybrid raw"),
    ("LaBSE+threshold vs raw LaBSE", "embedding-LaBSE thr", "embedding-LaBSE raw"),
    ("LaBSE+threshold vs MiniLM+threshold", "embedding-LaBSE thr", "embedding-MiniLM thr"),
    ("BM25 retriever vs MiniLM retriever", "llm-bm25 raw", "llm-deepseek raw"),
    ("hybrid retriever vs BM25 retriever", "llm-hybrid raw", "llm-bm25 raw"),
    ("gpt-4.1-mini vs deepseek", "llm-gpt raw", "llm-deepseek raw"),
)


def load_records(path: Path) -> dict[str, dict]:
    """Load the per-record detail written by the consistency runner."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_configurations(alignments: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Build every `<approach> <variant>` configuration as pair → F1.

    The variant `raw` scores the matcher's own output, `thr` scores the same
    output cut at the leave-one-out threshold.  The best mode of a pair stands
    for the approach, which is how the published comparison table aggregates.
    """
    configs: dict[str, dict[str, float]] = {}
    for approach, records in alignments.items():
        by_pair_raw: dict[str, list[float]] = {}
        by_pair_thr: dict[str, list[float]] = {}
        for record in records.values():
            pair = record["pair"]
            by_pair_raw.setdefault(pair, []).append(record["baseline"]["f1"])
            if "loo" in record:
                by_pair_thr.setdefault(pair, []).append(record["loo"]["f1"])
        if by_pair_raw:
            configs[f"{approach} raw"] = {p: max(v) for p, v in by_pair_raw.items()}
        if by_pair_thr:
            configs[f"{approach} thr"] = {p: max(v) for p, v in by_pair_thr.items()}
    return configs


def paired_test(
    first: dict[str, float],
    second: dict[str, float],
    rounds: int = BOOTSTRAP_ROUNDS,
) -> dict:
    """Compare two per-pair F1 vectors over the pairs they share."""
    pairs = sorted(set(first) & set(second))
    diffs = np.array([first[p] - second[p] for p in pairs])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(diffs, size=(rounds, len(diffs)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    nonzero = diffs[diffs != 0]
    if len(nonzero) >= 2:
        statistic, p_value = wilcoxon(nonzero)
    else:
        statistic, p_value = float("nan"), 1.0
    return {
        "pairs": len(pairs),
        "first_mean": mean(first[p] for p in pairs),
        "second_mean": mean(second[p] for p in pairs),
        "mean_difference": float(diffs.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "wins": int((diffs > 1e-9).sum()),
        "ties": int((np.abs(diffs) <= 1e-9).sum()),
        "losses": int((diffs < -1e-9).sum()),
    }


def print_report(results: list[dict], configs: dict[str, dict[str, float]]) -> None:
    """Print the configuration means and the paired comparisons."""
    print("Configurations (mean F1 over the shared pairs)")
    header = f"{'configuration':<36}{'pairs':>6}{'mean F1':>9}"
    print(header)
    print("-" * len(header))
    for name, values in sorted(configs.items(), key=lambda kv: -mean(kv[1].values())):
        print(f"{name:<36}{len(values):>6}{mean(values.values()):>9.3f}")

    print("\nPaired comparisons over the 21 OAEI pairs")
    header = (f"{'comparison':<36}{'first':>8}{'second':>8}{'diff':>8}"
              f"{'95% CI':>18}{'p':>9}{'W/T/L':>10}")
    print(header)
    print("-" * len(header))
    for r in results:
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
        print(f"{r['label']:<36}{r['first_mean']:>8.3f}{r['second_mean']:>8.3f}"
              f"{r['mean_difference']:>+8.3f}{ci:>18}{r['p_value']:>9.4f}{wtl:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired tests between matcher configurations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN / "alignments.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_IN / "stats.json")
    args = parser.parse_args()

    alignments = load_records(args.input)
    configs = build_configurations(alignments)

    results: list[dict] = []
    for label, first_name, second_name in COMPARISONS:
        if first_name not in configs or second_name not in configs:
            continue
        test = paired_test(configs[first_name], configs[second_name])
        results.append({"label": label, "first": first_name, "second": second_name, **test})

    print_report(results, configs)
    args.out.write_text(
        json.dumps({"configurations": configs, "comparisons": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
