"""Paired significance tests between matcher configurations.

Reads the per-record detail produced by `src.runners.consistency` and compares
configurations pair by pair, so the report can say not only "0.677 against
0.663" but whether that gap survives the spread over the 21 OAEI pairs.

Two configurations are built for every approach: the raw prediction, and the
prediction cut at the confidence threshold chosen leave-one-pair-out.  A
multi-representation (LLM) approach additionally has its `>=2-of-3` ensemble
record, and the comparisons whose subject *is* the aggregation use that record.
Each comparison reports the mean per-pair difference, a paired bootstrap
confidence interval, a Wilcoxon signed-rank test and the win/tie/loss count.
Every configuration is computed from the cached per-pair records, never from a
number written down here.

Multiplicity (finding M2.2 / 3.2): the corrected family has more than one
comparison, so the report also publishes Holm and Bonferroni adjusted p-values;
the family grows whenever a comparison is added, and the correction is applied
over the whole family (`len(results)`), not over a hardcoded count.

Aggregation (contract C5): the LLM row of a pair is **selection-free** — the
mean over its three representations, never the best one; the LOO selection
(`loo`), the `>=2-of-3` ensemble and the per-pair maximum (`oracle`) are
reported separately, the last one labelled as an oracle.

Averaging (finding 1.16): every configuration is macro-averaged over the pairs
(the published convention); pooled (micro) F1 is reported next to it for the
single-record configurations, where the pooled counts are well defined.

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
DEFAULT_REFERENCE = Path("data/processed/oaei/alignments.json")
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260601

# The three representations of an LLM approach, and its ensemble record.
LLM_MODES: tuple[str, ...] = ("concept", "concept-parent", "concept-children")
ENSEMBLE_MODE = "ensemble"

# Comparisons to report, as (label, first configuration, second configuration).
# The first three are the headline comparisons: the best single-record result
# (ruRoberta-large with the leave-one-pair-out threshold) against the string
# baseline and against the best LLM aggregation, and that aggregation against
# the string baseline.  LLM rows use `<approach> ensemble` (the >=2-of-3 rule)
# where the aggregation itself is the subject, and stay on `raw` (the mean over
# the three representations) where the comparison is the one published before.
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ruRoberta+thr vs StringEquiv", "embedding-ruRoberta-large thr", "string-equiv raw"),
    ("ruRoberta+thr vs hybrid ensemble", "embedding-ruRoberta-large thr", "llm-hybrid ensemble"),
    ("hybrid ensemble vs StringEquiv", "llm-hybrid ensemble", "string-equiv raw"),
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

# Diagnostic only, not part of the corrected comparisons: the embedding model is
# chosen on these same 21 pairs, so the claim needs its own numbers (finding 3.5).
# Both rows are on the raw (un-thresholded) embedding output, because the
# question is which model to embed with, not how to post-process it.
MODEL_SELECTIONS: tuple[tuple[str, str, str], ...] = (
    ("LaBSE vs MiniLM", "embedding-LaBSE raw", "embedding-MiniLM raw"),
    ("ruRoberta-large vs LaBSE", "embedding-ruRoberta-large raw", "embedding-LaBSE raw"),
)


def load_records(path: Path) -> dict[str, dict]:
    """Load the per-record detail written by the consistency runner."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _group_by_pair(records: dict[str, dict]) -> dict[str, dict[str | None, dict]]:
    """Group an approach's records as pair → mode → record."""
    by_pair: dict[str, dict[str | None, dict]] = {}
    for record in records.values():
        by_pair.setdefault(record["pair"], {})[record.get("mode")] = record
    return by_pair


def loo_mode_selection(by_pair: dict[str, dict[str | None, dict]]) -> dict[str, float]:
    """Per pair, the F1 of the mode chosen leave-one-pair-out (contract C5).

    For the held-out pair the mode with the best mean F1 over the *other* pairs
    wins; ties go to the first mode of `LLM_MODES`, so the choice is
    deterministic and never sees the pair it is applied to.
    """
    pairs = sorted(by_pair)
    out: dict[str, float] = {}
    for held in pairs:
        others = [p for p in pairs if p != held]
        scores = {
            mode: mean(
                by_pair[p][mode]["baseline"]["f1"]
                for p in others if mode in by_pair[p]
            )
            for mode in LLM_MODES
            if any(mode in by_pair[p] for p in others)
        }
        best_mode = max(scores, key=lambda m: (scores[m], -LLM_MODES.index(m)))
        out[held] = by_pair[held][best_mode]["baseline"]["f1"]
    return out


def build_configurations(alignments: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Build every `<approach> <variant>` configuration as pair → F1.

    Variants of a multi-mode approach (an LLM):
        `raw` — mean over the three representations; selection-free, this is the
            published row (contract C5).
        `thr` — mean over the leave-one-out-thresholded representations.
        `loo` — mode chosen leave-one-pair-out (`loo_mode_selection`).
        `ensemble` — the ensemble record, if the approach has one.
        `oracle` — best representation per pair; an oracle, never published.
    A single-record approach (embedding, string-equiv): `raw` is its own output,
    `thr` the same output cut at the leave-one-out threshold.
    """
    configs: dict[str, dict[str, float]] = {}
    for approach, records in alignments.items():
        by_pair = _group_by_pair(records)
        raw: dict[str, float] = {}
        thr: dict[str, float] = {}
        oracle: dict[str, float] = {}
        ensemble: dict[str, float] = {}
        multi_mode = False
        for pair, modes in by_pair.items():
            core = [modes[m] for m in LLM_MODES if m in modes]
            if core:
                multi_mode = True
                raw[pair] = mean(m["baseline"]["f1"] for m in core)
                oracle[pair] = max(m["baseline"]["f1"] for m in core)
                if all("loo" in m for m in core):
                    thr[pair] = mean(m["loo"]["f1"] for m in core)
            else:
                record = modes.get(None)
                if record is None:
                    continue
                raw[pair] = record["baseline"]["f1"]
                if "loo" in record:
                    thr[pair] = record["loo"]["f1"]
            if ENSEMBLE_MODE in modes:
                ensemble[pair] = modes[ENSEMBLE_MODE]["baseline"]["f1"]
        if raw:
            configs[f"{approach} raw"] = raw
        if thr:
            configs[f"{approach} thr"] = thr
        if multi_mode:
            configs[f"{approach} loo"] = loo_mode_selection(by_pair)
            configs[f"{approach} oracle"] = oracle
        if ensemble:
            configs[f"{approach} ensemble"] = ensemble
    return configs


def reference_gt_sizes(path: Path) -> dict[str, int]:
    """Number of reference pairs per canonical pair key."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    sizes: dict[str, int] = {}
    for record in raw:
        key = "↔".join(sorted([record["source"], record["target"]]))
        sizes[key] = len(record["matches"])
    return sizes


def pooled_configurations(
    alignments: dict[str, dict],
    gt_sizes: dict[str, int],
) -> dict[str, float]:
    """Pooled (micro) F1 of every single-record configuration.

    Pooling needs one prediction set per pair, so it is defined only for the
    approaches without representations (embedding, string-equiv); the
    multi-representation rows are macro by construction.  TP comes from the
    stored precision × predicted count, the reference size from `gt_sizes`.
    """
    pooled: dict[str, float] = {}
    for approach, records in alignments.items():
        by_pair = _group_by_pair(records)
        if any(modes.keys() - {None, ENSEMBLE_MODE} for modes in by_pair.values()):
            continue
        for variant in ("raw", "thr"):
            block_key = "baseline" if variant == "raw" else "loo"
            tp = pred = gt = 0
            found = False
            for pair, modes in by_pair.items():
                record = modes.get(None)
                if record is None or block_key not in record:
                    continue
                block = record[block_key]
                # The baseline block carries no count of its own; the record
                # level `matches` is the size of that prediction set.
                predicted = int(record["matches"] if variant == "raw" else block["matches"])
                tp += round(block["precision"] * predicted)
                pred += predicted
                gt += gt_sizes.get(pair, 0)
                found = True
            if found and pred and gt:
                pooled[f"{approach} {variant}"] = 2 * tp / (pred + gt)
    return pooled


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


def adjust_multiplicity(results: list[dict]) -> None:
    """Add Holm and Bonferroni adjusted p-values to every comparison in place.

    Holm is the step-down correction: sort the raw p ascending and multiply the
    i-th by `m - i`, keeping the running maximum so the sequence stays monotone;
    Bonferroni multiplies every raw p by `m`.  Both cap at 1.
    """
    m = len(results)
    order = sorted(range(m), key=lambda i: results[i]["p_value"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, results[index]["p_value"] * (m - rank))
        running = max(running, adjusted)
        results[index]["p_holm"] = min(1.0, running)
        results[index]["p_bonferroni"] = min(1.0, results[index]["p_value"] * m)


def oracle_variant(configs: dict[str, dict[str, float]], name: str) -> str | None:
    """The per-pair-maximum counterpart of a configuration, when it exists."""
    candidate = name.replace(" raw", " oracle")
    return candidate if name.endswith(" raw") and candidate in configs else None


def print_report(
    results: list[dict],
    configs: dict[str, dict[str, float]],
    pooled: dict[str, float],
    selections: list[dict] | None = None,
    oracle_results: list[dict] | None = None,
) -> None:
    """Print the configuration means and the paired comparisons."""
    print("Configurations (macro mean F1 over the shared pairs; pooled = micro)")
    header = f"{'configuration':<36}{'pairs':>6}{'macro F1':>10}{'pooled F1':>11}"
    print(header)
    print("-" * len(header))
    for name, values in sorted(configs.items(), key=lambda kv: -mean(kv[1].values())):
        pool = pooled.get(name)
        print(f"{name:<36}{len(values):>6}{mean(values.values()):>10.4f}"
              f"{(f'{pool:.4f}' if pool is not None else '—'):>11}")

    print("\nPaired comparisons over the 21 OAEI pairs")
    header = (f"{'comparison':<36}{'first':>8}{'second':>8}{'diff':>8}"
              f"{'95% CI':>18}{'p':>9}{'Holm':>9}{'Bonf.':>8}{'W/T/L':>10}")
    print(header)
    print("-" * len(header))
    for r in results:
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
        print(f"{r['label']:<36}{r['first_mean']:>8.3f}{r['second_mean']:>8.3f}"
              f"{r['mean_difference']:>+8.3f}{ci:>18}{r['p_value']:>9.4f}"
              f"{r['p_holm']:>9.4f}{r['p_bonferroni']:>8.4f}{wtl:>10}")
    print("* p is the raw Wilcoxon p-value; Holm/Bonferroni correct it for the "
          f"{len(results)} simultaneous comparisons.")

    if oracle_results:
        print("\nReference family: the same comparisons on the per-pair maximum")
        print("(`oracle`, the pre-C5 aggregation; shown so the published numbers of the")
        print("first review stay checkable).  Rows without a per-pair maximum on either")
        print("side (the ensemble and threshold configurations) are skipped.")
        header = f"{'comparison':<36}{'p':>9}{'Holm':>9}{'Bonf.':>8}"
        print(header)
        print("-" * len(header))
        for r in oracle_results:
            print(f"{r['label']:<36}{r['p_value']:>9.4f}{r['p_holm']:>9.4f}"
                  f"{r['p_bonferroni']:>8.4f}")

    if selections:
        print("\nModel selection (diagnostic, outside the corrected family)")
        for selection in selections:
            print(f"  {selection['label']}: diff={selection['mean_difference']:+.4f} "
                  f"95% CI=[{selection['ci_low']:+.4f}, {selection['ci_high']:+.4f}] "
                  f"p={selection['p_value']:.4f} W/T/L={selection['wins']}/"
                  f"{selection['ties']}/{selection['losses']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired tests between matcher configurations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_IN / "alignments.json")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                        help="reference alignments, for the pooled (micro) F1")
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
    adjust_multiplicity(results)

    # Reference family: identical comparisons, but every LLM row is its per-pair
    # maximum.  This is the family the first review corrected (0.0195 → Holm
    # 0.1562); it is published only as a diagnostic, because per-pair maxima are
    # oracle selection (contract C5).
    oracle_results: list[dict] = []
    for label, first_name, second_name in COMPARISONS:
        # Swap in the per-pair maximum where the row is a multi-mode approach;
        # comparisons of single-mode configurations stay as they are.
        first_oracle = oracle_variant(configs, first_name)
        second_oracle = oracle_variant(configs, second_name)
        # Строку без per-pair максимума хотя бы с одной стороны пропускаем:
        # иначе она повторила бы основной тест под другим заголовком.
        if first_oracle is None and second_oracle is None:
            continue
        test = paired_test(configs[first_oracle or first_name],
                           configs[second_oracle or second_name])
        oracle_results.append({
            "label": label,
            "first": first_oracle or first_name,
            "second": second_oracle or second_name,
            **test,
        })
    adjust_multiplicity(oracle_results)

    pooled = pooled_configurations(alignments, reference_gt_sizes(args.reference))

    selections: list[dict] = []
    for label, first_name, second_name in MODEL_SELECTIONS:
        if first_name not in configs or second_name not in configs:
            continue
        selections.append({
            "label": label,
            "note": "diagnostic only, not part of the multiplicity-corrected family",
            **paired_test(configs[first_name], configs[second_name]),
        })

    print_report(results, configs, pooled, selections, oracle_results)
    args.out.write_text(
        json.dumps({
            "configurations": configs,
            "comparisons": results,
            "comparisons_oracle_family": oracle_results,
            "pooled_f1": pooled,
            "model_selections": selections,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
