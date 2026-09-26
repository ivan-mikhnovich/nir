"""Check taxonomy integrity and alignment consistency across the result cache.

Runs the checks from `src.consistency` over the seven OAEI Conference
ontologies and over every cached matcher result, then measures two things:

* what happens to precision, recall and F1 when the flagged matches are
  dropped, so the consistency stage is a filter whose usefulness is measured;
* how F1 depends on the confidence threshold applied to the raw predictions,
  which is the post-processing the matchers leave to the caller.  The
  threshold is also selected leave-one-pair-out: it is picked on the other
  pairs and scored on the held-out one, because picking it on the full set
  would measure the evaluation set rather than the method.

Usage:
    uv run python -m src.runners.consistency
    uv run python -m src.runners.consistency --taxonomy-only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from src.cache import load_all_cached
from src.consistency import (
    CYCLE,
    DISJOINT_ANCESTOR,
    DUPLICATE_LABEL,
    ERROR_KINDS,
    INFO_KINDS,
    UNREACHABLE,
    WARNING_KINDS,
    AlignmentReport,
    filter_alignment,
    flag_quality,
    check_alignment,
    check_taxonomy,
)
from src.data_loader import find_gt, load_alignments, load_taxonomies
from src.metrics import evaluate_1to1
from src.taxonomy import Alignment, TaxonMatch, TaxonNode, Taxonomy

DEFAULT_DATA = Path("data/processed/oaei")
DEFAULT_OUT = Path("results/consistency")

# Filter variants: which flag kinds each one drops.
FILTERS: dict[str, tuple[str, ...]] = {
    "errors": ERROR_KINDS,
    "errors+warnings": ERROR_KINDS + WARNING_KINDS,
    "low-confidence": INFO_KINDS,
}

# Confidence thresholds swept over the raw predictions.
THRESHOLDS: tuple[float, ...] = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
)


def build_alignment(record: dict) -> Alignment | None:
    """Rebuild the predicted alignment from a cached result record."""
    pairs = record.get("match_pairs")
    if not pairs:
        return None
    return Alignment(
        source=record["source"],
        target=record["target"],
        matches=[
            TaxonMatch(
                source_id=p["source_id"],
                target_id=p["target_id"],
                confidence=p.get("confidence", 1.0),
            )
            for p in pairs
        ],
    )


def above(alignment: Alignment, threshold: float) -> Alignment:
    """Keep only the matches whose confidence reaches the threshold."""
    return Alignment(
        source=alignment.source,
        target=alignment.target,
        matches=[m for m in alignment.matches if m.confidence >= threshold],
    )


def sweep_key(threshold: float) -> str:
    """Key of one threshold inside a sweep mapping."""
    return f"{threshold:.2f}"


def check_all_taxonomies(taxonomies: dict) -> dict[str, dict]:
    """Run the internal checks over every loaded taxonomy."""
    out: dict[str, dict] = {}
    for name, tax in sorted(taxonomies.items()):
        report = check_taxonomy(tax)
        out[name] = {
            "node_count": report.node_count,
            "root_count": report.root_count,
            "max_depth": report.max_depth,
            "counts": report.counts(),
            "violations": [
                {"kind": v.kind, "severity": v.severity, "detail": v.detail}
                for v in report.violations
            ],
        }
    return out


def check_all_alignments(
    taxonomies: dict,
    alignments_raw: list[dict],
    confidence_threshold: float,
) -> tuple[dict[str, dict], int]:
    """Run the alignment checks over every cached record.

    Returns the per-record detail keyed by approach, plus the number of cached
    records that carried no match list (nothing to check).
    """
    out: dict[str, dict] = {}
    skipped = 0
    for approach, pairs in sorted(load_all_cached().items()):
        for pair, entry in sorted(pairs.items()):
            records: dict = {None: entry} if isinstance(entry.get("f1"), (int, float)) else entry
            for mode, record in sorted(records.items(), key=lambda kv: kv[0] or ""):
                alignment = build_alignment(record)
                gt = (
                    find_gt(record["source"], record["target"], alignments_raw)
                    if alignment is not None
                    else None
                )
                if alignment is None or gt is None:
                    skipped += 1
                    continue
                report: AlignmentReport = check_alignment(
                    taxonomies[record["source"]],
                    taxonomies[record["target"]],
                    alignment,
                    approach=approach,
                    pair=pair,
                    mode=mode,
                    confidence_threshold=confidence_threshold,
                )
                baseline = evaluate_1to1(alignment, gt)
                detail: dict = {
                    "pair": pair,
                    "mode": mode,
                    "matches": alignment.match_count,
                    "f1_cached": record.get("f1"),
                    "baseline": {
                        "precision": baseline.precision,
                        "recall": baseline.recall,
                        "f1": baseline.f1,
                    },
                    "violations": report.counts(),
                    "flagged": report.flagged_kinds(),
                    "filters": {},
                    "threshold_sweep": {},
                    "flag_quality": flag_quality(alignment, report, gt),
                }
                for name, kinds in FILTERS.items():
                    kept_alignment = filter_alignment(alignment, report, kinds)
                    kept = evaluate_1to1(kept_alignment, gt)
                    detail["filters"][name] = {
                        "matches": kept_alignment.match_count,
                        "precision": kept.precision,
                        "recall": kept.recall,
                        "f1": kept.f1,
                        "delta_f1": kept.f1 - baseline.f1,
                    }
                for t in THRESHOLDS:
                    swept = above(alignment, t)
                    metrics = evaluate_1to1(swept, gt)
                    detail["threshold_sweep"][sweep_key(t)] = {
                        "matches": swept.match_count,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1": metrics.f1,
                    }
                out.setdefault(approach, {})[f"{pair}|{mode or '-'}"] = detail
    return out, skipped


def loo_threshold(values: list[dict]) -> list[float]:
    """Pick the best threshold on the other records for every record.

    Returns the threshold chosen for each record in `values`, in the same
    order, so the caller can score the held-out record with it.
    """
    chosen: list[float] = []
    for i, value in enumerate(values):
        others = values[:i] + values[i + 1:]
        if not others:
            chosen.append(0.0)
            continue
        best = max(
            THRESHOLDS,
            key=lambda t: mean(o["threshold_sweep"][sweep_key(t)]["f1"] for o in others),
        )
        chosen.append(best)
    return chosen


def aggregate(alignments: dict[str, dict]) -> dict[str, dict]:
    """Average the per-record detail into one summary row per approach."""
    summary: dict[str, dict] = {}
    for approach, records in sorted(alignments.items()):
        values = list(records.values())
        if not values:
            continue
        kind_totals: dict[str, int] = {}
        for v in values:
            for kind, count in v["violations"].items():
                kind_totals[kind] = kind_totals.get(kind, 0) + count
        sweep_f1 = {
            sweep_key(t): mean(v["threshold_sweep"][sweep_key(t)]["f1"] for v in values)
            for t in THRESHOLDS
        }
        best_threshold = max(THRESHOLDS, key=lambda t: sweep_f1[sweep_key(t)])
        loo = loo_threshold(values)
        loo_scores = [v["threshold_sweep"][sweep_key(t)] for v, t in zip(values, loo)]
        for value, threshold, score in zip(values, loo, loo_scores):
            value["loo"] = {"threshold": threshold, **score}
        summary[approach] = {
            "records": len(values),
            "matches": sum(v["matches"] for v in values),
            "f1_mismatches": sum(
                1 for v in values
                if v["f1_cached"] is not None
                and abs(v["f1_cached"] - v["baseline"]["f1"]) > 1e-6
            ),
            "violation_totals": kind_totals,
            "flagged_totals": {
                kind: sum(v["flagged"].get(kind, 0) for v in values)
                for kind in sorted({k for v in values for k in v["flagged"]})
            },
            "baseline_f1": mean(v["baseline"]["f1"] for v in values),
            "baseline_precision": mean(v["baseline"]["precision"] for v in values),
            "baseline_recall": mean(v["baseline"]["recall"] for v in values),
            "filter_f1": {
                name: mean(v["filters"][name]["f1"] for v in values) for name in FILTERS
            },
            "filter_delta_f1": {
                name: mean(v["filters"][name]["delta_f1"] for v in values) for name in FILTERS
            },
            "filter_wins": {
                name: sum(1 for v in values if v["filters"][name]["delta_f1"] > 1e-9)
                for name in FILTERS
            },
            "filter_losses": {
                name: sum(1 for v in values if v["filters"][name]["delta_f1"] < -1e-9)
                for name in FILTERS
            },
            "threshold_sweep_f1": sweep_f1,
            "best_threshold": best_threshold,
            "best_threshold_f1": sweep_f1[sweep_key(best_threshold)],
            "loo_threshold_f1": mean(s["f1"] for s in loo_scores),
            "loo_threshold_precision": mean(s["precision"] for s in loo_scores),
            "loo_threshold_recall": mean(s["recall"] for s in loo_scores),
            "loo_threshold_choices": dict(Counter(loo)),
            "flag_precision": mean(v["flag_quality"]["error_precision"] for v in values),
            "flag_recall": mean(v["flag_quality"]["error_recall"] for v in values),
        }
    return summary


def aggregate_by_pair(alignments: dict[str, dict]) -> dict[str, dict]:
    """Best mode per pair, comparable with the published comparison table.

    Every approach picks its best mode per pair by the F1 of the leave-one-out
    thresholded prediction, then the result is averaged over the pairs.  Call
    this after `aggregate`, which annotates each record with its LOO score.
    """
    out: dict[str, dict] = {}
    for approach, records in sorted(alignments.items()):
        by_pair: dict[str, list[dict]] = {}
        for value in records.values():
            by_pair.setdefault(value["pair"], []).append(value)
        if not by_pair:
            continue
        best = [max(vs, key=lambda v: v["loo"]["f1"]) for vs in by_pair.values()]
        out[approach] = {
            "pairs": len(by_pair),
            "baseline_f1": mean(max(v["baseline"]["f1"] for v in vs) for vs in by_pair.values()),
            "loo_f1": mean(v["loo"]["f1"] for v in best),
            "loo_precision": mean(v["loo"]["precision"] for v in best),
            "loo_recall": mean(v["loo"]["recall"] for v in best),
            "thresholds": dict(Counter(v["loo"]["threshold"] for v in best)),
        }
    return out


def selftest() -> bool:
    """Verify that every taxonomy check fires on a deliberately broken taxonomy.

    A checker that reports nothing is worthless unless it can report something,
    so the self-test builds a taxonomy with a cycle, a duplicated label, a
    disjointness between a class and its own child, and a component without a
    root, then requires all four kinds to be found.
    """
    tax = Taxonomy(name="selftest", namespace="", root_id="a")
    tax.nodes = {
        "a": TaxonNode(id="a", name="A", parents=["b"], children=["c"], disjoint_with=["c"]),
        "b": TaxonNode(id="b", name="A", parents=["a"], children=[]),
        "c": TaxonNode(id="c", name="C", parents=["a"], children=[]),
    }
    found = check_taxonomy(tax).counts()
    expected = (CYCLE, DUPLICATE_LABEL, DISJOINT_ANCESTOR, UNREACHABLE)
    for kind in expected:
        print(f"  {kind:<20}{'found' if kind in found else 'MISSING'}")
    ok = all(kind in found for kind in expected)
    print(f"Self-test: {'passed' if ok else 'FAILED'}")
    return ok


def print_taxonomies(reports: dict[str, dict]) -> None:
    """Print the taxonomy-internal check results as a table."""
    print("Taxonomy integrity (7 OAEI ontologies)")
    print(f"{'ontology':<12}{'nodes':>6}{'roots':>7}{'depth':>7}  violations")
    for name, rep in sorted(reports.items(), key=lambda kv: -kv[1]["node_count"]):
        counts = rep["counts"]
        text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
        print(f"{name:<12}{rep['node_count']:>6}{rep['root_count']:>7}"
              f"{rep['max_depth']:>7}  {text}")


def print_alignments(summary: dict[str, dict]) -> None:
    """Print the alignment check and filter results as tables."""
    print("\nAlignment consistency and filter effect (cached records)")
    header = (f"{'approach':<22}{'recs':>5}{'matches':>8}{'F1 base':>9}"
              f"{'F1 err':>8}{'F1 err+warn':>13}{'F1 lowconf':>12}"
              f"{'Δerr':>8}{'flag P':>8}{'flag R':>8}")
    print(header)
    print("-" * len(header))
    for approach, s in summary.items():
        print(f"{approach:<22}{s['records']:>5}{s['matches']:>8}"
              f"{s['baseline_f1']:>9.3f}"
              f"{s['filter_f1']['errors']:>8.3f}"
              f"{s['filter_f1']['errors+warnings']:>13.3f}"
              f"{s['filter_f1']['low-confidence']:>12.3f}"
              f"{s['filter_delta_f1']['errors']:>+8.3f}"
              f"{s['flag_precision']:>8.2f}{s['flag_recall']:>8.2f}")

    print("\nViolations per kind (summed over records)")
    for approach, s in summary.items():
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(s["violation_totals"].items()))
        print(f"  {approach:<22}{kinds or 'none'}")

    print("\nFilter wins/losses over records")
    for approach, s in summary.items():
        parts = ", ".join(
            f"{name}: {s['filter_wins'][name]}/{s['filter_losses'][name]}"
            for name in FILTERS
        )
        print(f"  {approach:<22}{parts} (wins/losses)")

    print("\nMean F1 versus confidence threshold (raw predictions)")
    header = (f"{'approach':<22}" + "".join(f"{t:>7.2f}" for t in THRESHOLDS)
              + f"{'best':>8}{'LOO':>8}{'P@LOO':>8}{'R@LOO':>8}")
    print(header)
    print("-" * len(header))
    for approach, s in summary.items():
        row = "".join(f"{s['threshold_sweep_f1'][sweep_key(t)]:>7.3f}" for t in THRESHOLDS)
        print(f"{approach:<22}{row}"
              f"{s['best_threshold']:>8.2f}"
              f"{s['loo_threshold_f1']:>8.3f}"
              f"{s['loo_threshold_precision']:>8.3f}"
              f"{s['loo_threshold_recall']:>8.3f}")


def print_by_pair(by_pair: dict[str, dict]) -> None:
    """Print the best-mode-per-pair table with leave-one-out thresholds."""
    print("\nBest mode per pair (comparable with the published comparison table)")
    header = (f"{'approach':<22}{'pairs':>6}{'F1 raw':>9}{'F1 LOO':>9}"
              f"{'P@LOO':>8}{'R@LOO':>8}  thresholds")
    print(header)
    print("-" * len(header))
    for approach, s in by_pair.items():
        choices = ", ".join(f"{t:.2f}×{n}" for t, n in sorted(s["thresholds"].items()))
        print(f"{approach:<22}{s['pairs']:>6}{s['baseline_f1']:>9.3f}{s['loo_f1']:>9.3f}"
              f"{s['loo_precision']:>8.3f}{s['loo_recall']:>8.3f}  {choices}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check taxonomy and alignment consistency.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Confidence below which a match is flagged for review.")
    parser.add_argument("--taxonomy-only", action="store_true",
                        help="Skip the alignment checks over the result cache.")
    parser.add_argument("--selftest", action="store_true",
                        help="Verify that the checks fire on a deliberately broken taxonomy.")
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    taxonomies = load_taxonomies(args.data_dir)
    alignments_raw = load_alignments(args.data_dir / "alignments.json")

    taxonomy_reports = check_all_taxonomies(taxonomies)
    print_taxonomies(taxonomy_reports)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "taxonomies.json").write_text(
        json.dumps(taxonomy_reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.taxonomy_only:
        print(f"\nWrote {args.out / 'taxonomies.json'}")
        return

    alignments, skipped = check_all_alignments(taxonomies, alignments_raw, args.threshold)
    summary = aggregate(alignments)
    print_alignments(summary)
    by_pair = aggregate_by_pair(alignments)
    print_by_pair(by_pair)
    mismatches = sum(s["f1_mismatches"] for s in summary.values())
    print(f"\nCached F1 disagrees with the recomputed value in {mismatches} records.")
    if skipped:
        print(f"Skipped {skipped} cached records without a match list.")

    (args.out / "alignments.json").write_text(
        json.dumps(alignments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "by_pair.json").write_text(
        json.dumps(by_pair, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {args.out}/taxonomies.json, alignments.json, summary.json, by_pair.json")


if __name__ == "__main__":
    main()
