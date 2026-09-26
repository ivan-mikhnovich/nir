"""Check taxonomy integrity and alignment consistency across the result cache.

Runs the checks from `src.consistency` over the seven OAEI Conference
ontologies and over every cached matcher result, then measures two things:

* what happens to precision, recall and F1 when the flagged matches are
  dropped, so the consistency stage is a filter whose usefulness is measured.
  The flag is scored against the base error rate of the alignment it runs on,
  because a flag that marked every match would score exactly that rate (F3,
  review/review-consistency.md), and it is pooled over all matches of an
  approach rather than averaged over records, so that records without flags
  do not contribute a structural zero (F8);
* how F1 depends on the confidence threshold applied to the raw predictions,
  which is the post-processing the matchers leave to the caller.  This sweep
  is where low confidence is reported: cutting matches below a threshold is
  not a consistency check (`src.consistency`).  The threshold is also selected
  leave-one-pair-out: it is picked on the other pairs — never on another mode
  of the held-out pair — and scored on the held-out one, because picking it on
  the full set would measure the evaluation set rather than the method.

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

from src.cache import canonical_directions, load_all_cached, record_alignment
from src.consistency import (
    CARDINALITY,
    CYCLE,
    DANGLING_PARENT,
    DISJOINT_ANCESTOR,
    DUPLICATE_LABEL,
    ERROR_KINDS,
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
from src.taxonomy import Alignment, TaxonNode, Taxonomy

DEFAULT_DATA = Path("data/processed/oaei")
DEFAULT_OUT = Path("results/consistency")

# Filter variants: which flag kinds each one drops.  Low confidence is not a
# variant: it is the threshold sweep below, and it is numerically the same
# post-processing (`confidence < threshold`), so listing it here would present
# one number twice (F2, review/review-consistency.md).
FILTERS: dict[str, tuple[str, ...]] = {
    "errors": ERROR_KINDS,
    "errors+warnings": ERROR_KINDS + WARNING_KINDS,
}

# Confidence thresholds swept over the raw predictions.
THRESHOLDS: tuple[float, ...] = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
)


def cached_records(entry: dict) -> dict:
    """Return the records of one cached pair, keyed by mode (None for a plain record)."""
    return {None: entry} if isinstance(entry.get("f1"), (int, float)) else entry


def print_orientation_audit(
    cached: dict[str, dict],
    canonical: dict[str, tuple[str, str]],
) -> None:
    """Report cached records stored in the reverse of the canonical direction.

    The canonical direction comes from the reference alignments (contract C6),
    and every check runs in the direction a record was computed in, so a flipped
    file would be checked as the reverse pair.  The audit is printed rather than
    raised: the cache owner reorients the files
    (`src.runners.embedding --reorient`), and this run is a validation pass
    (findings 2.9/2.10).
    """
    flipped: list[tuple[str, str, str, tuple[str, str]]] = []
    for approach, pairs in sorted(cached.items()):
        for pair, entry in sorted(pairs.items()):
            for record in cached_records(entry).values():
                expected = canonical.get(pair)
                # Metric-only records (GNN, GNN baseline) carry no direction at
                # all: nothing to check here, and nothing to check in them
                # either (review F9 — the cache stores no match list).
                if expected is None or "source" not in record or "target" not in record:
                    continue
                if (record["source"], record["target"]) == expected:
                    continue
                flipped.append((approach, record["source"], record["target"], expected))
    if not flipped:
        print("Cache orientation: every record matches the canonical direction.")
        return
    print(f"Cache orientation: {len(flipped)} record(s) are stored in the reverse of the "
          f"canonical direction,")
    print("  so those pairs are checked as the reverse pair:")
    for approach, source, target, expected in flipped:
        print(f"  {approach:<24}{source} → {target} "
              f"(canonical {expected[0]} → {expected[1]})")


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
    cached: dict[str, dict],
    alignments_raw: list[dict],
    confidence_threshold: float,
) -> tuple[dict[str, dict], int]:
    """Run the alignment checks over every cached record.

    Returns the per-record detail keyed by approach, plus the number of cached
    records that carried no match list (nothing to check).  `confidence_threshold`
    only labels the low-confidence post-processing reported alongside the
    sweep; the consistency checks themselves never look at confidence.
    """
    out: dict[str, dict] = {}
    skipped = 0
    for approach, pairs in sorted(cached.items()):
        for pair, entry in sorted(pairs.items()):
            records: dict = cached_records(entry)
            for mode, record in sorted(records.items(), key=lambda kv: kv[0] or ""):
                alignment = record_alignment(record)
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
                )
                baseline = evaluate_1to1(alignment, gt)
                below_threshold = [
                    m.confidence for m in alignment.matches
                    if m.confidence < confidence_threshold
                ]
                detail: dict = {
                    "pair": pair,
                    "mode": mode,
                    "matches": alignment.match_count,
                    "gt_matches": gt.match_count,
                    "f1_cached": record.get("f1"),
                    "baseline": {
                        "precision": baseline.precision,
                        "recall": baseline.recall,
                        "f1": baseline.f1,
                    },
                    "violations": report.counts(),
                    "flagged": report.flagged_kinds(),
                    "cardinality_groups": report.cardinality_groups,
                    # The low-confidence cut-off is the threshold sweep, stored
                    # here only so the report can quote the cut's own numbers:
                    # matches dropped, and the highest confidence it drops (a
                    # match of 0.69999 must not be printed as "0.70 is below
                    # the 0.70 threshold", F11).
                    "low_confidence": {
                        "threshold": confidence_threshold,
                        "dropped": len(below_threshold),
                        "highest_dropped_confidence": (
                            max(below_threshold) if below_threshold else None
                        ),
                    },
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
                        # The denominator `evaluate_1to1` uses for precision: it
                        # keeps one match per source, so pooling precision over
                        # records must weight by sources, not by matches.
                        "sources": len({m.source_id for m in swept.matches}),
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1": metrics.f1,
                    }
                out.setdefault(approach, {})[f"{pair}|{mode or '-'}"] = detail
    return out, skipped


def loo_threshold(values: list[dict]) -> list[float]:
    """Pick the best threshold on the other pairs for every record.

    The pool excludes every other mode of the record's own pair: a threshold
    picked on another mode of the same pair was selected on the pair it is
    then scored on, which is exactly what the leave-one-out scheme is meant to
    rule out (F10, review/review-consistency.md).  Returns the threshold
    chosen for each record in `values`, in the same order, so the caller can
    score the held-out record with it.
    """
    chosen: list[float] = []
    for i, value in enumerate(values):
        others = [
            o for j, o in enumerate(values)
            if j != i and o["pair"] != value["pair"]
        ]
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
    """Average the per-record detail into one summary row per approach.

    Flag precision and recall are pooled over every match of the approach:
    averaging per-record ratios would count a record without flags as a
    precision of exactly zero, which understates the flag 2–5× (F8,
    review/review-consistency.md).  The base error rate of the alignment —
    what a flag that marked everything would score — is reported next to
    them, with the lift over it (F3).

    The threshold sweep is reported both as the mean over records and pooled
    over matches: pooled precision is Σ true positives / Σ predicted, the
    number the note quotes for the unthresholded run (REVIEW.md 1.8).
    """
    summary: dict[str, dict] = {}
    for approach, records in sorted(alignments.items()):
        values = list(records.values())
        if not values:
            continue
        kind_totals: dict[str, int] = {}
        for v in values:
            for kind, count in v["violations"].items():
                kind_totals[kind] = kind_totals.get(kind, 0) + count
        keys = [sweep_key(t) for t in THRESHOLDS]
        sweep_f1 = {k: mean(v["threshold_sweep"][k]["f1"] for v in values) for k in keys}
        sweep_precision = {
            k: mean(v["threshold_sweep"][k]["precision"] for v in values) for k in keys
        }
        sweep_recall = {k: mean(v["threshold_sweep"][k]["recall"] for v in values) for k in keys}
        pooled_precision: dict[str, float] = {}
        pooled_recall: dict[str, float] = {}
        sweep_matches: dict[str, int] = {}
        for k in keys:
            hits = sum(v["threshold_sweep"][k]["precision"] * v["threshold_sweep"][k]["sources"]
                       for v in values)
            gold = sum(v["threshold_sweep"][k]["recall"] * v["gt_matches"] for v in values)
            predicted = sum(v["threshold_sweep"][k]["matches"] for v in values)
            evaluated = sum(v["threshold_sweep"][k]["sources"] for v in values)
            sweep_matches[k] = predicted
            pooled_precision[k] = hits / evaluated if evaluated else 0.0
            pooled_recall[k] = hits / gold if gold else 0.0
        best_threshold = max(THRESHOLDS, key=lambda t: sweep_f1[sweep_key(t)])
        loo = loo_threshold(values)
        loo_scores = [v["threshold_sweep"][sweep_key(t)] for v, t in zip(values, loo)]
        for value, threshold, score in zip(values, loo, loo_scores):
            value["loo"] = {"threshold": threshold, **score}
        flag_matches = sum(v["flag_quality"]["matches"] for v in values)
        flag_flagged = sum(v["flag_quality"]["flagged"] for v in values)
        flag_errors = sum(v["flag_quality"]["errors"] for v in values)
        flag_flagged_errors = sum(v["flag_quality"]["flagged_errors"] for v in values)
        flagged_records = [v for v in values if v["flag_quality"]["flagged"]]
        flag_precision = flag_flagged_errors / flag_flagged if flag_flagged else 0.0
        base_error_rate = flag_errors / flag_matches if flag_matches else 0.0
        highest_dropped = [
            v["low_confidence"]["highest_dropped_confidence"] for v in values
            if v["low_confidence"]["highest_dropped_confidence"] is not None
        ]
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
            "cardinality_groups": {
                side: sum(v["cardinality_groups"][side] for v in values)
                for side in ("source", "target")
            },
            "low_confidence": {
                "threshold": values[0]["low_confidence"]["threshold"],
                "dropped": sum(v["low_confidence"]["dropped"] for v in values),
                "highest_dropped_confidence": max(highest_dropped) if highest_dropped else None,
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
            "threshold_sweep_precision": sweep_precision,
            "threshold_sweep_recall": sweep_recall,
            "threshold_sweep_matches": sweep_matches,
            "threshold_sweep_pooled_precision": pooled_precision,
            "threshold_sweep_pooled_recall": pooled_recall,
            "best_threshold": best_threshold,
            "best_threshold_f1": sweep_f1[sweep_key(best_threshold)],
            "loo_threshold_f1": mean(s["f1"] for s in loo_scores),
            "loo_threshold_precision": mean(s["precision"] for s in loo_scores),
            "loo_threshold_recall": mean(s["recall"] for s in loo_scores),
            "loo_threshold_choices": dict(Counter(loo)),
            # Pooled over every match of the approach (F8).
            "flag_precision_pooled": flag_precision,
            "flag_recall_pooled": (
                flag_flagged_errors / flag_errors if flag_errors else 0.0
            ),
            "flag_base_error_rate": base_error_rate,
            "flag_precision_lift": flag_precision - base_error_rate,
            "flag_coverage": flag_flagged / flag_matches if flag_matches else 0.0,
            # The same ratio restricted to the records that actually flagged,
            # for a reader who wants to see the effect of the structural zeros.
            "flag_records_with_flags": len(flagged_records),
            "flag_precision_flagged_records": (
                mean(v["flag_quality"]["error_precision"] for v in flagged_records)
                if flagged_records else 0.0
            ),
        }
    return summary


def aggregate_by_pair(alignments: dict[str, dict]) -> dict[str, dict]:
    """Best mode per pair, comparable with the published comparison table.

    Every approach picks its best mode per pair by the F1 of the leave-one-out
    thresholded prediction, then the result is averaged over the pairs.  That
    per-pair maximum is an oracle choice over modes and is labelled as such;
    `thresholds` records which cut-off the leave-one-out scheme picked.  Call
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
            "mode_selection": "per-pair maximum over modes (oracle)",
            "baseline_f1": mean(max(v["baseline"]["f1"] for v in vs) for vs in by_pair.values()),
            "loo_f1": mean(v["loo"]["f1"] for v in best),
            "loo_precision": mean(v["loo"]["precision"] for v in best),
            "loo_recall": mean(v["loo"]["recall"] for v in best),
            "thresholds": dict(Counter(v["loo"]["threshold"] for v in best)),
        }
    return out


def _check_counts(title: str, report, expected: dict[str, int]) -> bool:
    """Print the per-kind counts of one self-test case against its expectation."""
    found = report.counts()
    ok = found == expected
    print(f"  {title}: {'ok' if ok else 'MISMATCH'}")
    for kind in sorted(set(expected) | set(found)):
        want, got = expected.get(kind, 0), found.get(kind, 0)
        mark = "" if want == got else "   <-- expected"
        print(f"    {kind:<20}expected {want:<4}found {got}{mark}")
    return ok


def selftest() -> bool:
    """Verify that every taxonomy check fires on a deliberately broken taxonomy.

    A checker that reports nothing is worthless unless it can report something,
    so the self-test compares the per-kind COUNTS against their expectation —
    "the kind appeared" would still pass when one defect fires several checks.
    Three cases are run:

    * a broken taxonomy with one defect of every kind, including a class whose
      declared parent is absent (`dangling-parent`);
    * a broken link on its own, which must fire `dangling-parent` and leave the
      class unreachable — the trigger `unreachable-class` did not have before
      (F1, review/review-consistency.md);
    * a healthy taxonomy, the negative control: its report must be empty.
    """
    broken = Taxonomy(name="selftest", namespace="", root_id="a")
    broken.nodes = {
        "a": TaxonNode(id="a", name="A", parents=["b"], children=["c"], disjoint_with=["c"]),
        "b": TaxonNode(id="b", name="A", parents=["a"], children=[]),
        "c": TaxonNode(id="c", name="C", parents=["a", "absent"], children=[]),
    }
    broken_expected = {
        CYCLE: 1,
        DUPLICATE_LABEL: 1,
        DISJOINT_ANCESTOR: 1,
        DANGLING_PARENT: 1,
        UNREACHABLE: 1,
    }
    dangling = Taxonomy(name="selftest-dangling", namespace="", root_id="r")
    dangling.nodes = {
        "r": TaxonNode(id="r", name="Root", parents=[], children=[]),
        "x": TaxonNode(id="x", name="X", parents=["does_not_exist"], children=[]),
    }
    dangling_expected = {DANGLING_PARENT: 1, UNREACHABLE: 1}
    healthy = Taxonomy(name="selftest-healthy", namespace="", root_id="r")
    healthy.nodes = {
        "r": TaxonNode(id="r", name="Root", parents=[], children=["c1", "c2"]),
        "c1": TaxonNode(id="c1", name="Child One", parents=["r"], children=[]),
        "c2": TaxonNode(id="c2", name="Child Two", parents=["r"], children=[]),
    }
    print("Self-test: broken taxonomy, one defect of every kind")
    ok = _check_counts("broken", check_taxonomy(broken), broken_expected)
    print("\nSelf-test: a lone dangling parent (no cycle)")
    ok &= _check_counts("dangling", check_taxonomy(dangling), dangling_expected)
    print("\nSelf-test: healthy taxonomy (negative control)")
    ok &= _check_counts("healthy", check_taxonomy(healthy), {})
    print(f"\nSelf-test: {'passed' if ok else 'FAILED'}")
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
    print("\nAlignment consistency checks and filter effect (cached records)")
    header = (f"{'approach':<22}{'recs':>5}{'matches':>8}{'F1 base':>9}"
              f"{'F1 err':>8}{'F1 err+warn':>13}"
              f"{'Δerr':>8}{'flag P':>8}{'flag R':>8}{'base':>7}{'lift':>7}{'cover':>7}")
    print(header)
    print("-" * len(header))
    for approach, s in summary.items():
        print(f"{approach:<22}{s['records']:>5}{s['matches']:>8}"
              f"{s['baseline_f1']:>9.3f}"
              f"{s['filter_f1']['errors']:>8.3f}"
              f"{s['filter_f1']['errors+warnings']:>13.3f}"
              f"{s['filter_delta_f1']['errors']:>+8.3f}"
              f"{s['flag_precision_pooled']:>8.3f}{s['flag_recall_pooled']:>8.3f}"
              f"{s['flag_base_error_rate']:>7.3f}{s['flag_precision_lift']:>+7.3f}"
              f"{s['flag_coverage']:>7.3f}")
    print("  flag P/R are pooled over every match of the approach (F8); base is the share of")
    print("  wrong matches in the same alignments, so a flag that marked everything would score")
    print("  exactly base, and `lift` is the difference (F3)")

    print("\nViolations per kind: cardinality counts overloaded GROUPS, the others events;")
    print("the match count shows how many matches each kind flags (F5)")
    for approach, s in summary.items():
        parts = []
        for kind in sorted(set(s["violation_totals"]) | set(s["flagged_totals"])):
            events = s["violation_totals"].get(kind, 0)
            matches = s["flagged_totals"].get(kind, 0)
            if kind == CARDINALITY:
                sides = s["cardinality_groups"]
                parts.append(f"{kind}={events} groups (src {sides['source']}, "
                             f"tgt {sides['target']}) / {matches} matches")
            else:
                parts.append(f"{kind}={events} / {matches} matches")
        print(f"  {approach:<22}{'; '.join(parts) or 'none'}")
    source_groups = sum(s["cardinality_groups"]["source"] for s in summary.values())
    target_groups = sum(s["cardinality_groups"]["target"] for s in summary.values())
    if source_groups:
        print(f"\nCardinality: {source_groups} overloaded source groups, "
              f"{target_groups} overloaded target groups")
    else:
        print(f"\nCardinality: all {target_groups} overloaded groups are target-side — every "
              f"matcher emits at most one match per source, so the source side is injective by")
        print("construction and the 1:1 column is zero there by design, not by quality")

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

    dropped = sum(s["low_confidence"]["dropped"] for s in summary.values())
    thresholds = {s["low_confidence"]["threshold"] for s in summary.values()}
    highest = [
        s["low_confidence"]["highest_dropped_confidence"] for s in summary.values()
        if s["low_confidence"]["highest_dropped_confidence"] is not None
    ]
    cut = f"{sorted(thresholds)[0]:.2f}" if thresholds else "the cut-off"
    print(f"\nLow confidence is this sweep, not a consistency check (F2).  The strict cut at "
          f"{cut} drops {dropped} matches;")
    if highest:
        print(f"the highest-confidence match it drops is {max(highest):.4f} — printed to four "
              f"decimals, because rounding it to two")
        print('would read as "confidence 0.70 is below the 0.70 threshold" (F11).')
    print("Pooled precision and recall at every threshold are in summary.json, so the "
          "unthresholded")
    print("run can be quoted without recomputing it by hand.")
    for approach, s in summary.items():
        key = sweep_key(s["low_confidence"]["threshold"])
        print(f"  {approach:<22}sweep[{key}] pooled P={s['threshold_sweep_pooled_precision'][key]:.4f} "
              f"R={s['threshold_sweep_pooled_recall'][key]:.4f} "
              f"kept={s['threshold_sweep_matches'][key]}/{s['matches']} "
              f"dropped={s['low_confidence']['dropped']}")


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
                        help="Confidence of the low-confidence cut reported alongside the "
                             "sweep (post-processing, not a consistency check).")
    parser.add_argument("--taxonomy-only", action="store_true",
                        help="Skip the alignment checks over the result cache.")
    parser.add_argument("--selftest", action="store_true",
                        help="Verify the per-kind counts and the negative control of the "
                             "taxonomy checks.")
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

    cached = load_all_cached()
    canonical = canonical_directions(alignments_raw)
    print("\nCache orientation (reference-alignment direction, contract C6)")
    print_orientation_audit(cached, canonical)

    alignments, skipped = check_all_alignments(taxonomies, cached, alignments_raw, args.threshold)
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
