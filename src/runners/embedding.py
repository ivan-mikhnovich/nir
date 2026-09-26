"""Run embedding-based matching experiments on OAEI + synthetic data.

Iterates over multiple sentence-transformer models for comparison, provides the
description-mode component ablation and the postprocessing sweep over the 21
OAEI pairs (findings 1.2/2.2/3.6), and can re-orient the cached records into the
canonical pair direction (findings 2.9/3.11, contract C6).

Usage:
    uv run python -m src.runners.embedding                      # model sweep + synthetic + HITL
    uv run python -m src.runners.embedding --modes              # ablation, MiniLM, canonical direction
    uv run python -m src.runners.embedding --modes --model all --direction sorted
    uv run python -m src.runners.embedding --reorient           # rewrite flipped records (no API)
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from src.data_loader import find_gt, load_alignments, load_taxonomies, load_taxonomy
from src.matchers.embedding import DESCRIPTION_MODES, EmbeddingMatcher
from src.metrics import evaluate_1to1, MatchMetrics


# Models to evaluate (short name, full HF identifier).
MODELS: list[tuple[str, str]] = [
    ("LaBSE", "sentence-transformers/LaBSE"),
    ("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
    ("ruRoberta-large", "ai-forever/ruRoberta-large"),
    ("rubert-base", "DeepPavlov/rubert-base-cased"),
]

# Description modes of the component ablation, in the order of the table.
ABLATION_MODES: tuple[str, ...] = (
    "name_only",
    "name_path",
    "name+attributes",
    "name+disjoint+comment",
    "full",
)

# Postprocessing sweep on `name_only`: (rule, threshold).
POSTPROCESS_SWEEP: tuple[tuple[str, float | None], ...] = (
    ("argmax", None),
    ("dedupe_by_target", None),
    ("greedy_injective", None),
    ("greedy_injective", 0.5),
    ("greedy_injective", 0.7),
)


def make_gt_alignment(al_data: dict):
    from src.taxonomy import Alignment, TaxonMatch
    return Alignment(
        source=al_data["source"],
        target=al_data["target"],
        matches=[
            TaxonMatch(
                source_id=m["source_id"],
                target_id=m["target_id"],
                confidence=m.get("confidence", 1.0),
            )
            for m in al_data["matches"]
        ],
    )


def run_experiment_1(
    matcher: EmbeddingMatcher,
    taxonomies: dict,
    alignments_raw: list[dict],
) -> list[MatchMetrics]:
    """OAEI Conference Track: all 21 real cross-domain pairs."""
    all_metrics: list[MatchMetrics] = []
    for al_data in alignments_raw:
        src_name = al_data["source"]
        tgt_name = al_data["target"]
        if src_name not in taxonomies or tgt_name not in taxonomies:
            continue
        src = taxonomies[src_name]
        tgt = taxonomies[tgt_name]
        gt = make_gt_alignment(al_data)
        pred, scores = matcher.match(src, tgt)
        metrics = evaluate_1to1(pred, gt)
        all_metrics.append(metrics)
    return all_metrics


def run_experiment_2(
    matcher: EmbeddingMatcher,
    synth_dir: Path,
) -> dict[str, dict[str, MatchMetrics]]:
    """Synthetic variations × description modes."""
    base_tax = load_taxonomy(synth_dir / "base_cmt.json")
    scenarios = ["exact", "synonym", "structural", "attribute"]
    modes = ["full", "name_only", "name_path"]
    suffix_map = {"exact": "V1", "synonym": "SYN", "structural": "STR", "attribute": "ATTR"}

    from src.taxonomy import Alignment, TaxonMatch

    results: dict[str, dict[str, MatchMetrics]] = {}
    for mode in modes:
        results[mode] = {}
        for scenario in scenarios:
            var_path = synth_dir / f"{scenario}_cmt.json"
            if not var_path.exists():
                continue
            var_tax = load_taxonomy(var_path)
            pred, scores = matcher.match(base_tax, var_tax, description_mode=mode)

            # Ground truth: 1-to-1 by matching ID suffix.
            expected_suffix = suffix_map[scenario]
            gt_matches = [
                TaxonMatch(source_id=nid, target_id=f"{nid}_{expected_suffix}")
                for nid in base_tax.nodes
                if f"{nid}_{expected_suffix}" in var_tax.nodes
            ]
            gt = Alignment(source=base_tax.name, target=var_tax.name, matches=gt_matches)
            results[mode][scenario] = evaluate_1to1(pred, gt)
    return results


# ── Description-mode ablation (findings 1.2, 2.2, 3.6) ────────────────


def pair_directions(
    alignments_raw: list[dict],
    direction: str = "canonical",
) -> list[tuple[str, str]]:
    """Return every OAEI pair in the requested direction.

    `canonical` is the direction of the reference-alignment loader (contract
    C6, the direction `results/consistency/alignments.json` is built in);
    `sorted` is the alphabetical order the pre-fix embedding caches used.
    """
    pairs: list[tuple[str, str]] = []
    for al_data in alignments_raw:
        src_name, tgt_name = al_data["source"], al_data["target"]
        if direction == "sorted":
            src_name, tgt_name = sorted([src_name, tgt_name])
        pairs.append((src_name, tgt_name))
    return pairs


def _pair_score(
    matcher: EmbeddingMatcher,
    taxonomies: dict,
    alignments_raw: list[dict],
    src_name: str,
    tgt_name: str,
    **match_kwargs,
) -> dict:
    """Score one pair: F1/precision/recall plus the number of predicted pairs."""
    gt = find_gt(src_name, tgt_name, alignments_raw)
    pred, _scores = matcher.match(taxonomies[src_name], taxonomies[tgt_name], **match_kwargs)
    metrics = evaluate_1to1(pred, gt)
    return {
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "predicted": metrics.total_predicted,
    }


def run_ablation(
    model_short: str,
    model_id: str,
    direction: str = "canonical",
    out: Path | None = None,
) -> dict:
    """Component ablation + postprocessing sweep over the 21 OAEI pairs.

    Writes `results/embedding-modes-<model>-<direction>.json` and prints the
    tables.  The direction matters for one pair of the 21 (`confOf↔conference`,
    finding 2.9), hence the two flavours of the file.
    """
    data_proc = Path("data/processed")
    taxonomies = load_taxonomies(data_proc / "oaei")
    alignments_raw = load_alignments(data_proc / "oaei" / "alignments.json")
    pairs = pair_directions(alignments_raw, direction)

    print(f"\n{'=' * 72}")
    print(f"Description-mode ablation — {model_short} ({model_id}), direction={direction}")
    print("=" * 72)

    matcher = EmbeddingMatcher(model_name=model_id, description_mode="name_only")

    report: dict = {
        "model": model_short,
        "model_id": model_id,
        "direction": direction,
        "direction_note": (
            "canonical = direction of the reference-alignment loader "
            "(src.cache.canonical_directions); sorted = alphabetical direction "
            "used by the pre-fix results/embedding-* caches."
        ),
        "pairs": [f"{s}↔{t}" for s, t in pairs],
        "modes": {},
        "postprocess": {},
    }

    mode_scores: dict[str, dict[str, dict]] = {}
    for mode in ABLATION_MODES:
        rows = {
            f"{src}↔{tgt}": _pair_score(
                matcher, taxonomies, alignments_raw, src, tgt, description_mode=mode
            )
            for src, tgt in pairs
        }
        mode_scores[mode] = rows
        report["modes"][mode] = {
            "components": list(DESCRIPTION_MODES[mode]),
            "mean_f1": mean(r["f1"] for r in rows.values()),
            "per_pair": rows,
        }

    print("\nComponent ablation (mean over 21 pairs)")
    header = f"{'description':<26}{'components':<42}{'mean F1':>9}"
    print(header)
    print("-" * len(header))
    for mode in ABLATION_MODES:
        comps = "+".join(DESCRIPTION_MODES[mode])
        print(f"{mode:<26}{comps:<42}{report['modes'][mode]['mean_f1']:>9.4f}")

    print("\nPer pair (columns: " + ", ".join(ABLATION_MODES) + ")")
    header = f"{'pair':<20}" + "".join(f"{' ' + m:>14}" for m in ABLATION_MODES)
    print(header)
    print("-" * len(header))
    for pair in report["pairs"]:
        print(f"{pair:<20}" + "".join(f"{mode_scores[m][pair]['f1']:>14.4f}"
                                      for m in ABLATION_MODES))

    for rule, threshold in POSTPROCESS_SWEEP:
        key = rule if threshold is None else f"{rule}@{threshold:g}"
        rows = {
            f"{src}↔{tgt}": _pair_score(
                matcher, taxonomies, alignments_raw, src, tgt,
                description_mode="name_only", postprocess=rule, threshold=threshold,
            )
            for src, tgt in pairs
        }
        report["postprocess"][key] = {
            "rule": rule,
            "threshold": threshold,
            "mean_f1": mean(r["f1"] for r in rows.values()),
            "mean_predicted": mean(r["predicted"] for r in rows.values()),
            "per_pair": rows,
        }

    print("\nPostprocessing sweep of `name_only` (mean over 21 pairs)")
    header = f"{'rule':<20}{'threshold':>10}{'mean F1':>9}{'mean #pred':>12}"
    print(header)
    print("-" * len(header))
    for key, block in report["postprocess"].items():
        thr = "—" if block["threshold"] is None else f"{block['threshold']:.2f}"
        print(f"{block['rule']:<20}{thr:>10}{block['mean_f1']:>9.4f}"
              f"{block['mean_predicted']:>12.1f}")

    keys = list(report["postprocess"])
    print("\nPer pair (columns: " + ", ".join(keys) + ")")
    header = f"{'pair':<20}" + "".join(f"{' ' + k:>22}" for k in keys)
    print(header)
    print("-" * len(header))
    for pair in report["pairs"]:
        print(f"{pair:<20}" + "".join(
            f"{report['postprocess'][k]['per_pair'][pair]['f1']:>22.4f}" for k in keys
        ))

    if out is None:
        out = Path(f"results/embedding-modes-{model_short}-{direction}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return report


# ── Canonical-direction migration (findings 2.9/3.11, contract C6) ────


def reorient_caches(only: str = "all") -> None:
    """Rewrite every flipped embedding/string record into the canonical direction.

    Only the records that disagree with the direction of the reference-alignment
    loader are re-run, so the operation is free (CPU, cached models) and touches
    one pair of the 21.  Records of approaches outside this slice are reported
    and left alone.

    `only` limits the run to the embedding side (`embedding`) or to the
    StringEquiv baseline (`string`); the latter needs no model at all.
    """
    from src.cache import canonical_directions, misoriented_records
    from src.runners.compare import run_embedding, run_string_equiv

    def mine(approach: str) -> bool:
        if approach == "string-equiv":
            return only in ("all", "string")
        if approach.startswith("embedding-"):
            return only in ("all", "embedding")
        return False

    data_dir = Path("data/processed/oaei")
    alignments_raw = load_alignments(data_dir / "alignments.json")
    canonical = canonical_directions(alignments_raw)

    flipped = misoriented_records(canonical)
    if not flipped:
        print("All cached records already follow the canonical direction.")
        return

    for approach, records in sorted(flipped.items()):
        mark = "→ re-run" if mine(approach) else "→ left alone"
        for record in records:
            print(f"  {approach:<26}{record['pair']:<22}"
                  f"{record['source']} → {record['target']}"
                  f"   (canonical: {record['expected_source']} → {record['expected_target']})"
                  f"  {mark}")

    pairs = sorted({
        canonical[record["pair"]]
        for approach, records in flipped.items()
        if mine(approach)
        for record in records
    })
    if not pairs:
        print("Nothing to re-run for this slice.")
        return

    taxonomies = load_taxonomies(data_dir)
    watched = sorted(a for a in flipped if mine(a))
    before = cached_means(watched)
    if any(a.startswith("embedding-") for a in watched):
        run_embedding(taxonomies, alignments_raw, pairs)
    if "string-equiv" in watched:
        run_string_equiv(taxonomies, alignments_raw, pairs)

    _print_means(before, cached_means(watched),
                 "Cached mean F1 before/after the re-orientation")
    left = misoriented_records(canonical)
    print("Still flipped:", left if left else "none")


def cached_means(approaches: list[str] | None = None) -> dict[str, float]:
    """Mean cached F1 of every single-record approach, for before/after printing."""
    from src.cache import load_all_cached

    out: dict[str, float] = {}
    for approach, pairs in load_all_cached().items():
        if approaches is not None and approach not in approaches:
            continue
        records = [d for d in pairs.values() if isinstance(d, dict) and "f1" in d]
        if records:
            out[approach] = mean(d["f1"] for d in records)
    return out


def _print_means(before: dict[str, float], after: dict[str, float], title: str) -> None:
    """Print the cached mean F1 per approach before and after a rebuild."""
    print(f"\n{title}")
    header = f"{'approach':<24}{'before':>9}{'after':>9}{'delta':>9}"
    print(header)
    print("-" * len(header))
    for approach in sorted(set(before) | set(after)):
        old, new = before.get(approach), after.get(approach)
        delta = f"{new - old:+.4f}" if old is not None and new is not None else "—"
        print(f"{approach:<24}{(f'{old:.4f}' if old is not None else '—'):>9}"
              f"{(f'{new:.4f}' if new is not None else '—'):>9}{delta:>9}")


def rebuild_caches(
    only: str = "all",
    postprocess: str = "greedy_injective",
    model_spec: str = "all",
) -> None:
    """Re-run every OAEI pair of the embedding models (and StringEquiv) in the
    canonical direction, writing the `results/embedding-*` caches.

    One class of rule for every column of the comparison table (contract C1):
    `postprocess` decides which one the embedding records carry — the ablation
    file keeps the alternatives side by side.  StringEquiv emits at most one
    match per source, so no rule applies to it; its records are rebuilt as-is.
    """
    from src.cache import canonical_directions
    from src.runners.compare import save_results

    data_dir = Path("data/processed/oaei")
    alignments_raw = load_alignments(data_dir / "alignments.json")
    taxonomies = load_taxonomies(data_dir)
    canonical = canonical_directions(alignments_raw)
    pairs = sorted(canonical.values())

    approaches = [
        f"embedding-{short}" for short, _ in selected_models(model_spec)
    ] if only != "string" else []
    watched = approaches + (["string-equiv"] if only != "embedding" else [])
    before = cached_means(watched)

    if only != "string":
        for short_name, model_id in selected_models(model_spec):
            print(f"Rebuilding embedding-{short_name} "
                  f"({len(pairs)} pairs, postprocess={postprocess})…")
            matcher = EmbeddingMatcher(model_name=model_id, description_mode="name_only")
            for src_name, tgt_name in pairs:
                gt = find_gt(src_name, tgt_name, alignments_raw)
                t0 = time.perf_counter()
                pred, _scores = matcher.match(
                    taxonomies[src_name], taxonomies[tgt_name], postprocess=postprocess
                )
                elapsed = time.perf_counter() - t0
                metrics = evaluate_1to1(pred, gt)
                save_results(
                    f"embedding-{short_name}", src_name, tgt_name, metrics,
                    extra={"wall_time": elapsed, "model": model_id,
                           "postprocess": postprocess},
                    match_pairs=[
                        {"source_id": m.source_id, "target_id": m.target_id,
                         "confidence": m.confidence}
                        for m in pred.matches
                    ],
                )
                print(f"  {src_name}↔{tgt_name}: F1={metrics.f1:.4f}  {elapsed:.1f}s")

    if only != "embedding":
        from src.matchers.string_equiv import StringEquivMatcher

        print(f"Rebuilding string-equiv ({len(pairs)} pairs)…")
        string_matcher = StringEquivMatcher()
        for src_name, tgt_name in pairs:
            gt = find_gt(src_name, tgt_name, alignments_raw)
            t0 = time.perf_counter()
            pred, _details = string_matcher.match(taxonomies[src_name], taxonomies[tgt_name])
            elapsed = time.perf_counter() - t0
            metrics = evaluate_1to1(pred, gt)
            save_results(
                "string-equiv", src_name, tgt_name, metrics,
                extra={"wall_time": elapsed},
                match_pairs=[
                    {"source_id": m.source_id, "target_id": m.target_id,
                     "confidence": m.confidence}
                    for m in pred.matches
                ],
            )
            print(f"  {src_name}↔{tgt_name}: F1={metrics.f1:.4f}")

    _print_means(before, cached_means(watched),
                 f"Cached mean F1 before/after the rebuild ({postprocess})")


def selected_models(spec: str) -> list[tuple[str, str]]:
    """Resolve a `--model` spec (short name, comma-separated list, or `all`)."""
    if spec == "all":
        return list(MODELS)
    known = dict(MODELS)
    chosen: list[tuple[str, str]] = []
    for name in spec.split(","):
        name = name.strip()
        if name not in known:
            raise SystemExit(f"unknown model {name!r}; known: {sorted(known)}")
        chosen.append((name, known[name]))
    return chosen


def run_experiments():
    data_proc = Path("data/processed")
    taxonomies = load_taxonomies(data_proc / "oaei")
    alignments_raw = load_alignments(data_proc / "oaei" / "alignments.json")
    synth_dir = data_proc / "synthetic"

    # Collect results across models for LaTeX table.
    summary: dict[str, dict] = {}

    for short_name, model_id in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Model: {short_name} ({model_id})")
        print("=" * 60)

        t0 = time.perf_counter()
        matcher = EmbeddingMatcher(model_name=model_id)
        load_time = time.perf_counter() - t0

        summary[short_name] = {"load_time_s": load_time}

        # ── Experiment 1: OAEI ──
        print("\n  Experiment 1: OAEI Conference Track")
        metrics = run_experiment_1(matcher, taxonomies, alignments_raw)
        avg_p = sum(m.precision for m in metrics) / len(metrics)
        avg_r = sum(m.recall for m in metrics) / len(metrics)
        avg_f1 = sum(m.f1 for m in metrics) / len(metrics)
        print(f"    Average over {len(metrics)} pairs: "
              f"P={avg_p:.4f}  R={avg_r:.4f}  F1={avg_f1:.4f}")
        summary[short_name]["oaei"] = {"P": avg_p, "R": avg_r, "F1": avg_f1}

        # ── Experiment 2: Synthetic ──
        print("\n  Experiment 2: Synthetic variations")
        synth_res = run_experiment_2(matcher, synth_dir)
        for mode in synth_res:
            scores = [synth_res[mode][s].f1 for s in synth_res[mode]]
            avg_f1_syn = sum(scores) / len(scores)
            print(f"    mode={mode:12s}: avg F1={avg_f1_syn:.4f}")
        summary[short_name]["synthetic"] = {
            mode: {s: synth_res[mode][s].f1 for s in synth_res[mode]}
            for mode in synth_res
        }

        # ── Experiment 3: HITL confidence ──
        print("\n  Experiment 3: HITL confidence (cmt ↔ conference)")
        cmt = taxonomies.get("cmt")
        conf = taxonomies.get("conference")
        if cmt and conf:
            pred, unct = matcher.match_with_threshold(cmt, conf, threshold=0.7)
            scores = [m.confidence for m in pred.matches]
            print(f"    Mean confidence: {sum(scores)/len(scores):.4f}")
            print(f"    Uncertain cases: {len(unct)} / {len(pred.matches)}")
            summary[short_name]["hitl"] = {
                "mean_confidence": sum(scores) / len(scores),
                "uncertain_count": len(unct),
                "total_nodes": len(pred.matches),
            }

    # ── Final summary table ──
    print(f"\n{'=' * 60}")
    print("CROSS-MODEL SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<20} {'OAEI P':>8} {'OAEI R':>8} {'OAEI F1':>8} {'Synth avg':>10}")
    print("-" * 54)
    for short_name in summary:
        oaei = summary[short_name].get("oaei", {})
        synth_f1s = []
        for mode in summary[short_name].get("synthetic", {}):
            synth_f1s.extend(summary[short_name]["synthetic"][mode].values())
        syn_avg = sum(synth_f1s) / len(synth_f1s) if synth_f1s else 0.0
        print(f"{short_name:<20} {oaei.get('P', 0):8.4f} {oaei.get('R', 0):8.4f} "
              f"{oaei.get('F1', 0):8.4f} {syn_avg:10.4f}")

    print("\nDone!")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", "--ablation", dest="modes", action="store_true",
                        help="run the description-mode ablation + postprocessing sweep")
    parser.add_argument("--model", default="MiniLM",
                        help="short name(s) from " + ", ".join(n for n, _ in MODELS) + ", or 'all'")
    parser.add_argument("--direction", choices=("canonical", "sorted"), default="canonical",
                        help="pair direction of the ablation (default: canonical, contract C6)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path of the ablation JSON (default per model/direction)")
    parser.add_argument("--reorient", action="store_true",
                        help="re-run the flipped cached records into the canonical direction")
    parser.add_argument("--rebuild", action="store_true",
                        help="re-run all 21 pairs of the embedding/string approaches in the "
                             "canonical direction, with --postprocess as the matching rule")
    parser.add_argument("--postprocess", choices=("argmax", "dedupe_by_target", "greedy_injective"),
                        default="greedy_injective",
                        help="matching rule written into the rebuilt caches (default: greedy_injective)")
    parser.add_argument("--only", choices=("all", "embedding", "string"), default="all",
                        help="with --reorient/--rebuild: which side to re-run (default: all)")
    args = parser.parse_args()

    if args.reorient and args.rebuild:
        raise SystemExit("choose either --reorient or --rebuild")
    if args.reorient:
        if args.out is not None:
            raise SystemExit("--out does not apply to --reorient")
        reorient_caches(args.only)
        return
    if args.rebuild:
        if args.out is not None:
            raise SystemExit("--out does not apply to --rebuild")
        rebuild_caches(args.only, args.postprocess, args.model)
        return
    if args.modes:
        models = selected_models(args.model)
        if args.out is not None and len(models) > 1:
            raise SystemExit("--out applies to a single --model")
        for short_name, model_id in models:
            run_ablation(short_name, model_id, args.direction, args.out)
        return
    run_experiments()


if __name__ == "__main__":
    main()
