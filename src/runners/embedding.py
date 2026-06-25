"""Run embedding-based matching experiments on OAEI + synthetic data.

Iterates over multiple sentence-transformer models for comparison.
"""

import json
import time
from pathlib import Path

from src.data_loader import load_taxonomy
from src.matchers.embedding import EmbeddingMatcher
from src.metrics import evaluate_1to1, MatchMetrics


# Models to evaluate (short name, full HF identifier).
MODELS: list[tuple[str, str]] = [
    ("LaBSE", "sentence-transformers/LaBSE"),
    ("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
    ("ruRoberta-large", "ai-forever/ruRoberta-large"),
    ("rubert-base", "DeepPavlov/rubert-base-cased"),
]


def load_taxonomies(data_dir: Path) -> dict:
    """Load all processed taxonomy JSON files from a directory."""
    taxonomies: dict = {}
    for f in sorted(data_dir.glob("*.json")):
        if f.name == "alignments.json":
            continue
        tax = load_taxonomy(f)
        taxonomies[tax.name] = tax
    return taxonomies


def load_alignments(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


if __name__ == "__main__":
    run_experiments()
