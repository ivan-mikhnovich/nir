"""Run LLM-based matching experiments — LLMs4OM framework.

Evaluates the LLM matcher with three concept representations (C, CP, CC)
and ensemble on OAEI Conference pairs + synthetic variations.

Usage:
    uv run python -m src.runners.llm [--model MODEL] [--local]
"""

import argparse
import json
from pathlib import Path

from src.data_loader import load_taxonomy
from src.matchers.llm import LLMMatcher
from src.metrics import evaluate_1to1

LOCAL_BASE_URL = "http://127.0.0.1:12434/v1"


def load_taxonomies(data_dir: Path) -> dict:
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


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def run_experiments(model: str = "openai/gpt-4.1-mini", local: bool = False, deepseek: bool = False):
    data_proc = Path("data/processed")
    taxonomies = load_taxonomies(data_proc / "oaei")
    alignments_raw = load_alignments(data_proc / "oaei" / "alignments.json")

    if local:
        matcher = LLMMatcher(
            model=model,
            base_url=LOCAL_BASE_URL,
            api_key="not-needed",
            top_k=5,
            temperature=0.0,
            max_workers=1,
        )
        print(f"LLM model: {model} (local llama-swap, sequential)")
    elif deepseek:
        import os
        matcher = LLMMatcher(
            model=model,
            base_url=DEEPSEEK_BASE_URL,
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            top_k=5,
            temperature=0.0,
            max_workers=10,
            use_structured_output=False,  # DeepSeek does not support json_schema.
        )
        print(f"LLM model: {model} (DeepSeek direct, {matcher.max_workers} workers)")
    else:
        matcher = LLMMatcher(model=model, top_k=5, temperature=0.0)
        print(f"LLM model: {model} (kodikrouter, {matcher.max_workers} workers)")

    print("Approach: LLMs4OM (binary yes/no, top-k=5, per-source-node calls)")
    print("Post-processing: confidence > 0.7 + cardinality 1:1\n")

    # ── OAEI: three representative pairs ──
    pairs = [
        ("cmt", "conference"),
        ("cmt", "confOf"),
        ("conference", "confOf"),
    ]

    print("=" * 60)
    print("Experiment 1: OAEI Conference Track — LLMs4OM")
    print("=" * 60)

    from tqdm import tqdm

    for src_name, tgt_name in pairs:
        if src_name not in taxonomies or tgt_name not in taxonomies:
            continue
        src = taxonomies[src_name]
        tgt = taxonomies[tgt_name]

        gt = None
        for al_data in alignments_raw:
            if al_data["source"] == src_name and al_data["target"] == tgt_name:
                gt = make_gt_alignment(al_data)
                break
        if gt is None:
            continue

        n_src = src.node_count
        print(f"\n  {src_name} ↔ {tgt_name} ({n_src} vs {tgt.node_count} nodes)")

        modes = ["concept", "concept-parent", "concept-children"]
        mode_metrics: dict[str, float] = {}

        mode_pbar = tqdm(modes, desc=f"  {src_name}↔{tgt_name}", position=1, leave=False, unit="mode")
        for mode in mode_pbar:
            short = {"concept": "C", "concept-parent": "CP", "concept-children": "CC"}[mode]
            mode_pbar.set_postfix_str(f"mode={short}")
            pred, details = matcher.match(src, tgt, mode=mode,
                                          pbar_desc=f"    {short}", pbar_position=2)
            t = matcher.last_timing

            metrics = evaluate_1to1(pred, gt)
            n_matches = len([d for d in details if d.target_id])
            mode_metrics[mode] = metrics.f1
            tqdm.write(
                f"    [{short}] P={metrics.precision:.4f}  R={metrics.recall:.4f}  "
                f"F1={metrics.f1:.4f}  matches={n_matches}/{len(details)}  "
                f"api={t.get('api_time', 0):.0f}s  per_node={t.get('api_time', 0)/n_src:.1f}s"
            )
        mode_pbar.close()

        # Ensemble.
        print(f"    Ensemble ({src_name}↔{tgt_name})...")
        pred_ens, details_ens = matcher.match_ensemble(src, tgt)
        t = matcher.last_timing
        metrics_ens = evaluate_1to1(pred_ens, gt)
        n_matches = len([d for d in details_ens if d.target_id])
        print(f"    [ENS] P={metrics_ens.precision:.4f}  R={metrics_ens.recall:.4f}  "
              f"F1={metrics_ens.f1:.4f}  matches={n_matches}/{len(details_ens)}  "
              f"api={t.get('api_time', 0):.0f}s (sum of 3 modes)")

    # ── Synthetic variations ──
    print("\n" + "=" * 60)
    print("Experiment 2: Synthetic Variations (CMT) — concept mode only")
    print("=" * 60)

    synth_dir = data_proc / "synthetic"
    base_tax = load_taxonomy(synth_dir / "base_cmt.json")

    from src.taxonomy import Alignment, TaxonMatch

    scenarios = ["exact", "synonym", "structural"]
    suffix_map = {"exact": "V1", "synonym": "SYN", "structural": "STR"}

    for scenario in tqdm(scenarios, desc="  Synthetic", position=1, leave=False, unit="scenario"):
        var_path = synth_dir / f"{scenario}_cmt.json"
        if not var_path.exists():
            continue
        var_tax = load_taxonomy(var_path)

        expected_suffix = suffix_map[scenario]
        gt_matches = [
            TaxonMatch(source_id=nid, target_id=f"{nid}_{expected_suffix}")
            for nid in base_tax.nodes
            if f"{nid}_{expected_suffix}" in var_tax.nodes
        ]
        gt = Alignment(source=base_tax.name, target=var_tax.name, matches=gt_matches)

        pred, details = matcher.match(base_tax, var_tax, mode="concept",
                                      pbar_desc=f"    {scenario}", pbar_position=2)
        t = matcher.last_timing
        metrics = evaluate_1to1(pred, gt)
        n_matches = len([d for d in details if d.target_id])
        tqdm.write(
            f"  {scenario:12s}: P={metrics.precision:.4f}  R={metrics.recall:.4f}  "
            f"F1={metrics.f1:.4f}  matches={n_matches}/{len(details)}  "
            f"api={t.get('api_time', 0):.0f}s"
        )

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM taxonomy matching experiments (LLMs4OM).")
    parser.add_argument("--model", default="openai/gpt-4.1-mini", help="Model name.")
    parser.add_argument("--local", action="store_true", help="Use local llama-swap.")
    parser.add_argument("--deepseek", action="store_true", help="Use DeepSeek direct API.")
    args = parser.parse_args()

    if args.deepseek:
        run_experiments(args.model, local=False, deepseek=True)
    else:
        run_experiments(args.model, local=args.local)
