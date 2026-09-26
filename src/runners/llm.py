"""Run LLM-based matching experiments — LLMs4OM framework.

Standalone demo of the LLM matcher: the three concept representations (C, CP,
CC) and the ensemble on OAEI Conference pairs plus synthetic variations.  The
ensemble and the aggregation helpers are imported from the shared code paths
(`src.matchers.llm.ensemble_candidates`, `src.runners.compare`), so the demo
reports the same rule the cached comparison table uses.  The synthetic block
writes its numbers to `results/synthetic.json` (a file, not a subdirectory, so
`src.cache.load_all_cached` does not read it as another approach); the OAEI part
of the demo writes nothing to `results/` and costs one request per source node
per representation (3·|S| per pair, k candidate judgements per request).

Usage:
    uv run python -m src.runners.llm [--model MODEL] [--local]
    uv run python -m src.runners.llm --deepseek --only-stale
    uv run python -m src.runners.llm --all-pairs --pairs cmt-ekaw,ekaw-iasted
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.cache import canonical_directions
from src.data_loader import find_gt, load_alignments, load_taxonomies, load_taxonomy
from src.matchers.llm import ENSEMBLE_MODES, LLMMatcher, aggregate_mode_f1, ensemble_candidates
from src.matching_rules import to_alignment
from src.metrics import evaluate_1to1
from src.runners.compare import OAEI_SAMPLE_PAIRS, _get_all_pairs, select_pairs

LOCAL_BASE_URL = "http://127.0.0.1:12434/v1"
LLM_SHORT = {"concept": "C", "concept-parent": "CP", "concept-children": "CC"}


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


def run_experiments(model: str = "openai/gpt-4.1-mini", local: bool = False,
                    deepseek: bool = False, pairs: list[tuple[str, str]] | None = None):
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

    print("Approach: LLMs4OM (binary yes/no, top-k=5, one request per source node)")
    print("Post-processing: confidence > 0.7 + cardinality 1:1; "
          "ensemble = >=2-of-3 modes + greedy 1:1\n")

    # ── OAEI pairs (six by default, canonical direction) ──
    if pairs is None:
        pairs = select_pairs(OAEI_SAMPLE_PAIRS,
                             canonical=canonical_directions(alignments_raw))
    print(f"Pairs: {len(pairs)}")

    print("=" * 60)
    print("Experiment 1: OAEI Conference Track — LLMs4OM")
    print("=" * 60)

    from tqdm import tqdm

    mode_f1: dict[str, dict[str, float]] = {}
    ensemble_f1: dict[str, float] = {}

    for src_name, tgt_name in pairs:
        if src_name not in taxonomies or tgt_name not in taxonomies:
            continue
        src = taxonomies[src_name]
        tgt = taxonomies[tgt_name]

        gt = find_gt(src_name, tgt_name, alignments_raw)
        if gt is None:
            continue

        n_src = src.node_count
        print(f"\n  {src_name} ↔ {tgt_name} ({n_src} vs {tgt.node_count} nodes)")

        modes = list(ENSEMBLE_MODES)
        per_mode: dict[str, list[tuple[str, str, float]]] = {}

        mode_pbar = tqdm(modes, desc=f"  {src_name}↔{tgt_name}", position=1, leave=False, unit="mode")
        for mode in mode_pbar:
            short = LLM_SHORT[mode]
            mode_pbar.set_postfix_str(f"mode={short}")
            pred, details = matcher.match(src, tgt, mode=mode,
                                          pbar_desc=f"    {short}", pbar_position=2)
            t = matcher.last_timing

            metrics = evaluate_1to1(pred, gt)
            n_matches = len([d for d in details if d.target_id])
            per_mode[mode] = [
                (d.source_id, d.target_id, d.confidence) for d in details if d.target_id
            ]
            tqdm.write(
                f"    [{short}] P={metrics.precision:.4f}  R={metrics.recall:.4f}  "
                f"F1={metrics.f1:.4f}  matches={n_matches}/{len(details)}  "
                f"api={t.get('api_time', 0):.0f}s  requests={t.get('requests', 0)}  "
                f"per_node={t.get('api_time', 0)/n_src:.1f}s"
            )
            mode_f1.setdefault(f"{src_name}↔{tgt_name}", {})[mode] = metrics.f1
        mode_pbar.close()

        # Ensemble: the shared >=2-of-3 rule, no extra API request.
        selected = ensemble_candidates(per_mode)
        pred_ens = to_alignment(selected, src_name, tgt_name)
        metrics_ens = evaluate_1to1(pred_ens, gt)
        ensemble_f1[f"{src_name}↔{tgt_name}"] = metrics_ens.f1
        print(f"    [ENS] P={metrics_ens.precision:.4f}  R={metrics_ens.recall:.4f}  "
              f"F1={metrics_ens.f1:.4f}  matches={len(selected)}  "
              f"(>=2-of-3 of {len(modes)} modes, greedy 1:1, no extra requests)")

    # Summary with the shared aggregation helpers (also used by compare.py).
    if mode_f1:
        aggregates = aggregate_mode_f1(mode_f1, ensemble_f1)
        print(f"\n  Aggregation over {aggregates['n_pairs']} pairs: "
              f"mean_over_modes={aggregates['mean_over_modes']:.4f}  "
              f"loo_mode_selection={aggregates['loo_mode_selection']:.4f}  "
              f"ensemble={aggregates['ensemble']:.4f}  "
              f"oracle_max={aggregates['oracle_max']:.4f} (oracle)")

    # ── Synthetic variations ──
    print("\n" + "=" * 60)
    print("Experiment 2: Synthetic Variations (CMT) — concept mode only")
    print("=" * 60)

    synth_dir = data_proc / "synthetic"
    base_tax = load_taxonomy(synth_dir / "base_cmt.json")

    from src.taxonomy import Alignment, TaxonMatch

    scenarios = ["exact", "synonym", "structural", "attribute"]
    suffix_map = {
        "exact": "V1",
        "synonym": "SYN",
        "structural": "STR",
        "attribute": "ATTR",
    }

    synth_records: list[dict] = []
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
        synth_records.append({
            "scenario": scenario,
            "mode": "concept",
            "suffix": expected_suffix,
            "pairs": len(gt_matches),
            "matches": n_matches,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "api_time": t.get("api_time", 0.0),
            "requests": t.get("requests", 0),
            "retries": t.get("retries", 0),
        })

    # Артефакт прогона: числа сценариев сохраняются файлом, а не подкаталогом,
    # иначе `src.cache.load_all_cached` увидел бы в них ещё один подход.
    if synth_records:
        out_path = Path("results/synthetic.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": model,
            "provider": "deepseek" if deepseek else ("local" if local else "kodikrouter"),
            "top_k": matcher.top_k,
            "temperature": matcher.temperature,
            "mode": "concept",
            "scenarios": synth_records,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM taxonomy matching experiments (LLMs4OM).")
    parser.add_argument("--model", default="openai/gpt-4.1-mini", help="Model name.")
    parser.add_argument("--local", action="store_true", help="Use local llama-swap.")
    parser.add_argument("--deepseek", action="store_true", help="Use DeepSeek direct API.")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Run all 21 OAEI pairs (default: 6 sample pairs).")
    parser.add_argument("--pairs", type=str, default=None, metavar="SPEC",
                        help="Comma-separated pairs, e.g. `cmt-conference,ekaw-edas`.")
    parser.add_argument("--only-stale", action="store_true",
                        help="Only pairs touching cmt or ekaw (the 11 stale ones).")
    args = parser.parse_args()

    raw = load_alignments(Path("data/processed") / "oaei" / "alignments.json")
    all_pairs = _get_all_pairs(raw)
    known = {a["source"] for a in raw} | {a["target"] for a in raw}
    canonical = canonical_directions(raw)
    if args.pairs or args.only_stale or args.all_pairs:
        pairs = select_pairs(all_pairs, only_stale=args.only_stale,
                             spec=args.pairs, known=known, canonical=canonical)
    else:
        pairs = select_pairs(OAEI_SAMPLE_PAIRS, canonical=canonical)

    if args.deepseek:
        run_experiments(args.model, local=False, deepseek=True, pairs=pairs)
    else:
        run_experiments(args.model, local=args.local, pairs=pairs)
