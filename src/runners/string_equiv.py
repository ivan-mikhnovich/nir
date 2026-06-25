"""Run StringEquiv baseline on OAEI Conference pairs and synthetic variations.

Usage:
    uv run python -m src.runners.string_equiv
"""

import json
from pathlib import Path

from src.data_loader import load_taxonomy
from src.metrics import evaluate_1to1
from src.matchers.string_equiv import StringEquivMatcher


def load_taxonomies(data_dir: Path) -> dict:
    taxonomies: dict = {}
    for f in sorted(data_dir.glob("*.json")):
        if f.name == "alignments.json":
            continue
        tax = load_taxonomy(f)
        taxonomies[tax.name] = tax
    return taxonomies


def make_gt(make_gt_for_al_data):
    from src.taxonomy import Alignment, TaxonMatch
    def inner(al_data):
        return Alignment(
            source=al_data["source"],
            target=al_data["target"],
            matches=[
                TaxonMatch(source_id=m["source_id"], target_id=m["target_id"],
                          confidence=m.get("confidence", 1.0))
                for m in al_data["matches"]
            ],
        )
    return inner


def main() -> None:
    data_proc = Path("data/processed")
    taxonomies = load_taxonomies(data_proc / "oaei")
    with open(data_proc / "oaei" / "alignments.json", encoding="utf-8") as f:
        alignments_raw = json.load(f)
    matcher = StringEquivMatcher()

    # ── OAEI ──
    pairs = [
        ("cmt", "conference"),
        ("cmt", "confOf"),
        ("conference", "confOf"),
    ]

    print("StringEquiv baseline (case-insensitive label match)")
    print("=" * 50)

    for src_name, tgt_name in pairs:
        src = taxonomies[src_name]
        tgt = taxonomies[tgt_name]
        gt = None
        for al_data in alignments_raw:
            if al_data["source"] == src_name and al_data["target"] == tgt_name:
                from src.taxonomy import Alignment, TaxonMatch
                gt = Alignment(
                    source=al_data["source"],
                    target=al_data["target"],
                    matches=[
                        TaxonMatch(source_id=m["source_id"], target_id=m["target_id"],
                                  confidence=m.get("confidence", 1.0))
                        for m in al_data["matches"]
                    ],
                )
                break
        if gt is None:
            continue

        pred, details = matcher.match(src, tgt)
        metrics = evaluate_1to1(pred, gt)
        n = len(details)
        print(f"  {src_name}↔{tgt_name}: P={metrics.precision:.4f}  R={metrics.recall:.4f}  "
              f"F1={metrics.f1:.4f}  m={n}  t={matcher.last_timing['time']:.4f}s")

    # ── Synthetic ──
    print()
    synth_dir = data_proc / "synthetic"
    base_tax = load_taxonomy(synth_dir / "base_cmt.json")

    from src.taxonomy import Alignment as A, TaxonMatch as M
    for scenario in ["exact", "synonym", "structural"]:
        suffix = {"exact": "V1", "synonym": "SYN", "structural": "STR"}[scenario]
        var_path = synth_dir / f"{scenario}_cmt.json"
        if not var_path.exists():
            continue
        var_tax = load_taxonomy(var_path)
        gt = A(source=base_tax.name, target=var_tax.name, matches=[
            M(source_id=nid, target_id=f"{nid}_{suffix}")
            for nid in base_tax.nodes
            if f"{nid}_{suffix}" in var_tax.nodes
        ])
        pred, details = matcher.match(base_tax, var_tax)
        metrics = evaluate_1to1(pred, gt)
        print(f"  {scenario:12s}: P={metrics.precision:.4f}  R={metrics.recall:.4f}  "
              f"F1={metrics.f1:.4f}")


if __name__ == "__main__":
    main()
