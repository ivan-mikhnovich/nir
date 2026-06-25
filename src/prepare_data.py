"""Prepare OAEI Conference data: parse, generate synthetic variations, save."""

from pathlib import Path
import sys

from src.data_loader import (
    load_oaei_conference,
    generate_synthetic_variations,
    save_taxonomy,
    save_alignments,
)


def main():
    data_raw = Path("data/raw/oaei")
    data_processed = Path("data/processed")

    print("=" * 60)
    print("Step 1: Loading OAEI Conference ontologies and alignments...")
    print("=" * 60)

    taxonomies, alignments = load_oaei_conference(data_raw)
    print(f"Loaded {len(taxonomies)} ontologies:")
    for name, tax in sorted(taxonomies.items()):
        print(f"  {name}: {tax.node_count} nodes, root={tax.root_id}")
    print(f"Loaded {len(alignments)} reference alignments.")
    for a in alignments:
        print(f"  {a.source} <-> {a.target}: {a.match_count} matches")

    # Save processed OAEI data.  .
    oaei_dir = data_processed / "oaei"
    oaei_dir.mkdir(parents=True, exist_ok=True)
    for name, tax in taxonomies.items():
        save_taxonomy(tax, oaei_dir / f"{name}.json")
    save_alignments(alignments, oaei_dir / "alignments.json")
    print(f"\nSaved OAEI data to {oaei_dir}/")

    # -- Synthetic variations ----------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Generating synthetic variations...")
    print("=" * 60)

    # Use CMT as the base taxonomy (29 nodes — manageable).  .
    base_tax = taxonomies.get("cmt")
    if base_tax is None:
        print("ERROR: cmt not found in taxonomies!")
        sys.exit(1)

    variations = generate_synthetic_variations(base_tax, seed=42)

    synth_dir = data_processed / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)

    # Save base taxonomy there too.  .
    save_taxonomy(base_tax, synth_dir / "base_cmt.json")

    all_synth_alignments: list = []
    for scenario, (var_tax, alignment) in sorted(variations.items()):
        save_taxonomy(var_tax, synth_dir / f"{scenario}_cmt.json")
        all_synth_alignments.append(alignment)
        print(f"  {scenario}: {var_tax.node_count} nodes, {alignment.match_count} matches")

    save_alignments(all_synth_alignments, synth_dir / "synthetic_alignments.json")
    print(f"\nSaved synthetic data to {synth_dir}/")

    # -- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_oaei_matches = sum(a.match_count for a in alignments)
    total_synth_matches = sum(a.match_count for a in all_synth_alignments)
    print(f"OAEI pairs: {len(alignments)} ({total_oaei_matches} total matches)")
    print(f"Synthetic pairs: {len(all_synth_alignments)} ({total_synth_matches} total matches)")
    print(f"Total ontologies: {len(taxonomies) + len(variations) + 1}")
    print("\nDone!")


if __name__ == "__main__":
    main()
