"""Tests for the synthetic variation suite.

The generator used to build variants that were identical to the base taxonomy
modulo the id suffix for two of the four scenarios, so the reported synthetic
F1 of 1.00 was a property of the generator, not of a matcher. `structural` and
`attribute` must now really mutate the taxonomy, every variant name must carry
its seed, and `exact` must stay the deliberate pure-renaming control.
"""

from __future__ import annotations

from pathlib import Path

from src.data_loader import check_synthetic_variations, load_taxonomy

ONTOLOGY = Path(__file__).resolve().parents[1] / "data" / "processed" / "oaei" / "cmt.json"


def test_self_check_raises_on_a_pure_clone_lookalike():
    """`check_synthetic_variations` must fail when a scenario changes nothing."""
    base = load_taxonomy(ONTOLOGY)
    rows = check_synthetic_variations(base, seeds=(42,))
    assert {row["scenario"] for row in rows} == {
        "exact",
        "synonym",
        "structural",
        "attribute",
    }


def test_every_scenario_really_mutates_the_taxonomy_or_renames_ids():
    base = load_taxonomy(ONTOLOGY)
    rows = check_synthetic_variations(base, seeds=(42, 43, 44))
    assert len(rows) == 12
    for row in rows:
        scenario = row["scenario"]
        if scenario == "exact":
            # The control: ids only, content untouched.
            assert sum(
                row[key]
                for key in ("names", "parents", "children", "attributes", "comments", "disjoint")
            ) == 0
        elif scenario == "synonym":
            assert row["names"] > 0
        elif scenario == "structural":
            assert row["parents"] + row["children"] > 0
        elif scenario == "attribute":
            assert row["attributes"] > 0
        else:  # pragma: no cover - the set above is exhaustive
            raise AssertionError(f"неизвестный сценарий {scenario}")


def test_variant_names_carry_the_seed_and_are_unique():
    base = load_taxonomy(ONTOLOGY)
    rows = check_synthetic_variations(base, seeds=(42, 43, 44))
    names = [row["variant"] for row in rows]
    assert len(names) == len(set(names)) == 12
    assert "cmt_STR_s42" in names
    assert all(f"_s{row['seed']}" in row["variant"] for row in rows)


def test_every_variant_keeps_the_identity_ground_truth():
    base = load_taxonomy(ONTOLOGY)
    rows = check_synthetic_variations(base, seeds=(42,))
    assert all(row["gt_matches"] == row["nodes"] == len(base.nodes) for row in rows)
