"""Tests that the published LLM aggregations still agree with their caches.

The review (numbers F2) found the headline table mixed two incompatible
aggregations and that the repository carried more than one set of "published"
numbers, so a reader could not tell which rule produced which value.
`results/llm-aggregations.json` is now the machine-readable publication of the
four rules — `mean_over_modes`, `loo_mode_selection`, `ensemble`, `oracle_max`
— and this module recomputes each of them from the `results/llm-*/` records
and compares to the stored value (tolerance 1e-9).

The ensemble records are also checked to be injective and to come from the
shared ">=2 of 3" rule, not from the legacy plurality vote.

Note on a check that is *not* made: the ensemble F1 is *not* asserted to stay
inside the range of the per-mode F1 values of its pair. That invariant does
not hold in the data — the >=2-of-3 + greedy rule can beat every single
representation (e.g. `llm-bm25` `edas↔sigkdd`: ensemble 0.705883 > max mode
0.666667), because it builds a different alignment rather than averaging the
inputs. The structural invariants asserted instead are: the ensemble is
injective, and every accepted pair was voted for by at least two of the three
per-mode records.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pytest

from src.matchers.llm import ENSEMBLE_MIN_VOTES, ENSEMBLE_MODES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AGGREGATIONS = RESULTS / "llm-aggregations.json"

APPROACHES = ("llm-bm25", "llm-deepseek", "llm-gpt", "llm-hybrid")
MODES = tuple(ENSEMBLE_MODES)
AGGREGATE_KEYS = ("mean_over_modes", "loo_mode_selection", "ensemble", "oracle_max")
TOL = 1e-9


def load_records(approach: str) -> dict[str, dict[str, dict]]:
    """Return `pair -> mode -> record` for one approach, read from disk."""
    per_pair: dict[str, dict[str, dict]] = {}
    for path in sorted((RESULTS / approach).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        per_pair.setdefault(record["pair"], {})[record["mode"]] = record
    return per_pair


def per_pair_modes(records: dict[str, dict[str, dict]]) -> dict[str, dict[str, float]]:
    return {
        pair: {mode: modes[mode]["f1"] for mode in MODES}
        for pair, modes in records.items()
    }


def per_pair_ensemble(records: dict[str, dict[str, dict]]) -> dict[str, float]:
    return {pair: modes["ensemble"]["f1"] for pair, modes in records.items()}


def recompute(
    modes: dict[str, dict[str, float]],
    ensemble: dict[str, float],
) -> dict[str, float]:
    """Recompute the four published rules from the cached per-pair F1 values."""
    pairs = sorted(modes)
    all_modes = sorted({mode for by_mode in modes.values() for mode in by_mode})

    mean_over_modes = mean(
        mean(modes[pair][mode] for mode in all_modes if mode in modes[pair])
        for pair in pairs
    )

    loo_values = []
    for held_out in pairs:
        others = [pair for pair in pairs if pair != held_out]
        best_mode = max(
            all_modes,
            key=lambda mode: mean(
                modes[pair][mode] for pair in others if mode in modes[pair]
            ),
        )
        loo_values.append(modes[held_out].get(best_mode, 0.0))

    return {
        "mean_over_modes": mean_over_modes,
        "loo_mode_selection": mean(loo_values),
        "oracle_max": mean(max(by_mode.values()) for by_mode in modes.values()),
        "ensemble": mean(ensemble[pair] for pair in pairs if pair in ensemble),
        "n_pairs": len(pairs),
    }


def stored_aggregations() -> dict:
    return json.loads(AGGREGATIONS.read_text(encoding="utf-8"))["approaches"]


def test_aggregations_file_has_the_expected_shape():
    data = json.loads(AGGREGATIONS.read_text(encoding="utf-8"))
    assert set(data) == {"rules", "approaches"}
    assert set(data["approaches"]) == set(APPROACHES)
    for rule in AGGREGATE_KEYS:
        assert rule in data["rules"]
    for approach in APPROACHES:
        block = data["approaches"][approach]
        for key in (*AGGREGATE_KEYS, "n_pairs", "per_pair_f1", "plurality", "requests"):
            assert key in block


@pytest.mark.parametrize("approach", APPROACHES)
def test_published_aggregations_match_the_cached_records(approach):
    stored = stored_aggregations()[approach]
    records = load_records(approach)
    computed = recompute(per_pair_modes(records), per_pair_ensemble(records))

    assert computed["n_pairs"] == stored["n_pairs"] == len(records) == 21
    for key in AGGREGATE_KEYS:
        assert computed[key] == pytest.approx(stored[key], abs=TOL)


@pytest.mark.parametrize("approach", APPROACHES)
def test_per_pair_table_reproduces_every_cached_record(approach):
    stored = stored_aggregations()[approach]
    records = load_records(approach)
    modes = per_pair_modes(records)
    ensemble = per_pair_ensemble(records)

    assert set(stored["per_pair_f1"]) == set(records)
    for pair in records:
        table = stored["per_pair_f1"][pair]
        for mode in MODES:
            assert table[mode] == pytest.approx(modes[pair][mode], abs=TOL)
        assert table["ensemble"] == pytest.approx(ensemble[pair], abs=TOL)


@pytest.mark.parametrize("approach", APPROACHES)
def test_every_ensemble_record_is_injective_and_uses_the_voting_rule(approach):
    records = load_records(approach)
    for pair, by_mode in records.items():
        record = by_mode["ensemble"]
        assert record["rule"] == f">={ENSEMBLE_MIN_VOTES}-of-{len(MODES)}+greedy_injective"
        assert record["min_votes"] == ENSEMBLE_MIN_VOTES
        assert record["modes"] == list(MODES)

        matched = [
            (m["source_id"], m["target_id"]) for m in record["match_pairs"]
        ]
        sources = [src for src, _ in matched]
        targets = [tgt for _, tgt in matched]
        assert len(set(sources)) == len(sources), pair
        assert len(set(targets)) == len(targets), pair

        # A pair survives only with >=2 votes, so it must appear in at least
        # two of the per-mode records of the same pair.
        voted_for = {
            (m["source_id"], m["target_id"])
            for mode in MODES
            for m in by_mode[mode]["match_pairs"]
        }
        assert set(matched) <= voted_for, pair


@pytest.mark.parametrize("approach", APPROACHES)
def test_stored_f1_equals_its_predictions_against_the_ground_truth(approach):
    """Every stored F1 must be the F1 of the record's own `match_pairs`.

    This is the cache-side half of the publication guard: an aggregation can
    only be as good as the per-record values it averages, so each record's
    headline number is recomputed from its stored predictions and the shipped
    reference alignment.
    """
    from src.cache import record_alignment
    from src.data_loader import find_gt, load_alignments
    from src.metrics import evaluate_1to1

    ground_truth = load_alignments(
        ROOT / "data" / "processed" / "oaei" / "alignments.json"
    )
    records = load_records(approach)
    checked = 0
    for pair, by_mode in records.items():
        for mode, record in by_mode.items():
            built = record_alignment(record)
            gt = find_gt(record["source"], record["target"], ground_truth)
            assert built is not None and gt is not None, (pair, mode)
            assert evaluate_1to1(built, gt).f1 == pytest.approx(
                record["f1"], abs=TOL
            )
            checked += 1
    assert checked == 21 * (len(MODES) + 1)


@pytest.mark.parametrize("approach", APPROACHES)
def test_published_ensemble_is_not_the_legacy_plurality(approach):
    stored = stored_aggregations()[approach]
    plurality_mean = stored["plurality"]["mean"]
    assert plurality_mean is not None
    assert stored["ensemble"] > 0.0
    assert abs(stored["ensemble"] - plurality_mean) > TOL
