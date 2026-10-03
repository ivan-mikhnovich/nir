"""Behavioral safeguards for complete offline Jev evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.cache import pair_key
from src.matchers.jev import build_request
from src.matchers.llm import ENSEMBLE_MODES
from src.runners.jev_evaluate import (
    THRESHOLDS,
    calibration,
    choose_threshold,
    evaluate,
    load_collection,
    request_accounting,
    score_predictions,
    select_predictions,
)
from src.taxonomy import Alignment, TaxonMatch


def collection(tmp_path: Path) -> tuple[Path, Path, Path, dict, list[dict]]:
    """Create two genuinely scored pairs with complete attempted-request records."""
    experiment = tmp_path / "experiment"
    data = tmp_path / "data"
    experiment.mkdir()
    data.mkdir()
    reference = [
        {
            "source": source,
            "target": target,
            "matches": [
                {"source_id": "s1", "target_id": "t1"},
                {"source_id": "s2", "target_id": "t2"},
            ],
        }
        for source, target in (("a", "b"), ("a", "c"))
    ]
    manifest = {
        "schema_version": 1,
        "model": "typesafe/jev-1.13",
        "modes": list(ENSEMBLE_MODES),
        "thresholds": list(THRESHOLDS),
        "pairs": [],
    }
    rows = []
    baseline = {
        "llm-hybrid": {},
        "string-equiv": {},
        "embedding-ruRoberta-large": {},
    }
    for record in reference:
        source, target = record["source"], record["target"]
        pair = pair_key(source, target)
        manifest["pairs"].append(
            {
                "pair": pair,
                "source": source,
                "target": target,
                "candidates": {
                    s: [
                        {"target_id": t, "retrieval_score": 0.5}
                        for t in ("t1", "t2")
                    ]
                    for s in ("s1", "s2")
                },
                "descriptions": {
                    mode: {
                        "source": {"s1": "one", "s2": "two"},
                        "target": {"t1": "one", "t2": "two"},
                    }
                    for mode in ENSEMBLE_MODES
                },
            }
        )
        for mode in ENSEMBLE_MODES:
            for s, probabilities in (("s1", (0.9, 0.1)), ("s2", (0.8, 0.6))):
                rows.append(
                    {
                        "key": f"{pair}|{mode}|{s}",
                        "pair": pair,
                        "source": source,
                        "target": target,
                        "mode": mode,
                        "source_id": s,
                        "request": build_request(
                            "one" if s == "s1" else "two", ["one", "two"]
                        ),
                        "response": {
                            "model": "typesafe/jev-1.13-20260917",
                            "provider": "provider-a",
                            "id": f"{pair}-{mode}-{s}",
                            "answers": {
                                f"candidate_{i}": {"type": "noul", "noul": p}
                                for i, p in enumerate(probabilities)
                            },
                            "usage": {
                                "cost": 0.01,
                                "input_tokens": 10,
                                "output_tokens": 4,
                            },
                        },
                        "status_code": 200,
                        "timestamp": "2026-10-04T00:00:00Z",
                        "latency_seconds": 0.2,
                        "error": None,
                        "probabilities": [
                            {"source_id": s, "target_id": t, "probability": p}
                            for t, p in zip(("t1", "t2"), probabilities)
                        ],
                    }
                )
            baseline["llm-hybrid"][f"{pair}|{mode}"] = {
                "pair": pair,
                "mode": mode,
                "baseline": {"f1": 0.4},
            }
        baseline["llm-hybrid"][f"{pair}|ensemble"] = {
            "pair": pair,
            "mode": "ensemble",
            "baseline": {"f1": 0.5},
        }
        baseline["string-equiv"][pair] = {
            "pair": pair,
            "mode": None,
            "baseline": {"f1": 0.3},
        }
        baseline["embedding-ruRoberta-large"][pair] = {
            "pair": pair,
            "mode": None,
            "baseline": {"f1": 0.4},
            "loo": {"f1": 0.6},
        }
    (data / "alignments.json").write_text(
        json.dumps(reference), encoding="utf-8"
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    save_collection(experiment, manifest, rows)
    return experiment, data, baseline_path, manifest, rows


def save_collection(
    experiment: Path, manifest: dict, rows: list[dict]
) -> None:
    """Persist the fixture's actual manifest and attempted-request journal."""
    (experiment / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (experiment / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_all_candidates_are_available_to_global_injective_selection():
    predictions = select_predictions(
        [
            ("s1", "t1", 0.9),
            ("s1", "t2", 0.85),
            ("s2", "t1", 0.8),
            ("s2", "t2", 0.6),
        ],
        0.5,
    )
    assert predictions == [("s1", "t1", 0.9), ("s2", "t2", 0.6)]
    assert select_predictions([("s", "t", 0.7)], 0.7) == []


def test_empty_prediction_keeps_full_recall_denominator():
    gold = Alignment(
        "a", "b", [TaxonMatch("s1", "t1"), TaxonMatch("s2", "t2")]
    )
    metrics = score_predictions([], gold)
    assert metrics["total_ground_truth"] == 2
    assert metrics["total_predicted"] == metrics["true_positives"] == 0
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0


def test_threshold_selection_never_uses_held_pair_and_breaks_ties_lowest():
    sweep = {
        "held": {
            threshold: {"f1": float(threshold > 0.5)}
            for threshold in THRESHOLDS
        },
        "train": {
            threshold: {"f1": float(threshold <= 0.5)}
            for threshold in THRESHOLDS
        },
    }
    assert choose_threshold(sweep, "held") == 0
    sweep["held"] = {
        threshold: {"f1": 1 - block["f1"]}
        for threshold, block in sweep["held"].items()
    }
    assert choose_threshold(sweep, "held") == 0
    assert choose_threshold(sweep, "train") == 0


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing", "Incomplete collection"),
        ("duplicate", "Duplicate successful"),
        ("candidate_missing", "candidate coverage mismatch"),
        ("candidate_extra", "candidate coverage mismatch"),
        ("candidate_duplicate", "candidate coverage mismatch"),
        ("reverse_row", "reverse direction"),
        ("reverse_manifest", "reverse direction"),
        ("missing_pair", "every reference pair"),
        ("wrong_source", "Invalid candidate probability"),
        ("nan", "Invalid candidate probability"),
        ("payload", "request payload differs"),
        ("raw_response", "raw Jev response"),
        ("raw_incomplete", "answer keys"),
        ("unbilled_success", "billed usage.cost"),
    ],
)
def test_bad_collections_fail_instead_of_publishing(
    tmp_path, mutation, message
):
    experiment, data, baseline, manifest, rows = collection(tmp_path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "candidate_missing":
        rows[0]["probabilities"].pop()
    elif mutation == "candidate_extra":
        rows[0]["probabilities"][0]["target_id"] = "outside"
    elif mutation == "candidate_duplicate":
        rows[0]["probabilities"][1] = copy.deepcopy(
            rows[0]["probabilities"][0]
        )
    elif mutation == "reverse_row":
        rows[0]["source"], rows[0]["target"] = (
            rows[0]["target"],
            rows[0]["source"],
        )
    elif mutation == "reverse_manifest":
        spec = manifest["pairs"][0]
        spec["source"], spec["target"] = spec["target"], spec["source"]
    elif mutation == "missing_pair":
        manifest["pairs"].pop()
    elif mutation == "wrong_source":
        rows[0]["probabilities"][0]["source_id"] = "wrong"
    elif mutation == "nan":
        rows[0]["probabilities"][0]["probability"] = float("nan")
    elif mutation == "payload":
        rows[0]["request"]["state"]["source"] = "tampered source"
    elif mutation == "raw_response":
        rows[0]["response"]["answers"]["candidate_0"]["noul"] = 0.2
    elif mutation == "raw_incomplete":
        rows[0]["response"]["answers"].pop("candidate_0")
    elif mutation == "unbilled_success":
        rows[0]["response"]["usage"].pop("cost")
    save_collection(experiment, manifest, rows)
    with pytest.raises(ValueError, match=message):
        evaluate(experiment, data, baseline)


def test_reference_and_retrieval_denominators_include_unretrieved_gold(
    tmp_path,
):
    experiment, data, baseline, manifest, rows = collection(tmp_path)
    reference = json.loads(
        (data / "alignments.json").read_text(encoding="utf-8")
    )
    reference[0]["matches"].append(
        {"source_id": "unretrieved", "target_id": "outside"}
    )
    (data / "alignments.json").write_text(
        json.dumps(reference), encoding="utf-8"
    )
    report = evaluate(experiment, data, baseline)
    assert report["candidate_retrieval"]["total_gold"] == 5
    assert report["candidate_retrieval"]["retrieved_gold"] == 4
    assert report["candidate_retrieval"]["recall"] == 0.8
    assert report["candidate_retrieval"]["mean_candidate_count"] == 2
    metrics = report["configurations"]["concept_fixed_0.5"]
    assert metrics["pooled"]["f1"] == pytest.approx(8 / 9)
    assert metrics["macro"]["f1"] == pytest.approx(0.9)


def test_shared_threshold_ensemble_and_mode_selection_do_not_leak(tmp_path):
    experiment, data, baseline, manifest, rows = collection(tmp_path)
    report = evaluate(experiment, data, baseline)
    held = manifest["pairs"][0]["pair"]
    choices = {
        name: (
            block["per_pair"][held]["threshold"],
            block["per_pair"][held].get("mode"),
        )
        for name, block in report["configurations"].items()
        if name
        in (
            "concept_loo",
            "ensemble_shared_threshold_loo",
            "loo_mode_and_threshold_selection",
        )
    }
    reference = json.loads(
        (data / "alignments.json").read_text(encoding="utf-8")
    )
    reference[0]["matches"] = [{"source_id": "s1", "target_id": "t2"}]
    (data / "alignments.json").write_text(
        json.dumps(reference), encoding="utf-8"
    )
    changed = evaluate(experiment, data, baseline)
    for name, expected in choices.items():
        record = changed["configurations"][name]["per_pair"][held]
        assert (record["threshold"], record.get("mode")) == expected
    assert choices["ensemble_shared_threshold_loo"][0] == 0
    assert choices["loo_mode_and_threshold_selection"][1] == "concept"


def test_ensemble_threshold_is_selected_by_ensemble_not_individual_modes(
    tmp_path,
):
    experiment, data, baseline, manifest, rows = collection(tmp_path)
    held, training = [spec["pair"] for spec in manifest["pairs"]]
    for spec in manifest["pairs"]:
        for items in spec["candidates"].values():
            items.append({"target_id": "t3", "retrieval_score": 0.1})
        for descriptions in spec["descriptions"].values():
            descriptions["target"]["t3"] = "unrelated"
    for row in rows:
        source = row["source_id"]
        row["probabilities"].append(
            {"source_id": source, "target_id": "t3", "probability": 0.01}
        )
        if row["pair"] == training:
            values = (
                (0.9, 0.01, 0.01)
                if source == "s1"
                else (0.01, 0.4, 0.1 if row["mode"] == "concept" else 0.5)
            )
            for item, probability in zip(row["probabilities"], values):
                item["probability"] = probability
        spec = next(
            spec for spec in manifest["pairs"] if spec["pair"] == row["pair"]
        )
        descriptions = spec["descriptions"][row["mode"]]
        row["request"] = build_request(
            descriptions["source"][source],
            [
                descriptions["target"][item["target_id"]]
                for item in row["probabilities"]
            ],
        )
        row["response"]["answers"] = {
            f"candidate_{i}": {"type": "noul", "noul": item["probability"]}
            for i, item in enumerate(row["probabilities"])
        }
    save_collection(experiment, manifest, rows)
    report = evaluate(experiment, data, baseline)
    assert (
        report["configurations"]["concept_loo"]["per_pair"][held]["threshold"]
        == 0
    )
    assert (
        report["configurations"]["concept-parent_loo"]["per_pair"][held][
            "threshold"
        ]
        == 0.5
    )
    assert (
        report["configurations"]["ensemble_shared_threshold_loo"]["per_pair"][
            held
        ]["threshold"]
        == 0.5
    )


def test_reliability_bins_and_brier_use_candidates_not_selected_mapping():
    gold = {"a": Alignment("a", "b", [TaxonMatch("s", "yes")])}
    report = calibration(
        {"a": [("s", "yes", 1.0), ("s", "no", 0.5), ("other", "no", 0.0)]},
        gold,
    )
    assert report["candidate_pairs"] == 3
    assert report["brier"] == pytest.approx(1 / 12)
    assert report["ece_10_bins"] == pytest.approx(1 / 6)
    assert (
        report["positive_prevalence"]
        == report["always_negative_brier"]
        == pytest.approx(1 / 3)
    )
    assert report["reliability_bins"][9]["mean_probability"] == 1
    assert report["reliability_bins"][5]["positive_fraction"] == 0
    assert report["reliability_bins"][1]["mean_probability"] is None


def test_all_attempt_costs_and_missing_usage_are_separate_from_success_latency(
    tmp_path,
):
    experiment, data, baseline, manifest, rows = collection(tmp_path)
    failed = {
        **copy.deepcopy(rows[0]),
        "error": "rate limited",
        "status_code": 429,
        "latency_seconds": 99,
        "probabilities": [],
        "response": {
            "usage": {"cost": 0.02, "prompt_tokens": 3, "completion_tokens": 1}
        },
    }
    missing_usage = {
        **copy.deepcopy(failed),
        "response": {"error": "unavailable"},
    }
    rows = [failed, missing_usage, *rows]
    save_collection(experiment, manifest, rows)
    _, _, attempts, successes = load_collection(experiment, data)
    report = request_accounting(attempts, successes)
    assert report["cost"] == pytest.approx(0.14)
    assert report["input_tokens"] == 123
    assert report["output_tokens"] == 49
    assert report["cost_missing_attempts"] == 1
    assert report["error_attempts"] == 2
    assert report["successful_request_latency_seconds"]["max"] == 0.2
    assert report["successful_request_latency_seconds"]["count"] == 12


def test_exact_macro_pooled_and_comparison_pair_population(tmp_path):
    experiment, data, baseline, _, _ = collection(tmp_path)
    report = evaluate(experiment, data, baseline)
    metrics = report["configurations"]["ensemble_fixed_0.7"]
    assert metrics["macro"] == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3}
    assert metrics["pooled"]["true_positives"] == 2
    assert metrics["pooled"]["total_predicted"] == 2
    assert metrics["pooled"]["total_ground_truth"] == 4
    assert (
        report["configurations"]["ensemble_shared_threshold_loo"]["macro"][
            "f1"
        ]
        == 1
    )
    for comparison in report["comparisons"]["paired_tests"]:
        assert comparison["pairs"] == 2
        assert comparison["p_holm"] >= comparison["p_value"]
    assert evaluate(experiment, data, baseline) == report


def test_partial_historical_pair_coverage_is_not_silently_intersected(
    tmp_path,
):
    experiment, data, baseline, _, _ = collection(tmp_path)
    records = json.loads(baseline.read_text(encoding="utf-8"))
    records["string-equiv"].pop(next(iter(records["string-equiv"])))
    baseline.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="missing experiment pairs"):
        evaluate(experiment, data, baseline)


def test_reference_hash_prevents_evaluation_against_changed_ground_truth(
    tmp_path,
):
    experiment, data, baseline, manifest, rows = collection(tmp_path)
    reference_path = data / "alignments.json"
    manifest["input_hashes"] = {
        "alignments.json": hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
    }
    save_collection(experiment, manifest, rows)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference[0]["matches"].pop()
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(ValueError, match="input hashes differ"):
        evaluate(experiment, data, baseline)


def test_empty_candidate_sources_are_not_expected_http_requests(tmp_path):
    experiment, data, baseline, manifest, rows = collection(tmp_path)
    pair = manifest["pairs"][0]["pair"]
    manifest["pairs"][0]["candidates"]["s2"] = []
    rows = [
        row
        for row in rows
        if not (row["pair"] == pair and row["source_id"] == "s2")
    ]
    save_collection(experiment, manifest, rows)
    report = evaluate(experiment, data, baseline)
    assert report["requests"]["expected_requests"] == 9
    assert report["candidate_retrieval"]["total_gold"] == 4
    assert report["candidate_retrieval"]["recall"] == 0.75
    assert (
        report["configurations"]["concept_fixed_0.5"]["per_pair"][pair][
            "recall"
        ]
        == 0.5
    )
