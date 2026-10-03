"""Evaluate a complete Jev collection offline without changing canonical caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np

from src.cache import assert_orientation, canonical_directions, pair_key
from src.matchers.jev import build_request, parse_response
from src.matchers.llm import ENSEMBLE_MODES, ensemble_candidates
from src.matching_rules import Candidate, greedy_injective, to_alignment
from src.metrics import evaluate_1to1
from src.runners.stats import (
    adjust_multiplicity,
    build_configurations,
    paired_test,
)
from src.taxonomy import Alignment, TaxonMatch

THRESHOLDS = tuple([i / 20 for i in range(20)] + [0.99, 1.0])
BASELINES = Path("results/consistency/alignments.json")


def _number(
    value: object, minimum: float = 0, maximum: float = math.inf
) -> bool:
    """Check real finite numeric values without accepting JSON booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and minimum <= value <= maximum
    )


def load_collection(
    experiment_dir: Path, data_dir: Path
) -> tuple[dict, dict, list[dict], dict]:
    """Load and reject incomplete, ambiguous, or misoriented collections."""
    manifest = json.loads(
        (experiment_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if "input_hashes" in manifest:
        actual_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(data_dir.glob("*.json"))
        }
        if manifest["input_hashes"] != actual_hashes:
            raise ValueError(
                "Processed input hashes differ from the frozen manifest"
            )
    reference = json.loads(
        (data_dir / "alignments.json").read_text(encoding="utf-8")
    )
    canonical = canonical_directions(reference)
    if len(canonical) != len(reference):
        raise ValueError("Duplicate reference pair")
    gold = {
        pair_key(r["source"], r["target"]): Alignment(
            source=r["source"],
            target=r["target"],
            matches=[
                TaxonMatch(m["source_id"], m["target_id"])
                for m in r["matches"]
            ],
        )
        for r in reference
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("model") != "typesafe/jev-1.13"
    ):
        raise ValueError("Unsupported manifest schema or model")
    if manifest.get("modes") != list(ENSEMBLE_MODES):
        raise ValueError("Manifest must contain the three canonical modes")
    if manifest.get("thresholds") != list(THRESHOLDS):
        raise ValueError(
            "Manifest threshold grid differs from the experiment contract"
        )
    pairs = {}
    expected = {}
    for spec in manifest["pairs"]:
        key = spec["pair"]
        if key in pairs or key not in canonical:
            raise ValueError(f"Duplicate or unknown manifest pair: {key}")
        assert_orientation(spec, *canonical[key])
        if pair_key(spec["source"], spec["target"]) != key:
            raise ValueError(f"Noncanonical pair key: {key}")
        candidates = spec["candidates"]
        for mode in ENSEMBLE_MODES:
            descriptions = spec["descriptions"][mode]
            if set(descriptions["source"]) != set(candidates):
                raise ValueError(
                    f"Source candidate coverage mismatch: {key}/{mode}"
                )
            for source_id, items in candidates.items():
                targets = [item["target_id"] for item in items]
                if len(set(targets)) != len(targets) or not set(
                    targets
                ) <= set(descriptions["target"]):
                    raise ValueError(
                        f"Invalid candidate targets: {key}/{source_id}"
                    )
                if any(
                    not _number(item["retrieval_score"], -math.inf)
                    for item in items
                ):
                    raise ValueError(
                        f"Invalid retrieval score: {key}/{source_id}"
                    )
                if targets:
                    request = build_request(
                        descriptions["source"][source_id],
                        [
                            descriptions["target"][target_id]
                            for target_id in targets
                        ],
                    )
                    expected[f"{key}|{mode}|{source_id}"] = (
                        key,
                        mode,
                        source_id,
                        targets,
                        request,
                    )
        pairs[key] = spec
    if set(pairs) != set(gold) or len(pairs) < 2:
        raise ValueError(
            "Manifest must cover every reference pair and at least two pairs for LOO"
        )
    rows = []
    successes = {}
    with (experiment_dir / "requests.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key")
            if key not in expected:
                raise ValueError(
                    f"Unexpected request at line {line_number}: {key}"
                )
            pair, mode, source_id, targets, request = expected[key]
            if (row.get("pair"), row.get("mode"), row.get("source_id")) != (
                pair,
                mode,
                source_id,
            ):
                raise ValueError(f"Request identity mismatch: {key}")
            assert_orientation(row, *canonical[pair])
            if row.get("request") != request:
                raise ValueError(
                    f"Actual request payload differs from frozen descriptions: {key}"
                )
            if "error" not in row:
                raise ValueError(f"Missing request validation outcome: {key}")
            rows.append(row)
            if row.get("error") is not None:
                if row.get("probabilities"):
                    raise ValueError(
                        f"Failed request contains validated probabilities: {key}"
                    )
                continue
            if key in successes:
                raise ValueError(f"Duplicate successful request: {key}")
            if (
                not isinstance(row.get("response"), dict)
                or not isinstance(row.get("status_code"), int)
                or not 200 <= row["status_code"] < 300
            ):
                raise ValueError(f"Invalid successful response: {key}")
            if not _number(row.get("latency_seconds")):
                raise ValueError(f"Invalid successful request latency: {key}")
            probabilities = row.get("probabilities")
            if not isinstance(probabilities, list):
                raise ValueError(f"Missing successful probabilities: {key}")
            observed = []
            for item in probabilities:
                if item.get("source_id") != source_id or not _number(
                    item.get("probability"), maximum=1
                ):
                    raise ValueError(f"Invalid candidate probability: {key}")
                observed.append(item.get("target_id"))
            if len(observed) != len(set(observed)) or set(observed) != set(
                targets
            ):
                raise ValueError(
                    f"Probability candidate coverage mismatch: {key}"
                )
            parsed = parse_response(row["response"], source_id, targets)
            if probabilities != parsed:
                raise ValueError(
                    f"Stored probabilities differ from the raw Jev response: {key}"
                )
            successes[key] = row
    missing = set(expected) - set(successes)
    if missing:
        raise ValueError(
            f"Incomplete collection: {len(missing)} successful requests missing; first: {min(missing)}"
        )
    return manifest, gold, rows, successes


def select_predictions(
    candidates: list[Candidate], threshold: float
) -> list[Candidate]:
    """Apply the legacy strict cut before global one-to-one selection."""
    return greedy_injective(c for c in candidates if c[2] > threshold)


def score_predictions(
    predictions: list[Candidate], gold: Alignment, **selection: object
) -> dict:
    """Keep exact counts and the full injective prediction set."""
    metrics = asdict(
        evaluate_1to1(
            to_alignment(predictions, gold.source, gold.target), gold
        )
    )
    return {
        **metrics,
        **selection,
        "predictions": [
            {
                "source_id": source,
                "target_id": target,
                "probability": probability,
            }
            for source, target, probability in predictions
        ],
    }


def summarize(per_pair: dict[str, dict]) -> dict:
    """Report equally weighted pair means and independently pooled counts."""
    tp = sum(record["true_positives"] for record in per_pair.values())
    predicted = sum(record["total_predicted"] for record in per_pair.values())
    gold = sum(record["total_ground_truth"] for record in per_pair.values())
    return {
        "macro": {
            field: mean(record[field] for record in per_pair.values())
            for field in ("precision", "recall", "f1")
        },
        "pooled": {
            "true_positives": tp,
            "total_predicted": predicted,
            "total_ground_truth": gold,
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / gold if gold else 0.0,
            "f1": 2 * tp / (predicted + gold) if predicted + gold else 0.0,
        },
        "per_pair": per_pair,
    }


def choose_threshold(sweep: dict[str, dict[float, dict]], held: str) -> float:
    """Choose the lowest tied threshold using only other pairs' mean F1."""
    others = [pair for pair in sorted(sweep) if pair != held]
    if not others:
        raise ValueError("LOO threshold selection needs another pair")
    return max(
        THRESHOLDS,
        key=lambda threshold: (
            mean(sweep[pair][threshold]["f1"] for pair in others),
            -threshold,
        ),
    )


def calibration(
    candidates: dict[str, list[Candidate]], gold: dict[str, Alignment]
) -> dict:
    """Describe retrieved-pair probabilities against an incomplete reference."""
    bins = [
        {
            "lower": i / 10,
            "upper": (i + 1) / 10,
            "count": 0,
            "positives": 0,
            "probability_sum": 0.0,
        }
        for i in range(10)
    ]
    squared_error = positives = count = 0
    for pair in sorted(candidates):
        reference = gold[pair].as_pairs()
        for source, target, probability in candidates[pair]:
            label = int((source, target) in reference)
            count += 1
            positives += label
            squared_error += (probability - label) ** 2
            bucket = bins[min(int(probability * 10), 9)]
            bucket["count"] += 1
            bucket["positives"] += label
            bucket["probability_sum"] += probability
    ece = 0.0
    for bucket in bins:
        n = bucket["count"]
        bucket["mean_probability"] = (
            bucket.pop("probability_sum") / n if n else None
        )
        bucket["positive_fraction"] = bucket["positives"] / n if n else None
        if n:
            ece += n * abs(
                bucket["mean_probability"] - bucket["positive_fraction"]
            )
    return {
        "population": "All retrieved candidate pairs, before thresholding or injectivity; unlisted reference pairs are treated as negatives, although the reference is incomplete.",
        "candidate_pairs": count,
        "positives": positives,
        "positive_prevalence": positives / count if count else None,
        "always_negative_brier": positives / count if count else None,
        "brier": squared_error / count if count else None,
        "ece_10_bins": ece / count if count else None,
        "bin_convention": "[lower, upper), with probability 1 included in the last bin.",
        "reliability_bins": bins,
    }


def request_accounting(rows: list[dict], successes: dict[str, dict]) -> dict:
    """Account for observed usage on every attempt without inventing missing values."""

    def usage_block(attempts: list[dict]) -> dict:
        block = {
            "attempts": len(attempts),
            "successful_requests": sum(
                row.get("error") is None for row in attempts
            ),
        }
        for output, names in (
            ("cost", ("cost",)),
            ("input_tokens", ("input_tokens", "prompt_tokens")),
            ("output_tokens", ("output_tokens", "completion_tokens")),
        ):
            values = []
            for row in attempts:
                response = row.get("response") or {}
                usage = response.get("usage") or {}
                value = next(
                    (usage[name] for name in names if name in usage), None
                )
                if _number(value):
                    values.append(value)
            block[output] = sum(values) if values else None
            block[f"{output}_observed_attempts"] = len(values)
            block[f"{output}_missing_attempts"] = len(attempts) - len(values)
        return block

    latency = sorted(row["latency_seconds"] for row in successes.values())

    def available(field: str) -> list[str]:
        """List only metadata actually returned by the provider."""
        return sorted(
            {
                str(row["response"][field])
                for row in rows
                if isinstance(row.get("response"), dict)
                and row["response"].get(field) is not None
            }
        )

    errors = [row for row in rows if row.get("error") is not None]
    return {
        **usage_block(rows),
        "expected_requests": len(successes),
        "error_attempts": len(errors),
        "error_attempt_usage": usage_block(errors),
        "per_mode": {
            mode: usage_block([row for row in rows if row["mode"] == mode])
            for mode in ENSEMBLE_MODES
        },
        "successful_request_latency_seconds": {
            "count": len(latency),
            "min": min(latency) if latency else None,
            "mean": mean(latency) if latency else None,
            "max": max(latency) if latency else None,
            **{
                name: float(np.percentile(latency, q)) if latency else None
                for name, q in (("p50", 50), ("p90", 90), ("p95", 95))
            },
        },
        "models": available("model"),
        "providers": available("provider"),
        "response_ids": available("id"),
        "latency_scope": "Successful actual HTTP requests, not taxonomy-pair wall time; canonical cached api_time is not directly comparable.",
    }


def compare_baselines(
    configurations: dict[str, dict], path: Path, pairs: list[str]
) -> dict:
    """Compare matched-pair F1 vectors using the existing bootstrap and Holm rules."""
    if not path.exists():
        raise ValueError(f"Missing canonical comparison artifact: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    historical = build_configurations(records)
    names = [
        "llm-hybrid raw",
        "llm-hybrid ensemble",
        "string-equiv raw",
        "embedding-ruRoberta-large thr",
    ]
    for mode in ENSEMBLE_MODES:
        historical[f"llm-hybrid {mode}"] = {
            row["pair"]: row["baseline"]["f1"]
            for row in records["llm-hybrid"].values()
            if row.get("mode") == mode
        }
        names.append(f"llm-hybrid {mode}")
    baseline_vectors = {}
    for name in names:
        if name not in historical or not set(pairs) <= set(historical[name]):
            raise ValueError(
                f"Canonical comparison is missing experiment pairs: {name}"
            )
        baseline_vectors[name] = {
            pair: historical[name][pair] for pair in pairs
        }
    comparison_pairs = [
        (configuration, baseline)
        for configuration in (
            "ensemble_shared_threshold_loo",
            "loo_mode_and_threshold_selection",
        )
        for baseline in (
            "llm-hybrid ensemble",
            "string-equiv raw",
            "embedding-ruRoberta-large thr",
        )
    ] + [
        (f"{mode}_fixed_0.7", f"llm-hybrid {mode}") for mode in ENSEMBLE_MODES
    ]
    comparisons = []
    for name, baseline in comparison_pairs:
        first = {
            pair: configurations[name]["per_pair"][pair]["f1"]
            for pair in pairs
        }
        result = paired_test(first, baseline_vectors[baseline])
        if not math.isfinite(result["wilcoxon_statistic"]):
            result["wilcoxon_statistic"] = None
        comparisons.append({"first": name, "second": baseline, **result})
    adjust_multiplicity(comparisons)
    return {
        "artifact": str(path),
        "pairs": pairs,
        "baseline_f1": {
            name: {"mean": mean(vector.values()), "per_pair": vector}
            for name, vector in baseline_vectors.items()
        },
        "paired_tests": comparisons,
        "multiplicity_family": "All reported Jev-vs-baseline comparisons jointly adjusted by Holm and Bonferroni.",
        "provenance_limits": [
            "Historical LLM caches are non-contemporaneous and lack full pre-injective judgments and fallback provenance; this is not a perfectly controlled probability comparison.",
            "Historical calibrated ruRoberta uses its existing leave-one-pair-out calibration; its embedding model was chosen on these benchmark pairs.",
            "No historical probability calibration or single-request latency comparison is inferred from final mappings or cached pair-wall api_time.",
            "The same retrieval implementation/settings/texts are used, but historical candidate lists and omissions were not persisted, so identical old candidate populations cannot be established.",
            "The classifier interface and instructions are substituted: historical 'same entity' prompting versus explicit equivalent classes rather than related/subclass concepts; no architecture-only causal attribution is supported.",
        ],
    }


def evaluate(
    experiment_dir: Path, data_dir: Path, baseline_path: Path = BASELINES
) -> dict:
    """Recompute the complete experiment deterministically without network access."""
    manifest, gold, rows, successes = load_collection(experiment_dir, data_dir)
    pairs = sorted(gold)
    candidates = {
        mode: {pair: [] for pair in pairs} for mode in ENSEMBLE_MODES
    }
    for key in sorted(successes):
        row = successes[key]
        candidates[row["mode"]][row["pair"]].extend(
            (item["source_id"], item["target_id"], item["probability"])
            for item in row["probabilities"]
        )
    sweeps = {
        mode: {
            pair: {
                threshold: score_predictions(
                    select_predictions(candidates[mode][pair], threshold),
                    gold[pair],
                    threshold=threshold,
                )
                for threshold in THRESHOLDS
            }
            for pair in pairs
        }
        for mode in ENSEMBLE_MODES
    }
    configurations = {}
    for mode in ENSEMBLE_MODES:
        for threshold in (0.7, 0.5):
            configurations[f"{mode}_fixed_{threshold}"] = summarize(
                {pair: sweeps[mode][pair][threshold] for pair in pairs}
            )
        configurations[f"{mode}_loo"] = summarize(
            {
                pair: sweeps[mode][pair][choose_threshold(sweeps[mode], pair)]
                for pair in pairs
            }
        )
    ensemble_sweep = {pair: {} for pair in pairs}
    for pair in pairs:
        for threshold in THRESHOLDS:
            selected = {
                mode: select_predictions(candidates[mode][pair], threshold)
                for mode in ENSEMBLE_MODES
            }
            ensemble_sweep[pair][threshold] = score_predictions(
                ensemble_candidates(selected), gold[pair], threshold=threshold
            )
    configurations["ensemble_fixed_0.7"] = summarize(
        {pair: ensemble_sweep[pair][0.7] for pair in pairs}
    )
    configurations["ensemble_shared_threshold_loo"] = summarize(
        {
            pair: ensemble_sweep[pair][choose_threshold(ensemble_sweep, pair)]
            for pair in pairs
        }
    )
    for variant in ("fixed_0.7", "fixed_0.5", "loo"):
        per_pair = {}
        for pair in pairs:
            blocks = [
                configurations[f"{mode}_{variant}"]["per_pair"][pair]
                for mode in ENSEMBLE_MODES
            ]
            per_pair[pair] = {
                field: mean(block[field] for block in blocks)
                for field in ("precision", "recall", "f1")
            }
        configurations[f"mean_over_modes_{variant}"] = {
            "macro": {
                field: mean(row[field] for row in per_pair.values())
                for field in ("precision", "recall", "f1")
            },
            "per_pair": per_pair,
            "pooled_over_mode_decisions": summarize(
                {
                    f"{pair}|{mode}": configurations[f"{mode}_{variant}"][
                        "per_pair"
                    ][pair]
                    for pair in pairs
                    for mode in ENSEMBLE_MODES
                }
            )["pooled"],
            "population": "Representation average, not a single mapping; pooled mode decisions repeat each gold pair three times.",
        }
    selected_modes = {}
    for held in pairs:
        training_thresholds = {
            mode: choose_threshold(sweeps[mode], held)
            for mode in ENSEMBLE_MODES
        }
        selected_mode = max(
            ENSEMBLE_MODES,
            key=lambda mode: (
                mean(
                    sweeps[mode][pair][training_thresholds[mode]]["f1"]
                    for pair in pairs
                    if pair != held
                ),
                -ENSEMBLE_MODES.index(mode),
            ),
        )
        selected_modes[held] = {
            **sweeps[selected_mode][held][training_thresholds[selected_mode]],
            "mode": selected_mode,
        }
    configurations["loo_mode_and_threshold_selection"] = summarize(
        selected_modes
    )
    retrieval_per_pair = {}
    specs = {spec["pair"]: spec for spec in manifest["pairs"]}
    for pair in pairs:
        retrieved = {
            (source, item["target_id"])
            for source, items in specs[pair]["candidates"].items()
            for item in items
        }
        reference = gold[pair].as_pairs()
        retrieval_per_pair[pair] = {
            "retrieved_gold": len(retrieved & reference),
            "total_gold": len(reference),
            "candidate_pairs": len(retrieved),
            "source_nodes": len(specs[pair]["candidates"]),
            "recall": len(retrieved & reference) / len(reference)
            if reference
            else 0.0,
        }
    retrieved_gold = sum(
        row["retrieved_gold"] for row in retrieval_per_pair.values()
    )
    total_gold = sum(row["total_gold"] for row in retrieval_per_pair.values())
    source_nodes = sum(
        row["source_nodes"] for row in retrieval_per_pair.values()
    )
    return {
        "schema_version": 1,
        "model": manifest["model"],
        "experiment_manifest": manifest,
        "evaluation_protocol": {
            "thresholds": list(THRESHOLDS),
            "comparison": "probability > threshold",
            "injectivity": "Threshold every candidate first, then shared global greedy_injective.",
            "loo": "Only other pairs' macro F1 chooses thresholds/modes; threshold ties choose lowest, mode ties use manifest mode order.",
            "ensemble": ">=2 of 3 per-mode injective predictions, through ensemble_candidates; calibrated headline chooses one shared threshold by other pairs' ensemble F1.",
            "reference": str(data_dir / "alignments.json"),
        },
        "configurations": configurations,
        "candidate_retrieval": {
            "retrieved_gold": retrieved_gold,
            "total_gold": total_gold,
            "recall": retrieved_gold / total_gold if total_gold else 0.0,
            "mean_candidate_count": sum(
                row["candidate_pairs"] for row in retrieval_per_pair.values()
            )
            / source_nodes
            if source_nodes
            else 0.0,
            "source_nodes": source_nodes,
            "per_pair": retrieval_per_pair,
        },
        "calibration": {
            mode: calibration(candidates[mode], gold)
            for mode in ENSEMBLE_MODES
        },
        "requests": request_accounting(rows, successes),
        "comparisons": compare_baselines(configurations, baseline_path, pairs),
    }


def main() -> None:
    """Write a complete offline summary only after all validation succeeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path("experiments/jev")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/oaei")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/jev-experiment.json")
    )
    args = parser.parse_args()
    report = evaluate(args.experiment_dir, args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
