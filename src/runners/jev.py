"""Freeze hybrid retrieval and collect real Jev probabilities without reference leakage.

Prepare is offline; collect resumes only missing successful requests. Every HTTP
attempt is journalled, including errors. No failed call becomes an embedding match.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from src.cache import canonical_directions, pair_key
from src.data_loader import load_alignments, load_taxonomies
from src.matchers.hybrid import HybridLLMMatcher
from src.matchers.jev import (
    CONTEXT_TOKENS,
    CRITERIA,
    ENDPOINT,
    INPUT_USD_PER_TOKEN,
    INSTRUCTIONS,
    MAX_REQUEST_COST,
    MODEL,
    JevClient,
    build_request,
    parse_response,
)
from src.matchers.llm import ENSEMBLE_MODES

DEFAULT_EXPERIMENT = Path("experiments/jev")
DEFAULT_DATA = Path("data/processed/oaei")


def input_hashes(data_dir: Path) -> dict[str, str]:
    """Fingerprint all processed inputs, including the offline reference file."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(data_dir.glob("*.json"))
    }


def prepare(experiment_dir: Path, data_dir: Path) -> dict:
    """Freeze source orientation, candidates, representations, and input provenance."""
    manifest_path = experiment_dir / "manifest.json"
    hashes = input_hashes(data_dir)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["input_hashes"] != hashes:
            raise ValueError(
                "Processed inputs changed; do not resume this experiment"
            )
        return manifest
    taxonomies = load_taxonomies(data_dir)
    directions = canonical_directions(
        load_alignments(data_dir / "alignments.json")
    )
    retriever = HybridLLMMatcher(top_k=5, bm25_k=5, total_k=10)
    pairs = []
    for index, (source_name, target_name) in enumerate(directions.values(), 1):
        source, target = taxonomies[source_name], taxonomies[target_name]
        candidates = retriever.retrieve_candidates(source, target)
        pairs.append(
            {
                "pair": pair_key(source_name, target_name),
                "source": source_name,
                "target": target_name,
                "candidates": {
                    sid: [
                        {"target_id": tid, "retrieval_score": score}
                        for tid, score in rows
                    ]
                    for sid, rows in sorted(candidates.items())
                },
                "descriptions": {
                    mode: {
                        "source": {
                            sid: retriever._describe_concept(source, sid, mode)
                            for sid in sorted(source.nodes)
                        },
                        "target": {
                            tid: retriever._describe_concept(target, tid, mode)
                            for tid in sorted(target.nodes)
                        },
                    }
                    for mode in ENSEMBLE_MODES
                },
            }
        )
        print(
            f"[{index}/{len(directions)}] Prepared {pairs[-1]['pair']}",
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "model": MODEL,
        "endpoint": ENDPOINT,
        "pricing": {
            "input_usd_per_token": INPUT_USD_PER_TOKEN,
            "output_usd_per_token": 0.0,
            "context_tokens": CONTEXT_TOKENS,
            "source": "https://openrouter.ai/typesafe/jev-1.13",
        },
        "retriever": "HybridLLMMatcher",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "retrieval_description_mode": "name_only",
        "top_k": 5,
        "bm25_k": 5,
        "total_k": 10,
        "modes": list(ENSEMBLE_MODES),
        "thresholds": [round(i / 20, 2) for i in range(20)] + [0.99, 1.0],
        "instructions": INSTRUCTIONS,
        "criteria": CRITERIA,
        "input_hashes": hashes,
        "code_hashes": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "src/matchers/hybrid.py",
                "src/matchers/llm.py",
                "src/matchers/embedding.py",
                "src/matchers/jev.py",
                "src/matching_rules.py",
                "src/runners/jev.py",
            )
        },
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "torch",
                "sentence-transformers",
                "rank-bm25",
                "httpx",
            )
        },
        "pairs": pairs,
    }
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def expected_requests(manifest: dict) -> dict[str, dict]:
    """Build the complete request set solely from the frozen manifest."""
    if (
        manifest["model"] != MODEL
        or manifest["instructions"] != INSTRUCTIONS
        or manifest["criteria"] != CRITERIA
    ):
        raise ValueError(
            "Classifier contract changed; cannot resume this manifest"
        )
    tasks = {}
    for pair in manifest["pairs"]:
        for mode in manifest["modes"]:
            descriptions = pair["descriptions"][mode]
            for sid, candidates in pair["candidates"].items():
                if not candidates:
                    continue
                ids = [row["target_id"] for row in candidates]
                key = f"{pair['pair']}|{mode}|{sid}"
                tasks[key] = {
                    "key": key,
                    "pair": pair["pair"],
                    "source": pair["source"],
                    "target": pair["target"],
                    "mode": mode,
                    "source_id": sid,
                    "candidate_ids": ids,
                    "request": build_request(
                        descriptions["source"][sid],
                        [descriptions["target"][tid] for tid in ids],
                    ),
                }
    return tasks


def budget_charge(row: dict) -> float:
    """Reserve full-context cost when an interrupted attempt has unknown billing."""
    response = row.get("response")
    usage = response.get("usage") if isinstance(response, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(cost)
        and cost >= 0
    ):
        return float(cost)
    return MAX_REQUEST_COST


def load_journal(path: Path, tasks: dict[str, dict]) -> tuple[set[str], float]:
    """Validate previous successful responses and prevent mismatched cache reuse."""
    completed = set()
    charged = 0.0
    if not path.exists():
        return completed, charged
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            key = row["key"]
            if key not in tasks or row["request"] != tasks[key]["request"]:
                raise ValueError(f"Journal request mismatch at line {number}")
            charged += budget_charge(row)
            if row.get("error") is None:
                task = tasks[key]
                expected = parse_response(
                    row["response"], task["source_id"], task["candidate_ids"]
                )
                if (
                    row.get("status_code") != 200
                    or row.get("probabilities") != expected
                    or key in completed
                ):
                    raise ValueError(
                        f"Invalid or duplicate journal success at line {number}"
                    )
                completed.add(key)
    return completed, charged


def collect(
    experiment_dir: Path,
    manifest: dict,
    max_cost: float,
    workers: int,
    limit: int | None,
) -> None:
    """Collect missing requests in bounded batches, stopping on any API error."""
    tasks = expected_requests(manifest)
    journal_path = experiment_dir / "requests.jsonl"
    completed, charged = load_journal(journal_path, tasks)
    pending = [task for key, task in tasks.items() if key not in completed]
    if limit is not None:
        pending = pending[:limit]
    print(
        f"{len(completed)}/{len(tasks)} cached; {len(pending)} pending; budget ${max_cost:.2f}",
        flush=True,
    )
    if not pending:
        return
    client = JevClient()
    try:
        with (
            journal_path.open("a", encoding="utf-8") as stream,
            ThreadPoolExecutor(max_workers=workers) as executor,
        ):
            for start in range(0, len(pending), workers):
                batch = pending[start : start + workers]
                if charged + MAX_REQUEST_COST * len(batch) > max_cost:
                    raise RuntimeError(
                        f"Cost ceiling reached: reserved charges ${charged:.6f}, budget ${max_cost:.2f}"
                    )
                futures = {
                    executor.submit(
                        client.request,
                        task["request"],
                        task["source_id"],
                        task["candidate_ids"],
                    ): task
                    for task in batch
                }
                errors = []
                for future in as_completed(futures):
                    task = futures[future]
                    row = {
                        key: value
                        for key, value in task.items()
                        if key != "candidate_ids"
                    }
                    row.update(future.result())
                    stream.write(
                        json.dumps(
                            row, ensure_ascii=False, separators=(",", ":")
                        )
                        + "\n"
                    )
                    stream.flush()
                    charged += budget_charge(row)
                    if row["error"]:
                        errors.append(f"{row['key']}: {row['error']}")
                    else:
                        completed.add(row["key"])
                    print(
                        f"[{len(completed)}/{len(tasks)}] {row['pair']} {row['mode']} "
                        f"{row['source_id']} {row['latency_seconds']:.3f}s charges=${charged:.6f}",
                        flush=True,
                    )
                if errors:
                    raise RuntimeError(
                        "API errors saved; no fallback used.\n"
                        + "\n".join(errors)
                    )
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect"))
    parser.add_argument(
        "--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, help="Maximum new requests; use for a paid pilot."
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.max_cost)
        or args.max_cost <= 0
        or not 1 <= args.workers <= 8
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error(
            "Require positive finite cost/limit and one to eight workers"
        )
    manifest = prepare(args.experiment_dir, args.data_dir)
    tasks = expected_requests(manifest)
    payload_bytes = sum(
        len(json.dumps(task["request"], ensure_ascii=False).encode())
        for task in tasks.values()
    )
    print(
        f"Frozen {len(manifest['pairs'])} pairs, {len(tasks)} requests; "
        f"payload-byte upper estimate ${payload_bytes * INPUT_USD_PER_TOKEN:.4f}; "
        f"full-context reserve ${len(tasks) * MAX_REQUEST_COST:.4f}",
        flush=True,
    )
    if args.action == "collect":
        collect(
            args.experiment_dir,
            manifest,
            args.max_cost,
            args.workers,
            args.limit,
        )


if __name__ == "__main__":
    main()
