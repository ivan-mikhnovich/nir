"""Unified result cache — single source of truth for the cache layout.

A cached result lives in `results/<approach>/<first>_<second>[_<mode>].json`,
where the two taxonomy names are always in canonical (alphabetical) order, so
a pair is stored exactly once no matter which of its sides a runner visits
first.  The `pair` field inside the JSON uses the same order (`a↔b`).

The file name carries no direction: the record's own `source`/`target` fields
do, and the archive legitimately holds both orientations for the same pair
across approaches (finding 2.9).  A consumer that evaluates or combines records
across approaches must therefore (a) rebuild every record from its *own*
`source`/`target` and (b) check the orientation against one convention.  The
convention is the direction in which the reference-alignment loader enumerates
the pair (`canonical_directions`).

Usage:
    from src.cache import cache_path, load_all_cached, load_cached, pair_key

    alignments_raw = load_alignments("data/processed/oaei/alignments.json")
    canonical = canonical_directions(alignments_raw)
    data = load_all_cached(expect=canonical)   # raises on a flipped record
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path("results")

# Pair separator, both in canonical keys and in pair file names.
PAIR_SEPARATOR = "↔"

# Subdirectories of `results/` that hold derived artefacts, not matcher results.
VIZ_DIR = "viz"
CONSISTENCY_DIR = "consistency"
NON_RESULT_DIRS = frozenset({VIZ_DIR, CONSISTENCY_DIR})


def pair_key(a: str, b: str) -> str:
    """Return the canonical pair key, with both names sorted alphabetically."""
    return PAIR_SEPARATOR.join(sorted([a, b]))


def cache_path(approach: str, a: str, b: str, mode: str | None = None) -> Path:
    """Return the canonical cache path of a pair, creating the approach directory."""
    base = RESULTS_DIR / approach
    base.mkdir(parents=True, exist_ok=True)
    suffix = f"_{mode}" if mode else ""
    first, second = sorted([a, b])
    return base / f"{first}_{second}{suffix}.json"


def orientation(record: dict) -> tuple[str, str]:
    """Return the (source, target) direction a record was computed in."""
    return record["source"], record["target"]


def canonical_directions(alignments_raw: list[dict]) -> dict[str, tuple[str, str]]:
    """Derive the canonical direction of every pair from the reference alignments.

    The reference-alignment loader (`load_alignments`) enumerates each pair once
    and in a fixed order; that order is the canonical direction of the pair, so
    its `source`/`target` fields are the answer, and every cache is brought to
    it rather than the other way round (contract C6).
    """
    return {
        pair_key(record["source"], record["target"]): (record["source"], record["target"])
        for record in alignments_raw
    }


def assert_orientation(record: dict, source: str, target: str) -> None:
    """Raise `ValueError` when a record was not computed `source → target`."""
    actual = orientation(record)
    if actual != (source, target):
        raise ValueError(
            f"cache record for {pair_key(source, target)} is stored in the reverse "
            f"direction {actual[0]} → {actual[1]}, expected {source} → {target}"
        )


def record_alignment(record: dict):
    """Rebuild the predicted alignment from the record's own stored direction.

    Used instead of the caller's loop variables: a record in the reverse
    direction would otherwise be relabelled silently (findings 2.9, 2.10).
    Returns `None` when the record carries no match list.
    """
    pairs = record.get("match_pairs")
    if not pairs:
        return None
    from src.taxonomy import Alignment, TaxonMatch

    return Alignment(
        source=record["source"],
        target=record["target"],
        matches=[
            TaxonMatch(
                source_id=p["source_id"],
                target_id=p["target_id"],
                confidence=p.get("confidence", 1.0),
            )
            for p in pairs
        ],
    )


def load_cached(
    approach: str,
    a: str,
    b: str,
    mode: str | None = None,
    expect: tuple[str, str] | None = None,
) -> dict | None:
    """Load one cached result, or `None` when the pair is not cached yet.

    With `expect=(source, target)` the record is checked against that direction
    and a wrongly oriented record raises `ValueError` instead of being used.
    """
    path = cache_path(approach, a, b, mode)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        if expect is not None:
            assert_orientation(record, *expect)
        return record
    return None


def load_all_cached(expect: dict[str, tuple[str, str]] | None = None) -> dict[str, dict]:
    """Load every cached result, keyed by approach and canonical pair key.

    Pure results are stored as `all_data[approach][pair]`, results carrying a
    mode are grouped as `all_data[approach][pair][mode]`.  Records are returned
    as stored, so their direction is always available.

    With `expect` (from `canonical_directions`) every record is checked against
    the canonical direction of its pair and a flipped record raises
    `ValueError`; without it the archive loads as-is (old files keep working).
    Records that carry no `source`/`target` (e.g. the GNN aggregates) cannot be
    checked and are loaded unchanged.
    """
    all_data: dict[str, dict] = {}
    if not RESULTS_DIR.exists():
        return all_data
    for subdir in sorted(RESULTS_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name in NON_RESULT_DIRS:
            continue
        approach = subdir.name
        for f in sorted(subdir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            pair = d["pair"]
            if expect is not None and "source" in d and pair in expect:
                assert_orientation(d, *expect[pair])
            mode = d.get("mode")
            if approach not in all_data:
                all_data[approach] = {}
            if mode:
                if pair not in all_data[approach]:
                    all_data[approach][pair] = {}
                all_data[approach][pair][mode] = d
            else:
                all_data[approach][pair] = d
    return all_data


def misoriented_records(
    canonical: dict[str, tuple[str, str]],
) -> dict[str, list[dict]]:
    """Report cached records whose direction differs from the canonical one.

    Returns `approach → [{"pair", "source", "target", "expected_source",
    "expected_target", "path"}]` for every flipped record on disk.  Records
    without `source`/`target` (the GNN aggregates) carry no direction and are
    skipped.
    """
    out: dict[str, list[dict]] = {}
    if not RESULTS_DIR.exists():
        return out
    for subdir in sorted(RESULTS_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name in NON_RESULT_DIRS:
            continue
        for f in sorted(subdir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                record = json.load(fh)
            if "source" not in record or "target" not in record:
                continue
            pair = record.get("pair", pair_key(record["source"], record["target"]))
            expected = canonical.get(pair)
            if expected is None or orientation(record) == expected:
                continue
            out.setdefault(subdir.name, []).append({
                "pair": pair,
                "source": record["source"],
                "target": record["target"],
                "expected_source": expected[0],
                "expected_target": expected[1],
                "path": str(f),
            })
    return out
