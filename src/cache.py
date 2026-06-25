"""Unified result cache — single source of truth for the cache layout.

A cached result lives in `results/<approach>/<first>_<second>[_<mode>].json`,
where the two taxonomy names are always in canonical (alphabetical) order, so
a pair is stored exactly once no matter which of its sides a runner visits
first.  The `pair` field inside the JSON uses the same order (`a↔b`).

Usage:
    from src.cache import cache_path, load_all_cached, load_cached, pair_key
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path("results")

# Pair separator, both in canonical keys and in pair file names.
PAIR_SEPARATOR = "↔"

# Render-only subdirectory that is never part of the result cache.
VIZ_DIR = "viz"


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


def load_cached(approach: str, a: str, b: str, mode: str | None = None) -> dict | None:
    """Load one cached result, or `None` when the pair is not cached yet."""
    path = cache_path(approach, a, b, mode)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_all_cached() -> dict[str, dict]:
    """Load every cached result, keyed by approach and canonical pair key.

    Pure results are stored as `all_data[approach][pair]`, results carrying a
    mode are grouped as `all_data[approach][pair][mode]`.
    """
    all_data: dict[str, dict] = {}
    if not RESULTS_DIR.exists():
        return all_data
    for subdir in sorted(RESULTS_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name == VIZ_DIR:
            continue
        approach = subdir.name
        for f in sorted(subdir.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            pair = d["pair"]
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
