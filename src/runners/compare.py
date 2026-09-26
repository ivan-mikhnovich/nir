"""Compare all taxonomy matching approaches — unified table.

Runs Embedding, StringEquiv, LLM/gpt-4.1-mini, LLM/deepseek-v4-flash and the
BM25/hybrid retrievers over the OAEI Conference track.  By default six sample
pairs are run; the cache itself covers all 21 track pairs (`--all-pairs`).
Results are cached to `results/` to avoid re-running expensive LLM calls.

Pair direction.  The canonical direction of a pair is the direction of the
reference alignment (`alignments.json`), not the alphabetical one; the cache
key stays order-free.  A cached alignment is rebuilt from the record's own
`source`/`target` fields and checked against the canonical direction.

LLM aggregations.  `mean_over_modes`, `loo_mode_selection` and `ensemble` are
selection-free and are the numbers to publish; the per-pair maximum over the
representations is printed only as `oracle_max`/`(oracle)`, because it reads
the ground truth of the pair it is applied to.  The ensemble keeps a target
only when at least two of the three representations vote for it and reduces
the result to a 1:1 mapping (`src.matching_rules.greedy_injective`); it is
recomputed from the cached per-mode predictions and makes no API request.

Matchers are imported inside the runners, so `--table-only`, `--ensemble-only`
and `--dry-run` need neither a model nor a GPU.

Usage:
    uv run python -m src.runners.compare          # print table from cache.
    uv run python -m src.runners.compare --run     # run the cheap approaches, then print.
    uv run python -m src.runners.compare --approach embedding  # run specific approach.
    uv run python -m src.runners.compare --ensemble-only --dry-run  # no API, no writes.
    uv run python -m src.runners.compare --run --approach llm-deepseek \\
        --only-stale --force   # refresh the 11 cmt/ekaw pairs on DeepSeek direct.
"""

import argparse
import json
import time
from pathlib import Path

from src.cache import (
    cache_path,
    canonical_directions,
    load_all_cached,
    load_cached,
    pair_key,
    record_alignment,
)
from src.data_loader import find_gt, load_alignments, load_taxonomies
from src.matchers.llm import (
    ENSEMBLE_MIN_VOTES,
    ENSEMBLE_MODES,
    LLMMatcher,
    aggregate_mode_f1,
    ensemble_candidates,
)
from src.matching_rules import Candidate, to_alignment
from src.metrics import evaluate_1to1, MatchMetrics
from src.taxonomy import Alignment

LLM_MODES = list(ENSEMBLE_MODES)
LLM_SHORT = {"concept": "C", "concept-parent": "CP", "concept-children": "CC"}
LLM_APPROACHES = ["llm-bm25", "llm-deepseek", "llm-gpt", "llm-hybrid"]

# Taxonomies whose cached LLM results are stale after the ancestor-chain fix.
STALE_TAXONOMIES = ("cmt", "ekaw")

# Rule label stored with every ensemble record, so a recomputation can tell a
# record produced by the shared >=2-of-3 rule from an old plurality record.
ENSEMBLE_RULE = f">={ENSEMBLE_MIN_VOTES}-of-{len(ENSEMBLE_MODES)}+greedy_injective"

OAEI_SAMPLE_PAIRS = [
    ("cmt", "conference"),
    ("cmt", "confOf"),
    ("conference", "confOf"),
    ("cmt", "edas"),
    ("confOf", "ekaw"),
    ("conference", "ekaw"),
]


def _get_all_pairs(alignments_raw: list[dict]) -> list[tuple[str, str]]:
    """Return all pairs in the canonical direction of the reference alignments."""
    return [(a["source"], a["target"]) for a in alignments_raw if "source" in a]


def parse_pairs(spec: str, known: set[str]) -> list[tuple[str, str]]:
    """Parse a `--pairs` spec like `cmt-conference,ekaw-edas` in either direction."""
    pairs: list[tuple[str, str]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        for sep in ("-", ":", "↔", "/"):
            if sep in chunk:
                first, second = (p.strip() for p in chunk.split(sep, 1))
                break
        else:
            raise ValueError(f"Cannot parse pair {chunk!r}; use `source-target`.")
        if first not in known or second not in known:
            raise ValueError(f"Unknown taxonomy in pair {chunk!r}.")
        pairs.append((first, second))
    return pairs


def select_pairs(
    all_pairs: list[tuple[str, str]],
    only_stale: bool = False,
    spec: str | None = None,
    known: set[str] | None = None,
    canonical: dict[str, tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Select the pairs to run: explicit `spec` wins over `only_stale`.

    Every selection is brought to the canonical direction of the reference
    alignment, so `--pairs confOf-conference` runs the pair in the direction
    the cache and the ground truth use (contract C6).
    """
    if spec:
        selected = parse_pairs(spec, known or set())
    elif only_stale:
        selected = [(s, t) for s, t in all_pairs
                    if s in STALE_TAXONOMIES or t in STALE_TAXONOMIES]
    else:
        selected = list(all_pairs)
    if canonical:
        selected = [canonical.get(pair_key(s, t), (s, t)) for s, t in selected]
    return selected

# ── Cache ────────────────────────────────────────────────────────────


def save_results(approach: str, src_name: str, tgt_name: str, metrics: MatchMetrics,
                 extra: dict | None = None, mode: str | None = None,
                 match_pairs: list[dict] | None = None) -> None:
    data = {
        "approach": approach,
        "pair": pair_key(src_name, tgt_name),
        "source": src_name,
        "target": tgt_name,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "matches": metrics.true_positives,
        "total_predicted": metrics.total_predicted,
        "total_gt": metrics.total_ground_truth,
        **(extra or {}),
    }
    if mode:
        data["mode"] = mode
    if match_pairs is not None:
        data["match_pairs"] = match_pairs
    path = cache_path(approach, src_name, tgt_name, mode)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def target_collisions(pairs: list[Candidate]) -> int:
    """Count how many targets are used by more than one source."""
    seen: set[str] = set()
    extra = 0
    for _, tgt, _ in pairs:
        if tgt in seen:
            extra += 1
        seen.add(tgt)
    return extra


def plurality_candidates(per_mode: dict[str, list[Candidate]]) -> list[Candidate]:
    """Reproduce the OLD ensemble rule exactly: most-voted target, no 1:1.

    The rule took, per source node, the target with the most votes and broke
    ties by the first vote seen (`Counter.most_common` over the mode order).
    That tie-break is a property of the arrival order, not of the candidate
    set, which is why the rule was replaced; this function exists only to
    recover the published plurality numbers for the aggregation-rule ablation.
    It reproduces all 84 cached ensemble values exactly.
    """
    per_source: dict[str, dict[str, int]] = {}
    for mode in LLM_MODES:
        for src, tgt, _ in per_mode.get(mode, []):
            votes = per_source.setdefault(src, {})
            votes[tgt] = votes.get(tgt, 0) + 1

    out: list[Candidate] = []
    for src, votes in per_source.items():
        best = max(votes.values())
        best_tgt = next(tgt for tgt, count in votes.items() if count == best)
        out.append((src, best_tgt, best / len(LLM_MODES)))
    return sorted(out, key=lambda p: (-p[2], p[0], p[1]))


def cached_mode_candidates(
    approach: str,
    src_name: str,
    tgt_name: str,
) -> tuple[dict[str, list[Candidate]], dict[str, int | None]]:
    """Load the per-mode predictions of one cached pair, in canonical direction.

    Each record is read in its own stored direction and checked against the
    canonical one (contracts C3/C6), and its request count is carried along so
    the cost model stays measurable.
    """
    per_mode: dict[str, list[Candidate]] = {}
    mode_requests: dict[str, int | None] = {}
    for mode in LLM_MODES:
        record = load_cached(approach, src_name, tgt_name, mode, expect=(src_name, tgt_name))
        if record is None:
            raise FileNotFoundError(f"{approach} {src_name}↔{tgt_name} {mode}: no cached prediction")
        alignment = record_alignment(record)
        if alignment is None:
            raise FileNotFoundError(
                f"{approach} {src_name}↔{tgt_name} {mode}: cached record has no "
                f"reconstructible match_pairs (missing or empty)"
            )
        per_mode[mode] = [
            (m.source_id, m.target_id, m.confidence) for m in alignment.matches
        ]
        request_count = record.get("requests")
        mode_requests[mode] = int(request_count) if isinstance(request_count, (int, float)) else None
    return per_mode, mode_requests


def rebuild_ensemble(
    approach: str,
    src_name: str,
    tgt_name: str,
    gt: Alignment,
    write: bool = True,
) -> dict:
    """Recompute one ensemble record from cached per-mode predictions (no API).

    The shared `ensemble_candidates` rule is applied to the cached per-mode
    predictions and the result is stored with its `match_pairs`, the rule label
    and the request accounting.  Returns the before/after summary, including
    the legacy plurality F1 recomputed from the same data.
    """
    per_mode, mode_requests = cached_mode_candidates(approach, src_name, tgt_name)

    old = load_cached(approach, src_name, tgt_name, "ensemble")

    plurality = plurality_candidates(per_mode)
    plurality_metrics = evaluate_1to1(to_alignment(plurality, src_name, tgt_name), gt)

    selected = ensemble_candidates(per_mode)
    alignment = to_alignment(selected, src_name, tgt_name)
    metrics = evaluate_1to1(alignment, gt)

    old_map = {src: tgt for src, tgt, _ in plurality}
    new_map = {src: tgt for src, tgt, _ in selected}
    changed = sorted(
        (src, old_map.get(src, ""), new_map.get(src, ""))
        for src in set(old_map) | set(new_map)
        if old_map.get(src) != new_map.get(src)
    )

    if write:
        save_results(
            approach, src_name, tgt_name, metrics, mode="ensemble",
            extra={
                "rule": ENSEMBLE_RULE,
                "min_votes": ENSEMBLE_MIN_VOTES,
                "modes": LLM_MODES,
                "mode_requests": mode_requests,
                "requests": 0,
                "wall_time": 0.0,
            },
            match_pairs=[
                {"source_id": s, "target_id": t, "confidence": c} for s, t, c in selected
            ],
        )
    return {
        "approach": approach,
        "pair": pair_key(src_name, tgt_name),
        "old_f1": old["f1"] if old else None,
        "old_rule": old.get("rule") if old else None,
        "plurality_f1": plurality_metrics.f1,
        "new_f1": metrics.f1,
        "n_matches": len(selected),
        "plurality_target_collisions": target_collisions(plurality),
        "new_target_collisions": target_collisions(selected),
        "changed": changed,
    }


def recompute_ensembles(
    pairs: list[tuple[str, str]],
    alignments_raw: list[dict],
    approaches: list[str] | None = None,
    write: bool = True,
) -> list[dict]:
    """Recompute every LLM ensemble from the cached per-mode predictions.

    Pure cache work — no API request and no model — so the published ensemble
    numbers can be refreshed without paid calls.  Prints the before/after mean
    F1 per approach (the cached plurality value and the plurality rule
    recomputed from the same per-mode data, against the new >=2-of-3 rule) and
    one example where plurality and the new rule disagree.
    """
    from statistics import mean

    approaches = approaches or LLM_APPROACHES
    summaries: list[dict] = []

    print(f"{'approach':<14}{'pairs':>6}{'mean cached':>12}{'mean plurality':>15}"
          f"{'mean new':>10}{'changed':>9}{'collisions':>11}")
    print("-" * 77)
    for approach in approaches:
        cached_values: list[float] = []
        plurality_values: list[float] = []
        new_values: list[float] = []
        approach_summaries: list[dict] = []
        for src_name, tgt_name in pairs:
            gt = find_gt(src_name, tgt_name, alignments_raw)
            if gt is None:
                continue
            try:
                summary = rebuild_ensemble(approach, src_name, tgt_name, gt, write=write)
            except FileNotFoundError as e:
                print(f"  skip {e}")
                continue
            approach_summaries.append(summary)
            summaries.append(summary)
            if summary["old_f1"] is not None:
                cached_values.append(summary["old_f1"])
            plurality_values.append(summary["plurality_f1"])
            new_values.append(summary["new_f1"])
        changed = [s for s in approach_summaries
                   if abs(s["plurality_f1"] - s["new_f1"]) > 1e-9]
        collisions = sum(s["new_target_collisions"] for s in approach_summaries)
        cached_str = f"{mean(cached_values):.4f}" if cached_values else "—"
        print(f"{approach:<14}{len(approach_summaries):>6}{cached_str:>12}"
              f"{mean(plurality_values):>15.4f}{mean(new_values):>10.4f}"
              f"{len(changed):>9}{collisions:>11}")

    for summary in summaries:
        if summary["changed"]:
            print(f"\nplurality vs >=2-of-3 differ on {summary['pair']} "
                  f"({summary['approach']}): plurality F1={summary['plurality_f1']:.3f} "
                  f"new F1={summary['new_f1']:.3f}")
            for src, old_tgt, new_tgt in summary["changed"][:5]:
                print(f"    {src}: plurality {old_tgt or '—'} → new {new_tgt or '—'}")
            break
    return summaries


def llm_aggregations(all_data: dict) -> dict[str, dict[str, float]]:
    """Published aggregation rules per LLM approach, computed from the cache.

    `mean_over_modes`, `loo_mode_selection` and `ensemble` do not read the
    ground truth of the pair they are applied to; `oracle_max` does, and is
    kept only for reference.
    """
    out: dict[str, dict[str, float]] = {}
    for approach in LLM_APPROACHES:
        app_data = all_data.get(approach, {})
        per_pair_modes: dict[str, dict[str, float]] = {}
        per_pair_ensemble: dict[str, float] = {}
        for pair, record in app_data.items():
            if not isinstance(record, dict) or "mode" in record:
                continue
            modes = {
                mode: d["f1"]
                for mode, d in record.items()
                if isinstance(d, dict) and d.get("mode") != "ensemble" and "f1" in d
            }
            if modes:
                per_pair_modes[pair] = modes
            ens = record.get("ensemble")
            if isinstance(ens, dict) and "f1" in ens:
                per_pair_ensemble[pair] = ens["f1"]
        if per_pair_modes:
            out[approach] = aggregate_mode_f1(per_pair_modes, per_pair_ensemble or None)
    return out


# Retriever of every LLM approach, for the request-count accounting.
RETRIEVER_KIND = {
    "llm-gpt": "embedding-MiniLM",
    "llm-deepseek": "embedding-MiniLM",
    "llm-hybrid": "BM25+MiniLM",
    "llm-bm25": "BM25-only",
}

# Retrievers that give every source node at least one candidate, so the request
# count of a mode equals the number of source nodes exactly.
FULL_COVERAGE_RETRIEVERS = frozenset({"embedding-MiniLM", "BM25+MiniLM"})


def taxonomy_sizes(data_dir: Path = Path("data/processed/oaei")) -> dict[str, int]:
    """Node count of every processed taxonomy, for the request accounting."""
    sizes: dict[str, int] = {}
    for path in sorted(data_dir.glob("*.json")):
        if path.name == "alignments.json":
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        nodes = data.get("nodes") if isinstance(data, dict) else None
        if isinstance(nodes, dict):
            sizes[str(data.get("name", path.stem))] = len(nodes)
            sizes.setdefault(path.stem, len(nodes))
    return sizes


def bm25_request_counts(
    pair_list: list[tuple[str, str]],
    taxonomies: dict,
    top_k: int = 5,
) -> dict[tuple[str, str], int]:
    """Requests a BM25-only run issues per mode for a pair, recomputed locally.

    BM25 drops the source nodes whose best lexical score is zero, so its request
    count is below |S| and cannot be read off the cached records (which list the
    matched sources only).  The retrieval is deterministic and needs no model
    (`_embed_matcher` is lazy), so it is recomputed here through the same code
    path and the same `top_k`, and is exact for the cached runs.
    """
    from src.matchers.bm25_llm import BM25LLMMatcher

    matcher = BM25LLMMatcher(model="unused", top_k=top_k)
    counts: dict[tuple[str, str], int] = {}
    for src_name, tgt_name in pair_list:
        if src_name not in taxonomies or tgt_name not in taxonomies:
            continue
        candidates = matcher.retrieve_candidates(taxonomies[src_name], taxonomies[tgt_name])
        counts[(src_name, tgt_name)] = sum(1 for values in candidates.values() if values)
    return counts


def save_aggregations(
    path: str | Path = "results/llm-aggregations.json",
    alignments_raw: list[dict] | None = None,
    taxonomies: dict | None = None,
) -> dict:
    """Persist the published LLM aggregations in machine-readable form.

    Per approach: `mean_over_modes`, `loo_mode_selection`, `ensemble` and
    `oracle_max` under exactly those names (the last one is an oracle, the
    other three are selection-free), the per-pair F1 of every rule, the request
    accounting (`requests.per_pair` and `requests.total`, from the records'
    own count when present, else the source-node count, else — for BM25 — a
    model-free recomputation of the same retrieval) and the legacy `plurality`
    ensemble values kept for the aggregation-rule ablation.
    """
    from statistics import mean

    if alignments_raw is None:
        alignments_raw = load_alignments(Path("data/processed/oaei/alignments.json"))
    canonical = canonical_directions(alignments_raw)
    sizes = taxonomy_sizes()

    all_data = load_all_cached()
    aggregates = llm_aggregations(all_data)

    bm25_counts: dict[tuple[str, str], int] = {}
    if taxonomies is not None and all_data.get("llm-bm25"):
        bm25_pairs = [
            canonical.get(pair, (pair.split("↔")[0], pair.split("↔")[-1]))
            for pair in sorted(all_data["llm-bm25"])
        ]
        bm25_counts = bm25_request_counts(bm25_pairs, taxonomies)

    out: dict = {
        "rules": {
            "mean_over_modes": "mean over C/CP/CC, per pair then over pairs (selection-free)",
            "loo_mode_selection": "for pair P the mode with the best mean over the other pairs "
                                  "is applied to P (selection-free)",
            "ensemble": f"{ENSEMBLE_RULE} (selection-free)",
            "oracle_max": "per-pair maximum over C/CP/CC, chosen with the pair's own "
                          "ground truth (ORACLE, not deployable)",
            "plurality": "legacy rule: most-voted target per source, ties by the first vote, "
                         "no 1:1 (kept for the aggregation-rule ablation only)",
        },
        "approaches": {},
    }

    for approach in LLM_APPROACHES:
        app_data = all_data.get(approach)
        agg = aggregates.get(approach)
        if not app_data or not agg:
            continue

        per_pair_f1: dict[str, dict[str, float]] = {}
        plurality_per_pair: dict[str, float] = {}
        requests_per_pair: dict[str, dict[str, int | None]] = {}
        measured_entries = 0
        all_values_present = True

        for pair in sorted(app_data):
            record = app_data[pair]
            if not isinstance(record, dict):
                continue
            src_name, tgt_name = canonical.get(
                pair, (pair.split("↔")[0], pair.split("↔")[-1])
            )
            entry: dict[str, float] = {
                mode: d["f1"]
                for mode, d in record.items()
                if isinstance(d, dict) and d.get("mode") != "ensemble" and "f1" in d
            }
            ens = record.get("ensemble")
            if isinstance(ens, dict) and "f1" in ens:
                entry["ensemble"] = ens["f1"]

            try:
                per_mode, mode_requests = cached_mode_candidates(approach, src_name, tgt_name)
            except (FileNotFoundError, ValueError):
                per_mode, mode_requests = {}, {}

            if per_mode:
                plurality = plurality_candidates(per_mode)
                gt = find_gt(src_name, tgt_name, alignments_raw)
                if gt is not None:
                    plurality_f1 = evaluate_1to1(
                        to_alignment(plurality, src_name, tgt_name), gt
                    ).f1
                    entry["plurality"] = plurality_f1
                    plurality_per_pair[pair] = plurality_f1

            # Requests: the stored count when the record has one, otherwise the
            # source-node count — exact for a retriever that gives every source
            # node at least one candidate, an upper bound otherwise.
            n_src = sizes.get(src_name)
            covers_all = RETRIEVER_KIND.get(approach) in FULL_COVERAGE_RETRIEVERS
            pair_requests: dict[str, int | None] = {}
            for mode in LLM_MODES:
                stored = mode_requests.get(mode)
                if stored is not None:
                    pair_requests[mode] = stored
                    measured_entries += 1
                elif covers_all:
                    pair_requests[mode] = n_src
                else:
                    pair_requests[mode] = bm25_counts.get((src_name, tgt_name))
                if pair_requests[mode] is None:
                    all_values_present = False
            requests_per_pair[pair] = pair_requests
            per_pair_f1[pair] = entry

        retriever = RETRIEVER_KIND.get(approach, "unknown")
        covers_all = retriever in FULL_COVERAGE_RETRIEVERS
        request_values = [v for pair in requests_per_pair.values() for v in pair.values()]
        request_total = sum(v for v in request_values if v is not None) or None
        if measured_entries and measured_entries == len(request_values):
            basis = "records"
        elif covers_all:
            basis = "records+source_node_count"
        elif bm25_counts:
            basis = "records+retrieval_recomputed"
        else:
            basis = "records"
        approach_data = dict(agg)
        for key in ("mean_over_modes", "loo_mode_selection", "ensemble", "oracle_max"):
            approach_data.setdefault(key, None)
        approach_data.update({
            "n_pairs": len(per_pair_f1),
            "retriever": retriever,
            "requests": {
                "per_pair": requests_per_pair,
                "total": request_total,
                "exact": bool(all_values_present and request_total),
                "basis": basis,
                "measured_mode_entries": measured_entries,
                "note": (
                    "one request per source node per mode; a value comes from the "
                    "record's `requests` field when present, otherwise from the "
                    "source-node count (exact for a retriever covering every source) "
                    "or from a model-free recomputation of the same BM25 retrieval"
                ),
            },
            "plurality": {
                "mean": mean(plurality_per_pair.values()) if plurality_per_pair else None,
                "per_pair": plurality_per_pair,
            },
            "per_pair_f1": per_pair_f1,
        })
        out["approaches"][approach] = approach_data

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return out


EMBEDDING_MODELS: list[tuple[str, str]] = [
    ("LaBSE", "sentence-transformers/LaBSE"),
    ("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
    ("ruRoberta-large", "ai-forever/ruRoberta-large"),
    ("rubert-base", "DeepPavlov/rubert-base-cased"),
]


# ── Runners ───────────────────────────────────────────────────────────


def run_embedding(taxonomies: dict, alignments_raw: list[dict],
                  pairs: list[tuple[str, str]] | None = None) -> None:
    """Run all embedding models on given (or default) pairs."""
    from src.matchers.embedding import EmbeddingMatcher

    if pairs is None:
        pairs = OAEI_SAMPLE_PAIRS
    for short_name, model_id in EMBEDDING_MODELS:
        approach = f"embedding-{short_name}"
        print(f"Embedding ({short_name})…")
        matcher = EmbeddingMatcher(model_name=model_id, description_mode="name_only")
        for src_name, tgt_name in pairs:
            src = taxonomies[src_name]
            tgt = taxonomies[tgt_name]
            gt = find_gt(src_name, tgt_name, alignments_raw)
            if gt is None:
                continue
            t0 = time.perf_counter()
            pred, _scores = matcher.match(src, tgt)
            elapsed = time.perf_counter() - t0
            mp = [{"source_id": m.source_id, "target_id": m.target_id,
                   "confidence": m.confidence} for m in pred.matches]
            metrics = evaluate_1to1(pred, gt)
            save_results(approach, src_name, tgt_name, metrics,
                         extra={"wall_time": elapsed, "model": model_id},
                         match_pairs=mp)
            print(f"  {src_name}↔{tgt_name}: F1={metrics.f1:.4f}  {elapsed:.1f}s")


def run_string_equiv(taxonomies: dict, alignments_raw: list[dict],
                     pairs: list[tuple[str, str]] | None = None) -> None:
    """Run StringEquiv baseline on given (or default) pairs."""
    from src.matchers.string_equiv import StringEquivMatcher

    if pairs is None:
        pairs = OAEI_SAMPLE_PAIRS
    print("StringEquiv…")
    matcher = StringEquivMatcher()
    for src_name, tgt_name in pairs:
        src = taxonomies[src_name]
        tgt = taxonomies[tgt_name]
        gt = find_gt(src_name, tgt_name, alignments_raw)
        if gt is None:
            continue
        t0 = time.perf_counter()
        pred, _details = matcher.match(src, tgt)
        elapsed = time.perf_counter() - t0
        mp = [{"source_id": m.source_id, "target_id": m.target_id,
               "confidence": m.confidence} for m in pred.matches]
        metrics = evaluate_1to1(pred, gt)
        save_results("string-equiv", src_name, tgt_name, metrics,
                     extra={"wall_time": elapsed}, match_pairs=mp)
        print(f"  {src_name}↔{tgt_name}: F1={metrics.f1:.4f}  {elapsed:.3f}s")


def run_llm_gpt(taxonomies: dict, alignments_raw: list[dict],
                pairs: list[tuple[str, str]] | None = None,
                force: bool = False) -> None:
    """Run LLM/gpt-4.1-mini (kodikrouter)."""
    if pairs is None:
        pairs = OAEI_SAMPLE_PAIRS
    print("LLM (gpt-4.1-mini)…")
    matcher = LLMMatcher(model="openai/gpt-4.1-mini", top_k=5, temperature=0.0)
    _run_llm_impl("llm-gpt", matcher, taxonomies, alignments_raw, pairs, force=force)


def run_llm_deepseek(taxonomies: dict, alignments_raw: list[dict],
                     pairs: list[tuple[str, str]] | None = None,
                     force: bool = False) -> None:
    """Run LLM/deepseek-v4-flash (DeepSeek direct)."""
    if pairs is None:
        pairs = OAEI_SAMPLE_PAIRS
    import os
    print("LLM (deepseek-v4-flash)…")
    matcher = LLMMatcher(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        top_k=5,
        temperature=0.0,
        max_workers=10,
        use_structured_output=False,
    )
    _run_llm_impl("llm-deepseek", matcher, taxonomies, alignments_raw, pairs, force=force)


def _run_llm_impl(approach: str, matcher: LLMMatcher, taxonomies: dict,
                  alignments_raw: list[dict],
                  pairs: list[tuple[str, str]],
                  force: bool = False) -> None:
    """Common LLM runner: all pairs × all modes + ensemble.

    `force` ignores cached per-mode predictions and re-requests them (needed
    when a cached result is stale, e.g. after the ancestor-chain fix); the
    ensemble itself is always recomputed from the per-mode predictions and
    never costs an API request.
    """
    from tqdm import tqdm

    for src_name, tgt_name in pairs:
        src = taxonomies[src_name]
        tgt = taxonomies[tgt_name]
        gt = find_gt(src_name, tgt_name, alignments_raw)
        if gt is None:
            continue
        n_src = src.node_count

        # Per-mode predictions (fresh or from cache, always in canonical direction).
        mode_candidates: dict[str, list[Candidate]] = {}
        mode_requests: dict[str, int | None] = {}

        for mode in LLM_MODES:
            cached = None if force else load_cached(
                approach, src_name, tgt_name, mode, expect=(src_name, tgt_name)
            )
            alignment = record_alignment(cached) if cached else None
            if cached and alignment is not None:
                tqdm.write(f"  {src_name}↔{tgt_name} {LLM_SHORT[mode]}: "
                           f"cached F1={cached['f1']:.4f}")
                mode_candidates[mode] = [
                    (m.source_id, m.target_id, m.confidence) for m in alignment.matches
                ]
                request_count = cached.get("requests")
                mode_requests[mode] = (
                    int(request_count) if isinstance(request_count, (int, float)) else None
                )
                continue

            t0 = time.perf_counter()
            pred, details = matcher.match(src, tgt, mode=mode,
                                           pbar_desc=f"    {src_name}↔{tgt_name} {LLM_SHORT[mode]}",
                                           pbar_position=1)
            t = matcher.last_timing
            metrics = evaluate_1to1(pred, gt)
            mp = [{"source_id": d.source_id, "target_id": d.target_id,
                   "confidence": d.confidence}
                  for d in details if d.target_id]
            save_results(approach, src_name, tgt_name, metrics, mode=mode, extra={
                "wall_time": time.perf_counter() - t0,
                "api_time": t.get("api_time", 0),
                "per_node": t.get("api_time", 0) / n_src if n_src else 0,
                "requests": t.get("requests", 0),
                "judgements": t.get("judgements", 0),
                "retries": t.get("retries", 0),
            }, match_pairs=mp)
            mode_candidates[mode] = [
                (d.source_id, d.target_id, d.confidence) for d in details if d.target_id
            ]
            mode_requests[mode] = int(t.get("requests", 0))
            tqdm.write(f"  {src_name}↔{tgt_name} {LLM_SHORT[mode]}: "
                       f"F1={metrics.f1:.4f}  api={t.get('api_time', 0):.0f}s  "
                       f"requests={t.get('requests', 0)}")

        # Ensemble: shared >=2-of-3 rule, recomputed from the per-mode
        # predictions only — no API request.
        cached_ens = load_cached(approach, src_name, tgt_name, "ensemble")
        if cached_ens and cached_ens.get("rule") == ENSEMBLE_RULE and not force:
            tqdm.write(f"  {src_name}↔{tgt_name} ENS: cached")
        elif len(mode_candidates) == len(LLM_MODES):
            t0 = time.perf_counter()
            selected = ensemble_candidates(mode_candidates)
            ens_alignment = to_alignment(selected, src_name, tgt_name)
            metrics_ens = evaluate_1to1(ens_alignment, gt)
            save_results(approach, src_name, tgt_name, metrics_ens, mode="ensemble", extra={
                "rule": ENSEMBLE_RULE,
                "min_votes": ENSEMBLE_MIN_VOTES,
                "modes": LLM_MODES,
                "mode_requests": mode_requests,
                "requests": 0,
                "wall_time": time.perf_counter() - t0,
            }, match_pairs=[
                {"source_id": s, "target_id": t, "confidence": c} for s, t, c in selected
            ])
            tqdm.write(f"  {src_name}↔{tgt_name} ENS: F1={metrics_ens.f1:.4f}  "
                       f"matches={len(selected)}")


def run_llm_hybrid(taxonomies: dict, alignments_raw: list[dict],
                  pairs: list[tuple[str, str]] | None = None,
                  force: bool = False) -> None:
    """Run LLM/deepseek-v4-flash with BM25+embedding hybrid retrieval."""
    if pairs is None:
        pairs = OAEI_SAMPLE_PAIRS
    import os
    from src.matchers.hybrid import HybridLLMMatcher
    print("LLM hybrid (deepseek-v4-flash + BM25+MiniLM)…")
    matcher = HybridLLMMatcher(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        top_k=5,
        temperature=0.0,
        max_workers=10,
        use_structured_output=False,
    )
    _run_llm_impl("llm-hybrid", matcher, taxonomies, alignments_raw, pairs, force=force)


def run_llm_bm25(taxonomies: dict, alignments_raw: list[dict],
                 pairs: list[tuple[str, str]] | None = None,
                 force: bool = False) -> None:
    """Run LLM/deepseek-v4-flash with BM25-only lexical retrieval."""
    if pairs is None:
        pairs = OAEI_SAMPLE_PAIRS
    import os
    from src.matchers.bm25_llm import BM25LLMMatcher
    print("LLM bm25 (deepseek-v4-flash + BM25-only)…")
    matcher = BM25LLMMatcher(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        top_k=5,
        temperature=0.0,
        max_workers=10,
        use_structured_output=False,
    )
    _run_llm_impl("llm-bm25", matcher, taxonomies, alignments_raw, pairs, force=force)


# ── Table printer ─────────────────────────────────────────────────────


def print_table() -> None:
    all_data = load_all_cached()
    if not all_data:
        print("No cached results found. Run with --run first.")
        return

    # Collect all approaches, excluding "embedding" (legacy, pre-model-split).
    llm_apps = sorted(a for a in all_data if a.startswith("llm-"))
    emb_apps = sorted(a for a in all_data if a.startswith("embedding-"))
    other_apps = sorted(a for a in all_data
                         if a not in llm_apps and a not in emb_apps
                         and a != "embedding")
    approaches = emb_apps + other_apps + llm_apps
    oracle_headers = [f"{app} (oracle)" for app in llm_apps]
    headers = ["Pair"] + approaches + oracle_headers

    # Collect all pairs that have at least one cached result.
    all_pairs: set[str] = set()
    for app_data in all_data.values():
        for pk in app_data:
            all_pairs.add(pk)

    rows: list[dict] = []
    for pair in sorted(all_pairs):
        row: dict = {"pair": pair}
        for app in approaches:
            if app not in all_data:
                continue
            app_data = all_data[app]
            if pair not in app_data:
                row[app] = None
            elif app.startswith("llm-"):
                record = app_data[pair]
                # Main column: the selection-free ensemble.
                ens = record.get("ensemble") if isinstance(record, dict) else None
                row[app] = f"{ens['f1']:.3f}" if isinstance(ens, dict) else "—"
                # Oracle column: per-pair best representation, labelled as such.
                best_f1: float | None = None
                best_mode = ""
                for mode_key, d in record.items():
                    if not isinstance(d, dict) or d.get("mode", mode_key) == "ensemble":
                        continue
                    value = d.get("f1")
                    if value is None:
                        continue
                    if best_f1 is None or value > best_f1:
                        best_f1, best_mode = value, d.get("mode", mode_key)
                row[f"{app} (oracle)"] = (
                    f"{best_f1:.3f} ({best_mode})" if best_f1 is not None else "—"
                )
            else:
                d = app_data[pair]
                if isinstance(d, dict) and d.get("f1") is not None:
                    row[app] = f"{d['f1']:.3f}"
                else:
                    row[app] = "—"
        rows.append(row)

    # Aggregations: the published selection-free numbers, plus the oracle.
    # The legacy plurality mean comes from `results/llm-aggregations.json`,
    # because the ensemble cache itself now holds the >=2-of-3 records.
    legacy: dict = {}
    aggregations_path = Path("results/llm-aggregations.json")
    if aggregations_path.exists():
        with open(aggregations_path, encoding="utf-8") as fh:
            legacy = json.load(fh).get("approaches", {})

    print("\n## LLM aggregations — mean F1 over the pairs (selection-free first)\n")
    print(f"{'approach':<14}{'mean_over_modes':>17}{'loo_mode_selection':>20}"
          f"{'ensemble':>11}{'oracle_max':>12}{'plurality':>11}")
    for approach, agg in sorted(llm_aggregations(all_data).items()):
        plurality_mean = (legacy.get(approach, {}).get("plurality") or {}).get("mean")
        plurality_str = f"{plurality_mean:.4f}" if plurality_mean is not None else "—"
        print(f"{approach:<14}{agg['mean_over_modes']:>17.4f}"
              f"{agg['loo_mode_selection']:>20.4f}"
              f"{agg.get('ensemble', 0.0):>11.4f}{agg['oracle_max']:>12.4f}"
              f"{plurality_str:>11}")

    # Print markdown table.
    print("\n## Approach Comparison — OAEI Conference Track (F1)\n")

    # Header.
    col_widths = {"pair": max(len(r["pair"]) for r in rows)}
    for h in headers:
        if h in col_widths:
            continue
        col_widths[h] = max(len(h), max((len(str(r.get(h, ""))) for r in rows), default=0))

    def fmt_row(vals: list[str]) -> str:
        parts = [f" {v:<{col_widths.get(h, 12)}} " for v, h in zip(vals, headers)]
        return "|" + "|".join(parts) + "|"

    header_row = fmt_row(headers)
    sep_row = "|" + "|".join(f":{'-' * max(col_widths.get(h, 12) - 2, 1)}:" for h in headers) + "|"

    print("LLM columns are the >=2-of-3 ensemble; `(oracle)` is the per-pair best "
          "representation chosen with the pair's own ground truth.\n")
    print(header_row)
    print(sep_row)
    for r in rows:
        pair_display = r["pair"].replace("↔", " ↔ ")
        vals = [pair_display] + [str(r.get(h, "—")) for h in headers[1:]]
        display_headers = ["pair"] + headers[1:]
        parts = [f" {v:<{col_widths[h]}} " for v, h in zip(vals, display_headers)]
        print("|" + "|".join(parts) + "|")

    print()

    # Compact one-liner.
    print("## Summary (for text)\n")
    for r in rows:
        emb_str = " ".join(f"{k}={r.get(k, '—')}" for k in headers if k.startswith("embedding-"))
        print(f"  {r['pair']}: {emb_str}  "
              f"string-equiv={r.get('string-equiv', '—')}  "
              f"llm-gpt={r.get('llm-gpt', '—')}  "
              f"llm-deepseek={r.get('llm-deepseek', '—')}")


# ── Excel export ─────────────────────────────────────────────────────


def export_xlsx(path: str | Path = "results/comparison.xlsx") -> None:
    """Export full comparison table to a multi-sheet .xlsx file.

    One sheet per metric: F1, Precision, Recall, TP, Predicted, plus an
    `Oracle modes` sheet naming the representation behind each oracle column.
    Every LLM row is one configuration — the plain column is the selection-free
    >=2-of-3 ensemble, the `(oracle)` column one single representation — so the
    metric sheets of a row never mix configurations.  The best deployable cell
    of a row is bolded; oracle cells are excluded.
    """
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    all_data = load_all_cached()
    if not all_data:
        print("No cached data to export.")
        return

    # Collect approaches.
    llm_apps = sorted(a for a in all_data if a.startswith("llm-"))
    emb_apps = sorted(a for a in all_data if a.startswith("embedding-"))
    other_apps = sorted(a for a in all_data
                         if a not in llm_apps and a not in emb_apps
                         and a != "embedding")
    all_apps = emb_apps + other_apps + llm_apps

    # Collect all pairs.
    all_pairs: set[str] = set()
    for app_data in all_data.values():
        for pk in app_data:
            all_pairs.add(pk)
    sorted_pairs = sorted(all_pairs)

    # ── Styles ──
    bold = Font(bold=True)
    bold_blue = Font(bold=True, color="1F4E79")
    header_fill = PatternFill("solid", fgColor="D6E4F0")
    row_even = PatternFill("solid", fgColor="F2F2F2")
    thin_border = Border(
        left=Side(style="thin", color="B0B0B0"),
        right=Side(style="thin", color="B0B0B0"),
        top=Side(style="thin", color="B0B0B0"),
        bottom=Side(style="thin", color="B0B0B0"),
    )

    # ── Columns: (header, approach, use_oracle). ──
    # Every LLM row is ONE configuration: the plain column is the selection-free
    # >=2-of-3 ensemble, the `(oracle)` column is a single representation chosen
    # once by F1.  All metric sheets therefore show F1, precision, recall and TP
    # of the same configuration, never a mix of modes.
    columns: list[tuple[str, str, bool]] = (
        [(app, app, False) for app in all_apps]
        + [(f"{app} (oracle)", app, True) for app in llm_apps]
    )
    best_cols = [c for c, (_, _, oracle) in enumerate(columns, 2) if not oracle]

    def _oracle_mode(pd: dict) -> str | None:
        """Representation the oracle column uses: the best-F1 mode, ties by name."""
        candidates = {
            mk: md for mk, md in pd.items()
            if isinstance(md, dict) and md.get("mode", mk) != "ensemble"
        }
        if not candidates:
            return None
        return max(sorted(candidates), key=lambda mk: candidates[mk].get("f1") or -1.0)

    # ── Helper: get a scalar metric for an approach×pair×configuration. ──
    def _get_metric(app: str, pair: str, metric: str, oracle: bool = False) -> float | None:
        pd = all_data.get(app, {}).get(pair)
        if not isinstance(pd, dict):
            return None
        if not app.startswith("llm-"):
            value = pd.get(metric)
            return value if isinstance(value, (int, float)) else None
        if oracle:
            mode = _oracle_mode(pd)
            record = pd.get(mode) if mode else None
        else:
            record = pd.get("ensemble")
        if not isinstance(record, dict):
            return None
        value = record.get(metric)
        return value if isinstance(value, (int, float)) else None

    # ── Sheets: F1, Precision, Recall, TP, Predicted, GT ──
    sheets_def: list[tuple[str, str, str, bool]] = [
        ("F1", "f1", "0.0000", True),
        ("Precision", "precision", "0.0000", True),
        ("Recall", "recall", "0.0000", True),
        ("True Positives", "matches", "0", True),
        ("Predicted", "total_predicted", "0", False),
    ]

    wb = openpyxl.Workbook()
    # Remove default sheet.
    wb.remove(wb.active)

    for sheet_name, json_key, num_fmt, higher_is_better in sheets_def:
        ws = wb.create_sheet(title=sheet_name)

        extra_cols: list[str] = []
        extra_keys: list[str] = []
        if sheet_name == "Predicted":
            extra_cols = ["GT"]
            extra_keys = ["total_gt"]

        headers = ["Pair"] + [h for h, _, _ in columns] + extra_cols

        # Header row.
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font = bold_blue
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border

        # Bold extra-column headers (e.g., GT on Predicted).
        for ec_offset, _ in enumerate(extra_cols):
            c_extra = len(columns) + 2 + ec_offset
            ws.cell(row=1, column=c_extra).font = bold

        # Data rows.
        for r, pair in enumerate(sorted_pairs, 2):
            ws.cell(row=r, column=1, value=pair.replace("↔", " ↔ ")).border = thin_border
            row_vals: list[tuple[int, float]] = []
            row_cells: dict[int, float] = {}  # col → value for post-processing.

            for c, (_, app, oracle) in enumerate(columns, 2):
                value = _get_metric(app, pair, json_key, oracle=oracle)
                cell = ws.cell(row=r, column=c)
                if value is not None:
                    cell.value = round(value, 4) if isinstance(value, float) else value
                    cell.number_format = num_fmt
                    row_vals.append((c, value))
                    row_cells[c] = value
                else:
                    cell.value = "—"
                cell.alignment = Alignment(horizontal="center")
                cell.border = thin_border

            # Extra columns (e.g., GT on Predicted sheet).
            gt_val: float | None = None
            for ec_offset, (ec_name, ec_key) in enumerate(zip(extra_cols, extra_keys)):
                c_extra = len(columns) + 2 + ec_offset
                ec_val: float | None = None
                for app in all_apps:
                    v = _get_metric(app, pair, ec_key)
                    if v is not None:
                        ec_val = v
                        break
                cell = ws.cell(row=r, column=c_extra)
                if ec_val is not None:
                    cell.value = int(ec_val) if ec_val == int(ec_val) else round(ec_val, 4)
                else:
                    cell.value = "—"
                cell.alignment = Alignment(horizontal="center")
                cell.border = thin_border
                cell.font = bold  # GT cell always bold.
                gt_val = ec_val

            # Even row highlight.
            if r % 2 == 0:
                for c in range(1, len(headers) + 1):
                    ws.cell(row=r, column=c).fill = row_even

            # Bold cells that tie for best; oracle columns are excluded, they
            # are not deployable and would always win.
            best_vals = [(c, v) for c, v in row_vals if c in best_cols]
            if best_vals and sheet_name != "Predicted":
                best_val = (
                    max(v for _, v in best_vals) if higher_is_better
                    else min(v for _, v in best_vals)
                )
                for col_c, val in best_vals:
                    if val == best_val:
                        ws.cell(row=r, column=col_c).font = bold

            # Predicted sheet: bold cells where predicted == GT.
            if sheet_name == "Predicted" and gt_val is not None:
                for col_c, val in row_cells.items():
                    if col_c in best_cols and val == gt_val:
                        ws.cell(row=r, column=col_c).font = bold

        # Column widths.
        ws.column_dimensions["A"].width = 22
        for c in range(2, len(headers) + 1):
            ws.column_dimensions[get_column_letter(c)].width = 16

        ws.freeze_panes = "B2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(sorted_pairs) + 1}"

    # ── Sheet documenting the representation behind every oracle column. ──
    ws_modes = wb.create_sheet(title="Oracle modes")
    for c, h in enumerate(["Pair"] + llm_apps, 1):
        cell = ws_modes.cell(row=1, column=c, value=h)
        cell.font = bold_blue
        cell.fill = header_fill
        cell.border = thin_border
    for r, pair in enumerate(sorted_pairs, 2):
        ws_modes.cell(row=r, column=1, value=pair.replace("↔", " ↔ ")).border = thin_border
        for c, app in enumerate(llm_apps, 2):
            pd = all_data.get(app, {}).get(pair)
            mode = _oracle_mode(pd) if isinstance(pd, dict) else None
            cell = ws_modes.cell(row=r, column=c, value=mode or "—")
            cell.border = thin_border
    ws_modes.column_dimensions["A"].width = 22
    for c in range(2, len(llm_apps) + 2):
        ws_modes.column_dimensions[get_column_letter(c)].width = 16
    ws_modes.freeze_panes = "B2"

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))
    print(f"Exported: {path}")


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare taxonomy matching approaches.")
    parser.add_argument("--run", action="store_true",
                        help="Run the cheap experiments (LLM excluded to avoid API costs).")
    parser.add_argument("--approach",
                        choices=["embedding", "string-equiv", "llm-gpt", "llm-deepseek", "llm-hybrid", "llm-bm25"],
                        help="Run only this approach.")
    parser.add_argument("--table-only", action="store_true",
                        help="Only print the table from cache (no runs).")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Run on all 21 OAEI pairs (default: 6 sample pairs).")
    parser.add_argument("--pairs", type=str, default=None, metavar="SPEC",
                        help="Comma-separated pairs, e.g. `cmt-conference,ekaw-edas`.")
    parser.add_argument("--only-stale", action="store_true",
                        help="Only pairs touching cmt or ekaw (the 11 stale ones).")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached per-mode predictions and re-request them.")
    parser.add_argument("--ensemble-only", action="store_true",
                        help="Recompute all LLM ensembles from cached per-mode predictions (no API).")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --ensemble-only: print before/after without writing.")
    parser.add_argument("--aggregations", nargs="?", const="results/llm-aggregations.json",
                        metavar="PATH",
                        help="Write the published LLM aggregations to JSON "
                             "(default: results/llm-aggregations.json).")
    parser.add_argument("--xlsx", type=str, nargs="?", const="results/comparison.xlsx",
                        metavar="PATH",
                        help="Export full table to .xlsx (default: results/comparison.xlsx).")
    args = parser.parse_args()

    if args.dry_run and not args.ensemble_only:
        parser.error("--dry-run is only meaningful together with --ensemble-only")

    data_proc = Path("data/processed")
    alignments_raw = load_alignments(data_proc / "oaei" / "alignments.json")
    all_pairs = _get_all_pairs(alignments_raw)
    known = {a["source"] for a in alignments_raw} | {a["target"] for a in alignments_raw}
    canonical = canonical_directions(alignments_raw)

    if args.pairs or args.only_stale or args.all_pairs or args.ensemble_only:
        pairs = select_pairs(all_pairs, only_stale=args.only_stale,
                             spec=args.pairs, known=known, canonical=canonical)
    else:
        pairs = select_pairs(OAEI_SAMPLE_PAIRS, canonical=canonical)
    if args.run or args.approach or args.ensemble_only:
        print(f"Pairs: {len(pairs)}")

    if args.ensemble_only:
        recompute_ensembles(pairs, alignments_raw, write=not args.dry_run)
        if args.dry_run:
            return
        taxonomies = load_taxonomies(data_proc / "oaei")
        save_aggregations(alignments_raw=alignments_raw, taxonomies=taxonomies)
        if args.aggregations:
            save_aggregations(args.aggregations, alignments_raw=alignments_raw,
                              taxonomies=taxonomies)
        print_table()
        if args.xlsx:
            export_xlsx(args.xlsx)
        return

    if args.run or args.approach:
        taxonomies = load_taxonomies(data_proc / "oaei")
        approaches_to_run = [args.approach] if args.approach else [
            "embedding", "string-equiv",
            # LLM excluded from default --run to avoid unexpected API costs.
        ]
        runners = {
            "embedding": run_embedding,
            "string-equiv": run_string_equiv,
            "llm-gpt": run_llm_gpt,
            "llm-deepseek": run_llm_deepseek,
            "llm-hybrid": run_llm_hybrid,
            "llm-bm25": run_llm_bm25,
        }
        for app in approaches_to_run:
            runner = runners.get(app)
            if runner is None:
                print(f"Unknown approach: {app}")
            elif app.startswith("llm-"):
                runner(taxonomies, alignments_raw, pairs, force=args.force)
            else:
                runner(taxonomies, alignments_raw, pairs)

    print_table()

    if args.aggregations:
        save_aggregations(args.aggregations, alignments_raw=alignments_raw,
                          taxonomies=load_taxonomies(data_proc / "oaei"))

    if args.xlsx:
        export_xlsx(args.xlsx)


if __name__ == "__main__":
    main()
