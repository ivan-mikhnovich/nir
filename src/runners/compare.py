"""Compare all taxonomy matching approaches — unified table.

Runs Embedding, StringEquiv, LLM/gpt-4.1-mini, and LLM/deepseek-v4-flash over
the OAEI Conference track.  By default only three sample pairs are run; the
cache itself covers all 21 track pairs (`--all-pairs`).  Results are cached to
`results/` to avoid re-running expensive LLM calls.

Usage:
    uv run python -m src.runners.compare          # print table from cache.
    uv run python -m src.runners.compare --run     # run all experiments, then print.
    uv run python -m src.runners.compare --approach embedding  # run specific approach.
"""

import argparse
import json
import time
from pathlib import Path

from src.cache import cache_path, load_all_cached, load_cached, pair_key
from src.data_loader import load_taxonomy
from src.matchers.embedding import EmbeddingMatcher
from src.matchers.llm import LLMMatcher
from src.metrics import evaluate_1to1, MatchMetrics
from src.matchers.string_equiv import StringEquivMatcher
from src.taxonomy import Alignment, TaxonMatch

LLM_MODES = ["concept", "concept-parent", "concept-children"]
LLM_SHORT = {"concept": "C", "concept-parent": "CP", "concept-children": "CC"}
OAEI_SAMPLE_PAIRS = [
    ("cmt", "conference"),
    ("cmt", "confOf"),
    ("conference", "confOf"),
    ("cmt", "edas"),
    ("confOf", "ekaw"),
    ("conference", "ekaw"),
]


def _get_all_pairs(taxonomies: dict) -> list[tuple[str, str]]:
    """Return all pairs across the loaded taxonomies."""
    names = sorted(taxonomies.keys())
    return [(s, t) for i, s in enumerate(names) for t in names[i + 1:]]

# ── Helpers ──────────────────────────────────────────────────────────


def load_taxonomies(data_dir: Path) -> dict:
    taxonomies: dict = {}
    for f in sorted(data_dir.glob("*.json")):
        if f.name == "alignments.json":
            continue
        tax = load_taxonomy(f)
        taxonomies[tax.name] = tax
    return taxonomies


def load_alignments(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_gt(src_name: str, tgt_name: str, alignments_raw: list[dict]) -> Alignment | None:
    """Find ground-truth alignment, trying both source↔target orderings."""
    for al_data in alignments_raw:
        s = al_data["source"]
        t = al_data["target"]
        if (s == src_name and t == tgt_name) or (s == tgt_name and t == src_name):
            matches = [
                TaxonMatch(source_id=m["source_id"], target_id=m["target_id"],
                           confidence=m.get("confidence", 1.0))
                for m in al_data["matches"]
            ]
            # If we matched the reverse direction, swap source/target in matches.
            if s == tgt_name and t == src_name:
                matches = [
                    TaxonMatch(source_id=m.target_id, target_id=m.source_id,
                               confidence=m.confidence)
                    for m in matches
                ]
            return Alignment(
                source=src_name,
                target=tgt_name,
                matches=matches,
            )
    return None


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
    if match_pairs:
        data["match_pairs"] = match_pairs
    path = cache_path(approach, src_name, tgt_name, mode)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
                pairs: list[tuple[str, str]] | None = None) -> None:
    """Run LLM/gpt-4.1-mini (kodikrouter)."""
    if pairs is None:
        pairs = OAEI_SAMPLE_PAIRS
    print("LLM (gpt-4.1-mini)…")
    matcher = LLMMatcher(model="openai/gpt-4.1-mini", top_k=5, temperature=0.0)
    _run_llm_impl("llm-gpt", matcher, taxonomies, alignments_raw, pairs)


def run_llm_deepseek(taxonomies: dict, alignments_raw: list[dict],
                     pairs: list[tuple[str, str]] | None = None) -> None:
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
    _run_llm_impl("llm-deepseek", matcher, taxonomies, alignments_raw, pairs)


def _run_llm_impl(approach: str, matcher: LLMMatcher, taxonomies: dict,
                  alignments_raw: list[dict],
                  pairs: list[tuple[str, str]]) -> None:
    """Common LLM runner: all pairs × all modes + ensemble."""
    from collections import Counter
    from tqdm import tqdm

    for src_name, tgt_name in pairs:
        src = taxonomies[src_name]
        tgt = taxonomies[tgt_name]
        gt = find_gt(src_name, tgt_name, alignments_raw)
        if gt is None:
            continue
        n_src = src.node_count

        # Per-mode alignments (fresh or from cache).
        mode_preds: dict[str, Alignment] = {}

        for mode in LLM_MODES:
            cached = load_cached(approach, src_name, tgt_name, mode)
            if cached:
                tqdm.write(f"  {src_name}↔{tgt_name} {LLM_SHORT[mode]}: "
                           f"cached F1={cached['f1']:.4f}")
                # Reconstruct Alignment from cached match_pairs.
                mp = cached.get("match_pairs", [])
                if mp:
                    mode_preds[mode] = Alignment(
                        source=src_name, target=tgt_name,
                        matches=[TaxonMatch(m["source_id"], m["target_id"],
                                           m.get("confidence", 1.0))
                                 for m in mp],
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
            }, match_pairs=mp)
            mode_preds[mode] = pred
            tqdm.write(f"  {src_name}↔{tgt_name} {LLM_SHORT[mode]}: "
                       f"F1={metrics.f1:.4f}  api={t.get('api_time', 0):.0f}s")

        # Ensemble — majority vote from per-mode alignments, no API calls.
        if load_cached(approach, src_name, tgt_name, "ensemble"):
            tqdm.write(f"  {src_name}↔{tgt_name} ENS: cached")
        elif len(mode_preds) == len(LLM_MODES):
            t0 = time.perf_counter()
            # Majority vote: for each source node, count target votes across modes.
            ens_matches: dict[str, Counter[str]] = {}
            for mode, al in mode_preds.items():
                for m in al.matches:
                    if m.source_id not in ens_matches:
                        ens_matches[m.source_id] = Counter()
                    ens_matches[m.source_id][m.target_id] += 1

            ens_alignment = Alignment(
                source=src_name, target=tgt_name,
                matches=[
                    TaxonMatch(sid, votes.most_common(1)[0][0],
                              confidence=votes.most_common(1)[0][1] / len(LLM_MODES))
                    for sid, votes in ens_matches.items()
                ],
            )
            metrics_ens = evaluate_1to1(ens_alignment, gt)
            save_results(approach, src_name, tgt_name, metrics_ens, mode="ensemble", extra={
                "wall_time": time.perf_counter() - t0,
            })
            tqdm.write(f"  {src_name}↔{tgt_name} ENS: F1={metrics_ens.f1:.4f}")


def run_llm_hybrid(taxonomies: dict, alignments_raw: list[dict],
                  pairs: list[tuple[str, str]] | None = None) -> None:
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
    _run_llm_impl("llm-hybrid", matcher, taxonomies, alignments_raw, pairs)


def run_llm_bm25(taxonomies: dict, alignments_raw: list[dict],
                 pairs: list[tuple[str, str]] | None = None) -> None:
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
    _run_llm_impl("llm-bm25", matcher, taxonomies, alignments_raw, pairs)


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
    headers = ["Pair"] + approaches

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
                # Pick best F1 across C/CP/CC.
                best_f1 = 0.0
                best_mode = ""
                for mode_key, d in app_data[pair].items():
                    if not isinstance(d, dict):
                        continue
                    if d.get("mode", mode_key) == "ensemble":
                        continue
                    if d.get("f1", 0) > best_f1:
                        best_f1 = d["f1"]
                        best_mode = d.get("mode", mode_key)
                if best_f1 > 0:
                    row[app] = f"{best_f1:.3f} ({best_mode})"
                else:
                    row[app] = "—"
            else:
                d = app_data[pair]
                if isinstance(d, dict):
                    row[app] = f"{d.get('f1', 0):.3f}"
                else:
                    row[app] = "—"
        rows.append(row)

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

    One sheet per metric: F1, Precision, Recall, TP, Predicted, GT.
    Best value per row (highest for F1/P/R, closest-to-GT for TP) is bolded.
    """
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter

    all_data = load_all_cached()
    if not all_data:
        print("No cached data to export.")
        return

    # Collect approaches.
    llm_apps = [a for a in all_data if a.startswith("llm-")]
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

    # ── Helper: get a scalar metric for an approach×pair. ──
    def _get_metric(app: str, pair: str, metric: str) -> float | None:
        app_data = all_data.get(app, {})
        pd = app_data.get(pair)
        if not isinstance(pd, dict):
            return None
        if app.startswith("llm-"):
            best = 0.0
            for mk, md in pd.items():
                if isinstance(md, dict) and md.get("mode", mk) != "ensemble":
                    v = md.get(metric)
                    if isinstance(v, (int, float)) and v > best:
                        best = v
            return best if best > 0 else None
        v = pd.get(metric)
        return v if isinstance(v, (int, float)) else None

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

        headers = ["Pair"] + all_apps + extra_cols

        # Header row.
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font = bold_blue
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border

        # Bold extra-column headers (e.g., GT on Predicted).
        for ec_offset, _ in enumerate(extra_cols):
            c_extra = len(all_apps) + 2 + ec_offset
            ws.cell(row=1, column=c_extra).font = bold

        # Data rows.
        for r, pair in enumerate(sorted_pairs, 2):
            ws.cell(row=r, column=1, value=pair.replace("↔", " ↔ ")).border = thin_border
            row_vals: list[tuple[int, float]] = []
            row_cells: dict[int, float] = {}  # col → value for post-processing.

            for c, app in enumerate(all_apps, 2):
                value = _get_metric(app, pair, json_key)
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
                c_extra = len(all_apps) + 2 + ec_offset
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

            # Bold ALL cells that tie for best (skip Predicted — uses GT-match logic).
            if row_vals and sheet_name != "Predicted":
                if higher_is_better:
                    best_val = max(v for _, v in row_vals)
                else:
                    best_val = min(v for _, v in row_vals)
                for col_c, val in row_vals:
                    if val == best_val:
                        ws.cell(row=r, column=col_c).font = bold

            # Predicted sheet: bold cells where predicted == GT.
            if sheet_name == "Predicted" and gt_val is not None:
                for col_c, val in row_cells.items():
                    if val == gt_val:
                        ws.cell(row=r, column=col_c).font = bold

        # Column widths.
        ws.column_dimensions["A"].width = 22
        for c in range(2, len(headers) + 1):
            ws.column_dimensions[get_column_letter(c)].width = 16

        ws.freeze_panes = "B2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(sorted_pairs) + 1}"

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))
    print(f"Exported: {path}")


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare taxonomy matching approaches.")
    parser.add_argument("--run", action="store_true",
                        help="Run all experiments (slow for LLM).")
    parser.add_argument("--approach",
                        choices=["embedding", "string-equiv", "llm-gpt", "llm-deepseek", "llm-hybrid", "llm-bm25"],
                        help="Run only this approach.")
    parser.add_argument("--table-only", action="store_true",
                        help="Only print the table from cache (no runs).")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Run on all 21 OAEI pairs (default: 3 sample pairs).")
    parser.add_argument("--xlsx", type=str, nargs="?", const="results/comparison.xlsx",
                        metavar="PATH",
                        help="Export full table to .xlsx (default: results/comparison.xlsx).")
    args = parser.parse_args()

    data_proc = Path("data/processed")
    taxonomies = load_taxonomies(data_proc / "oaei")
    alignments_raw = load_alignments(data_proc / "oaei" / "alignments.json")

    pairs = _get_all_pairs(taxonomies) if args.all_pairs else OAEI_SAMPLE_PAIRS

    if args.run or args.approach:
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
            if app in runners:
                runners[app](taxonomies, alignments_raw, pairs)
            else:
                print(f"Unknown approach: {app}")

    print_table()

    if args.xlsx:
        export_xlsx(args.xlsx)


if __name__ == "__main__":
    main()
