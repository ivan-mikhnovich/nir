"""Render taxonomies and all available mappings to .dot via GraphViz (xdot).

Reads cached results from ``results/``.  Only renders if ``match_pairs``
is present — does NOT recompute anything.  Ground truth alignments
are read from ``data/processed/oaei/alignments.json``.

Usage:  uv run python -m src.viz.render

Output: results/viz/
          taxonomies/     .dot for each of 7 OAEI ontologies.
          gt/             .dot for 21 ground-truth pairs.
          embedding/      .dot for cached embedding results with match_pairs.
          string-equiv/   .dot for cached string-equiv results.
          llm-gpt/        .dot for cached GPT results.
          llm-deepseek/   .dot for cached DeepSeek results.
"""

from __future__ import annotations

import json
from pathlib import Path

import graphviz  # type: ignore

from src.data_loader import load_taxonomy
from src.taxonomy import Alignment, TaxonMatch

# ── Colours ───────────────────────────────────────────────────────────

SRC_FILL = "#e8f4fd"
SRC_STROKE = "#5ba3d9"
TGT_FILL = "#fde8e8"
TGT_STROKE = "#d95b5b"
MATCH_COLOR = "#2ca02c"
MATCH_PW = "2.5"
MATCH_EDGE_PW = "2.0"
EDGE_COLOR = "#b0b0b0"

FONT_NAME = "Arial"
FONT_SIZE = "10"
DPI = 144

OUT_DIR = Path("results/viz")
DATA_DIR = Path("data/processed/oaei")
RESULTS_DIR = Path("results")


def _safe_id(nid: str) -> str:
    """Replace GraphViz-unsafe characters in node IDs."""
    return nid.replace("#", "_").replace(".", "_").replace(" ", "_")


def _prefix_id(prefix: str, nid: str) -> str:
    """Qualify a node ID for a pair graph (avoids name collisions)."""
    return f"{prefix}_{_safe_id(nid)}"


# ── Single taxonomy ──────────────────────────────────────────────────

def render_taxonomy(json_path: Path, out_dir: Path) -> None:
    """Render a single taxonomy to .dot."""
    tax = load_taxonomy(json_path)
    name = tax.name

    dot = graphviz.Digraph(
        name=f"tax_{_safe_id(name)}",
        engine="dot",
    )
    dot.attr(rankdir="TB", nodesep="0.5", ranksep="0.9",
             bgcolor="transparent", pad="0", dpi=str(DPI))
    dot.attr("node", shape="box", style="rounded,filled",
             fontname=FONT_NAME, fontsize=FONT_SIZE,
             margin="0.14,0.08")
    dot.attr("edge", color=EDGE_COLOR, penwidth="0.9",
             arrowhead="none")

    for nid, node in tax.nodes.items():
        dot.node(
            _safe_id(nid), node.name,
            fillcolor=SRC_FILL, color=SRC_STROKE, penwidth="1.3",
        )

    for nid, node in tax.nodes.items():
        for pid in node.parents:
            if pid in tax.nodes:
                dot.edge(_safe_id(pid), _safe_id(nid))

    _write_dot(dot, out_dir, name)


# ── Pair with matches ────────────────────────────────────────────────

def render_pair(
    src_json: Path,
    tgt_json: Path,
    alignment: Alignment,
    out_dir: Path,
    label: str | None = None,
) -> None:
    """Render a taxonomy pair with matched nodes/edges to .dot."""
    src = load_taxonomy(src_json)
    tgt = load_taxonomy(tgt_json)
    src_name = src.name
    tgt_name = tgt.name
    pair_key = label or f"{src_name}-{tgt_name}"

    # Collect matches.
    matched_src: set[str] = set()
    matched_tgt: set[str] = set()
    match_edges: list[tuple[str, str]] = []
    for m in alignment.matches:
        if m.source_id in src.nodes and m.target_id in tgt.nodes:
            matched_src.add(m.source_id)
            matched_tgt.add(m.target_id)
            match_edges.append((m.source_id, m.target_id))

    dot = graphviz.Digraph(
        name=f"pair_{_safe_id(pair_key)}",
        engine="dot",
    )
    dot.attr(rankdir="TB", nodesep="0.5", ranksep="0.9",
             bgcolor="transparent", pad="0.2", dpi=str(DPI),
             compound="true")
    dot.attr("node", shape="box", style="rounded,filled",
             fontname=FONT_NAME, fontsize=FONT_SIZE,
             margin="0.14,0.08")

    # ── Source subgraph ──
    with dot.subgraph(name="cluster_src") as sub:
        sub.attr(label=src_name, style="dashed", color=SRC_STROKE,
                 fontname=FONT_NAME, fontsize="12")
        sub.attr("edge", color=EDGE_COLOR, penwidth="0.9",
                 arrowhead="none")
        for nid, node in src.nodes.items():
            sub.node(
                _prefix_id("s", nid), node.name,
                fillcolor=SRC_FILL,
                color=MATCH_COLOR if nid in matched_src else SRC_STROKE,
                penwidth=MATCH_PW if nid in matched_src else "1.3",
            )
        for nid, node in src.nodes.items():
            for pid in node.parents:
                if pid in src.nodes:
                    sub.edge(_prefix_id("s", pid), _prefix_id("s", nid))

    # ── Target subgraph ──
    with dot.subgraph(name="cluster_tgt") as sub:
        sub.attr(label=tgt_name, style="dashed", color=TGT_STROKE,
                 fontname=FONT_NAME, fontsize="12")
        sub.attr("edge", color=EDGE_COLOR, penwidth="0.9",
                 arrowhead="none")
        for nid, node in tgt.nodes.items():
            sub.node(
                _prefix_id("t", nid), node.name,
                fillcolor=TGT_FILL,
                color=MATCH_COLOR if nid in matched_tgt else TGT_STROKE,
                penwidth=MATCH_PW if nid in matched_tgt else "1.3",
            )
        for nid, node in tgt.nodes.items():
            for pid in node.parents:
                if pid in tgt.nodes:
                    sub.edge(_prefix_id("t", pid), _prefix_id("t", nid))

    # ── Match edges (between clusters) ──
    for s_nid, t_nid in match_edges:
        dot.edge(
            _prefix_id("s", s_nid),
            _prefix_id("t", t_nid),
            color=MATCH_COLOR, penwidth=MATCH_EDGE_PW,
            style="dashed", arrowhead="none",
        )

    _write_dot(dot, out_dir, pair_key)


def _write_dot(dot: graphviz.Digraph, out_dir: Path, name: str) -> None:
    """Write .dot source file."""
    safe_name = _safe_id(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / f"{safe_name}.dot"
    dot_path.write_text(dot.source, encoding="utf-8")


# ── Cached result helpers ────────────────────────────────────────────

def _alignment_from_match_pairs(
    mp: list[dict],
    source: str,
    target: str,
) -> Alignment:
    """Build an Alignment from cached match_pairs."""
    return Alignment(
        source=source, target=target,
        matches=[TaxonMatch(m["source_id"], m["target_id"],
                           m.get("confidence", 1.0))
                 for m in mp],
    )


def _count_and_skip(approach: str, rendered: int, skipped_simple: int, skipped_ens: int) -> None:
    """Print summary line for a result approach."""
    parts = [f"{rendered} files"]
    if skipped_simple:
        parts.append(f"{skipped_simple} no match_pairs")
    if skipped_ens:
        parts.append(f"{skipped_ens} ensemble (reconstructed)")
    print(f"=== {approach} ===  → {', '.join(parts)}\n")


# ── Ensemble reconstruction ──────────────────────────────────────────


def _reconstruct_ensemble(
    mode_map: dict[str, Path],
    src: str,
    tgt: str,
    app_out: Path,
    src_path: Path,
    tgt_path: Path,
) -> Alignment | None:
    """Build ensemble alignment from per-mode match_pairs (majority vote).

    No API calls — purely post-hoc from cached data.
    """
    from collections import Counter

    mode_alignments: dict[str, Alignment] = {}
    for mode, rp in mode_map.items():
        if mode in ("__no_mode__", "ensemble"):
            continue
        with open(rp, encoding="utf-8") as f:
            d = json.load(f)
        mp = d.get("match_pairs")
        if not mp:
            continue
        mode_alignments[mode] = _alignment_from_match_pairs(mp, src, tgt)

    if len(mode_alignments) < 2:
        return None

    ens_matches: dict[str, Counter[str]] = {}
    for mode, al in mode_alignments.items():
        for m in al.matches:
            if m.source_id not in ens_matches:
                ens_matches[m.source_id] = Counter()
            ens_matches[m.source_id][m.target_id] += 1

    n_modes = len(mode_alignments)
    return Alignment(
        source=src, target=tgt,
        matches=[
            TaxonMatch(sid, votes.most_common(1)[0][0],
                      confidence=votes.most_common(1)[0][1] / n_modes)
            for sid, votes in ens_matches.items()
        ],
    )


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} not found.  Run `make prepare-data` first.")
        return

    # Load taxonomy paths (lazy — only load full Taxonomy in render_*).
    taxonomies: dict[str, Path] = {}
    for json_path in sorted(DATA_DIR.glob("*.json")):
        name = json_path.stem
        if name == "alignments":
            continue
        taxonomies[name] = json_path

    # ── Render each taxonomy ──
    print("=== Taxonomies ===")
    tax_out = OUT_DIR / "taxonomies"
    for name in sorted(taxonomies):
        render_taxonomy(taxonomies[name], tax_out)
    n_tax = len(list(tax_out.glob("*.dot")))
    print(f"  → {n_tax} files\n")

    # ── Render GT pairs ──
    align_path = DATA_DIR / "alignments.json"
    with open(align_path, encoding="utf-8") as f:
        gt_raw = json.load(f)

    print("=== Ground Truth (21 pairs) ===")
    gt_out = OUT_DIR / "gt"
    for a in gt_raw:
        src_path = taxonomies.get(a["source"])
        tgt_path = taxonomies.get(a["target"])
        if not src_path or not tgt_path:
            continue
        alignment = _alignment_from_match_pairs(a["matches"], a["source"], a["target"])
        render_pair(src_path, tgt_path, alignment, gt_out)
    n_gt = len(list(gt_out.glob("*.dot")))
    print(f"  → {n_gt} files\n")

    # ── Render all cached result files ──
    if not RESULTS_DIR.exists():
        print("No results/ directory — nothing to render.")
        return

    # Walk subdirectories: each is an approach.
    by_approach: dict[str, list[Path]] = {}
    for subdir in sorted(RESULTS_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name == "viz":
            continue
        app = subdir.name
        by_approach[app] = sorted(subdir.glob("*.json"))

    for app in sorted(by_approach):
        app_out = OUT_DIR / app

        # Group files by (src, tgt).
        pair_data: dict[tuple[str, str], dict[str, Path]] = {}
        for rp in by_approach[app]:
            with open(rp, encoding="utf-8") as f:
                d = json.load(f)
            src = d.get("source")
            tgt = d.get("target")
            mode = d.get("mode")
            if not src or not tgt:
                continue
            key = (src, tgt)
            pair_data.setdefault(key, {})
            if mode:
                pair_data[key][mode] = rp
            else:
                pair_data[key]["__no_mode__"] = rp

        rendered = 0
        skipped_simple = 0
        skipped_ens = 0

        for (src, tgt), mode_map in sorted(pair_data.items()):
            src_path = taxonomies.get(src)
            tgt_path = taxonomies.get(tgt)
            if not src_path or not tgt_path:
                continue

            # Render individual mode files that have match_pairs.
            had_match = False
            for mode, rp in sorted(mode_map.items()):
                if mode == "__no_mode__":
                    label = f"{src}_{tgt}"
                else:
                    label = f"{src}_{tgt}_{mode}"

                with open(rp, encoding="utf-8") as f:
                    d = json.load(f)
                mp = d.get("match_pairs")
                if mp:
                    alignment = _alignment_from_match_pairs(mp, src, tgt)
                    render_pair(src_path, tgt_path, alignment, app_out, label=label)
                    rendered += 1
                    if mode != "__no_mode__" and mode != "ensemble":
                        had_match = True
                else:
                    skipped_simple += 1

            # Reconstruct ensemble from per-mode files (if skipping ensemble but have data).
            if had_match and "ensemble" in mode_map:
                # See if ensemble file has match_pairs already.
                ens_path = mode_map["ensemble"]
                with open(ens_path, encoding="utf-8") as f:
                    ens_d = json.load(f)
                if not ens_d.get("match_pairs"):
                    ens_alignment = _reconstruct_ensemble(
                        mode_map, src, tgt, app_out, src_path, tgt_path)
                    if ens_alignment is not None:
                        render_pair(src_path, tgt_path, ens_alignment, app_out,
                                    label=f"{src}_{tgt}_ensemble")
                        rendered += 1
                        skipped_ens += 1

        _count_and_skip(app, rendered, skipped_simple, skipped_ens)

    print(f"Done → {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
