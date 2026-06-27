"""GraphViz-based taxonomy renderer — DOT generation + position extraction.

Renders via GraphViz ``dot`` (pad=0, DPI=144), then extracts node
geometry from JSON output.  Node coordinates are converted to PNG
pixel space (y-flipped) so the viewer can overlay interactive zones.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import graphviz  # type: ignore
from PIL import Image

from src.taxonomy import Alignment, Taxonomy

# ── Colours ───────────────────────────────────────────────────────────

SRC_FILL = "#e8f4fd"
SRC_STROKE = "#5ba3d9"
TGT_FILL = "#fde8e8"
TGT_STROKE = "#d95b5b"
MATCH_STROKE = "#2ca02c"
MATCH_PW = "2.5"
EDGE_COLOR = "#b0b0b0"

FONT_NAME = "Arial"
FONT_SIZE = "10"
DPI = 144


@dataclass
class NodeBox:
    """A node's geometry in PNG pixel coordinates (origin = top-left)."""
    node_id: str
    label: str
    x: float          # centre x (px).
    y: float          # centre y (px, top=0).
    w: float          # width (px).
    h: float          # height (px).
    matched: bool


@dataclass
class GraphLayout:
    """Layout data for one taxonomy."""
    nodes: dict[str, NodeBox]
    png_bytes: bytes
    png_width: int
    png_height: int


def _safe_id(nid: str) -> str:
    return nid.replace("#", "_").replace(".", "_").replace(" ", "_")


def _build_dot(
    taxonomy: Taxonomy,
    fill: str = SRC_FILL,
    stroke: str = SRC_STROKE,
    matched_ids: set[str] | None = None,
) -> graphviz.Digraph:
    matched_ids = matched_ids or set()
    short = taxonomy.name.replace(".", "_").replace(" ", "_")
    dot = graphviz.Digraph(
        name=f"g_{short}", format="png", engine="dot",
    )
    dot.attr(rankdir="TB", nodesep="0.5", ranksep="0.9",
             bgcolor="transparent", pad="0", dpi=str(DPI))
    dot.attr("node", shape="box", style="rounded,filled",
             fontname=FONT_NAME, fontsize=FONT_SIZE,
             margin="0.14,0.08")

    for nid, node in taxonomy.nodes.items():
        attrs = {
            "fillcolor": fill,
            "color": MATCH_STROKE if nid in matched_ids else stroke,
            "penwidth": MATCH_PW if nid in matched_ids else "1.3",
        }
        dot.node(_safe_id(nid), node.name, **attrs)

    for nid, node in taxonomy.nodes.items():
        for pid in node.parents:
            if pid in taxonomy.nodes:
                dot.edge(_safe_id(pid), _safe_id(nid),
                         color=EDGE_COLOR, penwidth="0.9",
                         arrowhead="none")
    return dot


def layout_taxonomy(
    taxonomy: Taxonomy,
    fill: str = SRC_FILL,
    stroke: str = SRC_STROKE,
    matched_ids: set[str] | None = None,
) -> GraphLayout:
    """Render taxonomy via GraphViz, extract node positions in PNG pixels."""
    matched_ids = matched_ids or set()
    id_map = {_safe_id(nid): nid for nid in taxonomy.nodes}

    dot = _build_dot(taxonomy, fill=fill, stroke=stroke,
                     matched_ids=matched_ids)

    # JSON for node geometry.
    jdata = json.loads(dot.pipe(format="json"))
    bb_str: str = jdata.get("bb", "0,0,100,100")
    parts = [float(p) for p in bb_str.split(",")]
    bb_x0, bb_y0, bb_x1, bb_y1 = parts
    bb_w, bb_h = bb_x1 - bb_x0, bb_y1 - bb_y0

    # PNG for background.
    png_bytes = dot.pipe(format="png")
    img = Image.open(io.BytesIO(png_bytes))
    png_w, png_h = img.size

    # Scale factors: DOT point → PNG pixel.
    scale_x = png_w / bb_w if bb_w > 0 else 1.0
    scale_y = png_h / bb_h if bb_h > 0 else 1.0
    # Use uniform scale (both should be DPI/72).
    scale = min(scale_x, scale_y)

    nodes: dict[str, NodeBox] = {}
    for obj in jdata.get("objects", []):
        name = obj.get("name", "")
        nid = id_map.get(name)
        if nid is None:
            continue

        bbox = None
        label = ""
        for draw in obj.get("_draw_", []):
            op = draw.get("op", "")
            if op == "B":
                pts = draw.get("points", [])
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    bbox = (min(xs), min(ys), max(xs), max(ys))
            elif op == "T":
                label = draw.get("text", "")

        if bbox is None:
            continue

        lx, by, rx, ty = bbox
        # Convert DOT coords (y-up, origin=bottom-left) → PNG coords (y-down, origin=top-left).
        nodes[nid] = NodeBox(
            node_id=nid,
            label=label or taxonomy.nodes[nid].name,
            x=(lx + rx) / 2 * scale,
            y=(bb_h - (by + ty) / 2) * scale,
            w=(rx - lx) * scale,
            h=(ty - by) * scale,
            matched=nid in matched_ids,
        )

    return GraphLayout(
        nodes=nodes,
        png_bytes=png_bytes,
        png_width=png_w,
        png_height=png_h,
    )


def layout_pair(
    source: Taxonomy,
    target: Taxonomy,
    alignment: Alignment | None = None,
) -> tuple[GraphLayout, GraphLayout, dict[str, str], dict[str, str]]:
    """Layout both taxonomies, return graphs + mapping dicts."""
    s2t: dict[str, str] = {}
    t2s: dict[str, str] = {}
    msrc: set[str] = set()
    mtgt: set[str] = set()

    if alignment:
        for m in alignment.matches:
            if m.source_id in source.nodes and m.target_id in target.nodes:
                msrc.add(m.source_id)
                mtgt.add(m.target_id)
                s2t[m.source_id] = m.target_id
                t2s[m.target_id] = m.source_id

    gl_src = layout_taxonomy(source, fill=SRC_FILL, stroke=SRC_STROKE,
                             matched_ids=msrc)
    gl_tgt = layout_taxonomy(target, fill=TGT_FILL, stroke=TGT_STROKE,
                             matched_ids=mtgt)
    return gl_src, gl_tgt, s2t, t2s
