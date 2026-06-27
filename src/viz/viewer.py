"""Interactive taxonomy viewer — tkinter Canvas with GraphViz PNG background.

Usage:
    uv run python -m src.viz.viewer

Renders taxonomies via GraphViz ``dot``, displays PNG background on a
tkinter Canvas, and overlays invisible rectangles on nodes for hover
tooltips and click-to-navigate.

Mouse:
    Scroll wheel  — zoom in/out centred on cursor.
    Left drag     — pan.
    Hover on node — tooltip with label, parents, mapped pair.
    Click on mapped node — jump view to its counterpart.
    R key / Reset View button — restore initial zoom.
"""

from __future__ import annotations

import io
import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from src.cache import load_cached
from src.data_loader import load_taxonomy
from src.matchers.embedding import EmbeddingMatcher
from src.metrics import evaluate_1to1
from src.matchers.string_equiv import StringEquivMatcher
from src.taxonomy import Alignment, TaxonMatch, Taxonomy
from src.viz.graphviz_render import (
    GraphLayout,
    layout_pair,
    layout_taxonomy,
    SRC_FILL,
    SRC_STROKE,
)

DATA_DIR = Path("data/processed/oaei")


# ── Helpers ──────────────────────────────────────────────────────────


def _load_all_taxonomies() -> dict[str, Taxonomy]:
    taxs: dict[str, Taxonomy] = {}
    for f in sorted(DATA_DIR.glob("*.json")):
        if f.name == "alignments.json":
            continue
        t = load_taxonomy(f)
        taxs[t.name] = t
    return taxs


def _load_alignments() -> list[dict]:
    path = DATA_DIR / "alignments.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _make_gt(src: str, tgt: str, alignments_raw: list[dict]) -> Alignment | None:
    for al in alignments_raw:
        if al["source"] == src and al["target"] == tgt:
            return Alignment(
                source=src, target=tgt,
                matches=[TaxonMatch(m["source_id"], m["target_id"],
                                    confidence=m.get("confidence", 1.0))
                         for m in al["matches"]],
            )
    return None


def _cached_alignment(cached: dict, src: str, tgt: str) -> Alignment:
    mp = cached.get("match_pairs", [])
    return Alignment(
        source=src, target=tgt,
        matches=[TaxonMatch(m["source_id"], m["target_id"],
                            confidence=m.get("confidence", 1.0))
                 for m in mp],
    )


# ── Viewer ───────────────────────────────────────────────────────────


class TaxonomyViewer:
    """Canvas-based interactive taxonomy viewer."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Taxonomy & Mapping Viewer")
        self.root.geometry("1500x850")

        # Data.
        self.taxonomies = _load_all_taxonomies()
        self.alignments_raw = _load_alignments()

        # Selection.
        self.selected_src: str | None = None
        self.selected_tgt: str | None = None

        # Matchers (lazy).
        self._embed_matcher: EmbeddingMatcher | None = None
        self._string_matcher: StringEquivMatcher | None = None

        # Layout cache.
        self._gl_src: GraphLayout | None = None
        self._gl_tgt: GraphLayout | None = None
        self._s2t: dict[str, str] = {}
        self._t2s: dict[str, str] = {}
        self._confidence: dict[tuple[str, str], float] = {}
        self._tgt_offset_x: int = 0

        # Interaction state.
        self._drag_start: tuple[float, float] | None = None
        self._hover_nid: str | None = None
        self._click_nid: str | None = None
        self._click_xy: tuple[float, float] | None = None
        # Hotzone item → (node_id, is_target).
        self._hot_items: dict[int, tuple[str, bool]] = {}
        # PIL image refs (must be kept alive).
        self._tk_images: list[ImageTk.PhotoImage] = []

        # Build.
        self._build_sidebar()
        self._build_main()
        self._connect_events()
        self._update_display()

    # ── UI construction ───────────────────────────────────────────────

    def _build_sidebar(self):
        frame = ttk.Frame(self.root, width=230)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
        frame.pack_propagate(False)

        ttk.Label(frame, text="Taxonomies (Ctrl+click for 2)",
                  font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 2))

        self.tax_list = tk.Listbox(frame, selectmode=tk.EXTENDED, height=10,
                                   exportselection=False)
        self.tax_list.pack(fill=tk.X, pady=(0, 6))
        for name in sorted(self.taxonomies.keys()):
            self.tax_list.insert(tk.END, name)
        self.tax_list.bind("<<ListboxSelect>>", self._on_tax_select)

        ttk.Label(frame, text="Alignment", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(4, 2))

        self.align_var = tk.StringVar(value="none")
        self.align_frame = ttk.Frame(frame)
        self.align_frame.pack(fill=tk.X)

        for key, label in [
            ("none", "None (taxonomy only)"),
            ("gt", "Ground truth"),
            ("embedding", "Embedding (LaBSE)"),
            ("string-equiv", "StringEquiv"),
            ("llm-gpt-concept", "LLM/gpt — C"),
            ("llm-gpt-concept-parent", "LLM/gpt — CP"),
            ("llm-gpt-concept-children", "LLM/gpt — CC"),
            ("llm-gpt-ensemble", "LLM/gpt — ENS"),
            ("llm-deepseek-concept", "LLM/ds — C"),
            ("llm-deepseek-concept-parent", "LLM/ds — CP"),
            ("llm-deepseek-concept-children", "LLM/ds — CC"),
            ("llm-deepseek-ensemble", "LLM/ds — ENS"),
        ]:
            rb = ttk.Radiobutton(self.align_frame, text=label,
                                 variable=self.align_var,
                                 value=key, command=self._on_align_change)
            rb.pack(anchor=tk.W)

        self.info_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.info_var, font=("", 8),
                  wraplength=210).pack(anchor=tk.W, pady=(8, 0))

        ttk.Button(frame, text="Reset View",
                   command=self._reset_view).pack(anchor=tk.W, pady=(8, 2))
        ttk.Label(frame,
                  text="Scroll = zoom  |  Drag = pan\n"
                       "Hover = info  |  Click = go to pair\n"
                       "R = reset view",
                  font=("", 7), foreground="gray").pack(anchor=tk.W)

    def _build_main(self):
        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.canvas = tk.Canvas(right, bg="#f5f5f5", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _connect_events(self):
        self.canvas.bind("<MouseWheel>", self._on_wheel)      # Win / macOS.
        self.canvas.bind("<Button-4>", self._on_wheel_up)     # Linux up.
        self.canvas.bind("<Button-5>", self._on_wheel_down)   # Linux down.
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_move)
        self.root.bind("r", lambda e: self._reset_view())
        self.root.bind("R", lambda e: self._reset_view())

    # ── Selection callbacks ───────────────────────────────────────────

    def _on_tax_select(self, _event):
        sel = self.tax_list.curselection()
        if len(sel) == 0:
            self.selected_src = None
            self.selected_tgt = None
        elif len(sel) == 1:
            self.selected_src = self.tax_list.get(sel[0])
            self.selected_tgt = None
        else:
            self.selected_src = self.tax_list.get(sel[0])
            self.selected_tgt = self.tax_list.get(sel[1])
        self._update_display()

    def _on_align_change(self):
        self._update_display()

    # ── Display update ────────────────────────────────────────────────

    def _update_display(self):
        """Full rebuild: layout, render, place interactive objects."""
        self.canvas.delete("all")
        self._tk_images.clear()
        self._hot_items.clear()
        self._hide_state()

        if self.selected_src is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Select one or two taxonomies from the list.\n"
                     "Use Ctrl+click for two.",
                font=("", 12), fill="gray",
            )
            return

        src_tax = self.taxonomies.get(self.selected_src)
        if src_tax is None:
            return
        tgt_tax = self.taxonomies.get(self.selected_tgt or "")

        # Resolve alignment.
        alignment, mode_label = self._resolve_alignment(src_tax, tgt_tax)

        # Layout via GraphViz.
        if tgt_tax is not None:
            self._gl_src, self._gl_tgt, self._s2t, self._t2s = layout_pair(
                src_tax, tgt_tax, alignment)
            self._confidence = {}
            if alignment:
                for m in alignment.matches:
                    if (m.source_id in src_tax.nodes and
                            m.target_id in tgt_tax.nodes):
                        self._confidence[(m.source_id, m.target_id)] = m.confidence
            self._render_pair(mode_label)
        else:
            self._gl_src = layout_taxonomy(src_tax, fill=SRC_FILL,
                                           stroke=SRC_STROKE)
            self._gl_tgt = None
            self._s2t = {}
            self._t2s = {}
            self._render_single()

        # Info.
        info = f"{src_tax.name} ({src_tax.node_count})"
        if tgt_tax is not None:
            info += f" ↔ {tgt_tax.name} ({tgt_tax.node_count})"
            if alignment and mode_label not in ("Ground truth",):
                gt = _make_gt(src_tax.name, tgt_tax.name, self.alignments_raw)
                if gt:
                    m = evaluate_1to1(alignment, gt)
                    info += f"\nF1={m.f1:.3f}  P={m.precision:.3f}  R={m.recall:.3f}"
            elif alignment:
                info += f"\nGT: {len(alignment.matches)} matches"
        if mode_label:
            info += f"\n{mode_label}"
        self.info_var.set(info)

    def _resolve_alignment(self, src: Taxonomy, tgt: Taxonomy):
        align_key = self.align_var.get()
        alignment: Alignment | None = None
        label = ""

        if not tgt:
            return None, ""

        if align_key == "gt":
            alignment = _make_gt(src.name, tgt.name, self.alignments_raw)
            label = "Ground truth"
        elif align_key == "embedding":
            cached = load_cached("embedding-LaBSE", src.name, tgt.name)
            if cached:
                alignment = _cached_alignment(cached, src.name, tgt.name)
                label = "Embedding (cached)"
            else:
                if self._embed_matcher is None:
                    self._embed_matcher = EmbeddingMatcher(
                        model_name="sentence-transformers/LaBSE")
                alignment, _ = self._embed_matcher.match(src, tgt)
                label = "Embedding (LaBSE)"
        elif align_key == "string-equiv":
            cached = load_cached("string-equiv", src.name, tgt.name)
            if cached:
                alignment = _cached_alignment(cached, src.name, tgt.name)
                label = "StringEquiv (cached)"
            else:
                if self._string_matcher is None:
                    self._string_matcher = StringEquivMatcher()
                alignment, _ = self._string_matcher.match(src, tgt)
                label = "StringEquiv"
        elif align_key.startswith("llm-"):
            approach = "llm-gpt" if "gpt" in align_key else "llm-deepseek"
            mode = align_key.split("-", 2)[-1]
            cached = load_cached(approach, src.name, tgt.name, mode)
            if cached:
                alignment = _cached_alignment(cached, src.name, tgt.name)
                f1_str = f" F1={cached['f1']:.3f}"
                label = f"{approach} {mode}{f1_str}"
            else:
                label = f"{approach} {mode} (not cached)"

        return alignment, label

    # ── Rendering ─────────────────────────────────────────────────────

    def _render_single(self):
        gl = self._gl_src
        if gl is None:
            return
        img = ImageTk.PhotoImage(Image.open(io.BytesIO(gl.png_bytes)))
        self._tk_images.append(img)
        self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
        self._place_hotzones(gl, is_target=False, off_x=0)

    def _render_pair(self, mode_label: str):
        gl_src = self._gl_src
        gl_tgt = self._gl_tgt
        if gl_src is None or gl_tgt is None:
            return

        # Place backgrounds side by side.
        img_src = ImageTk.PhotoImage(Image.open(io.BytesIO(gl_src.png_bytes)))
        img_tgt = ImageTk.PhotoImage(Image.open(io.BytesIO(gl_tgt.png_bytes)))
        self._tk_images.extend([img_src, img_tgt])

        gap = 80
        self._tgt_offset_x = gl_src.png_width + gap

        self.canvas.create_image(0, 0, image=img_src, anchor=tk.NW)
        self.canvas.create_image(self._tgt_offset_x, 0, image=img_tgt, anchor=tk.NW)

        self._place_hotzones(gl_src, is_target=False, off_x=0)
        self._place_hotzones(gl_tgt, is_target=True, off_x=self._tgt_offset_x)

        # Mapping lines.
        self._draw_mapping_lines(gl_src, gl_tgt)

        # Legend below graphs.
        max_h = max(gl_src.png_height, gl_tgt.png_height)
        self.canvas.create_text(
            self._tgt_offset_x // 2, max_h + 16,
            text=f"{mode_label}: {len(self._s2t)} matches",
            font=("", 8), fill="#555555",
        )

    def _place_hotzones(self, gl: GraphLayout, is_target: bool, off_x: int):
        for nid, nb in gl.nodes.items():
            x1 = off_x + nb.x - nb.w / 2
            y1 = nb.y - nb.h / 2
            x2 = off_x + nb.x + nb.w / 2
            y2 = nb.y + nb.h / 2
            item = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill="", outline="",   # invisible.
            )
            self._hot_items[item] = (nid, is_target)

    def _draw_mapping_lines(self, gl_src: GraphLayout, gl_tgt: GraphLayout):
        for sid, tid in self._s2t.items():
            sb = gl_src.nodes.get(sid)
            tb = gl_tgt.nodes.get(tid)
            if sb is None or tb is None:
                continue
            conf = self._confidence.get((sid, tid), 0.5)
            g = int(40 + 150 * conf)
            r = int(40 + 50 * (1 - conf))
            b = int(40 + 100 * (1 - conf))
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas.create_line(
                sb.x, sb.y,
                self._tgt_offset_x + tb.x, tb.y,
                fill=color, width=1.0 + 1.5 * conf,
                dash=(4, 3),
            )

    # ── Mouse events ──────────────────────────────────────────────────

    def _canvas_xy(self, event):
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def _on_wheel(self, event):
        """Windows/macOS scroll wheel."""
        cx, cy = self._canvas_xy(event)
        factor = 1.15 if event.delta > 0 else 1.0 / 1.15
        self.canvas.scale("all", cx, cy, factor, factor)
        self._hide_tooltip()

    def _on_wheel_up(self, event):
        cx, cy = self._canvas_xy(event)
        self.canvas.scale("all", cx, cy, 1.15, 1.15)
        self._hide_tooltip()

    def _on_wheel_down(self, event):
        cx, cy = self._canvas_xy(event)
        self.canvas.scale("all", cx, cy, 1.0 / 1.15, 1.0 / 1.15)
        self._hide_tooltip()

    def _on_press(self, event):
        self._drag_start = self._canvas_xy(event)
        self._click_xy = self._canvas_xy(event)
        item = self._find_hotzone(event)
        if item is not None:
            self._click_nid, _ = self._hot_items[item]
        else:
            self._click_nid = None

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        cur = self._canvas_xy(event)
        sx, sy = self._drag_start
        dx, dy = cur[0] - sx, cur[1] - sy
        self.canvas.move("all", dx, dy)
        self._drag_start = cur
        self._hide_tooltip()

    def _on_release(self, event):
        if self._click_nid is not None and self._click_xy is not None:
            cur = self._canvas_xy(event)
            if (abs(cur[0] - self._click_xy[0]) < 3 and
                    abs(cur[1] - self._click_xy[1]) < 3):
                self._navigate_to_pair(self._click_nid)
        self._drag_start = None
        self._click_nid = None
        self._click_xy = None

    def _on_move(self, event):
        if self._drag_start is not None:
            return
        item = self._find_hotzone(event)
        if item is not None:
            nid, is_tgt = self._hot_items[item]
            if nid != self._hover_nid:
                self._show_tooltip(event, nid, is_tgt)
        else:
            self._hide_tooltip()

    # ── Hit testing ───────────────────────────────────────────────────

    def _find_hotzone(self, event) -> int | None:
        cx, cy = self._canvas_xy(event)
        overlapping = self.canvas.find_overlapping(cx, cy, cx, cy)
        for item in overlapping:
            if item in self._hot_items:
                return item
        return None

    # ── Tooltip ───────────────────────────────────────────────────────

    def _show_tooltip(self, event, nid: str, is_target: bool):
        self._hide_tooltip()
        self._hover_nid = nid

        # Build tooltip text.
        tax = (self.taxonomies.get(self.selected_tgt or "")
               if is_target
               else self.taxonomies.get(self.selected_src or ""))
        lines = [f"Node: {nid}"]
        if tax:
            node = tax.nodes.get(nid)
            if node:
                lines.append(f"Label: {node.name}")
                if node.parents:
                    pnames = ", ".join(
                        tax.nodes[p].name for p in node.parents
                        if p in tax.nodes
                    )
                    if pnames:
                        lines.append(f"Parents: {pnames}")

        # Mapped pair.
        if is_target:
            sid = self._t2s.get(nid)
            src_tax = self.taxonomies.get(self.selected_src or "")
            if sid and src_tax and sid in src_tax.nodes:
                conf = self._confidence.get((sid, nid), 0)
                lines.append(f"↔ {src_tax.nodes[sid].name}  (conf={conf:.2f})")
        else:
            tid = self._s2t.get(nid)
            tgt_tax = self.taxonomies.get(self.selected_tgt or "")
            if tid and tgt_tax and tid in tgt_tax.nodes:
                conf = self._confidence.get((nid, tid), 0)
                lines.append(f"↔ {tgt_tax.nodes[tid].name}  (conf={conf:.2f})")

        cx, cy = self._canvas_xy(event)
        text = "\n".join(lines)
        tid = self.canvas.create_text(
            cx + 14, cy + 10, text=text, anchor=tk.NW,
            font=("", 8), fill="#222222",
            tags=("tooltip",),
        )
        bbox = self.canvas.bbox(tid)
        if bbox:
            self.canvas.create_rectangle(
                bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3,
                fill="#ffffcc", outline="#aaaaaa",
                tags=("tooltip",),
            )
            self.canvas.tag_raise("tooltip")
        # Keep ref for cleanup.
        self._tooltip_ids = self.canvas.find_withtag("tooltip")

    def _hide_tooltip(self):
        self.canvas.delete("tooltip")
        self._hover_nid = None

    def _hide_state(self):
        self._hide_tooltip()
        self._hover_nid = None
        self._drag_start = None
        self._click_nid = None
        self._click_xy = None

    # ── Navigate ──────────────────────────────────────────────────────

    def _navigate_to_pair(self, nid: str):
        """Jump view: center on counterpart node."""
        # Determine which side we're on and where the counterpart is.
        if nid in self._s2t:
            # nid is in source; counterpart in target.
            tid = self._s2t[nid]
            gl = self._gl_tgt
            off_x = self._tgt_offset_x
        elif nid in self._t2s:
            tid = self._t2s[nid]
            gl = self._gl_src
            off_x = 0
        else:
            return

        if gl is None:
            return
        nb = gl.nodes.get(tid)
        if nb is None:
            return

        tx = off_x + nb.x
        ty = nb.y
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # Reset to identity transform, then center on target.
        self._reset_view()
        self.canvas.move("all", cw / 2 - tx, ch / 2 - ty)

    def _reset_view(self):
        """Full re-render to restore initial view."""
        self._hide_state()
        self._update_display()

    # ── Run ───────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


def main():
    app = TaxonomyViewer()
    app.run()


if __name__ == "__main__":
    main()
