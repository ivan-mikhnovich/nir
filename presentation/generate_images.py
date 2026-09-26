#!/usr/bin/env python3
"""Генерирует изображения для презентации НИР.

Запускается из Makefile перед компиляцией Beamer.
При каждом запуске перегенерирует три изображения.
"""

import os
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")


def _make_comparison_six_pairs(path: str) -> None:
    """Столбчатая диаграмма: F1 всех подходов на шести LLM-парах."""
    pairs = [
        "cmt\n↔\nconfOf",
        "cmt\n↔\nconference",
        "confOf\n↔\nconference",
        "cmt\n↔\nedas",
        "confOf\n↔\nekaw",
        "conference\n↔\nekaw",
    ]

    labels = [
        "StringEquiv",
        "LaBSE\n(без порога)",
        "ruRoberta-large\n(порог)",
        "GNN",
        "LLM: BM25+DS\n(ансамбль)",
        "LLM: гибрид+DS\n(ансамбль)",
    ]

    # F1 per approach per pair (единое правило 1:1; порог по leave-one-pair-out;
    # LLM — ансамбль «не менее двух из трёх»).
    data = {
        "StringEquiv":            [0.533, 0.375, 0.737, 1.000, 0.552, 0.457],
        "LaBSE (без порога)":     [0.359, 0.500, 0.367, 0.432, 0.552, 0.463],
        "ruRoberta-large (порог)":[0.533, 0.444, 0.737, 1.000, 0.687, 0.500],
        "GNN":                    [0.359, 0.400, 0.327, 0.432, 0.586, 0.390],
        "LLM: BM25+DS (ансамбль)":[0.429, 0.375, 0.778, 0.933, 0.667, 0.424],
        "LLM: гибрид+DS (ансамбль)":[0.667, 0.588, 0.667, 0.875, 0.667, 0.514],
    }

    n_pairs = len(pairs)
    n_groups = len(labels)
    x = np.arange(n_pairs)
    width = 0.13
    offsets = np.linspace(
        -(n_groups - 1) * width / 2,
        (n_groups - 1) * width / 2,
        n_groups,
    )

    # Цветовая схема: baseline → embedding → GNN → LLM.
    colors = [
        "#95A5A6",  # StringEquiv (grey).
        "#3498DB",  # LaBSE raw.
        "#1F618D",  # ruRoberta-large с порогом.
        "#8E44AD",  # GNN.
        "#E67E22",  # BM25 + DS.
        "#27AE60",  # Гибрид + DS.
    ]

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for i, (label, vals) in enumerate(data.items()):
        bars = ax.bar(
            x + offsets[i], vals, width, label=label,
            color=colors[i], edgecolor="white", linewidth=0.3,
        )
        # Подписать значения над высокими столбцами.
        for bar, val in zip(bars, vals):
            if val >= 0.6:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center", fontsize=6.5, fontweight="bold",
                )

    ax.set_title(
        "F1 на шести показательных парах: единое правило 1:1, "
        "порог по leave-one-pair-out, LLM — ансамбль «не менее двух из трёх»",
        fontsize=10,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=9)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))

    ax.legend(
        loc="upper left", fontsize=7.5, ncol=3,
        framealpha=0.9, edgecolor="#ccc",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Сгенерирован: {os.path.relpath(path, SCRIPT_DIR)}")


def _make_matching_diagram(path: str) -> None:
    """GraphViz-иллюстрация: мэтчинг двух таксономий.

    Рисует две таксономии рядом с матч-стрелками между ними.
    """
    dot_path = path.replace(".png", ".dot")

    dot = [
        "digraph matching {",
        "  rankdir=TB",
        '  bgcolor="transparent"',
        "  nodesep=0.7",
        "  ranksep=0.8",
        '  node [shape=box, style="rounded,filled", fontname="Arial",',
        '        fontsize=13, margin="0.15,0.10"]',
        '  edge [color="#909090", penwidth=1.1, arrowhead=none]',
        "",
        "  // ── Source (left) ──",
        '  subgraph cluster_src {',
        '    label="CMT"',
        '    style=dashed',
        '    color="#5ba3d9"',
        '    fontsize=14',
        "",
        '    s_Document   [label="Document",    fillcolor="#e8f4fd", color="#5ba3d9" penwidth=1.3]',
        '    s_Paper      [label="Paper",       fillcolor="#e8f4fd", color="#2ca02c" penwidth=2.5]',
        '    s_Poster     [label="Poster",      fillcolor="#e8f4fd", color="#2ca02c" penwidth=2.5]',
        '    s_Person     [label="Person",      fillcolor="#e8f4fd", color="#2ca02c" penwidth=2.5]',
        '    s_Author     [label="Author",      fillcolor="#e8f4fd", color="#2ca02c" penwidth=2.5]',
        '    s_Reviewer   [label="Reviewer",    fillcolor="#e8f4fd", color="#5ba3d9" penwidth=1.3]',
        "",
        "    s_Document -> s_Paper",
        "    s_Document -> s_Poster",
        "    s_Person -> s_Author",
        "    s_Person -> s_Reviewer",
        "  }",
        "",
        "  // ── Target (right) ──",
        '  subgraph cluster_tgt {',
        '    label="ConfOf"',
        '    style=dashed',
        '    color="#d95b5b"',
        '    fontsize=14',
        "",
        '    t_Contribution  [label="Contribution",   fillcolor="#fde8e8", color="#d95b5b" penwidth=1.3]',
        '    t_Paper         [label="Paper",          fillcolor="#fde8e8", color="#2ca02c" penwidth=2.5]',
        '    t_Poster        [label="Poster",         fillcolor="#fde8e8", color="#2ca02c" penwidth=2.5]',
        '    t_Human         [label="Person",         fillcolor="#fde8e8", color="#2ca02c" penwidth=2.5]',
        '    t_Author        [label="Author",         fillcolor="#fde8e8", color="#2ca02c" penwidth=2.5]',
        '    t_Committee     [label="CommitteeMember", fillcolor="#fde8e8", color="#d95b5b" penwidth=1.3]',
        "",
        "    t_Contribution -> t_Paper",
        "    t_Contribution -> t_Poster",
        "    t_Human -> t_Author",
        "    t_Human -> t_Committee",
        "  }",
        "",
        "  // ── Match edges ──",
        '  edge [color="#2ca02c", penwidth=2.0, style=dashed, arrowhead=none]',
        "  s_Paper  -> t_Paper",
        "  s_Poster -> t_Poster",
        "  s_Person -> t_Human",
        "  s_Author -> t_Author",
        "}",
    ]

    dot_text = "\n".join(dot)
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_text)

    # Render via dot (GraphViz).
    if shutil.which("dot"):
        subprocess.run(
            ["dot", "-Tpng", "-Gdpi=180", "-o", path, dot_path],
            check=True,
        )
        print(f"Сгенерирован: {os.path.relpath(path, SCRIPT_DIR)}")
    else:
        # Fallback: just write the dot and warn.
        print("GraphViz 'dot' not found — skipping PNG render.")
        print(f"Написан .dot: {os.path.relpath(dot_path, SCRIPT_DIR)}")


def _make_gnn_architecture(path: str) -> None:
    """GraphViz-диаграмма: архитектура Siamese GraphSAGE."""
    dot_path = path.replace(".png", ".dot")

    dot = [
        "digraph gnn {",
        "  rankdir=LR",
        '  bgcolor="transparent"',
        "  nodesep=0.4",
        "  ranksep=0.7",
        '  node [shape=box, style="rounded,filled", fontname="Arial",',
        '        fontsize=11, margin="0.12,0.08"]',
        '  edge [arrowhead=normal, penwidth=1.2]',
        "",
        '  subgraph cluster_src {',
        '    label="Таксономия A (source)"',
        '    style=dashed',
        '    color="#5ba3d9"',
        '    fontsize=13',
        "",
        '    src_graph [label="Граф (узлы + рёбра)", fillcolor="#e8f4fd"]',
        '    src_bert  [label="MiniLM (384-dim)", fillcolor="#d4e6f1"]',
        '    src_gnn   [label="GraphSAGE\n(1 слой, residual)\n384 → 384", fillcolor="#aed6f1"]',
        '    src_emb   [label="L2-norm\n(384-dim)", fillcolor="#85c1e9"]',
        "",
        "    src_graph -> src_bert -> src_gnn -> src_emb",
        "  }",
        "",
        '  subgraph cluster_tgt {',
        '    label="Таксономия B (target)"',
        '    style=dashed',
        '    color="#d95b5b"',
        '    fontsize=13',
        "",
        '    tgt_graph [label="Граф (узлы + рёбра)", fillcolor="#fde8e8"]',
        '    tgt_bert  [label="MiniLM (384-dim)", fillcolor="#f5c6c6"]',
        '    tgt_gnn   [label="GraphSAGE\n(1 слой, residual)\n384 → 384", fillcolor="#f1948a"]',
        '    tgt_emb   [label="L2-norm\n(384-dim)", fillcolor="#e74c3c", fontcolor=white]',
        "",
        "    tgt_graph -> tgt_bert -> tgt_gnn -> tgt_emb",
        "  }",
        "",
        '  cosine [label="Cosine similarity\nматрица [Nₛ × Nₜ]", shape=box, style="rounded,filled",',
        '           fillcolor="#f9e79f", fontsize=12]',
        '  match  [label="Жадный отбор 1:1\nпо убыванию сходства", shape=box, style="rounded,filled",',
        '           fillcolor="#abebc6", fontsize=12]',
        "",
        '  src_emb -> cosine [color="#5ba3d9", penwidth=1.5]',
        '  tgt_emb -> cosine [color="#d95b5b", penwidth=1.5]',
        '  cosine -> match [penwidth=1.5]',
        "}",
    ]

    dot_text = "\n".join(dot)
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot_text)

    if shutil.which("dot"):
        subprocess.run(
            ["dot", "-Tpng", "-Gdpi=150", "-o", path, dot_path],
            check=True,
        )
        print(f"Сгенерирован: {os.path.relpath(path, SCRIPT_DIR)}")
    else:
        print("GraphViz 'dot' not found — skipping PNG render.")
        print(f"Написан .dot: {os.path.relpath(dot_path, SCRIPT_DIR)}")


def main() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)

    path_6 = os.path.join(IMAGES_DIR, "comparison_six_pairs.png")
    _make_comparison_six_pairs(path_6)

    path_m = os.path.join(IMAGES_DIR, "matching_diagram.png")
    _make_matching_diagram(path_m)

    path_g = os.path.join(IMAGES_DIR, "gnn_architecture.png")
    _make_gnn_architecture(path_g)


if __name__ == "__main__":
    main()
