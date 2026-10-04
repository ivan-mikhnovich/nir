#!/usr/bin/env python3
"""Генерирует изображения для презентации НИР.

Запускается из Makefile перед компиляцией Beamer.
При каждом запуске перегенерирует две схемы.
"""

import os
import shutil
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")


def _make_matching_diagram(path: str) -> None:
    """GraphViz-иллюстрация: условное сопоставление двух таксономий.

    Показывает иерархии классов и соответствия между ними.
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
        "  // Source taxonomy.",
        '  subgraph cluster_src {',
        '    label="Таксономия A (пример)"',
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
        "  // Target taxonomy.",
        '  subgraph cluster_tgt {',
        '    label="Таксономия B (пример)"',
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
        "  // Class correspondences.",
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
        '    label="Таксономия A"',
        '    style=dashed',
        '    color="#5ba3d9"',
        '    fontsize=13',
        "",
        '    src_graph [label="Имена классов", fillcolor="#e8f4fd"]',
        '    src_edges [label="Связи is-a", fillcolor="#e8f4fd"]',
        '    src_bert  [label="MiniLM\\nпризнаки: 384", fillcolor="#d4e6f1"]',
        '    src_gnn   [label="GraphSAGE\\nодин слой: 384 → 384", fillcolor="#aed6f1"]',
        '    src_emb   [label="L2-нормировка", fillcolor="#85c1e9"]',
        "",
        "    src_graph -> src_bert -> src_gnn -> src_emb",
        "    src_edges -> src_gnn",
        "  }",
        "",
        '  subgraph cluster_tgt {',
        '    label="Таксономия B"',
        '    style=dashed',
        '    color="#d95b5b"',
        '    fontsize=13',
        "",
        '    tgt_graph [label="Имена классов", fillcolor="#fde8e8"]',
        '    tgt_edges [label="Связи is-a", fillcolor="#fde8e8"]',
        '    tgt_bert  [label="MiniLM\\nпризнаки: 384", fillcolor="#f5c6c6"]',
        '    tgt_gnn   [label="GraphSAGE\\nодин слой: 384 → 384", fillcolor="#f1948a"]',
        '    tgt_emb   [label="L2-нормировка", fillcolor="#e74c3c", fontcolor=white]',
        "",
        "    tgt_graph -> tgt_bert -> tgt_gnn -> tgt_emb",
        "    tgt_edges -> tgt_gnn",
        "  }",
        "",
        '  cosine [label="Косинусное сходство\\nматрица [Nₛ × Nₜ]", shape=box, style="rounded,filled",',
        '           fillcolor="#f9e79f", fontsize=12]',
        '  match  [label="Жадный отбор 1:1\nпо убыванию сходства", shape=box, style="rounded,filled",',
        '           fillcolor="#abebc6", fontsize=12]',
        "",
        '  src_gnn -> tgt_gnn [label="Общие параметры", style=dashed, dir=none, constraint=false]',
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

    path_m = os.path.join(IMAGES_DIR, "matching_diagram.png")
    _make_matching_diagram(path_m)

    path_g = os.path.join(IMAGES_DIR, "gnn_architecture.png")
    _make_gnn_architecture(path_g)


if __name__ == "__main__":
    main()
