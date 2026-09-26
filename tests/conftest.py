"""Shared helpers for the test suite."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.taxonomy import Alignment, TaxonMatch, Taxonomy, TaxonNode


def build_taxonomy(
    edges: dict[str, list[str]] | None = None,
    **nodes: dict,
) -> Taxonomy:
    """Build a taxonomy from a ``node id → parent ids`` map.

    Extra nodes can be passed as keyword arguments, e.g.
    ``build_taxonomy({"a": []}, b={"name": "B", "parents": ["a"]})``.
    """
    edges = dict(edges or {})
    ids = list(edges) + list(nodes)
    graph: dict[str, dict] = {}
    for node_id in ids:
        graph[node_id] = {"parents": list(edges.get(node_id, [])), "name": node_id}
    for node_id, extra in nodes.items():
        graph[node_id].update(extra)

    taxa = Taxonomy(name="t", namespace="http://example.org/", nodes={})
    for node_id, spec in graph.items():
        taxa.nodes[node_id] = TaxonNode(
            id=node_id,
            name=spec["name"],
            parents=list(spec["parents"]),
            children=[],
            attributes=dict(spec.get("attributes", {})),
            disjoint_with=list(spec.get("disjoint_with", [])),
            comment=spec.get("comment", ""),
            depth=spec.get("depth", 0),
        )
    for node_id, spec in graph.items():
        for parent in spec["parents"]:
            if parent in taxa.nodes:
                taxa.nodes[parent].children.append(node_id)
    return taxa


def build_alignment(
    source: str,
    target: str,
    pairs: list[tuple[str, str]],
    confidence: float = 1.0,
) -> Alignment:
    """Build an alignment from ``(source_id, target_id)`` pairs."""
    return Alignment(
        source=source,
        target=target,
        matches=[TaxonMatch(s, t, confidence=confidence) for s, t in pairs],
    )


@pytest.fixture
def taxonomy() -> Callable[..., Taxonomy]:
    """Factory fixture returning :func:`build_taxonomy`."""
    return build_taxonomy


@pytest.fixture
def alignment() -> Callable[..., Alignment]:
    """Factory fixture returning :func:`build_alignment`."""
    return build_alignment
