"""Tests for the ancestor chain that becomes a node's path text.

The chain feeds the textual description of a node (`name_path` and `full`
modes), so a change of its definition silently changes what every model sees.
The contract pinned here is the one the docstring promises: the *shortest*
chain, resolved by node id so the result depends on the graph alone and not on
the order of the parent lists in the JSON.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

from src.taxonomy import Taxonomy, TaxonNode

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "oaei"


def build(edges: dict[str, list[str]]) -> Taxonomy:
    """Build a taxonomy from a ``node id → parent ids`` map."""
    nodes: dict[str, TaxonNode] = {}
    for node_id in edges:
        nodes[node_id] = TaxonNode(id=node_id, name=node_id)
    for node_id, parents in edges.items():
        nodes[node_id].parents = list(parents)
        for parent in parents:
            nodes[parent].children.append(node_id)
    return Taxonomy(name="t", namespace="http://example.org/", nodes=nodes)


def load_real(name: str) -> Taxonomy:
    """Load a processed taxonomy without going through the data loader."""
    raw = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    return Taxonomy(
        name=raw["name"],
        namespace=raw["namespace"],
        root_id=raw.get("root_id"),
        nodes={node_id: TaxonNode(**node) for node_id, node in raw["nodes"].items()},
    )


def test_chain_takes_the_shortest_path():
    # Two chains reach `c`: r → b → c and the direct r → c.
    tax = build({"r": [], "a": ["r"], "b": ["a"], "c": ["b", "r"]})
    assert tax.get_representative_chain("c") == ["r", "c"]
    assert tax.get_representative_chain("b") == ["r", "a", "b"]
    assert tax.get_representative_chain("r") == ["r"]


def test_chain_is_independent_of_parent_order():
    tax = build({"r1": [], "r2": [], "b": ["r1", "r2"]})
    for first, second in itertools.permutations(["r1", "r2"]):
        tax.nodes["b"].parents = [first, second]
        assert tax.get_representative_chain("b") == ["r1", "b"]


def test_chain_of_an_unknown_node_is_empty():
    assert build({"r": []}).get_representative_chain("missing") == []


def test_ancestors_equal_the_transitive_closure():
    tax = build({"r": [], "a": ["r"], "b": ["a"], "c": ["b", "r"], "d": ["c", "b"]})
    assert tax.get_ancestors("d") == {"c", "b", "a", "r"}
    assert tax.get_ancestors("r") == set()


def test_a_cycle_does_not_make_a_node_its_own_ancestor():
    tax = build({"x": ["y"], "y": ["x"]})
    assert tax.get_representative_chain("x") == ["x"]
    assert tax.get_ancestors("x") == {"y"}


def test_real_chains_are_stable_under_parent_permutation():
    for ontology in ("cmt", "confOf", "conference", "edas", "ekaw", "iasted", "sigkdd"):
        base = load_real(ontology)
        expected = {nid: base.get_representative_chain(nid) for nid in base.nodes}

        reshuffled = load_real(ontology)
        rng = random.Random(20260926)
        for node in reshuffled.nodes.values():
            rng.shuffle(node.parents)

        got = {nid: reshuffled.get_representative_chain(nid) for nid in reshuffled.nodes}
        assert got == expected, f"{ontology}: цепочка зависит от порядка родителей"
