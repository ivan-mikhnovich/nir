"""Data structures for taxonomy representation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaxonNode:
    """A single node in a taxonomy tree."""

    id: str  # Unique identifier (e.g., 'cmt#Author').
    name: str  # Human-readable name (e.g., 'Author').
    parents: list[str] = field(default_factory=list)  # IDs of parent nodes ('is-a' links).
    children: list[str] = field(default_factory=list)  # IDs of child nodes.
    attributes: dict[str, str] = field(default_factory=dict)  # Key-value properties.
    disjoint_with: list[str] = field(default_factory=list)  # IDs of disjoint classes.
    comment: str = ""  # rdfs:comment text.
    depth: int = 0  # Distance from root.


@dataclass
class Taxonomy:
    """A taxonomy (hierarchy of classes with 'is-a' relations)."""

    name: str  # Short name (e.g., 'cmt').
    namespace: str  # Ontology namespace URI.
    nodes: dict[str, TaxonNode] = field(default_factory=dict)
    root_id: str | None = None  # ID of the root node (usually owl:Thing).

    def get_path(self, node_id: str) -> list[str]:
        """Return the path from root to node as a list of node names."""
        path: list[str] = []
        current = node_id
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            node = self.nodes.get(current)
            if node is None:
                break
            path.append(node.name)
            if node.parents:
                current = node.parents[0]  # Take first parent for path.
            else:
                break
        return list(reversed(path))

    def get_textual_description(self, node_id: str) -> str:
        """Build a textual representation of a node for embedding/LLM input."""
        node = self.nodes.get(node_id)
        if node is None:
            return ""

        parts: list[str] = []
        parts.append(f"Name: {node.name}")

        path = self.get_path(node_id)
        if len(path) > 1:
            parts.append(f"Path: {' -> '.join(path)}")

        if node.attributes:
            attr_str = ", ".join(f"{k}: {v}" for k, v in sorted(node.attributes.items()))
            parts.append(f"Attributes: {attr_str}")

        if node.disjoint_with:
            disjoint_names = [
                self.nodes[d].name if d in self.nodes else d
                for d in node.disjoint_with
            ]
            parts.append(f"Disjoint with: {', '.join(disjoint_names)}")

        if node.comment:
            parts.append(f"Comment: {node.comment}")

        return "\n".join(parts)

    def to_networkx(self):
        """Convert taxonomy to a NetworkX directed graph."""
        import networkx as nx
        g = nx.DiGraph()
        for node_id, node in self.nodes.items():
            g.add_node(node_id, name=node.name, depth=node.depth)
        for node_id, node in self.nodes.items():
            for parent_id in node.parents:
                if parent_id in self.nodes:
                    g.add_edge(parent_id, node_id)
        return g

    def to_adjacency(
        self,
    ) -> tuple[list[str], "torch.Tensor"]:  # noqa: F821
        """Build normalized adjacency matrix for GNN.

        Returns (node_ids in order, normalized adjacency as torch tensor).
        Uses symmetric normalization: D^{-1/2} A D^{-1/2}.
        Self-loops are added so each node attends to itself.
        Edges are directed parent→child, made undirected for message passing.
        """
        import numpy as np
        import torch

        node_ids = sorted(self.nodes.keys())
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)

        adj = np.zeros((n, n), dtype=np.float32)
        for node_id, node in self.nodes.items():
            i = id_to_idx[node_id]
            for parent_id in node.parents:
                if parent_id in id_to_idx:
                    j = id_to_idx[parent_id]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0  # Make undirected.

        # Self-loops.
        np.fill_diagonal(adj, 1.0)

        # Symmetric normalization: D^{-1/2} A D^{-1/2}.
        deg = adj.sum(axis=1)
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg))
        d_inv_sqrt = np.diag(deg_inv_sqrt)
        adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

        return node_ids, torch.from_numpy(adj_norm).float()

    @property
    def node_count(self) -> int:
        return len(self.nodes)


@dataclass
class TaxonMatch:
    """A single mapping between two taxonomy nodes."""

    source_id: str  # Node ID in source taxonomy.
    target_id: str  # Node ID in target taxonomy.
    confidence: float = 1.0  # Ground-truth or predicted confidence.


@dataclass
class Alignment:
    """A set of matches between two taxonomies."""

    source: str  # Source taxonomy name.
    target: str  # Target taxonomy name.
    matches: list[TaxonMatch] = field(default_factory=list)

    def as_pairs(self) -> set[tuple[str, str]]:
        """Return matches as a set of (source_id, target_id) pairs."""
        return {(m.source_id, m.target_id) for m in self.matches}

    @property
    def match_count(self) -> int:
        return len(self.matches)
