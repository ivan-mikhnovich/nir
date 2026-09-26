"""Data structures for taxonomy representation."""

from __future__ import annotations

from collections import deque
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

    def get_representative_chain(self, node_id: str) -> list[str]:
        """Return one shortest chain of names from a top-level class to the node.

        There is no single root to walk from: the parser leaves top-level
        classes unlinked from `owl:Thing`, so these taxonomies have many of
        them, and a class may have several parents.  This method therefore
        returns the *shortest* chain, and resolves ties between equally short
        chains by node id, so the result depends on the graph alone and not on
        the order of the parent lists.  A node that has no chain to a top-level
        class (a cycle) yields just its own name.
        """
        node = self.nodes.get(node_id)
        if node is None:
            return []

        queue: deque[list[str]] = deque([[node_id]])
        seen = {node_id}
        while queue:
            chain = queue.popleft()
            parents = sorted(
                parent for parent in self.nodes[chain[-1]].parents
                if parent in self.nodes
            )
            if not parents:
                return [self.nodes[nid].name for nid in reversed(chain)]
            for parent in parents:
                if parent not in seen:
                    seen.add(parent)
                    queue.append(chain + [parent])

        return [node.name]

    def get_ancestors(self, node_id: str) -> set[str]:
        """Return every strict ancestor of a node, tolerating cycles.

        A cycle must not make a node its own ancestor, so the starting node is
        removed from the result even when the traversal reaches it again.
        """
        node = self.nodes.get(node_id)
        seen: set[str] = set()
        stack = list(node.parents) if node is not None else []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            parent = self.nodes.get(current)
            if parent is not None:
                stack.extend(parent.parents)
        seen.discard(node_id)
        return seen

    def get_textual_description(self, node_id: str) -> str:
        """Build a textual representation of a node for embedding/LLM input."""
        node = self.nodes.get(node_id)
        if node is None:
            return ""

        parts: list[str] = []
        parts.append(f"Name: {node.name}")

        path = self.get_representative_chain(node_id)
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
