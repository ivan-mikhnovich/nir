"""Siamese GraphSAGE matcher for taxonomy alignment.

Trains a GNN encoder on synthetic taxonomy pairs with known ground-truth
mappings, then applies it to real OAEI Conference data for evaluation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..taxonomy import Alignment, TaxonMatch, Taxonomy


class GraphSAGELayer(nn.Module):
    """One GraphSAGE layer with mean aggregation.

    For each node: aggregate neighbour features via normalised adjacency,
    concatenate with self-features, apply linear + non-linearity.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neigh = nn.Linear(in_dim, out_dim)
        self.linear_out = nn.Linear(2 * out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [N, in_dim].
            adj: Normalised adjacency [N, N].

        Returns:
            Updated node features [N, out_dim].
        """
        neigh = adj @ x  # [N, in_dim] — mean aggregation.
        self_out = F.relu(self.linear_self(x))
        neigh_out = F.relu(self.linear_neigh(neigh))
        combined = torch.cat([self_out, neigh_out], dim=-1)
        out = F.relu(self.linear_out(combined))
        # Residual connection when input and output dimensions match.
        if x.shape[-1] == self.out_dim:
            out = out + x
        return self.dropout(out)


class LinearSiamese(nn.Module):
    """Siamese linear projection baseline — no graph conv, just a learned
    transformation of node features followed by cosine similarity."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(self, x: torch.Tensor, _adj: torch.Tensor) -> torch.Tensor:
        """Project node features (adj ignored)."""
        return self.proj(x)

    def forward(
        self,
        x_src: torch.Tensor,
        adj_src: torch.Tensor,
        x_tgt: torch.Tensor,
        adj_tgt: torch.Tensor,
    ) -> torch.Tensor:
        emb_src = self.encode(x_src, adj_src)
        emb_tgt = self.encode(x_tgt, adj_tgt)
        emb_src_norm = F.normalize(emb_src, p=2, dim=-1)
        emb_tgt_norm = F.normalize(emb_tgt, p=2, dim=-1)
        return emb_src_norm @ emb_tgt_norm.T


class SiameseGraphSAGE(nn.Module):
    """Two-branch GNN with shared weights for taxonomy matching."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            layers.append(GraphSAGELayer(d_in, hidden_dim, dropout))
        self.gnn_layers = nn.ModuleList(layers)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Encode one taxonomy into node embeddings.

        Args:
            x: Node features [N, in_dim].
            adj: Normalised adjacency [N, N].

        Returns:
            Node embeddings [N, out_dim].
        """
        h = x
        for layer in self.gnn_layers:
            h = layer(h, adj)
        return self.output_proj(h)

    def forward(
        self,
        x_src: torch.Tensor,
        adj_src: torch.Tensor,
        x_tgt: torch.Tensor,
        adj_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Encode both taxonomies and compute cosine similarity matrix.

        Args:
            x_src: Source node features [N_s, in_dim].
            adj_src: Source adjacency [N_s, N_s].
            x_tgt: Target node features [N_t, in_dim].
            adj_tgt: Target adjacency [N_t, N_t].

        Returns:
            Cosine similarity matrix [N_s, N_t].
        """
        emb_src = self.encode(x_src, adj_src)  # [N_s, out_dim]
        emb_tgt = self.encode(x_tgt, adj_tgt)  # [N_t, out_dim]
        emb_src_norm = F.normalize(emb_src, p=2, dim=-1)
        emb_tgt_norm = F.normalize(emb_tgt, p=2, dim=-1)
        return emb_src_norm @ emb_tgt_norm.T  # [N_s, N_t]


class GNNMatcher:
    """Matches two taxonomies using a Siamese GraphSAGE encoder.

    Node features are generated from a sentence-transformers model
    (default: MiniLM, 384-dim).  The GNN adds structural context via
    message passing over the parent-child graph.
    """

    def __init__(
        self,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 2,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.embedder_model_name = embedder_model

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # Initialise model (in_dim set after embedder is loaded).
        self.model: SiameseGraphSAGE | None = None
        self.in_dim: int | None = None

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    # ------------------------------------------------------------------
    # Embedding helpers.
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-load the sentence-transformers embedder."""
        import atexit

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.embedder_model_name)
        atexit.register(lambda: model.cpu() if hasattr(model, "cpu") else None)
        return model

    def _encode_taxonomy(self, tax: Taxonomy) -> tuple[list[str], torch.Tensor]:
        """Build node features for a taxonomy using BERT embeddings.

        Only the class name is used (name_only mode) — path information
        is provided structurally by the GNN via adjacency.

        Returns:
            (node_ids in order, feature tensor [N, in_dim]).
        """
        embedder = self._get_embedder()
        node_ids = sorted(tax.nodes.keys())
        texts = [tax.nodes[nid].name for nid in node_ids]
        embeddings = embedder.encode(texts, show_progress_bar=False)
        features = torch.from_numpy(embeddings).float()
        return node_ids, features

    # ------------------------------------------------------------------
    # Checkpoint I/O.
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a trained model from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.in_dim = ckpt["in_dim"]
        self.hidden_dim = ckpt.get("hidden_dim", 256)
        self.out_dim = ckpt.get("out_dim", 128)
        self.num_layers = ckpt.get("num_layers", 2)

        self.model = SiameseGraphSAGE(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights and metadata to a file."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        ckpt: dict = {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "num_layers": self.num_layers,
            "model_state": self.model.state_dict(),
        }
        torch.save(ckpt, path)

    # ------------------------------------------------------------------
    # Matching.
    # ------------------------------------------------------------------

    def match(
        self,
        source: Taxonomy,
        target: Taxonomy,
        threshold: float = 0.5,
    ) -> Alignment:
        """Match two taxonomies.

        Returns 1-to-1 greedy matching: for each source node, pick the
        best unmatched target node above the similarity threshold.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not initialised.  Call train() or load_checkpoint() first."
            )

        src_ids, src_feat = self._encode_taxonomy(source)
        tgt_ids, tgt_feat = self._encode_taxonomy(target)

        _, adj_src = source.to_adjacency()
        _, adj_tgt = target.to_adjacency()

        src_feat = src_feat.to(self.device)
        tgt_feat = tgt_feat.to(self.device)
        adj_src = adj_src.to(self.device)
        adj_tgt = adj_tgt.to(self.device)

        with torch.no_grad():
            sim = self.model(src_feat, adj_src, tgt_feat, adj_tgt)  # [N_s, N_t]

        sim = sim.cpu().numpy()

        # Greedy 1-to-1 matching.
        matches: list[TaxonMatch] = []
        used_tgt: set[int] = set()
        # Sort source nodes by confidence descending.
        candidates: list[tuple[float, int, int]] = []
        for i in range(len(src_ids)):
            for j in range(len(tgt_ids)):
                candidates.append((float(sim[i, j]), i, j))
        candidates.sort(key=lambda x: x[0], reverse=True)

        for score, i, j in candidates:
            if j in used_tgt or score < threshold:
                continue
            matches.append(TaxonMatch(
                source_id=src_ids[i],
                target_id=tgt_ids[j],
                confidence=score,
            ))
            used_tgt.add(j)

        return Alignment(
            source=source.name,
            target=target.name,
            matches=matches,
        )
