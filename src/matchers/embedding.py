"""Embedding-based taxonomy matcher using sentence-transformers."""

from __future__ import annotations

import numpy as np

from src.matching_rules import (
    Candidate,
    dedupe_by_target,
    greedy_injective,
    to_alignment,
)
from src.taxonomy import Alignment, TaxonMatch, Taxonomy

# Text built for a node, as (label → components). `name_path` is the one legacy
# label: it joins the chain without the `Path:` block, exactly as the cached
# ablation was produced. Every other label is the corresponding subset of
# `Taxonomy.get_textual_description`, in the same order and with the same block
# headers, so `full` is bit-identical to it.
DESCRIPTION_MODES: dict[str, tuple[str, ...]] = {
    "name_only": ("name",),
    "name_path": ("name", "path"),
    "name+attributes": ("name", "attributes"),
    "name+disjoint+comment": ("name", "attributes", "disjoint", "comment"),
    "full": ("name", "path", "attributes", "disjoint", "comment"),
}

# Postprocessing modes, applied to the cosine matrix (contract C1). `argmax` is
# the nearest-target rule behind the published baseline; the other two are
# injective variants from `src.matching_rules`.
POSTPROCESS_MODES: tuple[str, ...] = ("argmax", "dedupe_by_target", "greedy_injective")


def node_text(tax: Taxonomy, node_id: str, mode: str) -> str:
    """Build the text of one node for the given description mode.

    Args:
        tax: The taxonomy holding the node.
        node_id: Node id.
        mode: One of `DESCRIPTION_MODES`.

    Returns:
        The text encoded for that node.

    Raises:
        ValueError: When `mode` is unknown.
    """
    if mode not in DESCRIPTION_MODES:
        raise ValueError(
            f"unknown description mode {mode!r}; known: {sorted(DESCRIPTION_MODES)}"
        )
    if mode == "name_only":
        # The published baseline: the bare class name, nothing else.
        return tax.nodes[node_id].name
    if mode == "name_path":
        # Legacy: the chain includes the node's own name, no `Path:` header.
        return " -> ".join(tax.get_representative_chain(node_id))

    node = tax.nodes[node_id]
    blocks: dict[str, str] = {"name": f"Name: {node.name}"}
    path = tax.get_representative_chain(node_id)
    if len(path) > 1:
        blocks["path"] = f"Path: {' -> '.join(path)}"
    if node.attributes:
        attributes = ", ".join(f"{k}: {v}" for k, v in sorted(node.attributes.items()))
        blocks["attributes"] = f"Attributes: {attributes}"
    if node.disjoint_with:
        disjoint = ", ".join(
            tax.nodes[d].name if d in tax.nodes else d for d in node.disjoint_with
        )
        blocks["disjoint"] = f"Disjoint with: {disjoint}"
    if node.comment:
        blocks["comment"] = f"Comment: {node.comment}"
    return "\n".join(blocks[c] for c in DESCRIPTION_MODES[mode] if c in blocks)


class EmbeddingMatcher:
    """Matches taxonomy nodes using cosine similarity of text embeddings.

    Pipeline: encode every node text with the sentence-transformers model
    (`normalize_embeddings=True`, so the dot product is the cosine), build the
    full source×target similarity matrix, then reduce it to an alignment with
    one postprocessing rule.

    Description modes (`node_text`):
        `name_only` — the class name.
        `name_path` — the name plus one shortest ancestor chain, joined by ` -> `.
        `name+attributes` — `Name:` + `Attributes:` blocks.
        `name+disjoint+comment` — `Name:` + `Attributes:` + `Disjoint with:` +
            `Comment:` (the reviewer's shorthand for the component ablation).
        `full` — `Name:` + `Path:` + `Attributes:` + `Disjoint with:` + `Comment:`,
            i.e. `Taxonomy.get_textual_description`.

    Postprocessing modes (contract C1):
        `argmax` — the nearest target of every source; one pair per source and
            therefore *not* injective on the target side (the published
            baseline).
        `dedupe_by_target` — nearest target per source, then the best source per
            target (`src.matching_rules.dedupe_by_target`).
        `greedy_injective` — greedy 1:1 over the whole matrix
            (`src.matching_rules.greedy_injective`).

    All three are deterministic; `argmax` keeps the source order of the output,
    the other two order pairs by descending confidence.
    """

    def __init__(self, model_name: str = "sentence-transformers/LaBSE",
                 description_mode: str = "name_only"):
        """Initialize the matcher with a sentence-transformers model.

        Args:
            model_name: HuggingFace model identifier.
            description_mode: How to describe nodes for embedding; one of
                `DESCRIPTION_MODES`, default `name_only`.
        """
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.description_mode = description_mode
        self.model = SentenceTransformer(model_name)
        try:
            self._dim = self.model.get_embedding_dimension()
        except AttributeError:
            self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode_taxonomy(
        self,
        tax: Taxonomy,
        description_mode: str | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Encode all nodes of a taxonomy as text embeddings.

        Args:
            tax: The taxonomy to encode.
            description_mode: One of `DESCRIPTION_MODES`, default the matcher's.

        Returns:
            (node_ids, embeddings) where embeddings shape is (N, dim).
        """
        node_ids = sorted(tax.nodes.keys())
        mode = description_mode or self.description_mode
        texts = [node_text(tax, nid, mode) for nid in node_ids]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return node_ids, embeddings

    def similarity(
        self,
        source: Taxonomy,
        target: Taxonomy,
        description_mode: str | None = None,
    ) -> tuple[list[str], list[str], np.ndarray]:
        """Encode both sides and return the cosine similarity matrix."""
        src_ids, src_embs = self.encode_taxonomy(source, description_mode)
        tgt_ids, tgt_embs = self.encode_taxonomy(target, description_mode)
        # Cosines: both sides are L2-normalised, so the dot product is the cosine.
        return src_ids, tgt_ids, np.dot(src_embs, tgt_embs.T)

    def candidates(
        self,
        source: Taxonomy,
        target: Taxonomy,
        description_mode: str | None = None,
    ) -> list[Candidate]:
        """Return every (source_id, target_id, cosine) triple of the pair."""
        src_ids, tgt_ids, sim = self.similarity(source, target, description_mode)
        return [
            (src_id, tgt_ids[j], float(sim[i, j]))
            for i, src_id in enumerate(src_ids)
            for j in range(len(tgt_ids))
        ]

    def match(
        self,
        source: Taxonomy,
        target: Taxonomy,
        description_mode: str | None = None,
        postprocess: str = "argmax",
        threshold: float | None = None,
    ) -> tuple[Alignment, list[float]]:
        """Match nodes between two taxonomies.

        Args:
            source: Source taxonomy.
            target: Target taxonomy.
            description_mode: One of `DESCRIPTION_MODES`, default the matcher's.
            postprocess: One of `POSTPROCESS_MODES`; which rule turns the cosine
                matrix into the alignment.
            threshold: Optional confidence floor; lower-scoring pairs are dropped
                after the rule (so `argmax` keeps today's behaviour when omitted).

        Returns:
            (alignment, similarity_scores) with the cosine similarity of every
            accepted match.

        Raises:
            ValueError: On an unknown postprocessing mode.
        """
        if postprocess not in POSTPROCESS_MODES:
            raise ValueError(
                f"unknown postprocess mode {postprocess!r}; known: {list(POSTPROCESS_MODES)}"
            )
        src_ids, tgt_ids, sim = self.similarity(source, target, description_mode)

        if postprocess == "argmax":
            # Nearest target of every source, in source order (equivalent to
            # `per_source_best` on the full matrix, but keeps the cached order).
            pairs: list[Candidate] = []
            for i, src_id in enumerate(src_ids):
                best_j = int(np.argmax(sim[i]))
                best_score = float(sim[i, best_j])
                if threshold is not None and best_score < threshold:
                    continue
                pairs.append((src_id, tgt_ids[best_j], best_score))
        else:
            candidates = [
                (src_id, tgt_ids[j], float(sim[i, j]))
                for i, src_id in enumerate(src_ids)
                for j in range(len(tgt_ids))
            ]
            rule = dedupe_by_target if postprocess == "dedupe_by_target" else greedy_injective
            pairs = rule(candidates, threshold)

        alignment = to_alignment(pairs, source.name, target.name)
        return alignment, [conf for _src, _tgt, conf in pairs]

    def match_with_threshold(
        self,
        source: Taxonomy,
        target: Taxonomy,
        threshold: float = 0.7,
        description_mode: str = "full",
    ) -> tuple[Alignment, list[dict]]:
        """Match with confidence threshold, flagging low-confidence cases.

        Demo/calibration path only (this is why its default description mode
        differs from the class default: it is the human-in-the-loop showcase).

        Returns:
            (alignment, uncertain_cases) where uncertain_cases is a list
            of dicts with source_id, best_match, score, and alternative
            candidates for human review.
        """
        src_ids, tgt_ids, sim_matrix = self.similarity(source, target, description_mode)

        matches: list[TaxonMatch] = []
        uncertain: list[dict] = []

        for i, src_id in enumerate(src_ids):
            # Get top-2 indices and scores.
            sorted_indices = np.argsort(sim_matrix[i])[::-1]
            best_score = float(sim_matrix[i, sorted_indices[0]])
            second_score = float(sim_matrix[i, sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
            best_tgt = tgt_ids[sorted_indices[0]]

            matches.append(TaxonMatch(
                source_id=src_id,
                target_id=best_tgt,
                confidence=best_score,
            ))

            # Flag uncertain: low confidence or close competitors.
            if best_score < threshold or (best_score - second_score < 0.1 and second_score > 0.3):
                alternatives = []
                for j in sorted_indices[1:4]:  # Top 2-4 alternatives.
                    alt_score = float(sim_matrix[i, j])
                    if alt_score > 0.3:
                        alternatives.append({
                            "target_id": tgt_ids[j],
                            "target_name": target.nodes[tgt_ids[j]].name,
                            "score": round(alt_score, 4),
                        })
                uncertain.append({
                    "source_id": src_id,
                    "source_name": source.nodes[src_id].name,
                    "best_match_id": best_tgt,
                    "best_match_name": target.nodes[best_tgt].name,
                    "best_score": round(best_score, 4),
                    "second_score": round(second_score, 4),
                    "alternatives": alternatives,
                })

        alignment = Alignment(
            source=source.name,
            target=target.name,
            matches=matches,
        )
        return alignment, uncertain
