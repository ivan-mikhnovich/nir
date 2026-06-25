"""Embedding-based taxonomy matcher using sentence-transformers."""

from __future__ import annotations

import numpy as np

from src.taxonomy import Alignment, TaxonMatch, Taxonomy


class EmbeddingMatcher:
    """Matches taxonomy nodes using cosine similarity of text embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/LaBSE",
                 description_mode: str = "name_only"):
        """Initialize the matcher with a sentence-transformers model.

        Args:
            model_name: HuggingFace model identifier.
            description_mode: How to describe nodes for embedding.
                'name_only': just the class name (best for OAEI).
                'name_path': name + path from root.
                'full': name + path + attributes + disjoint + comment.
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
            description_mode:
                - 'full': name + path + properties + disjoint + comment.
                - 'name_only': just the class name.
                - 'name_path': name + path from root.

        Returns:
            (node_ids, embeddings) where embeddings shape is (N, dim).
        """
        node_ids = sorted(tax.nodes.keys())
        mode = description_mode or self.description_mode
        texts: list[str] = []
        for nid in node_ids:
            if mode == "full":
                text = tax.get_textual_description(nid)
            elif mode == "name_only":
                text = tax.nodes[nid].name
            elif mode == "name_path":
                path = tax.get_path(nid)
                text = f"{' -> '.join(path)}"
            else:
                text = tax.get_textual_description(nid)
            texts.append(text)

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return node_ids, embeddings

    def match(
        self,
        source: Taxonomy,
        target: Taxonomy,
        description_mode: str | None = None,
        top_k: int = 3,
    ) -> tuple[Alignment, list[float]]:
        """Match nodes between two taxonomies.

        For each source node, finds the best matching target node(s)
        by cosine similarity.

        Args:
            source: Source taxonomy.
            target: Target taxonomy.
            description_mode: How to describe nodes for embedding.
            top_k: Number of top candidates to record.

        Returns:
            (alignment, similarity_scores) where alignment contains the
            best match for each source node, and similarity_scores is
            a list of the cosine similarities for those matches.
        """
        src_ids, src_embs = self.encode_taxonomy(source, description_mode)
        tgt_ids, tgt_embs = self.encode_taxonomy(target, description_mode)

        # Cosine similarity matrix (already normalized, so dot product = cosine).
        # Shape: (N_src, N_tgt).
        sim_matrix = np.dot(src_embs, tgt_embs.T)

        matches: list[TaxonMatch] = []
        all_scores: list[float] = []

        for i, src_id in enumerate(src_ids):
            # Find best match.
            best_j = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i, best_j])
            best_tgt_id = tgt_ids[best_j]

            matches.append(TaxonMatch(
                source_id=src_id,
                target_id=best_tgt_id,
                confidence=best_score,
            ))
            all_scores.append(best_score)

        alignment = Alignment(
            source=source.name,
            target=target.name,
            matches=matches,
        )
        return alignment, all_scores

    def match_with_threshold(
        self,
        source: Taxonomy,
        target: Taxonomy,
        threshold: float = 0.7,
        description_mode: str = "full",
    ) -> tuple[Alignment, list[dict]]:
        """Match with confidence threshold, flagging low-confidence cases.

        Returns:
            (alignment, uncertain_cases) where uncertain_cases is a list
            of dicts with source_id, best_match, score, and alternative
            candidates for human review.
        """
        src_ids, src_embs = self.encode_taxonomy(source, description_mode)
        tgt_ids, tgt_embs = self.encode_taxonomy(target, description_mode)

        sim_matrix = np.dot(src_embs, tgt_embs.T)

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
