"""BM25-only LLM matcher with lexical retrieval (no embeddings).

Extends LLMMatcher from llm.py.  Uses BM25 instead of BERT embeddings
for candidate retrieval — catches exact and partial lexical matches
like StringEquiv, but allows partial overlap ("Author" ↔ "AuthorNotReviewer").
LLM does final binary classification and cardinality filtering.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.matchers.llm import LLMMatcher
from src.taxonomy import Taxonomy


class BM25LLMMatcher(LLMMatcher):
    """LLMMatcher with BM25-only candidate retrieval (no BERT).

    Replaces embedding-based top-k with BM25 lexical search.
    """

    def __init__(self, bm25_k: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.bm25_k = bm25_k or self.top_k

    # ── BM25-only retrieval ────────────────────────────────────────────

    def _get_top_k_targets(
        self, source: Taxonomy, target: Taxonomy,
    ) -> dict[str, list[tuple[str, float]]]:
        """Pre-compute top-k candidates via BM25 lexical search.

        Each target node is a BM25 document (tokenised class name).
        Each source node queries the index; top-bm25_k results returned.
        Scores are raw BM25 scores (not normalised), since they are
        only used for ordering, not for the LLM prompt.
        """
        import numpy as np

        cache_key = (source.name, target.name)
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        tgt_ids = sorted(target.nodes.keys())
        tgt_texts = [
            self._tokenize(target.nodes[nid].name)
            for nid in tgt_ids
        ]
        bm25 = BM25Okapi(tgt_texts)

        result: dict[str, list[tuple[str, float]]] = {}

        for src_id in sorted(source.nodes.keys()):
            src_text = self._tokenize(source.nodes[src_id].name)
            bm25_raw = bm25.get_scores(src_text)
            top_idx = np.argsort(bm25_raw)[::-1][: self.bm25_k]
            candidates: list[tuple[str, float]] = [
                (tgt_ids[j], float(bm25_raw[j]))
                for j in top_idx if float(bm25_raw[j]) > 0
            ]
            result[src_id] = candidates

        self._embed_cache[cache_key] = result
        return result

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 (lowercased word split)."""
        return text.lower().split()
