"""Hybrid LLM matcher with BM25 + embedding candidate retrieval.

Extends LLMMatcher from llm.py.  Only the candidate retrieval step
(`_get_top_k_targets`) is overridden — the LLM classification and
post-processing stages are inherited verbatim.

BM25 catches exact and partial lexical matches (where StringEquiv excels),
embeddings catch semantic correspondences (where BERT excels).  The union
of both candidate sets is sent to the LLM for final binary classification.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.matchers.llm import LLMMatcher
from src.taxonomy import Taxonomy


class HybridLLMMatcher(LLMMatcher):
    """LLMMatcher with BM25 + embedding hybrid candidate retrieval.

    Constructor identical to LLMMatcher plus one optional parameter:
        - bm25_k: Number of BM25 candidates per source node (default = top_k).
    """

    def __init__(self, bm25_k: int | None = None, total_k: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.bm25_k = bm25_k or self.top_k
        self.total_k = total_k or min(self.top_k + self.bm25_k, 10)

    # ── Hybrid retrieval ────────────────────────────────────────────────

    def _get_top_k_targets(
        self, source: Taxonomy, target: Taxonomy,
    ) -> dict[str, list[tuple[str, float]]]:
        """Pre-compute top-k candidates via BM25 + embedding union.

        For each source node, both retrievers produce a ranked list.
        The two lists are merged: each target node gets the maximum
        confidence from whichever retriever ranked it highest (BM25
        scores are min-max-normalised per query to [0, 1]).
        The merged list is truncated to self.top_k.
        """
        import numpy as np

        cache_key = (source.name, target.name)
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        # ── BM25 index on target taxonomy ──
        tgt_ids = sorted(target.nodes.keys())
        tgt_texts = [
            self._tokenize(target.nodes[nid].name)
            for nid in tgt_ids
        ]
        bm25 = BM25Okapi(tgt_texts)

        # ── Embedding retrieval (reuse parent's MiniLM) ──
        src_ids, src_embs = self._embed_matcher.encode_taxonomy(
            source, description_mode="name_only",
        )
        tgt_ids_emb, tgt_embs = self._embed_matcher.encode_taxonomy(
            target, description_mode="name_only",
        )
        sim = np.dot(src_embs, tgt_embs.T)

        result: dict[str, list[tuple[str, float]]] = {}

        for i, src_id in enumerate(src_ids):
            # ── Embedding candidates (always included first) ──
            top_emb_idx = np.argsort(sim[i])[::-1][: self.top_k]
            emb_candidates: list[tuple[str, float]] = [
                (tgt_ids_emb[j], float(sim[i, j])) for j in top_emb_idx
            ]
            seen: set[str] = {tid for tid, _ in emb_candidates}

            # ── BM25 candidates (added on top, no replacement) ──
            src_text = self._tokenize(source.nodes[src_id].name)
            bm25_raw = bm25.get_scores(src_text)
            top_bm25_idx = np.argsort(bm25_raw)[::-1][: self.bm25_k]
            bm25_candidates: list[tuple[str, float]] = []
            for j in top_bm25_idx:
                raw_score = float(bm25_raw[j])
                if raw_score <= 0:
                    continue
                tid = tgt_ids[j]
                if tid in seen:
                    continue
                seen.add(tid)
                bm25_candidates.append((tid, raw_score))

            # ── Merge: emb first, then new BM25, truncate to total_k ──
            merged = emb_candidates + bm25_candidates
            result[src_id] = merged[: self.total_k]

        self._embed_cache[cache_key] = result
        return result

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 (lowercased word split)."""
        return text.lower().split()
