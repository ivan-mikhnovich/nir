"""String-equivalence baseline for taxonomy matching.

Simple lowercase label matching — OAEI official lower bound.
For each source node, matches to the first target node with
identical (lowercased) label, if any.
"""

import time

from src.taxonomy import Taxonomy, Alignment, TaxonMatch


class StringEquivMatcher:
    """Match nodes by exact (case-insensitive) label equality."""

    def __init__(self) -> None:
        self.last_timing: dict[str, float] = {}

    def match(self, source: Taxonomy, target: Taxonomy) -> tuple[Alignment, list[TaxonMatch]]:
        t0 = time.perf_counter()

        # Build lookup: lowercase label → list of target ids.
        tgt_index: dict[str, list[str]] = {}
        for tid, node in target.nodes.items():
            key = node.name.strip().lower()
            tgt_index.setdefault(key, []).append(tid)

        matches: list[TaxonMatch] = []
        for sid, src_node in source.nodes.items():
            key = src_node.name.strip().lower()
            candidates = tgt_index.get(key, [])
            if candidates:
                # Take first match (deterministic).
                matches.append(TaxonMatch(
                    source_id=sid,
                    target_id=candidates[0],
                    confidence=1.0,
                ))

        alignment = Alignment(
            source=source.name,
            target=target.name,
            matches=matches,
        )

        dt = time.perf_counter() - t0
        self.last_timing["time"] = dt
        self.last_timing["n_source"] = source.node_count
        self.last_timing["n_target"] = target.node_count

        return alignment, matches
