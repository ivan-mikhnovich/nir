"""Shared matching rules: turning raw candidate scores into alignments.

A matcher that can emit more than one match per source node must pass its raw
candidates through one of these rules before evaluation, so that every column
of the comparison table is produced by one class of postprocessing. Candidates
are ``(source_id, target_id, confidence)`` triples; every rule here is
deterministic and does not depend on the input order.
"""

from __future__ import annotations

from collections.abc import Iterable

Candidate = tuple[str, str, float]


def sort_candidates(pairs: Iterable[Candidate]) -> list[Candidate]:
    """Sort candidates by descending confidence, then by ids."""
    return sorted(pairs, key=lambda p: (-p[2], p[0], p[1]))


def per_source_best(pairs: Iterable[Candidate]) -> list[Candidate]:
    """Keep the best candidate of every source node.

    This is the nearest-target rule: each source gets exactly one match, so the
    mapping is injective on the source side and may collide on the target side.
    """
    best: dict[str, Candidate] = {}
    for src, tgt, conf in sort_candidates(pairs):
        if src not in best:
            best[src] = (src, tgt, conf)
    return sorted(best.values(), key=lambda p: (-p[2], p[0], p[1]))


def dedupe_by_target(
    pairs: Iterable[Candidate],
    threshold: float | None = None,
) -> list[Candidate]:
    """Reduce the nearest-target rule to an injective one.

    The per-source best candidates are kept first, then, for every target, only
    its best match survives. Every source appears at most once (it had one
    candidate) and every target appears at most once (by construction).
    """
    best: dict[str, Candidate] = {}
    for src, tgt, conf in per_source_best(pairs):
        if threshold is not None and conf < threshold:
            continue
        best.setdefault(tgt, (src, tgt, conf))
    return sorted(best.values(), key=lambda p: (-p[2], p[0], p[1]))


def greedy_injective(
    pairs: Iterable[Candidate],
    threshold: float | None = None,
) -> list[Candidate]:
    """Greedy one-to-one selection over all candidates.

    Candidates are taken in descending confidence order and are accepted only
    when neither their source nor their target has been used yet. Ties are
    broken by source and target id, so the result does not depend on the input
    order nor on the order in which parallel classifications finished. When
    ``threshold`` is given, lower-confidence candidates are dropped before the
    selection.
    """
    used_src: set[str] = set()
    used_tgt: set[str] = set()
    out: list[Candidate] = []
    for src, tgt, conf in sort_candidates(pairs):
        if threshold is not None and conf < threshold:
            continue
        if src in used_src or tgt in used_tgt:
            continue
        used_src.add(src)
        used_tgt.add(tgt)
        out.append((src, tgt, conf))
    return out


def to_alignment(
    pairs: Iterable[Candidate],
    source: str,
    target: str,
):
    """Build an ``Alignment`` from selected candidates."""
    from src.taxonomy import Alignment, TaxonMatch

    return Alignment(
        source=source,
        target=target,
        matches=[
            TaxonMatch(src, tgt, confidence=conf)
            for src, tgt, conf in pairs
        ],
    )
