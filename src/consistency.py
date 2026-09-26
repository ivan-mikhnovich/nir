"""Semantic consistency checking for taxonomies and their alignments.

Two families of checks make up the second half of the topic:

* Taxonomy-internal integrity — cycles in the `is-a` graph, duplicate labels,
  declared disjointness between a class and its own ancestor, classes whose
  declared parent is absent from the taxonomy, and classes with no chain of
  parents ending at a root.
* Alignment consistency — violations of the 1:1 cardinality, matches that
  contradict declared disjointness or the subsumption hierarchy of the source.

The alignment checks double as a post-filter: `check_alignment` records, for
every match, the kinds it violates, so a caller can drop the flagged matches
and measure whether the filter improves precision.

Confidence is deliberately NOT one of the checks. A confidence cut-off is the
post-processing the matchers leave to the caller, and it selects on the match's
own confidence alone, so it can say nothing about consistency that the
threshold sweep of `src.runners.consistency` does not already say: the two are
numerically identical, and the sweep is where that number belongs (F2,
review/review-consistency.md).

Usage:
    from src.consistency import check_alignment, check_taxonomy
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.taxonomy import Alignment, Taxonomy

# Kinds are plain strings (not an enum), so they survive a JSON round trip.
CYCLE = "is-a-cycle"
DUPLICATE_LABEL = "duplicate-label"
DISJOINT_ANCESTOR = "disjoint-ancestor"
DANGLING_PARENT = "dangling-parent"
UNREACHABLE = "unreachable-class"
CARDINALITY = "cardinality-violation"
DISJOINTNESS = "disjointness-violation"
SUBSUMPTION = "subsumption-violation"

# Severity per kind: an error contradicts the ontology or the 1:1 setting,
# a warning is a structural anomaly.
SEVERITY: dict[str, str] = {
    CYCLE: "error",
    DUPLICATE_LABEL: "warning",
    DISJOINT_ANCESTOR: "error",
    DANGLING_PARENT: "error",
    UNREACHABLE: "info",
    CARDINALITY: "error",
    DISJOINTNESS: "error",
    SUBSUMPTION: "warning",
}

# Kinds of the alignment checks proper, and the flags a filter may drop matches
# for.  Low confidence is absent on purpose: it is the threshold sweep, not a
# consistency check (see the module docstring).
ALIGNMENT_KINDS = (CARDINALITY, DISJOINTNESS, SUBSUMPTION)

# Kinds worth treating as "the alignment claims something the ontology denies".
ERROR_KINDS = (CARDINALITY, DISJOINTNESS)
WARNING_KINDS = (SUBSUMPTION,)


@dataclass(frozen=True)
class Violation:
    """A single detected inconsistency."""

    kind: str
    detail: str
    source_id: str | None = None
    target_id: str | None = None

    @property
    def severity(self) -> str:
        """Severity of this violation, derived from its kind."""
        return SEVERITY[self.kind]


@dataclass
class TaxonomyReport:
    """Consistency report of a single taxonomy."""

    name: str
    node_count: int
    root_count: int
    max_depth: int
    violations: list[Violation] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Number of violations per kind."""
        return _count_kinds(self.violations)


@dataclass
class AlignmentReport:
    """Consistency report of a single alignment."""

    approach: str
    pair: str
    mode: str | None
    match_count: int
    violations: list[Violation] = field(default_factory=list)
    flagged: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # Overloaded groups per side — the unit a cardinality violation is counted
    # in, as opposed to `flagged`, which counts the matches such a group
    # invalidates (F5, review/review-consistency.md).
    cardinality_groups: dict[str, int] = field(
        default_factory=lambda: {"source": 0, "target": 0}
    )

    def counts(self) -> dict[str, int]:
        """Number of violations per kind."""
        return _count_kinds(self.violations)

    def flagged_kinds(self) -> dict[str, int]:
        """Number of matches carrying each flag kind."""
        return _count_flags(self.flagged)


# ── Shared helpers ───────────────────────────────────────────────────


def _count_kinds(violations: list[Violation]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in violations:
        out[v.kind] = out.get(v.kind, 0) + 1
    return out


def _count_flags(flagged: dict[tuple[str, str], list[str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for kinds in flagged.values():
        for k in kinds:
            out[k] = out.get(k, 0) + 1
    return out


def _sample(items: list[str], limit: int = 5) -> str:
    """Format up to `limit` items, appending an ellipsis when truncated."""
    head = ", ".join(items[:limit])
    return head if len(items) <= limit else f"{head}, …"


def _normalized(name: str) -> str:
    """Normalize a label for duplicate detection (case and whitespace only)."""
    return " ".join(name.casefold().split())


def _is_descendant(tax: Taxonomy, node_id: str, ancestor_id: str) -> bool:
    """True when `ancestor_id` is a strict ancestor of `node_id`."""
    if ancestor_id == node_id:
        return False
    return ancestor_id in tax.get_ancestors(node_id)


def _roots(tax: Taxonomy) -> list[str]:
    """Return the top-level classes, which declare no parent at all.

    A class whose parents are all absent from the taxonomy is deliberately not
    a root: that would silently turn a broken `is-a` link into a top-level
    class (F1, review/review-consistency.md).  Such a class is reported as
    `dangling-parent` and stays unreachable unless another declared parent
    reaches a root.
    """
    return sorted(nid for nid, node in tax.nodes.items() if not node.parents)


def _children_map(tax: Taxonomy) -> dict[str, list[str]]:
    """Invert the parent relation, ignoring links to unknown nodes."""
    children: dict[str, list[str]] = {}
    for nid, node in tax.nodes.items():
        for parent in node.parents:
            if parent in tax.nodes:
                children.setdefault(parent, []).append(nid)
    return children


def _disjoint_pairs(tax: Taxonomy) -> list[tuple[str, str]]:
    """Return declared disjointness pairs in canonical order, without duplicates."""
    pairs: set[tuple[str, str]] = set()
    for nid, node in tax.nodes.items():
        for d in node.disjoint_with:
            if d in tax.nodes and d != nid:
                pairs.add(tuple(sorted((nid, d))))
    return sorted(pairs)


def _find_cycles(tax: Taxonomy) -> list[list[str]]:
    """Return one node list per cycle in the `is-a` graph, closed on itself."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(tax.nodes, WHITE)
    cycles: list[list[str]] = []
    for start in sorted(tax.nodes):
        if color[start] != WHITE:
            continue
        color[start] = GRAY
        path = [start]
        stack: list[list] = [[start, 0]]  # Node and next parent index.
        while stack:
            nid, idx = stack[-1]
            parents = [p for p in tax.nodes[nid].parents if p in tax.nodes]
            if idx < len(parents):
                stack[-1][1] = idx + 1
                parent = parents[idx]
                if color[parent] == GRAY:
                    cycles.append(path[path.index(parent):] + [parent])
                elif color[parent] == WHITE:
                    color[parent] = GRAY
                    path.append(parent)
                    stack.append([parent, 0])
            else:
                color[nid] = BLACK
                path.pop()
                stack.pop()
    return cycles


# ── Taxonomy-internal checks ─────────────────────────────────────────


def check_taxonomy(tax: Taxonomy) -> TaxonomyReport:
    """Run every taxonomy-internal integrity check."""
    violations: list[Violation] = []

    for cycle in _find_cycles(tax):
        names = " → ".join(tax.nodes[n].name if n in tax.nodes else n for n in cycle)
        violations.append(Violation(
            kind=CYCLE,
            detail=f"inheritance cycle: {names}",
            source_id=cycle[0],
        ))

    for nid, node in sorted(tax.nodes.items()):
        missing = sorted({p for p in node.parents if p not in tax.nodes})
        if missing:
            violations.append(Violation(
                kind=DANGLING_PARENT,
                detail=(f"{nid} declares parent(s) the taxonomy does not contain: "
                        f"{_sample(missing)}"),
                source_id=nid,
            ))

    by_label: dict[str, list[str]] = {}
    for nid, node in tax.nodes.items():
        by_label.setdefault(_normalized(node.name), []).append(nid)
    for label, ids in sorted(by_label.items()):
        if len(ids) > 1:
            violations.append(Violation(
                kind=DUPLICATE_LABEL,
                detail=(f"label '{label}' is shared by {len(ids)} classes: "
                        f"{_sample(sorted(ids))}"),
                source_id=ids[0],
            ))

    scopes = {nid: tax.get_ancestors(nid) | {nid} for nid in tax.nodes}
    for x, y in _disjoint_pairs(tax):
        clash = sorted(nid for nid, scope in scopes.items() if x in scope and y in scope)
        if clash:
            violations.append(Violation(
                kind=DISJOINT_ANCESTOR,
                detail=(f"{x} and {y} are declared disjoint, yet both are "
                        f"ancestor-or-self of {_sample(clash)}"),
                source_id=clash[0],
            ))

    # The parser leaves top-level classes unlinked from owl:Thing, so a taxonomy
    # has many roots by construction.  A class counts as unreachable only when no
    # chain of declared parents ends at one of those roots: a cycle, or a
    # declared parent that is absent from the taxonomy.
    roots = _roots(tax)
    children = _children_map(tax)
    reachable = set(roots)
    stack = list(roots)
    while stack:
        cur = stack.pop()
        for child in children.get(cur, ()):
            if child not in reachable:
                reachable.add(child)
                stack.append(child)
    orphans = sorted(set(tax.nodes) - reachable)
    if orphans:
        violations.append(Violation(
            kind=UNREACHABLE,
            detail=(f"{len(orphans)} of {tax.node_count} classes have no chain of "
                    f"parents ending at a root class: {_sample(orphans)}"),
            source_id=orphans[0],
        ))

    return TaxonomyReport(
        name=tax.name,
        node_count=tax.node_count,
        root_count=len(roots),
        max_depth=max((n.depth for n in tax.nodes.values()), default=0),
        violations=violations,
    )


# ── Alignment checks ─────────────────────────────────────────────────


def check_alignment(
    src: Taxonomy,
    tgt: Taxonomy,
    alignment: Alignment,
    *,
    approach: str = "",
    pair: str = "",
    mode: str | None = None,
) -> AlignmentReport:
    """Run every alignment check and record which kinds each match violates.

    The cardinality check needs only the alignment; the disjointness and
    subsumption checks compare it against both taxonomies.  All of them are
    order-independent: an overloaded source or target is kept as a *set* of
    nodes and every combination of those sets is tested, so shuffling
    `alignment.matches` cannot change a count (F4,
    review/review-consistency.md).

    A cardinality violation is counted per overloaded GROUP — an LLM matcher
    is source-injective by construction, so a single overloaded target group
    invalidates several matches while counting once (F5).
    """
    violations: list[Violation] = []
    flagged: dict[tuple[str, str], list[str]] = {}

    def flag(key: tuple[str, str], kind: str) -> None:
        kinds = flagged.setdefault(key, [])
        if kind not in kinds:
            kinds.append(kind)

    by_source: dict[str, set[str]] = {}
    by_target: dict[str, set[str]] = {}
    for m in alignment.matches:
        by_source.setdefault(m.source_id, set()).add(m.target_id)
        by_target.setdefault(m.target_id, set()).add(m.source_id)

    overloaded_sources = sorted(sid for sid, tids in by_source.items() if len(tids) > 1)
    overloaded_targets = sorted(tid for tid, sids in by_target.items() if len(sids) > 1)
    for sid in overloaded_sources:
        tids = sorted(by_source[sid])
        for tid in tids:
            flag((sid, tid), CARDINALITY)
        violations.append(Violation(
            kind=CARDINALITY,
            detail=f"source {sid} is matched to {len(tids)} targets: {_sample(tids)}",
            source_id=sid,
        ))
    for tid in overloaded_targets:
        sids = sorted(by_target[tid])
        for sid in sids:
            flag((sid, tid), CARDINALITY)
        violations.append(Violation(
            kind=CARDINALITY,
            detail=f"target {tid} is matched to {len(sids)} sources: {_sample(sids)}",
            target_id=tid,
        ))

    # Declared disjointness must survive the alignment: two classes that one
    # ontology declares disjoint cannot be matched by classes the other
    # ontology relates by subsumption (this would make them overlap).  One
    # violation is counted per declared pair and side once any combination of
    # its matches breaks the constraint, and every match of a breaking
    # combination is flagged.
    for side, tax, other, mapping in (
        ("source", src, tgt, by_source),
        ("target", tgt, src, by_target),
    ):
        for x, y in _disjoint_pairs(tax):
            if x not in mapping or y not in mapping:
                continue
            clash: tuple[str, str] | None = None
            for mx in sorted(mapping[x]):
                for my in sorted(mapping[y]):
                    if mx != my and not _is_descendant(other, mx, my) \
                            and not _is_descendant(other, my, mx):
                        continue
                    keys = [(x, mx), (y, my)] if side == "source" else [(mx, x), (my, y)]
                    for key in keys:
                        flag(key, DISJOINTNESS)
                    if clash is None:
                        clash = (mx, my)
            if clash is None:
                continue
            mx, my = clash
            keys = [(x, mx), (y, my)] if side == "source" else [(mx, x), (my, y)]
            relation = ("both land on the same class" if mx == my
                        else f"{other.name} relates them by subsumption")
            violations.append(Violation(
                kind=DISJOINTNESS,
                detail=(f"{x} and {y} are declared disjoint in {tax.name}, yet the "
                        f"alignment maps them to {mx} and {my}: {relation}"),
                source_id=keys[0][0],
                target_id=keys[0][1],
            ))

    # A child may not be matched below where its own parent was matched: the
    # alignment has to preserve the subsumption direction of the source.  A
    # parent matched to several targets gives one candidate per target, and the
    # match is reported once.
    for m in alignment.matches:
        node = src.nodes.get(m.source_id)
        if node is None:
            continue
        for parent_id in node.parents:
            for parent_target in sorted(by_source.get(parent_id, ())):
                if m.target_id == parent_target:
                    continue
                if _is_descendant(tgt, m.target_id, parent_target):
                    continue
                flag((m.source_id, m.target_id), SUBSUMPTION)
                violations.append(Violation(
                    kind=SUBSUMPTION,
                    detail=(f"{m.source_id} ⊑ {parent_id} and {parent_id} ↔ "
                            f"{parent_target}, but {m.target_id} is not a "
                            f"descendant of {parent_target}"),
                    source_id=m.source_id,
                    target_id=m.target_id,
                ))
                break
            else:
                continue
            break

    return AlignmentReport(
        approach=approach,
        pair=pair,
        mode=mode,
        match_count=alignment.match_count,
        violations=violations,
        flagged=flagged,
        cardinality_groups={
            "source": len(overloaded_sources),
            "target": len(overloaded_targets),
        },
    )


def filter_alignment(
    alignment: Alignment,
    report: AlignmentReport,
    kinds: tuple[str, ...] = ALIGNMENT_KINDS,
) -> Alignment:
    """Drop every match flagged with any of `kinds`, keeping the rest."""
    dropped = set(kinds)
    kept = [
        m for m in alignment.matches
        if not dropped.intersection(report.flagged.get((m.source_id, m.target_id), ()))
    ]
    return Alignment(source=alignment.source, target=alignment.target, matches=kept)


def flag_quality(
    alignment: Alignment,
    report: AlignmentReport,
    ground_truth: Alignment,
    kinds: tuple[str, ...] = ALIGNMENT_KINDS,
) -> dict[str, float]:
    """Score the flags as a detector of wrong matches.

    `error_precision` is the share of flagged matches that are wrong, and
    `error_recall` is the share of wrong matches that got flagged. Both are
    measured against the ground truth, and `base_error_rate` (the share of
    wrong matches in the whole alignment) is reported next to them, because a
    flag that marked every match would score exactly that rate: a flag is
    useful only when its `precision_lift` over the base rate is positive
    (F3, review/review-consistency.md).

    `flagged_errors` and `errors` are the raw sums, so a caller can pool the
    ratio over several alignments instead of averaging per-record ratios
    (F8).
    """
    gt = ground_truth.as_pairs()
    dropped = set(kinds)
    flagged_keys = {
        key for key, ks in report.flagged.items() if dropped.intersection(ks)
    }
    flagged_errors = sum(1 for key in flagged_keys if key not in gt)
    missed_errors = sum(
        1 for m in alignment.matches
        if (m.source_id, m.target_id) not in gt
        and (m.source_id, m.target_id) not in flagged_keys
    )
    total_errors = flagged_errors + missed_errors
    matches = alignment.match_count
    precision = flagged_errors / len(flagged_keys) if flagged_keys else 0.0
    base_error_rate = total_errors / matches if matches else 0.0
    return {
        "matches": float(matches),
        "flagged": float(len(flagged_keys)),
        "flagged_errors": float(flagged_errors),
        "errors": float(total_errors),
        "error_precision": precision,
        "error_recall": flagged_errors / total_errors if total_errors else 0.0,
        "base_error_rate": base_error_rate,
        "precision_lift": precision - base_error_rate,
    }
