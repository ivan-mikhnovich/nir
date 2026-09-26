"""Tests for the consistency checks.

Two review findings are pinned here. First, `unreachable-class` used to be
unable to fire on a dangling parent link, because a node whose parent is
missing was treated as a root; a dangling parent must now be reported as its
own kind and must leave the node unreachable. Second, the violation counts used
to depend on the order of `alignment.matches` (a target with several sources
kept whichever source came last), so shuffling the list changed the numbers.
"""

from __future__ import annotations

import random

from src import consistency as cons


def test_a_healthy_taxonomy_reports_nothing(taxonomy):
    report = cons.check_taxonomy(taxonomy({"r": [], "a": ["r"], "b": ["a", "r"]}))
    assert report.counts() == {}
    assert report.violations == []


def test_cycles_are_reported(taxonomy):
    report = cons.check_taxonomy(taxonomy({"x": ["y"], "y": ["x"]}))
    assert report.counts().get(cons.CYCLE, 0) >= 1


def test_a_dangling_parent_is_reported_and_leaves_the_node_unreachable(taxonomy):
    report = cons.check_taxonomy(taxonomy({"r": [], "x": ["missing"]}))
    counts = report.counts()
    assert counts.get(cons.DANGLING_PARENT) == 1
    assert cons.UNREACHABLE in counts, "узел с отсутствующим родителем обязан быть недостижим"
    assert report.root_count == 1, "узел с висячей ссылкой не должен считаться корнем"


def test_duplicate_labels_are_reported_case_insensitively(taxonomy):
    tax = taxonomy(
        {"a": [], "b": []},
        a={"name": "Author"},
        b={"name": "  author "},
    )
    assert cons.check_taxonomy(tax).counts().get(cons.DUPLICATE_LABEL) == 1


def test_two_disjoint_classes_in_one_scope_are_reported(taxonomy):
    tax = taxonomy(
        {"r": [], "x": ["r"], "y": ["r"], "z": ["x", "y"]},
        x={"disjoint_with": ["y"]},
    )
    assert cons.check_taxonomy(tax).counts().get(cons.DISJOINT_ANCESTOR, 0) >= 1


def test_an_overloaded_target_counts_once_but_flags_every_match(taxonomy, alignment):
    src = taxonomy({"a": [], "b": []})
    tgt = taxonomy({"c": []})
    report = cons.check_alignment(
        src,
        tgt,
        alignment("s", "t", [("a", "c"), ("b", "c")]),
        approach="test",
        pair="s↔t",
    )
    assert report.match_count == 2
    assert report.counts().get(cons.CARDINALITY) == 1
    assert len(report.flagged) == 2


def test_a_source_injective_alignment_has_no_cardinality_violation(taxonomy, alignment):
    src = taxonomy({"a": [], "b": []})
    tgt = taxonomy({"c": [], "d": []})
    report = cons.check_alignment(src, tgt, alignment("s", "t", [("a", "c"), ("b", "d")]))
    assert cons.CARDINALITY not in report.counts()


def test_shuffling_the_matches_does_not_change_the_counts(taxonomy, alignment):
    src = taxonomy({"a": [], "b": ["a"], "c": []})
    tgt = taxonomy({"x": [], "y": ["x"]})
    pairs = [("a", "y"), ("b", "x"), ("c", "y"), ("c", "x")]
    base = cons.check_alignment(src, tgt, alignment("s", "t", pairs), pair="s↔t")
    base_counts = base.counts()
    base_flags = {key: sorted(kinds) for key, kinds in base.flagged.items()}
    for seed in range(5):
        shuffled = pairs[:]
        random.Random(seed).shuffle(shuffled)
        report = cons.check_alignment(src, tgt, alignment("s", "t", shuffled), pair="s↔t")
        assert report.counts() == base_counts
        assert {k: sorted(v) for k, v in report.flagged.items()} == base_flags


def test_low_confidence_is_not_a_consistency_check():
    assert cons.ALIGNMENT_KINDS == (cons.CARDINALITY, cons.DISJOINTNESS, cons.SUBSUMPTION)
    assert "low-confidence" not in cons.SEVERITY
