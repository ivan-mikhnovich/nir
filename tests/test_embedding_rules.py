"""Tests for the embedding post-processing rules.

The review (embedding F2/F4) found the published embedding baseline used the
nearest-target `argmax` rule alone, which keeps one match per source but lets
several sources claim the same target — so the column was never the 1:1
mapping the note described, and the alternative rules existed only in the
runner. The matcher now offers the three documented modes
(`POSTPROCESS_MODES`): `argmax` (repeats targets), `dedupe_by_target` (keeps
the best claim per target) and `greedy_injective` (may reassign a source to
its next-best candidate). All three are exercised here on a hand-made cosine
matrix, and the runner records which rule produced a cached record.

No model is loaded: the similarity matrix is fed in through the same
`similarity()` seam the encoder writes to.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.matchers.embedding import POSTPROCESS_MODES, EmbeddingMatcher

SRC_IDS = ("s0", "s1", "s2")
TGT_IDS = ("t0", "t1")
# Hand-made cosines: s0 and s1 both prefer t0, s2 prefers t1.
SIM = np.array(
    [
        [0.90, 0.60],
        [0.80, 0.75],
        [0.50, 0.70],
    ]
)


def build_matcher(
    sim: np.ndarray = SIM,
    src_ids: tuple[str, ...] = SRC_IDS,
    tgt_ids: tuple[str, ...] = TGT_IDS,
) -> EmbeddingMatcher:
    """Build an `EmbeddingMatcher` around a fixed similarity matrix."""
    matcher = object.__new__(EmbeddingMatcher)
    matcher.model_name = "stub"
    matcher.description_mode = "name_only"
    matcher._dim = int(sim.shape[1])

    def _similarity(_source, _target, _description_mode=None):
        return list(src_ids), list(tgt_ids), sim

    matcher.similarity = _similarity
    return matcher


def sources(taxonomy) -> tuple:
    return taxonomy({node: [] for node in SRC_IDS}), taxonomy({node: [] for node in TGT_IDS})


def test_argmax_returns_one_match_per_source_and_may_reuse_targets(taxonomy):
    source, target = sources(taxonomy)
    alignment, scores = build_matcher().match(source, target, postprocess="argmax")

    pairs = alignment.as_pairs()
    assert pairs == {("s0", "t0"), ("s1", "t0"), ("s2", "t1")}
    # One prediction per source, but t0 carries two sources: not injective.
    assert {src for src, _ in pairs} == set(SRC_IDS)
    assert len({tgt for _, tgt in pairs}) == 2
    assert scores == pytest.approx([0.90, 0.80, 0.70])


def test_dedupe_by_target_is_injective_and_keeps_the_best_claim(taxonomy):
    source, target = sources(taxonomy)
    alignment, _ = build_matcher().match(source, target, postprocess="dedupe_by_target")

    pairs = alignment.as_pairs()
    assert pairs == {("s0", "t0"), ("s2", "t1")}
    # For the contested t0 the stronger claim (s0, 0.90) wins.
    assert ("s1", "t0") not in pairs
    assert len({src for src, _ in pairs}) == len(pairs)
    assert len({tgt for _, tgt in pairs}) == len(pairs)


def test_greedy_injective_reassigns_a_source_to_its_second_best(taxonomy):
    source, target = sources(taxonomy)
    matcher = build_matcher()

    alignment, _ = matcher.match(source, target, postprocess="greedy_injective")
    pairs = alignment.as_pairs()
    assert pairs == {("s0", "t0"), ("s1", "t1")}
    # s1's best (t0) is taken, so it falls back to t1; s2's t1 is then taken too.
    assert ("s1", "t0") not in pairs
    assert ("s1", "t1") in pairs
    assert ("s2", "t1") not in pairs
    assert len({src for src, _ in pairs}) == len(pairs)
    assert len({tgt for _, tgt in pairs}) == len(pairs)

    # The nearest-target rule instead drops s1 entirely, so the two modes
    # genuinely differ on the same matrix.
    deduped, _ = matcher.match(source, target, postprocess="dedupe_by_target")
    assert ("s1", "t1") not in deduped.as_pairs()


def test_threshold_drops_weak_candidates(taxonomy):
    source, target = sources(taxonomy)
    matcher = build_matcher()

    uncut, _ = matcher.match(source, target, postprocess="argmax")
    assert ("s2", "t1") in uncut.as_pairs()  # s2's best is 0.70

    cut, _ = matcher.match(source, target, postprocess="argmax", threshold=0.72)
    assert ("s2", "t1") not in cut.as_pairs()
    assert cut.as_pairs() == {("s0", "t0"), ("s1", "t0")}

    deduped, _ = matcher.match(
        source, target, postprocess="dedupe_by_target", threshold=0.72
    )
    assert deduped.as_pairs() == {("s0", "t0")}

    greedy, _ = matcher.match(
        source, target, postprocess="greedy_injective", threshold=0.78
    )
    assert greedy.as_pairs() == {("s0", "t0")}


def test_unknown_postprocess_mode_is_rejected(taxonomy):
    source, target = sources(taxonomy)
    with pytest.raises(ValueError):
        build_matcher().match(source, target, postprocess="not-a-rule")


def test_runner_records_the_chosen_postprocess_rule(tmp_path, monkeypatch):
    """`save_results` is the runner's recording helper; it must keep the rule.

    The rebuild path passes `extra={"wall_time":…, "model":…, "postprocess":…}`,
    so the produced record carries the rule that built its `match_pairs`. The
    write is redirected to `tmp_path`; `results/` is never touched.
    """
    from src.runners import compare

    monkeypatch.setattr(
        compare,
        "cache_path",
        lambda approach, a, b, mode=None: tmp_path / f"{approach}_{a}_{b}.json",
    )
    metrics = SimpleNamespace(
        precision=1.0,
        recall=1.0,
        f1=1.0,
        true_positives=1,
        total_predicted=1,
        total_ground_truth=1,
    )
    compare.save_results(
        "embedding-stub",
        "cmt",
        "confOf",
        metrics,
        extra={"wall_time": 0.0, "model": "stub", "postprocess": "greedy_injective"},
        match_pairs=[
            {"source_id": "cmt#A", "target_id": "confof#B", "confidence": 1.0}
        ],
    )

    record = json.loads(
        (tmp_path / "embedding-stub_cmt_confOf.json").read_text(encoding="utf-8")
    )
    assert record["postprocess"] == "greedy_injective"
    assert record["postprocess"] in POSTPROCESS_MODES
    assert record["model"] == "stub"
    assert record["pair"] == "cmt↔confOf"
