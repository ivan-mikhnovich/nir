"""Train and evaluate Siamese GraphSAGE for taxonomy matching.

Uses leave-one-out cross-validation on the 21 OAEI Conference track
pairs — standard protocol in ontology matching (GraphMatcher, LogMap).

Protocols (`--fold-mode`):
  * `pair-loo` (default) — transductive pair leave-one-out: one pair is held
    out and the model trains on the other 20.  The held-out pair's ontologies
    usually still appear in training (as endpoints of the other pairs and as
    synthetic variants — see `--no-synthetic`), so this measures matching on
    *seen* ontologies and its folds are not ontology-disjoint (finding G3).
  * `ontology-loo` — leave-one-ontology-out: every pair containing the
    held-out ontology (6 of them) is excluded from training, together with
    every synthetic variant derived from either endpoint of those pairs, so
    the fold never sees the held-out ontology's names, graph or variants
    (finding G3).

Reproducibility and checkpoints (findings G8, G11):
  * every run is seeded from `--seed` before anything random happens, and the
    seed plus the full configuration are stored in every per-pair record and
    in every checkpoint;
  * the weights that a fold is evaluated with (the final epoch, which is what
    `_evaluate_model` receives) are written to
    `results/gnn/fold-<k>.pt` / `results/gnn/fold-onto-<name>.pt`;
    `results/gnn/model.pt` is only written by `--train` (full-data diagnostic)
    and is therefore *not* a leave-one-out model;
  * every record carries `match_pairs`, so the consistency audit can rebuild
    the predicted alignment (finding F9).

Usage:
    # Train + evaluate full leave-one-out (21 folds, ~40 min).
    uv run python -m src.runners.gnn --evaluate-all

    # Ontology-level leave-one-out (7 folds × 6 pairs).
    uv run python -m src.runners.gnn --evaluate-all --fold-mode ontology-loo

    # Cheap smoke: one fold, one epoch.
    uv run python -m src.runners.gnn --evaluate-all --folds 1 --epochs 1

    # Train on all pairs (diagnostic), no evaluation.
    uv run python -m src.runners.gnn --train

    # Train on 20 pairs, test on one specific pair.
    uv run python -m src.runners.gnn --test-pair cmt↔conference

    # Re-evaluate from a saved checkpoint.
    uv run python -m src.runners.gnn --evaluate
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..data_loader import load_oaei_conference
from ..matching_rules import Candidate, greedy_injective, to_alignment
from ..matchers.gnn import LinearSiamese, SiameseGraphSAGE
from ..metrics import MatchMetrics, evaluate_1to1
from ..taxonomy import Alignment, Taxonomy


RESULTS_DIR = Path("results/gnn")
CHECKPOINT_PATH = RESULTS_DIR / "model.pt"
BASELINE_DIR = Path("results/gnn-baseline")
DATA_DIR = Path("data/raw/oaei")


# ---------------------------------------------------------------------------
# Configuration and seeding.
# ---------------------------------------------------------------------------


def _run_config(args: argparse.Namespace, fold: str | None = None) -> dict:
    """Return the full configuration of a run (finding G11).

    Stored verbatim in every per-pair record and in every checkpoint, so a
    published number can be traced back to the exact hyperparameters.
    """
    return {
        "seed": args.seed,
        "embedder": args.embedder,
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "neg_ratio": args.neg_ratio,
        "synthetic_seeds": 0 if args.no_synthetic else args.synthetic_seeds,
        "linear": args.linear,
        "fold_mode": args.fold_mode,
        "fold": fold,
    }


def _seed_everything(seed: int) -> None:
    """Seed every random source used by training (finding G11)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training helpers.
# ---------------------------------------------------------------------------


def _build_train_data(
    taxonomies: dict[str, Taxonomy],
    all_alignments: list[Alignment],
    held_out_pairs: set[str] | None = None,
    num_synthetic_seeds: int = 3,
    held_out_ontologies: set[str] | None = None,
) -> list[tuple[Taxonomy, Taxonomy, Alignment]]:
    """Build training pairs from OAEI data + synthetic augmentation.

    Synthetic pairs (variations of real ontologies) are added to increase
    the number of training examples.  Only source ontologies that appear in
    real training pairs are used for synthetic generation, and every ontology
    named in ``held_out_ontologies`` is excluded from both the real pairs and
    the synthetic generation — so an ontology-level hold-out really removes
    that ontology from training, including its variants (finding G3).

    Args:
        taxonomies: All loaded taxonomies keyed by name.
        all_alignments: All 21 reference alignments.
        held_out_pairs: Set of pair keys to exclude from training.
        num_synthetic_seeds: Synthetic variations per source ontology.
        held_out_ontologies: Ontology names to exclude completely.

    Returns:
        List of (source_tax, target_tax, ground_truth_alignment) triples.
    """
    from ..data_loader import generate_synthetic_variations

    if held_out_pairs is None:
        held_out_pairs = set()
    if held_out_ontologies is None:
        held_out_ontologies = set()

    train_pairs: list[tuple[Taxonomy, Taxonomy, Alignment]] = []
    source_names_used: set[str] = set()

    for gt in all_alignments:
        pair_key = f"{gt.source}↔{gt.target}"
        if pair_key in held_out_pairs:
            continue
        if gt.source in held_out_ontologies or gt.target in held_out_ontologies:
            continue
        src = taxonomies.get(gt.source)
        tgt = taxonomies.get(gt.target)
        if src is None or tgt is None:
            continue
        train_pairs.append((src, tgt, gt))
        source_names_used.add(gt.source)

    # Add synthetic variations for ontologies appearing in real training pairs.
    for src_name in sorted(source_names_used):
        if src_name in held_out_ontologies:
            continue
        src_tax = taxonomies[src_name]
        for seed in range(num_synthetic_seeds):
            variations = generate_synthetic_variations(
                src_tax, seed=seed * 7 + 13,
            )
            for _scenario, (variant, gt_align) in variations.items():
                gt_align.source = src_tax.name
                gt_align.target = variant.name
                train_pairs.append((src_tax, variant, gt_align))

    return train_pairs


def _report_train_data(
    train_pairs: list[tuple[Taxonomy, Taxonomy, Alignment]],
) -> dict:
    """Report the effective size of a fold's training set (finding G4).

    The nominal numbers count every entry of ``train_pairs``; the distinct
    numbers collapse entries that share the same (source, target) taxonomy
    pair, which is what a name-keyed feature cache actually trains on.

    Returns:
        Dict with the nominal and distinct triple/positive counts.
    """
    by_pair: dict[tuple[str, str], Alignment] = {}
    for src_tax, tgt_tax, gt_align in train_pairs:
        by_pair.setdefault((src_tax.name, tgt_tax.name), gt_align)
    nominal_pos = sum(len(gt.as_pairs()) for _, _, gt in train_pairs)
    distinct_pos = sum(len(gt.as_pairs()) for gt in by_pair.values())
    variants = sorted({
        name
        for src_tax, tgt_tax, _ in train_pairs
        for name in (src_tax.name, tgt_tax.name)
        if name not in ("cmt", "conference", "confOf", "edas", "ekaw", "iasted", "sigkdd")
    })
    report = {
        "nominal_triples": len(train_pairs),
        "distinct_graph_pairs": len(by_pair),
        "nominal_positives": nominal_pos,
        "distinct_positives": distinct_pos,
        "variant_taxonomies": len(variants),
    }
    print(f"Training triples: {report['nominal_triples']} nominal, "
          f"{report['distinct_graph_pairs']} distinct graph pairs")
    print(f"Positive examples: {nominal_pos} nominal, {distinct_pos} on distinct data")
    print(f"Synthetic variant taxonomies: {report['variant_taxonomies']}")
    return report


def _precompute_cache(
    taxonomies: dict[str, Taxonomy],
    embedder_model: str,
    device: str,
) -> dict[str, tuple[list[str], torch.Tensor, torch.Tensor]]:
    """Pre-compute BERT features and adjacency for all taxonomies.

    Returns dict: name → (node_ids, features[N,in_dim], adjacency[N,N]).
    All tensors are on the target device.
    """
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embedder_model)
    cache: dict[str, tuple[list[str], torch.Tensor, torch.Tensor]] = {}

    for name, tax in tqdm(list(taxonomies.items()), desc="Precomputing features"):
        node_ids, adj = tax.to_adjacency()
        texts = [tax.nodes[nid].name for nid in node_ids]
        feats = torch.from_numpy(
            embedder.encode(texts, show_progress_bar=False)
        ).float().to(device)
        adj = adj.to(device)
        cache[name] = (node_ids, feats, adj)

    return cache


def _train_epoch(
    model: SiameseGraphSAGE,
    pairs: list[tuple[Taxonomy, Taxonomy, Alignment]],
    cache: dict[str, tuple[list[str], torch.Tensor, torch.Tensor]],
    device: str,
    optimizer: torch.optim.Optimizer,
    neg_ratio: int = 3,
) -> float:
    """Train one epoch over all (source, target, gt) triples.

    Features and adjacency are fetched from the pre-computed cache.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for src_tax, tgt_tax, gt_align in tqdm(pairs, desc="Training", unit="pair"):
        gt_pairs = gt_align.as_pairs()  # {(src_id, tgt_id), ...}

        src_name = src_tax.name
        tgt_name = tgt_tax.name

        src_ids, src_feat, src_adj = cache[src_name]
        tgt_ids, tgt_feat, tgt_adj = cache[tgt_name]

        sim = model(src_feat, src_adj, tgt_feat, tgt_adj)  # [N_s, N_t]

        src_to_idx = {nid: i for i, nid in enumerate(src_ids)}
        tgt_to_idx = {nid: i for i, nid in enumerate(tgt_ids)}

        pos_scores: list[torch.Tensor] = []
        neg_scores: list[torch.Tensor] = []

        for src_id, tgt_id in gt_pairs:
            if src_id not in src_to_idx or tgt_id not in tgt_to_idx:
                continue
            i = src_to_idx[src_id]
            j = tgt_to_idx[tgt_id]
            pos_scores.append(sim[i, j])

            neg_candidates = [k for k in range(len(tgt_ids)) if k != j]
            if not neg_candidates:
                continue
            neg_sample = random.sample(
                neg_candidates,
                min(neg_ratio, len(neg_candidates)),
            )
            for nj in neg_sample:
                neg_scores.append(sim[i, nj])

        if not pos_scores:
            continue

        pos_tensor = torch.stack(pos_scores)
        neg_tensor = (
            torch.stack(neg_scores) if neg_scores
            else torch.tensor([], device=device)
        )

        loss_pos = F.mse_loss(pos_tensor, torch.ones_like(pos_tensor))
        loss_neg = (
            F.mse_loss(neg_tensor, torch.zeros_like(neg_tensor))
            if neg_scores else torch.tensor(0.0, device=device)
        )
        loss = loss_pos + loss_neg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _save_checkpoint(
    model: SiameseGraphSAGE,
    path: Path,
    in_dim: int,
    args: argparse.Namespace,
    fold: str | None = None,
) -> None:
    """Save evaluated weights and the full run configuration (G8, G11).

    The tensor layout keeps the keys `GNNMatcher.load_checkpoint` reads
    (`in_dim`, `hidden_dim`, `out_dim`, `num_layers`, `model_state`) and adds
    `config` with the seed, the hyperparameters and the fold label.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "in_dim": in_dim,
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "num_layers": args.num_layers,
        "linear": args.linear,
        "config": _run_config(args, fold),
        "model_state": model.state_dict(),
    }, path)


# ---------------------------------------------------------------------------
# Evaluation helpers.
# ---------------------------------------------------------------------------


def _similarity_candidates(
    sim: np.ndarray,
    src_ids: list[str],
    tgt_ids: list[str],
) -> list[Candidate]:
    """Return every (source_id, target_id, confidence) candidate of a matrix."""
    return [
        (src_ids[i], tgt_ids[j], float(sim[i, j]))
        for i in range(len(src_ids))
        for j in range(len(tgt_ids))
    ]


def _match(
    sim: np.ndarray,
    src_ids: list[str],
    tgt_ids: list[str],
    source: str,
    target: str,
    threshold: float | None = None,
) -> Alignment:
    """Reduce a similarity matrix to a 1:1 alignment via the shared rule.

    Candidates go through :func:`matching_rules.greedy_injective`, so at most
    one target per source *and* at most one source per target (findings
    2.1/1.10: the removed local matcher never excluded sources, so it emitted
    one match per target node and the number of predictions was
    ``|targets|``, not ``min(|S|, |T|)``).
    """
    pairs = _similarity_candidates(sim, src_ids, tgt_ids)
    return to_alignment(greedy_injective(pairs, threshold), source, target)


def _evaluate_model(
    model: SiameseGraphSAGE,
    taxonomies: dict[str, Taxonomy],
    test_alignments: list[Alignment],
    cache: dict[str, tuple[list[str], torch.Tensor, torch.Tensor]],
) -> dict[str, tuple[MatchMetrics, Alignment]]:
    """Evaluate on a list of test alignments.

    Returns:
        Dict mapping pair key → (metrics, predicted alignment).  The alignment
        is stored as `match_pairs` so the consistency audit can rebuild it.
    """
    model.eval()
    results: dict[str, tuple[MatchMetrics, Alignment]] = {}

    for gt_align in tqdm(test_alignments, desc="Evaluating", unit="pair"):
        src_name = gt_align.source
        tgt_name = gt_align.target
        if src_name not in cache or tgt_name not in cache:
            continue

        src_ids, src_feat, src_adj = cache[src_name]
        tgt_ids, tgt_feat, tgt_adj = cache[tgt_name]

        with torch.no_grad():
            sim = model(src_feat, src_adj, tgt_feat, tgt_adj)

        sim_np = sim.cpu().numpy()
        pred = _match(sim_np, src_ids, tgt_ids, gt_align.source, gt_align.target)

        metrics = evaluate_1to1(pred, gt_align)
        results[f"{gt_align.source}↔{gt_align.target}"] = (metrics, pred)

    return results


def _save_results(
    pair_key: str,
    metrics: MatchMetrics,
    alignment: Alignment,
    approach: str = "gnn",
    results_dir: Path = RESULTS_DIR,
    meta: dict | None = None,
) -> None:
    """Cache per-pair evaluation results as JSON.

    The key set is the one `src.runners.compare.save_results` writes, so the
    comparison table and the consistency audit read GNN records on the same
    code path as every other approach; `match_pairs`, the run seed and the
    full configuration travel with the metrics (findings F9, G11).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    # Canonical (sorted) pair key for consistent cross-approach merging.
    parts = sorted(pair_key.split("↔"))
    pair_key_canonical = "↔".join(parts)
    fname = pair_key_canonical.replace("↔", "_").lower()
    fpath = results_dir / f"{fname}.json"
    data = {
        "pair": pair_key_canonical,
        "approach": approach,
        "source": alignment.source,
        "target": alignment.target,
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "matches": metrics.true_positives,
        "total_predicted": metrics.total_predicted,
        "total_gt": metrics.total_ground_truth,
        "match_pairs": [
            {
                "source_id": m.source_id,
                "target_id": m.target_id,
                "confidence": m.confidence,
            }
            for m in alignment.matches
        ],
        **(meta or {}),
    }
    fpath.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core training procedure.
# ---------------------------------------------------------------------------


def _train_model(
    taxonomies: dict[str, Taxonomy],
    all_alignments: list[Alignment],
    embedder_model: str,
    held_out_pairs: set[str] | None,
    args: argparse.Namespace,
    held_out_ontologies: set[str] | None = None,
    checkpoint_path: Path = CHECKPOINT_PATH,
    fold_label: str | None = None,
    base_cache: dict[str, tuple[list[str], torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[SiameseGraphSAGE, dict[str, tuple[list[str], torch.Tensor, torch.Tensor]]]:
    """Train a SiameseGraphSAGE from scratch.

    The model is seeded deterministically (same seed for every fold), and the
    weights that this function returns — the final epoch, which the caller
    evaluates — are written to ``checkpoint_path`` (finding G8).

    Args:
        taxonomies: All OAEI taxonomies.
        all_alignments: All 21 reference alignments.
        embedder_model: Sentence-transformers model name.
        held_out_pairs: Pairs to exclude from training (for leave-one-out).
        args: CLI arguments.
        held_out_ontologies: Ontologies to exclude completely (ontology-LOO).
        checkpoint_path: Where to save the evaluated weights.
        fold_label: Human-readable fold name stored in the checkpoint.
        base_cache: Feature cache of the real taxonomies, reused across folds
            instead of re-encoding them for every fold.

    Returns:
        (trained model on the specified device, feature cache extended with the
        fold's synthetic variants).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    _seed_everything(args.seed)
    config = _run_config(args, fold_label)
    print(f"Seed: {args.seed}")
    print(f"Config: {config}")

    # Features and adjacency of the real taxonomies (shared across folds),
    # extended below with the synthetic variants this fold trains on.
    cache = dict(base_cache) if base_cache is not None else _precompute_cache(
        taxonomies, embedder_model, device,
    )

    # Build training pairs first, then extend cache with synthetic variants.
    train_pairs = _build_train_data(
        taxonomies, all_alignments, held_out_pairs,
        num_synthetic_seeds=0 if args.no_synthetic else args.synthetic_seeds,
        held_out_ontologies=held_out_ontologies,
    )
    _report_train_data(train_pairs)

    # Extend cache with any taxonomy variants not yet cached.
    extra_taxonomies: dict[str, Taxonomy] = {}
    for src_tax, tgt_tax, _gt in train_pairs:
        for tx in (src_tax, tgt_tax):
            if tx.name not in cache and tx.name not in extra_taxonomies:
                extra_taxonomies[tx.name] = tx
    if extra_taxonomies:
        extra_cache = _precompute_cache(extra_taxonomies, embedder_model, device)
        cache.update(extra_cache)

    # Infer input dimension from first cached feature tensor.
    _, first_feat, _ = next(iter(cache.values()))
    in_dim = first_feat.shape[1]
    total_pos = sum(len(gt.as_pairs()) for _, _, gt in train_pairs)
    print(f"Training pairs: {len(train_pairs)}, "
          f"total positive examples: {total_pos}")
    print(f"Hidden dim: {args.hidden_dim}, Output dim: {args.out_dim}")
    print(f"Layers: {args.num_layers}, Dropout: {args.dropout}")
    print(f"LR: {args.lr}, Weight decay: {args.weight_decay}")
    print()

    model_cls = LinearSiamese if args.linear else SiameseGraphSAGE
    init_kwargs: dict = {
        "in_dim": in_dim,
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "dropout": args.dropout,
    }
    if not args.linear:
        init_kwargs["num_layers"] = args.num_layers
    model = model_cls(**init_kwargs).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        random.shuffle(train_pairs)
        avg_loss = _train_epoch(
            model, train_pairs, cache, device, optimizer,
            neg_ratio=args.neg_ratio,
        )
        print(f"Epoch {epoch + 1}/{args.epochs}, loss={avg_loss:.6f}")

    # The caller evaluates these weights, so these are the ones on disk.
    _save_checkpoint(model, checkpoint_path, in_dim, args, fold_label)
    print(f"Evaluated weights saved to {checkpoint_path}\n")
    return model, cache


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _train_all(args: argparse.Namespace) -> None:
    """Train on all 21 pairs and save checkpoint.

    The saved `results/gnn/model.pt` holds the final-epoch weights of this
    full-data run; leave-one-out models are written per fold by
    `--evaluate-all` (finding G8).
    """
    print("=== Training Siamese GraphSAGE (all 21 pairs) ===\n")
    taxonomies, alignments = load_oaei_conference(DATA_DIR)
    _train_model(taxonomies, alignments, args.embedder, None, args,
                 fold_label="all-pairs")
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")


def _evaluate_from_checkpoint(args: argparse.Namespace) -> None:
    """Evaluate a trained model on all 21 pairs."""
    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint at {CHECKPOINT_PATH}.  Run --train first.")
        return

    print("=== Evaluating Siamese GraphSAGE ===\n")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt.get("linear"):
        model = LinearSiamese(
            in_dim=ckpt["in_dim"],
            hidden_dim=ckpt["hidden_dim"],
            out_dim=ckpt["out_dim"],
        ).to(device)
    else:
        model = SiameseGraphSAGE(
            in_dim=ckpt["in_dim"],
            hidden_dim=ckpt["hidden_dim"],
            out_dim=ckpt["out_dim"],
            num_layers=ckpt["num_layers"],
        ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {"seed": args.seed, "config": ckpt.get("config", _run_config(args, "checkpoint"))}
    taxonomies, alignments = load_oaei_conference(DATA_DIR)
    cache = _precompute_cache(taxonomies, args.embedder, device)

    results = _evaluate_model(model, taxonomies, alignments, cache)

    print()
    print(f"{'Pair':<25s} {'F1':>6s}  {'P':>6s}  {'R':>6s}")
    print("-" * 47)
    for pair_key in sorted(results.keys()):
        metrics, pred = results[pair_key]
        _save_results(pair_key, metrics, pred, meta=meta)
        print(f"{pair_key:<25s} {metrics.f1:6.4f}  {metrics.precision:6.4f}  {metrics.recall:6.4f}")

    avg_f1 = np.mean([metrics.f1 for metrics, _ in results.values()])
    print("-" * 47)
    print(f"{'AVERAGE':<25s} {avg_f1:6.4f}")


def _test_single_pair(args: argparse.Namespace) -> None:
    """Train leaving out one specific pair, then evaluate on that pair."""
    test_pair = args.test_pair
    print(f"=== Leave-one-out: test pair = {test_pair} ===\n")

    taxonomies, alignments = load_oaei_conference(DATA_DIR)

    all_pair_keys = {f"{a.source}↔{a.target}" for a in alignments}

    # Canonicalise: try both orderings.
    pair_a, pair_b = test_pair.split("↔") if "↔" in test_pair else ("", "")
    pair_forward = f"{pair_a}↔{pair_b}"
    pair_reverse = f"{pair_b}↔{pair_a}"

    if pair_forward in all_pair_keys:
        held_out = {pair_forward}
    elif pair_reverse in all_pair_keys:
        held_out = {pair_reverse}
    else:
        print(f"Unknown pair: {test_pair}")
        print(f"Available: {sorted(all_pair_keys)}")
        return

    # Train on 20 pairs.
    print(f"Holding out: {held_out}")
    held_label = "↔".join(sorted(next(iter(held_out)).split("↔")))
    checkpoint_path = RESULTS_DIR / f"fold-test-{held_label.replace('↔', '_').lower()}.pt"
    model, cache = _train_model(taxonomies, alignments, args.embedder, held_out, args,
                                checkpoint_path=checkpoint_path,
                                fold_label=f"pair {held_label}")

    # Evaluate only on the test pair.
    test_alignments = [
        a for a in alignments
        if f"{a.source}↔{a.target}" in held_out
        or f"{a.target}↔{a.source}" in held_out
    ]

    results = _evaluate_model(model, taxonomies, test_alignments, cache)
    meta = {"seed": args.seed, "config": _run_config(args, f"pair {held_label}")}

    print()
    for pair_key, (metrics, pred) in sorted(results.items()):
        _save_results(pair_key, metrics, pred, meta=meta)
        print(f"{pair_key}: F1={metrics.f1:.4f}, P={metrics.precision:.4f}, R={metrics.recall:.4f}")


def _make_folds(
    args: argparse.Namespace,
    taxonomies: dict[str, Taxonomy],
    alignments: list[Alignment],
) -> list[dict]:
    """Build the leave-one-out folds for the requested protocol.

    `pair-loo` yields one fold per reference pair (held_out_pairs = that pair);
    `ontology-loo` yields one fold per ontology, holding out every pair that
    contains it (6 pairs each) and the ontology itself (finding G3).

    Args:
        args: CLI arguments (`fold_mode`, `folds`).
        taxonomies: Loaded taxonomies.
        alignments: The 21 reference alignments.

    Returns:
        List of fold dicts with label, held-out pairs/ontologies, test pairs
        and the checkpoint path of the fold.
    """
    folds: list[dict] = []
    if args.fold_mode == "ontology-loo":
        for onto in sorted(taxonomies):
            held = {
                f"{a.source}↔{a.target}"
                for a in alignments
                if onto in (a.source, a.target)
            }
            folds.append({
                "label": f"ontology {onto}",
                "held_pairs": held,
                "held_ontologies": {onto},
                "test_pairs": held,
                "checkpoint": RESULTS_DIR / f"fold-onto-{onto}.pt",
            })
    else:
        for align in alignments:
            key = f"{align.source}↔{align.target}"
            folds.append({
                "label": f"pair {key}",
                "held_pairs": {key},
                "held_ontologies": set(),
                "test_pairs": {key},
                "checkpoint": RESULTS_DIR / f"fold-{len(folds) + 1}.pt",
            })
    return folds


def _evaluate_all_loocv(args: argparse.Namespace) -> None:
    """Leave-one-out cross-validation in the direction given by --fold-mode.

    `pair-loo` runs 21 folds (train on 20 pairs, test on one); `ontology-loo`
    runs 7 folds (hold out one ontology and its 6 pairs).  `--folds N` keeps
    only the first N folds, which makes a cheap smoke run possible.
    """
    print(f"=== Leave-one-out cross-validation ({args.fold_mode}) ===\n")

    taxonomies, alignments = load_oaei_conference(DATA_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    folds = _make_folds(args, taxonomies, alignments)
    if args.folds and args.folds < len(folds):
        print(f"--folds {args.folds}: running the first {args.folds} of {len(folds)} folds.")
        folds = folds[:args.folds]

    # Pre-compute once for all folds.
    cache = _precompute_cache(taxonomies, args.embedder, device)

    all_results: dict[str, tuple[MatchMetrics, Alignment]] = {}

    for i, fold in enumerate(folds):
        print(f"{'─' * 60}")
        print(f"Fold {i + 1}/{len(folds)}: testing {fold['label']}")
        print(f"{'─' * 60}")

        model, cache = _train_model(
            taxonomies, alignments, args.embedder, fold["held_pairs"], args,
            held_out_ontologies=fold["held_ontologies"],
            checkpoint_path=fold["checkpoint"],
            fold_label=fold["label"],
            base_cache=cache,
        )

        test_alignments = [
            a for a in alignments
            if f"{a.source}↔{a.target}" in fold["test_pairs"]
        ]
        results = _evaluate_model(model, taxonomies, test_alignments, cache)
        all_results.update(results)

        # Cache immediately, so even a truncated (smoke) run writes records.
        meta = {"seed": args.seed, "config": _run_config(args, fold["label"])}
        for pair_key, (metrics, pred) in sorted(results.items()):
            _save_results(pair_key, metrics, pred, meta=meta)

    # Final table.
    print()
    print("=" * 47)
    print(f"  Full leave-one-out results ({args.fold_mode})")
    print("=" * 47)
    print(f"{'Pair':<25s} {'F1':>6s}  {'P':>6s}  {'R':>6s}")
    print("-" * 47)
    for pair_key in sorted(all_results.keys()):
        metrics, _pred = all_results[pair_key]
        print(f"{pair_key:<25s} {metrics.f1:6.4f}  {metrics.precision:6.4f}  {metrics.recall:6.4f}")

    avg_f1 = np.mean([metrics.f1 for metrics, _ in all_results.values()])
    print("-" * 47)
    print(f"{'AVERAGE':<25s} {avg_f1:6.4f}")


def _evaluate_baseline(args: argparse.Namespace) -> None:
    """Evaluate the raw embedder features with the GNN's matcher.

    Trains nothing: this is the reference point for the GNN measured with
    the identical post-processing (`matching_rules.greedy_injective`), so the
    two numbers are comparable.  Results go to `results/gnn-baseline/`.
    """
    print("=== Raw embedder baseline (same 1:1 matcher, no training) ===\n")

    taxonomies, alignments = load_oaei_conference(DATA_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = _precompute_cache(taxonomies, args.embedder, device)

    meta = {"seed": args.seed, "config": _run_config(args, "baseline")}
    f1_scores: list[float] = []
    print(f"{'Pair':<25s} {'F1':>6s}  {'P':>6s}  {'R':>6s}")
    print("-" * 47)
    for gt_align in alignments:
        if gt_align.source not in cache or gt_align.target not in cache:
            continue
        src_ids, src_feat, _ = cache[gt_align.source]
        tgt_ids, tgt_feat, _ = cache[gt_align.target]
        sim = F.normalize(src_feat, p=2, dim=-1) @ F.normalize(tgt_feat, p=2, dim=-1).T
        pred = _match(
            sim.cpu().numpy(), src_ids, tgt_ids, gt_align.source, gt_align.target,
        )
        metrics = evaluate_1to1(pred, gt_align)
        _save_results(f"{gt_align.source}↔{gt_align.target}", metrics, pred,
                      approach="gnn-baseline", results_dir=BASELINE_DIR, meta=meta)
        f1_scores.append(metrics.f1)
        print(f"{gt_align.source}↔{gt_align.target:<20s} {metrics.f1:6.4f}  "
              f"{metrics.precision:6.4f}  {metrics.recall:6.4f}")

    print("-" * 47)
    print(f"{'AVERAGE':<25s} {np.mean(f1_scores):6.4f}")
    print(f"\nSaved to {BASELINE_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Siamese GraphSAGE for taxonomy matching.",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train on all 21 pairs and save checkpoint.",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate from saved checkpoint on all 21 pairs.",
    )
    parser.add_argument(
        "--test-pair", type=str, default=None,
        help="Train on 20 pairs, test on this one (e.g., 'cmt↔conference').",
    )
    parser.add_argument(
        "--evaluate-all", action="store_true",
        help="Full leave-one-out CV: 21 folds (train 20, test 1).",
    )
    parser.add_argument(
        "--fold-mode", choices=("pair-loo", "ontology-loo"), default="pair-loo",
        help="pair-loo: hold out one pair (21 folds); "
             "ontology-loo: hold out one ontology and its 6 pairs (7 folds).",
    )
    parser.add_argument(
        "--folds", type=int, default=0,
        help="Run only the first N folds (0 = all), for a cheap smoke run.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed, recorded in every record and checkpoint.",
    )
    parser.add_argument(
        "--embedder", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model (default: MiniLM).",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=384,
        help="Hidden dimension (default: 384, the published configuration).",
    )
    parser.add_argument(
        "--out-dim", type=int, default=384,
        help="Output dimension (default: 384, the published configuration).",
    )
    parser.add_argument(
        "--num-layers", type=int, default=1,
        help="Number of GraphSAGE layers (default: 1).",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="Dropout rate (default: 0.5).",
    )
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Epochs (default: 60).",
    )
    parser.add_argument(
        "--lr", type=float, default=0.003,
        help="Learning rate (default: 0.003).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="Weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--neg-ratio", type=int, default=3,
        help="Negative pairs per positive pair (default: 3).",
    )
    parser.add_argument(
        "--no-synthetic", action="store_true",
        help="Disable synthetic data augmentation.",
    )
    parser.add_argument(
        "--synthetic-seeds", type=int, default=3,
        help="Synthetic seeds per source ontology (default: 3).",
    )
    parser.add_argument(
        "--linear", action="store_true",
        help="Use LinearSiamese (projection only, no graph convolution) as baseline.",
    )

    parser.add_argument(
        "--baseline", action="store_true",
        help="Evaluate raw embedder features with the same 1:1 matcher.",
    )

    args = parser.parse_args()

    _seed_everything(args.seed)

    if args.baseline:
        _evaluate_baseline(args)
    elif args.evaluate_all:
        _evaluate_all_loocv(args)
    elif args.test_pair:
        _test_single_pair(args)
    elif args.train:
        _train_all(args)
    elif args.evaluate:
        _evaluate_from_checkpoint(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
