"""Train and evaluate Siamese GraphSAGE for taxonomy matching.

Uses leave-one-out cross-validation on the 21 OAEI Conference track
pairs — standard protocol in ontology matching (GraphMatcher, LogMap).

Usage:
    # Train + evaluate full leave-one-out (21 folds, ~40 min).
    uv run python -m src.runners.gnn --evaluate-all

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
from ..matchers.gnn import LinearSiamese, SiameseGraphSAGE
from ..metrics import MatchMetrics, evaluate_1to1
from ..taxonomy import Alignment, TaxonMatch, Taxonomy


RESULTS_DIR = Path("results/gnn")
CHECKPOINT_PATH = RESULTS_DIR / "model.pt"
BASELINE_DIR = Path("results/gnn-baseline")
DATA_DIR = Path("data/raw/oaei")


# ---------------------------------------------------------------------------
# Training helpers.
# ---------------------------------------------------------------------------


def _build_train_data(
    taxonomies: dict[str, Taxonomy],
    all_alignments: list[Alignment],
    held_out_pairs: set[str] | None = None,
    num_synthetic_seeds: int = 3,
) -> list[tuple[Taxonomy, Taxonomy, Alignment]]:
    """Build training pairs from OAEI data + synthetic augmentation.

    Synthetic pairs (variations of real ontologies) are added to increase
    the number of training examples.  Only source ontologies that appear
    in real training pairs are used for synthetic generation.

    Args:
        taxonomies: All loaded taxonomies keyed by name.
        all_alignments: All 21 reference alignments.
        held_out_pairs: Set of pair keys to exclude from training.
        num_synthetic_seeds: Synthetic variations per source ontology.

    Returns:
        List of (source_tax, target_tax, ground_truth_alignment) triples.
    """
    from ..data_loader import generate_synthetic_variations

    if held_out_pairs is None:
        held_out_pairs = set()

    train_pairs: list[tuple[Taxonomy, Taxonomy, Alignment]] = []
    source_names_used: set[str] = set()

    for gt in all_alignments:
        pair_key = f"{gt.source}↔{gt.target}"
        if pair_key in held_out_pairs:
            continue
        src = taxonomies.get(gt.source)
        tgt = taxonomies.get(gt.target)
        if src is None or tgt is None:
            continue
        train_pairs.append((src, tgt, gt))
        source_names_used.add(gt.source)

    # Add synthetic variations for ontologies appearing in real training pairs.
    for src_name in sorted(source_names_used):
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


# ---------------------------------------------------------------------------
# Evaluation helpers.
# ---------------------------------------------------------------------------


def _greedy_match(
    sim: np.ndarray,
    src_ids: list[str],
    tgt_ids: list[str],
) -> list[TaxonMatch]:
    """Greedy 1-to-1 matching by descending similarity, no threshold.

    Every source node gets at most one target and vice versa, so the number
    of predicted matches is min(|src|, |tgt|).
    """
    used_tgt: set[int] = set()
    matches: list[TaxonMatch] = []
    candidates = sorted(
        (
            (float(sim[i, j]), i, j)
            for i in range(len(src_ids))
            for j in range(len(tgt_ids))
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    for score, i, j in candidates:
        if j in used_tgt:
            continue
        matches.append(
            TaxonMatch(source_id=src_ids[i], target_id=tgt_ids[j], confidence=score)
        )
        used_tgt.add(j)
    return matches


def _evaluate_model(
    model: SiameseGraphSAGE,
    taxonomies: dict[str, Taxonomy],
    test_alignments: list[Alignment],
    cache: dict[str, tuple[list[str], torch.Tensor, torch.Tensor]],
) -> dict[str, MatchMetrics]:
    """Evaluate on a list of test alignments."""
    model.eval()
    results: dict[str, MatchMetrics] = {}

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
        matches = _greedy_match(sim_np, src_ids, tgt_ids)

        pred = Alignment(
            source=gt_align.source, target=gt_align.target, matches=matches,
        )
        metrics = evaluate_1to1(pred, gt_align)
        results[f"{gt_align.source}↔{gt_align.target}"] = metrics

    return results


def _save_results(
    pair_key: str,
    metrics: MatchMetrics,
    approach: str = "gnn",
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Cache per-pair evaluation results as JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)
    # Canonical (sorted) pair key for consistent cross-approach merging.
    parts = sorted(pair_key.split("↔"))
    pair_key_canonical = "↔".join(parts)
    fname = pair_key_canonical.replace("↔", "_").lower()
    fpath = results_dir / f"{fname}.json"
    data = {
        "pair": pair_key_canonical,
        "approach": approach,
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "matches": metrics.true_positives,
        "total_predicted": metrics.total_predicted,
        "total_gt": metrics.total_ground_truth,
    }
    fpath.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Core training procedure.
# ---------------------------------------------------------------------------


def _train_model(
    taxonomies: dict[str, Taxonomy],
    all_alignments: list[Alignment],
    embedder_model: str,
    held_out_pairs: set[str] | None,
    args: argparse.Namespace,
) -> SiameseGraphSAGE:
    """Train a SiameseGraphSAGE from scratch.

    Args:
        taxonomies: All OAEI taxonomies.
        all_alignments: All 21 reference alignments.
        embedder_model: Sentence-transformers model name.
        held_out_pairs: Pairs to exclude from training (for leave-one-out).
        args: CLI arguments.

    Returns:
        Trained model on the specified device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Pre-compute features and adjacency once for all ontologies
    # (including synthetic variants that will appear in train_pairs).
    cache = _precompute_cache(taxonomies, embedder_model, device)

    # Build training pairs first, then extend cache with synthetic variants.
    train_pairs = _build_train_data(
        taxonomies, all_alignments, held_out_pairs,
        num_synthetic_seeds=0 if args.no_synthetic else args.synthetic_seeds,
    )

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

    best_loss = float("inf")
    for epoch in range(args.epochs):
        random.shuffle(train_pairs)
        avg_loss = _train_epoch(
            model, train_pairs, cache, device, optimizer,
            neg_ratio=args.neg_ratio,
        )
        print(f"Epoch {epoch + 1}/{args.epochs}, loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "in_dim": in_dim,
                "hidden_dim": args.hidden_dim,
                "out_dim": args.out_dim,
                "num_layers": args.num_layers,
                "linear": args.linear,
                "model_state": model.state_dict(),
            }, CHECKPOINT_PATH)

    print(f"Best loss: {best_loss:.6f}\n")
    return model


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _train_all(args: argparse.Namespace) -> None:
    """Train on all 21 pairs and save checkpoint."""
    print("=== Training Siamese GraphSAGE (all 21 pairs) ===\n")
    taxonomies, alignments = load_oaei_conference(DATA_DIR)
    _train_model(taxonomies, alignments, args.embedder, None, args)
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

    taxonomies, alignments = load_oaei_conference(DATA_DIR)
    cache = _precompute_cache(taxonomies, args.embedder, device)

    results = _evaluate_model(model, taxonomies, alignments, cache)

    print()
    print(f"{'Pair':<25s} {'F1':>6s}  {'P':>6s}  {'R':>6s}")
    print("-" * 47)
    for pair_key in sorted(results.keys()):
        m = results[pair_key]
        _save_results(pair_key, m)
        print(f"{pair_key:<25s} {m.f1:6.4f}  {m.precision:6.4f}  {m.recall:6.4f}")

    avg_f1 = np.mean([m.f1 for m in results.values()])
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
    model = _train_model(taxonomies, alignments, args.embedder, held_out, args)
    cache = _precompute_cache(taxonomies, args.embedder, "cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate only on the test pair.
    test_alignments = [
        a for a in alignments
        if f"{a.source}↔{a.target}" in held_out
        or f"{a.target}↔{a.source}" in held_out
    ]

    results = _evaluate_model(model, taxonomies, test_alignments, cache)

    print()
    for pair_key, m in sorted(results.items()):
        _save_results(pair_key, m)
        print(f"{pair_key}: F1={m.f1:.4f}, P={m.precision:.4f}, R={m.recall:.4f}")


def _evaluate_all_loocv(args: argparse.Namespace) -> None:
    """Full leave-one-out: train on 20 pairs, test on 1, repeat 21 times."""
    print("=== Leave-one-out cross-validation (21 folds) ===\n")

    taxonomies, alignments = load_oaei_conference(DATA_DIR)
    all_pair_keys = [f"{a.source}↔{a.target}" for a in alignments]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-compute once for all folds.
    cache = _precompute_cache(taxonomies, args.embedder, device)

    all_results: dict[str, MatchMetrics] = {}

    for i, held_pair in enumerate(all_pair_keys):
        print(f"{'─' * 60}")
        print(f"Fold {i + 1}/{len(all_pair_keys)}: testing {held_pair}")
        print(f"{'─' * 60}")

        model = _train_model(
            taxonomies, alignments, args.embedder, {held_pair}, args,
        )

        test_alignments = [
            a for a in alignments
            if f"{a.source}↔{a.target}" == held_pair
        ]
        results = _evaluate_model(model, taxonomies, test_alignments, cache)
        all_results.update(results)

    # Final table.
    print()
    print("=" * 47)
    print("  Full leave-one-out results")
    print("=" * 47)
    print(f"{'Pair':<25s} {'F1':>6s}  {'P':>6s}  {'R':>6s}")
    print("-" * 47)
    for pair_key in sorted(all_results.keys()):
        m = all_results[pair_key]
        _save_results(pair_key, m)
        print(f"{pair_key:<25s} {m.f1:6.4f}  {m.precision:6.4f}  {m.recall:6.4f}")

    avg_f1 = np.mean([m.f1 for m in all_results.values()])
    print("-" * 47)
    print(f"{'AVERAGE':<25s} {avg_f1:6.4f}")


def _evaluate_baseline(args: argparse.Namespace) -> None:
    """Evaluate the raw embedder features with the GNN's greedy matcher.

    Trains nothing: this is the reference point for the GNN measured with
    the identical post-processing, so the two numbers are comparable.
    Results go to `results/gnn-baseline/`.
    """
    print("=== Raw embedder baseline (same greedy matcher, no training) ===\n")

    taxonomies, alignments = load_oaei_conference(DATA_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = _precompute_cache(taxonomies, args.embedder, device)

    f1_scores: list[float] = []
    print(f"{'Pair':<25s} {'F1':>6s}  {'P':>6s}  {'R':>6s}")
    print("-" * 47)
    for gt_align in alignments:
        if gt_align.source not in cache or gt_align.target not in cache:
            continue
        src_ids, src_feat, _ = cache[gt_align.source]
        tgt_ids, tgt_feat, _ = cache[gt_align.target]
        sim = F.normalize(src_feat, p=2, dim=-1) @ F.normalize(tgt_feat, p=2, dim=-1).T
        matches = _greedy_match(sim.cpu().numpy(), src_ids, tgt_ids)
        pred = Alignment(
            source=gt_align.source, target=gt_align.target, matches=matches,
        )
        metrics = evaluate_1to1(pred, gt_align)
        _save_results(f"{gt_align.source}↔{gt_align.target}", metrics,
                      approach="gnn-baseline", results_dir=BASELINE_DIR)
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
        "--embedder", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model (default: MiniLM).",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Hidden dimension (default: 128).",
    )
    parser.add_argument(
        "--out-dim", type=int, default=64,
        help="Output dimension (default: 64).",
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
        help="Evaluate raw embedder features with the same greedy matcher.",
    )

    args = parser.parse_args()

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
