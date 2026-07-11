"""Scoring method outputs against ground truth.

Two families of metric, matching the two halves of the question:

* **Recovery** — did the method find the planted concepts? Hungarian-matched
  cosine similarity, plus coverage/redundancy (which separates the SAE capture
  regimes: shattering, compact capture, dilution) and, for circular concepts,
  the fraction of the ring covered.
* **Importance faithfulness** — did the method rank concepts by their true
  causal importance for the label? Spearman correlation against generator-level
  ground truth, obtained by intervening in the data generator.

Plus **sufficiency**: can a simple probe on the discovered concepts reproduce the
label — i.e. did the method find everything the label needs?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from .datagen import World
from .methods.base import positive_prob


# ---------------------------------------------------------------------------
# ground-truth importance
# ---------------------------------------------------------------------------


def model_importance(model, world: World, n: int = 3000, seed: int = 777) -> np.ndarray:
    """Model-level causal importance per concept.

    Toggle each concept in the generator, keep the noise fixed, and measure the
    change in the model's positive-class probability. This is the closest
    ground-truth analogue of what ablation/patching approximate.
    Single-vector datasets only.
    """
    rng = np.random.default_rng(seed)
    ns = world.sample_latents(n, rng)
    eps = rng.normal(0, world.noise, size=(n, world.dim)) if world.noise > 0 else 0.0
    X = world.embed(ns, rng, add_noise=False) + eps
    Xt = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        base = positive_prob(model, Xt)
    imp = np.zeros(world.n_concepts)
    for i in range(world.n_concepts):
        ns2 = world.toggle(ns, i, rng)
        X2 = world.embed(ns2, rng, add_noise=False) + eps
        with torch.no_grad():
            p2 = positive_prob(model, torch.as_tensor(X2, dtype=torch.float32))
        imp[i] = (p2 - base).abs().mean().item()
    return imp


def sequence_data_importance(world: World, seq_len: int, n: int = 3000,
                             seed: int = 778) -> np.ndarray:
    """Data-level importance for a sequence label: toggle a concept in every
    token and measure the change in the aggregated sequence label."""
    rng = np.random.default_rng(seed)
    per_token = {c.name: np.zeros((n, seq_len), dtype=np.int64) for c in world.concepts}
    for t in range(seq_len):
        ns_t = world.sample_latents(n, rng)
        for name, v in ns_t.items():
            per_token[name][:, t] = v
    base = world.seq_formula(per_token)
    imp = np.zeros(world.n_concepts)
    for i, c in enumerate(world.concepts):
        toggled = {k: v.copy() for k, v in per_token.items()}
        for t in range(seq_len):
            col = {c2.name: toggled[c2.name][:, t] for c2 in world.concepts}
            tog = world.toggle(col, i, rng)
            toggled[c.name][:, t] = tog[c.name]
        y2 = world.seq_formula(toggled)
        imp[i] = np.mean(np.abs(y2 - base))
    return imp


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


@dataclass
class RecoveryResult:
    mean_matched_cosine: float
    per_true_best_cosine: list[float]
    coverage: float                       # fraction of true dirs with a match > thr
    redundancy: float                     # avg # discovered atoms per true dir (> thr)
    regime: str                           # shattering | compact | dilution | partial
    n_discovered: int
    n_true: int
    ring_coverage: float | None = None    # fraction of circle covered, if circular


def _abs_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.abs(A @ B.T)


def recovery(discovered: np.ndarray, world: World, thr: float = 0.7) -> RecoveryResult:
    true_dirs = world.anchor_unit()                       # (M, D)
    cos = _abs_cos(discovered, true_dirs)                  # (n_disc, M)
    per_true_best = cos.max(axis=0)                        # best discovered per true dir

    # Hungarian assignment discovered<->true on the shared min dimension.
    C = 1.0 - cos
    row, col = linear_sum_assignment(C)
    matched = cos[row, col]
    mean_matched = float(matched.mean()) if len(matched) else 0.0

    covered = per_true_best > thr
    coverage = float(covered.mean())
    # redundancy: how many discovered atoms align with each covered true dir
    counts = (cos > thr).sum(axis=0)                       # per true dir
    redundancy = float(counts[covered].mean()) if covered.any() else 0.0
    regime = _regime(coverage, redundancy, discovered.shape[0], true_dirs.shape[0])

    ring_cov = _ring_coverage(discovered, world, thr)

    return RecoveryResult(
        mean_matched_cosine=mean_matched,
        per_true_best_cosine=[float(x) for x in per_true_best],
        coverage=coverage, redundancy=redundancy, regime=regime,
        n_discovered=int(discovered.shape[0]), n_true=int(true_dirs.shape[0]),
        ring_coverage=ring_cov,
    )


def _regime(coverage: float, redundancy: float, n_disc: int, n_true: int) -> str:
    if coverage < 0.6:
        return "partial"
    if redundancy >= 2.5:
        return "shattering" if n_disc > 3 * n_true else "dilution"
    return "compact"


def _ring_coverage(discovered: np.ndarray, world: World, thr: float) -> float | None:
    """For a circular concept, what fraction of the ring do discovered atoms hit?

    We project discovered directions onto the ring's 2D plane and bin their
    angles; coverage is the fraction of angular bins that a discovered atom
    points into. Dilution shows up as many atoms scattered around the ring.
    """
    circ = [c for c in world.concepts if c.kind == "circular"]
    if not circ:
        return None
    c = circ[0]
    u = world.anchors[c.dir_index[0]]
    v = world.anchors[c.dir_index[1]]
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    D = discovered / (np.linalg.norm(discovered, axis=1, keepdims=True) + 1e-12)
    au = D @ u
    av = D @ v
    inplane = np.sqrt(au ** 2 + av ** 2)                   # how much lies in the ring plane
    on_ring = inplane > thr
    if not on_ring.any():
        return 0.0
    angles = np.arctan2(av[on_ring], au[on_ring])
    bins = np.floor((angles + np.pi) / (2 * np.pi) * c.n_positions).astype(int) % c.n_positions
    return float(len(np.unique(bins)) / c.n_positions)


# ---------------------------------------------------------------------------
# importance faithfulness
# ---------------------------------------------------------------------------


def aggregate_to_concepts(dir_scores: np.ndarray, world: World) -> np.ndarray:
    """Sum per-anchor-row scores into per-concept scores."""
    ids = world.anchor_concept_ids()
    out = np.zeros(world.n_concepts)
    for row, ci in enumerate(ids):
        out[ci] += dir_scores[row]
    return out


@dataclass
class FaithfulnessResult:
    spearman: float
    detects_interaction: bool             # nonzero score on XOR-coupled concepts
    per_concept_score: list[float]
    gt_importance: list[float]


def faithfulness(concept_scores: np.ndarray, gt_importance: np.ndarray,
                 interaction_concepts: list[int] | None = None) -> FaithfulnessResult:
    if np.allclose(concept_scores, concept_scores[0]) or np.allclose(gt_importance, gt_importance[0]):
        rho = 0.0
    else:
        rho = float(spearmanr(concept_scores, gt_importance).statistic)
    detects = False
    if interaction_concepts:
        s = concept_scores[interaction_concepts]
        thr = 0.1 * (concept_scores.max() + 1e-12)
        detects = bool(np.all(s > thr))
    return FaithfulnessResult(
        spearman=rho, detects_interaction=detects,
        per_concept_score=[float(x) for x in concept_scores],
        gt_importance=[float(x) for x in gt_importance],
    )


# ---------------------------------------------------------------------------
# sufficiency
# ---------------------------------------------------------------------------


def sufficiency(discovered: np.ndarray, E: np.ndarray, y: np.ndarray) -> float:
    """Train a logistic-regression probe from discovered-concept coefficients to
    the label; return held-out accuracy. High = the method found what the label
    needs. Interactions (XOR) are added as pairwise products so a *sufficient*
    linear set can still express them."""
    proj = E @ discovered.T                                # (N, m)
    n = len(proj)
    if n < 20:
        return float("nan")
    cut = int(0.7 * n)
    Xtr, Xte = proj[:cut], proj[cut:]
    ytr, yte = y[:cut], y[cut:]
    # allow interactions: augment with pairwise products of the top components
    def aug(P):
        k = min(P.shape[1], 8)
        prods = [P[:, i] * P[:, j] for i in range(k) for j in range(i, k)]
        return np.concatenate([P] + [np.stack(prods, axis=1)], axis=1) if prods else P
    Xtr, Xte = aug(Xtr), aug(Xte)
    if len(np.unique(ytr)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=500, C=1.0).fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean())


# ---------------------------------------------------------------------------
# top-level per-method evaluation
# ---------------------------------------------------------------------------


@dataclass
class MethodEval:
    name: str
    recovery: RecoveryResult | None = None
    faithfulness: FaithfulnessResult | None = None
    sufficiency: float | None = None
    importance_per_concept: list[float] = field(default_factory=list)
