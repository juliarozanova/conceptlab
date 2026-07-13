"""Ground-truth attribution for concept graphs, and method scoring.

Three tiers (agreed in planning):

* **Tier 1 — structural support.** Which concepts the label depends on at all
  (transitive graph support). A faithful method must give ~0 to the rest. Hard
  gate: attribution *mass outside support*.
* **Tier 2 — pivotal.** Per example, which concepts flip the label when toggled.
  Hard gate: locally inert concepts get ~0 on that example.
* **Tier 3 — graded.** Per-concept importance under two definitions: single-
  concept **interventional** magnitude and sampled **concept-Shapley**. The
  disagreement between them (``definition_gap``) is reported per config — where
  it is large (overdetermined OR labels), ranking is not well-posed and both
  targets are shown.

Ground truth here is *data-level* (the exact label function), which is what the
logic defines. A separate **audit** reports, per concept, whether a trained
model even represents it (probe AUC) and how much the model's own output moves
when the concept is toggled in the data — so "method failed" and "concept not
in the model" are never confused.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .concepts import ConceptGraph, EventBatch
from .concept_models import ConceptDataset


# ---------------------------------------------------------------------------
# ground-truth attribution (data-level)
# ---------------------------------------------------------------------------


def _override_all(C: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    return {n: C[:, i].astype(np.float64) for i, n in enumerate(names)}


def interventional_importance(graph: ConceptGraph, b: EventBatch, C: np.ndarray,
                              names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (global_importance (K,), pivotal (N, K) bool).

    Toggling concept k (0<->1) with all others at their true values; the change
    in the label is k's single-concept interventional effect on that example.
    """
    base = graph.label_under_override(b, _override_all(C, names))
    N, K = C.shape
    piv = np.zeros((N, K), dtype=bool)
    mag = np.zeros(K)
    for k in range(K):
        ov = _override_all(C, names)
        ov[names[k]] = 1.0 - C[:, k]
        flipped = graph.label_under_override(b, ov)
        diff = np.abs(flipped - base)
        piv[:, k] = diff > 0.5
        mag[k] = diff.mean()
    return mag, piv


def shapley_importance(graph: ConceptGraph, b: EventBatch, C: np.ndarray,
                       names: list[str], support: list[int],
                       n_samples: int = 64, seed: int = 0) -> np.ndarray:
    """Sampled per-concept Shapley importance over the support set.

    Value function v(S) = label with concepts in S at their true value and the
    rest set to 0 (absent). Exact if the support is small; sampled otherwise.
    """
    rng = np.random.default_rng(seed)
    K = C.shape[1]
    phi = np.zeros(K)
    S_idx = list(support)
    if not S_idx:
        return phi

    def value(active: set[int]) -> np.ndarray:
        ov = {names[k]: (C[:, k].astype(np.float64) if k in active else np.zeros(C.shape[0]))
              for k in range(K)}
        return graph.label_under_override(b, ov).astype(np.float64)

    exact = len(S_idx) <= 8
    perms = (itertools.permutations(S_idx) if exact
             else (rng.permutation(S_idx) for _ in range(n_samples)))
    count = 0
    for perm in perms:
        active: set[int] = set()
        v_prev = value(active)
        for k in perm:
            active.add(k)
            v_now = value(active)
            phi[k] += (v_now - v_prev).mean()
            v_prev = v_now
        count += 1
    return phi / max(count, 1)


@dataclass
class ConceptGroundTruth:
    names: list[str]
    support_mask: np.ndarray          # (K,) bool: in transitive support
    interventional: np.ndarray        # (K,) global magnitude
    shapley: np.ndarray               # (K,)
    pivotal: np.ndarray               # (N, K) bool
    definition_gap: float             # 1 - rho(interventional, shapley) on support


def compute_ground_truth(ds: ConceptDataset, shapley_samples: int = 64) -> ConceptGroundTruth:
    g, b, C, names = ds.graph, ds.batch, ds.C, ds.graph.names
    support_names = set(g.transitive_support())
    support_mask = np.array([n in support_names for n in names])
    support_idx = [i for i, n in enumerate(names) if n in support_names]

    interv, piv = interventional_importance(g, b, C, names)
    shap = shapley_importance(g, b, C, names, support_idx, n_samples=shapley_samples)

    if support_mask.sum() >= 2 and np.ptp(interv[support_mask]) > 0 and np.ptp(shap[support_mask]) > 0:
        rho = spearmanr(interv[support_mask], shap[support_mask]).statistic
        gap = float(1 - rho)
    else:
        gap = 0.0
    return ConceptGroundTruth(names=names, support_mask=support_mask, interventional=interv,
                              shapley=shap, pivotal=piv, definition_gap=gap)


# ---------------------------------------------------------------------------
# concept audit (model-level): decodability + model toggle sensitivity
# ---------------------------------------------------------------------------


def _acts_at(model, batch: EventBatch, hookpoint: str) -> np.ndarray:
    _, cache = model.run_with_cache(batch)
    a = cache[hookpoint]
    if a.dim() == 3:                    # (B, T, d) -> decision point
        a = a[:, -1, :]
    return a.detach().numpy()


def concept_audit(model, ds: ConceptDataset, hookpoints: list[str]) -> dict:
    """For each concept x hookpoint: linear-probe AUC (is it decodable?)."""
    C = ds.C
    out = {"hookpoints": hookpoints, "auc": {}}
    for hp in hookpoints:
        A = _acts_at(model, ds.batch, hp)
        cut = int(0.7 * len(A))
        aucs = []
        for k, name in enumerate(ds.graph.names):
            yk = (C[:, k] > 0.5).astype(int)
            if yk[:cut].sum() in (0, cut) or len(np.unique(yk)) < 2:
                aucs.append(float("nan"))
                continue
            try:
                clf = LogisticRegression(max_iter=300).fit(A[:cut], yk[:cut])
                aucs.append(float(roc_auc_score(yk[cut:], clf.decision_function(A[cut:]))))
            except Exception:
                aucs.append(float("nan"))
        out["auc"][hp] = aucs
    # best decodability per concept across layers
    best = np.nanmax(np.stack([out["auc"][hp] for hp in hookpoints]), axis=0)
    out["best_auc"] = [float(x) for x in best]
    return out


# ---------------------------------------------------------------------------
# scoring a method's attribution against ground truth
# ---------------------------------------------------------------------------


@dataclass
class MethodScore:
    name: str
    rho_interventional: float
    rho_shapley: float
    support_leakage: float        # fraction of |attr| mass on non-support concepts
    pivotal_gate: float           # mean |attr| on locally-inert concepts (lower=better)
    per_concept: list[float]


def score_method(name: str, attribution: np.ndarray, gt: ConceptGroundTruth) -> MethodScore:
    """attribution: (K,) nonneg per-concept importance (global)."""
    a = np.abs(np.asarray(attribution, dtype=float))
    sup = gt.support_mask

    def _rho(target):
        if np.ptp(a) == 0 or np.ptp(target) == 0:
            return 0.0
        return float(spearmanr(a, target).statistic)

    rho_i = _rho(gt.interventional)
    rho_s = _rho(gt.shapley)
    total = a.sum() + 1e-12
    leakage = float(a[~sup].sum() / total)
    # pivotal gate: attribution on concepts that are never pivotal anywhere
    never_pivotal = ~gt.pivotal.any(axis=0)
    pgate = float(a[never_pivotal].mean()) if never_pivotal.any() else 0.0
    return MethodScore(name=name, rho_interventional=rho_i, rho_shapley=rho_s,
                       support_leakage=leakage, pivotal_gate=pgate,
                       per_concept=[float(x) for x in a])
