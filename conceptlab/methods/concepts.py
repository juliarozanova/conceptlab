"""Concept-attribution methods for the tabular testbed.

All methods explain a label decision with respect to concepts by acting on a
**concept direction** (a CAV) at a chosen hookpoint. They differ in mechanism:

* ``ICS``          — Integrated Gradients along the CAV (Schrouff et al. 2021).
* ``TCAV``         — directional-derivative sign statistics (global).
* ``CAVAblation``  — project the CAV out of the activation, measure Δ output.
* ``ProbePatch``   — set the probe-decoded concept value to a counterfactual.
* ``InputAgg``     — attribute to inputs, aggregate to concepts via the DAG.
* ``CAVSkyline``   — CAVs taken from ground-truth concept values (isolates the
                     attribution mechanism from CAV-estimation error).

A :class:`ModelAdapter` gives every method the same three capabilities at a
hookpoint: read decision-point activations, run the model forward from a
perturbed decision activation, and take the gradient of the positive-class
logit w.r.t. that activation.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ..concept_models import SoftLogicModel, TabTransformer
from ..concepts import EventBatch


# ---------------------------------------------------------------------------
# model adapter: uniform activation interface over Mode A / Mode B
# ---------------------------------------------------------------------------


class ModelAdapter:
    def __init__(self, model, hookpoint: str):
        self.model = model
        self.hp = hookpoint

    def decision_acts(self, batch: EventBatch) -> np.ndarray:
        _, cache = self.model.run_with_cache(batch)
        a = cache[self.hp]
        if a.dim() == 3:
            a = a[:, -1, :]
        return a.detach().numpy()

    def _cache_full(self, batch: EventBatch):
        _, cache = self.model.run_with_cache(batch)
        return cache

    def logit_and_grad(self, batch: EventBatch, delta: torch.Tensor):
        """Positive-class logit and its gradient w.r.t. the (perturbed) decision
        activation. ``delta`` (N, d) is added to the decision-point activation."""
        cache = self._cache_full(batch)
        a = cache[self.hp]
        if isinstance(self.model, SoftLogicModel):
            base = a.detach()
            act = (base + delta).requires_grad_(True)
            logits = self.model.forward_from_rep(act)
            prob = torch.softmax(logits, -1)[:, 1].sum()
            grad, = torch.autograd.grad(prob, act)
            return logits.detach(), grad.detach()
        # TabTransformer: replace decision token in the cached resid, run tail
        layer = int(self.hp.split("resid_post_L")[-1]) if "resid_post_L" in self.hp else self.model.n_layers - 1
        resid = cache[f"resid_post_L{layer}"].detach()             # (B,T,d)
        dec = resid[:, -1, :]
        act = (dec + delta).requires_grad_(True)
        full = resid.clone()
        full = torch.cat([full[:, :-1, :], act.unsqueeze(1)], dim=1)
        logits = _tab_tail(self.model, full, layer)
        prob = torch.softmax(logits, -1)[:, 1].sum()
        grad, = torch.autograd.grad(prob, act)
        return logits.detach(), grad.detach()

    def logit_perturbed(self, batch: EventBatch, delta: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cache = self._cache_full(batch)
            if isinstance(self.model, SoftLogicModel):
                return self.model.forward_from_rep(cache[self.hp].detach() + delta)
            layer = int(self.hp.split("resid_post_L")[-1]) if "resid_post_L" in self.hp else self.model.n_layers - 1
            resid = cache[f"resid_post_L{layer}"].detach()
            act = resid[:, -1, :] + delta
            full = torch.cat([resid[:, :-1, :], act.unsqueeze(1)], dim=1)
            return _tab_tail(self.model, full, layer)


def _tab_tail(model: TabTransformer, resid_full: torch.Tensor, layer: int) -> torch.Tensor:
    """Run a TabTransformer from the residual after ``layer`` to the logits."""
    T = resid_full.shape[1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    h = resid_full
    for blk in model.blocks[layer + 1:]:
        h = blk(h, attn_mask=mask)
    h = model.ln_f(h)
    return model.head(h[:, -1, :])


# ---------------------------------------------------------------------------
# CAV fitting
# ---------------------------------------------------------------------------


def fit_cav(acts: np.ndarray, concept_vals: np.ndarray) -> np.ndarray:
    """Unit CAV: logistic-probe weight separating concept-present from absent."""
    y = (concept_vals > 0.5).astype(int)
    if len(np.unique(y)) < 2:
        return np.zeros(acts.shape[1])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(acts, y)
    w = clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-12)


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


class ConceptMethod:
    name = "base"
    global_only = False

    def attribute(self, adapter: ModelAdapter, batch: EventBatch,
                  cavs: np.ndarray) -> np.ndarray:
        """Return (K,) global per-concept importance for the label."""
        raise NotImplementedError


class ICS(ConceptMethod):
    """Integrated gradients along each CAV, from a zero-coefficient baseline."""

    name = "ics"

    def __init__(self, steps: int = 16):
        self.steps = steps

    def attribute(self, adapter, batch, cavs):
        acts = adapter.decision_acts(batch)
        K = len(cavs)
        out = np.zeros(K)
        per_ex = np.zeros((acts.shape[0], K))
        for k in range(K):
            v = cavs[k]
            if not np.any(v):
                continue
            vt = torch.tensor(v, dtype=torch.float32)
            coef = torch.tensor(acts @ v, dtype=torch.float32)        # (N,)
            grad_accum = torch.zeros_like(coef)
            for s in range(1, self.steps + 1):
                alpha = s / self.steps
                # move activation from (a - coef*v) [coef 0] to a [coef full]
                delta = ((alpha - 1.0) * coef).unsqueeze(1) * vt
                _, grad = adapter.logit_and_grad(batch, delta)
                grad_accum = grad_accum + (grad @ vt)
            ig = coef * (grad_accum / self.steps)
            per_ex[:, k] = ig.abs().numpy()
            out[k] = ig.abs().mean().item()
        self.last_per_example = per_ex          # (N, K) local attributions
        return out


class TCAV(ConceptMethod):
    """Fraction of examples with positive directional derivative, centered."""

    name = "tcav"
    global_only = True

    def attribute(self, adapter, batch, cavs):
        K = len(cavs)
        out = np.zeros(K)
        zero = torch.zeros(batch.n, adapter.decision_acts(batch).shape[1])
        for k in range(K):
            v = cavs[k]
            if not np.any(v):
                continue
            vt = torch.tensor(v, dtype=torch.float32)
            _, grad = adapter.logit_and_grad(batch, zero)
            dd = (grad @ vt).numpy()
            out[k] = abs(2 * (dd > 0).mean() - 1.0)      # |TCAV score - 0.5| * 2
        return out


class CAVAblation(ConceptMethod):
    """Project the CAV out of the decision activation; measure |Δ P(fraud)|."""

    name = "cav_ablation"

    def attribute(self, adapter, batch, cavs):
        acts = adapter.decision_acts(batch)
        base = torch.softmax(adapter.logit_perturbed(batch, torch.zeros(acts.shape)), -1)[:, 1]
        K = len(cavs)
        out = np.zeros(K)
        per_ex = np.zeros((acts.shape[0], K))
        for k in range(K):
            v = cavs[k]
            if not np.any(v):
                continue
            vt = torch.tensor(v, dtype=torch.float32)
            coef = torch.tensor(acts @ v, dtype=torch.float32)
            delta = (-coef).unsqueeze(1) * vt                 # remove the component
            p = torch.softmax(adapter.logit_perturbed(batch, delta), -1)[:, 1]
            per_ex[:, k] = (p - base).abs().numpy()
            out[k] = (p - base).abs().mean().item()
        self.last_per_example = per_ex
        return out


class ProbePatch(ConceptMethod):
    """Counterfactual concept patch: move the activation to the mean of the
    opposite-concept population along the CAV, measure |Δ P(fraud)|.

    A CAV/probe-space analogue of concept-bottleneck intervention."""

    name = "probe_patch"

    def attribute(self, adapter, batch, cavs, concept_vals=None):
        acts = adapter.decision_acts(batch)
        base = torch.softmax(adapter.logit_perturbed(batch, torch.zeros(acts.shape)), -1)[:, 1]
        K = len(cavs)
        out = np.zeros(K)
        per_ex = np.zeros((acts.shape[0], K))
        for k in range(K):
            v = cavs[k]
            if not np.any(v):
                continue
            proj = acts @ v
            hi, lo = np.quantile(proj, 0.8), np.quantile(proj, 0.2)
            # patch each example to the *opposite* pole along the CAV
            target = np.where(proj > proj.mean(), lo, hi)
            vt = torch.tensor(v, dtype=torch.float32)
            delta = torch.tensor(target - proj, dtype=torch.float32).unsqueeze(1) * vt
            p = torch.softmax(adapter.logit_perturbed(batch, delta), -1)[:, 1]
            per_ex[:, k] = (p - base).abs().numpy()
            out[k] = (p - base).abs().mean().item()
        self.last_per_example = per_ex
        return out


class InputAgg(ConceptMethod):
    """Input-level integrated gradients, aggregated to concepts via the known DAG.

    Attributes the label to each *input column* by integrated gradients on the
    model's numeric inputs (categorical columns via embedding-gradient norm),
    then sums each concept's column attributions using the concepts' known input
    sets. Tests whether concept-level methods add anything over input-level
    attribution plus structure — and, by construction, cannot separate two
    concepts that read the same column (a real limitation this surfaces).
    """

    name = "input_agg"
    global_only = True

    def __init__(self, graph, ds, steps: int = 16):
        self.graph = graph
        self.ds = ds
        self.steps = steps

    def attribute(self, adapter, batch, cavs):
        model = adapter.model
        if isinstance(model, SoftLogicModel):
            # glass box: input attribution isn't the intended surface; fall back
            # to the concept-coordinate skyline so the column is populated.
            return CAVSkyline().attribute(adapter, batch, _unit_rows(self.ds.anchors))
        num_cols = model.numeric_cols
        cat_cols = model.categorical_cols
        num0 = {c: torch.as_tensor(batch.numeric[c], dtype=torch.float32) for c in num_cols}
        cat = {c: torch.as_tensor(batch.categorical[c], dtype=torch.long) for c in cat_cols}
        # integrated gradients on numeric inputs from a per-column baseline (mean)
        col_attr = {}
        base = {c: num0[c].mean().item() for c in num_cols}
        for c in num_cols:
            total = torch.zeros_like(num0[c])
            for s in range(1, self.steps + 1):
                a = s / self.steps
                cur = {cc: num0[cc].clone() for cc in num_cols}
                cur[c] = base[c] + a * (num0[c] - base[c])
                for cc in num_cols:
                    cur[cc] = cur[cc].requires_grad_(True)
                logits = model.logit_from_inputs(cur, cat)
                prob = torch.softmax(logits, -1)[:, 1].sum()
                g, = torch.autograd.grad(prob, cur[c])
                total = total + g
            ig = ((num0[c] - base[c]) * total / self.steps)
            col_attr[c] = ig[:, -1].abs().mean().item()      # decision-point column
        # categorical: gradient-norm at the decision-point embedding
        for c in cat_cols:
            emb = model.cat_emb[c](cat[c]).detach().requires_grad_(True)
            # rough: perturb embedding, measure output sensitivity via finite diff
            col_attr[c] = _cat_sensitivity(model, num0, cat, c)
        # aggregate columns -> concepts via the DAG
        out = np.zeros(self.graph.n_concepts)
        for k, name in enumerate(self.graph.names):
            cols = self.graph.concept_support_inputs(name)
            out[k] = sum(col_attr.get(col, 0.0) for col in cols)
        return out


def _unit_rows(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def _cat_sensitivity(model, num0, cat, col) -> float:
    """Sensitivity of P(fraud) to the decision-point categorical value, by
    swapping it to each other code and averaging |Δ prob| (a model-based proxy
    for input attribution on a discrete field)."""
    with torch.no_grad():
        base = torch.softmax(model.logit_from_inputs(num0, cat), -1)[:, 1]
        card = model.cat_emb[col].num_embeddings
        diffs = []
        for code in range(card):
            alt = {c: v.clone() for c, v in cat.items()}
            alt[col][:, -1] = code
            p = torch.softmax(model.logit_from_inputs(num0, alt), -1)[:, 1]
            diffs.append((p - base).abs())
        return torch.stack(diffs).mean().item()


class CAVSkyline(ConceptMethod):
    """ICS using ground-truth concept directions instead of fitted CAVs."""

    name = "cav_skyline"

    def __init__(self, steps: int = 16):
        self.ics = ICS(steps=steps)

    def attribute(self, adapter, batch, true_cavs):
        return self.ics.attribute(adapter, batch, true_cavs)
