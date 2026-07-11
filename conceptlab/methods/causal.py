"""Causal importance by intervening on a concept direction.

* ``DirectionAblation`` — remove the component along ``d`` (project it out) and
  measure how much the model's positive-class probability changes.
* ``ActivationPatching`` — replace the component along ``d`` with its dataset
  mean, measuring the change.

These are doubly useful: methods under test *and* a strong reference for
importance, since intervening through the trained model is the closest
in-activation analogue of the generator-level causal ground truth.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import InterpMethod, MethodContext, positive_prob


class _InterventionScorer(InterpMethod):
    can_score = True

    def __init__(self, batch: int = 1024):
        self.batch = batch

    def fit(self, ctx: MethodContext) -> "_InterventionScorer":
        self.ctx = ctx
        # Precompute mean coefficient per (batch of) directions lazily per call.
        return self

    def _intervene(self, X: torch.Tensor, d: torch.Tensor, mean_coef: float) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def score_directions(self, directions: np.ndarray) -> np.ndarray:
        ctx = self.ctx
        model = ctx.model
        X = ctx.X_sample
        base = positive_prob(model, X)
        scores = np.zeros(len(directions))
        for j, dnp in enumerate(directions):
            d = torch.as_tensor(dnp, dtype=torch.float32)
            mean_coef = (X * d).sum(dim=-1).mean().item()
            diffs = []
            for i in range(0, len(X), self.batch):
                xb = X[i : i + self.batch]
                xint = self._intervene(xb, d, mean_coef)
                p = positive_prob(model, xint)
                diffs.append((p - base[i : i + self.batch]).abs())
            scores[j] = torch.cat(diffs).mean().item()
        return scores


class DirectionAblation(_InterventionScorer):
    name = "ablation"

    def _intervene(self, X, d, mean_coef):
        coef = (X * d).sum(dim=-1, keepdim=True)
        return X - coef * d


class ActivationPatching(_InterventionScorer):
    name = "patching"

    def _intervene(self, X, d, mean_coef):
        coef = (X * d).sum(dim=-1, keepdim=True)
        return X - coef * d + mean_coef * d
