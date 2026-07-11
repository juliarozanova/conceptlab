"""Gradient attribution: integrated gradients along a concept direction.

For a unit direction ``d``, the coefficient of an input embedding along ``d`` is
``c = x . d``. Integrated Gradients attributes the model's positive-class
probability to the presence of that component by integrating the gradient as the
component is scaled from 0 (removed) to 1 (present):

    IG(d) = mean_x  | c(x) * integral_0^1  d f / d c  at  x_alpha  d_alpha |

with ``x_alpha = x - (1 - alpha) * c(x) * d``. This is the natural home for the
"integrated conceptual sensitivity" idea — swap the path or the target scalar to
prototype variants against known ground-truth importance.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import InterpMethod, MethodContext, positive_prob


class IntegratedGradients(InterpMethod):
    name = "integrated_gradients"
    can_score = True

    def __init__(self, steps: int = 24, batch: int = 512):
        self.steps = steps
        self.batch = batch

    def fit(self, ctx: MethodContext) -> "IntegratedGradients":
        self.ctx = ctx
        return self

    def _last_dim_component(self, X: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return (X * d).sum(dim=-1, keepdim=True)

    def score_directions(self, directions: np.ndarray) -> np.ndarray:
        ctx = self.ctx
        model = ctx.model
        X = ctx.X_sample
        scores = np.zeros(len(directions))
        for j, dnp in enumerate(directions):
            d = torch.as_tensor(dnp, dtype=torch.float32)
            total = 0.0
            count = 0
            for i in range(0, len(X), self.batch):
                xb = X[i : i + self.batch]
                c = self._last_dim_component(xb, d)          # (B,...,1)
                grad_accum = torch.zeros_like(c)
                for s in range(1, self.steps + 1):
                    alpha = s / self.steps
                    x_alpha = (xb - (1 - alpha) * c * d).detach().requires_grad_(True)
                    f = positive_prob(model, x_alpha).sum()
                    g, = torch.autograd.grad(f, x_alpha)
                    grad_accum = grad_accum + (g * d).sum(dim=-1, keepdim=True)
                ig = (c.squeeze(-1) * (grad_accum.squeeze(-1) / self.steps))
                total += ig.abs().sum().item()
                count += ig.numel()
            scores[j] = total / max(count, 1)
        return scores
