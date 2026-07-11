"""Baseline / calibration methods.

* ``LinearProbeSkyline`` — supervised recovery using the ground-truth anchor
  coefficients. It is a *skyline*: the best a linear method could do given the
  answer. Real unsupervised methods should be judged relative to it.
* ``PCAMethod`` / ``ICAMethod`` — unsupervised linear decompositions, the floor
  any learned method should beat.
* ``LabelProbeImportance`` — importance from a supervised label probe's
  sensitivity to each direction; a cheap importance baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import Ridge

from .base import InterpMethod, MethodContext, unit


class LinearProbeSkyline(InterpMethod):
    name = "probe_skyline"
    can_discover = True

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, ctx: MethodContext) -> "LinearProbeSkyline":
        self.ctx = ctx
        # Regress embeddings' coefficients: weight vector per anchor row is the
        # recovered direction. Ridge over E -> each true coefficient column.
        reg = Ridge(alpha=self.alpha)
        reg.fit(ctx.E, ctx.coeffs)               # coeffs: (n, M)
        W = reg.coef_                             # (M, D)
        self._dirs = unit(W)
        return self

    def discovered_concepts(self) -> np.ndarray:
        return self._dirs


class PCAMethod(InterpMethod):
    name = "pca"
    can_discover = True

    def __init__(self, n_components: int | None = None, overcomplete: float = 1.0):
        self.n_components = n_components
        self.overcomplete = overcomplete

    def fit(self, ctx: MethodContext) -> "PCAMethod":
        self.ctx = ctx
        k = self.n_components or int(round(ctx.n_dirs * self.overcomplete))
        k = min(k, ctx.dim, len(ctx.E))
        pca = PCA(n_components=k, random_state=0).fit(ctx.E)
        self._dirs = unit(pca.components_)
        return self

    def discovered_concepts(self) -> np.ndarray:
        return self._dirs


class ICAMethod(InterpMethod):
    name = "ica"
    can_discover = True

    def __init__(self, n_components: int | None = None, overcomplete: float = 1.0):
        self.n_components = n_components
        self.overcomplete = overcomplete

    def fit(self, ctx: MethodContext) -> "ICAMethod":
        self.ctx = ctx
        k = self.n_components or int(round(ctx.n_dirs * self.overcomplete))
        k = min(k, ctx.dim, len(ctx.E))
        ica = FastICA(n_components=k, random_state=0, max_iter=500, whiten="unit-variance")
        ica.fit(ctx.E)
        # rows of the mixing matrix are the directions in data space
        self._dirs = unit(ica.mixing_.T)
        return self

    def discovered_concepts(self) -> np.ndarray:
        return self._dirs


class LabelProbeImportance(InterpMethod):
    name = "label_probe"
    can_score = True

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, ctx: MethodContext) -> "LabelProbeImportance":
        self.ctx = ctx
        return self

    def score_directions(self, directions: np.ndarray) -> np.ndarray:
        """Importance = |weight| of a linear label-probe on each direction's
        coefficient. Purely linear, so it should miss XOR-type interactions."""
        ctx = self.ctx
        y = ctx.model(ctx.X_sample).argmax(-1).detach().numpy()
        # project sample embeddings onto directions
        if ctx.is_sequence:
            E = ctx.X_sample.reshape(len(ctx.X_sample), -1, ctx.dim).mean(dim=1).numpy()
        else:
            E = ctx.X_sample.numpy()
        proj = E @ directions.T                    # (N, k)
        reg = Ridge(alpha=self.alpha).fit(proj, y.astype(float))
        return np.abs(reg.coef_)
