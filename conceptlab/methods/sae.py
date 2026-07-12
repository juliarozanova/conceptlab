"""Sparse autoencoders over the embedding space.

Two variants:

* ``ReluSAE``  — ReLU encoder with an L1 sparsity penalty on activations.
* ``TopKSAE``  — hard top-K sparsity (the BatchTopK-style constraint).

The decoder columns are the learned dictionary directions. On a known geometry
these reproduce the SAE **capture regimes** from the literature: with an
overcomplete dictionary on a circular concept, atoms tend to *dilute* (each
covers an arc), which the recovery/coverage metrics surface directly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import InterpMethod, MethodContext, unit


class _SAEModule(nn.Module):
    def __init__(self, dim: int, n_features: int):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(dim))
        self.W_enc = nn.Parameter(torch.randn(dim, n_features) * (1.0 / dim ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.randn(n_features, dim) * (1.0 / n_features ** 0.5))

    def encode_pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


class _BaseSAE(InterpMethod):
    can_discover = True

    def __init__(self, overcomplete: float = 4.0, n_features: int | None = None,
                 epochs: int = 300, lr: float = 3e-3, seed: int = 0,
                 alive_threshold: float = 0.01):
        self.overcomplete = overcomplete
        self.n_features = n_features
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.alive_threshold = alive_threshold

    def _n_features(self, ctx) -> int:
        return self.n_features or max(4, int(round(ctx.n_dirs * self.overcomplete)))

    def _activate(self, pre):
        raise NotImplementedError

    def _penalty(self, feats):
        return torch.tensor(0.0)

    def fit(self, ctx: MethodContext) -> "_BaseSAE":
        self.ctx = ctx
        torch.manual_seed(self.seed)
        X = torch.as_tensor(ctx.E, dtype=torch.float32)
        # center-ish init of decoder bias at data mean helps
        m = self._n_features(ctx)
        sae = _SAEModule(ctx.dim, m)
        with torch.no_grad():
            sae.b_dec.data = X.mean(0)
        sae.normalize_decoder()
        opt = torch.optim.Adam(sae.parameters(), lr=self.lr)
        n = len(X)
        bs = min(1024, n)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                x = X[idx]
                pre = sae.encode_pre(x)
                feats = self._activate(pre)
                recon = sae.decode(feats)
                loss = F.mse_loss(recon, x) + self._penalty(feats)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sae.normalize_decoder()
        with torch.no_grad():
            feats = self._activate(sae.encode_pre(X))
            self._usage = (feats.abs() > 1e-6).float().mean(0).numpy()  # firing rate per atom
            self._all_dirs = unit(sae.W_dec.detach().numpy())
        self._sae = sae
        return self

    def discovered_concepts(self) -> np.ndarray:
        """Alive dictionary atoms only (firing rate above ``alive_threshold``).

        Dead atoms are randomly-initialised directions the SAE never uses;
        including them pollutes recovery/redundancy statistics. Excluding them
        is standard SAE practice, and the alive count is itself a finding.
        """
        alive = self._usage > self.alive_threshold
        return self._all_dirs[alive] if alive.any() else self._all_dirs

    def feature_usage(self) -> np.ndarray:
        """Firing rate of each dictionary atom (for dead-feature diagnostics)."""
        return self._usage


class ReluSAE(_BaseSAE):
    name = "relu_sae"

    def __init__(self, l1: float = 3e-3, **kw):
        super().__init__(**kw)
        self.l1 = l1

    def _activate(self, pre):
        return F.relu(pre)

    def _penalty(self, feats):
        return self.l1 * feats.abs().mean()


class TopKSAE(_BaseSAE):
    name = "topk_sae"

    def __init__(self, k: int = 4, **kw):
        super().__init__(**kw)
        self.k = k

    def _activate(self, pre):
        pre = F.relu(pre)
        k = min(self.k, pre.shape[-1])
        vals, idx = torch.topk(pre, k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, idx, vals)
        return out
