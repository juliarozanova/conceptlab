"""Method interface and shared context.

All methods operate in the **input embedding space** R^D — the space where
concepts are planted as Gaussian blobs. This keeps two things well defined and
comparable across model types:

* **discovery** decomposes the embeddings into directions that are directly
  comparable to the known anchors; and
* **importance** perturbs those embeddings and forwards them through the trained
  model, so ground-truth importance (obtained by toggling a concept in the data
  generator) and the method's score live in the same space.

A method advertises which capabilities it supports via ``can_discover`` /
``can_score``; the evaluation harness only asks for what a method provides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..datagen import Dataset, World


@dataclass
class MethodContext:
    """Everything a method may need, precomputed once per run."""

    model: nn.Module
    world: World
    is_sequence: bool
    E: np.ndarray              # (n_items, D) embeddings for discovery (tokens flattened)
    coeffs: np.ndarray         # (n_items, M) ground-truth anchor coefficients (skyline only)
    X_sample: torch.Tensor     # (N, ...) raw model inputs, for importance interventions
    device: str = "cpu"

    @property
    def dim(self) -> int:
        return self.E.shape[1]

    @property
    def n_dirs(self) -> int:
        return self.world.n_dirs


def build_context(model: nn.Module, dataset: Dataset, n_items: int = 4000,
                  n_sample: int = 1500, seed: int = 0) -> MethodContext:
    """Assemble a :class:`MethodContext` from a trained model and dataset."""
    rng = np.random.default_rng(seed)
    world = dataset.world

    # Embeddings for discovery: flatten tokens for sequences.
    if dataset.is_sequence:
        E_all = dataset.X.reshape(-1, dataset.dim)
        # per-token coefficients aligned with flattened embeddings
        coeff_tokens = []
        T = dataset.X.shape[1]
        for t in range(T):
            ns_t = {name: v[:, t] for name, v in dataset.Z.items()}
            coeff_tokens.append(world.anchor_coeffs(ns_t))
        C_all = np.stack(coeff_tokens, axis=1).reshape(-1, world.n_dirs)
    else:
        E_all = dataset.X
        C_all = world.anchor_coeffs(dataset.Z)

    idx = rng.permutation(len(E_all))[:n_items]
    E = E_all[idx].astype(np.float64)
    C = C_all[idx].astype(np.float64)

    sidx = rng.permutation(len(dataset.X))[:n_sample]
    X_sample = torch.as_tensor(dataset.X[sidx], dtype=torch.float32)

    return MethodContext(model=model, world=world, is_sequence=dataset.is_sequence,
                         E=E, coeffs=C, X_sample=X_sample)


class InterpMethod:
    """Base class. Override the capabilities you support."""

    name: str = "base"
    can_discover: bool = False
    can_score: bool = False

    def fit(self, ctx: MethodContext) -> "InterpMethod":
        self.ctx = ctx
        return self

    def discovered_concepts(self) -> np.ndarray:
        """(m, D) unit direction vectors discovered in embedding space."""
        raise NotImplementedError

    def concept_importance(self) -> np.ndarray:
        """(m,) importance of each discovered concept for the label.

        Default: score the discovered directions with the model via ablation, so
        every discovery method also gets an importance readout for the report.
        """
        from .causal import DirectionAblation
        dirs = self.discovered_concepts()
        return DirectionAblation().fit(self.ctx).score_directions(dirs)

    def score_directions(self, directions: np.ndarray) -> np.ndarray:
        """(k,) importance of each *given* unit direction for the label."""
        raise NotImplementedError


# ---- shared model-forward helpers -----------------------------------------


def positive_prob(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """P(y=1) for binary tasks; for multi-class, prob of the argmax-mean class."""
    logits = model(X)
    probs = torch.softmax(logits, dim=-1)
    if probs.shape[-1] == 2:
        return probs[:, 1]
    return probs.max(dim=-1).values


def project_out(X: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Remove the component of X along unit direction d (last-dim broadcast)."""
    coef = (X * d).sum(dim=-1, keepdim=True)
    return X - coef * d


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-12)
