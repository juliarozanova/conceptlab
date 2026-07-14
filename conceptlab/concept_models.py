"""The two model modes for the tabular-concepts testbed.

**Mode A (glass box)** — a differentiable function that maps event columns to the
concept representation via hand-built soft logic, embeds the (soft) concepts into
R^D with fixed anchors, and reads out the label. Nothing is trained except
optionally the head. Concept CAVs are known exactly (the anchors), so it is the
oracle mode: it isolates *attribution-mechanism* error from CAV-estimation error.

**Mode B (from scratch)** — a ``TabTransformer`` that consumes per-field event
embeddings fused into one token per event, runs a causal transformer, and is
trained on the label alone. It is never told about concepts; whether it
represents them is an empirical question the concept audit answers.

Both expose ``forward(...)`` returning label logits and ``run_with_cache`` giving
activations at named hookpoints for the concept methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .concepts import ConceptGraph, EventBatch, SoftContext
from .datagen import _orthonormal_basis
from .models import _Block


# ---------------------------------------------------------------------------
# concept -> representation embedding (shared by both modes)
# ---------------------------------------------------------------------------


def build_anchors(n_concepts: int, dim: int, seed: int = 0, geometry: str = "orthogonal") -> np.ndarray:
    rng = np.random.default_rng(seed)
    if geometry == "orthogonal" and n_concepts <= dim:
        return _orthonormal_basis(n_concepts, dim, rng)
    a = rng.standard_normal((n_concepts, dim))
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Mode A: glass-box soft logic
# ---------------------------------------------------------------------------


class SoftLogicModel(nn.Module):
    """inputs -> soft concepts -> concept embedding -> label.

    ``tau`` controls how crisp the logic is; it is the main stress knob. The
    concept representation is ``rep = C_soft @ anchors``; the head maps rep to a
    binary label. With ``train_head=False`` the head is set to the ground-truth
    label direction so the whole model is a pure oracle.
    """

    def __init__(self, graph: ConceptGraph, anchors: np.ndarray, tau: float = 0.1,
                 noise: float = 0.0):
        super().__init__()
        self.graph = graph
        self.tau = tau
        self.noise = noise
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.dim = anchors.shape[1]
        self.head = nn.Linear(self.dim, 2)

    # concept coordinates for a batch (soft, differentiable in numeric cols)
    def soft_concepts(self, ctx: SoftContext) -> torch.Tensor:
        return self.graph.concept_matrix_soft(ctx)

    def representation(self, ctx: SoftContext) -> torch.Tensor:
        C = self.soft_concepts(ctx)                    # (N, K)
        rep = C @ self.anchors                         # (N, D)
        if self.noise > 0:
            rep = rep + self.noise * torch.randn_like(rep)
        return rep

    def forward_from_rep(self, rep: torch.Tensor) -> torch.Tensor:
        return self.head(rep)

    def ctx_from_batch(self, b: EventBatch, requires_grad: bool = False) -> SoftContext:
        num = {}
        for k, v in b.numeric.items():
            t = torch.tensor(v, dtype=torch.float32)
            if requires_grad:
                t.requires_grad_(True)
            num[k] = t
        return SoftContext(numeric=num, categorical=b.categorical, latents=b.latents, tau=self.tau)

    def forward(self, b: EventBatch) -> torch.Tensor:
        ctx = self.ctx_from_batch(b)
        return self.forward_from_rep(self.representation(ctx))

    @torch.no_grad()
    def run_with_cache(self, b: EventBatch):
        ctx = self.ctx_from_batch(b)
        rep = self.representation(ctx)
        logits = self.forward_from_rep(rep)
        return logits, {"rep": rep.detach()}

    def hookpoints(self):
        return ["rep"]


# ---------------------------------------------------------------------------
# Mode B: TabTransformer trained from scratch
# ---------------------------------------------------------------------------


class TabTransformer(nn.Module):
    """One fused token per event, causal transformer, last-token readout.

    Numeric fields use a small piecewise-linear (2-knot) encoding; categoricals
    use embedding tables. Field embeddings are concatenated and projected to the
    event token — the TabBERT-at-toy-scale shape used in the fraud model.
    """

    def __init__(self, numeric_cols: list[str], categorical_cardinalities: dict[str, int],
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 4, n_classes: int = 2,
                 max_len: int = 64, field_dim: int = 16):
        super().__init__()
        self.numeric_cols = numeric_cols
        self.categorical_cols = list(categorical_cardinalities)
        self.field_dim = field_dim
        # numeric: linear lift of [x, relu(x-b1), relu(x-b2)] -> field_dim
        self.num_proj = nn.ModuleDict({c: nn.Linear(3, field_dim) for c in numeric_cols})
        self.num_knots = nn.ParameterDict(
            {c: nn.Parameter(torch.tensor([-1.0, 1.0])) for c in numeric_cols})
        self.cat_emb = nn.ModuleDict(
            {c: nn.Embedding(card, field_dim) for c, card in categorical_cardinalities.items()})
        n_fields = len(numeric_cols) + len(self.categorical_cols)
        self.fuse = nn.Linear(n_fields * field_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(_Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.n_layers = n_layers
        self.d_model = d_model

    def _tokens(self, num: dict[str, torch.Tensor], cat: dict[str, torch.Tensor]) -> torch.Tensor:
        B, T = next(iter({**num, **cat}.values())).shape
        fields = []
        for c in self.numeric_cols:
            x = num[c].unsqueeze(-1)                    # (B,T,1)
            b1, b2 = self.num_knots[c]
            ple = torch.cat([x, F.relu(x - b1), F.relu(x - b2)], dim=-1)
            fields.append(self.num_proj[c](ple))
        for c in self.categorical_cols:
            fields.append(self.cat_emb[c](cat[c]))
        h = torch.cat(fields, dim=-1)                  # (B,T,n_fields*field_dim)
        return self.fuse(h)                            # (B,T,d_model)

    def embed_fields(self, num: dict, cat: dict) -> dict:
        """Per-field embeddings (each (B, T, field_dim)), in model field order.

        The differentiable surface for input-level attribution: categorical codes
        have no continuous path, but their embeddings do — IG interpolates here.
        """
        out = {}
        for c in self.numeric_cols:
            x = num[c].unsqueeze(-1)
            b1, b2 = self.num_knots[c]
            ple = torch.cat([x, F.relu(x - b1), F.relu(x - b2)], dim=-1)
            out[c] = self.num_proj[c](ple)
        for c in self.categorical_cols:
            out[c] = self.cat_emb[c](cat[c])
        return out

    def acts_from_field_embeddings(self, field_embs: dict, layer: int) -> torch.Tensor:
        """Residual-stream activation at ``resid_post_L{layer}``, decision position,
        computed from (possibly grad-enabled / interpolated) field embeddings."""
        fields = [field_embs[c] for c in self.numeric_cols + self.categorical_cols]
        h = self.fuse(torch.cat(fields, dim=-1))
        B, T, _ = h.shape
        h = h + self.pos[:, :T, :]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        for i, blk in enumerate(self.blocks):
            h = blk(h, attn_mask=mask)
            if i == layer:
                return h[:, -1, :]
        return h[:, -1, :]

    def logit_from_inputs(self, num: dict, cat: dict) -> torch.Tensor:
        """Differentiable forward from explicit numeric tensors (for input-level
        attribution). ``num`` values are torch tensors (grad-enabled by caller);
        ``cat`` are long tensors."""
        h = self._tokens(num, cat)
        B, T, _ = h.shape
        h = h + self.pos[:, :T, :]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            h = blk(h, attn_mask=mask)
        h = self.ln_f(h)
        return self.head(h[:, -1, :])

    def forward(self, batch: EventBatch, cache=None) -> torch.Tensor:
        num = {c: torch.as_tensor(batch.numeric[c], dtype=torch.float32) for c in self.numeric_cols}
        cat = {c: torch.as_tensor(batch.categorical[c], dtype=torch.long) for c in self.categorical_cols}
        h = self._tokens(num, cat)
        if cache is not None:
            cache["embed"] = h.detach()
        B, T, _ = h.shape
        h = h + self.pos[:, :T, :]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        for i, blk in enumerate(self.blocks):
            h = blk(h, attn_mask=mask)
            if cache is not None:
                cache[f"resid_post_L{i}"] = h.detach()
        h = self.ln_f(h)
        pooled = h[:, -1, :]                            # decision point = last event
        if cache is not None:
            cache["pooled"] = pooled.detach()
        return self.head(pooled)

    @torch.no_grad()
    def run_with_cache(self, batch: EventBatch):
        cache: dict = {}
        logits = self.forward(batch, cache=cache)
        return logits, cache

    def hookpoints(self):
        return ["embed"] + [f"resid_post_L{i}" for i in range(self.n_layers)] + ["pooled"]


# ---------------------------------------------------------------------------
# dataset tying world + graph together
# ---------------------------------------------------------------------------


@dataclass
class ConceptDataset:
    batch: EventBatch
    C: np.ndarray                # (N, K) hard concept values at decision point
    y: np.ndarray                # (N,) label
    graph: ConceptGraph
    anchors: np.ndarray          # (K, D) true concept directions in rep space
    world: object                # SequenceWorld (for resimulation)

    @property
    def concept_names(self):
        return self.graph.names


def make_concept_dataset(world, graph: ConceptGraph, n: int, dim: int,
                         offset: int = 0, anchor_seed: int = 0,
                         geometry: str = "orthogonal") -> ConceptDataset:
    batch = world.sample(n, offset=offset)
    C = graph.concept_matrix_hard(batch)
    y = graph.label_hard(batch)
    anchors = build_anchors(graph.n_concepts, dim, seed=anchor_seed, geometry=geometry)
    return ConceptDataset(batch=batch, C=C, y=y, graph=graph, anchors=anchors, world=world)
