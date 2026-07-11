"""Toy models with activation caching.

Both models expose ``run_with_cache(x) -> (logits, cache)`` where ``cache`` maps
a hookpoint name to the activation at that point (a TransformerLens-style API,
hand-rolled to avoid a heavy dependency at toy scale). Interpretability methods
fit on whichever hookpoint the experiment selects.

The embeddings produced by :mod:`conceptlab.datagen` play the role of a frozen
input embedding layer, so the transformer consumes token vectors directly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyMLP(nn.Module):
    """A small MLP over a single embedding vector x in R^D.

    Hookpoints: ``input``, ``hidden0`` ... ``hidden{L-1}`` (post-activation),
    ``pre_logits``.
    """

    def __init__(self, dim: int, hidden: int = 64, n_layers: int = 2, n_classes: int = 2):
        super().__init__()
        self.dim = dim
        self.n_classes = n_classes
        self.in_proj = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers - 1))
        self.head = nn.Linear(hidden, n_classes)
        self.hidden = hidden
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        if cache is not None:
            cache["input"] = x.detach()
        h = F.relu(self.in_proj(x))
        if cache is not None:
            cache["hidden0"] = h.detach()
        for i, blk in enumerate(self.blocks, start=1):
            h = F.relu(blk(h))
            if cache is not None:
                cache[f"hidden{i}"] = h.detach()
        if cache is not None:
            cache["pre_logits"] = h.detach()
        return self.head(h)

    @torch.no_grad()
    def run_with_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cache: dict = {}
        logits = self.forward(x, cache=cache)
        return logits, cache

    def hookpoints(self) -> list[str]:
        return ["input"] + [f"hidden{i}" for i in range(self.n_layers)] + ["pre_logits"]


class _Block(nn.Module):
    """Pre-norm transformer block (bidirectional by default)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ToyTransformer(nn.Module):
    """A 1-2 layer transformer over a sequence of event embeddings (T, D).

    A learned CLS token is prepended and its final residual is the pooled
    representation used for classification.

    Hookpoints: ``embed`` (post input-projection, per token), ``resid_post_L{i}``
    (per token, includes the CLS slot at index 0), ``pooled`` (CLS residual).
    """

    def __init__(self, dim: int, d_model: int = 64, n_layers: int = 1, n_heads: int = 4,
                 n_classes: int = 2, max_len: int = 64, causal: bool = False):
        super().__init__()
        self.dim = dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.causal = causal
        self.in_proj = nn.Linear(dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(_Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        # x: (B, T, D)
        B, T, _ = x.shape
        h = self.in_proj(x)
        if cache is not None:
            cache["embed"] = h.detach()
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)             # (B, T+1, d_model)
        h = h + self.pos[:, : T + 1, :]
        attn_mask = None
        if self.causal:
            L = T + 1
            attn_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        for i, blk in enumerate(self.blocks):
            h = blk(h, attn_mask=attn_mask)
            if cache is not None:
                cache[f"resid_post_L{i}"] = h.detach()
        h = self.ln_f(h)
        pooled = h[:, 0, :]                          # CLS
        if cache is not None:
            cache["pooled"] = pooled.detach()
        return self.head(pooled)

    @torch.no_grad()
    def run_with_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cache: dict = {}
        logits = self.forward(x, cache=cache)
        return logits, cache

    def hookpoints(self) -> list[str]:
        return ["embed"] + [f"resid_post_L{i}" for i in range(self.n_layers)] + ["pooled"]


def build_model(kind: str, dim: int, n_classes: int = 2, **kw) -> nn.Module:
    if kind == "mlp":
        return ToyMLP(dim=dim, n_classes=n_classes, **kw)
    if kind == "transformer":
        return ToyTransformer(dim=dim, n_classes=n_classes, **kw)
    raise ValueError(f"unknown model kind: {kind}")


def token_activations(cache: dict, hookpoint: str, drop_cls: bool = True) -> torch.Tensor:
    """Return activations at a hookpoint flattened to (n_items, d).

    For per-token transformer hookpoints this flattens (B, T, d) -> (B*T, d),
    dropping the CLS slot by default so concept-recovery methods see only real
    event tokens.
    """
    a = cache[hookpoint]
    if a.dim() == 3:
        if drop_cls and hookpoint.startswith("resid_post"):
            a = a[:, 1:, :]
        return a.reshape(-1, a.shape[-1])
    return a
