"""Training loop for the toy models.

A model that hasn't learned the label function ``f`` invalidates every downstream
interpretability comparison, so :func:`train_model` returns the achieved accuracy
and callers should assert it clears a threshold before running methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .datagen import Dataset
from .models import build_model


@dataclass
class TrainConfig:
    kind: str = "mlp"                     # mlp | transformer
    epochs: int = 60
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 0
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class TrainResult:
    model: nn.Module
    train_acc: float
    val_acc: float
    history: dict[str, list[float]]


def _tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def train_model(train: Dataset, val: Dataset, cfg: TrainConfig) -> TrainResult:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    n_classes = int(max(train.y.max(), val.y.max())) + 1
    model = build_model(cfg.kind, dim=train.dim, n_classes=n_classes, **cfg.model_kwargs)

    Xtr, ytr = _tensor(train.X), torch.as_tensor(train.y, dtype=torch.long)
    Xva, yva = _tensor(val.X), torch.as_tensor(val.y, dtype=torch.long)

    # class-balanced loss: the labels (AND/XOR of concepts) are often skewed.
    counts = torch.bincount(ytr, minlength=n_classes).float()
    weight = (counts.sum() / (n_classes * counts.clamp(min=1)))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    hist: dict[str, list[float]] = {"loss": [], "train_acc": [], "val_acc": []}
    n = len(ytr)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            opt.zero_grad()
            logits = model(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx], weight=weight)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            tr_acc = (model(Xtr).argmax(1) == ytr).float().mean().item()
            va_acc = (model(Xva).argmax(1) == yva).float().mean().item()
        hist["loss"].append(ep_loss / n)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)

    return TrainResult(model=model, train_acc=hist["train_acc"][-1],
                       val_acc=hist["val_acc"][-1], history=hist)
