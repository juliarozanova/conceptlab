"""Training for the tabular-concepts models.

Mode A's head can be fit in closed form-ish (or by a few steps) since the
representation is a known linear function of concepts; Mode B (TabTransformer) is
trained end-to-end on the label and must clear an accuracy bar before any
interpretability method is trusted on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .concept_models import ConceptDataset, SoftLogicModel, TabTransformer


@dataclass
class TabTrainResult:
    model: object
    train_acc: float
    val_acc: float
    history: dict


def fit_soft_logic(train: ConceptDataset, tau: float = 0.1, noise: float = 0.0,
                   epochs: int = 150, lr: float = 5e-2) -> SoftLogicModel:
    """Fit only the linear head of the glass-box model to the label."""
    model = SoftLogicModel(train.graph, train.anchors, tau=tau, noise=noise)
    y = torch.as_tensor(train.y, dtype=torch.long)
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(train.batch)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    return model


def train_tab_transformer(train: ConceptDataset, val: ConceptDataset,
                          epochs: int = 40, lr: float = 2e-3, batch_size: int = 256,
                          seed: int = 0, **model_kw) -> TabTrainResult:
    torch.manual_seed(seed)
    cards = {c: int(max(train.batch.categorical[c].max(), val.batch.categorical[c].max())) + 1
             for c in train.batch.categorical}
    model = TabTransformer(numeric_cols=list(train.batch.numeric),
                           categorical_cardinalities=cards,
                           max_len=train.batch.T, **model_kw)
    ytr = torch.as_tensor(train.y, dtype=torch.long)
    yva = torch.as_tensor(val.y, dtype=torch.long)
    counts = torch.bincount(ytr, minlength=2).float()
    weight = counts.sum() / (2 * counts.clamp(min=1))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n = train.batch.n
    hist = {"loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size].numpy()
            sub = _index_batch(train.batch, idx)
            opt.zero_grad()
            logits = model(sub)
            loss = F.cross_entropy(logits, ytr[idx], weight=weight)
            loss.backward()
            opt.step()
            ep += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            tr = (model(train.batch).argmax(1) == ytr).float().mean().item()
            va = (model(val.batch).argmax(1) == yva).float().mean().item()
        hist["loss"].append(ep / n)
        hist["train_acc"].append(tr)
        hist["val_acc"].append(va)
    return TabTrainResult(model=model, train_acc=hist["train_acc"][-1],
                          val_acc=hist["val_acc"][-1], history=hist)


def _index_batch(batch, idx):
    from .concepts import EventBatch
    num = {k: v[idx] for k, v in batch.numeric.items()}
    cat = {k: v[idx] for k, v in batch.categorical.items()}
    lat = {k: v[idx] for k, v in batch.latents.items()}
    out = EventBatch(num, cat, lat, batch.decision)
    if hasattr(batch, "_seq_ids"):
        out._seq_ids = batch._seq_ids[idx]
    return out
