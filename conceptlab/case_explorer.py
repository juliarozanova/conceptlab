"""Per-case drill-down data for the fraud attribution report.

For the top-N model-scored (predicted-fraud) events this computes everything the
interactive case explorer needs:

* the raw event-sequence table (display values via the contract codebooks);
* per-concept **local** attribution scores (ICS / CAV-ablation / probe-patch,
  taken from each method's per-example vectors) next to the ground-truth
  defining-concept mask;
* the **input→concept cube**: for each concept, how much each input cell
  (event t × field) influences the concept's detection in the model — computed
  two ways: **IG-vs-probe** (integrated gradients of the CAV probe score along
  an embedding-space path from a baseline event window) and **cell occlusion**
  (set the cell to baseline, measure the change in the CAV coefficient);
* the ground-truth 0/1 cell mask rendered from the contract's
  ``concept_input_deps`` (which cells the concept's definition actually reads).

Baselines are the same for both methods: numeric → column mean, categorical →
modal code. All heatmaps are exported normalized per (case, concept, method).
"""

from __future__ import annotations

import numpy as np
import torch

from .concepts import EventBatch
from .contract import ContractDataset


# ---------------------------------------------------------------------------
# GT cell masks from concept_input_deps
# ---------------------------------------------------------------------------


def render_gt_mask(deps: list, T: int, fields: list[str]) -> np.ndarray:
    """(T, F) 0/1 mask from [(field, scope), ...]."""
    M = np.zeros((T, len(fields)), dtype=int)
    fidx = {f: j for j, f in enumerate(fields)}
    for field, scope in deps:
        if field not in fidx:
            continue
        j = fidx[field]
        if scope == "decision":
            M[T - 1, j] = 1
        elif scope == "history":
            M[: T - 1, j] = 1
        elif scope == "all":
            M[:, j] = 1
        elif scope.startswith("window:"):
            k = int(scope.split(":")[1])
            M[max(0, T - k):, j] = 1
    return M


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def _baselines(ds: ContractDataset):
    num_base = {c: float(v.mean()) for c, v in ds.numeric.items()}
    cat_base = {c: int(np.bincount(v.reshape(-1)).argmax()) for c, v in ds.categorical.items()}
    return num_base, cat_base


# ---------------------------------------------------------------------------
# input -> concept cubes
# ---------------------------------------------------------------------------


def occlusion_cube(model, ds: ContractDataset, idx: int, cavs: np.ndarray,
                   layer: int, num_base, cat_base) -> np.ndarray:
    """(K, T, F) |Δ CAV coefficient| when each cell is set to its baseline.

    One batched forward covers all T*F single-cell variants; every concept's
    coefficient is read from the same activations, so the whole cube costs one
    batch per case.
    """
    fields = list(ds.numeric) + list(ds.categorical)
    T = ds.seq_len
    F_ = len(fields)
    n_var = T * F_ + 1

    num = {c: np.repeat(ds.numeric[c][idx][None], n_var, axis=0).copy() for c in ds.numeric}
    cat = {c: np.repeat(ds.categorical[c][idx][None], n_var, axis=0).copy() for c in ds.categorical}
    v = 1  # variant 0 = original
    for t in range(T):
        for f in fields:
            if f in num:
                num[f][v, t] = num_base[f]
            else:
                cat[f][v, t] = cat_base[f]
            v += 1

    batch = EventBatch(num, cat, latents={}, decision=-1)
    with torch.no_grad():
        _, cache = model.run_with_cache(batch)
    acts = cache[f"resid_post_L{layer}"][:, -1, :].numpy()          # (n_var, d)
    coeff = acts @ cavs.T                                            # (n_var, K)
    delta = np.abs(coeff[1:] - coeff[0])                             # (T*F, K)
    return delta.T.reshape(len(cavs), T, F_)


def ig_probe_cube(model, ds: ContractDataset, idx: int, cavs: np.ndarray,
                  layer: int, num_base, cat_base, steps: int = 12) -> np.ndarray:
    """(K, T, F) integrated gradients of each concept's probe score.

    Path: field embeddings interpolated from the baseline window's embeddings to
    the actual window's. Cell attribution = |Σ_dims (Δemb ⊙ avg grad)| at (t, f).
    """
    fields = list(ds.numeric) + list(ds.categorical)
    T = ds.seq_len
    K = len(cavs)

    num_a = {c: torch.tensor(ds.numeric[c][idx][None], dtype=torch.float32) for c in ds.numeric}
    cat_a = {c: torch.tensor(ds.categorical[c][idx][None], dtype=torch.long) for c in ds.categorical}
    num_b = {c: torch.full_like(num_a[c], num_base[c]) for c in num_a}
    cat_b = {c: torch.full_like(cat_a[c], cat_base[c]) for c in cat_a}

    with torch.no_grad():
        emb_a = model.embed_fields(num_a, cat_a)     # field -> (1, T, fd)
        emb_b = model.embed_fields(num_b, cat_b)

    # batched interpolation over steps: (steps, T, fd) per field
    alphas = torch.linspace(1.0 / steps, 1.0, steps).view(steps, 1, 1)
    interp = {f: (emb_b[f] + alphas * (emb_a[f] - emb_b[f])).detach().requires_grad_(True)
              for f in fields}

    acts = model.acts_from_field_embeddings(interp, layer)          # (steps, d)
    cav_t = torch.tensor(cavs, dtype=torch.float32)                  # (K, d)
    scores = torch.sigmoid(acts @ cav_t.T).sum(0)                    # (K,) summed over steps

    cube = np.zeros((K, T, len(fields)))
    for k in range(K):
        grads = torch.autograd.grad(scores[k], list(interp.values()),
                                    retain_graph=(k < K - 1), allow_unused=True)
        for j, f in enumerate(fields):
            g = grads[j]
            if g is None:
                continue
            avg_g = g.mean(0)                                       # (T, fd)
            delta = (emb_a[f][0] - emb_b[f][0])                     # (T, fd)
            cube[k, :, j] = (delta * avg_g).sum(-1).abs().numpy()
    return cube


# ---------------------------------------------------------------------------
# assembling case data
# ---------------------------------------------------------------------------


def _norm2(a):
    m = np.max(a)
    return np.round(a / m, 2).tolist() if m > 1e-12 else np.zeros_like(a).round(2).tolist()


def _display_table(ds: ContractDataset, idx: int, codebooks: dict) -> list:
    fields = list(ds.numeric) + list(ds.categorical)
    T = ds.seq_len
    rows = []
    for t in range(T):
        row = []
        for f in fields:
            if f in ds.numeric:
                v = float(ds.numeric[f][idx, t])
                row.append(f"{v:.2f}" if f != "hour" else f"{int(v):02d}h")
            else:
                code = int(ds.categorical[f][idx, t])
                row.append(str(codebooks.get(f, {}).get(str(code), code)))
        rows.append(row)
    return rows


def build_cases(model, ds: ContractDataset, te: np.ndarray, p_scores: np.ndarray,
                cavs: np.ndarray, layer: int, method_per_example: dict,
                concept_input_deps: dict, codebooks: dict, n_cases: int = 20) -> dict:
    """Assemble the case-explorer JSON payload.

    ``te`` are eval-split indices into ``ds``; ``p_scores`` the model's P(fraud)
    aligned with ``te``; ``method_per_example`` maps method name -> (len(te), K)
    local attribution matrices (aligned with ``te``).
    """
    fields = list(ds.numeric) + list(ds.categorical)
    T = ds.seq_len
    names = ds.concept_names
    order = np.argsort(-p_scores)[:n_cases]
    num_base, cat_base = _baselines(ds)

    gt_masks = {n: render_gt_mask(concept_input_deps.get(n, []), T, fields).tolist()
                for n in names}

    cases = []
    for rank, j in enumerate(order):
        gi = int(te[j])                                     # global index
        occ = occlusion_cube(model, ds, gi, cavs, layer, num_base, cat_base)
        ig = ig_probe_cube(model, ds, gi, cavs, layer, num_base, cat_base)
        typ = str(ds.typology[gi])
        defining = set(ds.typology_defining.get(typ, []))
        cases.append({
            "rank": rank, "score": round(float(p_scores[j]), 3),
            "label": int(ds.y[gi]), "typology": typ,
            "table": _display_table(ds, gi, codebooks),
            "concepts": {
                n: {
                    "value": int(ds.C[gi, k]),
                    "gt_defining": int(n in defining),
                    "methods": {m: float(f"{method_per_example[m][j, k]:.3g}")
                                for m in method_per_example},
                    "ig": _norm2(ig[k]),
                    "occ": _norm2(occ[k]),
                } for k, n in enumerate(names)
            },
        })

    return {
        "fields": fields, "T": T, "concept_names": names,
        "concept_levels": ds.concept_levels,
        "gt_masks": gt_masks,
        "methods": list(method_per_example),
        "cases": cases,
        "baseline_note": "numeric baseline = column mean; categorical baseline = modal code",
    }
