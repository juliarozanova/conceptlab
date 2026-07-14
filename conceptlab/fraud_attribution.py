"""Concept-attribution on contract (fraud) data — the transfer demonstration.

Trains a TabTransformer on a loaded contract dataset, fits CAVs for each concept
on its activations, runs the concept-attribution methods, and grades them against
the **typology-defining** ground truth: on a fraud event of a given typology, the
correct explanation attributes to that typology's defining concepts (e.g. a card-
testing fraud -> short_burst, micro_amount, card_not_present) and not to
correlated-but-noncausal concepts (on_holiday).

Two views are reported:
* **global** — Spearman of each method's mean per-concept attribution against the
  fraud-frequency-weighted defining-concept mask;
* **per-typology hit-rate** — for each typology, does the method rank its defining
  concepts above the rest (mean rank-AUC over fraud events of that typology)?
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .concept_models import TabTransformer
from .concepts import EventBatch
from .contract import ContractDataset, load_contract
from .concept_eval import concept_audit
from .methods.concepts import ModelAdapter, fit_cav, ICS, TCAV, CAVAblation, ProbePatch


def _index(batch: EventBatch, idx):
    return EventBatch({k: v[idx] for k, v in batch.numeric.items()},
                      {k: v[idx] for k, v in batch.categorical.items()},
                      latents={}, decision=-1)


def _train_tab(ds: ContractDataset, cut: int, epochs: int, seed: int):
    torch.manual_seed(seed)
    batch = ds.event_batch()
    cards = {c: int(ds.categorical[c].max()) + 1 for c in ds.categorical}
    model = TabTransformer(numeric_cols=list(ds.numeric), categorical_cardinalities=cards,
                           d_model=64, n_layers=2, n_heads=4, field_dim=12, max_len=ds.seq_len)
    y = torch.as_tensor(ds.y, dtype=torch.long)
    tr = np.arange(cut)
    counts = torch.bincount(y[tr], minlength=2).float()
    weight = counts.sum() / (2 * counts.clamp(min=1))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(cut)
        for i in range(0, cut, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            logits = model(_index(batch, idx))
            F.cross_entropy(logits, y[idx], weight=weight).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        te = np.arange(cut, ds.n)
        p = torch.softmax(model(_index(batch, te)), -1)[:, 1].numpy()
    auc = float(roc_auc_score(ds.y[te], p)) if len(np.unique(ds.y[te])) > 1 else float("nan")
    return model, auc, te


@dataclass
class FraudAttrResult:
    name: str
    global_rho: float
    per_concept: list


def _grade(name, attr, gt_global):
    a = np.abs(attr)
    rho = float(spearmanr(a, gt_global).statistic) if np.ptp(a) > 0 and np.ptp(gt_global) > 0 else 0.0
    return FraudAttrResult(name=name, global_rho=rho, per_concept=[float(x) for x in a])


def run_fraud_attribution(contract_path: str | Path, out_dir: str | Path,
                          epochs: int = 30, seed: int = 0) -> dict:
    from .fraud_report import write_fraud_report
    t0 = time.time()
    ds = load_contract(contract_path)
    cut = int(0.7 * ds.n)
    model, auc, te = _train_tab(ds, cut, epochs, seed)

    batch = ds.event_batch()
    # cap the attribution sample for cost (IG runs steps x concepts forward passes)
    if len(te) > 1000:
        te = np.random.default_rng(0).choice(te, 1000, replace=False)
    te_batch = _index(batch, te)
    C_te = ds.C[te]
    y_te = ds.y[te]
    typ_te = ds.typology[te]
    attr_gt_te = ds.attribution_gt[te]

    # fraud-frequency-weighted defining-concept mask = global attribution GT
    fraud_mask = y_te == 1
    gt_global = attr_gt_te[fraud_mask].mean(0) if fraud_mask.any() else attr_gt_te.mean(0)

    # audit: is each concept decodable in the trained model?
    audit = _audit(model, te_batch, C_te, model.hookpoints())

    ad = ModelAdapter(model, "resid_post_L0")
    acts = ad.decision_acts(te_batch)
    cavs = np.stack([fit_cav(acts, C_te[:, k]) for k in range(ds.n_concepts)])

    # explain the model *on fraud events*: which concepts drive its fraud calls?
    fr_idx = np.where(y_te == 1)[0]
    fraud_batch = _index(te_batch, fr_idx)
    methods = {}
    for M in [ICS(steps=10), TCAV(), CAVAblation(), ProbePatch()]:
        attr = M.attribute(ad, fraud_batch, cavs)
        methods[M.name] = _grade(M.name, attr, gt_global).__dict__

    # ---- case explorer: top-N model-scored events, local drill-down --------
    from .case_explorer import build_cases
    with torch.no_grad():
        p_te = torch.softmax(model(te_batch), -1)[:, 1].numpy()
    # top-15 overall plus the 5 highest-scored *false positives* — the model's
    # confident mistakes are the most instructive explanation cases.
    top_all = np.argsort(-p_te)
    top15 = top_all[:15]
    fp_order = top_all[y_te[top_all] == 0]
    top_fp = np.array([i for i in fp_order if i not in set(top15)][:5], dtype=int)
    top_local = np.concatenate([top15, top_fp]) if len(top_fp) else top15
    top_local = top_local[np.argsort(-p_te[top_local])]
    case_batch = _index(te_batch, top_local)
    per_example = {}
    for M in [ICS(steps=10), CAVAblation(), ProbePatch()]:
        M.attribute(ad, case_batch, cavs)
        per_example[M.name] = M.last_per_example
    cases_payload = build_cases(
        model, ds, te=te[top_local], p_scores=p_te[top_local], cavs=cavs, layer=0,
        method_per_example=per_example,
        concept_input_deps=ds.concept_input_deps or {},
        codebooks=ds.codebooks or {}, n_cases=len(top_local))

    # per-typology hit-rate: does the method rank defining concepts above others
    # on that typology's fraud events? uses per-concept global attribution as the
    # ranking (a global explanation) scored against each typology's defining set.
    per_typ = {}
    for t_ in sorted(set(typ_te)):
        if t_ == "legit":
            continue
        defining = set(ds.typology_defining.get(t_, []))
        if not defining:
            per_typ[t_] = None       # first_party: no defining concepts (Bayes floor)
            continue
        target = np.array([1 if n in defining else 0 for n in ds.concept_names])
        per_typ[t_] = {}
        for mname, mres in methods.items():
            a = np.abs(np.array(mres["per_concept"]))
            per_typ[t_][mname] = (float(roc_auc_score(target, a))
                                  if target.sum() and target.sum() < len(target) else float("nan"))

    agg = {
        "name": Path(contract_path).name, "model_auc": auc,
        "concept_names": ds.concept_names, "concept_levels": ds.concept_levels,
        "gt_global": [float(x) for x in gt_global],
        "methods": methods, "per_typology_hit": per_typ,
        "audit": audit, "typology_defining": ds.typology_defining,
        "runtime_s": round(time.time() - t0, 1), "n": int(ds.n),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "metrics.json").write_text(json.dumps(agg, indent=2))
    agg["cases"] = cases_payload           # embedded in the report, not metrics.json
    write_fraud_report(agg, out_dir)
    return agg


def _audit(model, batch, C, hookpoints):
    from sklearn.linear_model import LogisticRegression
    out = {"hookpoints": hookpoints, "auc": {}}
    for hp in hookpoints:
        _, cache = model.run_with_cache(batch)
        a = cache[hp]
        if a.dim() == 3:
            a = a[:, -1, :]
        a = a.detach().numpy()
        cut = int(0.7 * len(a))
        aucs = []
        for k in range(C.shape[1]):
            yk = (C[:, k] > 0.5).astype(int)
            if len(np.unique(yk[:cut])) < 2 or len(np.unique(yk[cut:])) < 2:
                aucs.append(float("nan"))
                continue
            try:
                clf = LogisticRegression(max_iter=300).fit(a[:cut], yk[:cut])
                aucs.append(float(roc_auc_score(yk[cut:], clf.decision_function(a[cut:]))))
            except Exception:
                aucs.append(float("nan"))
        out["auc"][hp] = aucs
    out["best_auc"] = [float(x) for x in np.nanmax(np.stack([out["auc"][hp] for hp in hookpoints]), 0)]
    return out
