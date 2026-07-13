"""Run a tabular-concept experiment: both modes, all methods, GT + audit, report."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .concept_configs import CONFIGS, ConceptConfig
from .concept_eval import compute_ground_truth, concept_audit, score_method
from .concept_models import make_concept_dataset
from .concept_train import fit_soft_logic, train_tab_transformer
from .methods.concepts import (ModelAdapter, fit_cav, ICS, TCAV, CAVAblation,
                               ProbePatch, InputAgg, CAVSkyline)


def _unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def _run_mode_a(tr, va, gt, cfg: ConceptConfig) -> dict:
    model = fit_soft_logic(tr, tau=cfg.tau)
    import torch
    with torch.no_grad():
        acc = float((model(va.batch).argmax(1).numpy() == va.y).mean())
    ad = ModelAdapter(model, "rep")
    acts = ad.decision_acts(va.batch)
    cavs = np.stack([fit_cav(acts, va.C[:, k]) for k in range(tr.graph.n_concepts)])
    true_cavs = _unit(tr.anchors)
    scores = {}
    for M in [ICS(), TCAV(), CAVAblation(), ProbePatch(), CAVSkyline()]:
        arg = true_cavs if M.name == "cav_skyline" else cavs
        attr = M.attribute(ad, va.batch, arg)
        scores[M.name] = score_method(M.name, attr, gt).__dict__
    return {"val_acc": acc, "hookpoint": "rep", "scores": scores}


def _run_mode_b(tr, va, gt, cfg: ConceptConfig, seed: int) -> dict:
    res = train_tab_transformer(tr, va, epochs=cfg.modeB_epochs, d_model=48,
                                n_layers=2, n_heads=4, field_dim=12, seed=seed)
    audit = concept_audit(res.model, va, res.model.hookpoints())
    ad = ModelAdapter(res.model, "resid_post_L0")
    acts = ad.decision_acts(va.batch)
    cavs = np.stack([fit_cav(acts, va.C[:, k]) for k in range(tr.graph.n_concepts)])
    scores = {}
    for M in [ICS(), TCAV(), CAVAblation(), ProbePatch()]:
        attr = M.attribute(ad, va.batch, cavs)
        scores[M.name] = score_method(M.name, attr, gt).__dict__
    # input_agg uses the DAG directly (mode-independent reference)
    ia = InputAgg(tr.graph, va)
    scores["input_agg"] = score_method("input_agg", ia.attribute(ad, va.batch, cavs), gt).__dict__
    return {"val_acc": res.val_acc, "hookpoint": "resid_post_L0",
            "scores": scores, "audit": audit}


def run_concept_experiment(cfg: ConceptConfig, out_dir: str | Path) -> dict:
    from .concept_report import write_concept_report
    out_dir = Path(out_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    per_seed = []
    first = None
    for seed in cfg.seeds:
        world, graph = cfg.build()
        tr = make_concept_dataset(world, graph, cfg.n_train, dim=cfg.dim, offset=0, anchor_seed=seed)
        va = make_concept_dataset(world, graph, cfg.n_val, dim=cfg.dim, offset=cfg.n_train, anchor_seed=seed)
        gt = compute_ground_truth(va)
        modeA = _run_mode_a(tr, va, gt, cfg)
        modeB = _run_mode_b(tr, va, gt, cfg, seed)
        per_seed.append({"gt": gt, "modeA": modeA, "modeB": modeB, "tr": tr, "va": va})
        if first is None:
            first = per_seed[-1]

    agg = _aggregate(per_seed, cfg)
    agg["runtime_s"] = round(time.time() - t0, 1)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(_json_safe(agg), f, indent=2)
    write_concept_report(agg, first, cfg, out_dir)
    return agg


def _aggregate(per_seed, cfg) -> dict:
    gt0 = per_seed[0]["gt"]
    names = gt0.names

    def avg_scores(mode):
        methods = per_seed[0][mode]["scores"].keys()
        out = {}
        for m in methods:
            keys = ["rho_interventional", "rho_shapley", "support_leakage", "pivotal_gate"]
            out[m] = {k: float(np.mean([s[mode]["scores"][m][k] for s in per_seed])) for k in keys}
            out[m]["per_concept"] = per_seed[0][mode]["scores"][m]["per_concept"]
        return out

    return {
        "name": cfg.name,
        "description": cfg.description,
        "concept_names": names,
        "support_mask": [bool(x) for x in gt0.support_mask],
        "gt_interventional": [float(x) for x in gt0.interventional],
        "gt_shapley": [float(x) for x in gt0.shapley],
        "definition_gap": float(np.mean([s["gt"].definition_gap for s in per_seed])),
        "modeA_val_acc": float(np.mean([s["modeA"]["val_acc"] for s in per_seed])),
        "modeB_val_acc": float(np.mean([s["modeB"]["val_acc"] for s in per_seed])),
        "modeA_scores": avg_scores("modeA"),
        "modeB_scores": avg_scores("modeB"),
        "audit": per_seed[0]["modeB"]["audit"],
        "seeds": cfg.seeds,
        "tau": cfg.tau,
    }


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items() if k not in ("tr", "va", "gt")}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS) + ["all"])
    ap.add_argument("--out", default="docs/concept_runs")
    args = ap.parse_args(argv)
    names = list(CONFIGS) if args.config == "all" else [args.config]
    for n in names:
        print(f"[conceptlab] running concept experiment '{n}' ...")
        agg = run_concept_experiment(CONFIGS[n], args.out)
        print(f"  modeA acc={agg['modeA_val_acc']:.3f} modeB acc={agg['modeB_val_acc']:.3f} "
              f"def_gap={agg['definition_gap']:.2f} ({agg['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
