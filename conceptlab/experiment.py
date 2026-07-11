"""End-to-end experiment orchestration.

A run is: generate -> train -> build method context -> run every configured
method -> evaluate against ground truth -> write a self-contained HTML report.
Multiple seeds are averaged. The config is a plain dict (loaded from YAML).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import eval as ev
from .datagen import ConceptSpec, make_dataset
from .methods import build_context, build_method
from .train import TrainConfig, train_model


@dataclass
class RunConfig:
    name: str
    spec: dict
    model: dict = field(default_factory=dict)
    methods: list[str] = field(default_factory=list)
    method_kwargs: dict = field(default_factory=dict)
    n_train: int = 6000
    n_val: int = 1500
    seeds: list[int] = field(default_factory=lambda: [0])
    min_accuracy: float = 0.85
    interaction_concepts: list[int] | None = None

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        return RunConfig(**d)


def _train_one(cfg: RunConfig, seed: int):
    spec_d = dict(cfg.spec)
    spec_d["seed"] = seed
    spec = ConceptSpec.from_dict(spec_d)
    train = make_dataset(spec, cfg.n_train, split_seed=0)
    val = make_dataset(spec, cfg.n_val, split_seed=1)
    mcfg = TrainConfig(
        kind=cfg.model.get("kind", "mlp"),
        epochs=cfg.model.get("epochs", 60),
        lr=cfg.model.get("lr", 3e-3),
        batch_size=cfg.model.get("batch_size", 256),
        weight_decay=cfg.model.get("weight_decay", 1e-4),
        model_kwargs=cfg.model.get("model_kwargs", {}),
        seed=seed,
    )
    res = train_model(train, val, mcfg)
    return spec, train, val, res


def _evaluate_seed(cfg: RunConfig, spec: ConceptSpec, train, val, res) -> dict:
    world = train.world
    ctx = build_context(res.model, train, seed=0)
    true_dirs = world.anchor_unit()

    # ground-truth importance per concept
    if train.is_sequence:
        gt_imp = ev.sequence_data_importance(world, spec.seq_len)
    else:
        gt_imp = ev.model_importance(res.model, world)

    # sufficiency data: sample-level embeddings + labels
    if val.is_sequence:
        E_suff = val.X.reshape(len(val.X), -1, val.dim).mean(axis=1)
    else:
        E_suff = val.X
    y_suff = val.y

    out: dict[str, ev.MethodEval] = {}
    for name in cfg.methods:
        kwargs = cfg.method_kwargs.get(name, {})
        method = build_method(name, **kwargs).fit(ctx)
        me = ev.MethodEval(name=name)
        if getattr(method, "can_discover", False):
            disc = method.discovered_concepts()
            me.recovery = ev.recovery(disc, world)
            me.sufficiency = ev.sufficiency(disc, E_suff, y_suff)
            me.importance_per_concept = list(
                ev.aggregate_to_concepts(method.concept_importance(), world)
            ) if disc.shape[0] == world.n_dirs else []
        if getattr(method, "can_score", False):
            dir_scores = method.score_directions(true_dirs)
            concept_scores = ev.aggregate_to_concepts(dir_scores, world)
            me.faithfulness = ev.faithfulness(
                concept_scores, gt_imp, cfg.interaction_concepts
            )
            me.importance_per_concept = list(concept_scores)
        out[name] = me

    return {
        "methods": {k: asdict(v) for k, v in out.items()},
        "train_acc": res.train_acc,
        "val_acc": res.val_acc,
        "gt_importance": list(gt_imp),
        "history": res.history,
    }


def _average_seeds(per_seed: list[dict], method_names: list[str]) -> dict:
    """Average scalar metrics across seeds; keep the first seed's structure."""
    agg = {"val_acc": float(np.mean([s["val_acc"] for s in per_seed])),
           "train_acc": float(np.mean([s["train_acc"] for s in per_seed])),
           "methods": {}}
    for name in method_names:
        recs = [s["methods"][name] for s in per_seed]
        m: dict[str, Any] = {"name": name}

        def avg(path, sub):
            vals = [r[path][sub] for r in recs if r.get(path) and r[path].get(sub) is not None]
            return float(np.mean(vals)) if vals else None

        if recs[0].get("recovery"):
            m["recovery"] = {
                "mean_matched_cosine": avg("recovery", "mean_matched_cosine"),
                "coverage": avg("recovery", "coverage"),
                "redundancy": avg("recovery", "redundancy"),
                "regime": recs[0]["recovery"]["regime"],
                "n_discovered": recs[0]["recovery"]["n_discovered"],
                "n_true": recs[0]["recovery"]["n_true"],
                "ring_coverage": avg("recovery", "ring_coverage"),
            }
        if recs[0].get("faithfulness"):
            m["faithfulness"] = {
                "spearman": avg("faithfulness", "spearman"),
                "detects_interaction": bool(recs[0]["faithfulness"]["detects_interaction"]),
            }
        suffs = [r["sufficiency"] for r in recs if r.get("sufficiency") is not None
                 and not (isinstance(r["sufficiency"], float) and np.isnan(r["sufficiency"]))]
        m["sufficiency"] = float(np.mean(suffs)) if suffs else None
        agg["methods"][name] = m
    return agg


def run_experiment(cfg: RunConfig, out_dir: str | Path) -> dict:
    from .report import write_report  # local import to avoid plotly at import time

    out_dir = Path(out_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    per_seed = []
    first = None
    for seed in cfg.seeds:
        spec, train, val, res = _train_one(cfg, seed)
        if res.val_acc < cfg.min_accuracy:
            print(f"  [warn] seed {seed}: val_acc={res.val_acc:.3f} < "
                  f"min_accuracy={cfg.min_accuracy}; model may not have learned f.")
        per_seed.append(_evaluate_seed(cfg, spec, train, val, res))
        if first is None:
            first = (spec, train, val, res)

    agg = _average_seeds(per_seed, cfg.methods)
    spec, train, val, res = first
    agg.update({
        "name": cfg.name,
        "config": asdict(cfg),
        "seeds": cfg.seeds,
        "runtime_s": round(time.time() - t0, 1),
        "gt_importance": per_seed[0]["gt_importance"],
        "concept_names": [c.name for c in train.world.concepts],
        "label": spec.label if not train.is_sequence else spec.sequence_label,
        "is_sequence": train.is_sequence,
    })

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(agg, f, indent=2)

    write_report(agg, first, per_seed[0], out_dir)
    return agg
