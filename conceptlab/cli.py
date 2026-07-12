"""Command-line entry point: run an experiment from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .experiment import RunConfig, run_experiment


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a conceptlab interpretability experiment.")
    ap.add_argument("--config", required=True, help="path to a YAML experiment config")
    ap.add_argument("--out", default="docs/runs", help="output directory (default: docs/runs/)")
    args = ap.parse_args(argv)

    cfg_dict = yaml.safe_load(Path(args.config).read_text())
    cfg = RunConfig.from_dict(cfg_dict)
    print(f"[conceptlab] running '{cfg.name}' with {len(cfg.methods)} methods over seeds {cfg.seeds} ...")
    agg = run_experiment(cfg, args.out)

    print(f"[conceptlab] done in {agg['runtime_s']}s · val_acc={agg['val_acc']:.3f}")
    for name, m in agg["methods"].items():
        rec = m.get("recovery")
        fa = m.get("faithfulness")
        bits = []
        if rec:
            bits.append(f"recov={rec['mean_matched_cosine']:.3f} ({rec['regime']})")
        if fa:
            bits.append(f"faith_rho={fa['spearman']:.3f}")
        if m.get("sufficiency") is not None:
            bits.append(f"suff={m['sufficiency']:.3f}")
        print(f"    {name:22s} " + "  ".join(bits))
    print(f"[conceptlab] report: {Path(args.out)/cfg.name/'report.html'}  ·  index: {Path(args.out)/'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
