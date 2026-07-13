"""Self-contained HTML report for a tabular-concept experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .report import (BLUE, AQUA, YELLOW, VIOLET, RED, GREEN, MUTED, _CSS, _div)


def _bars_gt(names, interv, shap, support, first=False):
    fig = go.Figure()
    fig.add_bar(name="interventional", x=names, y=_norm(interv), marker_color=BLUE)
    fig.add_bar(name="Shapley", x=names, y=_norm(shap), marker_color=AQUA)
    # mark non-support concepts
    ns = [names[i] for i, s in enumerate(support) if not s]
    for nm in ns:
        fig.add_annotation(x=nm, y=1.02, text="✕ not causal", showarrow=False,
                           font=dict(color=RED, size=10), yref="y")
    fig.update_layout(barmode="group", height=320)
    fig.update_yaxes(title_text="importance (max-norm)", range=[0, 1.15])
    return _div(fig, first)


def _method_bars(names, gt_interv, method_scores: dict, support, first=False):
    order = list(range(len(names)))
    fig = go.Figure()
    fig.add_bar(name="ground truth", x=names, y=_norm(np.array(gt_interv)), marker_color=MUTED)
    colors = [BLUE, AQUA, YELLOW, VIOLET, RED, GREEN]
    for i, (m, sc) in enumerate(method_scores.items()):
        fig.add_bar(name=m, x=names, y=_norm(np.array(sc["per_concept"])),
                    marker_color=colors[i % len(colors)])
    fig.update_layout(barmode="group", height=340)
    fig.update_yaxes(title_text="attribution (max-norm)")
    return _div(fig, first)


def _audit_heatmap(names, audit, first=False):
    hps = audit["hookpoints"]
    z = np.array([audit["auc"][hp] for hp in hps])   # (n_hp, K)
    fig = go.Figure(go.Heatmap(
        z=z, x=names, y=hps, zmin=0.5, zmax=1.0, colorscale="Blues",
        colorbar=dict(title="AUC"), hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(height=90 + 26 * len(hps))
    return _div(fig, first)


def _norm(a):
    a = np.abs(np.asarray(a, float))
    m = a.max()
    return a / m if m > 1e-12 else a


def _scorecard(scores: dict, support_names) -> str:
    rows = []
    for m, s in scores.items():
        rows.append(
            f"<tr><td class='name'>{m}</td>"
            f"<td>{s['rho_interventional']:+.2f}</td>"
            f"<td>{s['rho_shapley']:+.2f}</td>"
            f"<td>{s['support_leakage']:.2f}</td>"
            f"<td>{s['pivotal_gate']:.3f}</td></tr>")
    return "\n".join(rows)


def write_concept_report(agg: dict, first, cfg, out_dir: Path) -> Path:
    names = agg["concept_names"]
    fig_gt = _bars_gt(names, agg["gt_interventional"], agg["gt_shapley"], agg["support_mask"], first=True)
    fig_mA = _method_bars(names, agg["gt_interventional"], agg["modeA_scores"], agg["support_mask"])
    fig_mB = _method_bars(names, agg["gt_interventional"], agg["modeB_scores"], agg["support_mask"])
    fig_audit = _audit_heatmap(names, agg["audit"])

    html = _TEMPLATE.format(
        name=agg["name"], description=agg["description"],
        modeA_acc=f"{agg['modeA_val_acc']:.3f}", modeB_acc=f"{agg['modeB_val_acc']:.3f}",
        def_gap=f"{agg['definition_gap']:.2f}", tau=agg["tau"], seeds=agg["seeds"],
        runtime=agg["runtime_s"], n_concepts=len(names),
        scorecard_A=_scorecard(agg["modeA_scores"], names),
        scorecard_B=_scorecard(agg["modeB_scores"], names),
        fig_gt=fig_gt, fig_mA=fig_mA, fig_mB=fig_mB, fig_audit=fig_audit,
    ).replace("%%CSS%%", _CSS)
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    _update_index(out_dir.parent)
    return path


def _update_index(runs_dir: Path):
    import json
    cards = []
    for mp in sorted(runs_dir.glob("*/metrics.json")):
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        cards.append(
            f"<a class='card' href='{mp.parent.name}/report.html'>"
            f"<div class='ct'>{m['name']}</div>"
            f"<div class='cs'>{m['description'][:130]}…</div>"
            f"<div class='cm'>mode A {m['modeA_val_acc']:.2f} · mode B {m['modeB_val_acc']:.2f} · "
            f"def-gap {m['definition_gap']:.2f}</div></a>")
    html = _INDEX.format(cards="\n".join(cards) or "<p>No runs.</p>").replace("%%CSS%%", _CSS)
    (runs_dir / "index.html").write_text(html, encoding="utf-8")


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>conceptlab · {name}</title><style>%%CSS%%</style></head><body>
<h1>conceptlab — {name}</h1>
<div class="sub">Compositional concept attribution vs known ground truth.
&nbsp;·&nbsp; <a href="index.html">all concept runs</a>
&nbsp;·&nbsp; <a href="../index.html">methodology</a></div>
<p class="card-note">{description}</p>
<div class="meta">
  <span>mode A (glass box) acc <b>{modeA_acc}</b></span>
  <span>mode B (from scratch) acc <b>{modeB_acc}</b></span>
  <span>concepts <b>{n_concepts}</b></span>
  <span>definition gap <b>{def_gap}</b></span>
  <span>τ <b>{tau}</b></span>
  <span>seeds <b>{seeds}</b></span>
  <span>{runtime}s</span>
</div>

<h2>Ground truth — the two definitions</h2>
<p class="card-note">Per-concept causal importance from the exact label function. Interventional =
single-concept toggle magnitude; Shapley = average marginal contribution. Where they disagree
(large definition gap) the label is overdetermined and ranking is not uniquely defined. Concepts
marked ✕ are structurally irrelevant — faithful methods must give them ~0.</p>
{fig_gt}

<h2>Mode A — glass box (soft logic oracle, hookpoint = rep)</h2>
<p class="card-note">The representation is a known function of the concepts, so CAVs are exact. This
isolates the attribution mechanism from CAV-estimation error. ρ = Spearman of the method's
per-concept attribution vs ground truth; leakage = attribution mass on non-causal concepts;
pivotal-gate = attribution on never-pivotal concepts (lower is better).</p>
<table><thead><tr><th class="name">method</th><th>ρ interventional</th><th>ρ Shapley</th>
<th>support leakage</th><th>pivotal gate</th></tr></thead><tbody>{scorecard_A}</tbody></table>
{fig_mA}

<h2>Mode B — trained from scratch (TabTransformer, hookpoint = resid_post_L0)</h2>
<p class="card-note">The model learned the label from raw events and was never told about concepts;
CAVs are fitted on its activations. Attribution quality is now bounded by how well each concept is
represented — see the audit below.</p>
<table><thead><tr><th class="name">method</th><th>ρ interventional</th><th>ρ Shapley</th>
<th>support leakage</th><th>pivotal gate</th></tr></thead><tbody>{scorecard_B}</tbody></table>
{fig_mB}

<h2>Concept audit — is each concept even in the model?</h2>
<p class="card-note">Linear-probe AUC for each concept at each hookpoint of the trained model.
Attribution to a concept is only meaningful where it is decodable; a low-AUC concept that gets ~0
attribution is a correct result, not a method failure.</p>
{fig_audit}
</body></html>
"""

_INDEX = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>conceptlab · concept runs</title><style>%%CSS%%</style></head><body>
<h1>conceptlab — concept attribution runs</h1>
<div class="sub">Compositional interpretability vs known ground truth.
&nbsp;·&nbsp; <a href="../index.html">methodology</a></div>
{cards}
</body></html>
"""
