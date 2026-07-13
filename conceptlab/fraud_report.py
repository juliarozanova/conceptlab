"""HTML report for concept attribution on contract (fraud) data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .report import BLUE, AQUA, YELLOW, VIOLET, RED, GREEN, MUTED, _CSS, _div


def _norm(a):
    a = np.abs(np.asarray(a, float))
    m = a.max()
    return a / m if m > 1e-12 else a


def _method_bars(names, gt, methods, first=False):
    fig = go.Figure()
    fig.add_bar(name="defining-concept GT", x=names, y=_norm(gt), marker_color=MUTED)
    colors = [BLUE, AQUA, YELLOW, VIOLET, RED]
    for i, (m, r) in enumerate(methods.items()):
        fig.add_bar(name=m, x=names, y=_norm(r["per_concept"]), marker_color=colors[i % len(colors)])
    fig.update_layout(barmode="group", height=380)
    fig.update_yaxes(title_text="attribution (max-norm)")
    fig.update_xaxes(tickangle=-40)
    return _div(fig, first)


def _hit_heatmap(per_typ, method_names, first=False):
    typs = [t for t in per_typ if per_typ[t] is not None]
    z = np.array([[per_typ[t].get(m, np.nan) for m in method_names] for t in typs])
    fig = go.Figure(go.Heatmap(z=z, x=method_names, y=typs, zmin=0.3, zmax=1.0,
                               colorscale="Blues", colorbar=dict(title="rank-AUC"),
                               text=np.round(z, 2), texttemplate="%{text}",
                               hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(height=90 + 30 * len(typs))
    return _div(fig, first)


def _audit_heatmap(names, audit, first=False):
    hps = audit["hookpoints"]
    z = np.array([audit["auc"][hp] for hp in hps])
    fig = go.Figure(go.Heatmap(z=z, x=names, y=hps, zmin=0.5, zmax=1.0, colorscale="Blues",
                               colorbar=dict(title="AUC"),
                               hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(height=90 + 26 * len(hps))
    fig.update_xaxes(tickangle=-40)
    return _div(fig, first)


def _scorecard(methods):
    return "\n".join(
        f"<tr><td class='name'>{m}</td><td>{r['global_rho']:+.2f}</td></tr>"
        for m, r in methods.items())


def write_fraud_report(agg: dict, out_dir: Path) -> Path:
    names = agg["concept_names"]
    method_names = list(agg["methods"])
    # the per-typology heatmap appears first in the document, so it carries the
    # inlined Plotly.js
    fig_hit = _hit_heatmap(agg["per_typology_hit"], method_names, first=True)
    fig_bars = _method_bars(names, np.array(agg["gt_global"]), agg["methods"])
    fig_audit = _audit_heatmap(names, agg["audit"])

    defn = agg["typology_defining"]
    defn_rows = "".join(
        f"<tr><td class='name'>{t}</td><td>{', '.join(cs) if cs else '<i>none (Bayes floor)</i>'}</td></tr>"
        for t, cs in defn.items())

    # data-driven headline sentence for the per-typology note
    hit = agg["per_typology_hit"]
    ics_scores = {t: d["ics"] for t, d in hit.items() if d and d.get("ics") == d.get("ics")}
    headline = ""
    if ics_scores:
        best = max(ics_scores, key=ics_scores.get)
        worst = min(ics_scores, key=ics_scores.get)
        headline = (f"Here ICS/CAV-ablation are strongest on <b>{best}</b> "
                    f"({ics_scores[best]:.2f}) and weakest on <b>{worst}</b> "
                    f"({ics_scores[worst]:.2f}).")

    html = _TEMPLATE.format(
        name=agg["name"], auc=f"{agg['model_auc']:.3f}", n=agg["n"],
        runtime=agg["runtime_s"], n_concepts=len(names),
        scorecard=_scorecard(agg["methods"]), defn_rows=defn_rows, headline=headline,
        fig_bars=fig_bars, fig_hit=fig_hit, fig_audit=fig_audit,
    ).replace("%%CSS%%", _CSS)
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>conceptlab · fraud attribution</title><style>%%CSS%%</style></head><body>
<h1>conceptlab — concept attribution on fraud data</h1>
<div class="sub">Explaining a trained fraud model's decisions with respect to concepts, graded
against typology-defining ground truth. &nbsp;·&nbsp; <a href="../index.html">methodology</a></div>
<p class="card-note">A TabTransformer is trained on synthetic transactions (from <code>fraudgen</code>,
loaded through the data contract). For each fraud event we know <i>why</i> it is fraud — the
defining concepts of its typology — so we can grade whether an explanation points at the right
concepts and not at correlated-but-noncausal ones (e.g. on_holiday).</p>
<div class="meta">
  <span>fraud model AUC <b>{auc}</b></span>
  <span>sequences <b>{n}</b></span>
  <span>concepts <b>{n_concepts}</b></span>
  <span>{runtime}s</span>
</div>

<h2>Typology → defining concepts (the attribution ground truth)</h2>
<table><tbody>{defn_rows}</tbody></table>

<h2>Per-typology hit-rate — the headline result</h2>
<p class="card-note">Rank-AUC of each method's concept ranking against each typology's defining set.
1.0 = the method ranks exactly that typology's concepts on top. This is the meaningful view for
fraud: the causal methods (ICS, CAV-ablation, probe-patch) recover the right concepts per
typology, and their strength tracks how decodable those concepts are in the model (see the audit
below). {headline} TCAV is the outlier — it over-attributes to nearly every concept (see the bar
chart: its bars dominate regardless of ground truth), so its rankings are weak and its global ρ
can even go negative. first_party has no defining concepts (undetectable — the Bayes floor) and
is omitted.</p>
{fig_hit}

<h2>Global attribution is ill-posed for heterogeneous fraud</h2>
<p class="card-note">Grey = fraud-frequency-weighted defining-concept mask. The ρ below is near zero
even for the good methods — <b>not a method failure but a finding</b>: because different typologies
are driven by different concepts, no single global concept ranking is correct, and a mean over all
fraud events blends incompatible explanations. The transfer lesson for real fraud models: concept
explanations must be local (per case / per typology), never a global feature-importance bar.</p>
<table><thead><tr><th class="name">method</th><th>ρ vs global defining-concept mask</th></tr></thead>
<tbody>{scorecard}</tbody></table>
{fig_bars}

<h2>Concept audit — decodability in the trained model</h2>
<p class="card-note">Linear-probe AUC per concept per layer. Attribution is only meaningful where a
concept is decodable; low-AUC concepts correctly receiving ~0 attribution is a success.</p>
{fig_audit}
</body></html>
"""
