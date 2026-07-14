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

    import json as _json
    cases_html = ""
    if agg.get("cases"):
        cases_html = _CASE_EXPLORER.replace("%%CASES_JSON%%", _json.dumps(agg["cases"]))

    html = _TEMPLATE.format(
        name=agg["name"], auc=f"{agg['model_auc']:.3f}", n=agg["n"],
        runtime=agg["runtime_s"], n_concepts=len(names),
        scorecard=_scorecard(agg["methods"]), defn_rows=defn_rows, headline=headline,
        fig_bars=fig_bars, fig_hit=fig_hit, fig_audit=fig_audit,
    ).replace("%%CSS%%", _CSS).replace("%%CASE_EXPLORER%%", cases_html)
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
%%CASE_EXPLORER%%
</body></html>
"""

_CASE_EXPLORER = r"""
<h2>Case explorer — drill into individual decisions</h2>
<p class="card-note">The 20 highest-scored events (the model's most confident fraud calls),
badged <b>TP</b> (true fraud, with typology) or <b>FP</b> (actually legitimate). Click a case to
see its per-concept local attributions next to the ground truth; click a concept row to see
<i>which input cells</i> influence that concept — model attribution (IG vs the concept probe,
toggleable to cell occlusion) side by side with the ground-truth causal cells from the concept's
definition. Baselines: numeric = column mean, categorical = modal code. Deep-linkable via
<code>#case=0&amp;concept=cvv_failure</code>.</p>
<style>
.cx { display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap; }
.cx-list { flex:0 0 250px; max-height:560px; overflow-y:auto; border:1px solid rgba(137,135,129,.3);
  border-radius:10px; }
.cx-item { padding:8px 12px; cursor:pointer; border-bottom:1px solid rgba(137,135,129,.18);
  font-size:12.5px; display:flex; gap:8px; align-items:center; }
.cx-item:hover { background:rgba(42,120,214,.08); }
.cx-item.sel { background:rgba(42,120,214,.16); }
.cx-badge { font-size:10px; font-weight:700; padding:1px 7px; border-radius:9px; color:#fff; }
.cx-badge.tp { background:#e34948; } .cx-badge.fp { background:#eda100; color:#1a1a19; }
.cx-typ { color:#898781; font-size:11px; }
.cx-main { flex:1 1 620px; min-width:0; }
.cx-tables { display:flex; gap:16px; flex-wrap:wrap; }
.cx-tables table { font-size:12px; flex:1 1 300px; }
.cx-tables td, .cx-tables th { padding:3px 8px; }
tr.cx-crow { cursor:pointer; } tr.cx-crow:hover td { background:rgba(42,120,214,.10); }
tr.cx-crow.sel td { background:rgba(42,120,214,.20); }
.cx-bar { display:inline-block; height:9px; background:#2a78d6; border-radius:2px;
  vertical-align:middle; margin-right:5px; }
.cx-def1 { color:#008300; font-weight:700; } .cx-def0 { color:#898781; }
.cx-hm { margin-top:18px; display:flex; gap:18px; flex-wrap:wrap; }
.cx-hm .pane { flex:1 1 420px; min-width:0; overflow-x:auto; }
.cx-hm h3 { margin:4px 0 6px; font-size:13.5px; }
.cx-hm table { border-collapse:collapse; font-size:10.5px; font-variant-numeric:tabular-nums; }
.cx-hm td, .cx-hm th { border:1px solid rgba(137,135,129,.22); padding:2px 5px; text-align:right;
  white-space:nowrap; }
.cx-hm th { color:#898781; font-weight:600; }
.cx-hm tr.dec td { border-top:2px solid #e34948; }
.cx-toggle { margin:6px 0; }
.cx-toggle button { font:inherit; font-size:12px; padding:3px 12px; border-radius:14px;
  border:1px solid rgba(137,135,129,.4); background:transparent; color:inherit; cursor:pointer; }
.cx-toggle button.on { border-color:#2a78d6; color:#2a78d6; font-weight:700; }
.cx-note { color:#898781; font-size:11.5px; margin:4px 0; }
</style>
<div class="cx">
  <div class="cx-list" id="cxList"></div>
  <div class="cx-main">
    <div id="cxDetail"><p class="cx-note">Select a case on the left.</p></div>
    <div id="cxHeat"></div>
  </div>
</div>
<script>
const CX = %%CASES_JSON%%;
let selCase = null, selConcept = null, hmMode = "ig";

function hashState() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { c: p.get("case"), k: p.get("concept"), m: p.get("mode") };
}
function setHash() {
  const bits = [];
  if (selCase !== null) bits.push("case=" + selCase);
  if (selConcept) bits.push("concept=" + selConcept);
  if (hmMode !== "ig") bits.push("mode=" + hmMode);
  history.replaceState(null, "", "#" + bits.join("&"));
}

function renderList() {
  const el = document.getElementById("cxList");
  el.innerHTML = CX.cases.map((c, i) => {
    const badge = c.label ? `<span class="cx-badge tp">TP</span>` : `<span class="cx-badge fp">FP</span>`;
    const typ = c.label ? c.typology : "legit";
    return `<div class="cx-item ${i===selCase?'sel':''}" onclick="selectCase(${i})">
      <b>#${i}</b> ${badge} <span>p=${c.score.toFixed(2)}</span> <span class="cx-typ">${typ}</span></div>`;
  }).join("");
}

function selectCase(i) {
  selCase = i; selConcept = null; setHash(); renderList(); renderDetail();
  document.getElementById("cxHeat").innerHTML = "";
}

function renderDetail() {
  const c = CX.cases[selCase];
  const methods = CX.methods;
  // sort concepts by first method's local attribution
  const names = [...CX.concept_names].sort((a, b) =>
    c.concepts[b].methods[methods[0]] - c.concepts[a].methods[methods[0]]);
  const maxv = {};
  methods.forEach(m => { maxv[m] = Math.max(...names.map(n => c.concepts[n].methods[m]), 1e-9); });

  const fmt = v => v === 0 ? "0" : (v >= 0.001 ? v.toFixed(3) : v.toExponential(1));
  const mrows = names.map(n => {
    const cc = c.concepts[n];
    const cells = methods.map(m => {
      const w = Math.round(60 * cc.methods[m] / maxv[m]);
      return `<td><span class="cx-bar" style="width:${w}px"></span>${fmt(cc.methods[m])}</td>`;
    }).join("");
    return `<tr class="cx-crow ${n===selConcept?'sel':''}" onclick="selectConcept('${n}')">
      <td class="name">${n}</td>${cells}</tr>`;
  }).join("");

  const grows = names.map(n => {
    const cc = c.concepts[n];
    const d = cc.gt_defining ? `<span class="cx-def1">1 — defining</span>` : `<span class="cx-def0">0</span>`;
    return `<tr class="cx-crow ${n===selConcept?'sel':''}" onclick="selectConcept('${n}')">
      <td class="name">${n}</td><td>${cc.value}</td><td>${d}</td></tr>`;
  }).join("");

  const gtTitle = c.label
    ? `ground truth — typology <b>${c.typology}</b>`
    : `ground truth — <b>legitimate</b> (false positive: no defining concepts)`;
  document.getElementById("cxDetail").innerHTML = `
    <p class="cx-note">Case #${selCase} · model score ${c.score.toFixed(3)} ·
      ${c.label ? "true fraud (" + c.typology + ")" : "actually legitimate — the explanation shows what fooled the model"}
      · click a concept row for the input heatmap</p>
    <div class="cx-tables">
      <table><thead><tr><th class="name">method attribution (local)</th>
        ${methods.map(m => `<th>${m}</th>`).join("")}</tr></thead><tbody>${mrows}</tbody></table>
      <table><thead><tr><th class="name">${gtTitle}</th><th>concept value</th><th>defining?</th></tr></thead>
        <tbody>${grows}</tbody></table>
    </div>`;
}

function selectConcept(n) { selConcept = n; setHash(); renderDetail(); renderHeat(); }
function setMode(m) { hmMode = m; setHash(); renderHeat(); }

function heatTable(vals, isGT) {
  const T = CX.T, F = CX.fields;
  const c = CX.cases[selCase];
  let h = `<table><thead><tr><th>t</th>${F.map(f => `<th>${f}</th>`).join("")}</tr></thead><tbody>`;
  for (let t = 0; t < T; t++) {
    const dec = (t === T - 1) ? " class=\"dec\"" : "";
    h += `<tr${dec}><th>${t === T - 1 ? "▶ " + t : t}</th>`;
    for (let j = 0; j < F.length; j++) {
      const v = vals[t][j];
      const bg = isGT ? `rgba(0,131,0,${v ? 0.45 : 0})` : `rgba(42,120,214,${(0.85 * v).toFixed(2)})`;
      h += `<td style="background:${bg}">${c.table[t][j]}</td>`;
    }
    h += "</tr>";
  }
  return h + "</tbody></table>";
}

function renderHeat() {
  if (selCase === null || !selConcept) return;
  const cc = CX.cases[selCase].concepts[selConcept];
  const gt = CX.gt_masks[selConcept];
  document.getElementById("cxHeat").innerHTML = `
    <div class="cx-toggle">model attribution for <b>${selConcept}</b>:
      <button class="${hmMode==='ig'?'on':''}" onclick="setMode('ig')">IG vs concept probe</button>
      <button class="${hmMode==='occ'?'on':''}" onclick="setMode('occ')">cell occlusion</button>
    </div>
    <div class="cx-hm">
      <div class="pane"><h3>model: which cells influence the concept</h3>${heatTable(cc[hmMode], false)}</div>
      <div class="pane"><h3>ground truth: cells the definition reads</h3>${heatTable(gt, true)}</div>
    </div>
    <p class="cx-note">Blue intensity = normalized attribution (${hmMode === 'ig' ?
      'integrated gradients of the concept probe along an embedding path from the baseline window' :
      'change in the concept-probe score when the cell is set to its baseline'}).
      Green = the cells this concept's definition of record actually reads. The red-topped row is
      the decision event. ${CX.baseline_note}.</p>`;
}

(function init() {
  renderList();
  const h = hashState();
  if (h.m === "occ") hmMode = "occ";
  if (h.c !== null && CX.cases[+h.c]) { selectCase(+h.c); }
  if (h.k && selCase !== null && CX.concept_names.includes(h.k)) { selectConcept(h.k); }
})();
</script>
"""
