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


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---------------------------------------------------------------------------
# causal graph (inline SVG) + definitions
# ---------------------------------------------------------------------------


def _causal_svg(g: dict) -> str:
    """Inline SVG of the pure-mediation chain: inputs → concepts (by level) → label.

    Support concepts are filled; structurally-irrelevant distractors are dashed
    and muted; the label node is red. Self-contained and theme-neutral (palette
    colors legible on both surfaces; secondary ink is the shared muted gray)."""
    inputs = g["input_columns"]
    concepts = g["concepts"]
    levels = sorted({c["level"] for c in concepts})
    # columns: inputs | concepts grouped by level | label
    columns = [("inputs", [{"id": c, "text": c, "kind": "input"} for c in inputs])]
    for lv in levels:
        nodes = [{"id": c["name"], "text": c["name"], "kind": "concept",
                  "support": c["support"], "level": lv}
                 for c in concepts if c["level"] == lv]
        columns.append((f"L{lv}", nodes))
    columns.append(("label", [{"id": "__label__", "text": "label", "kind": "label",
                               "expr": g["label_expr"]}]))

    box_w, box_h, row_gap, margin, gap = 150, 44, 60, 24, 88
    lab_w = max(box_w, 26 + 7 * len(g["label_expr"]))
    # width-aware column placement so a wide label box always clears its neighbour
    widths = [lab_w if title == "label" else box_w for title, _ in columns]
    col_cx, x = [], margin
    for w in widths:
        col_cx.append(x + w / 2)
        x += w + gap
    W = x - gap + margin
    n_rows = max(len(nodes) for _, nodes in columns)
    H = margin * 2 + n_rows * row_gap

    pos = {}
    for i, (_, nodes) in enumerate(columns):
        cx = col_cx[i]
        m = len(nodes)
        y0 = (H - m * row_gap) / 2 + row_gap / 2
        for j, nd in enumerate(nodes):
            cy = y0 + j * row_gap
            pos[nd["id"]] = (cx, cy, nd)

    # edges first (under the boxes)
    edge_svg = []
    for e in g["edges"]:
        if e["src"] not in pos or e["dst"] not in pos:
            continue
        x1, y1, _ = pos[e["src"]]
        x2, y2, _ = pos[e["dst"]]
        sx = x1 + box_w / 2
        ex = x2 - (lab_w if e["dst"] == "__label__" else box_w) / 2
        mx = (sx + ex) / 2
        if e["kind"] == "input":
            col, w, op = MUTED, 1.0, 0.5
        elif e["kind"] == "label":
            col, w, op = BLUE, 2.0, 0.9
        else:
            col, w, op = MUTED, 1.4, 0.7
        edge_svg.append(
            f'<path d="M{sx:.0f},{y1:.0f} C{mx:.0f},{y1:.0f} {mx:.0f},{y2:.0f} {ex:.0f},{y2:.0f}" '
            f'fill="none" stroke="{col}" stroke-width="{w}" opacity="{op}"/>')

    node_svg = []
    for nid, (cx, cy, nd) in pos.items():
        w = lab_w if nd["kind"] == "label" else box_w
        x, y = cx - w / 2, cy - box_h / 2
        if nd["kind"] == "input":
            fill, stroke, ink, dash, sub = "transparent", MUTED, MUTED, "", "input"
        elif nd["kind"] == "label":
            fill, stroke, ink, dash, sub = RED, RED, "#fff", "", _esc(nd["expr"])
        elif nd.get("support"):
            fill, stroke, ink, dash, sub = BLUE, BLUE, "#fff", "", f"L{nd['level']} · causal"
        else:
            fill, stroke, ink, dash, sub = "transparent", MUTED, MUTED, "6 3", f"L{nd['level']} · ✕ not causal"
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        node_svg.append(
            f'<g><rect x="{x:.0f}" y="{y:.0f}" rx="9" width="{w:.0f}" height="{box_h}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash_attr}/>'
            f'<text x="{cx:.0f}" y="{cy-3:.0f}" text-anchor="middle" '
            f'font-size="13" font-weight="700" fill="{ink}" '
            f'font-family="system-ui,sans-serif">{_esc(nd["text"])}</text>'
            f'<text x="{cx:.0f}" y="{cy+12:.0f}" text-anchor="middle" font-size="9.5" '
            f'fill="{ink}" opacity="0.85" font-family="system-ui,sans-serif">{sub}</text></g>')

    # column headers
    head_svg = []
    for i, (title, _) in enumerate(columns):
        cx = col_cx[i]
        head_svg.append(
            f'<text x="{cx:.0f}" y="14" text-anchor="middle" font-size="10.5" '
            f'fill="{MUTED}" font-weight="600" letter-spacing="0.5" '
            f'font-family="system-ui,sans-serif">{_esc(title.upper())}</text>')

    return (f'<div style="overflow-x:auto"><svg viewBox="0 0 {W:.0f} {H+20:.0f}" '
            f'width="100%" style="max-width:{W:.0f}px;min-width:520px" '
            f'font-family="system-ui,sans-serif">'
            + "".join(head_svg) + "".join(edge_svg) + "".join(node_svg) + "</svg></div>")


def _definitions_table(g: dict) -> str:
    rows = []
    for c in g["concepts"]:
        tag = ("<span class='cx-def1'>causal</span>" if c["support"]
               else "<span class='cx-def0'>✕ distractor</span>")
        rows.append(
            f"<tr><td class='name'>{_esc(c['name'])}</td><td>L{c['level']}</td>"
            f"<td class='mono'>{_esc(c['expr'])}</td>"
            f"<td>{', '.join(_esc(x) for x in c['inputs']) or '—'}</td><td>{tag}</td></tr>")
    label_row = (f"<tr><td class='name'>label</td><td>—</td>"
                 f"<td class='mono'>{_esc(g['label_expr'])}</td><td>—</td>"
                 f"<td><span class='cx-def1'>decision</span></td></tr>")
    return (f"<table><thead><tr><th class='name'>concept</th><th>level</th>"
            f"<th class='name'>definition</th><th class='name'>inputs</th><th>role</th>"
            f"</tr></thead><tbody>{''.join(rows)}{label_row}</tbody></table>")


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

    graph = agg.get("graph") or (first["tr"].graph.structural_export())
    svg_dag = _causal_svg(graph)
    defs_table = _definitions_table(graph)

    import json as _json
    cases_html = ""
    if agg.get("cases"):
        cases_html = _CASE_EXPLORER.replace("%%CASES_JSON%%", _json.dumps(agg["cases"]))

    html = _TEMPLATE.format(
        name=agg["name"], description=agg["description"],
        modeA_acc=f"{agg['modeA_val_acc']:.3f}", modeB_acc=f"{agg['modeB_val_acc']:.3f}",
        def_gap=f"{agg['definition_gap']:.2f}", tau=agg["tau"], seeds=agg["seeds"],
        runtime=agg["runtime_s"], n_concepts=len(names),
        label_expr=_esc(graph["label_expr"]),
        svg_dag=svg_dag, defs_table=defs_table,
        scorecard_A=_scorecard(agg["modeA_scores"], names),
        scorecard_B=_scorecard(agg["modeB_scores"], names),
        fig_gt=fig_gt, fig_mA=fig_mA, fig_mB=fig_mB, fig_audit=fig_audit,
    ).replace("%%CSS%%", _CSS).replace("%%CASE_EXPLORER%%", cases_html)
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
<title>conceptlab · {name}</title><style>%%CSS%%</style>
<style>
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
.cx-def1 {{ color:#008300; font-weight:700; }} .cx-def0 {{ color:#898781; }}
td.mono {{ white-space: nowrap; }}
</style></head><body>
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

<h2>Causal graph — inputs → concepts → label</h2>
<p class="card-note">The data-generating chain is pure mediation: the label depends on the inputs
<i>only</i> through concepts. Filled blue nodes are concepts the label actually depends on
(transitive support); dashed nodes are structurally-irrelevant distractors a faithful method must
give ~0. Blue edges into the red label node are the causal concept→label references.</p>
{svg_dag}
<p class="card-note" style="margin-top:8px">label := <code>{label_expr}</code></p>

<h2>Ground-truth definitions</h2>
<p class="card-note">Each concept's definition of record (the differentiable soft-logic expression),
its input columns, and whether it is causal for the label. This is the ground truth the attribution
methods are graded against — both globally (below) and per case (the case explorer).</p>
{defs_table}

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
%%CASE_EXPLORER%%
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


_CASE_EXPLORER = r"""
<h2>Case explorer — Mode A vs Mode B, per decision</h2>
<p class="card-note">A handful of validation events, chosen for variety (the models' most confident
calls, A/B disagreements, locally overdetermined examples, and clean single-cause ones). Click a
case: its per-concept <b>local</b> attribution is shown for <b>Mode A (glass box)</b> and
<b>Mode B (from scratch)</b> side by side, method by method — <b>ICS</b> first. The
<span class="cx-def1">green</span> rows are the concepts that are <b>pivotal here</b>: toggling one
flips the label on this very example. That per-case pivotal set is the local ground truth an
explanation should point at — the glass-box oracle should hit it exactly, and the gap to Mode B is
the learned-representation cost. Deep-linkable via <code>#case=0</code>.</p>
<style>
.cx { display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap; }
.cx-list { flex:0 0 240px; max-height:620px; overflow-y:auto; border:1px solid rgba(137,135,129,.3);
  border-radius:10px; }
.cx-item { padding:8px 12px; cursor:pointer; border-bottom:1px solid rgba(137,135,129,.18);
  font-size:12.5px; }
.cx-item:hover { background:rgba(42,120,214,.08); }
.cx-item.sel { background:rgba(42,120,214,.16); }
.cx-l1 { display:flex; gap:7px; align-items:center; }
.cx-l2 { color:#898781; font-size:11px; margin-top:2px; }
.cx-badge { font-size:10px; font-weight:700; padding:1px 7px; border-radius:9px; color:#fff; }
.cx-badge.tp { background:#e34948; } .cx-badge.fp { background:#eda100; color:#1a1a19; }
.cx-main { flex:1 1 640px; min-width:0; }
.cx-note { color:#898781; font-size:11.5px; margin:4px 0; }
.cx-attr { overflow-x:auto; }
.cx-attr table { border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums;
  min-width:100%; }
.cx-attr th, .cx-attr td { padding:3px 8px; border-bottom:1px solid rgba(137,135,129,.18);
  text-align:right; white-space:nowrap; }
.cx-attr td.name, .cx-attr th.name { text-align:left; }
.cx-attr tr.piv td { background:rgba(0,131,0,.09); }
.cx-attr tr.piv td.name { font-weight:700; }
.cx-attr .grp { text-align:center; font-size:10.5px; letter-spacing:.4px; color:#898781;
  border-bottom:2px solid rgba(137,135,129,.35); }
.cx-attr .mA { border-left:2px solid rgba(42,120,214,.35); }
.cx-attr .mB { border-left:2px solid rgba(74,58,167,.35); }
.cx-attr td.ics { background:rgba(42,120,214,.06); }
.cx-bar { display:inline-block; height:8px; background:#2a78d6; border-radius:2px;
  vertical-align:middle; margin-right:5px; }
.cx-bar.b { background:#4a3aa7; }
.cx-ev { border-collapse:collapse; font-size:10.5px; font-variant-numeric:tabular-nums; margin-top:6px; }
.cx-ev td, .cx-ev th { border:1px solid rgba(137,135,129,.22); padding:2px 6px; text-align:right;
  white-space:nowrap; }
.cx-ev th { color:#898781; font-weight:600; } .cx-ev tr.dec td { border-top:2px solid #e34948; }
details { margin-top:12px; } summary { cursor:pointer; color:#898781; font-size:12px; }
</style>
<div class="cx">
  <div class="cx-list" id="cxList"></div>
  <div class="cx-main"><div id="cxDetail"><p class="cx-note">Select a case on the left.</p></div></div>
</div>
<script>
const CX = %%CASES_JSON%%;
let selCase = null;

function badge(c){ return c.label ? '<span class="cx-badge tp">fraud</span>'
  : '<span class="cx-badge fp">legit</span>'; }

function renderList(){
  document.getElementById("cxList").innerHTML = CX.cases.map((c,i)=>{
    const piv = c.pivotal_concepts.length ? c.pivotal_concepts.join(", ") : "none pivotal";
    return `<div class="cx-item ${i===selCase?'sel':''}" onclick="selectCase(${i})">
      <div class="cx-l1"><b>#${i}</b> ${badge(c)}
        <span>pA ${c.p.A.toFixed(2)} · pB ${c.p.B.toFixed(2)}</span></div>
      <div class="cx-l2">pivotal: ${piv}</div></div>`;
  }).join("");
}

function selectCase(i){
  selCase = i;
  history.replaceState(null,"","#case="+i);
  renderList(); renderDetail();
}

function fmt(v){ return v===0 ? "0" : (Math.abs(v)>=0.001 ? v.toFixed(3) : v.toExponential(1)); }

function renderDetail(){
  const c = CX.cases[selCase];
  const mA = CX.methods.A, mB = CX.methods.B;
  // order concepts: pivotal first, then by Mode A ICS (fallback first method)
  const key = mA.includes("ics") ? "ics" : mA[0];
  const names = [...CX.concept_names].sort((a,b)=>{
    const pa=c.concepts[a].pivotal, pb=c.concepts[b].pivotal;
    if(pa!==pb) return pb-pa;
    return (c.concepts[b].A[key]||0)-(c.concepts[a].A[key]||0);
  });
  // per-column max for bar scaling
  const maxA={}, maxB={};
  mA.forEach(m=> maxA[m]=Math.max(...names.map(n=>c.concepts[n].A[m]||0),1e-9));
  mB.forEach(m=> maxB[m]=Math.max(...names.map(n=>c.concepts[n].B[m]||0),1e-9));

  const cell=(v,max,mode,m)=>{
    const w=Math.round(52*(v||0)/max);
    const ics = m==="ics" ? " ics":"";
    return `<td class="${mode==='A'?'':''}${ics}"><span class="cx-bar ${mode==='B'?'b':''}"
      style="width:${w}px"></span>${fmt(v||0)}</td>`;
  };
  const rows = names.map(n=>{
    const cc=c.concepts[n];
    const gt = cc.pivotal ? '<span class="cx-def1">pivotal</span>'
      : (cc.support ? '<span class="cx-def0">causal</span>'
                    : '<span class="cx-def0">✕ distractor</span>');
    const aCells = mA.map((m,j)=>cell(cc.A[m], maxA[m], 'A', m).replace('<td class="',
      '<td class="'+(j===0?'mA ':'')) ).join("");
    const bCells = mB.map((m,j)=>cell(cc.B[m], maxB[m], 'B', m).replace('<td class="',
      '<td class="'+(j===0?'mB ':'')) ).join("");
    return `<tr class="${cc.pivotal?'piv':''}"><td class="name">${n}</td>
      <td>${cc.value}</td><td>${gt}</td><td>${cc.interventional}</td>
      ${aCells}${bCells}</tr>`;
  }).join("");

  const grp = (cls,label,span)=>`<th class="grp ${cls}" colspan="${span}">${label}</th>`;
  const sub = (arr,cls)=>arr.map((m,j)=>`<th class="${j===0?cls:''}">${m}</th>`).join("");

  document.getElementById("cxDetail").innerHTML = `
    <p class="cx-note">Case #${selCase} · event ${c.idx} · ${c.label?'true fraud':'legitimate'} ·
      Mode A P(fraud) <b>${c.p.A.toFixed(3)}</b> · Mode B P(fraud) <b>${c.p.B.toFixed(3)}</b> ·
      ${c.n_pivotal} pivotal concept(s) here</p>
    <div class="cx-attr"><table>
      <thead>
        <tr><th class="name"></th>${grp('','local ground truth',3)}
          ${grp('mA','Mode A — glass box',mA.length)}${grp('mB','Mode B — from scratch',mB.length)}</tr>
        <tr><th class="name">concept</th><th>value</th><th>role</th><th>interv</th>
          ${sub(mA,'mA')}${sub(mB,'mB')}</tr>
      </thead><tbody>${rows}</tbody></table></div>
    <details><summary>event sequence (${CX.T} events × ${CX.fields.length} fields; ▶ = decision)</summary>
      ${eventTable(c)}</details>`;
}

function eventTable(c){
  let h=`<table class="cx-ev"><thead><tr><th>t</th>${CX.fields.map(f=>`<th>${f}</th>`).join("")}</tr></thead><tbody>`;
  for(let t=0;t<CX.T;t++){
    const dec = t===CX.T-1 ? ' class="dec"':'';
    h+=`<tr${dec}><th>${t===CX.T-1?'▶ '+t:t}</th>${c.table[t].map(v=>`<td>${v}</td>`).join("")}</tr>`;
  }
  return h+"</tbody></table>";
}

(function init(){
  renderList();
  const p = new URLSearchParams(location.hash.slice(1));
  const c = p.get("case");
  if(c!==null && CX.cases[+c]) selectCase(+c); else selectCase(0);
})();
</script>
"""
