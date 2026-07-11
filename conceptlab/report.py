"""Self-contained HTML reports (Plotly inlined; theme-aware chrome).

Charts follow the project data-viz method: single-series bar charts put method
identity on the axis (no cycled hues); the importance comparison is the only
multi-series chart and carries a legend. All chart chrome uses the muted-gray
ink that is legible on both the light and dark surface, and backgrounds are
transparent so the page's ``prefers-color-scheme`` background shows through.

Validated categorical palette (light-mode hexes, legible on both surfaces):
blue #2a78d6, aqua #1baf7a, yellow #eda100, violet #4a3aa7, red #e34948.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA

# palette (validated in references/palette.md)
BLUE, AQUA, YELLOW, VIOLET, RED, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#008300"
MUTED = "#898781"          # axis/label ink — identical in light & dark
GRID = "rgba(137,135,129,0.20)"

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=MUTED, size=13),
    margin=dict(l=60, r=20, t=10, b=50),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, tickfont=dict(color=MUTED)),
    legend=dict(font=dict(color=MUTED), orientation="h", y=1.12, x=0),
    colorway=[BLUE, AQUA, YELLOW, VIOLET, RED, GREEN],
)


def _div(fig: go.Figure, first: bool) -> str:
    fig.update_layout(**_LAYOUT)
    return pio.to_html(fig, include_plotlyjs=("inline" if first else False),
                       full_html=False, config={"displayModeBar": False})


def _bar(names, values, ylabel, color=BLUE, yrange=None, first=False) -> str:
    fig = go.Figure(go.Bar(
        x=names, y=values, marker_color=color, marker_line_width=0,
        text=[f"{v:.2f}" if v is not None else "" for v in values],
        textposition="outside", textfont=dict(color=MUTED),
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_yaxes(title_text=ylabel, range=yrange)
    fig.update_layout(height=300)
    return _div(fig, first)


def _importance_fig(concept_names, gt, method_scores: dict, first=False) -> str:
    """Grouped bars: ground-truth importance vs each scoring method, per concept."""
    gt = np.asarray(gt, dtype=float)
    order = np.argsort(-gt)[:12]
    xs = [concept_names[i] for i in order]
    fig = go.Figure()
    fig.add_bar(name="ground truth", x=xs, y=_norm(gt)[order], marker_color=MUTED)
    colors = [BLUE, AQUA, YELLOW, VIOLET, RED]
    for k, (mname, sc) in enumerate(method_scores.items()):
        fig.add_bar(name=mname, x=xs, y=_norm(np.asarray(sc, float))[order],
                    marker_color=colors[k % len(colors)])
    fig.update_layout(barmode="group", height=340)
    fig.update_yaxes(title_text="importance (max-normalised)")
    return _div(fig, first)


def _projection_fig(spec, train, res, disc_dirs, disc_name, first=False) -> str:
    world = train.world
    X = train.X.reshape(-1, train.dim) if train.is_sequence else train.X
    y = None if train.is_sequence else train.y
    n = min(1500, len(X))
    idx = np.random.default_rng(0).permutation(len(X))[:n]
    Xs = X[idx]
    pca = PCA(n_components=2).fit(Xs)
    P = pca.transform(Xs)
    comp = pca.components_                       # (2, D)

    fig = go.Figure()
    if y is not None:
        ys = y[idx]
        for lab, col, nm in [(0, BLUE, "label 0"), (1, RED, "label 1")]:
            m = ys == lab
            fig.add_scatter(x=P[m, 0], y=P[m, 1], mode="markers", name=nm,
                            marker=dict(size=5, color=col, opacity=0.45, line=dict(width=0)),
                            hoverinfo="skip")
    else:
        fig.add_scatter(x=P[:, 0], y=P[:, 1], mode="markers", name="events",
                        marker=dict(size=5, color=BLUE, opacity=0.4), hoverinfo="skip")

    scale = 1.15 * np.abs(P).max()
    # true concept anchors projected into the PCA plane
    A = world.anchor_unit() @ comp.T             # (M, 2)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9) * scale
    names = []
    for c in world.concepts:
        names += [c.name] * len(c.dir_index)
    for j in range(len(A)):
        fig.add_scatter(x=[0, A[j, 0]], y=[0, A[j, 1]], mode="lines",
                        line=dict(color=GREEN, width=2),
                        name="true concept" if j == 0 else None,
                        showlegend=(j == 0), hovertext=names[j], hoverinfo="text")
    # discovered directions
    if disc_dirs is not None and len(disc_dirs):
        Dp = disc_dirs @ comp.T
        Dp = Dp / (np.linalg.norm(Dp, axis=1, keepdims=True) + 1e-9) * scale * 0.9
        for j in range(min(len(Dp), 40)):
            fig.add_scatter(x=[0, Dp[j, 0]], y=[0, Dp[j, 1]], mode="lines",
                            line=dict(color=YELLOW, width=1, dash="dot"),
                            name=f"discovered ({disc_name})" if j == 0 else None,
                            showlegend=(j == 0), hoverinfo="skip")
    fig.update_layout(height=460)
    fig.update_xaxes(title_text="PC1", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="PC2")
    return _div(fig, first)


def _norm(a: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(a))
    return a / m if m > 1e-12 else a


def _scorecard_rows(methods: dict) -> str:
    rows = []
    for name, m in methods.items():
        rec = m.get("recovery") or {}
        fa = m.get("faithfulness") or {}
        suff = m.get("sufficiency")
        rows.append(
            f"<tr><td class='name'>{name}</td>"
            f"<td>{_fmt(rec.get('mean_matched_cosine'))}</td>"
            f"<td>{_fmt(rec.get('coverage'))}</td>"
            f"<td>{_fmt(rec.get('redundancy'))}</td>"
            f"<td>{rec.get('regime','—')}</td>"
            f"<td>{_fmt(fa.get('spearman'))}</td>"
            f"<td>{'yes' if fa.get('detects_interaction') else ('—' if not fa else 'no')}</td>"
            f"<td>{_fmt(suff)}</td></tr>"
        )
    return "\n".join(rows)


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "—"


def write_report(agg: dict, first, seed0_detail: dict, out_dir: Path) -> Path:
    spec, train, val, res = first
    methods = agg["methods"]
    concept_names = agg["concept_names"]
    gt = agg["gt_importance"]

    # pick a discovery method to visualise in the projection
    disc_name = next((n for n in ("relu_sae", "pca", "ica", "topk_sae", "probe_skyline")
                      if n in methods and methods[n].get("recovery")), None)
    disc_dirs = None
    if disc_name:
        from .methods import build_context, build_method
        ctx = build_context(res.model, train, seed=0)
        kw = agg["config"].get("method_kwargs", {}).get(disc_name, {})
        disc_dirs = build_method(disc_name, **kw).fit(ctx).discovered_concepts()

    # scoring-method importance vectors from seed-0 detail
    score_methods = {n: d["importance_per_concept"]
                     for n, d in seed0_detail["methods"].items()
                     if d.get("faithfulness") and d.get("importance_per_concept")}

    # figures
    disc_methods = [n for n in methods if methods[n].get("recovery")]
    scor_methods = [n for n in methods if methods[n].get("faithfulness")]
    fig_recov = _bar(disc_methods, [methods[n]["recovery"]["mean_matched_cosine"] for n in disc_methods],
                     "matched cosine", color=BLUE, yrange=[0, 1.08], first=True)
    fig_suff = _bar(disc_methods, [methods[n].get("sufficiency") for n in disc_methods],
                    "probe accuracy", color=AQUA, yrange=[0, 1.08])
    fig_faith = _bar(scor_methods, [methods[n]["faithfulness"]["spearman"] for n in scor_methods],
                     "Spearman ρ vs ground truth", color=VIOLET, yrange=[-1.08, 1.08])
    fig_imp = _importance_fig(concept_names, gt, score_methods)
    fig_proj = _projection_fig(spec, train, res, disc_dirs, disc_name or "—")

    label = agg["label"]
    html = _TEMPLATE.format(
        name=agg["name"],
        label=label,
        model_kind=agg["config"]["model"].get("kind", "mlp"),
        val_acc=f"{agg['val_acc']:.3f}", train_acc=f"{agg['train_acc']:.3f}",
        n_concepts=train.world.n_concepts, dim=train.dim,
        geometry=spec.geometry, seeds=agg["seeds"], runtime=agg["runtime_s"],
        scorecard=_scorecard_rows(methods),
        fig_recov=fig_recov, fig_suff=fig_suff, fig_faith=fig_faith,
        fig_imp=fig_imp, fig_proj=fig_proj,
    ).replace("%%CSS%%", _CSS)
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    update_index(out_dir.parent)
    return path


def update_index(runs_dir: Path) -> Path:
    """Rebuild runs/index.html linking every run with its headline metrics."""
    cards = []
    for mp in sorted(runs_dir.glob("*/metrics.json")):
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        best_rec = max((v["recovery"]["mean_matched_cosine"]
                        for v in m["methods"].values() if v.get("recovery")), default=None)
        cards.append(
            f"<a class='card' href='{mp.parent.name}/report.html'>"
            f"<div class='ct'>{m['name']}</div>"
            f"<div class='cs'>label <code>{m.get('label','')}</code></div>"
            f"<div class='cm'>val acc {m['val_acc']:.3f} &middot; "
            f"best recovery {(_fmt(best_rec))} &middot; {len(m['methods'])} methods</div></a>"
        )
    html = _INDEX_TEMPLATE.format(cards="\n".join(cards) or "<p>No runs yet.</p>").replace("%%CSS%%", _CSS)
    path = runs_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


_CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 32px; background: #f9f9f7; color: #0b0b0b; line-height: 1.5; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 32px 0 8px;
  color: #52514e; border-bottom: 1px solid rgba(137,135,129,0.3); padding-bottom: 4px; }
.sub { color: #898781; font-size: 13px; margin-bottom: 8px; }
code { background: rgba(137,135,129,0.15); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
.meta { display: flex; flex-wrap: wrap; gap: 18px; font-size: 13px; color: #52514e; margin: 12px 0; }
.meta b { color: #0b0b0b; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid rgba(137,135,129,0.2); }
th { color: #898781; font-weight: 600; } td.name, th.name { text-align: left; }
td.name { font-weight: 600; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.card-note { color: #898781; font-size: 12px; margin: 2px 0 0; }
@media (prefers-color-scheme: dark) {
  body { background: #0d0d0d; color: #fff; } h2 { color: #c3c2b7; }
  .meta { color: #c3c2b7; } .meta b { color: #fff; } td.name { color: #fff; }
}
@media (max-width: 780px) { .grid2 { grid-template-columns: 1fr; } }
.card { display:block; text-decoration:none; color:inherit; padding:16px 18px;
  border:1px solid rgba(137,135,129,0.3); border-radius:10px; margin:10px 0; }
.card:hover { border-color:#2a78d6; }
.ct { font-weight:700; font-size:16px; } .cs { color:#898781; font-size:13px; margin:2px 0; }
.cm { color:#52514e; font-size:13px; }
@media (prefers-color-scheme: dark) { .cm { color:#c3c2b7; } }
"""

_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>conceptlab · {name}</title><style>%%CSS%%</style></head><body>
<h1>conceptlab — {name}</h1>
<div class="sub">Interpretability methods scored against known ground truth.</div>
<div class="meta">
  <span>label <code>{label}</code></span>
  <span>model <b>{model_kind}</b></span>
  <span>val acc <b>{val_acc}</b></span>
  <span>concepts <b>{n_concepts}</b></span>
  <span>dim <b>{dim}</b></span>
  <span>geometry <b>{geometry}</b></span>
  <span>seeds <b>{seeds}</b></span>
  <span>{runtime}s</span>
</div>

<h2>Scorecard</h2>
<table><thead><tr>
  <th class="name">method</th><th>recovery cos</th><th>coverage</th><th>redundancy</th>
  <th>regime</th><th>faithfulness ρ</th><th>interaction?</th><th>sufficiency</th>
</tr></thead><tbody>{scorecard}</tbody></table>
<p class="card-note">recovery cos: Hungarian-matched cosine to true concepts · coverage/redundancy/regime:
SAE capture regime · faithfulness ρ: Spearman vs generator causal importance · sufficiency: label probe accuracy on discovered concepts.</p>

<div class="grid2">
  <div><h2>Concept recovery</h2>{fig_recov}</div>
  <div><h2>Sufficiency (probe accuracy)</h2>{fig_suff}</div>
</div>
<div class="grid2">
  <div><h2>Importance faithfulness</h2>{fig_faith}</div>
  <div><h2>Per-concept importance</h2>{fig_imp}</div>
</div>

<h2>Embedding space — true vs discovered concepts</h2>
<p class="card-note">PCA projection of the embeddings. Green solid = true concept anchors;
yellow dotted = discovered directions. Points colored by label.</p>
{fig_proj}
</body></html>
"""

_INDEX_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>conceptlab · runs</title><style>%%CSS%%</style></head><body>
<h1>conceptlab — runs</h1>
<div class="sub">Cross-run comparison. Click a run for its full report.</div>
{cards}
</body></html>
"""
