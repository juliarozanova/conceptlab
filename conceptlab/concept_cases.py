"""Per-case drill-down for the tabular-concept report.

Assembles, for a handful of informative validation examples, the data the case
explorer needs to compare **Mode A (glass box)** and **Mode B (from scratch)**
side by side:

* the raw event-sequence table (numeric formatted, categorical as codes);
* per-concept **local** attribution from every per-example method (ICS first)
  in *both* modes;
* the **local ground truth** for that example — each concept's true value, its
  structural support flag, and whether it is **pivotal** here (toggling it flips
  the label). Pivotal is the per-case analogue of the fraud "defining" mask: it
  is exactly the set of concepts a faithful local explanation should point at.

Case selection is deterministic and picks for variety: the model's most
confident calls, confident disagreements between the modes, locally
overdetermined examples (≥2 pivotal concepts), and clean single-cause examples.
"""

from __future__ import annotations

import numpy as np

from .concepts import EventBatch


# ---------------------------------------------------------------------------
# case selection
# ---------------------------------------------------------------------------


def select_cases(y: np.ndarray, pA: np.ndarray, pB: np.ndarray,
                 pivotal: np.ndarray, n: int = 16) -> list[int]:
    """Deterministic, variety-seeking pick of example indices into the val set."""
    n_piv = pivotal.sum(axis=1)
    picked: list[int] = []

    def take(order, k, cond=None):
        c = 0
        for i in order:
            i = int(i)
            if cond is not None and not cond(i):
                continue
            if i not in picked:
                picked.append(i)
                c += 1
                if c >= k:
                    break

    pos = np.argsort(-pB)                                   # confident B calls
    take(pos, 5, lambda i: y[i] == 1)                      # true positives
    take(np.argsort(-pB), 3, lambda i: y[i] == 0)          # confident false positives
    take(np.argsort(-np.abs(pA - pB)), 3)                  # A/B disagreements
    take(np.argsort(-n_piv), 3, lambda i: n_piv[i] >= 2)   # locally overdetermined
    take(pos, 2, lambda i: n_piv[i] == 1)                  # clean single-cause
    take(pos, n)                                            # fill
    return picked[:n]


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------


def _fmt_num(col: str, v: float) -> str:
    return f"{v:.2f}"


def _display_table(batch: EventBatch, gi: int, numeric_fields, categorical_fields) -> list:
    T = batch.T
    rows = []
    for t in range(T):
        row = []
        for f in numeric_fields:
            row.append(_fmt_num(f, float(batch.numeric[f][gi, t])))
        for f in categorical_fields:
            row.append(str(int(batch.categorical[f][gi, t])))
        rows.append(row)
    return rows


def _g(v):
    return float(f"{float(v):.3g}")


def build_concept_cases(graph, va, gt, modeA: dict, modeB: dict,
                        n_cases: int = 16) -> dict:
    """Assemble the JSON payload for the two-mode case explorer.

    ``modeA``/``modeB`` are the dicts returned by the mode runners; each must
    carry ``per_example`` (method -> (N, K) local attributions) and ``p`` (the
    model's P(label=1) over the val batch).
    """
    names = graph.names
    K = len(names)
    numeric_fields = list(va.batch.numeric)
    categorical_fields = list(va.batch.categorical)
    fields = numeric_fields + categorical_fields

    pA, pB = np.asarray(modeA["p"]), np.asarray(modeB["p"])
    C = va.C
    y = va.y
    piv = gt.pivotal
    support = gt.support_mask
    interv = gt.interventional

    idx = select_cases(y, pA, pB, piv, n=n_cases)
    methodsA = list(modeA["per_example"])
    methodsB = list(modeB["per_example"])

    cases = []
    for gi in idx:
        concepts = {}
        for k, nm in enumerate(names):
            concepts[nm] = {
                "value": int(C[gi, k] > 0.5),
                "pivotal": int(piv[gi, k]),
                "support": int(bool(support[k])),
                "interventional": _g(interv[k]),
                "A": {m: _g(modeA["per_example"][m][gi, k]) for m in methodsA},
                "B": {m: _g(modeB["per_example"][m][gi, k]) for m in methodsB},
            }
        cases.append({
            "idx": int(gi),
            "label": int(y[gi]),
            "n_pivotal": int(piv[gi].sum()),
            "pivotal_concepts": [names[k] for k in range(K) if piv[gi, k]],
            "p": {"A": round(float(pA[gi]), 3), "B": round(float(pB[gi]), 3)},
            "table": _display_table(va.batch, gi, numeric_fields, categorical_fields),
            "concepts": concepts,
        })

    return {
        "fields": fields,
        "numeric_fields": numeric_fields,
        "categorical_fields": categorical_fields,
        "T": va.batch.T,
        "concept_names": names,
        "levels": {c.name: c.level for c in graph.concepts.values()},
        "methods": {"A": methodsA, "B": methodsB},
        "cases": cases,
        "note": ("pivotal = toggling this concept flips the label on THIS example "
                 "(the local ground truth a faithful explanation must match)."),
    }
