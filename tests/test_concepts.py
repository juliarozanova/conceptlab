"""Tests for the tabular-concepts machinery: DSL, soft logic, resimulate, GT."""

import numpy as np
import torch

from conceptlab.concepts import (Concept, ConceptGraph, GT, LT, EQ, IN_SET, COUNT_W,
                                  EPISODE, SoftContext, A, ref, AND, OR, XOR, NOT)
from conceptlab.concept_models import make_concept_dataset
from conceptlab.concept_eval import compute_ground_truth
from conceptlab.tabular import (TableSpec, NumericCol, CategoricalCol, LatentSpec, SequenceWorld)


def _world():
    spec = TableSpec(
        numeric=[NumericCol("amount", base_mean=4.0, base_std=1.0),
                 NumericCol("dt", base_mean=0.8, base_std=0.5)],
        categorical=[CategoricalCol("channel", 3),
                     CategoricalCol("country", 6, latent_dist={"on_trip": [0.05, 0.05, 0.225, 0.225, 0.225, 0.225]})],
        latents=[LatentSpec("on_trip", p_start=0.1, p_stop=0.2)], seq_len=10, seed=0)
    return SequenceWorld(spec)


def _graph():
    return ConceptGraph([
        Concept("amount_gt", 0, A(GT("amount", 4.5))),
        Concept("cnp", 0, A(EQ("channel", 0))),
        Concept("foreign", 0, A(IN_SET("country", {2, 3, 4, 5}))),
        Concept("burst", 1, A(COUNT_W(LT("dt", 0.5), window=5, k=3))),
        Concept("on_holiday", 1, A(EPISODE("on_trip"))),
    ], label=OR(AND(ref("burst"), ref("cnp")), AND(ref("foreign"), ref("amount_gt"))))


def test_soft_logic_matches_hard_at_low_tau():
    w, g = _world(), _graph()
    b = w.sample(300)
    C = g.concept_matrix_hard(b)
    ctx = SoftContext(numeric={k: torch.tensor(v, dtype=torch.float32) for k, v in b.numeric.items()},
                      categorical=b.categorical, latents=b.latents, tau=1e-3)
    Cs = (g.concept_matrix_soft(ctx).detach().numpy() > 0.5).astype(int)
    assert np.array_equal(Cs, C.astype(int))
    ys = (g.label_soft(ctx).detach().numpy() > 0.5).astype(int)
    assert np.array_equal(ys, g.label_hard(b))


def test_transitive_support_excludes_distractor():
    g = _graph()
    sup = set(g.transitive_support())
    assert {"amount_gt", "cnp", "foreign", "burst"} <= sup
    assert "on_holiday" not in sup           # distractor, not in the label


def test_resimulate_reuses_noise():
    w = _world()
    b = w.sample(200)
    b0 = w.resimulate(b, "on_trip", 0)
    # amount has no latent shift -> identical; country changes only where trip was on
    assert np.allclose(b0.numeric["amount"], b.numeric["amount"])
    off = b.latents["on_trip"] == 0
    assert np.array_equal(b0.categorical["country"][off], b.categorical["country"][off])


def test_override_gives_causal_label_effect():
    w, g = _world(), _graph()
    b = w.sample(500)
    C = g.concept_matrix_hard(b)
    names = g.names
    base = g.label_under_override(b, {n: C[:, i].astype(float) for i, n in enumerate(names)})
    assert np.array_equal(base, g.label_hard(b))
    # forcing foreign and amount_gt on should raise the label rate
    ov = {n: C[:, i].astype(float) for i, n in enumerate(names)}
    ov["foreign"] = np.ones(b.n)
    ov["amount_gt"] = np.ones(b.n)
    forced = g.label_under_override(b, ov)
    assert forced.mean() >= base.mean()


def test_ground_truth_distractor_is_zero():
    w, g = _world(), _graph()
    ds = make_concept_dataset(w, g, 1200, dim=16)
    gt = compute_ground_truth(ds)
    k = g.names.index("on_holiday")
    assert gt.interventional[k] == 0.0
    assert not gt.support_mask[k]
    # a causal concept has nonzero importance
    assert gt.interventional[g.names.index("amount_gt")] > 0
