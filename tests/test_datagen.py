"""Ground-truth must be trustworthy before anything is built on it."""

import numpy as np
import pytest

from conceptlab.datagen import ConceptSpec, GroupSpec, World, make_dataset
from conceptlab.labeldsl import LabelFormula


# ---- label DSL truth tables ------------------------------------------------


def test_label_and_or_xor():
    z0 = np.array([0, 0, 1, 1])
    z1 = np.array([0, 1, 0, 1])
    ns = {"z0": z0, "z1": z1}
    assert list(LabelFormula("AND(z0, z1)")(ns)) == [0, 0, 0, 1]
    assert list(LabelFormula("OR(z0, z1)")(ns)) == [0, 1, 1, 1]
    assert list(LabelFormula("XOR(z0, z1)")(ns)) == [0, 1, 1, 0]
    assert list(LabelFormula("z0 & z1")(ns)) == [0, 0, 0, 1]
    assert list(LabelFormula("z0 ^ z1")(ns)) == [0, 1, 1, 0]


def test_label_nested_xor_of_and():
    z = {"z0": np.array([1, 1, 0]), "z1": np.array([1, 0, 0]), "z2": np.array([1, 0, 1])}
    # (z0 AND z1) XOR z2
    out = LabelFormula("XOR(AND(z0, z1), z2)")(z)
    assert list(out) == [0, 0, 1]


def test_label_maj_thresh_in():
    z = {"a": np.array([1, 1, 0]), "b": np.array([1, 0, 0]), "c": np.array([0, 0, 0])}
    assert list(LabelFormula("MAJ(a, b, c)")(z)) == [1, 0, 0]
    assert list(LabelFormula("THRESH(2, a, b, c)")(z)) == [1, 0, 0]
    ring = {"ring": np.array([0, 3, 5, 7])}
    assert list(LabelFormula("IN(ring, 2, 5)")(ring)) == [0, 1, 1, 0]


def test_label_support():
    f = LabelFormula("XOR(AND(z0, z1), z2)")
    assert set(f.support) == {"z0", "z1", "z2"}
    assert f.references("z0") and not f.references("z9")


def test_label_rejects_unsafe():
    with pytest.raises(Exception):
        LabelFormula("__import__('os').system('echo hi')")
    with pytest.raises(Exception):
        LabelFormula("z0 + z1")  # arithmetic not allowed


def test_sequence_aggregators():
    a = np.array([[0, 1, 0], [0, 0, 0]])
    assert list(LabelFormula("ANY(a)")({"a": a})) == [1, 0]
    assert list(LabelFormula("LAST(a)")({"a": a})) == [0, 0]
    assert list(LabelFormula("FIRST(a)")({"a": a})) == [0, 0]
    assert list(LabelFormula("COUNT_GE(a, 1)")({"a": a})) == [1, 0]


# ---- geometry sanity -------------------------------------------------------


def test_orthogonal_anchors_are_orthonormal():
    spec = ConceptSpec(dim=16, groups=[GroupSpec(kind="point", n=6)], geometry="orthogonal")
    w = World(spec)
    A = w.true_directions()
    gram = A @ A.T
    off = gram - np.diag(np.diag(gram))
    assert np.allclose(np.diag(gram), 1.0, atol=1e-6)
    assert np.abs(off).max() < 1e-6


def test_correlated_group_hits_target_cosine():
    spec = ConceptSpec(dim=32, groups=[GroupSpec(kind="correlated", n=2, cos_sim=0.6)])
    w = World(spec)
    A = w.true_directions()
    cos = A[0] @ A[1] / (np.linalg.norm(A[0]) * np.linalg.norm(A[1]))
    assert abs(cos - 0.6) < 1e-6


def test_circle_lies_on_a_ring():
    spec = ConceptSpec(dim=8, groups=[GroupSpec(kind="circle", n_positions=8, name="ring")],
                       noise=0.0, label="IN(ring, 0, 3)")
    w = World(spec)
    rng = np.random.default_rng(0)
    ns = {"ring": np.arange(8)}
    X = w.embed(ns, rng, add_noise=False)
    # every ring point has (near) unit radius in the plane it spans
    radii = np.linalg.norm(X, axis=1)
    assert np.allclose(radii, radii[0], atol=1e-6)
    # consecutive positions are equally spaced in angle
    x0 = X / radii[:, None]
    dots = np.sum(x0[:-1] * x0[1:], axis=1)
    assert np.allclose(dots, dots[0], atol=1e-6)


def test_superposition_more_concepts_than_dims():
    spec = ConceptSpec(dim=8, groups=[GroupSpec(kind="point", n=20)], geometry="random")
    w = World(spec)
    assert w.n_concepts == 20
    assert w.anchors.shape == (20, 8)
    norms = np.linalg.norm(w.anchors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


# ---- dataset + causal importance ------------------------------------------


def test_dataset_label_matches_formula():
    spec = ConceptSpec(dim=32, groups=[GroupSpec(kind="point", n=4)], label="AND(z0, z1)", noise=0.0)
    ds = make_dataset(spec, n=500)
    expected = (ds.Z["z0"] & ds.Z["z1"]).astype(np.int64)
    assert np.array_equal(ds.y, expected)


def test_data_importance_matches_support():
    # y depends only on z0, z1 -> those have importance, z2, z3 do not.
    spec = ConceptSpec(dim=32, groups=[GroupSpec(kind="point", n=4)], label="AND(z0, z1)")
    w = World(spec)
    imp = w.data_importance(n=6000)
    assert imp[0] > 0.1 and imp[1] > 0.1
    assert imp[2] < 1e-9 and imp[3] < 1e-9


def test_xor_importance_is_nonzero_for_all_three():
    spec = ConceptSpec(dim=32, groups=[GroupSpec(kind="point", n=4)], label="XOR(AND(z0, z1), z2)")
    w = World(spec)
    imp = w.data_importance(n=8000)
    # z2 flips the label whenever it is toggled -> importance ~0.5
    assert imp[2] > 0.4
    # z0, z1 matter only through the AND, so smaller but nonzero
    assert imp[0] > 0.05 and imp[1] > 0.05
    assert imp[3] < 1e-9


def test_determinism():
    spec = ConceptSpec(dim=16, groups=[GroupSpec(kind="point", n=3)], seed=123)
    a = make_dataset(spec, n=200)
    b = make_dataset(spec, n=200)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)


def test_sequence_dataset_shapes_and_label():
    spec = ConceptSpec(
        dim=16, groups=[GroupSpec(kind="point", n=3)],
        sequence=True, seq_len=6, label="AND(z0, z1)",
        sequence_label="ANY(z0) & LAST(z1)", noise=0.0,
    )
    ds = make_dataset(spec, n=300)
    assert ds.X.shape == (300, 6, 16)
    expected = (ds.Z["z0"].max(axis=1).astype(bool) & ds.Z["z1"][:, -1].astype(bool)).astype(np.int64)
    assert np.array_equal(ds.y, expected)
