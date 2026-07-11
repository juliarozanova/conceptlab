"""Synthetic concept data with fully known ground truth.

The generator builds an embedding space out of *concept anchors* (directions in
R^D), samples latent concept activations ``z``, composes an embedding
``x = sum_i z_i * anchor_i + noise``, and assigns a label ``y = f(z)`` from the
label DSL. Because we own the generative process, we can also *intervene* on it
(toggle a concept, regenerate) to obtain causal ground-truth importance — the
thing method-based attribution is ultimately trying to approximate.

Concept groups support three geometries:

* ``point``      — independent binary concepts, one anchor direction each.
* ``correlated`` — binary concepts whose anchors sit at a chosen cosine angle
  (non-orthogonal superposition of a related pair/group).
* ``circle``     — one ordinal concept traced around a ring in a 2D plane
  (reproduces the weekday-circle geometry from the literature).

Set the top-level ``geometry`` to ``"random"`` with more concepts than
dimensions to enter the **superposition** regime that motivates SAEs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .labeldsl import LabelFormula

# ---------------------------------------------------------------------------
# Spec dataclasses (YAML-friendly)
# ---------------------------------------------------------------------------


@dataclass
class GroupSpec:
    """One concept group. ``kind`` selects the geometry."""

    kind: str = "point"            # point | correlated | circle
    n: int = 1                     # point/correlated: number of binary concepts
    name_prefix: str = "z"         # concept names become f"{prefix}{i}"
    p_active: float = 0.5          # marginal activation prob (binary concepts)
    cos_sim: float = 0.6           # correlated: target cosine between anchors
    n_positions: int = 8           # circle: number of ring positions
    name: str = "ring"             # circle: concept name
    noise: Optional[float] = None  # per-group blob noise override


@dataclass
class ConceptSpec:
    """Declarative ground truth for a dataset."""

    dim: int = 32
    groups: list[GroupSpec] = field(default_factory=list)
    label: str = "AND(z0, z1)"
    geometry: str = "orthogonal"   # orthogonal | random(=superposition)
    noise: float = 0.1             # global blob noise sigma
    anchor_scale: float = 1.0
    seed: int = 0

    # sequence (transformer) options -- ignored for single-vector datasets
    sequence: bool = False
    seq_len: int = 12
    sequence_label: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "ConceptSpec":
        d = dict(d)
        groups = [GroupSpec(**g) for g in d.pop("groups", [])]
        return ConceptSpec(groups=groups, **d)


# ---------------------------------------------------------------------------
# Compiled world
# ---------------------------------------------------------------------------


@dataclass
class Concept:
    name: str
    kind: str                 # "binary" | "circular"
    dir_index: list[int]      # rows of the anchor matrix this concept owns
    n_positions: int = 0      # circular only
    p_active: float = 0.5     # binary only


class World:
    """Compiled, immutable ground truth: anchors + concept metadata + label."""

    def __init__(self, spec: ConceptSpec):
        self.spec = spec
        self.dim = spec.dim
        self.noise = spec.noise
        rng = np.random.default_rng(spec.seed)

        anchors: list[np.ndarray] = []
        concepts: list[Concept] = []

        # A shared orthonormal basis to draw near-clean directions from.
        basis = _orthonormal_basis(spec.dim, spec.dim, rng)
        next_basis = 0

        def take_orthonormal() -> np.ndarray:
            nonlocal next_basis
            if spec.geometry == "orthogonal" and next_basis < spec.dim:
                v = basis[next_basis]
                next_basis += 1
                return v.copy()
            # superposition / out of basis: random unit vector
            v = rng.standard_normal(spec.dim)
            return v / (np.linalg.norm(v) + 1e-12)

        for g in spec.groups:
            if g.kind == "point":
                for i in range(g.n):
                    a = take_orthonormal()
                    concepts.append(Concept(f"{g.name_prefix}{i}", "binary", [len(anchors)], p_active=g.p_active))
                    anchors.append(a)
            elif g.kind == "correlated":
                # First anchor free; the rest sit at cos_sim to the first.
                base = take_orthonormal()
                for i in range(g.n):
                    if i == 0:
                        a = base
                    else:
                        perp = take_orthonormal()
                        perp = perp - (perp @ base) * base
                        perp = perp / (np.linalg.norm(perp) + 1e-12)
                        a = g.cos_sim * base + np.sqrt(max(0.0, 1 - g.cos_sim ** 2)) * perp
                        a = a / (np.linalg.norm(a) + 1e-12)
                    concepts.append(Concept(f"{g.name_prefix}{i}", "binary", [len(anchors)], p_active=g.p_active))
                    anchors.append(a)
            elif g.kind == "circle":
                u = take_orthonormal()
                v = take_orthonormal()
                v = v - (v @ u) * u
                v = v / (np.linalg.norm(v) + 1e-12)
                idx = [len(anchors), len(anchors) + 1]
                anchors.append(u)
                anchors.append(v)
                concepts.append(Concept(g.name, "circular", idx, n_positions=g.n_positions))
            else:
                raise ValueError(f"unknown group kind: {g.kind}")

        self.anchors = np.stack(anchors, axis=0) * spec.anchor_scale  # (M, D)
        self.concepts = concepts
        self.concept_index = {c.name: i for i, c in enumerate(concepts)}
        self.label_formula = LabelFormula(spec.label)
        self.seq_formula = LabelFormula(spec.sequence_label) if spec.sequence_label else None

    # -- helpers -------------------------------------------------------------
    @property
    def n_concepts(self) -> int:
        return len(self.concepts)

    @property
    def binary_concepts(self) -> list[int]:
        return [i for i, c in enumerate(self.concepts) if c.kind == "binary"]

    def true_directions(self) -> np.ndarray:
        """One representative unit direction per concept (for recovery scoring).

        Circular concepts contribute their first plane axis; use
        :meth:`concept_dir_rows` when both plane axes are needed.
        """
        rows = [c.dir_index[0] for c in self.concepts]
        A = self.anchors[rows]
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)

    def concept_dir_rows(self, concept: int) -> np.ndarray:
        return self.anchors[self.concepts[concept].dir_index]

    @property
    def n_dirs(self) -> int:
        """Total number of atomic anchor directions (circular concepts own 2)."""
        return len(self.anchors)

    def anchor_concept_ids(self) -> list[int]:
        """For each anchor row, the index of the concept that owns it."""
        ids = [0] * self.n_dirs
        for ci, c in enumerate(self.concepts):
            for row in c.dir_index:
                ids[row] = ci
        return ids

    def anchor_unit(self) -> np.ndarray:
        """All anchor rows as unit vectors (the targets for concept recovery)."""
        A = self.anchors
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)

    def anchor_coeffs(self, ns: dict[str, np.ndarray]) -> np.ndarray:
        """Ground-truth coefficient of each anchor row per sample, (n, M).

        Binary concept -> its z value on its single row; circular concept ->
        (cos theta, sin theta) on its two rows. This is exactly the linear
        combination :meth:`embed` uses (before noise), so a regression of the
        embeddings onto these coefficients recovers the anchor directions.
        """
        n = len(next(iter(ns.values())))
        C = np.zeros((n, self.n_dirs), dtype=np.float64)
        for c in self.concepts:
            val = ns[c.name]
            if c.kind == "binary":
                C[:, c.dir_index[0]] = val
            else:
                theta = 2 * np.pi * val / c.n_positions
                C[:, c.dir_index[0]] = np.cos(theta)
                C[:, c.dir_index[1]] = np.sin(theta)
        return C

    # -- sampling ------------------------------------------------------------
    def sample_latents(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Return a namespace mapping concept name -> activation array (n,)."""
        ns: dict[str, np.ndarray] = {}
        for c in self.concepts:
            if c.kind == "binary":
                ns[c.name] = (rng.random(n) < c.p_active).astype(np.int64)
            else:  # circular
                ns[c.name] = rng.integers(0, c.n_positions, size=n)
        return ns

    def embed(self, ns: dict[str, np.ndarray], rng: np.random.Generator,
              add_noise: bool = True) -> np.ndarray:
        """Build embeddings x (n, D) from a latent namespace."""
        n = len(next(iter(ns.values())))
        x = np.zeros((n, self.dim), dtype=np.float64)
        for c in self.concepts:
            val = ns[c.name]
            if c.kind == "binary":
                x += val[:, None] * self.anchors[c.dir_index[0]][None, :]
            else:
                theta = 2 * np.pi * val / c.n_positions
                u = self.anchors[c.dir_index[0]]
                v = self.anchors[c.dir_index[1]]
                x += np.cos(theta)[:, None] * u[None, :] + np.sin(theta)[:, None] * v[None, :]
        if add_noise and self.noise > 0:
            x += rng.normal(0, self.noise, size=x.shape)
        return x

    # -- ground-truth importance --------------------------------------------
    def toggle(self, ns: dict[str, np.ndarray], concept: int,
               rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Return a copy of the latent namespace with one concept toggled.

        Binary concepts flip 0<->1; circular concepts advance to a random
        *different* position. Used for both data-level and model-level causal
        importance.
        """
        c = self.concepts[concept]
        out = {k: v.copy() for k, v in ns.items()}
        if c.kind == "binary":
            out[c.name] = 1 - out[c.name]
        else:
            n = len(out[c.name])
            shift = rng.integers(1, c.n_positions, size=n)
            out[c.name] = (out[c.name] + shift) % c.n_positions
        return out

    def data_importance(self, n: int = 4000, seed: int = 12345) -> np.ndarray:
        """Data-level causal importance: how much toggling each concept changes
        the label, E[|f(toggle_i(z)) - f(z)|]. Depends only on ``f``, not any model.
        """
        rng = np.random.default_rng(seed)
        ns = self.sample_latents(n, rng)
        base = self._label(ns)
        imp = np.zeros(self.n_concepts)
        for i in range(self.n_concepts):
            flipped = self._label(self.toggle(ns, i, rng))
            imp[i] = np.mean(np.abs(flipped - base))
        return imp

    # -- labels --------------------------------------------------------------
    def _label(self, ns: dict[str, np.ndarray]) -> np.ndarray:
        return self.label_formula(ns)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    X: np.ndarray                 # (N, D) or (N, T, D)
    y: np.ndarray                 # (N,)
    Z: dict[str, np.ndarray]      # latent namespace: name -> (N,) or (N, T)
    world: World
    is_sequence: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.world.dim

    def concept_matrix(self) -> np.ndarray:
        """Binary concept activations as a dense (N, n_concepts) matrix.

        For single-vector datasets this is the per-sample latent; for sequences
        it is the sequence-level ``ANY`` aggregate (a reasonable per-concept
        summary for probing). Circular concepts contribute their raw position.
        """
        cols = []
        for c in self.world.concepts:
            v = self.Z[c.name]
            if self.is_sequence:
                v = v.max(axis=1) if c.kind == "binary" else v[:, -1]
            cols.append(v)
        return np.stack(cols, axis=1).astype(np.float64)


def make_dataset(spec: ConceptSpec, n: int, split_seed: int = 0) -> Dataset:
    """Generate a single-vector or sequence dataset from a spec."""
    world = World(spec)
    rng = np.random.default_rng(spec.seed + 1000 * split_seed + 7)
    if not spec.sequence:
        ns = world.sample_latents(n, rng)
        X = world.embed(ns, rng).astype(np.float32)
        y = world._label(ns)
        return Dataset(X=X, y=y, Z=ns, world=world, is_sequence=False,
                       meta={"n": n, "split_seed": split_seed})

    # sequence dataset: T tokens per sample, each an independently sampled event
    if world.seq_formula is None:
        raise ValueError("sequence dataset requires spec.sequence_label")
    T = spec.seq_len
    per_token: dict[str, np.ndarray] = {c.name: np.zeros((n, T), dtype=np.int64) for c in world.concepts}
    X = np.zeros((n, T, world.dim), dtype=np.float32)
    for t in range(T):
        ns_t = world.sample_latents(n, rng)
        X[:, t, :] = world.embed(ns_t, rng).astype(np.float32)
        for name, v in ns_t.items():
            per_token[name][:, t] = v
    y = world.seq_formula(per_token)
    return Dataset(X=X, y=y, Z=per_token, world=world, is_sequence=True,
                   meta={"n": n, "split_seed": split_seed, "seq_len": T})


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------


def _orthonormal_basis(k: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """k orthonormal row vectors in R^dim (k <= dim)."""
    k = min(k, dim)
    a = rng.standard_normal((dim, k))
    q, _ = np.linalg.qr(a)
    return q[:, :k].T.copy()


def spec_to_serializable(spec: ConceptSpec) -> dict:
    """Round-trippable dict for saving the exact ground-truth spec with a run."""
    d = dataclasses.asdict(spec)
    return d
