"""Concept graphs over tabular event sequences.

The chain is  inputs (event columns) -> concepts -> label, with the label
depending on inputs *only* through concepts (pure mediation). "Direct" input
influence is represented as **level-0 identity concepts** — single-column
predicates — so input-level and concept-level attribution are the same DAG read
at two granularities.

A :class:`ConceptGraph` holds named concepts, each with an explicit ``level``
and a definition that is either

* an **expression** over atoms and other concepts (differentiable soft logic),
* or a reference to a **generator latent** (``EPISODE``) — high-level concepts
  like "on holiday" that are world state, not a row formula.

Everything evaluates on an :class:`EventBatch`: a batch of N sequences of T
events, columns stored column-major. Concepts are evaluated at each sequence's
**decision point** (the last event), with window atoms looking back over the
sequence — the "explain this transaction given its history" framing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# batch container
# ---------------------------------------------------------------------------


@dataclass
class EventBatch:
    """A batch of N sequences, T events each, columns column-major.

    numeric columns: float (N, T); categorical columns: int codes (N, T);
    latents: generator state (N, T). ``decision`` is the index used as the
    per-sequence decision point (default: last event).
    """

    numeric: dict[str, np.ndarray]
    categorical: dict[str, np.ndarray]
    latents: dict[str, np.ndarray] = field(default_factory=dict)
    decision: int = -1

    @property
    def n(self) -> int:
        any_col = next(iter({**self.numeric, **self.categorical}.values()))
        return any_col.shape[0]

    @property
    def T(self) -> int:
        any_col = next(iter({**self.numeric, **self.categorical}.values()))
        return any_col.shape[1]

    def col(self, name: str) -> np.ndarray:
        if name in self.numeric:
            return self.numeric[name]
        if name in self.categorical:
            return self.categorical[name]
        if name in self.latents:
            return self.latents[name]
        raise KeyError(f"unknown column/latent: {name}")


# ---------------------------------------------------------------------------
# atoms — the leaves of a concept expression
# ---------------------------------------------------------------------------


class Atom:
    """A predicate over the batch, evaluated at the decision point.

    ``hard`` returns a {0,1} numpy array (N,); ``soft`` returns a differentiable
    torch tensor (N,) that equals ``hard`` at temperature 0. ``inputs`` lists the
    columns/latents read (for the structural support DAG).
    """

    name: str = "atom"
    inputs: tuple[str, ...] = ()

    def hard(self, b: EventBatch) -> np.ndarray:
        raise NotImplementedError

    def soft(self, ctx: "SoftContext") -> torch.Tensor:
        raise NotImplementedError


@dataclass
class SoftContext:
    """Differentiable view of a batch for soft-logic evaluation.

    ``numeric`` holds torch tensors (N, T) with grad enabled where an attribution
    path is wanted; ``categorical`` / ``latents`` stay numpy (discrete, no path).
    """

    numeric: dict[str, torch.Tensor]
    categorical: dict[str, np.ndarray]
    latents: dict[str, np.ndarray]
    tau: float
    decision: int = -1


class GT(Atom):
    """numeric column > threshold."""

    def __init__(self, col: str, v: float):
        self.col, self.v = col, v
        self.name = f"{col}>{v:g}"
        self.inputs = (col,)

    def hard(self, b):
        return (b.numeric[self.col][:, b.decision] > self.v).astype(np.float64)

    def soft(self, ctx):
        x = ctx.numeric[self.col][:, ctx.decision]
        return torch.sigmoid((x - self.v) / ctx.tau)


class LT(Atom):
    def __init__(self, col: str, v: float):
        self.col, self.v = col, v
        self.name = f"{col}<{v:g}"
        self.inputs = (col,)

    def hard(self, b):
        return (b.numeric[self.col][:, b.decision] < self.v).astype(np.float64)

    def soft(self, ctx):
        x = ctx.numeric[self.col][:, ctx.decision]
        return torch.sigmoid((self.v - x) / ctx.tau)


class EQ(Atom):
    """categorical column == code (identity; non-differentiable in the code)."""

    def __init__(self, col: str, code: int, label: Optional[str] = None):
        self.col, self.code = col, code
        self.name = label or f"{col}=={code}"
        self.inputs = (col,)

    def hard(self, b):
        return (b.categorical[self.col][:, b.decision] == self.code).astype(np.float64)

    def soft(self, ctx):
        v = (ctx.categorical[self.col][:, ctx.decision] == self.code).astype(np.float64)
        return torch.tensor(v, dtype=torch.float32)


class IN_SET(Atom):
    def __init__(self, col: str, codes: set[int], label: Optional[str] = None):
        self.col, self.codes = col, set(codes)
        self.name = label or f"{col}∈{sorted(self.codes)}"
        self.inputs = (col,)

    def hard(self, b):
        return np.isin(b.categorical[self.col][:, b.decision], list(self.codes)).astype(np.float64)

    def soft(self, ctx):
        v = np.isin(ctx.categorical[self.col][:, ctx.decision], list(self.codes)).astype(np.float64)
        return torch.tensor(v, dtype=torch.float32)


class COUNT_W(Atom):
    """>= k events in the trailing window matching a row predicate.

    Reproduces windowed concepts like "short burst" (>= k events in last w).
    Soft version uses a sigmoid over the (soft) count.
    """

    def __init__(self, pred: Atom, window: int, k: int, label: Optional[str] = None):
        self.pred, self.window, self.k = pred, window, k
        self.name = label or f"count({pred.name},w={window})>={k}"
        self.inputs = pred.inputs

    def _counts_hard(self, b: EventBatch) -> np.ndarray:
        d = b.decision if b.decision >= 0 else b.T + b.decision
        lo = max(0, d - self.window + 1)
        # evaluate the predicate at every timestep in the window
        counts = np.zeros(b.n)
        for t in range(lo, d + 1):
            sub = EventBatch(b.numeric, b.categorical, b.latents, decision=t)
            counts += self.pred.hard(sub)
        return counts

    def hard(self, b):
        return (self._counts_hard(b) >= self.k).astype(np.float64)

    def soft(self, ctx):
        d = ctx.decision if ctx.decision >= 0 else next(iter(ctx.numeric.values())).shape[1] + ctx.decision
        lo = max(0, d - self.window + 1)
        total = None
        for t in range(lo, d + 1):
            sub = SoftContext(ctx.numeric, ctx.categorical, ctx.latents, ctx.tau, decision=t)
            s = self.pred.soft(sub)
            total = s if total is None else total + s
        return torch.sigmoid((total - (self.k - 0.5)) / (ctx.tau * self.k + 1e-6))


class NOVEL(Atom):
    """decision-point categorical value not seen earlier in the sequence."""

    def __init__(self, col: str, label: Optional[str] = None):
        self.col = col
        self.name = label or f"novel({col})"
        self.inputs = (col,)

    def hard(self, b):
        d = b.decision if b.decision >= 0 else b.T + b.decision
        cur = b.categorical[self.col][:, d]
        hist = b.categorical[self.col][:, :d]
        seen = (hist == cur[:, None]).any(axis=1) if d > 0 else np.zeros(b.n, bool)
        return (~seen).astype(np.float64)

    def soft(self, ctx):
        b = EventBatch({k: v for k, v in ctx.numeric.items()}, ctx.categorical, ctx.latents, ctx.decision)
        # categorical -> not differentiable; reuse hard
        return torch.tensor(self.hard(_ctx_to_batch(ctx)), dtype=torch.float32)


class EPISODE(Atom):
    """Reads a generator latent at the decision point — a high-level concept
    that is world state, not a row formula (e.g. ``on_holiday``)."""

    def __init__(self, latent: str, label: Optional[str] = None):
        self.latent = latent
        self.name = label or latent
        self.inputs = (latent,)

    def hard(self, b):
        return (b.latents[self.latent][:, b.decision] > 0.5).astype(np.float64)

    def soft(self, ctx):
        v = (ctx.latents[self.latent][:, ctx.decision] > 0.5).astype(np.float64)
        return torch.tensor(v, dtype=torch.float32)


def _ctx_to_batch(ctx: SoftContext) -> EventBatch:
    num = {k: v.detach().numpy() for k, v in ctx.numeric.items()}
    return EventBatch(num, ctx.categorical, ctx.latents, ctx.decision)


# ---------------------------------------------------------------------------
# boolean combinators (product t-norm; differentiable, exact at {0,1})
# ---------------------------------------------------------------------------


class Node:
    """Expression node: an Atom, a concept reference, or a boolean op."""

    def inputs(self, graph: "ConceptGraph") -> set[str]:
        raise NotImplementedError

    def hard(self, b, graph):
        raise NotImplementedError

    def soft(self, ctx, graph):
        raise NotImplementedError


class Leaf(Node):
    def __init__(self, atom: Atom):
        self.atom = atom

    def inputs(self, graph):
        return set(self.atom.inputs)

    def hard(self, b, graph):
        return self.atom.hard(b)

    def soft(self, ctx, graph):
        return self.atom.soft(ctx)


class Ref(Node):
    """Reference to another concept by name."""

    def __init__(self, name: str):
        self.ref = name

    def inputs(self, graph):
        return graph.concepts[self.ref].expr.inputs(graph)

    def hard(self, b, graph):
        return graph.eval_concept_hard(self.ref, b)

    def soft(self, ctx, graph):
        return graph.eval_concept_soft(self.ref, ctx)


class BoolOp(Node):
    def __init__(self, op: str, *children: Node):
        self.op, self.children = op, children

    def inputs(self, graph):
        out: set[str] = set()
        for c in self.children:
            out |= c.inputs(graph)
        return out

    def hard(self, b, graph):
        vals = [c.hard(b, graph) for c in self.children]
        return _reduce_hard(self.op, vals)

    def soft(self, ctx, graph):
        vals = [c.soft(ctx, graph) for c in self.children]
        return _reduce_soft(self.op, vals)


def _reduce_hard(op, vals):
    if op == "NOT":
        return 1.0 - vals[0]
    out = vals[0]
    for v in vals[1:]:
        if op == "AND":
            out = out * v
        elif op == "OR":
            out = 1 - (1 - out) * (1 - v)
        elif op == "XOR":
            out = out + v - 2 * out * v
    return out


def _reduce_soft(op, vals):
    if op == "NOT":
        return 1.0 - vals[0]
    out = vals[0]
    for v in vals[1:]:
        if op == "AND":
            out = out * v
        elif op == "OR":
            out = 1 - (1 - out) * (1 - v)
        elif op == "XOR":
            out = out + v - 2 * out * v
    return out


# convenience builders
def A(atom: Atom) -> Leaf:
    return Leaf(atom)


def ref(name: str) -> Ref:
    return Ref(name)


def AND(*c): return BoolOp("AND", *c)
def OR(*c): return BoolOp("OR", *c)
def XOR(*c): return BoolOp("XOR", *c)
def NOT(c): return BoolOp("NOT", c)


# ---------------------------------------------------------------------------
# concept graph
# ---------------------------------------------------------------------------


@dataclass
class Concept:
    name: str
    level: int
    expr: Node


class ConceptGraph:
    """Named concepts + a label expression that references concepts only."""

    def __init__(self, concepts: list[Concept], label: Node):
        self.concepts = {c.name: c for c in concepts}
        self.order = [c.name for c in concepts]
        self.label = label
        self._check_mediation()

    def _check_mediation(self):
        # the label may reference concept names only (enforced structurally)
        for name in _refs_in(self.label):
            if name not in self.concepts:
                raise ValueError(f"label references '{name}' which is not a concept "
                                 f"(mediation requires the label to depend on inputs only via concepts)")

    @property
    def names(self) -> list[str]:
        return self.order

    @property
    def n_concepts(self) -> int:
        return len(self.order)

    def input_columns(self) -> list[str]:
        cols: set[str] = set()
        for c in self.concepts.values():
            cols |= c.expr.inputs(self)
        return sorted(cols)

    # -- evaluation (memoised per call via a cache dict) ---------------------
    def eval_concept_hard(self, name: str, b: EventBatch, cache=None) -> np.ndarray:
        cache = cache if cache is not None else {}
        if name in cache:
            return cache[name]
        v = self.concepts[name].expr.hard(b, self)
        cache[name] = v
        return v

    def eval_concept_soft(self, name: str, ctx: SoftContext, cache=None) -> torch.Tensor:
        cache = cache if cache is not None else {}
        if name in cache:
            return cache[name]
        v = self.concepts[name].expr.soft(ctx, self)
        cache[name] = v
        return v

    def concept_matrix_hard(self, b: EventBatch) -> np.ndarray:
        """(N, K) hard concept values at the decision point."""
        cache: dict = {}
        return np.stack([self.eval_concept_hard(n, b, cache) for n in self.order], axis=1)

    def concept_matrix_soft(self, ctx: SoftContext) -> torch.Tensor:
        cache: dict = {}
        return torch.stack([self.eval_concept_soft(n, ctx, cache) for n in self.order], axis=1)

    def label_hard(self, b: EventBatch) -> np.ndarray:
        return (self.label.hard(b, self) > 0.5).astype(np.int64)

    # -- override evaluation (the engine of attribution ground truth) --------
    def _eval_cached(self, node: Node, b: EventBatch, cache: dict) -> np.ndarray:
        """Evaluate a node, resolving concept references through ``cache`` so
        overridden concepts short-circuit their own definition and everything
        downstream recomputes using the overridden value."""
        if isinstance(node, Leaf):
            return node.atom.hard(b)
        if isinstance(node, Ref):
            if node.ref in cache:
                return cache[node.ref]
            v = self._eval_cached(self.concepts[node.ref].expr, b, cache)
            cache[node.ref] = v
            return v
        if isinstance(node, BoolOp):
            vals = [self._eval_cached(c, b, cache) for c in node.children]
            return _reduce_hard(node.op, vals)
        raise TypeError(node)

    def label_under_override(self, b: EventBatch, overrides: dict[str, np.ndarray]) -> np.ndarray:
        """Label with some concepts forced to given values (float in [0,1]).

        This is the interventional primitive: overriding concept k propagates to
        every downstream concept and to the label, giving the causal effect of k
        *as the true label function sees it*.
        """
        cache = {k: np.asarray(v, dtype=np.float64) for k, v in overrides.items()}
        val = self._eval_cached(self.label, b, cache)
        return (val > 0.5).astype(np.int64)

    def label_soft(self, ctx: SoftContext) -> torch.Tensor:
        return self.label.soft(ctx, self)

    # -- structural support (tier 1) ----------------------------------------
    def label_support_concepts(self) -> list[str]:
        """Direct concept references in the label expression."""
        return [n for n in _refs_in(self.label)]

    def transitive_support(self) -> list[str]:
        """All concepts the label depends on, transitively through the graph.

        This is the tier-1 structural support: any concept not in this set is
        structurally irrelevant, and a faithful method must give it ~0.
        """
        seen: set[str] = set()
        stack = list(self.label_support_concepts())
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(_refs_in(self.concepts[n].expr))
        # preserve declaration order
        return [n for n in self.order if n in seen]

    def concept_support_inputs(self, name: str) -> set[str]:
        return self.concepts[name].expr.inputs(self)

    def to_json(self) -> dict:
        return {
            "concepts": [
                {"name": c.name, "level": c.level, "inputs": sorted(c.expr.inputs(self))}
                for c in self.concepts.values()
            ],
            "label_support": self.label_support_concepts(),
            "input_columns": self.input_columns(),
        }


def _refs_in(node: Node) -> list[str]:
    out: list[str] = []
    if isinstance(node, Ref):
        out.append(node.ref)
    elif isinstance(node, BoolOp):
        for c in node.children:
            out += _refs_in(c)
    return out
