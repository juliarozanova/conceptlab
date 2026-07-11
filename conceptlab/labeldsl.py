"""A tiny, safe label DSL.

Label formulas are written as strings over concept names, e.g.::

    (z0 & z1) ^ z2                 # XOR of an AND with a third concept
    MAJ(z0, z1, z2)               # majority vote
    THRESH(2, z0, z1, z2, z3)     # at least 2 active
    IN(ring, 2, 4)               # circular concept position in [2, 4]

For sequence (transformer) datasets the per-token concept arrays are aggregated
across positions before being combined::

    ANY(IN(ring, 2, 4)) & LAST(z0)

Evaluation is done by walking a restricted Python AST — no ``eval`` on the raw
string — so a config can never execute arbitrary code. Every operand is a numpy
array so a whole batch is evaluated at once.
"""

from __future__ import annotations

import ast
from typing import Iterable

import numpy as np

# ---- vectorised primitives -------------------------------------------------


def _reduce(op, args):
    out = np.asarray(args[0], dtype=bool)
    for a in args[1:]:
        out = op(out, np.asarray(a, dtype=bool))
    return out


def AND(*args):
    return _reduce(np.logical_and, args)


def OR(*args):
    return _reduce(np.logical_or, args)


def XOR(*args):
    return _reduce(np.logical_xor, args)


def NOT(a):
    return np.logical_not(np.asarray(a, dtype=bool))


def MAJ(*args):
    s = np.sum([np.asarray(a, dtype=int) for a in args], axis=0)
    return s > (len(args) / 2.0)


def THRESH(k, *args):
    s = np.sum([np.asarray(a, dtype=int) for a in args], axis=0)
    return s >= int(k)


def IN(positions, lo, hi):
    p = np.asarray(positions)
    return (p >= int(lo)) & (p <= int(hi))


# ---- sequence aggregators (collapse the token axis, axis=1) ----------------


def ANY(a):
    return np.asarray(a, dtype=bool).any(axis=1)


def ALL(a):
    return np.asarray(a, dtype=bool).all(axis=1)


def LAST(a):
    return np.asarray(a, dtype=bool)[:, -1]


def FIRST(a):
    return np.asarray(a, dtype=bool)[:, 0]


def COUNT_GE(a, k):
    return np.asarray(a, dtype=int).sum(axis=1) >= int(k)


_FUNCS = {
    "AND": AND, "OR": OR, "XOR": XOR, "NOT": NOT, "MAJ": MAJ,
    "THRESH": THRESH, "IN": IN, "ANY": ANY, "ALL": ALL, "LAST": LAST,
    "FIRST": FIRST, "COUNT_GE": COUNT_GE,
}

_BINOPS = {ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or, ast.BitXor: np.logical_xor}


class LabelFormula:
    """A parsed, reusable label formula.

    Parameters
    ----------
    expr:
        The formula string.
    """

    def __init__(self, expr: str):
        self.expr = expr
        self.tree = ast.parse(expr, mode="eval")
        self._validate(self.tree.body)

    # -- validation: only whitelisted node types may appear ------------------
    def _validate(self, node):
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOPS:
                raise ValueError(f"operator {type(node.op).__name__} not allowed; use & | ^")
            self._validate(node.left)
            self._validate(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.Invert, ast.Not)):
                raise ValueError("only ~ / not unary ops allowed")
            self._validate(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ValueError(f"unknown function in label: {ast.dump(node.func)}")
            for a in node.args:
                self._validate(a)
        elif isinstance(node, ast.Name):
            pass
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("only numeric constants allowed")
        else:
            raise ValueError(f"disallowed expression node: {type(node).__name__}")

    # -- concept names actually referenced (the formula's support) -----------
    @property
    def support(self) -> list[str]:
        names: list[str] = []
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Name) and n.id not in _FUNCS and n.id not in names:
                names.append(n.id)
        return names

    # -- evaluation ----------------------------------------------------------
    def _eval(self, node, ns):
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](self._eval(node.left, ns), self._eval(node.right, ns))
        if isinstance(node, ast.UnaryOp):
            return np.logical_not(self._eval(node.operand, ns))
        if isinstance(node, ast.Call):
            args = [self._eval(a, ns) for a in node.args]
            return _FUNCS[node.func.id](*args)
        if isinstance(node, ast.Name):
            if node.id not in ns:
                raise KeyError(f"concept '{node.id}' referenced by label not in namespace")
            return ns[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        raise ValueError(f"cannot evaluate node {type(node).__name__}")

    def __call__(self, namespace: dict) -> np.ndarray:
        """Evaluate the formula, returning an integer {0,1} label array."""
        out = self._eval(self.tree.body, namespace)
        return np.asarray(out, dtype=np.int64)

    def references(self, name: str) -> bool:
        return name in self.support
