"""Synthetic tabular event sequences with latent episodes and counterfactuals.

A :class:`TableSpec` declares columns; a :class:`SequenceWorld` samples batches
of event sequences whose distributions are modulated by **latent episodes**
(sticky 0/1 states like ``on_trip`` and bursty ``session`` state). Those latents
are the substrate for high-level concepts and are exported as ground truth.

The critical property is **replayable noise**: every random draw is keyed by
``(seed, sequence, column, timestep)`` via a counter-based generator, so
:meth:`SequenceWorld.resimulate` can rebuild a batch with a latent toggled while
reusing every other draw — the paired counterfactual that makes model-level
concept attribution computable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .concepts import EventBatch


def _key_rng(*parts) -> np.random.Generator:
    """Deterministic Generator seeded by hashing the parts (counter-based)."""
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


@dataclass
class NumericCol:
    name: str
    base_mean: float = 0.0
    base_std: float = 1.0
    lognormal: bool = False
    # per-latent additive shift on the (log-)mean when the latent is active
    latent_shift: dict[str, float] = field(default_factory=dict)


@dataclass
class CategoricalCol:
    name: str
    cardinality: int
    # base categorical distribution (None -> uniform); Zipf if zipf>0
    zipf: float = 0.0
    # per-latent override distribution (name -> probability vector)
    latent_dist: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class LatentSpec:
    """A sticky binary episode latent (Markov on/off with dwell)."""

    name: str
    p_start: float = 0.05        # prob of switching on at a step when off
    p_stop: float = 0.25         # prob of switching off at a step when on
    p_active_init: float = 0.1   # prob active at t=0


@dataclass
class TableSpec:
    numeric: list[NumericCol]
    categorical: list[CategoricalCol]
    latents: list[LatentSpec] = field(default_factory=list)
    seq_len: int = 16
    seed: int = 0


class SequenceWorld:
    """Samples event-sequence batches from a :class:`TableSpec`."""

    def __init__(self, spec: TableSpec):
        self.spec = spec
        self.numeric = {c.name: c for c in spec.numeric}
        self.categorical = {c.name: c for c in spec.categorical}
        self.latents = {l.name: l for l in spec.latents}
        self._cat_base = {}
        for c in spec.categorical:
            if c.zipf > 0:
                w = 1.0 / np.power(np.arange(1, c.cardinality + 1), c.zipf)
                self._cat_base[c.name] = w / w.sum()
            else:
                self._cat_base[c.name] = np.full(c.cardinality, 1.0 / c.cardinality)

    # -- latent episodes -----------------------------------------------------
    def _sample_latents(self, seq_ids: np.ndarray) -> dict[str, np.ndarray]:
        T = self.spec.seq_len
        out = {}
        for name, spec in self.latents.items():
            arr = np.zeros((len(seq_ids), T), dtype=np.int64)
            for i, sid in enumerate(seq_ids):
                rng = _key_rng(self.spec.seed, "latent", name, int(sid))
                state = int(rng.random() < spec.p_active_init)
                for t in range(T):
                    if state == 0 and rng.random() < spec.p_start:
                        state = 1
                    elif state == 1 and rng.random() < spec.p_stop:
                        state = 0
                    arr[i, t] = state
            out[name] = arr
        return out

    # -- columns given latents ----------------------------------------------
    def _sample_columns(self, seq_ids: np.ndarray, latents: dict[str, np.ndarray]):
        T = self.spec.seq_len
        N = len(seq_ids)
        numeric, categorical = {}, {}

        for name, c in self.numeric.items():
            arr = np.zeros((N, T))
            for i, sid in enumerate(seq_ids):
                rng = _key_rng(self.spec.seed, "num", name, int(sid))
                z = rng.standard_normal(T)
                mean = np.full(T, c.base_mean)
                for lat, shift in c.latent_shift.items():
                    mean = mean + shift * latents[lat][i]
                vals = mean + c.base_std * z
                arr[i] = np.exp(vals) if c.lognormal else vals
            numeric[name] = arr

        for name, c in self.categorical.items():
            arr = np.zeros((N, T), dtype=np.int64)
            base = self._cat_base[name]
            for i, sid in enumerate(seq_ids):
                rng = _key_rng(self.spec.seed, "cat", name, int(sid))
                u = rng.random(T)
                for t in range(T):
                    # pick the active latent override if any latent is on
                    dist = base
                    for lat, d in c.latent_dist.items():
                        if latents.get(lat) is not None and latents[lat][i, t]:
                            dist = np.asarray(d)
                            break
                    arr[i, t] = np.searchsorted(np.cumsum(dist), u[t] * dist.sum())
            categorical[name] = np.clip(arr, 0, c.cardinality - 1)
        return numeric, categorical

    # -- public API ----------------------------------------------------------
    def sample(self, n: int, offset: int = 0) -> EventBatch:
        seq_ids = np.arange(offset, offset + n)
        latents = self._sample_latents(seq_ids)
        numeric, categorical = self._sample_columns(seq_ids, latents)
        b = EventBatch(numeric, categorical, latents)
        b._seq_ids = seq_ids  # type: ignore[attr-defined]
        return b

    def resimulate(self, batch: EventBatch, toggle_latent: str, value: int) -> EventBatch:
        """Rebuild the batch with one latent forced to ``value`` everywhere it is
        currently the opposite, reusing every other random draw.

        Because column noise is keyed by (seed, sequence, column) independent of
        the latent, only the latent-conditioned distribution changes — the
        paired counterfactual conceptlab tier-3 needs. Numeric columns shift by
        their ``latent_shift`` deterministically; categorical columns re-decode
        under the (possibly) changed per-step distribution using the same U draw.
        """
        seq_ids = getattr(batch, "_seq_ids", np.arange(batch.n))
        latents = {k: v.copy() for k, v in batch.latents.items()}
        latents[toggle_latent] = np.full_like(latents[toggle_latent], value)
        numeric, categorical = self._sample_columns(seq_ids, latents)
        out = EventBatch(numeric, categorical, latents)
        out._seq_ids = seq_ids  # type: ignore[attr-defined]
        return out
