"""Consume a data-contract dataset (e.g. from fraudgen) for concept attribution.

This is the transfer bridge: conceptlab reads the versioned parquet contract a
data source exports — event sequences, per-decision concept values, and (where
available) attribution ground truth — with no code dependency on the producer.
The loaded dataset plugs into the same :class:`TabTransformer` and concept
methods used on the synthetic conceptlab configs, so a method validated there is
run here, on fraud-shaped data, against the typology-defining ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .concepts import EventBatch


@dataclass
class ContractDataset:
    numeric: dict           # name -> (N, T)
    categorical: dict       # name -> (N, T) int
    concept_names: list
    C: np.ndarray           # (N, K) concept values at decision point
    concept_levels: list
    y: np.ndarray           # (N,) label
    typology: np.ndarray    # (N,) str
    attribution_gt: np.ndarray   # (N, K) 1 where concept defines this event's label
    typology_defining: dict
    seq_len: int
    concept_input_deps: dict = None   # concept -> [(field, scope)] (nullable)
    codebooks: dict = None            # field -> {code: label} (nullable)

    def event_batch(self) -> EventBatch:
        return EventBatch(self.numeric, self.categorical, latents={}, decision=-1)

    @property
    def n(self):
        return len(self.y)

    @property
    def n_concepts(self):
        return len(self.concept_names)


def load_contract(path: str | Path) -> ContractDataset:
    path = Path(path)
    graph = json.loads((path / "concept_graph.json").read_text())
    T = graph["seq_len"]
    num_cols = graph["numeric_cols"]
    cat_cols = graph["categorical_cols"]

    ev = pd.read_parquet(path / "events.parquet").sort_values(["seq_id", "t"])
    N = ev["seq_id"].nunique()
    numeric = {c: ev[c].to_numpy().reshape(N, T).astype(np.float32) for c in num_cols}
    categorical = {c: ev[c].to_numpy().reshape(N, T).astype(np.int64) for c in cat_cols}

    cdf = pd.read_parquet(path / "concepts.parquet")
    names = [c["name"] for c in graph["concepts"]]
    levels = [c["level"] for c in graph["concepts"]]
    cpiv = cdf.pivot(index="seq_id", columns="concept", values="value")[names]
    C = cpiv.to_numpy().astype(np.int64)

    lab = pd.read_parquet(path / "labels.parquet").sort_values("seq_id")
    y = lab["fraud"].to_numpy().astype(np.int64)
    typ = lab["typology"].to_numpy().astype(str)

    agt = pd.read_parquet(path / "ground_truth" / "attribution_gt.parquet")
    apiv = agt.pivot(index="seq_id", columns="concept", values="defines")[names]
    attribution = apiv.to_numpy().astype(np.int64)

    return ContractDataset(numeric=numeric, categorical=categorical, concept_names=names,
                           C=C, concept_levels=levels, y=y, typology=typ,
                           attribution_gt=attribution,
                           typology_defining=graph.get("typology_defining", {}), seq_len=T,
                           concept_input_deps=graph.get("concept_input_deps") or {},
                           codebooks=graph.get("codebooks") or {})
