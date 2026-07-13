"""Tabular-concept experiment configs.

Concept graphs are code (the DSL is Python), so configs are builder functions
returning a :class:`ConceptConfig`. Each ships a fraud-flavored schema so the
configs read like the transfer target, but the machinery is schema-generic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .concepts import (Concept, ConceptGraph, GT, LT, EQ, IN_SET, COUNT_W, NOVEL,
                       EPISODE, A, ref, AND, OR, XOR, NOT)
from .tabular import TableSpec, NumericCol, CategoricalCol, LatentSpec, SequenceWorld


@dataclass
class ConceptConfig:
    name: str
    description: str
    build: Callable[[], tuple]     # -> (SequenceWorld, ConceptGraph)
    dim: int = 16
    n_train: int = 4000
    n_val: int = 1500
    tau: float = 0.1
    seeds: list[int] = field(default_factory=lambda: [0])
    modeB_epochs: int = 30


def _fraud_schema(seed=0, seq_len=12, extra_country_trip=True):
    country_kw = {"latent_dist": {"on_trip": [0.05, 0.05, 0.225, 0.225, 0.225, 0.225]}} if extra_country_trip else {}
    return TableSpec(
        numeric=[NumericCol("amount", base_mean=4.0, base_std=1.2),
                 NumericCol("dt", base_mean=0.8, base_std=0.6)],
        categorical=[CategoricalCol("channel", 3),
                     CategoricalCol("country", 6, **country_kw)],
        latents=[LatentSpec("on_trip", p_start=0.08, p_stop=0.2, p_active_init=0.15)],
        seq_len=seq_len, seed=seed)


# --- configs ----------------------------------------------------------------


def _compound():
    world = SequenceWorld(_fraud_schema())
    concepts = [
        Concept("amount_gt", 0, A(GT("amount", 4.5))),
        Concept("cnp", 0, A(EQ("channel", 0, "cnp"))),
        Concept("foreign", 0, A(IN_SET("country", {2, 3, 4, 5}, "foreign"))),
        Concept("short_burst", 1, A(COUNT_W(LT("dt", 0.6), window=6, k=3, label="burst"))),
        Concept("on_holiday", 1, A(EPISODE("on_trip", "on_holiday"))),   # distractor
    ]
    # card-testing typology AND foreign-cashout typology
    label = OR(AND(ref("short_burst"), ref("cnp")), AND(ref("foreign"), ref("amount_gt")))
    return world, ConceptGraph(concepts, label=label)


def _overdetermined():
    world = SequenceWorld(_fraud_schema(seed=1))
    concepts = [
        Concept("amount_gt", 0, A(GT("amount", 4.2))),
        Concept("cnp", 0, A(EQ("channel", 0, "cnp"))),
        Concept("foreign", 0, A(IN_SET("country", {3, 4, 5}, "foreign"))),
        Concept("on_holiday", 1, A(EPISODE("on_trip", "on_holiday"))),
    ]
    # three overlapping single-concept "typologies" -> heavy OR, redundant causes
    label = OR(ref("amount_gt"), ref("cnp"), ref("foreign"))
    return world, ConceptGraph(concepts, label=label)


def _correlated():
    # 'on_holiday' is made to correlate strongly with the label, but is NOT in it;
    # a good method must not attribute the label to it. This is the CAV-confounding
    # test: without decorrelated controls, a naive concept detector = a label detector.
    spec = _fraud_schema(seed=2)
    # make amount jump during trips so that on_holiday correlates with the amount
    # concept that DOES drive the label
    spec.numeric[0].latent_shift = {"on_trip": 1.5}
    world = SequenceWorld(spec)
    concepts = [
        Concept("amount_gt", 0, A(GT("amount", 5.0))),
        Concept("cnp", 0, A(EQ("channel", 0, "cnp"))),
        Concept("on_holiday", 1, A(EPISODE("on_trip", "on_holiday"))),   # confounder
    ]
    label = AND(ref("amount_gt"), ref("cnp"))
    return world, ConceptGraph(concepts, label=label)


CONFIGS = {
    "tab_compound": ConceptConfig(
        name="tab_compound",
        description="Compound concepts vs their constituents: label = (burst AND cnp) OR "
                    "(foreign AND amount>500). Tests separating a compound typology from its "
                    "level-0 parts; on_holiday is a structurally-irrelevant distractor.",
        build=_compound, seeds=[0, 1]),
    "tab_overdetermined": ConceptConfig(
        name="tab_overdetermined",
        description="Redundant causes: label = amount>500 OR cnp OR foreign. Interventional and "
                    "Shapley ground truth disagree (large definition gap); both are shown and "
                    "single-concept toggling is often inert where the label is jointly satisfied.",
        build=_overdetermined, seeds=[0, 1]),
    "tab_correlated": ConceptConfig(
        name="tab_correlated",
        description="CAV confounding: on_holiday is correlated with fraud (trips raise the amount "
                    "that drives the label) but is NOT causal. A faithful method must not attribute "
                    "the decision to it — the failure mode that motivates decorrelated controls.",
        build=_correlated, seeds=[0, 1]),
}
