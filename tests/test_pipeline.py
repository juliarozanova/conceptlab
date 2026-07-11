"""Fast integration tests encoding the benchmark's key expected separations.

These are deliberately small (few samples/epochs) so they run in CI, but they
assert the qualitative results the whole testbed exists to produce:

* the supervised probe *skyline* recovers orthogonal concepts near-perfectly;
* on an XOR label, a purely linear importance method mis-ranks the coupled
  concepts while a causal (ablation) method ranks them correctly.
"""

import numpy as np

from conceptlab.datagen import ConceptSpec, GroupSpec, make_dataset
from conceptlab.eval import aggregate_to_concepts, faithfulness, model_importance, recovery
from conceptlab.methods import build_context, build_method
from conceptlab.train import TrainConfig, train_model


def _train(spec, n=4000, **tk):
    tr = make_dataset(spec, n, 0)
    va = make_dataset(spec, n // 4, 1)
    res = train_model(tr, va, TrainConfig(**tk))
    return tr, va, res


def test_skyline_recovers_orthogonal_concepts():
    spec = ConceptSpec(dim=24, groups=[GroupSpec(kind="point", n=5)], label="AND(z0, z1)", noise=0.1)
    tr, va, res = _train(spec, n=3000, epochs=30)
    ctx = build_context(res.model, tr, seed=0)
    disc = build_method("probe_skyline").fit(ctx).discovered_concepts()
    rec = recovery(disc, tr.world)
    assert rec.mean_matched_cosine > 0.95


def test_linear_importance_fails_but_causal_succeeds_on_xor():
    spec = ConceptSpec(dim=24, groups=[GroupSpec(kind="point", n=6)],
                       label="XOR(AND(z0, z1), z2)", geometry="orthogonal", noise=0.05)
    tr, va, res = _train(spec, n=8000, epochs=80, lr=2e-3,
                         model_kwargs={"hidden": 128, "n_layers": 3})
    assert res.val_acc > 0.9, f"model did not learn XOR (val_acc={res.val_acc})"

    ctx = build_context(res.model, tr, seed=0)
    world = tr.world
    true_dirs = world.anchor_unit()
    gt = model_importance(res.model, world)

    def concept_faith(method_name):
        s = build_method(method_name).fit(ctx).score_directions(true_dirs)
        return faithfulness(aggregate_to_concepts(s, world), gt).spearman

    rho_causal = concept_faith("ablation")
    rho_linear = concept_faith("label_probe")
    # the causal method should track ground truth much better than the linear one
    assert rho_causal > 0.6
    assert rho_causal - rho_linear > 0.3


def test_sae_beats_pca_on_recovery_in_superposition():
    spec = ConceptSpec(dim=20, groups=[GroupSpec(kind="point", n=28)],
                       label="AND(z0, z1)", geometry="random", noise=0.05)
    tr, va, res = _train(spec, n=6000, epochs=40, model_kwargs={"hidden": 128, "n_layers": 2})
    ctx = build_context(res.model, tr, seed=0)
    pca = recovery(build_method("pca").fit(ctx).discovered_concepts(), tr.world)
    sae = recovery(build_method("relu_sae", overcomplete=4.0, epochs=200).fit(ctx).discovered_concepts(), tr.world)
    assert sae.mean_matched_cosine > pca.mean_matched_cosine
