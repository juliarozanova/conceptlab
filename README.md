# conceptlab

A ground-truth benchmark for **validating and comparing transformer interpretability methods**.

🌐 **[Live reports on GitHub Pages](https://juliarozanova.github.io/conceptlab/runs/index.html)** — cross-run comparison, easy_blobs, superposition_xor, circle_transformer.

Real-model interpretability has no ground truth — "did the SAE find the *real* features?" is unanswerable. `conceptlab` sidesteps this by generating synthetic data where the concepts, their geometry, and their causal importance for the label are **known by construction**. Interpretability methods are then scored on how well they

1. **recover** the planted concepts from a trained model's activations, and
2. **attribute label importance** to those concepts correctly.

Because the data is synthetic, we also get **causal ground-truth importance** for free — by intervening in the data generator (flip a concept, regenerate, measure the change in the model's output) rather than approximating it with a method under test.

## Design

Three cleanly separated layers so any method can be compared on any dataset/model:

```
GROUND TRUTH (known)          MODEL (trained)              METHODS (under test)
concept anchors, geometry --> toy model learns          --> each method sees only
label function f(concepts)    label from embeddings         (model, data), outputs:
causal importance             activations recorded          - discovered concepts
                                                            - importance scores
                                        EVAL: score method output against ground truth
```

## Quickstart

```bash
uv sync                                             # create env (CPU torch)
uv run conceptlab-run --config configs/easy_blobs.yaml
uv run conceptlab-run --config configs/superposition_xor.yaml
uv run conceptlab-run --config configs/circle_transformer.yaml
open docs/runs/index.html                           # cross-run comparison
```

**Pre-generated reports** are live on GitHub Pages:
- **[Cross-run index](https://juliarozanova.github.io/conceptlab/runs/index.html)** — headlines from all runs
- **[easy_blobs](https://juliarozanova.github.io/conceptlab/runs/easy_blobs/report.html)** — harness sanity check
- **[superposition_xor](https://juliarozanova.github.io/conceptlab/runs/superposition_xor/report.html)** — SAEs beat PCA; linear importance fails on XOR
- **[circle_transformer](https://juliarozanova.github.io/conceptlab/runs/circle_transformer/report.html)** — SAE dilution on circular concepts

Each run generates data, trains a toy model, runs every configured interpretability
method, evaluates against ground truth, and writes a self-contained `report.html`.

## What's inside

- `conceptlab/datagen.py` — `ConceptSpec`, geometry (point blobs, circular manifolds, correlated pairs, superposition), a small label DSL (AND/OR/XOR/majority/threshold), and generator-level causal importance.
- `conceptlab/models.py` — `ToyMLP` and `ToyTransformer`, both exposing `run_with_cache`.
- `conceptlab/methods/` — linear probe (skyline), PCA/ICA, ReLU & TopK SAEs, integrated gradients, direction ablation & activation patching, behind one `InterpMethod` interface.
- `conceptlab/eval.py` — Hungarian-matched concept recovery, coverage/redundancy (shattering vs. compact vs. dilution regimes), importance faithfulness, sufficiency.
- `conceptlab/report.py` — Plotly HTML reports.

## Starter experiments (they double as harness self-tests)

![conceptlab report](docs/preview.png)

*Above: the `circle_transformer` report. In the embedding-space panel the eight
ring-position blobs form a circle; the true concept is two axes (green) while the
SAE's discovered directions (yellow) fan out around the whole ring — the
**dilution regime** made visible.*

| Config | What it checks | Result |
|---|---|---|
| `easy_blobs` | Harness sanity: probe skyline should ace recovery; causal importance should isolate the label's concepts. | probe recovery **0.997**, ICA **1.00**, PCA 0.64; ablation/IG isolate z0,z1 ✓ |
| `superposition_xor` | SAEs should beat PCA on recovery; linear-only importance should fail on the XOR-coupled concepts. | recovery: SAE **0.57–0.62** > PCA **0.41**; faithfulness: causal **0.75–0.88** vs linear label-probe **0.03** ✓ |
| `circle_transformer` | Reproduces the SAE **dilution regime** on a circular concept. | SAE ring-coverage **0.5** vs linear baselines **0.25**, higher redundancy ✓ |

The linear label-probe's near-zero faithfulness (0.03) on the XOR label — while
ablation, patching and integrated gradients all score 0.75+ — is the benchmark's
discriminating result: it separates methods that capture concept *interactions*
from those that only see linear importance.

## Adding your own method

Subclass `InterpMethod` (see `conceptlab/methods/base.py`), set `can_discover`
and/or `can_score`, implement `discovered_concepts()` / `score_directions()`, and
register it in `conceptlab/methods/__init__.py`. Everything operates in the input
embedding space R^D, so a discovery method returns `(m, D)` unit directions and a
scoring method takes/returns per-direction importances. The integrated-gradients
implementation is the natural starting point for an "integrated conceptual
sensitivity" variant — swap the integration path or the target scalar.

See `plan.md` for the full design rationale.
