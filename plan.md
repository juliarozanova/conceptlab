# Interpretability Methods Testbed — Ground-Truth Concept Benchmark

## Context

Julia wants a reusable end-to-end experimental harness for **comparing and validating transformer interpretability methods** against known ground truth. The core idea: because real-model interpretability has no ground truth ("did the SAE find the *real* features?" is unanswerable), we build synthetic data where the concepts, their geometry, and their causal importance for the label are all **known by construction**. Interpretability methods are then scored on how well they (a) recover the concepts from activations, and (b) attribute label importance correctly.

This directly connects to themes in her knowledge wiki (`vibe_knowledge/knowledge_wiki/Mechanistic Interpretability/`): SAE capture regimes (shattering / compact / dilution), curved concept geometry (circles), superposition, and causal faithfulness. The testbed should be able to *reproduce these known phenomena on demand* — that's the validation that the harness itself works.

**Location**: `/mnt/d/dashboard/interpretability_comparisons/` (currently empty). Not a git repo yet — `git init` it as part of setup.

**Decisions made with user**:
- Staged model classes: Stage 1 = MLP on single vectors, Stage 2 = 1–2 layer transformer on token sequences. Same datagen + eval harness serves both.
- Visualisation: self-contained static HTML report per experiment run (Plotly embedded).
- Methods v1: all of — linear probes + PCA/ICA baselines, SAEs (ReLU+L1 and TopK), gradient attribution (IG-style, extensible to her "integrated conceptual sensitivity" idea), causal ablation/patching.

## Core Design Principle

Three-layer separation, so any method can be compared on any dataset/model:

```
GROUND TRUTH (known)          MODEL (trained)              METHODS (under test)
concept anchors, geometry --> toy model learns          --> each method sees only
label function f(concepts)    label from embeddings         (model, data), outputs:
causal importance             activations recorded          - discovered concepts
                                                            - importance scores
                                        EVAL: score method output against ground truth
```

The unique leverage of synthetic data: **causal ground-truth importance** is computed by intervening in the *data generator* (flip concept c, regenerate, measure Δlabel / Δmodel-output) — no method-based approximation needed.

## Package layout

```
interpretability_comparisons/
├── pyproject.toml            # uv project; deps: torch, numpy, plotly, scikit-learn, pyyaml, jinja2
├── conceptlab/
│   ├── datagen.py            # ConceptSpec, WorldModel, dataset generation
│   ├── models.py             # ToyMLP, ToyTransformer, activation hooks
│   ├── train.py              # training loop, checkpointing
│   ├── methods/
│   │   ├── base.py           # InterpMethod interface
│   │   ├── baselines.py      # LinearProbe (skyline), PCA, ICA
│   │   ├── sae.py            # ReLU+L1 SAE, TopK SAE
│   │   ├── attribution.py    # IG along directions; pluggable for custom methods
│   │   └── causal.py         # activation ablation, patching
│   ├── eval.py               # recovery + faithfulness metrics
│   ├── report.py             # HTML report builder
│   └── registry.py           # method/dataset registries for config-driven runs
├── configs/                  # YAML experiment configs
├── runs/                     # per-run output dirs (config copy, checkpoints, metrics.json, report.html)
└── experiments/              # thin scripts: run_experiment.py --config configs/x.yaml
```

## Component specs

### 1. Data generator (`datagen.py`)

`ConceptSpec` — declarative ground truth:
- **k concepts**, each an anchor direction in R^D with Gaussian blob noise σ (per-concept)
- **Geometry options per concept group**: point blob (default), *circle* (ordinal concept traced around a ring — reproduces the weekday-circle phenomenon), correlated pairs (non-orthogonal anchors, cos-sim configurable)
- **Superposition switch**: k > D with anchors as near-orthogonal random directions — the regime SAEs are motivated by
- **Binary/continuous concept activations** z ∈ {0,1}^k or [0,1]^k, with configurable co-occurrence structure

Sample: `x = Σᵢ zᵢ · anchorᵢ + ε`, label `y = f(z)` where `f` is a composite from a small DSL: AND/OR/XOR/majority/threshold over concept subsets (e.g. `y = (z₁ AND z₂) XOR z₃`). XOR-type labels matter — they defeat purely linear importance methods, giving the benchmark discriminative power.

Stage 2 (sequences): each token draws its own z; sequence label composes across positions (e.g. "concept A anywhere AND concept B at last token"). Token embedding = the same blob construction, used as a **frozen embedding layer**.

Every dataset serialises its full ground truth (anchors, f, z per sample) to the run dir.

**Ground-truth causal importance**: `true_importance(cᵢ) = E[|model(x with zᵢ flipped) − model(x)|]`, computed via the generator. Both data-level (does f depend on cᵢ) and model-level (did the model *learn* to use cᵢ) versions — the gap between them is itself interesting.

### 2. Models (`models.py`)

- `ToyMLP`: 2–3 hidden layers on x ∈ R^D
- `ToyTransformer`: 1–2 layers, few heads, frozen concept-blob embeddings, CLS-style readout
- Both expose `run_with_cache(x) -> (output, dict[hookpoint, activation])` (TransformerLens-style, but hand-rolled — no heavyweight dep for toy scale)
- `train.py`: standard loop, saves checkpoint + training curves; assert model reaches high accuracy before methods run (a model that hasn't learned f invalidates the comparison)

### 3. Method interface (`methods/base.py`)

```python
class InterpMethod:
    def fit(self, model, dataset, hookpoint) -> None
    def discovered_concepts(self) -> list[Direction | Manifold]   # unsupervised discovery
    def concept_importance(self) -> np.ndarray                    # importance of each discovered concept for label
    def score_directions(self, directions) -> np.ndarray          # optional: importance of *given* (true) directions
```

`score_directions` lets attribution methods (IG, ablation) be evaluated on ground-truth directions independently of discovery quality — separating the two halves of the question ("find concepts" vs "rank their importance") is a key design point.

v1 implementations: `LinearProbe` (supervised skyline), `PCA`, `ICA`, `ReluSAE`, `TopKSAE`, `IntegratedGradients` (path integral of ∂output/∂(direction coefficient) — the natural home for the "integrated conceptual sensitivity" idea), `DirectionAblation`, `ActivationPatching`.

### 4. Evaluation (`eval.py`)

**Concept recovery** (discovered vs true):
- Hungarian-matched max cosine similarity per true concept + mean (MCC-style score from disentanglement literature)
- Coverage/redundancy: how many discovered features per true concept (detects shattering vs compact vs dilution regimes directly)
- Circular concepts: fit discovered feature set to the true ring; report fraction of ring covered

**Importance faithfulness**:
- Spearman correlation between method's importance ranking and ground-truth causal importance
- Detection of interaction effects: does the method notice that z₁, z₂ matter jointly under XOR?

**Sufficiency/completeness**: train a small probe from method's discovered-concept coefficients to the label; accuracy = "did the method find everything the label needs?"

### 5. Reports (`report.py`)

One self-contained `report.html` per run (Plotly JS inlined, jinja2 template):
- 2D/3D projection of embedding/activation space (PCA axes): true blobs coloured by concept, discovered directions overlaid as arrows, circular manifolds as curves
- Per-method scorecard table (recovery, faithfulness, sufficiency)
- Cross-method comparison bar charts
- Ground-truth panel: concept spec, label formula, model accuracy
- A top-level `runs/index.html` listing all runs for cross-run comparison

Follow the `dataviz` skill when building the report styling/palette.

### 6. Config-driven experiments

YAML config = {concept spec, label formula, model, methods list, seeds}. `run_experiment.py` executes: generate → train → run methods → eval → report. Multiple seeds averaged. Ship 3 starter configs that double as harness validation:

1. **`easy_blobs.yaml`** — 6 orthogonal concepts, D=32, MLP, `y = z₁ AND z₂`. Everything should ace this; it validates the harness, not the methods.
2. **`superposition_xor.yaml`** — 40 concepts in D=16, `y = (z₁ AND z₂) XOR z₃`. SAEs should beat PCA; linear-only importance methods should visibly fail on the XOR pair.
3. **`circle_transformer.yaml`** — Stage 2: one circular concept (8 positions on a ring) + point concepts, 1-layer transformer, label depends on ring position range. Expect to reproduce the SAE dilution regime from the wiki notes.

## Implementation order

1. `git init`, `uv init`, package skeleton
2. `datagen.py` + unit tests (label formula correctness, geometry sanity) — the ground truth must be trustworthy before anything else
3. `models.py` + `train.py`; verify `easy_blobs` trains to ~100%
4. `methods/`: baselines first (probe/PCA validates eval plumbing), then SAE, attribution, causal
5. `eval.py` metrics
6. `report.py` HTML reports
7. Run the 3 starter configs end-to-end; check expected qualitative results (see Verification)

## Verification

- **Unit**: label DSL truth tables; datagen geometry (anchor cos-sims, ring shape); Hungarian matching on hand-built cases where the answer is known.
- **End-to-end sanity (the real test of the harness)**: on `easy_blobs`, the linear probe skyline must hit ~1.0 recovery and the causal ground-truth importance must match the label formula (only z₁, z₂ matter). If the skyline can't ace the easy case, the harness is broken, not the method.
- **Known-phenomena reproduction**: `superposition_xor` should show SAE > PCA on recovery; `circle_transformer` should show SAE dilution on the ring (multiple features covering arcs). Reproducing these published qualitative results is the strongest evidence the benchmark measures the right thing.
- Open `report.html` for each run and visually confirm blobs/directions/scorecards render in both light and dark themes.
