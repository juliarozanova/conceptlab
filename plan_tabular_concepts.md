# Tabular Concepts Testbed — Implementation Plan (branch: `tabular-concepts`)

## Context

Extends conceptlab from embedding-space blobs to the full chain **tabular events → intermediate concepts → label**, to validate *compositional* interpretability methods — centrally Integrated Conceptual Sensitivity (ICS: Integrated Gradients × Concept Activation Vectors, Schrouff et al. 2021, arXiv:2106.08641) — against exact ground truth. The target application is the fraud testbed in `fraud/`: a causal TabBERT model predicting fraud, whose decisions we want to explain with respect to (a) input values and (b) compound intermediate concepts ("short burst", "unusual country", typologies). This suite exists to establish, before trusting any explanation of the fraud model, which methods survive when the right answer is known.

Companion document: `fraud/plan.md` §8 (concept & explainability layer) implements the data contract this plan defines.

## Agreed design decisions

1. **Pure mediation with identity concepts.** The label depends on inputs *only* through the concept layer. "Direct" input influence is represented by **level-0 identity concepts** (single-column predicates or passthroughs: `amount>500`, `channel=CNP`, `country≠home`). Compound concepts (level 1+) are logical combinations of level-0s. Consequence: input-level and concept-level attribution are the same DAG read at different granularity, and the test "can a method tell a compound concept from its own constituents?" falls out naturally.
2. **Mode A (glass box): hand-built soft logic.** Differentiable relaxations of the concept formulas — nothing trained except (optionally) the label head. Analytic ground truth for gradients and concept coordinates.
3. **Mode B (from scratch): end-to-end causal transformer**, never told about concepts. Methods are graded against **model-level** interventional truth (toggle concept in the generator → Δ model output), with a mandatory **concept representation audit** (probes per layer × position). The data-level vs model-level gap is exported as a shortcut-detection signal.
4. **Sequences from day one.** Event streams + causal transformer are the primary setting; `seq_len: 1` configs degenerate to row-level and serve as the debugging rung.
5. **Three-tier attribution ground truth** (below). Tiers 1–2 are hard gates; tier 3 grades ranking, with both interventional and Shapley definitions exported and their disagreement flagged.
6. **Methods v1:** ICS, TCAV, CAV-projection ablation, concept-probe patching, input-level attribution aggregated through the DAG.
7. **Interop = documented data contract** (`CONTRACT.md`, defined here, implemented by fraudgen and eventually by real-data exporters).

---

## 1. Tabular world (`conceptlab/tabular.py`)

A declarative `TableSpec`: named columns with types — `numeric` (distribution params), `categorical` (cardinality, optionally Zipf), `timestamp`. Event streams are sampled per sequence: a base process draws rows; designated **latent episode variables** (e.g. `on_trip ∈ {0,1}` with sticky transitions, `session` bursts via a simple self-exciting intensity) modulate row distributions across time — these latents are the substrate for high-level concepts and are exported as ground truth.

A toy-fraud flavored default schema ships (`amount, mcc, channel, country, entry_mode, Δt, merchant_id`) so configs read like the fraud target, but the machinery is schema-generic.

## 2. Concept DSL extension (`conceptlab/labeldsl.py`)

Extend the existing safe-AST DSL with:

- **Row atoms (level 0):** `GT(col, v)`, `LT`, `EQ(col, v)`, `IN_SET(col, ...)`, `PASS(col)` (normalized passthrough). Each atom *is* an identity concept.
- **Temporal/window atoms:** `COUNT_W(pred, window)` (events matching pred in trailing window), `ANY_W`, `SUM_W`, `TIME_SINCE(pred)`, `NOVEL(col)` (value unseen in this sequence's history), `EPISODE(name)` (reads a generator latent — the only atom not computable from rows alone).
- **Levels:** a `ConceptGraph` object holds named concepts with explicit `level` and `parents`; the label formula may reference concepts only (mediation is enforced structurally — the label node's parents are concepts, never raw columns).

`ConceptGraph` serializes to the contract (§7) and exposes `support()`, per-example evaluation, and per-example **pivotality** (tier 2) by re-evaluation under toggles.

## 3. Soft logic / Mode A (`conceptlab/softlogic.py`)

Compile a `ConceptGraph` to a differentiable torch module:

- `GT(col,v)` → `σ((x_col − v)/τ)`; `EQ` on categoricals → indicator via (fixed) embedding dot products; `AND` → product t-norm; `OR` → `1−∏(1−·)`; `XOR` → soft-XOR; window atoms → soft counts over the sequence (masked cumulative sums, differentiable in amounts/times but not in event count — documented).
- Temperature **τ** is a config axis: τ→0 recovers the crisp logic; larger τ makes the oracle "neural-textured". Reported per-τ; gradient starvation at low τ is an expected, *labeled* finding, not noise.
- Output: soft concept vector `ĉ` per (event, sequence) → embedded into R^D via the existing `World`-style anchors (concepts are the "latents", reusing the blob machinery and its geometry options) → fixed or trained linear head → label.
- Oracle exports: exact concept coordinates (anchor directions = true CAVs), exact per-example soft concept values, analytic gradients.

## 4. Mode B model (`conceptlab/models.py` extension)

`TabTransformer`: per-field embeddings (numeric: piecewise-linear encoding; categorical: embedding tables) fused into **one token per event** (concat → projection), then the existing causal `ToyTransformer` trunk over event tokens, CLS/last-token readout. This mirrors the fraud TabBERT shape at toy scale. The **input-attribution surface** is defined as the per-field embedding vectors (categoricals have no continuous path in raw space — this choice is part of the contract).

## 5. Ground truth (`conceptlab/eval.py` extension)

Per config, exported per example:

- **Tier 1 — structural:** support masks over concepts and over input columns (via the DAG). *Gate:* no attribution mass outside support (tolerance ε).
- **Tier 2 — pivotal:** which concepts (and level-0 atoms) flip this example's label / model output when toggled. *Gate:* locally inert concepts receive ~0.
- **Tier 3 — graded:** (a) interventional magnitude per concept (existing toggle machinery, extended to latents via **paired counterfactual re-simulation**: regenerate the sequence with the latent flipped, same noise); (b) **sampled concept-Shapley** (tens of concepts → cheap at toy scale). Both exported; per-config `definition_gap = 1 − ρ(GT_int, GT_shap)` flagged in reports. Both computed against the label function (data-level) and against each trained model (model-level; Mode B grading target).

**Concept audit:** for every concept × hookpoint × position(-pool): probe AUC + CAV fit quality. Attribution scores for a concept are reported alongside its best decodability; "method failed" vs "concept not represented" are distinguished everywhere.

## 6. Methods (`conceptlab/methods/concepts.py`)

All five consume the contract's concept example sets (never ground-truth directions, except the skyline):

| method | mechanism | notes |
|---|---|---|
| `ics` | IG along CAV directions (Schrouff et al.) | path/baseline choices exposed as kwargs — the extension point for **ICS variants** |
| `tcav` | directional-derivative sign statistics | global; graded on tier-1/global targets only |
| `cav_ablation` | project CAV out of activations → Δ output | causal-in-activation rival |
| `probe_patch` | set probe-decoded concept value to counterfactual, decode back | concept-bottleneck-style intervention |
| `input_agg` | input-level IG/ablation on field embeddings, aggregated to concepts via the known DAG | tests whether concept-level methods add anything over input-level + structure |

Plus `cav_skyline`: CAVs fit on ground-truth concept values (not example sets) — separates "CAV estimation error" from "attribution mechanism error".

## 7. Data contract (`CONTRACT.md` + `conceptlab/contract.py`)

Versioned schemas, all parquet/JSON, nullable GT columns so real data can comply:

- `events.parquet` — the model-input schema (strict subset; concept columns structurally excluded → leakage impossible by construction).
- `concepts.parquet` — `(sequence_id, event_id|window, concept_id, value, level)`.
- `concept_graph.json` — DSL serialization: atoms, parents, levels, label formula.
- `concept_sets/{concept}/` — positive/negative example manifests for CAV fitting, **stratified against the label and against correlated sibling concepts** (decorrelation is a validation-time assertion: max |corr(concept, label)| within a set below threshold).
- `attribution_gt.parquet` — tiers 1–3, model-level and data-level, nullable.

## 8. Experiments (v1 configs)

1. `tab_easy` — seq_len 1, few concepts, low τ + high τ: harness sanity; all gates should pass for the skyline; τ-starvation measured.
2. `tab_compound` — compound vs constituent test: label uses `burst AND cnp`; does each method separate the compound concept from its level-0 parts? (identity-concept design pays off here).
3. `tab_overdetermined` — OR-heavy label (two "typologies"): tier-3 definition gap is large by construction; methods scored against both definitions.
4. `tab_correlated` — burst⇢fraud correlation 0.95 with planted decorrelated controls on/off: the CAV-confounding experiment. Expect: without controls every method's "burst" attribution is secretly a fraud detector; with controls the good methods recover.
5. `tab_modeB` — same data, Modes A and B side by side; audit-conditioned scoring; data-vs-model gap reported (shortcut detection).

Each run reuses the existing report pipeline with new panels: gate pass/fail matrix, tier-3 ranking ρ per definition, audit heatmap (concept × layer × position), τ-sweep curves.

## 9. Pitfalls → design requirements (from the grill)

- **Concept correlation kills CAVs** → decorrelated controls are *planted*, example sets stratified, and decorrelation asserted at validation time (§7).
- **Latent concepts need counterfactual re-simulation** → generator API: `resimulate(sequence, toggle_latent, same_noise=True)` from day one (§5).
- **CAV placement is position-dependent in causal models** → audit and methods sweep hookpoint × position; never a single-layer assumption (§5).
- **Soft-logic saturation** → τ is an experiment axis, reported per-τ (§3).
- **Categorical IG paths** → input-attribution surface = field embeddings, stated in the contract (§4).
- **Overdetermination is common, not edge** → tier 3 dual-definition policy; high-gap configs labeled "ranking not well-posed" (§5).
- **Leakage** → model-input schema is a structural subset of the export (§7).

## 10. Implementation order

1. DSL extension + `ConceptGraph` (+ tests: truth tables incl. window atoms, support/pivotality on hand-built cases).
2. Tabular world + latent episodes + paired counterfactual re-simulation (+ determinism tests).
3. Soft-logic compiler (Mode A) — verify: τ→0 matches crisp DSL exactly on all configs.
4. GT tiers 1–3 (+ tests: pivotality vs brute force; Shapley sampling vs exact on ≤10 concepts).
5. Contract module + exporter; `tab_easy` end-to-end with skyline only.
6. `TabTransformer` (Mode B) + audit.
7. The five methods (+ skyline); experiments 1–5; report panels.
8. Freeze contract v1; hand `CONTRACT.md` reference to `fraud/plan.md` §8.

## Verification

- Skyline passes all gates on every config (else harness bug).
- τ→0 Mode A equals crisp logic bit-for-bit on labels and concepts.
- `tab_correlated` without controls shows the predicted CAV confounding (a *positive* result for the harness).
- Tier-3 GT sanity: on `tab_overdetermined`, interventional GT is all-zero on jointly-sufficient examples while Shapley is not (the §5 worked example, asserted in tests).
- Mode B audit: at least one config where a concept is undetectable and all methods correctly score ~0 model-level importance for it.
