# Domain Concept Taxonomy & Rich Synthetic World — Plan (branch: `domain-concepts`)

## Why this plan exists

The v1 fraudgen implementation made a design error worth naming: it treated a handful of
*illustrative* concepts ("card not present", "amount>500", "short burst", "on holiday") as the
complete concept ontology, and built a minimal world just big enough to emit them. That inverts
the correct dependency. The concept set should be **derived from fraud domain knowledge** —
known fraud patterns, issuer/acquirer verification signals, and realistic payment behaviour —
and the synthetic world should be **rich enough to make every concept in that taxonomy
genuinely occur**, with both fraudulent and legitimate generative pathways.

This plan defines (§1) the full concept catalog (~70 concepts across 5 families and 4 levels),
(§2) the world-model refinements each concept family requires, (§3) the scale and data-shape
requirements for comparing **causal TabBERT variants at ~20M parameters**, (§4) contract v2,
and (§5) validation. It supersedes the concept sections of fraudgen v1; the v1 code is the
skeleton to be extended, not the spec.

Two principles carried over, one upgraded:

- **Every concept keeps a definition of record** (a DSL formula over rows/windows/entity state,
  or a named generator latent) and is computed *during generation*.
- **Decorrelation is now a spectrum, not a uniform target.** v1 forced controls toward zero
  correlation with fraud. Realistic domain structure is a spectrum: `impossible_travel` is
  almost always fraud (that's why analysts use it); `night_transaction` barely correlates.
  Each concept now declares a **target fraud-correlation band**, and the benchmark's hard case
  is distinguishing "correlated because defining" from "correlated because confounded" at
  matched correlation strengths — a much stronger test for CAV-style methods than uniformly
  decorrelated controls.

---

## 1. The concept catalog

Levels: **L0** row-computable identity/predicate · **L1** window/behavioral (recent sequence)
· **L2** entity-history/latent-state (long horizon or generator state) · **L3** fraud-pattern
composites and typologies. Corr band: target |corr(concept, fraud)| range the generator is
tuned to produce (with the legit pathway that keeps it there).

### 1.1 Transaction-intrinsic (L0)

| concept | definition sketch | corr band | legit pathway |
|---|---|---|---|
| `card_not_present` | channel = ecommerce/MOTO | 0.1–0.2 | most online commerce |
| `token_wallet` | Apple/Google-Pay token entry | ~0 (mild −) | wallet adopters |
| `magstripe_fallback` | entry = swipe on chip card | 0.2–0.35 | degraded terminals |
| `atm_withdrawal` | MCC/entry = ATM cash | ~0 | normal cash use |
| `moto` | mail/telephone order | 0.1–0.2 | legacy billers |
| `recurring_flag` | scheme recurring indicator | mild − | subscriptions |
| `micro_amount` | amount < €2 | 0.15–0.3 | parking, vending, trials |
| `round_amount` | amount ∈ {50,100,200,…} | 0.05–0.15 | gifts, transfers, fuel presets |
| `just_below_limit` | amount in (0.9·L, L) for a known auth/step-up limit L | 0.2–0.4 | budget-constrained shoppers (planted) |
| `amount_tier_high` | amount > customer-currency €500 | 0.1–0.25 | electronics, travel, rent-like |
| `psych_price` | x.99 / x.95 pattern | ~0 | retail pricing |
| `fenceable_mcc` | electronics, jewelry, luxury | 0.15–0.3 | normal shopping |
| `gift_card_mcc` | prepaid/gift-card merchants | 0.25–0.4 | gifting season (planted spikes) |
| `gambling_mcc` | betting/casino | 0.1–0.25 | recreational bettors |
| `crypto_mcc` | crypto on-ramps | 0.2–0.35 | crypto-active segment |
| `money_transfer_mcc` | remittance/P2P rails | 0.15–0.3 | remittance-sending segment |
| `fuel_pump_preauth` | AFD pre-auth pattern | ~0 | drivers |
| `digital_goods` | game credits, in-app | 0.1–0.2 | gamers, app stores |
| `foreign_country` | merchant country ≠ home | 0.1–0.25 | travel, cross-border ecom |
| `high_risk_corridor` | merchant country in configured risk list | 0.2–0.4 | diaspora/remittance corridors |
| `cross_border_ecom` | CNP + foreign acquirer | 0.15–0.3 | global marketplaces |
| `currency_foreign` | txn currency ≠ card currency | 0.1–0.2 | travel/import shopping |
| `night_local` | 00:00–05:00 cardholder-local | 0.05–0.15 | shift workers, insomniac shoppers |
| `weekend` | Sat/Sun local | ~0 | everyone |
| `is_declined` | auth declined | 0.1–0.25 | NSF, expired cards |
| `nsf_decline` | decline code = insufficient funds | ~0–0.1 | low balances |
| `do_not_honor` | decline code = DNH | 0.2–0.35 | issuer risk rules firing |
| `cvv_failure` | CVV mismatch result | 0.35–0.5 | typos (rare) |
| `avs_mismatch` | address verification fail | 0.25–0.4 | moved recently, typos |
| `three_ds_failed` | 3DS challenge failed/abandoned | 0.3–0.45 | UX abandonment |
| `three_ds_frictionless` | 3DS exemption path | mild − | trusted flows |
| `pin_verified` | PIN present & correct | strong − | POS chip+PIN |

### 1.2 Behavioral / windowed (L1)

| concept | definition sketch | corr band | legit pathway |
|---|---|---|---|
| `short_burst` | ≥k txns in trailing Δt | 0.1–0.25 | shopping sessions, checkout retries |
| `rapid_fire` | ≥3 txns < 60s apart | 0.3–0.5 | double-tap retries (rare) |
| `multi_merchant_burst` | burst spanning ≥m distinct merchants | 0.25–0.4 | mall/market crawls (planted) |
| `same_merchant_retry` | repeated attempts, same merchant, minutes | 0.15–0.3 | payment retries after decline |
| `escalating_amounts` | monotone amount ramp within session | 0.2–0.4 | upsell sessions (planted, weak) |
| `amount_stepping_down` | declined amount followed by smaller retry | 0.3–0.5 | genuine balance probing (rare) |
| `micro_then_large` | micro txn then >20× txn within window | 0.4–0.6 | trial-then-subscribe (planted, weak) |
| `new_merchant` | merchant unseen in customer history | ~0–0.1 | exploration |
| `new_mcc` | MCC unseen in history | 0.05–0.15 | life changes |
| `new_country` | country unseen in history | 0.15–0.3 | first trips |
| `new_device` | device id unseen | 0.2–0.35 | phone upgrades (planted churn) |
| `new_ip_subnet` | IP block unseen | 0.15–0.3 | ISP churn, VPNs |
| `first_cnp_after_pos_history` | first-ever CNP for a POS-only customer | 0.2–0.4 | first online purchase |
| `impossible_travel` | two POS txns violating geo-speed | 0.7–0.9 | (almost none — near-pure signal, by design) |
| `atm_after_foreign_pos` | ATM cash shortly after foreign POS | 0.2–0.4 | travel cash needs |
| `dormancy_break` | first txn after >30d silence | 0.1–0.25 | seasonal cards, returns from abroad |
| `velocity_spike` | rate ≫ personal baseline | 0.15–0.3 | holidays, emergencies |

### 1.3 Entity-history & latent-state (L2)

| concept | definition sketch | corr band | legit pathway |
|---|---|---|---|
| `amount_z_extreme` | amount z-score > 3 vs personal history | 0.2–0.4 | big-ticket life purchases |
| `unusual_mcc_for_customer` | MCC outside personal top-p mass | 0.1–0.2 | exploration, gifts |
| `unusual_hour_for_customer` | hour outside personal profile | 0.1–0.2 | schedule changes |
| `unusual_country` | country outside personal geo distribution | 0.3–0.5 | genuinely rare trips |
| `on_holiday` | travel episode latent active | ~0–0.1 | vacations (EPISODE) |
| `life_event_shift` | distribution-change latent (move, new job) | ~0 | real life events (EPISODE) |
| `seasonal_peak` | seasonal spend latent (December, sales) | ~0 | holidays (EPISODE) |
| `payday_window` | within k days after salary credit | ~0 | payday cycles |
| `account_young` | account age < 60d | 0.15–0.3 | genuinely new customers |
| `credit_util_spike` | utilization jump > x pts in window | 0.2–0.4 | big purchases, emergencies |
| `balance_drain` | txn consumes > y% of available | 0.3–0.5 | rent-like payments |
| `spend_trend_ramp` | weeks-scale upward spend trend | 0.15–0.3 | income growth (planted) |
| `fresh_merchant` | merchant age < 30d | 0.2–0.4 | genuinely new businesses |
| `tail_merchant` | merchant popularity in bottom decile | 0.1–0.25 | niche commerce |
| `merchant_volume_spike` | merchant's own volume ≫ its baseline | 0.2–0.4 | viral products, sales |
| `hot_merchant` | merchant fraud-rate latent elevated | 0.4–0.6 | (mislabeled hot spots — planted noise) |
| `cpp_exposed` | card visited a breached merchant in breach window | 0.3–0.5 | (exposure ≠ fraud; most exposed cards stay clean) |
| `device_many_cards` | device associated with ≥n distinct cards | 0.4–0.6 | families, shared tablets |
| `shared_device_cluster` | device seen across unrelated customers | 0.3–0.5 | public terminals |

### 1.4 Fraud-pattern composites (L3a) — compositional intermediates

These are the mid-level compounds that make **compound-vs-constituent** attribution testable at
multiple depths (the tab_compound experiment, but with realistic content):

| concept | composition (DSL over lower levels) |
|---|---|
| `test_then_cashout` | `micro_then_large ∧ card_not_present ∧ (cvv_failure ∨ is_declined in window)` |
| `enumeration_signature` | many-cards-one-merchant latent ∧ `rapid_fire` ∧ `cvv_failure` |
| `geo_takeover_signature` | `new_device ∧ unusual_country ∧ (avs_mismatch ∨ three_ds_failed)` |
| `bustout_ramp` | `spend_trend_ramp ∧ credit_util_spike ∧ account_young`… inverted for nurture |
| `cash_conversion` | `fenceable_mcc ∨ gift_card_mcc ∨ crypto_mcc ∨ atm_withdrawal` (fence channel) |
| `verification_evasion` | `magstripe_fallback ∨ three_ds_failed ∨ moto` |
| `laundering_pattern` | `round_amount ∧ money_transfer_mcc ∧ velocity_spike` |

### 1.5 Typologies (L3b) — campaign truth with defining-concept sets

Expanded from v1's four to eleven; each lists its defining set (per-event attribution GT):

1. **`card_testing`** — {`rapid_fire`, `micro_amount`, `card_not_present`, `cvv_failure`}
2. **`enumeration_attack`** (BIN attack: many cards, one merchant, sequential PANs) —
   {`enumeration_signature`, `micro_amount`}; detectable primarily *merchant-side* — rewards
   merchant embeddings, not customer history
3. **`stolen_card_cnp`** (cash-out after testing) — {`test_then_cashout`, `cash_conversion`, `unusual_country`}
4. **`counterfeit_pos`** (skimming; follows CPP exposure) — {`cpp_exposed`, `magstripe_fallback`, `impossible_travel`}
5. **`lost_stolen_pos`** — {`pin_verified`=false context, `atm_withdrawal`, `velocity_spike`, `unusual_hour_for_customer`}
6. **`account_takeover`** — {`geo_takeover_signature`, `new_ip_subnet`, `balance_drain`}
7. **`synthetic_identity_bustout`** — {`bustout_ramp`, `account_young`, `credit_util_spike`}
8. **`merchant_bustout`** — {`fresh_merchant`, `merchant_volume_spike`, `hot_merchant`}
9. **`transaction_laundering`** — {`laundering_pattern`, `tail_merchant`}
10. **`refund_abuse`** — {`same_merchant_retry` variant on refund rail, `digital_goods`}
11. **`first_party_friendly`** — {} (Bayes floor, unchanged)

Plus a **novel typology reserved for the test window** (config-selectable from a held-out
parameterization) to keep the NLL-anomaly/generalization test from the original fraud plan.

**Catalog totals:** ~32 L0 + ~17 L1 + ~19 L2 + 7 L3a + 11 L3b ≈ **86 concepts**. Julia's original
examples all appear — as ordinary members.

---

## 2. World-model refinements (what the data must have so the concepts are real)

Each family above forces generator capabilities; this is the gap list vs fraudgen v1:

1. **Continuous timelines, not fixed windows.** Per-customer marked point process over an
   18-month simulated window (diurnal × weekly × payday intensity, Hawkes sessions), with
   training samples cut as **variable-length trailing histories** (see §3). Required by: all L2
   concepts, dormancy, trends, seasonality, label maturity.
2. **Auth & verification micro-model.** Issuer rule stack producing declines with reason codes,
   CVV/AVS/3DS/PIN results correlated with (but not determined by) fraud, and step-up limits
   that make `just_below_limit` meaningful. Required by: the entire verification concept group —
   the strongest real-world fraud features, entirely absent from v1.
3. **Richer event schema** (~22 fields): ts, amount, currency, mcc, merchant_id,
   merchant_country, entry_mode, channel, device_id, ip_subnet, auth_result, decline_code,
   cvv_result, avs_result, three_ds_result, pin_present, recurring_flag, balance_after,
   credit_limit, available_credit, terminal_id (POS), acquirer_country.
4. **High-cardinality merchant universe** (~100K–1M, Zipf, lifecycle: birth/ramp/death, breach
   events, per-merchant fraud-propensity latent). Required by: merchant-side concepts,
   `enumeration_attack`, CPP, and — critically — the **merchant-embedding comparisons** in the
   TabBERT architecture bake-off.
5. **Device/IP graph.** Device sets per customer with churn (upgrades), shared devices
   (households, public terminals), fraud-operator device reuse across victims. Required by:
   device concepts and ATO realism.
6. **Card & account state.** Balances, credit limits, utilization; salary credits (payday);
   card reissue after confirmed fraud (behavior persists, id changes).
7. **Latent episode system** (generalizing v1's `on_trip`): travel, life-event shift, seasonal
   peaks, merchant breach windows, campaign phases — all exported, all resimulable
   (counterfactual API from the grilled plan is unchanged and now pays off at scale).
8. **Label mechanics restored** from the original fraud plan: chargeback reporting delay
   (Gamma, median ~25d), 10–15% never-reported fraud, `label_reported` vs `label_true`,
   maturity-aware splits.
9. **Correlation-band tuning loop.** A calibration pass measures each concept's realized
   fraud-correlation and adjusts the paired legit pathway's quota until it lands in its band —
   the §1 bands are *specs, asserted in validation*, not hopes.

## 3. Scale & shape for 20M-parameter causal TabBERT comparison

A 20M-parameter model comparison is meaningless on 6K fixed windows. Targets:

- **Volume:** default config **~50M events / 200K customers**; stretch config 500M+ events via
  the sharded generation scheme from the original plan (global planning pass → independent
  customer shards, counter-based RNG). One fused token per event ⇒ ~50M tokens default —
  enough to differentiate 20M-param architectures on pretraining behavior; the stretch config
  approaches Chinchilla-adequate scale for fine-grained ranking.
- **Sequence shape:** trailing histories of **64–512 events** (heavy-tailed, realistic), packed
  with varlen attention for pretraining; decision-scoring evaluation at the last event.
  This is what makes the KV-cache vs SSM-hybrid vs snapshot history-amortization comparison
  from the architecture discussion actually testable.
- **Cardinalities:** merchant_id 100K–1M (Zipf α≈1), device_id ~500K, terminal_id ~200K —
  the embedding-table / hashed-embedding / InfoNCE techniques need real cardinality pressure.
- **Fraud rate:** 0.1–0.3% of events (windows oversampled for the explainability suite, which
  needs balanced concept example sets — the two consumers get different *samplings* of the
  same world).
- **Planted architecture-discriminating signals** (unchanged in spirit from the original fraud
  plan, now with concept-level provenance): long-baseline ATO (lookback 50–500), merchant-side
  enumeration (embedding signal), Δt bursts (time-encoding signal), novel test-window typology
  (NLL signal), difficulty tiers (cascade/early-exit routing).
- **Reference bake-off suite** (fraudgen side, definition of done): logistic → GBM-on-windows →
  **causal TabBERT ~20M** (event-token fusion, PLE numerics, learned merchant embeddings) with
  required separations: TabBERT > GBM overall; TabBERT edge concentrated on long-baseline and
  merchant-relational provenance tiers; window-truncation ablation degrades exactly those.

## 4. Contract v2 (conceptlab side)

- **Variable-length sequences**: events.parquet gains `history_len`; loaders pad/pack.
- **Event-level concept values** (not just decision-point): `concepts.parquet` keyed by
  (seq_id, t, concept) for window/L1 concepts; L2/L3 stay decision-level. Enables per-position
  CAV placement studies (where in the sequence is `short_burst` decodable?).
- **Concept metadata**: family, level, definition-of-record string, target corr band, realized
  corr, legit-pathway description — the report can then show spec-vs-realized per concept.
- **Example-set manifests** per concept with stratification report (label balance, sibling-
  concept balance within the family — matched-correlation confounder pairs included
  deliberately as the hard CAV test).
- conceptlab additions: a taxonomy-aware audit (decodability by family × level × position), and
  a **matched-correlation confounder experiment** replacing `tab_correlated`'s single case:
  for each of ~5 (defining, confounder) pairs at matched |corr|, does the method separate them?

## 5. Validation (definition of done)

1. Every §1 concept occurs with frequency ≥ configured floor, in both fraud and legit pathways
   (except declared near-pure signals like `impossible_travel`).
2. Realized fraud-correlation of every concept inside its declared band (the calibration loop
   converged) — published as a spec-vs-realized table in the dataset report.
3. Typology recall structure: GBM baseline strong on velocity typologies, weak on
   long-baseline ones; `first_party_friendly` at base rate (Bayes floor); enumeration attack
   invisible to customer-history-only models (merchant-side signal).
4. Leakage: no single raw field AUC > 0.75; concept columns structurally excluded from the
   model-input schema (unchanged).
5. Concept-attribution transfer: per-typology defining-set rank-AUC for causal methods
   exceeds TCAV by a margin on every typology with a nonempty defining set, and matched-
   correlation confounders are rejected (leakage < threshold) — the v1 finding, now tested
   across 11 typologies and ~86 concepts.
6. 20M TabBERT bake-off separations from §3 hold on the default config.

## 6. Implementation order

1. Contract v2 + concept-metadata machinery in conceptlab (small, unblocks everything).
2. fraudgen world upgrade: continuous timelines + auth micro-model + schema v2 (biggest lift).
3. Concept catalog implementation family-by-family (L0 → L1 → L2 → L3a), each with its legit
   pathway and correlation-band calibration; catalog is config-driven so subsets are cheap.
4. Typologies 5→11 with defining sets; novel-typology holdout mechanism.
5. Label mechanics (delay/noise/maturity).
6. Scale pass: sharded generation, packing-friendly export, 50M-event default config.
7. Bake-off suite (logistic/GBM/20M TabBERT) + validation report v2.
8. conceptlab: taxonomy audit + matched-correlation confounder experiments + refreshed reports.

**Non-goals (unchanged):** P2P/mule graphs, adaptive adversaries, FX realism, GNN exports.

## Honest scoping note

§1–§2 (catalog + world upgrade at ~5–10M events) is a self-contained deliverable that already
makes the explainability suite dramatically richer. The 50M+/20M-TabBERT bake-off (§3, step 6–7)
is a compute-bounded second phase — CPU generation is fine but model training at that scale
wants a GPU box; the plan keeps it separable so phase 1 doesn't block on hardware.
