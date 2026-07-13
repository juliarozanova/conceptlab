#!/usr/bin/env python3
"""Generate docs/wiki/concepts.html from the concept catalog.

The catalog here is the documentation source of truth for the domain-concepts
taxonomy (plan_domain_concepts.md §1). Regenerate with:

    .venv/bin/python scripts/build_wiki_concepts.py
"""

from __future__ import annotations

from pathlib import Path

# (name, level, definition, why-it-matters, corr_lo, corr_hi, legit_pathway, implemented)
# corr band None -> composite/typology (composition or defining set goes in `legit`)
L0 = [
    ("card_not_present", "The purchase happened without the physical card — online, in-app, or by phone.",
     "CNP removes the chip/PIN safety net, so most stolen-card fraud happens here. Its CAV should be trivially findable — a good smoke test for any concept method.", .1, .2, "most online commerce", True),
    ("token_wallet", "Paid via a phone wallet (Apple/Google Pay) using a device-bound token.",
     "Tokens are hard to steal, so this concept should get <i>negative</i> fraud attribution — a nice sanity check that methods can express protective concepts.", .0, .05, "wallet adopters", False),
    ("magstripe_fallback", "A chip card was swiped on its magnetic stripe instead of using the chip.",
     "Skimmed (counterfeit) cards can't fake the chip, so cloners force the stripe. A classic verification-evasion tell.", .2, .35, "degraded terminals", False),
    ("atm_withdrawal", "Cash taken out at an ATM.",
     "Cash is the end of the audit trail — fraudsters convert to cash fast. But most ATM use is mundane, so attribution must depend on <i>context</i>, not the flag alone.", .0, .05, "normal cash use", True),
    ("moto", "A mail-order / telephone-order payment (card details read out loud).",
     "The weakest verification rail still in use; a small but real fraud channel.", .1, .2, "legacy billers", False),
    ("recurring_flag", "The network marks this as a subscription-style repeat payment.",
     "Recurring payments are pre-agreed and low-risk — another protective concept.", .0, .05, "subscriptions", False),
    ("micro_amount", "A very small payment (under ~€2).",
     "Card testers 'ping' stolen cards with tiny charges to see if they work. Legit micro-payments (parking, vending) keep this from being a pure fraud flag.", .15, .3, "parking, vending, app trials", True),
    ("round_amount", "A suspiciously tidy number: 50, 100, 200…",
     "Humans buying things produce messy totals; transfers, laundering and gift-card loads produce round ones.", .05, .15, "gifts, fuel presets", False),
    ("just_below_limit", "An amount just under a known authorization or step-up threshold (e.g. €498 when checks start at €500).",
     "Fraudsters <i>structure</i> amounts to stay under extra verification. Detecting this tests whether models learn thresholds — and whether numeric encodings (PLE) can even represent them.", .2, .4, "budget-capped shoppers (planted)", False),
    ("amount_tier_high", "A large payment (over ~€500).",
     "Bigger tickets, bigger losses — fraud concentrates value here. On its own it's weak; combined with novelty concepts it becomes decisive.", .1, .25, "electronics, travel, rent-like bills", True),
    ("psych_price", "A retail-style price ending in .99 or .95.",
     "Real merchants price psychologically; fraudulent test amounts rarely bother. Mildly protective.", .0, .05, "retail pricing", False),
    ("fenceable_mcc", "The merchant sells easily-resellable goods (electronics, jewelry, luxury).",
     "Stolen money becomes goods that become cash. The 'cash-out' half of most stolen-card stories.", .15, .3, "normal shopping", True),
    ("gift_card_mcc", "A prepaid / gift-card merchant.",
     "Gift cards are near-cash with no name attached — a favorite laundering exit.", .25, .4, "gifting seasons (planted spikes)", False),
    ("gambling_mcc", "A betting or casino merchant.",
     "Fast, semi-anonymous value movement; also a genuine hobby — a good medium-correlation test case.", .1, .25, "recreational bettors", False),
    ("crypto_mcc", "A cryptocurrency on-ramp.",
     "Irreversible, pseudonymous — the modern cash-out. High-risk but with a real legitimate user base.", .2, .35, "crypto-active customers", False),
    ("money_transfer_mcc", "A remittance or money-transfer merchant.",
     "Where laundering and mule activity route funds; also how millions send money home. Context is everything.", .15, .3, "remittance-sending families", False),
    ("fuel_pump_preauth", "The pre-authorization pattern of an automated fuel dispenser.",
     "A distinctive-but-benign pattern; useful as a 'should be ignored' concept in attribution tests.", .0, .05, "drivers", False),
    ("digital_goods", "In-app purchases, game credits, digital content.",
     "Instant delivery = instant loss; heavily targeted by both third- and first-party fraud.", .1, .2, "gamers, app stores", False),
    ("foreign_country", "The merchant's country differs from the cardholder's home country.",
     "Stolen card data crosses borders in seconds; people travel more slowly. Confounded by real travel — which is exactly why the on_holiday latent exists.", .1, .25, "travel, cross-border shopping", True),
    ("high_risk_corridor", "The merchant sits in a configured high-risk country list.",
     "Mirrors real issuer country-risk rules. Tests whether attribution respects list-membership concepts.", .2, .4, "diaspora corridors", False),
    ("cross_border_ecom", "Online purchase from a foreign merchant.",
     "The intersection of CNP and foreign — a compound identity concept whose parts are also concepts. Can methods keep them apart?", .15, .3, "global marketplaces", False),
    ("currency_foreign", "Charged in a currency other than the card's.",
     "Mostly redundant with foreign_country — deliberately so: a correlated-sibling test for CAV separation.", .1, .2, "travel, imports", False),
    ("night_local", "Between midnight and 5am in the cardholder's local time.",
     "Fraud bots don't sleep, but neither do shift workers. A weak signal that must not be over-attributed.", .05, .15, "shift workers, night owls", False),
    ("weekend", "Saturday or Sunday, local time.",
     "Near-zero signal by design — a null concept that any faithful method should ignore.", .0, .05, "everyone", False),
    ("is_declined", "The issuer said no.",
     "Declines trail fraud (testing, drained balances) but mostly reflect innocent problems. The <i>pattern</i> of declines matters, not the flag.", .1, .25, "insufficient funds, expired cards", True),
    ("nsf_decline", "Declined for insufficient funds.",
     "The most innocent decline reason — helps methods separate decline <i>causes</i>.", .0, .1, "low balances", False),
    ("do_not_honor", "Declined with the catch-all 'do not honor' code.",
     "The issuer's own risk engine already fired. Tests whether models pick up second-hand risk signals — and whether attribution notices.", .2, .35, "issuer risk rules", False),
    ("cvv_failure", "The security code on the back of the card was wrong.",
     "Data-breach card dumps often lack CVV, so fraudsters guess. One of the strongest single verification tells.", .35, .5, "typos (rare)", False),
    ("avs_mismatch", "The billing address didn't match.",
     "Stolen identities come with stale addresses. Confounded by people who move house.", .25, .4, "recent movers, typos", False),
    ("three_ds_failed", "A 3-D Secure challenge (bank app confirmation) failed or was abandoned.",
     "The cardholder's own phone says no. Strong — but abandoned checkouts are common enough to keep it honest.", .3, .45, "UX abandonment", False),
    ("three_ds_frictionless", "3DS passed without a challenge (trusted-flow exemption).",
     "Protective: the risk engine already trusted this flow.", .0, .05, "trusted flows", False),
    ("pin_verified", "Chip + correct PIN at a terminal.",
     "The gold standard of card verification — strongly protective, and a good 'negative attribution' test.", .0, .05, "POS chip+PIN", False),
]

L1 = [
    ("short_burst", "Several transactions squeezed into a short window.",
     "Bursts are how card-testing looks — and how a Saturday shopping trip looks. The generator plants both, so a burst CAV encodes <i>burstiness</i>, not fraud.", .1, .25, "shopping sessions, checkout retries", True),
    ("rapid_fire", "Three or more transactions less than a minute apart.",
     "Human hands don't move that fast; scripts do. Much stronger than short_burst — the pair tests whether methods track signal <i>strength</i>, not just presence.", .3, .5, "double-tap retries (rare)", False),
    ("multi_merchant_burst", "A burst spread across several different merchants.",
     "Testers rotate merchants to dodge per-merchant velocity rules; mall crawls do the same innocently.", .25, .4, "market/mall trips (planted)", False),
    ("same_merchant_retry", "Repeated attempts at one merchant within minutes.",
     "Usually a failed payment being retried; occasionally a fraudster hammering one door.", .15, .3, "payment retries", False),
    ("escalating_amounts", "Amounts climbing steadily within a session.",
     "Fraudsters probe upward to find the card's ceiling.", .2, .4, "upsell funnels (weak, planted)", False),
    ("amount_stepping_down", "A decline followed by a smaller retry.",
     "Balance probing: 'no to €500? try €300.' The temporal <i>order</i> matters — a genuine test of sequence models over bag-of-features ones.", .3, .5, "honest balance guessing (rare)", False),
    ("micro_then_large", "A tiny payment followed shortly by one 20× bigger.",
     "The classic test-then-cash-out signature in miniature. Order-dependent, window-dependent — exactly what attention should catch.", .4, .6, "trial-then-subscribe (weak)", False),
    ("new_merchant", "First time this customer visits this merchant.",
     "Almost everyone shops somewhere new sometimes — near-null alone, decisive in combination.", .0, .1, "exploration", False),
    ("new_mcc", "First purchase ever in this merchant category.",
     "Life changes and gifts produce these innocently; account takeover produces them in clusters.", .05, .15, "life changes", False),
    ("new_country", "First transaction ever in this country.",
     "First trips happen — but paired with other novelty concepts it's an ATO cornerstone.", .15, .3, "first trips", False),
    ("new_device", "A device never seen on this account before.",
     "The single loudest ATO alarm. The generator plants routine phone upgrades so its CAV stays honest.", .2, .35, "phone upgrades (planted churn)", True),
    ("new_ip_subnet", "Connection from an unfamiliar network block.",
     "Credential thieves connect from their networks, not yours. VPNs and ISP churn provide the innocent path.", .15, .3, "ISP churn, VPNs", False),
    ("first_cnp_after_pos_history", "A card that has only ever been used in person suddenly appears online.",
     "Stolen card numbers surface online even when the plastic never left the wallet. A beautiful long-history concept: invisible to short-window models.", .2, .4, "first online purchase", False),
    ("impossible_travel", "Two in-person transactions whose locations couldn't be reached at any sane speed.",
     "Physics as a fraud rule. Deliberately near-pure (corr 0.7–0.9): the benchmark needs at least one concept where high attribution is simply <i>correct</i>.", .7, .9, "(almost none — by design)", False),
    ("atm_after_foreign_pos", "Cash withdrawal shortly after a foreign in-person purchase.",
     "A cash-out chaser following a counterfeit purchase; also just… travelers needing cash.", .2, .4, "travel cash", False),
    ("dormancy_break", "First activity after 30+ silent days.",
     "Fraudsters love forgotten cards — nobody is watching the statements. Returning travelers and seasonal cards keep it honest.", .1, .25, "seasonal use, returns from abroad", False),
    ("velocity_spike", "Spending rate far above this customer's own baseline.",
     "Everything about this concept is <i>relative to personal history</i> — it cannot be computed from one event, making it a pure test of history-aware models.", .15, .3, "holidays, emergencies", False),
]

L2 = [
    ("amount_z_extreme", "An amount more than 3 'personal standard deviations' above this customer's usual.",
     "Not 'is €800 a lot?' but 'is €800 a lot *for Maria*?' Tests whether models build per-entity baselines — and whether attribution credits the deviation, not the raw amount.", .2, .4, "big-ticket life purchases", False),
    ("unusual_mcc_for_customer", "A category this customer essentially never shops in.",
     "A vegetarian's card buying at a steakhouse. Personal-profile deviation, long-horizon by nature.", .1, .2, "gifts, exploration", False),
    ("unusual_hour_for_customer", "Activity at an hour this customer is normally silent.",
     "Same idea, time-of-day edition. Weak alone; a brick in the ATO wall.", .1, .2, "schedule changes", False),
    ("unusual_country", "A country outside this customer's whole geographic history.",
     "The distributional version of new_country: not just 'first time' but 'far outside the pattern'.", .3, .5, "genuinely rare trips", True),
    ("on_holiday", "The customer's travel episode latent is active — the <i>generator knows</i> they're traveling.",
     "The flagship confounder: travel explains away foreign/unusual activity. A faithful method must give it ~0 fraud attribution even though it co-occurs with scary-looking concepts.", .0, .1, "vacations (latent EPISODE)", True),
    ("life_event_shift", "A latent life change (move, new job) shifting the whole spending distribution.",
     "Innocent distribution shift — the stress test for every 'deviation from baseline' concept above.", .0, .05, "real life events (latent)", False),
    ("seasonal_peak", "A seasonal spending latent (December, sales events) is active.",
     "Everyone's velocity spikes together in December; models must not read the season as an attack.", .0, .05, "holidays (latent)", False),
    ("payday_window", "Within a few days after the salary lands.",
     "Spending rhythms follow paydays; a benign periodic structure sequence models should absorb.", .0, .05, "payday cycles", False),
    ("account_young", "The account is less than ~60 days old.",
     "Synthetic identities are always young; so is every genuine new customer. Base-rate reasoning required.", .15, .3, "new customers", False),
    ("credit_util_spike", "Credit utilization jumped sharply.",
     "Bust-outs max the line before vanishing. Emergencies do too, once.", .2, .4, "emergencies, big purchases", False),
    ("balance_drain", "One transaction consumes most of the available balance/limit.",
     "ATO end-game: take everything. Rent-like payments are the honest twin.", .3, .5, "rent-like payments", False),
    ("spend_trend_ramp", "Weeks-long upward trend in spending.",
     "The synthetic-identity 'nurture then bust' arc starts as a gentle, credible ramp — only visible over long horizons.", .15, .3, "income growth (planted)", False),
    ("fresh_merchant", "The merchant itself is less than ~30 days old.",
     "Fraudulent merchants are born, cash out, and die young. Every real business is also new once — merchant-side base rates.", .2, .4, "genuinely new businesses", False),
    ("tail_merchant", "A merchant in the bottom decile of popularity.",
     "The long tail is where collusion hides — and where niche commerce lives. Tests merchant-embedding quality directly.", .1, .25, "niche shops", False),
    ("merchant_volume_spike", "A merchant's own volume far above its baseline.",
     "Bust-out merchants spike before vanishing; viral products spike innocently.", .2, .4, "viral products, sales", False),
    ("hot_merchant", "The generator's per-merchant fraud-propensity latent is elevated.",
     "Some merchants really are riskier. Exported as truth so methods can be tested on second-hand (entity-level) risk concepts.", .4, .6, "mislabeled hot spots (planted noise)", False),
    ("cpp_exposed", "This card previously visited a merchant during its (latent) breach window.",
     "The common-point-of-compromise mechanism: exposure travels through a <i>shared merchant</i>, not this customer's behavior. Only merchant-aware models can even see it.", .3, .5, "most exposed cards stay clean", False),
    ("device_many_cards", "One device is associated with unusually many cards.",
     "Fraud operators run many victims from one machine; families share tablets. An entity-graph concept.", .4, .6, "families, shared tablets", False),
    ("shared_device_cluster", "A device seen across seemingly unrelated customers.",
     "Public terminals vs. mule herders — same observable, different worlds. Great hard case.", .3, .5, "public terminals", False),
]

# composites: (name, definition, why, composition)
L3A = [
    ("test_then_cashout", "The full card-testing arc: tiny probes (often failing CVV), then a big spend.",
     "A <i>compound</i> concept whose parts are themselves concepts. The benchmark's central question: do methods credit the compound, its parts, or double-count both?",
     "micro_then_large ∧ card_not_present ∧ (cvv_failure ∨ declined in window)"),
    ("enumeration_signature", "One merchant being hammered by many cards in rapid, CVV-failing succession.",
     "Lives on the <i>merchant</i> side — invisible in any single customer's history. Separates merchant-aware architectures from customer-sequence ones.",
     "many-cards-one-merchant latent ∧ rapid_fire ∧ cvv_failure"),
    ("geo_takeover_signature", "New device + unfamiliar country + failing address/3DS checks.",
     "The ATO chord: each note is weak, the chord is loud. Tests interaction-aware attribution (linear methods hear only notes).",
     "new_device ∧ unusual_country ∧ (avs_mismatch ∨ three_ds_failed)"),
    ("bustout_ramp", "A young account ramping spend and utilization toward a cliff.",
     "Long-horizon compound: none of its parts fire on any single event.",
     "spend_trend_ramp ∧ credit_util_spike ∧ account_young"),
    ("cash_conversion", "Any of the 'turn credit into cash' channels.",
     "A disjunction (OR) — the overdetermination generator. When two channels fire at once, single-toggle importance goes to zero and Shapley-style credit is required.",
     "fenceable_mcc ∨ gift_card_mcc ∨ crypto_mcc ∨ atm_withdrawal"),
    ("verification_evasion", "Any route around strong verification.",
     "Another disjunction, verification edition.",
     "magstripe_fallback ∨ three_ds_failed ∨ moto"),
    ("laundering_pattern", "Round amounts moving fast through transfer rails.",
     "A conjunction of individually-benign concepts — the anti-pattern of geo_takeover: here <i>no</i> part should get much solo credit.",
     "round_amount ∧ money_transfer_mcc ∧ velocity_spike"),
]

# typologies: (name, definition, why, defining set, implemented)
L3B = [
    ("card_testing", "A batch of stolen card numbers being validated with rapid micro-charges.",
     "The highest-volume fraud pattern in the wild.", "rapid_fire, micro_amount, card_not_present, cvv_failure", True),
    ("enumeration_attack", "A BIN attack: guessing card numbers in sequence against one merchant.",
     "Pure merchant-side signal — the planted reason merchant embeddings should win benchmarks.", "enumeration_signature, micro_amount", False),
    ("stolen_card_cnp", "Validated stolen cards being cashed out online.",
     "The cash-out that follows testing; the bread-and-butter CNP fraud story.", "test_then_cashout, cash_conversion, unusual_country", False),
    ("counterfeit_pos", "Cloned physical cards used in person after a skimming breach.",
     "The typology that <i>requires</i> CPP exposure — relational evidence across customers.", "cpp_exposed, magstripe_fallback, impossible_travel", False),
    ("lost_stolen_pos", "A physically lost/stolen card used before the owner notices.",
     "Old-fashioned and fast: cash and goods before the freeze.", "atm_withdrawal, velocity_spike, unusual_hour_for_customer", False),
    ("account_takeover", "Credentials stolen; the attacker becomes the customer.",
     "The long-baseline typology: everything looks plausible except against months of history.", "geo_takeover_signature, new_ip_subnet, balance_drain", True),
    ("synthetic_identity_bustout", "A fabricated identity nurtures good behavior, then maxes out and vanishes.",
     "Months of patience, one cliff — only lifetime-scale models see the shape.", "bustout_ramp, account_young, credit_util_spike", False),
    ("merchant_bustout", "A merchant builds plausible volume, spikes with stolen-card traffic, and dies.",
     "Fraud where the <i>merchant</i> is the criminal.", "fresh_merchant, merchant_volume_spike, hot_merchant", False),
    ("transaction_laundering", "A tail merchant washing payments for an unseen business.",
     "The quiet typology: individually boring events, collectively a pipe.", "laundering_pattern, tail_merchant", False),
    ("refund_abuse", "Systematic exploitation of refund flows, mostly in digital goods.",
     "First-party-adjacent: the 'customer' is real, the claims aren't.", "same_merchant_retry (refund rail), digital_goods", False),
    ("first_party_friendly", "A real customer disputing purchases they genuinely made.",
     "Behaviorally identical to innocence — <i>zero</i> defining concepts. The Bayes floor that keeps every method honest about its confidence.", "(none — undetectable by design)", True),
]

FAMILIES = [
    ("Transaction-intrinsic (level 0)", "l0", L0, """
Row-level facts readable off a single transaction — the payment channel, the amount's shape,
the merchant's category, the geography, and (new in this taxonomy) the <b>verification
signals</b>: CVV, address checks, 3-D Secure, PIN, decline reason codes. These are the
"identity concepts": each is simultaneously an input column and a concept, which is what lets
the testbed compare input-level and concept-level attribution on the same footing.
<br><br><b>Interpretability deduction:</b> level-0 concepts are the easy case — any method that
can't find and rank these has no business with the harder ones. They also include deliberately
protective (negative-attribution) and null concepts, testing that methods can say "this made
fraud <i>less</i> likely" and "this didn't matter at all"."""),
    ("Behavioral / windowed (level 1)", "l1", L1, """
Patterns over the <b>recent sequence</b> — bursts, retries, escalation, novelty, and physical
impossibilities. None of these exist in a single row; they are properties of order and timing.
<br><br><b>Interpretability deduction:</b> these concepts separate genuinely sequential models
from bag-of-features ones. If a method attributes a decision to <code>micro_then_large</code>,
it is implicitly claiming the model used <i>order</i> — and because we planted the pattern, we
can check that claim exactly."""),
    ("Entity-history & latent-state (level 2)", "l2", L2, """
Deviations from an entity's <b>own long-run profile</b> (customer, merchant, or device), plus
concepts that are really <b>latent world-states</b> the generator knows about (travel episodes,
life events, merchant breaches). Latents deserve special attention: <code>on_holiday</code> is
not computable from any row — it is a hidden cause. The generator exports it as truth and can
<i>re-simulate the world with it switched off</i> (a counterfactual), which is how we measure
its real causal effect.
<br><br><b>Interpretability deduction:</b> this family carries the confounding tests. Travel
<i>causes</i> foreign, unusual-looking activity without causing fraud; a faithful method must
attribute fraud decisions to the fraud-causing concepts and not the innocent latent riding
alongside them."""),
    ("Fraud-pattern composites (level 3a)", "l3", None, """
Named <b>combinations</b> of lower concepts — the "chords" analysts actually recognize:
test-then-cash-out, the takeover signature, cash-conversion channels. Written in the same logic
DSL, so their internal structure (AND vs OR) is known exactly.
<br><br><b>Interpretability deduction:</b> compounds make credit assignment testable at depth.
ANDs create <i>interaction</i> effects (each part is worthless alone — linear attribution
fails); ORs create <i>overdetermination</i> (single-toggle importance collapses to zero and
fair-share definitions like Shapley become necessary). Both failure modes are planted on
purpose."""),
    ("Typologies (level 3b)", "l3", None, """
The eleven fraud <b>storylines</b> the generator can inject, each hijacking a customer's (or
merchant's) stream. Every typology declares its <b>defining concept set</b> — the ground-truth
answer to "why is this transaction fraud?" — which is what fraud-model explanations are graded
against.
<br><br><b>Interpretability deduction:</b> because different typologies use different concepts,
no single global concept ranking can be right — explanations must be local. The typology layer
is what turns that from an opinion into a measurable property of the benchmark."""),
]


def _corrbar(lo, hi):
    left = lo * 100
    width = max((hi - lo) * 100, 3)
    return (f'<div class="corrbar" title="target fraud-correlation {lo:.2f}–{hi:.2f}">'
            f'<span style="left:{left:.0f}%;width:{width:.0f}%"></span></div>')


def _card(name, level_cls, definition, why, meta_html, implemented):
    impl = '<span class="chip impl">implemented</span>' if implemented else ""
    lvl = {"l0": "L0 · row", "l1": "L1 · window", "l2": "L2 · history/latent", "l3": "L3 · pattern"}[level_cls]
    return f"""<div class="ccard">
  <div class="head"><span class="name">{name}</span><span class="chip {level_cls}">{lvl}</span>{impl}</div>
  <div class="def">{definition}</div>
  <div class="why">{why}</div>
  <div class="meta">{meta_html}</div>
</div>"""


def build() -> str:
    sections = []
    for title, cls, rows, intro in FAMILIES:
        cards = []
        if cls in ("l0", "l1", "l2"):
            for name, definition, why, lo, hi, legit, impl in rows:
                meta = (f"corr band {lo:.2f}–{hi:.2f} · legit pathway: {legit}" + _corrbar(lo, hi))
                cards.append(_card(name, cls, definition, why, meta, impl))
        elif "composites" in title:
            for name, definition, why, comp in L3A:
                meta = f"composition: <code>{comp}</code>"
                cards.append(_card(name, cls, definition, why, meta, False))
        else:
            for name, definition, why, defining, impl in L3B:
                meta = f"defining concepts: <b>{defining}</b>"
                cards.append(_card(name, cls, definition, why, meta, impl))
        n = len(cards)
        sections.append(f"<h2>{title} — {n} concepts</h2>\n<p class='family-intro'>{intro}</p>\n"
                        f"<div class='grid'>\n" + "\n".join(cards) + "\n</div>")

    total = len(L0) + len(L1) + len(L2) + len(L3A) + len(L3B)
    body = "\n".join(sections)
    return _PAGE.format(total=total, body=body)


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>conceptlab wiki · concept catalog</title>
<link rel="stylesheet" href="wiki.css"></head><body><main>
<h1>The concept catalog</h1>
<div class="sub">Every concept the synthetic world can label — {total} concepts across five
families. Cards marked <span class="chip impl">implemented</span> exist in fraudgen v1; the
rest are specified in the <code>domain-concepts</code> plan.</div>
<div class="nav">
  <a href="index.html">Overview</a>
  <a href="generation.html">How the data is generated</a>
  <a class="here" href="concepts.html">Concept catalog</a>
  <a href="interpretability.html">Making interpretability deductions</a>
</div>

<div class="callout">
<b>How to read a card.</b> Each box gives the plain-English definition, why fraud analysts care,
and — where it changes what we can conclude — what the concept lets us <i>test</i>. The thin red
bar shows the <b>target correlation band</b>: how strongly this concept is allowed to co-occur
with fraud in the generated data. The band matters because the generator also plants a
<b>legitimate pathway</b> for every concept (shown in the footer): bursts also come from
shopping trips, new devices also come from phone upgrades. Without those innocent twins, a
detector for any concept would secretly just be a fraud detector — and every "concept
explanation" would be circular.</div>

<div class="statbox"><b>Stats note — "correlation" here.</b> We use correlation loosely to mean
"how much more often this concept is true on fraud than on legitimate transactions", scaled to
0–1. A band of 0.05–0.15 means "barely informative"; 0.7–0.9 means "almost always fraud when
present". The point of <i>bands</i> (rather than forcing everything to zero) is realism: some
concepts, like <code>impossible_travel</code>, really are near-proof of fraud, and a benchmark
that pretends otherwise teaches methods the wrong lesson.</div>

{body}

<footer>conceptlab wiki · generated by <code>scripts/build_wiki_concepts.py</code> — edit the
catalog there, not this file.</footer>
</main></body></html>
"""


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "docs" / "wiki" / "concepts.html"
    out.write_text(build(), encoding="utf-8")
    total = len(L0) + len(L1) + len(L2) + len(L3A) + len(L3B)
    print(f"wrote {out} ({total} concepts)")
