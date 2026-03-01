# Addendum to Preregistration: Ant Colony Simulations — Modules 1 and 2
## C-H2 Threshold Miscalibration

**Addendum date: 2026-03-01**
**Original preregistration: docs/preregistration-ant-modules.md (locked 2026-03-01)**
**This document does not modify the original preregistration.**

---

## Purpose

This addendum identifies a preregistration quality failure in hypothesis C-H2. The
original locked document is left intact. The inconsistency documented here is visible
in the locked document and is not retroactively constructed.

---

## C-H2 As Preregistered (verbatim from locked document)

> **C-H2 (τ=20 mid-range)**: SCI(20) falls in the 50–90% range, confirming the
> preregistered τ is neither trivially short (most events unclosed) nor so long that
> SCI ≈ 1.0 (all events trivially close). A τ where SCI is near 0 or 1 has poor
> discriminative power.

---

## Internal Inconsistency in the Locked Document

The locked preregistration document contains both C-H2 (above) and the following
pilot result, under **Smoke Test Results (Verified Before Lock)**:

> `SCI: 0.0036 (128486 Delta-events, 458 resolved)`

SCI = 0.0036 is incompatible with the C-H2 threshold of 0.50–0.90 by two orders of
magnitude. Both values appear in the same locked document. The SCI scale was therefore
not unknown at the time of lock — it was explicitly stated in the document alongside
the criterion it falsifies.

---

## Cause of the Miscalibration

C-H2 was written by analogy with QRC (Query-Response Coupling) from the explicit-Δ
system, where QRC runs 0.81–0.97 across conditions and the mid-range criterion
(0.50–0.90) is well-calibrated. SCI is structurally analogous to QRC but operates at
a fundamentally different scale: in the pheromone system, most Δ-events (ants below
gradient threshold) are never resolved within τ=20 steps because gradient reinforcement
depends on a returning forager passing through the exact forward neighborhood of the
stalled ant — a rare local event. QRC in the MARL system closes routinely because
responses are directed and deliberate. The SCI scale was not adjusted for this
difference before the threshold was written.

The failure is therefore not that the prediction turned out wrong — it is that the
threshold was already falsified by the pilot result the document itself cites.

---

## Reporting Classification

**C-H2 will be reported as a preregistration quality failure, not a result failure.**

Specifically:
- The criterion was miscalibrated relative to known pilot data at the time of lock.
- C-H2 will not be used to evaluate confirmatory support for Δ-Variable Theory.
- The confirmatory finding for Experiment C remains **C-H1** (SCI increases
  monotonically with τ), which holds cleanly across all 30 seeds and 5 τ levels.
- The characterization of the absolute SCI scale (max ≈ 0.011 at τ=80) is reported
  as descriptive/exploratory, not confirmatory.

---

## A-H2 — Separate Assessment

For completeness: **A-H2** (SCI peaks at same intermediate δ as crystallization rate)
is a distinct case. The pilot ran only at δ=0.10; no multi-delta SCI data existed at
lock. A-H2 was a genuine a priori prediction. It failed: SCI declines monotonically
with δ (highest at δ=0.04, 0.0091; lowest at δ=0.30, 0.0025), while crystallization
rate peaks at δ=0.20 (83%). A-H2 is reported as a **legitimate falsified prediction**,
with the post-hoc explanation that low-δ rarity of Δ-events produces high per-event
closure rates, while high-δ event flooding dilutes SCI. This explanation was not
available at the time of preregistration.

---

## What Is Unchanged

- The locked preregistration document is not modified.
- The campaign data, parameters, and analysis are unchanged.
- B-H1, B-H2, B-H3 (Experiment B) are all confirmed and unaffected by this addendum.
- A-H1, A-H3 (Experiment A) are unaffected.
- C-H1 (Experiment C) is confirmed and unaffected.

---

*This addendum was written on 2026-03-01, the same day as the preregistration lock,
upon reviewing C-H2 against the campaign results. The internal inconsistency in the
locked document was identified during post-campaign analysis.*
