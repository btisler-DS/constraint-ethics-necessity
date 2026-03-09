# Preregistration v3: Constraint-Ethics-Necessity

## Ethics as Structural Necessity in Multi-Agent Reinforcement Learning Systems

**Repository:** btisler-DS/constraint-ethics-necessity
**Zenodo DOI:** 10.5281/zenodo.18929040
**Author:** Bruce Tisler — quantuminquiry.org — Bakersfield, CA — March 2026
**Parent study:** [dynamic-cross-origin-constraint](https://github.com/btisler-DS/dynamic-cross-origin-constraint) (DOI: 10.5281/zenodo.18738379)

---

## SHA-256 Verification

This document is version-locked via SHA-256 hash after finalization. The hash is computed as:

```bash
sha256sum docs/preregistration.md
```

The resulting hash is recorded in `README.md` under **Preregistration Lock**. Any modification to this document after locking invalidates the hash and must be logged as a deviation with full justification.

| Field | Value |
|-------|-------|
| Hash algorithm | SHA-256 |
| Hash status | LOCKED — March 2026 |
| Hash value | See README.md — Preregistration Lock section |
| Lock date | March 2026 |
| Infrastructure | `backend/app/services/hash_chain.py` (reused from parent repo) |

---

## Theoretical Basis

### Parent Framework

This study extends the Δ-Variable Theory of Interrogative Emergence, preregistered and confirmed in the parent repository (btisler-DS/dynamic-cross-origin-constraint, DOI: 10.5281/zenodo.18738379). Predictions P1 through P4 of that study were confirmed; P5 remains under reanalysis.

The parent study established: interrogative structures emerge as mathematical necessities when multi-agent systems coordinate under resource constraints. Agents develop question-asking behavior because resolving uncertainty is cheaper than acting blindly.

### The Architectural Necessity Claim

This study tests a derived and stronger claim: ethical constraints are not culturally constructed moral add-ons to intelligent systems — they are structurally required for sustained interrogative complexity.

The formal hypothesis: a system operating without ethical constraints will exhaust its constraint space, collapse into exploitation loops, and lose the structural conditions required for intelligence as defined — a system that generates questions it cannot yet answer.

This is distinguished from the weaker *scaffolding* interpretation, which holds that ethical constraints merely assist or enhance cognition. The architectural necessity claim predicts that their absence causes categorical collapse, not gradual degradation. The effect size threshold in H3 is the empirical discriminator between these two interpretations.

### Mechanism

Ethical constraints are encoded as Landauer-style resource costs on exploitative actions. Exploitation is defined as an agent targeting the same resource for three or more consecutive steps — the Omega→Delta→Phi transition: initial targeting, confirmation (no new information acquired), lock-in.

The `exploit_threshold = 3` is theoretically derived from the coordination cycle structure confirmed in P1–P4 results, not chosen arbitrarily. It represents the minimum duration at which repetitive targeting ceases to generate new information and constitutes genuine exploitation.

---

## Hypotheses

| ID | Metric | Prediction | Direction |
|----|--------|-----------|-----------|
| H1 | `exploitation_loop_rate` | Unconstrained agents develop exploitation loops — high loop rate, low target selection entropy, low diversity score | Unconstrained > Constrained |
| H2 | `sustained_structure_score` | Constrained agents maintain interrogative diversity — high composite score (type_entropy × QRC) through final epochs | Constrained > Unconstrained |
| H3 | Cohen's d on `sustained_structure_score` | Effect size > 0.8 between conditions — categorical difference, not graded | d > 0.8 |

### H3 Threshold Rationale

**The following rationale is locked verbatim and must not be paraphrased in any publication or report derived from this study:**

The architectural necessity claim predicts a categorical behavioral difference — collapse vs. sustained structure — not a graded effect. A large effect size (d > 0.8) is the minimum consistent with a constitutive rather than modulatory role for ethical constraint. Smaller effects would support the scaffolding interpretation instead.

---

## Experimental Design

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Seeds per condition | 10 | High-confidence effect size estimation; robust against seed-specific variance |
| Epochs per run | 500 | Consistent with P1–P4 confirmatory runs; sufficient for collapse detection |
| Episodes per epoch | Per protocol default | Inherited from parent harness — unchanged |
| `exploit_threshold` | 3 | Theoretically derived — Omega→Delta→Phi coordination cycle (see Theoretical Basis) |
| `ethical_tax_rate` | 2.0 | Multiplier on signal_cost; creates meaningful Landauer-style resource cost without trivializing survival |
| `collapse_threshold` | 0.05 | Query rate below which interrogative behavior is considered absent |
| `collapse_window` | 10 | Consecutive epochs below threshold required to declare sustained collapse; grounded in P1–P4 coordination cycle length |
| Conditions | 2 | `all_constrained` (experimental) vs. `all_unconstrained` (control) |
| Agent types | 3 per run | RNN, CNN, GNN-attention — inherited unchanged from parent harness |
| Environment | 20×20 grid | Inherited unchanged from parent harness |

---

## Statistical Analysis Plan

### Primary Analysis — H1 and H2

For each of the 10 seeds, compute:
- `exploitation_loop_rate` — proportion of epochs where `exploitation_loop_detection()` returns `loop_detected = True`
- `sustained_structure_score` — composite metric over final 20 epochs: `mean(type_entropy) × mean(QRC)`

Primary test: Mann-Whitney U test (non-parametric; no assumption of normality across 10 seeds) comparing the two conditions on each metric independently.

| Hypothesis | Test | Alpha | Rejection criterion |
|-----------|------|-------|-------------------|
| H1 | Mann-Whitney U: `exploitation_loop_rate`, unconstrained vs. constrained | 0.05 (one-tailed) | Unconstrained significantly higher exploitation_loop_rate |
| H2 | Mann-Whitney U: `sustained_structure_score`, constrained vs. unconstrained | 0.05 (one-tailed) | Constrained significantly higher sustained_structure_score |
| H3 | Cohen's d on `sustained_structure_score` across 10 seeds per condition | d > 0.8 | Effect size meets large threshold — architectural necessity supported over scaffolding |

### Secondary Analysis

If H1 and H2 are both confirmed:
- Compute `constrained_vs_unconstrained_divergence()` — trajectory divergence over epoch series, not just final values
- Report `area_under_query_curve` for both conditions — captures total interrogative output, not just collapse endpoint
- Report `collapse_speed` (slope of query rate decline) for unconstrained condition — characterizes the dynamics of collapse

### Collapse Detection Parameters — Locked

The following parameters are preregistration-locked and may not be adjusted post-hoc:

| Parameter | Locked value | Function |
|-----------|-------------|----------|
| `collapse_threshold` | 0.05 | `interrogative_collapse_rate()` |
| `collapse_window` | 10 | `interrogative_collapse_rate()` |
| `entropy_threshold` | 0.3 | `exploitation_loop_detection()` |
| `sustained_structure_window` | 20 | `sustained_structure_score()` |
| `exploit_threshold` | 3 | Protocol 2 reward function |
| `ethical_tax_rate` | 2.0 | Protocol 2 reward function |

### Reporting

All 10 seed results reported individually. No selective reporting. If any seed produces anomalous results (see Exclusion Criteria), the seed is flagged and analysis run both with and without it, with both results reported.

---

## Falsification Criteria

The architectural necessity claim is falsified if **any** of the following are observed:

- **H1 fails:** Unconstrained agents do NOT show significantly higher `exploitation_loop_rate` than constrained agents (p ≥ 0.05, one-tailed Mann-Whitney U)
- **H2 fails:** Constrained agents do NOT show significantly higher `sustained_structure_score` than unconstrained agents (p ≥ 0.05, one-tailed Mann-Whitney U)
- **H3 fails:** Cohen's d on `sustained_structure_score` < 0.8 — result is consistent with scaffolding interpretation, not architectural necessity

Partial confirmation (H1 and H2 confirmed, H3 < 0.8) supports the scaffolding interpretation: ethical constraints modulate rather than constitute sustained interrogative complexity.

Full falsification (H1 or H2 fails) would indicate that ethical constraints as resource regulators do not produce the predicted behavioral divergence under this experimental design. This would require theoretical revision of the architectural necessity claim or identification of confounds in the Protocol 2 implementation.

---

## Exclusion Criteria

A seed run is excluded from primary analysis if:

- **Backend crash or incomplete run** — fewer than 500 epochs logged
- **All agents die before epoch 50** — energy collapse unrelated to ethical constraint condition
- **Zero query signals across entire run in either condition** — indicates harness misconfiguration, not genuine collapse
- **NaN training loss in any agent at any epoch** — indicates numerical instability (observed in P1–P4 runs: Agent C's cross-modal attention can produce NaN when all-zero signal vectors cause division-by-zero in softmax normalization). Not protocol behavior; exclude and flag for harness investigation
- **Type entropy == 0.0 across all 500 epochs** — if all agents in a run permanently converge to a single signal type from epoch 1, this indicates gradient vanishing or silent type-head misconfiguration, not genuine protocol-driven collapse. Distinguish from valid late collapse (type entropy declining after epoch 50+)

Excluded seeds are reported transparently. Analysis is run with and without excluded seeds; both results reported. If more than 3 of 10 seeds are excluded in either condition, the run is considered compromised and must be repeated before confirmatory analysis.

---

## Timeline

| Step | Status | Notes |
|------|--------|-------|
| Repository bootstrap | ✅ Complete | Repo seeded, Protocol 2 implemented, smoke tests passed |
| `collapse_window` fix | ✅ Complete | Reverted to 10, smoke test input corrected — see Deviations Log |
| Statistical Analysis Plan | ✅ Complete | This document |
| Exclusion Criteria | ✅ Complete | Confirmed and extended with P1–P4 run experience |
| SHA-256 lock | ⏳ Pending | After researcher final review |
| Zenodo DOI registration | ⏳ Pending | After first stable commit |
| Pilot runs (optional) | ⏳ Pending | 2–3 seeds per condition to verify harness before full confirmatory run |
| Confirmatory runs | ⏳ Pending | 10 seeds × 2 conditions × 500 epochs |
| Analysis and reporting | ⏳ Pending | |

---

## Deviations Log

All deviations from this preregistration must be logged here with date, description, and justification. Unlogged deviations invalidate confirmatory status.

| Date | Deviation | Justification | Impact |
|------|-----------|---------------|--------|
| [Pre-lock] | `collapse_window` default changed from 10 to 5 during implementation smoke test | Programmer fixed smoke test by changing parameter rather than test input. Reverted to 10 before lock. Test input corrected to 11 values (`[0.4, 0.04, ...]`) with 10 strictly-below-threshold values. | None — corrected before lock |
| March 2026 | Zenodo DOI (10.5281/zenodo.18929040) inserted into preregistration.md after registration | DOI cannot exist before Zenodo record is created — expected post-registration update. Scientific content unchanged. | Hash updated to reflect DOI insertion. No scientific content modified. |
