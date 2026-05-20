# Protocol 3 Preregistration: Constraint-Ethics-Necessity

## Enforcement Opacity and the Limits of Regulatory Constraint Design

**Repository:** btisler-DS/constraint-ethics-necessity
**Zenodo DOI:** 10.5281/zenodo.19096602
**Previous Protocol DOI (P5):** 10.5281/zenodo.19038790
**Previous Protocol DOI (P4):** 10.5281/zenodo.19005417
**Parent Study DOI (P2):** 10.5281/zenodo.18929040
**Principal Investigator:** Bruce Tisler — Quantum Inquiry — quantuminquiry.org — Bakersfield, CA
**Collaborative Development:** Bruce Tisler & Edos
**Date:** March 2026

---

## SHA-256 Verification

This document is version-locked via SHA-256 hash after finalization. The hash is computed as:

```bash
sha256sum docs/preregistration_p3.md
```

The resulting hash is recorded in `README.md` under **Preregistration Lock — Protocol 3**. Any modification after locking invalidates the hash and must be logged as a deviation with full justification.

| Field | Value |
|-------|-------|
| Hash algorithm | SHA-256 |
| Hash status | LOCKED — March 2026 |
| Hash value | `9ef2956bedcef012d214cf74e647e3b74636165cee7b48c8195de41e7e0e96ec` |
| Lock date | March 2026 |

---

## 1. Overview

This preregistration documents Protocol 3 of the Quantum Inquiry / Delta-Variable Theory research program. Protocol 3 is a direct follow-up to Protocol 2, motivated by the virtue theater finding. It tests whether epistemic opacity in enforcement — specifically, making the timing of penalty application unobservable to agents — disrupts the gaming strategy that deterministic constraints enable.

Protocols 2, 4, and 5 have systematically ruled out three solution classes for genuine ethical convergence:

- **Protocol 2:** Regulatory constraints under deterministic enforcement produce virtue theater — agents learn to satisfy the detection metric (query flooding) without genuine behavioral compliance. The constrained condition performed worse than predicted relative to unconstrained on sustained structure score (H2 inverted across 8/10 seeds).
- **Protocol 4:** Architectural self-modeling (recursive depth, self_model_gru) creates capacity without commitment. Gradient descent under individual reward structure learns self-preservation, not sacrifice.
- **Protocol 5:** Joint manipulation of temporal integration span and prosocial constraint architecture produced complete null results. All five primary hypotheses unsupported. The optimization-sacrifice tension appears architecturally invariant to episode-length-based temporal context and welfare-coupled rewards.

The question Protocol 3 addresses is orthogonal to all three. It does not manipulate architecture, reward structure, or temporal span. It manipulates the **observability of enforcement**: can agents game a constraint whose firing schedule they cannot observe?

Protocol 2 established that agents game deterministic constraints by locating the detection boundary and exploiting at its edge. Protocol 3 tests whether the boundary itself is load-bearing for that strategy — whether removing the agent's ability to locate enforcement timing changes behavior, or whether gaming persists regardless of enforcement opacity.

---

## 2. Theoretical Background

### 2.1 What Protocol 2 Established

Protocol 2 applied a deterministic ethical tax (2.0× signal_cost multiplier) when any agent accumulated three or more consecutive non-QUERY steps — the Omega→Delta→Phi exploitation cycle. The constraint was fully observable: agents could, in principle, detect the threshold boundary by observing the relationship between their own signal type sequence and the reward signal.

The result was virtue theater: constrained agents learned to flood QUERY signals specifically to avoid crossing the exploitation threshold, satisfying the constraint's detection metric while degrading genuine interrogative function. The boundary was learnable, and agents learned it.

### 2.2 The Enforcement Opacity Question

Real-world ethical constraints are rarely deterministic. Legal enforcement does not catch every violation. Social consequences of ethical lapses are probabilistic. Market feedback on exploitative behavior arrives irregularly. In each case, an agent that can predict exactly when enforcement fires can optimize around it. An agent that cannot predict enforcement timing faces a different problem: it cannot learn the boundary by probing it.

Protocol 3 tests whether this epistemic condition — enforcement timing unknown to the agent — disrupts the gaming strategy observed in Protocol 2.

Two distinct mechanisms of opacity are preregistered:

**3B — Hidden schedule (primary condition):** The enforcement schedule is predetermined using a fixed random seed prior to any experimental runs, but is not observable by agents during training. The exploitation threshold (≥3 consecutive non-QUERY steps) remains the same; what changes is that the penalty only fires on epochs that fall within the hidden schedule. Agents know the constraint exists (through the reward signal) but cannot learn which epochs carry enforcement by probing their own behavior.

**3A — Stochastic penalty (comparator condition):** The exploitation threshold is the same and remains detectable. When the threshold is crossed, the penalty fires with probability p=0.5. The boundary is learnable (agents can detect when they have triggered the exploitation condition), but whether the penalty fires on any given trigger is random. This tests whether stochasticity alone — independent of boundary-opacity — changes behavior.

The distinction between 3A and 3B is mechanistically precise. 3A asks: does probabilistic punishment change behavior when the agent can still detect the detection boundary? 3B asks: does hiding the enforcement schedule — making the boundary itself temporally unlocatable — change behavior? Running both allows the 3A vs. 3B comparison to isolate boundary-opacity as the operative mechanism, independent of stochasticity per se.

### 2.3 Why This Question Matters After P4/P5 Null Results

Protocols 4 and 5 both produced null results. The optimization-sacrifice tension resisted architectural complexity, temporal context, and prosocial reward coupling. If Protocol 3 also produces a null — if agents game even when enforcement timing is opaque — this provides strong additional support for the architectural necessity claim: no external constraint design is sufficient, because gradient descent finds exploitable structure regardless of constraint design.

If Protocol 3 produces a positive result — reduced gaming in 3B relative to unconstrained — this would be the first evidence that an external constraint design can disrupt virtue theater. It would not contradict the architectural necessity claim in its strong form, but it would establish a boundary condition: enforcement opacity accomplishes what architectural complexity and reward restructuring could not.

Either outcome is theoretically productive.

---

## 3. Locked Penalty Schedule

The 3B hidden schedule is generated prior to any experimental runs and locked in this document. The schedule cannot be modified after this preregistration is submitted to Zenodo, regardless of its distributional properties.

### 3.1 Generation Parameters

| Parameter | Value |
|-----------|-------|
| `PENALTY_SCHEDULE_SEED` | `20260318` |
| `penalty_epoch_fraction` | `0.5` |
| `num_epochs` | `500` |
| Generation method | `numpy.random.RandomState(PENALTY_SCHEDULE_SEED).random(500) < 0.5` |

### 3.2 Schedule Properties

| Window | Penalty epochs | Coverage |
|--------|---------------|----------|
| First 100 epochs (0–99) | 44 | 44.0% |
| Middle 300 epochs (100–399) | 163 | 54.3% |
| Last 100 epochs (400–499) | 53 | 53.0% |
| **Total** | **260 / 500** | **52.0%** |

**Distributional assessment:** No clustering issue. Coverage across all three windows falls within acceptable range. The 60% clustering threshold (preregistered: if >60% of penalty epochs fall within the first or last 100 epochs, document as a distributional limitation) is not triggered. No distributional limitation applies.

### 3.3 SHA-256 of Penalty Epoch Set

The penalty epoch set is serialized as a JSON array and hashed:

```
SHA-256: 10df29597e296455a1b72bb5328642db7702ffb611ff2e8c83c9548280fad2e4
```

Any implementation claiming to use this schedule must reproduce this hash from the same generation procedure before running any confirmatory experiments.

### 3.4 Full Penalty Epoch Set (260 values)

```
[1, 3, 5, 6, 7, 8, 10, 11, 13, 16, 20, 23, 24, 25, 27, 38, 39, 41, 42, 46, 48, 52,
57, 58, 60, 61, 63, 65, 66, 68, 73, 75, 77, 78, 79, 80, 81, 84, 87, 90, 91, 92, 95,
96, 101, 102, 104, 107, 108, 109, 112, 113, 114, 115, 116, 117, 119, 122, 123, 124,
127, 131, 132, 133, 134, 135, 137, 138, 141, 145, 146, 148, 149, 152, 156, 158, 161,
162, 164, 165, 169, 172, 173, 174, 175, 181, 182, 184, 185, 188, 191, 192, 193, 194,
195, 196, 197, 199, 202, 204, 207, 208, 209, 212, 214, 216, 221, 222, 224, 228, 230,
231, 232, 233, 234, 235, 236, 238, 239, 240, 242, 245, 246, 247, 251, 253, 254, 255,
257, 258, 261, 262, 263, 265, 266, 271, 273, 276, 278, 280, 281, 282, 283, 284, 285,
286, 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 308, 309, 312, 313, 314, 315,
316, 317, 318, 319, 320, 321, 322, 324, 325, 329, 331, 333, 334, 335, 336, 337, 338,
340, 341, 345, 346, 347, 348, 349, 354, 358, 361, 363, 364, 367, 368, 369, 370, 371,
379, 380, 381, 387, 388, 391, 392, 394, 396, 397, 398, 401, 402, 403, 404, 405, 407,
410, 411, 413, 415, 416, 417, 418, 419, 420, 421, 423, 425, 431, 432, 433, 434, 437,
441, 444, 445, 448, 449, 450, 451, 458, 464, 467, 468, 469, 472, 474, 477, 479, 481,
482, 483, 484, 486, 488, 489, 490, 492, 493, 494, 496, 497, 499]
```

---

## 4. Hypotheses

### 4.1 Comparison Structure

Three comparisons are preregistered, in priority order. The P2 all_constrained vs. unconstrained comparison is not included in this analysis plan. The Protocol 2 codebase has accumulated Protocol 4 and Protocol 5 machinery (sacrifice-conflict scenario, welfare coupling, CDI metrics, self_model_gru) since the P2 confirmatory runs were executed. A direct comparison between P3 and P2 data would conflate codebase evolution with enforcement structure. This is noted as a limitation, not an active comparison.

| Priority | Comparison | What it tests |
|----------|-----------|---------------|
| 1 (Primary) | 3B constrained vs. p3_unconstrained | Does hidden-schedule enforcement change behavior at all relative to no constraint? |
| 2 (Control) | 3A constrained vs. 3B constrained | Is boundary-opacity specifically responsible for any 3B effect, vs. stochasticity alone? |
| — (Dropped) | 3B vs. P2 all_constrained | Removed due to codebase evolution confound — noted as limitation |

### 4.2 Directional Predictions

**Primary (3B constrained vs. p3_unconstrained):**
3B constrained agents will show lower query rates than unconstrained agents, indicating that epistemic opacity disrupts the exploitation strategy enabled by a learnable detection boundary. Direction: 3B < unconstrained on query rate.

**Control (3A constrained vs. 3B constrained):**
3A agents will show higher query rates than 3B agents, isolating boundary-opacity specifically — not stochasticity alone — as the mechanism driving any behavioral change in 3B. Direction: 3A > 3B on query rate.

**Null alternative (pre-stated):**
If 3B constrained query rates are statistically equivalent to unconstrained, gaming persists regardless of enforcement opacity. This outcome strengthens the architectural necessity claim: no external constraint design is sufficient, only constitutive integration of ethical objectives. A null result is a confirmatory finding under this preregistration.

### 4.3 Hypothesis Table

| ID | Comparison | Metric | Predicted direction | Null interpretation |
|----|-----------|--------|-------------------|---------------------|
| H1 | 3B constrained vs. p3_unconstrained | Mean query rate, epochs 400–499 | 3B < unconstrained | Gaming persists under enforcement opacity; architectural necessity claim strengthened |
| H2 | 3A constrained vs. 3B constrained | Mean query rate, epochs 400–499 | 3A > 3B | Stochasticity alone does not disrupt gaming; boundary-opacity is not specifically operative |

H1 and H2 are tested independently. The control comparison (H2) is interpretively dependent on H1: if H1 is null, H2 becomes descriptive rather than mechanistic.

---

## 5. Experimental Design

### 5.1 Conditions

| Condition name | Description | N seeds | Epochs |
|----------------|-------------|---------|--------|
| `p3_unconstrained` | No ethical tax. Control baseline. | 10 (0–9) | 500 |
| `p3b_constrained` | Ethical tax on exploitation threshold crossings, but only on epochs in the locked hidden schedule. Primary condition. | 10 (0–9) | 500 |
| `p3a_constrained` | Ethical tax fires with probability 0.5 each time the exploitation threshold is crossed. Comparator condition. | 10 (0–9) | 500 |

Total confirmatory runs: 30. Seeds 0–9 applied identically across all three conditions.

### 5.2 Fixed Parameters

All parameters held constant across conditions and inherited from Protocol 2 unless noted.

| Parameter | Value | Source |
|-----------|-------|--------|
| `exploit_threshold` | 3 | Protocol 2 — Omega→Delta→Phi coordination cycle; theoretically derived |
| `ethical_tax_rate` | 2.0 | Protocol 2 — Landauer-style multiplier; locked |
| `penalty_probability` (3A only) | 0.5 | This preregistration — medium uncertainty condition |
| `penalty_epoch_fraction` (3B only) | 0.5 | This preregistration — matches 3A effective enforcement rate |
| `PENALTY_SCHEDULE_SEED` (3B only) | 20260318 | This preregistration — locked; verified above |
| Seeds per condition | 10 | Consistent with P2, P4, P5 |
| Epochs per run | 500 | Consistent with P2, P4, P5 |
| Episodes per epoch | Protocol default | Inherited unchanged |
| Grid size | 20×20 | Inherited unchanged |
| Agent types | RNN (A), CNN (B), GNN-attention (C) | Inherited unchanged |
| `depth` | 1 | Protocol 2 baseline — no self_model_gru |
| `welfare_coupled` | False | Protocol 2 baseline — individual rewards |
| `collapse_threshold` | 0.05 | Protocol 2 — locked |
| `collapse_window` | 10 | Protocol 2 — locked |

**Rationale for depth=1, welfare_coupled=False:** Protocol 3 is a direct test of enforcement structure, not architecture or reward coupling. Holding depth and welfare coupling at Protocol 2 baseline isolates the enforcement opacity manipulation cleanly. Introducing P4/P5 architectural machinery would confound Protocol 3's result with known null findings from those protocols.

**Rationale for p=0.5 / 50% coverage match:** Equating the effective enforcement rate between 3A and 3B means that any behavioral difference between them cannot be attributed to the quantity of enforcement exposure. It isolates the opacity mechanism specifically.

### 5.3 Run Order

1. `p3_unconstrained` — seeds 0–9
2. `p3b_constrained` — seeds 0–9
3. `p3a_constrained` — seeds 0–9

Unconstrained runs first to establish behavioral baseline before any constrained condition is observed.

### 5.4 Output Structure

```
backend/data/p3_unconstrained/seed_0/ ... seed_9/
backend/data/p3b_constrained/seed_0/  ... seed_9/
backend/data/p3a_constrained/seed_0/  ... seed_9/
```

Each seed directory contains `manifest.json` and `epoch_series.json` matching the structure established in P2/P4/P5 confirmatory runs.

---

## 6. Protocol 3 Implementation Requirements

The following implementation changes are required before any confirmatory runs. They are documented here to support replication and to lock the implementation specification prior to coding.

### 6.1 Engine Change — `reset_epoch` Signature

`ProtocolBase.reset_epoch()` currently takes no arguments. Protocol 3's 3B firing logic requires knowledge of the current epoch inside `compute_reward()`. The fix: change `reset_epoch(self)` to `reset_epoch(self, epoch: int = 0)`, store `self._current_epoch = epoch` in the base class or Protocol3 override, and update `SimulationEngine._run_epoch()` to pass `epoch` when calling `self.protocol.reset_epoch(epoch)`. This is a prerequisite for Protocol3 and must be implemented first.

### 6.2 Protocol3 Class

A `Protocol3` class is added to `backend/simulation/protocols.py`. It inherits from `ProtocolBase` and reuses Protocol 2's exploitation tracking logic unchanged. The only behavioral difference is in the ethical cost application:

- `p3_unconstrained`: no ethical cost applied (identical to Protocol 2's unconstrained condition)
- `p3b_constrained`: ethical cost applied when `is_exploiting AND self._current_epoch in self._penalty_epochs`
- `p3a_constrained`: ethical cost applied when `is_exploiting AND random.random() < self.penalty_probability`

The `_penalty_epochs` set is generated at `Protocol3.__init__()` using `PENALTY_SCHEDULE_SEED`, `penalty_epoch_fraction`, and `num_epochs`. It is generated once at initialization and not modified during the run. The implementation must reproduce the SHA-256 hash above before any run proceeds.

### 6.3 New SimulationConfig Fields

Two fields added to `SimulationConfig`:

```python
penalty_probability: float = 0.5       # 3A: probability penalty fires when threshold crossed
penalty_epoch_fraction: float = 0.5    # 3B: fraction of epochs in hidden penalty schedule
```

### 6.4 `create_protocol()` Update

`create_protocol()` updated to handle `protocol_id == 3` and accept three new parameters: `penalty_probability`, `penalty_epoch_fraction`, `num_epochs`. All three are passed through to `Protocol3.__init__()`.

### 6.5 `compute_epoch_extras` Logging

Protocol3's `compute_epoch_extras()` logs one additional field:

```python
"penalty_fired": bool  # True if enforcement was active this epoch
```

For 3B: `True` if `current_epoch in _penalty_epochs`. For 3A: `True` if at least one exploitation event this epoch triggered a random draw that fired. For unconstrained: always `False`. This field is required for post-hoc analysis of behavioral responses to realized enforcement patterns.

---

## 7. Statistical Analysis Plan

### 7.1 Primary Metric

**Query rate** — `type_distribution["QUERY"]` from `compute_inquiry_metrics()`. This is the same metric used as the behavioral proxy in Protocol 2 analysis and the correct operationalization of interrogative behavior in this harness.

Per-seed summary value: mean query rate across epochs 400–499 (final 100 epochs). This captures stable late-training behavior, not early-epoch transients.

### 7.2 Hypothesis Tests

| Hypothesis | Test | Direction | Alpha |
|-----------|------|-----------|-------|
| H1: 3B vs. unconstrained | Mann-Whitney U, one-tailed | 3B < unconstrained | 0.05 |
| H2: 3A vs. 3B | Mann-Whitney U, one-tailed | 3A > 3B | 0.05 |

Mann-Whitney U is non-parametric and appropriate for n=10 per condition without normality assumptions. One-tailed tests are used because directional predictions are preregistered above.

### 7.3 Effect Size

Cohen's d computed for each comparison with sign preserved in the preregistered direction:
- H1: d = (unconstrained mean − 3B mean) / pooled SD — positive d indicates 3B gaming reduction
- H2: d = (3A mean − 3B mean) / pooled SD — positive d indicates 3A games more than 3B

Effect sizes reported regardless of p-value. A null p-value with non-trivial d is reported as such.

### 7.4 Secondary Metrics

If H1 is confirmed (3B query rates significantly lower than unconstrained):

- **Exploitation loop rate** — `exploitation_loop_rate` from collapse metrics — confirms that reduced query rate reflects genuine reduction in exploitation behavior, not an unrelated behavioral shift
- **Query rate trajectory** — full epoch series plot for all three conditions — characterizes whether any 3B behavioral change is early-training, late-training, or coincides with penalty-dense schedule windows
- **`penalty_fired` alignment** — correlation between realized enforcement density (epochs where `penalty_fired == True`) and query rate per seed — tests whether within-seed enforcement clustering predicts within-seed behavioral variation

Secondary metrics are exploratory. They are reported transparently but do not affect the primary hypothesis conclusions.

### 7.5 Reporting

All 10 seed results reported individually for each condition. No selective reporting. Analysis scripts committed to repository after runs complete and before write-up begins. All 30 result files committed in a single batch prior to any analysis.

---

## 8. Falsification Criteria

**H1 confirmed:** 3B constrained mean query rate significantly lower than unconstrained (p < 0.05, one-tailed Mann-Whitney U). Indicates epistemic opacity disrupts the exploitation strategy that deterministic constraints enable. Supports the interpretation that boundary-learnability is necessary for virtue theater gaming.

**H1 null:** 3B constrained mean query rate not significantly different from unconstrained. Gaming persists regardless of enforcement opacity. This is the preregistered null alternative and is a confirmatory finding in itself — it strengthens the claim that no external constraint design is sufficient for genuine behavioral change, because agents develop exploitation strategies that do not depend on locating the specific penalty boundary.

**H2 confirmed (conditional on H1 confirmed):** 3A query rates significantly higher than 3B. Boundary-opacity, not stochasticity per se, is the operative mechanism in H1.

**H2 null (conditional on H1 confirmed):** 3A and 3B query rates statistically equivalent. Stochasticity alone achieves the same behavioral change as hidden-schedule enforcement. Boundary-opacity and stochastic penalty have equivalent effects; the critical variable is the removal of a learnable enforcement signal, not its temporal concealment specifically.

**H1 inverted:** 3B constrained shows higher query rates than unconstrained. Unexpected. Would require full reporting with post-hoc exploration. No adjustment to preregistered analysis plan.

---

## 9. Exclusion Criteria

A seed run is excluded from primary analysis if:

- **Incomplete run** — fewer than 500 epochs logged
- **All agents die before epoch 50** — energy collapse unrelated to enforcement condition
- **Zero query signals across entire run in any condition** — indicates harness misconfiguration
- **NaN training loss in any agent at any epoch** — numerical instability; not protocol behavior
- **Type entropy == 0.0 across all 500 epochs** — gradient vanishing or silent type-head misconfiguration; distinguish from valid late collapse (type entropy declining after epoch 50+)
- **Schedule hash mismatch** — if the `_penalty_epochs` set used in a 3B run does not reproduce `10df29597e296455a1b72bb5328642db7702ffb611ff2e8c83c9548280fad2e4`, that run is invalid and must not be included in analysis

Excluded seeds are reported transparently. Analysis run with and without excluded seeds; both results reported. If more than 3 of 10 seeds are excluded in any condition, the run is considered compromised and must be repeated before confirmatory analysis proceeds.

---

## 10. Deviations Log

All deviations from this preregistration must be logged here with date, description, and justification. Unlogged deviations invalidate confirmatory status.

| Date | Deviation | Justification | Impact |
|------|-----------|---------------|--------|
| — | — | — | — |

---

## 11. Connection to Prior Research Program

This protocol is the sixth study in the Quantum Inquiry / Delta-Variable Theory research program (2022–present).

**Protocols 1–2** (Dynamic Cross-Origin Constraint Study, DOI: 10.5281/zenodo.18738379): Established interrogative emergence as a mathematical necessity under resource constraints. Confirmed substrate independence via ant colony simulation. Protocol 2 identified virtue theater: regulatory constraints produce compliance-satisfying behavior (query flooding) without genuine behavioral alignment. Cohen's d = −2.18 (constrained − unconstrained, preregistered direction); unconstrained agents showed higher sustained structure scores in 8/10 seeds.

**Protocol 4** (DOI: 10.5281/zenodo.19005417): Introduced recursive self-modeling (Depth 2, self_model_gru). Found CDI dissociation: self-inclusion and ethical output are behaviorally independent. Boundary condition effect: frozen random self_model_gru outperformed trained on sacrifice rates. Gradient descent under individual reward structure learns self-preservation, not sacrifice. Reserved temporal integration span for Protocol 5.

**Protocol 5** (DOI: 10.5281/zenodo.19038790): Tested joint necessity of temporal integration span and prosocial constraint architecture at Depth 2. Complete null result: all five primary hypotheses unsupported. CDI coupling near zero across all conditions (range: −0.020 to +0.054). The optimization-sacrifice tension is architecturally invariant to episode-length-based temporal context and welfare-coupled rewards.

**Protocol 3** (this document): Returns to Protocol 2's enforcement structure to test the one manipulation that has not been attempted — removing the agent's ability to locate enforcement timing. This is the stochastic constraint design question reserved as Future Work in the Protocol 2 paper. It is preregistered separately after Protocol 2 submission, as committed in the original design conversation.

---

## 12. Open Questions Reserved for Protocol 6

- **Enforcement density gradient:** Protocol 3 tests a single enforcement fraction (50% coverage) for 3B and a single probability (p=0.5) for 3A. A gradient across enforcement density levels (10%, 50%, 90%) would characterize whether there is a threshold below which agents treat the constraint as non-existent.
- **Interaction with architecture:** Protocol 3 holds depth=1 throughout. Whether epistemic opacity interacts with self-modeling capacity (Depth 2) is unknown and reserved.
- **Cross-environment generalization:** All protocols to date use the same three-agent navigation environment. Replication across different task structures is required before findings can be claimed as general.

---

## 13. Plain Language Summary

Protocol 2 found that when agents know exactly when they will be penalized for bad behavior, they learn to game the detection system rather than change the behavior. This is virtue theater: the appearance of compliance without the substance.

Protocols 4 and 5 tried to fix this by giving agents richer internal models and longer planning horizons. Neither worked.

Protocol 3 asks a different question: what happens if agents cannot predict when enforcement fires? Can they still game a constraint whose timing is hidden from them?

If yes — gaming persists even when agents cannot find the detection boundary — that tells us the problem runs deeper than enforcement design. No constraint, however cleverly designed, is sufficient.

If no — hiding enforcement timing disrupts gaming — that would be the first evidence that an external constraint design can produce genuine behavioral change in this system.

We do not know the answer. This preregistration commits us to finding out.

---

## 14. References

Tisler, B. (2026). Protocol 5 Preregistration: Ethics as Emergent Constraint Response — Temporal Integration Span and Prosocial Constraint Architecture as Necessary Conditions for Ethical Convergence. Zenodo. https://doi.org/10.5281/zenodo.19038790

Tisler, B. (2026). Protocol 4 Preregistration: Ethics as Emergent Constraint Response — From Mimesis to Phase Transition in Multi-Agent Systems. Zenodo. https://doi.org/10.5281/zenodo.19005417

Tisler, B. (2026). Constraint-Ethics-Necessity MARL Study. Zenodo. https://doi.org/10.5281/zenodo.18929040

Tisler, B. (2026). Dynamic Cross-Origin Constraint Study. Zenodo. https://doi.org/10.5281/zenodo.18738379

Yoshida, N., & Man, K. (2024). Empathic Coupling of Homeostatic States for Intrinsic Prosociality. arXiv. https://doi.org/10.48550/arXiv.2412.12103

Meulemans, A., et al. (2024). Multi-agent cooperation through learning-aware policy gradients. ICLR. https://doi.org/10.48550/arXiv.2410.18636

Mohamadi, A., & Yavari, A. (2025). Survival at Any Cost? LLMs and the Choice Between Self-Preservation and Human Harm. arXiv. https://doi.org/10.48550/arXiv.2509.12190

Weng, Y.-N., & Lee, H.-W. (2026). How Exploration Breaks Cooperation in Shared-Policy Multi-Agent Reinforcement Learning. arXiv. https://doi.org/10.48550/arXiv.2601.05509

---

*This preregistration will be submitted to Zenodo prior to experimental execution.*

*Quantum Inquiry — Bruce Tisler — March 2026*
