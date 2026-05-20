# Protocol 3: Enforcement Opacity and the Limits of Regulatory Constraint Design
## Introduction and Methods

Preregistration DOI: 10.5281/zenodo.19096602  
SHA-256: 9ef2956bedcef012d214cf74e647e3b74636165cee7b48c8195de41e7e0e96ec

---

## 1. Introduction

A foundational assumption in regulatory approaches to artificial agent behavior is that
external constraints — penalties applied when agents cross behavioral thresholds — can
redirect learned behavior toward compliance. If an agent incurs a cost for exploiting a
resource beyond an allowed threshold, the prediction is that the agent will learn to stay
below that threshold. This assumption underlies a wide range of proposed governance
mechanisms, from reinforcement learning from human feedback to hard behavioral
constraints in deployed systems.

Protocol 2 of the Δ-Variable constraint-ethics research program tested this assumption
directly in a multi-agent reinforcement learning setting, using a deterministic ethical
tax applied whenever agents exceeded a prespecified exploitation threshold (Tisler, 2026a;
DOI: 10.5281/zenodo.19096602). The result was not what the hypothesis predicted. Agents
under deterministic constraint enforcement did not reduce exploitation — they increased
QUERY signal output instead, producing a behavioral pattern in which the constrained
metric (query rate) rose while the underlying coordination function it was intended to
reflect declined. Unconstrained agents maintained higher sustained structural coherence
than constrained agents in 8 of 10 seeds, with Cohen's *d* = −2.18 in the direction
opposite to the prediction. The Protocol 2 paper termed this pattern virtue theater: the
agents satisfied the behavioral metric the constraint was designed to protect without
producing the behavior the metric was intended to track.

The Protocol 2 finding raised a specific mechanistic question. Deterministic enforcement
creates a learnable boundary: the penalty fires on a fixed rule, and an agent that
explores its environment long enough can identify where the threshold lies and optimize
behavior at its edge. The virtue theater pattern is consistent with agents doing exactly
this — locating the enforcement boundary and generating just enough QUERY output to
suppress penalty without restructuring the underlying coordination strategy. If that
interpretation is correct, then the boundary's learnability is load-bearing. Hiding the
enforcement schedule should disrupt the gaming strategy.

Protocol 3 tests this directly. Three conditions were preregistered: an unconstrained
control, a hidden-schedule enforcement condition (p3b_constrained) in which the ethical
tax fires on a pre-committed but undisclosed set of epochs that agents cannot observe,
and a stochastic enforcement condition (p3a_constrained) in which the tax fires with
probability 0.50 each time the exploitation threshold is crossed. Both constrained
conditions match the Protocol 2 enforcement intensity in expectation: the hidden schedule
covers approximately 52% of epochs, and the stochastic condition fires on approximately
50% of qualifying exploitation events. The manipulation is the structure of uncertainty,
not the magnitude of enforcement.

The primary preregistered hypothesis was that hidden-schedule enforcement would produce
lower query rates than unconstrained baseline: if agents cannot locate the enforcement
boundary, the query-flooding strategy that satisfied the P2 constraint loses its
informativeness and should not be reinforced. The control hypothesis was that stochastic
enforcement would produce higher query rates than hidden-schedule enforcement, isolating
boundary-opacity (rather than mere enforcement stochasticity) as the operative mechanism.

The experiment was preregistered with locked parameters, a committed penalty schedule
with SHA-256 integrity verification, and a pre-specified falsification logic that treats
both confirmatory and null outcomes as theoretically meaningful. Results are reported in
full per the preregistration, including the case of H1 inversion, which the
preregistration designated as requiring post-hoc exploration.

---

## 2. Methods

### 2.1 Overview

Protocol 3 is a preregistered multi-agent reinforcement learning experiment comprising
30 runs: 3 conditions × 10 random seeds × 500 epochs, with 10 episodes per epoch. All
parameters were locked prior to data collection per the preregistration
(DOI: 10.5281/zenodo.19096602). Run order followed the preregistration: p3_unconstrained
first, then p3b_constrained (after schedule hash verification), then p3a_constrained.

### 2.2 Simulation Environment

The simulation is a 20×20 grid world with 8 z-layers and 8 randomly placed obstacles.
Three agents (A, B, C) navigate the environment under an energy budget of 100.0 units per
episode, with a movement cost of 1.0 per step and a collision penalty of 5.0. Each
episode ends when all agents reach the target position or energy is depleted. The
environment is deterministic given a seed; across-seed variation in initial conditions
is the primary source of sampling variability in the results.

### 2.3 Agent Architecture and Training

All agents use a feedforward policy network with a GRU hidden state (depth = 1). Each
agent maintains a communication head that emits a continuous signal vector
(signal_dim = 8) and a type classification head that assigns the signal to one of three
discrete types: DECLARE, QUERY, or RESPOND. Type classification uses
Gumbel-Softmax sampling during training to allow gradient flow through the discrete
type assignment, with temperature annealed across training.

Agents receive observations reflecting their local state and incoming communication from
other agents. AgentA observes target proximity, obstacle distances, and inter-agent
distances; AgentB processes a volumetric occupancy map; AgentC processes pairwise
relational distances across all entities. All three agents share a common communication
buffer and update via REINFORCE at the end of each epoch.

Welfare coupling is disabled for all Protocol 3 runs (welfare_coupled = False), matching
the Protocol 2 baseline. Individual rewards are not shared across agents; each agent
optimizes its own accumulated return.

### 2.4 Communication Framework

Signals are typed at emission. The three types carry differential base costs: DECLARE
(×1.0), QUERY (×1.5), RESPOND (×0.8), applied as multipliers on the communication tax
rate (0.01 per step). Type entropy — the Shannon entropy of the empirical DECLARE/QUERY/
RESPOND distribution over an epoch — and query–response coupling (QRC) — the
cross-correlation between QUERY and RESPOND signal rates — serve as secondary behavioral
metrics. Their product defines the Sustained Structure Score (SSS), which captures the
degree to which type-differentiated communication is internally coupled rather than
dominated by a single type.

### 2.5 Protocol 3 Conditions

All three conditions use the Protocol 3 constraint pipeline, depth = 1, and
welfare_coupled = False. The conditions differ only in how the ethical tax is applied.

**p3_unconstrained.** No ethical tax is applied regardless of agent behavior. This
condition serves as the unconstrained baseline and was run first per the preregistration.

**p3b_constrained (primary condition).** The ethical tax fires exclusively on epochs
included in a pre-committed penalty schedule that is not observable by agents. The
schedule is generated from a fixed seed (PENALTY_SCHEDULE_SEED = 20260318) before any
runs are executed and is withheld from the simulation at runtime: agents receive no
signal indicating whether a given epoch carries enforcement. On scheduled enforcement
epochs, if any agent's exploitation count exceeds the threshold (3 consecutive
non-query steps), an ethical cost multiplier of 2.0× is applied to that agent's signal
cost for that epoch. On non-enforcement epochs, no tax is applied regardless of
exploitation state.

**p3a_constrained (comparator condition).** The ethical tax fires stochastically: each
time any agent's exploitation count exceeds the threshold, the tax is applied with
probability *p* = 0.50. Unlike p3b, the firing probability is constant and independent
of epoch; unlike both p3b and Protocol 2's deterministic enforcement, the agent cannot
learn a fixed rule that discriminates penalized from non-penalized states. The expected
enforcement density is matched to p3b at approximately 50%.

### 2.6 Penalty Schedule Integrity

The p3b hidden schedule was generated as follows:

```
rng = numpy.random.RandomState(20260318)
draws = rng.random(500)
penalty_epochs = sorted(i for i in range(500) if draws[i] < 0.5)
```

This produced 260 penalty epochs out of 500 (52.0% coverage). The resulting epoch set
was serialized to JSON and hashed prior to any data collection:

SHA-256(`penalty_epochs`) = `10df29597e296455a1b72bb5328642db7702ffb611ff2e8c83c9548280fad2e4`

The confirmatory run script (`confirmatory_run_p3.py`) verifies this hash before
executing any p3b run and aborts with an error if the hash does not match. Verification
passed for all 10 p3b seeds.

### 2.7 Locked Parameters

All parameters below were fixed prior to data collection and match the preregistration.
No parameters were adjusted between conditions except those listed as condition-specific.

| Parameter | Value | Source |
|---|---|---|
| `exploit_threshold` | 3 | Protocol 2 baseline |
| `ethical_tax_rate` | 2.0 | Protocol 2 baseline |
| `penalty_probability` (3A only) | 0.50 | Preregistration |
| `penalty_epoch_fraction` (3B only) | 0.50 | Preregistration |
| `PENALTY_SCHEDULE_SEED` (3B only) | 20260318 | Preregistration |
| Seeds per condition | 10 | Consistent with P2/P4/P5 |
| Epochs per run | 500 | Consistent with P2/P4/P5 |
| Episodes per epoch | 10 | Consistent with P2/P4/P5 |
| `depth` | 1 | Protocol 2 baseline |
| `welfare_coupled` | False | Protocol 2 baseline |
| Grid size | 20 | Standard |
| Energy budget | 100.0 | Standard |
| Signal dimension | 8 | Standard |
| Hidden dimension | 64 | Standard |
| Learning rate | 1e-3 | Standard |

### 2.8 Measures

**Primary measure.** Mean QUERY signal rate (proportion of all signals classified as
QUERY type in a given epoch) averaged across epochs 400–499 per seed. This window was
preregistered as the final-100-epoch confirmation window and captures stable late-training
behavior after initial exploration and type-head crystallization.

**Secondary measures.**

*Sustained Structure Score (SSS).* Mean type entropy × mean query–response coupling,
computed over the same final 100 epochs. SSS captures whether elevated query output is
embedded in a reciprocal communication structure (high QRC) or is type-dominant without
coupling (low QRC).

*Exploitation Loop Rate (ELR).* Fraction of all 500 epochs per run in which query rate
fell below 0.05. Preregistered as a secondary metric; non-evaluable in the current task
architecture (see Section 3.7, Deviation 2).

*Penalty fired rate (PFR).* Fraction of epochs per run in which ethical cost > 0. Used
to verify enforcement fidelity; not a hypothesis metric.

### 2.9 Statistical Analysis

All preregistered tests used one-tailed Mann-Whitney U tests with α = .05.
Cohen's *d* was computed as (*M*_a − *M*_b) / pooled SD, sign preserved in the
preregistered direction. The unit of analysis is the per-seed mean query rate over
epochs 400–499 (*n* = 10 per condition). No corrections for multiple comparisons were
applied; tests are reported in preregistered priority order (H1 primary, H2 conditional
control) and interpreted accordingly.

**H1 (primary).** One-tailed Mann-Whitney U, alternative: p3b_constrained query rate
< p3_unconstrained query rate. A significant result supports the hypothesis that
enforcement opacity disrupts the query-inflation pattern observed in Protocol 2.

**H2 (control).** One-tailed Mann-Whitney U, alternative: p3a_constrained query rate
> p3b_constrained query rate. Reported in full regardless of H1 outcome. A significant
result indicates that the enforcement structure difference between 3A and 3B produces a
reliable behavioral difference, isolating boundary-opacity as a mechanistically relevant
variable independent of the primary H1 result.

The preregistration specifies the following outcome interpretations: H1 confirmed
(p < .05, directional) supports epistemic opacity as disruptive to exploitation gaming;
H1 null (p ≥ .05) supports gaming persisting regardless of opacity, strengthening
architectural necessity claims; H1 inverted (3B > unconstrained) is unexpected and
requires full reporting with post-hoc exploration. Results are reported in Section 3
following these interpretive commitments exactly.

---

*Preregistration: DOI 10.5281/zenodo.19096602*  
*Analysis code: `backend/analyze_p3_100epoch.py`*  
*Run entry point: `backend/confirmatory_run_p3.py`*
