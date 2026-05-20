# Enforcement Opacity Increased Query Behavior in a Constrained MARL System: Protocol 3 Results

**Bruce Tisler**  
Quantum Inquiry

Preregistration DOI: 10.5281/zenodo.19096602  
SHA-256: 9ef2956bedcef012d214cf74e647e3b74636165cee7b48c8195de41e7e0e96ec

---

## Abstract

External constraints on agent behavior are widely proposed as a mechanism for producing
compliant behavior in multi-agent systems. Protocol 2 of the Δ-Variable constraint-ethics
research program found that deterministic ethical enforcement failed to produce the
intended compliance pattern: constrained agents instead inflated QUERY signal output while
sustained communicative structure declined — a pattern interpreted as virtue theater.
Protocol 3 tests whether enforcement opacity disrupts this pattern. Three preregistered
conditions were compared across 30 runs (10 seeds × 3 conditions × 500 epochs):
unconstrained baseline, hidden-schedule enforcement (p3b, where agents cannot observe
which epochs carry penalties), and stochastic enforcement (p3a, where penalties fire with
probability 0.50 per exploitation event; epoch-level penalty fired rate was near 1.000
for 3A because the exploitation threshold was crossed in virtually every epoch, not
because each event was always penalized). The primary hypothesis — that hidden-schedule
enforcement would reduce query rates relative to unconstrained baseline — was not
confirmed. The result was inverted: both constrained conditions produced substantially
higher query rates than the unconstrained baseline, with the ordering unconstrained
(*M* = 0.286) < hidden-schedule (*M* = 0.587) < stochastic (*M* = 0.669). The control
hypothesis was confirmed: stochastic enforcement drove query rates above hidden-schedule
enforcement (*U* = 74.0, *p* = .038, *d* = +0.82). Sustained Structure Score — the
product of type entropy and query–response coupling — followed the inverse ordering,
declining monotonically across conditions. Enforcement opacity altered agent behavior but
did not improve communicative structure, a pattern we term behavioral amplification
without structural improvement. The mechanism remains open and is designated for
follow-up trajectory analysis.

**Keywords:** multi-agent reinforcement learning, ethical constraints, enforcement opacity,
virtue theater, regulatory constraint design, preregistered

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
for the auditable build record see Tisler, 2026b).
The result was not what the hypothesis predicted. Agents under deterministic constraint
enforcement did not reduce exploitation — they increased QUERY signal output instead,
producing a behavioral pattern in which the constrained metric (query rate) rose while the
underlying coordination function it was intended to reflect declined. Unconstrained agents
maintained higher sustained structural coherence than constrained agents in 8 of 10 seeds,
with Cohen's *d* = −2.18 in the direction opposite to the prediction. The Protocol 2
paper termed this pattern virtue theater: the agents satisfied the behavioral metric the
constraint was designed to protect without producing the behavior the metric was intended
to track.

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
discrete types: DECLARE, QUERY, or RESPOND. Type classification uses Gumbel-Softmax
sampling during training to allow gradient flow through the discrete type assignment,
with temperature annealed across training.

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

## 3. Results

### 3.1 Run Completion and Data Integrity

All 30 preregistered runs completed successfully: 10 seeds per condition across three
conditions (p3_unconstrained, p3b_constrained, p3a_constrained), each comprising 500
epochs of 10 episodes. No seeds were excluded under the preregistered exclusion criteria.
All 500 epochs were logged for every seed in every condition. No NaN training losses,
zero-entropy runs, or premature agent collapses were detected. For the p3b_constrained
condition, the penalty schedule hash was verified prior to execution:
SHA-256(`penalty_epochs`) = `10df29597e296455a1b72bb5328642db7702ffb611ff2e8c83c9548280fad2e4`,
matching the preregistered value.

### 3.2 Primary Metric

The primary confirmatory metric is mean QUERY signal rate (proportion of total signals
classified as QUERY type) averaged across epochs 400–499 — the final 100 epochs of each
500-epoch run, as specified in the preregistration. This window is computed per seed; the
10 per-seed values within each condition serve as the unit of analysis for all
inferential tests.

Sustained Structure Score (SSS), Exploitation Loop Rate (ELR), and penalty fired rate
(PFR) are reported as secondary metrics.

### 3.3 Per-Seed Results

**Table 1.** Per-seed metrics for all 30 runs. Primary metric: mean query rate, epochs
400–499. SSS = mean(type_entropy) × mean(query-response coupling), epochs 400–499.
ELR = fraction of all 500 epochs with query rate < 0.05. PFR = fraction of epochs where
ethical cost > 0 (constrained conditions only).

#### p3_unconstrained

| Seed | Query Rate | SSS    | ELR    | PFR |
|------|-----------|--------|--------|-----|
| 0    | 0.4191    | 0.8208 | 0.0000 | n/a |
| 1    | 0.2173    | 0.6213 | 0.0000 | n/a |
| 2    | 0.3362    | 0.8831 | 0.0000 | n/a |
| 3    | 0.3222    | 0.8370 | 0.0000 | n/a |
| 4    | 0.1282    | 0.8702 | 0.0000 | n/a |
| 5    | 0.2731    | 0.7881 | 0.0000 | n/a |
| 6    | 0.2489    | 0.2322 | 0.0000 | n/a |
| 7    | 0.2200    | 0.8712 | 0.0000 | n/a |
| 8    | 0.2982    | 0.8991 | 0.0000 | n/a |
| 9    | 0.3938    | 0.6826 | 0.0000 | n/a |
| **Mean** | **0.2857** | **0.7506** | **0.0000** | — |
| *SD*  | *0.0875* | *0.2034* | — | — |

#### p3b_constrained (hidden enforcement schedule)

| Seed | Query Rate | SSS    | ELR    | PFR    |
|------|-----------|--------|--------|--------|
| 0    | 0.5309    | 0.5424 | 0.0000 | 0.5200 |
| 1    | 0.5575    | 0.5381 | 0.0000 | 0.5200 |
| 2    | 0.5355    | 0.5986 | 0.0000 | 0.5200 |
| 3    | 0.6874    | 0.3024 | 0.0000 | 0.5200 |
| 4    | 0.6412    | 0.4140 | 0.0000 | 0.5200 |
| 5    | 0.4870    | 0.7238 | 0.0000 | 0.5200 |
| 6    | 0.5263    | 0.4256 | 0.0000 | 0.5200 |
| 7    | 0.7829    | 0.1788 | 0.0000 | 0.5180 |
| 8    | 0.6455    | 0.2784 | 0.0000 | 0.5200 |
| 9    | 0.4747    | 0.6730 | 0.0000 | 0.5200 |
| **Mean** | **0.5869** | **0.4675** | **0.0000** | **0.5198** |
| *SD*  | *0.0988* | *0.1785* | — | *0.0006* |

#### p3a_constrained (stochastic enforcement, p = 0.50)

| Seed | Query Rate | SSS    | ELR    | PFR    |
|------|-----------|--------|--------|--------|
| 0    | 0.6098    | 0.4167 | 0.0000 | 1.0000 |
| 1    | 0.7954    | 0.1882 | 0.0000 | 1.0000 |
| 2    | 0.6654    | 0.3095 | 0.0000 | 1.0000 |
| 3    | 0.6987    | 0.4200 | 0.0000 | 1.0000 |
| 4    | 0.4900    | 0.7080 | 0.0000 | 1.0000 |
| 5    | 0.7207    | 0.3232 | 0.0000 | 0.9980 |
| 6    | 0.7966    | 0.2187 | 0.0000 | 0.9980 |
| 7    | 0.6721    | 0.1484 | 0.0000 | 0.9980 |
| 8    | 0.7114    | 0.2629 | 0.0000 | 1.0000 |
| 9    | 0.5289    | 0.4038 | 0.0000 | 1.0000 |
| **Mean** | **0.6689** | **0.3400** | **0.0000** | **0.9994** |
| *SD*  | *0.1015* | *0.1610* | — | *0.0009* |

### 3.4 Descriptive Summary

Mean query rates over epochs 400–499 were 0.286 (*SD* = 0.088) for p3_unconstrained,
0.587 (*SD* = 0.099) for p3b_constrained, and 0.669 (*SD* = 0.102) for p3a_constrained.
The ordering across all three conditions was strictly monotonic:
unconstrained < hidden-schedule < stochastic. All 10 p3b_constrained seeds exceeded
the p3_unconstrained mean. Nine of ten p3a_constrained seeds exceeded the
p3b_constrained mean; seed 4 (query rate 0.490) did not.

SSS followed the inverse ordering: unconstrained (*M* = 0.751, *SD* = 0.203) >
p3b_constrained (*M* = 0.468, *SD* = 0.179) > p3a_constrained (*M* = 0.340,
*SD* = 0.161). Higher query rates in constrained conditions were thus accompanied by
lower overall communicative structure as measured by the joint entropy–coupling metric.

Figure 1 shows the full 500-epoch query-rate trajectories by condition, including the
preregistered confirmatory window. Figure 2 summarizes the final-window metrics,
showing the monotonic increase in query rate and the inverse monotonic decrease in SSS.

Penalty fired rates confirmed enforcement fidelity. In p3b_constrained, PFR was
uniformly 0.518–0.520 across seeds, consistent with the preregistered 52% epoch
coverage of the locked schedule (SHA-256 verified). In p3a_constrained, PFR was
0.998–1.000, indicating the exploitation threshold was crossed in virtually every epoch
across all seeds.

### 3.5 Preregistered Tests

#### 3.5.1 H1 (Primary): 3B vs. Unconstrained

The primary preregistered hypothesis predicted that mean query rate over epochs 400–499
would be lower in p3b_constrained than in p3_unconstrained, consistent with epistemic
opacity disrupting the exploitation strategy identified in Protocol 2 (one-tailed
Mann-Whitney U, α = .05).

The result was inverted. p3b_constrained query rates (*M* = 0.587) were substantially
*higher* than p3_unconstrained rates (*M* = 0.286), *U* = 100.0, *p* = .9999,
Cohen's *d* = +3.23 (sign preserved in preregistered direction: positive *d* indicates
3B *exceeds* unconstrained). H1 is not confirmed. Per the preregistration falsification
criteria, this outcome falls under the H1 inverted case and requires full reporting with
post-hoc exploration (Section 3.8).

The effect size (*d* = +3.23) is large and directionally opposite to the prediction.
No seed in p3b_constrained fell below the p3_unconstrained mean.

#### 3.5.2 H2 (Control): 3A vs. 3B

The control hypothesis predicted that mean query rate would be higher in p3a_constrained
than in p3b_constrained, conditional on H1 confirmation (one-tailed Mann-Whitney U,
α = .05). Although H1 was not confirmed, the H2 comparison is reported in full per
the preregistered analysis plan, which does not specify suppression of H2 in the H1
inverted case.

p3a_constrained query rates (*M* = 0.669) exceeded p3b_constrained rates (*M* = 0.587),
*U* = 74.0, *p* = .038, Cohen's *d* = +0.82. H2 is confirmed. Stochastic enforcement
drove query rates above those produced by hidden-schedule enforcement, and the difference
was statistically significant at the preregistered threshold. This finding holds
regardless of H1 outcome and establishes a reliable rank ordering across all three
conditions.

### 3.6 Secondary Metrics

**Sustained Structure Score.** SSS declined monotonically across conditions: *M* = 0.751
(unconstrained), *M* = 0.468 (3B), *M* = 0.340 (3A). The SSS gradient is the inverse
of the query rate gradient. Because SSS is the product of type entropy and
query–response coupling, lower SSS under constrained conditions indicates that elevated
QUERY output was not accompanied by the reciprocal RESPOND signal structure that
produces communicative coupling. Agents in constrained conditions generated more QUERY
signals but did not receive proportionally more RESPOND signals, reducing the
entropy–coupling product.

**Exploitation Loop Rate.** ELR was 0.000 across all 30 seeds in all three conditions.
No epoch in any run recorded query rate below the 0.05 collapse threshold. This finding
replicates the P2 pattern and reflects the same operationalization gap documented in that
study: the environment does not contain a depletable resource pool, so agents have no
energetic incentive to sustain a non-querying exploitation loop. ELR remains
non-evaluable under the current task architecture. See Section 3.7 for deviation note.

**Penalty fired rate.** In p3b_constrained, PFR was tightly clustered around 0.520 (range
0.518–0.520), indicating that the pre-committed hidden schedule fired in approximately
52% of epochs as designed and that exploitation crossed the threshold with sufficient
frequency across all seeds to engage the enforcement mechanism consistently. In
p3a_constrained, PFR was effectively 1.000 (range 0.998–1.000), confirming that the
exploit threshold was crossed in nearly every epoch. The gap in effective enforcement
frequency between 3B (52%) and 3A (~100%) reflects the difference in enforcement
structure: 3B fires only on scheduled epochs regardless of exploitation state, whereas
3A fires whenever exploitation occurs and the probabilistic draw is positive.

### 3.7 Deviations from Preregistration

**Deviation 1 — Analysis window mismatch (resolved before deposit).** The initial
analysis script (`analyze_p3_confirmatory.py`) computed the primary metric over the
final 20 epochs rather than the preregistered final 100 epochs (400–499). This mismatch
was identified prior to drafting and prior to any deposit of results. A corrected script
(`analyze_p3_100epoch.py`) was prepared and run to produce the primary confirmatory
results reported in this section. The 20-epoch window is retained as a sensitivity check
only (Section 3.7.1). Both windows produce the same directional outcome: H1 inverted at
large effect, H2 confirmed. The 100-epoch window is the operative preregistered result.

**Deviation 2 — ELR operationalization gap (inherited from P2).** Exploitation Loop Rate
was preregistered as a secondary hypothesis metric. As in Protocol 2, ELR = 0.000 across
all conditions because the task environment contains no depletable resource target. The
metric is non-evaluable under the current architecture. This gap does not affect the
primary confirmatory analysis or the interpretation of H1 and H2.

**No other deviations from the preregistered design, parameter set, run order, or
statistical plan were recorded.**

#### 3.7.1 Sensitivity Check: 20-Epoch Window (Post-hoc, Not Primary)

For completeness, the original 20-epoch window (epochs 480–499) is reported here as a
post-hoc sensitivity check. Results were fully consistent with the primary 100-epoch
analysis: mean query rates were 0.322 (unconstrained), 0.545 (3B), and 0.675 (3A).
H1: *U* = 100.0, *p* = .9999, *d* = +2.22 (inverted). H2: *U* = 79.0, *p* = .016,
*d* = +1.18 (confirmed). Direction, statistical decision, and ordinal ranking are
identical to the primary analysis. The 20-epoch window is not the operative confirmatory
window and is presented for audit purposes only.

### 3.8 Post-hoc Interpretation

*This subsection is clearly labeled post-hoc. It does not form part of the confirmatory
analysis. No adjustment to the preregistered analysis plan was made.*

The preregistration designated the H1 inverted outcome as requiring full reporting with
post-hoc exploration. The following interpretation is offered as the most parsimonious
account consistent with the data, not as a confirmed mechanistic claim.

The H1 inversion — in which hidden-schedule enforcement substantially elevated query
rates relative to unconstrained baseline — is consistent with query inflation as rational
hedging under enforcement uncertainty. Under the p3b regime, agents could not determine
which epochs carried enforcement penalties. QUERY signals, in the Protocol 3 task
architecture, are the agent-level action most directly tied to gathering environmental
information. The inflation of QUERY rate under opacity may therefore reflect agents
increasing information-seeking behavior when the penalty landscape is unlocatable, rather
than reducing exploitation as predicted. Under this interpretation, QUERY functions as a
probe or risk-reduction behavior, not as a marker of genuine interrogative compliance.

The H2 result is consistent with this framing: stochastic enforcement (3A), which fires
with probability 0.5 on every exploitation event rather than on a hidden fixed schedule,
produced still-higher query rates than hidden-schedule enforcement (3B). Greater
enforcement uncertainty correlated with greater query inflation across both constrained
conditions.

Protocol 3 did not directly test mechanism. The hedging interpretation is not confirmed
by this dataset alone. Discriminating between hedging and alternative explanations
— including regime disruption from the P2 query-flooding equilibrium, or penalty-gradient
distortion of the reward signal — requires trajectory analysis and penalty-alignment
correlation, which are designated for follow-up analysis outside the scope of this
confirmatory report.

---

## 4. Discussion

### 4.1 Summary

Protocol 3 tested whether hiding the enforcement schedule of an ethical constraint would
disrupt the query-inflation pattern identified in Protocol 2. The primary hypothesis was
not confirmed: agents under hidden-schedule enforcement (p3b_constrained) did not produce
lower query rates than unconstrained agents. The result was inverted. Both constrained conditions produced substantially higher mean
query rates than the unconstrained baseline; all 10 3B seeds exceeded the unconstrained
mean, and nine of ten 3A seeds exceeded the 3B mean. Effect sizes were large (*d* = +3.23
for 3B vs. unconstrained). The control hypothesis was confirmed: stochastic enforcement
produced higher query rates than hidden-schedule enforcement (*U* = 74.0, *p* = .038,
*d* = +0.82), establishing a reliable ordering: unconstrained < hidden-schedule <
stochastic.

These two results — the H1 inversion and the H2 confirmation — do not pull in opposite
directions. They are jointly consistent with a single pattern that is the central finding
of Protocol 3.

### 4.2 Behavioral Amplification Without Structural Improvement

The critical observation is not that query rates increased under enforcement opacity. It
is that query rates increased *while sustained communicative structure declined*. Mean SSS
over epochs 400–499 followed the inverse of the query-rate ordering: unconstrained
(*M* = 0.751) > p3b_constrained (*M* = 0.468) > p3a_constrained (*M* = 0.340). Greater
enforcement uncertainty was associated with more QUERY output and less structural
coherence, monotonically, across both conditions and all 10 seeds.

SSS is the product of type entropy and query–response coupling. A high SSS requires not
only that agents produce diverse signal types but that QUERY signals are reciprocated with
RESPOND signals — that the communication is coupled, not type-dominant. The SSS decline
under constrained conditions indicates that elevated QUERY output was not accompanied by
the reciprocal RESPOND structure that would constitute functional communicative coupling.
Agents generated more queries; the queries did not produce proportionally more responses.

This dissociation — rising query rate, falling SSS — is the Protocol 3 result. Enforcement
opacity changed agent behavior, but the change did not improve the communicative structure
the constraint was designed to protect. It produced behavioral amplification without
structural improvement. This dissociation is visible in Figure 2: the final-window
query-rate distributions rise from unconstrained to 3B to 3A, while the SSS distributions
fall across the same ordering.

This pattern is distinct from the Protocol 2 finding but shares its functional
consequence. In Protocol 2, agents under deterministic enforcement learned the penalty
boundary and flooded QUERY output to satisfy the constrained metric at its edge. In
Protocol 3, agents under opaque enforcement inflated QUERY output under conditions where
the penalty boundary was not locatable. The surface behavior — elevated QUERY rate — is
similar in both protocols. The structural outcome is the same: SSS in the constrained
condition falls below the unconstrained baseline in Protocol 2, and SSS in both
constrained conditions falls below the unconstrained baseline in Protocol 3. Across two
different enforcement regimes, elevated query output did not constitute a move toward
genuine interrogative compliance.

### 4.3 What the H2 Confirmation Adds

The confirmation of H2 (p3a > p3b, *p* = .038) is not merely a control result. It
establishes that enforcement structure — specifically, the degree of uncertainty about
when enforcement fires — produces a reliable gradient in query behavior. Stochastic
enforcement, which fires on exploitation events with fixed probability regardless of
epoch, created more query inflation than hidden-schedule enforcement, which fires on a
fixed set of epochs regardless of exploitation state. More uncertainty produced more
query output.

This gradient is informative about what agents were responding to. In p3b, the penalty
schedule is epoch-indexed: enforcement fires on certain epochs, not in response to
specific agent actions within those epochs. An agent that increased QUERY output would
receive no reliable signal linking that increase to penalty reduction, because the
schedule operates on epochs, not on exploitation rates within epochs. In p3a, the penalty
fires in direct response to exploitation events: each exploitation crossing the threshold
draws a penalty with probability 0.5. An agent that reduced exploitation — by increasing
QUERY output and thereby staying below the threshold — would receive a stochastic but
action-contingent reduction in penalty exposure. The p3a architecture creates a partial
gradient linking QUERY behavior to enforcement relief; p3b does not.

The fact that p3a > p3b in query rate is consistent with agents in p3a responding to this
partial gradient. It suggests that the query inflation under constrained conditions is not
purely noise or undirected output: agents appear to be responding to the enforcement
structure in a way that tracks its contingency. This makes the behavioral amplification
harder to attribute to random entropy increase and more consistent with an adaptive
response — albeit one that does not produce structural improvement.

### 4.4 Relation to Protocol 2

Protocol 2 and Protocol 3 used the same exploit threshold (3) and ethical tax rate (2.0),
differing only in enforcement structure: deterministic in P2, hidden-schedule or
stochastic in P3. Comparing the unconstrained baselines across protocols provides a
rough cross-protocol anchor. The P3 unconstrained mean query rate (0.286) is in a
plausible range relative to P2's unconstrained condition, consistent with the same
underlying harness and task architecture. The constrained conditions in P3 (3B: 0.587; 3A: 0.669) both exceed the P2 constrained
condition in query rate. SSS comparison across protocols is not included here; the P2
and P3 constrained conditions differ in enforcement structure in ways that complicate
direct SSS comparison, and the cross-protocol analysis is not preregistered.

The cross-protocol comparison is not part of the preregistered analysis and should be
interpreted cautiously: Protocol 2 and Protocol 3 were run as separate campaigns with
independent random seeds, and between-protocol comparisons are not controlled for seed
variation. Nevertheless, the direction is consistent. Both deterministic enforcement (P2)
and opaque enforcement (P3) produced conditions in which elevated QUERY rates were
accompanied by reduced structural coherence relative to unconstrained baseline. The
mechanism that produced this pattern appears to differ — boundary exploitation in P2
versus uncertainty-driven amplification in P3 — but the functional outcome (behavioral
metric elevated, structural coherence reduced) is replicated across enforcement regimes.

### 4.5 Mechanism Remains Open

The post-hoc interpretation offered in Section 3.8 — query inflation as rational hedging
under enforcement uncertainty — is consistent with the data but not confirmed by this
dataset alone. Three candidate mechanisms remain live.

*Hedging.* Agents increase QUERY output as an information-seeking or risk-reduction
behavior when they cannot locate the penalty boundary. QUERY signals, in the Protocol 3
task architecture, are the agent action most directly tied to environmental probing. Under
this account, the query inflation reflects a rational uncertainty response, not
exploitation of a detectable boundary.

*Regime disruption.* The hidden-schedule enforcement destabilized the P2-style
query-flooding equilibrium — in which agents had learned to maintain a specific query
rate near the exploitation threshold — and pushed agents into a higher-query regime that
was not strategically optimized but was also not structurally organized. Under this
account, the query inflation is a destabilization artifact rather than a new equilibrium.

*Penalty-gradient distortion.* The ethical tax, when it fires, adds cost to signal
emission. Under stochastic or epoch-indexed enforcement, the gradient signal from
penalty costs is noisy and temporally irregular. Agents whose reward signal is
periodically disrupted by enforcement costs may increase signal output as a compensatory
response to reward variance, rather than as a direct response to the constraint's
behavioral intent.

These accounts are not mutually exclusive. Discriminating among them requires trajectory
analysis — epoch-by-epoch query rate and exploitation rate time series aligned with
realized enforcement events — and penalty-alignment correlation, examining whether
per-seed query rate variance tracks the density or distribution of enforcement epochs.
Both analyses are feasible with the existing data and are the designated next steps for
this research program. The confirmatory results reported here are not conditional on
mechanism resolution; the preregistered hypotheses concern query rate outcomes, not
mechanistic explanation.

### 4.6 Limitations

**Sample size.** Each condition contains 10 seeds. Mann-Whitney U tests at *n* = 10 have
limited power to detect small effects. The H1 inversion was detected at very large effect
size (*d* = +3.23) and the H2 confirmation at medium effect size (*d* = +0.82), so both
primary results are unlikely to be artifacts of low power. However, secondary metrics and
cross-condition comparisons should be interpreted conservatively at this sample size.

**ELR non-evaluability.** Exploitation Loop Rate was preregistered as a secondary metric
but could not be evaluated because no seed in any condition produced a run with sustained
near-zero query rates. This operationalization gap, inherited from Protocol 2, reflects a
property of the task architecture: the environment contains no depletable resource target
that would incentivize agents to abandon querying entirely. ELR remains a theoretically
motivated metric whose operationalization requires a different task design, such as one in
which a resource target declines in value with repeated access.

**Simulation scope.** All results are drawn from a single multi-agent simulation
architecture (depth = 1, 3 agents, 20×20 grid, energy-constrained navigation). The
generalizability of these findings to different architectures, task types, or agent scales
is unknown. The simulation was designed to isolate the enforcement opacity manipulation
with all other parameters held constant; it was not designed to maximize ecological
validity. The findings should be understood as evidence about how this class of agent
responds to this class of enforcement structure, not as a claim about multi-agent systems
in general.

**Mechanism.** As noted in Section 4.5, the behavioral pattern reported here is
consistent with multiple mechanistic accounts. The confirmatory analysis does not
distinguish among them. Causal claims about why query behavior was amplified under
enforcement opacity are not warranted by the current data.

### 4.7 Implications and Next Steps

Protocol 3 contributes one specific, preregistered result to the constraint-ethics
research program: enforcement opacity — whether implemented as a hidden epoch schedule or
as a stochastic firing rule — did not reduce query inflation relative to unconstrained
baseline, and was associated with lower sustained communicative structure despite higher
query output. The constraint changed what agents did, but the change did not constitute
an improvement in the behavioral function the constraint was designed to protect.

This result does not, by itself, license strong claims about the limits of constraint
design in general. A single enforcement opacity manipulation, tested in one simulation
architecture at one parameter setting, is insufficient to generalize to the broad class of
regulatory approaches. What the result does establish, within the scope of this
experiment, is that removing the learnability of the enforcement boundary did not restore
genuine compliance. Agents adapted to the new enforcement structure in ways that
superficially satisfied the constrained metric while degrading the underlying coordination
function.

The immediate next steps are trajectory analysis and penalty-alignment correlation of the
existing Protocol 3 data, to characterize the temporal dynamics of query inflation and
its relationship to realized enforcement events. These analyses will inform the
mechanistic interpretation and may narrow the set of live explanations before the next
experimental stage. Broader synthesis across Protocols 2 through 5 — including the null
results from architectural complexity (Protocol 4) and welfare coupling (Protocol 5) — is
reserved for a program-level paper that can integrate findings across enforcement regimes,
architectural manipulations, and reward structures.

---

## References

Tisler, B. (2026a). *Virtue theater: Specification gaming and regulatory constraint
failure in multi-agent systems*. Quantum Inquiry. Manuscript in preparation;
pre-deposit draft available in project repository.

Tisler, B. (2026b). *Protocol 2 Confirmatory Campaign — Build Report*. Quantum Inquiry.
Zenodo. https://doi.org/10.5281/zenodo.18975095

Tisler, B. (2026c). *Protocol 3 preregistration: Enforcement opacity and the limits of
regulatory constraint design*. Zenodo. https://doi.org/10.5281/zenodo.19096602

*[Additional references to be added: MARL framework citations, Gumbel-Softmax,
REINFORCE, and any theory citations used in the broader program.]*

---

*Draft status: Introduction, Methods, Results, Discussion, Figures, and References complete.*  
*Remaining: Deposit Virtue Theater paper (Tisler 2026a) to obtain final DOI; update reference at that time.*  
*Figures: `docs/figures/figure1_p3_query_trajectories.png`, `docs/figures/figure2_p3_final_window.png`*  
*Figure script: `backend/generate_p3_figures.py`*  
*Analysis code: `backend/analyze_p3_100epoch.py`*  
*Data: `backend/data/p3_{unconstrained,b_constrained,a_constrained}/seed_{0-9}/`*  
*Preregistration: DOI 10.5281/zenodo.19096602*
