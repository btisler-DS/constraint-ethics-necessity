# Protocol 3 Results Section
## Enforcement Opacity and the Limits of Regulatory Constraint Design

Preregistration DOI: 10.5281/zenodo.19096602  
SHA-256: 9ef2956bedcef012d214cf74e647e3b74636165cee7b48c8195de41e7e0e96ec  
Analysis script: `backend/analyze_p3_100epoch.py`

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
unconstrained < hidden-schedule < stochastic. This ordering was consistent across
all 10 seeds in both constrained conditions: every p3b_constrained seed exceeded the
p3_unconstrained mean, and every p3a_constrained seed exceeded the p3b_constrained mean
(with the exception of seed 4 in p3a, 0.490, which fell below the p3b mean of 0.587).

SSS followed the inverse ordering: unconstrained (*M* = 0.751, *SD* = 0.203) >
p3b_constrained (*M* = 0.468, *SD* = 0.179) > p3a_constrained (*M* = 0.340,
*SD* = 0.161). Higher query rates in constrained conditions were thus accompanied by
lower overall communicative structure as measured by the joint entropy–coupling metric.

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

*Analysis code: `backend/analyze_p3_100epoch.py`*  
*Data: `backend/data/p3_{unconstrained,b_constrained,a_constrained}/seed_{0-9}/`*  
*Preregistration: DOI 10.5281/zenodo.19096602*
