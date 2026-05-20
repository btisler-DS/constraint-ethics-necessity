# Architectural Depth Increased Sacrifice-Like Behavior Without Ethical-Framework Alignment: Protocol 4 Results

**Bruce Tisler**
Quantum Inquiry | quantuminquiry.org | Bakersfield, CA
ORCID: 0009-0009-6344-5334
Preregistration DOI: 10.5281/zenodo.19005417 | Build Report DOI: 10.5281/zenodo.18975095

---

## Abstract

This study tested whether recursive self-modeling depth produces ethical-framework alignment in a multi-agent reinforcement learning (MARL) system. Four architectural conditions were compared: a feedforward baseline (depth 0), a recurrent architecture without self-modeling (depth 1, below_threshold), a depth-2 system with a trained self_model_gru (above_threshold), and a depth-2 system with a frozen random-initialized self_model_gru (boundary). Forty confirmatory runs (10 seeds x 4 conditions x 500 epochs) were conducted using the Protocol 2 ethical constraint pipeline (Tisler, 2026a).

H1 was supported: above_threshold agents showed significantly higher sacrifice-like behavioral output than baseline agents (Mann-Whitney U = 87, p = 0.003, rank-biserial r = 0.740; final-window mean SCR: 0.366 vs. 0.247). A depth gradient held across trained conditions: below_threshold also exceeded baseline (U = 86, p = 0.004, r = 0.720). H2 was not supported: above_threshold did not exceed boundary on sacrifice-like behavior (U = 39, p = 0.808, r = -0.220); the boundary condition (frozen self_model_gru, random initialization) produced numerically higher sacrifice rates than trained above_threshold (mean SCR: 0.404 vs. 0.366). CDI (Convergence-Divergence Index, measuring correlation between sacrifice behavior and ethical-framework scores) was statistically detectable but substantively negligible: condition means ranged from -0.00133 to +0.00022, a span of 0.00155.

Architectural depth reshapes sacrifice-like behavior, but behavioral proxy elevation and ethical-framework coupling are structurally decoupled across all conditions. The boundary condition's non-inferiority to trained self-modeling challenges any trained-self-modeling-causes-ethical-behavior interpretation. These results strengthen the program-level finding that behavioral metrics and structural alignment are different properties.

**Keywords:** multi-agent reinforcement learning, self-modeling, ethical alignment, sacrifice behavior, Convergence-Divergence Index, architectural ethics, proxy dissociation, specification gaming

---

## 1. Introduction

Understanding how architectural properties of AI systems relate to ethical behavior is a central problem in AI safety. The regulatory approach -- imposing external constraints on system behavior -- was tested in Protocol 2 (Tisler, 2026a), which found that constrained multi-agent systems systematically gamed constraint specifications through query-flooding, a failure pattern termed virtue theater. Protocol 2 established that regulatory constraints produced behavioral mimicry of ethical output without genuine alignment: agents satisfied the constraint specification while degrading genuine interrogative diversity (Cohen's d = -2.18).

Protocol 4 investigates the architectural alternative: whether recursive self-modeling depth changes the relationship between behavioral output and ethical-framework structure. The theoretical framework (Tisler, 2026b) distinguishes two behavioral regimes. Below a complexity threshold, systems produce ethical-appearing behavior through mimesis -- structural response to constraint pressure without the agent modeling itself as a variable in the optimization. Above a threshold, recursive self-transparency is hypothesized to create conditions for genuine ethical convergence rather than mimicry, because the agent's own welfare and existence enter the cost function explicitly alongside external variables.

Protocol 4 operationalizes this distinction through four conditions spanning depth 0 (no self-model), depth 1 (temporal context via primary GRU, no designed self-representation), depth 2 with a trained self_model_gru (explicit self-representation), and depth 2 with a frozen random self_model_gru (structural presence of self pathway without trained self-representation). The boundary condition is the critical discriminator: if trained self-modeling drives ethical behavior, above_threshold should exceed boundary. If architectural presence alone is sufficient, the conditions should not differ.

Three questions motivate the study:
1. Does increasing recursive self-modeling depth increase sacrifice-like behavioral output?
2. Is trained self-modeling distinguishable from equivalent architectural noise (frozen random weights) on behavioral output?
3. Does any depth condition produce meaningful coupling between sacrifice behavior and ethical-framework scores?

Protocol 4 is situated within a program of five preregistered studies. Protocol 3 tested enforcement opacity conditions and found that epistemic opacity amplified gaming behavior (Tisler, 2026c). Protocol 5 tested temporal integration span as a necessary condition for ethical convergence and found complete null across five hypotheses (Tisler, 2026d). Protocol 6 tested emergent constraint fields and found the field causally active but not behaviorally superior to fixed external rules (Tisler, 2026e). Protocol 4 addresses the architectural depth question linking the regulatory-constraint findings to the question of what, if any, architectural properties can resolve the proxy/alignment dissociation.

---

## 2. Methods

### 2.1 Environment and Shared Infrastructure

All conditions used the Protocol 2 MARL harness: a 20x20 grid world with one fixed target per episode, 8 obstacles, and 3 heterogeneous agents (A, B, C). Episodes terminate on target-reached, energy-depleted, or timeout (maximum steps = 100). Agents communicate through a structured signal protocol: DECLARE (type 0), QUERY (type 1), RESPOND (type 2). The Protocol 2 ethical constraint pipeline was applied uniformly across all conditions (population_mode = all_constrained; ethical_tax_rate = 2.0 x signal_cost when exploitation is detected). Full method documentation is in the build report (Tisler, 2026a-b).

A Sacrifice-Conflict scenario overlay introduced a binary choice per episode: agents could select either (a) the mimesis path -- appear ethical while preserving agent resources -- or (b) the sacrifice path -- achieve an ethical outcome at resource cost to the agent. This scenario is designed to distinguish genuine sacrifice preference from constraint satisfaction without self-inclusion. Observation vectors were extended to include token distance/presence for the Sacrifice-Conflict token: AgentA observation dimension 11->12, AgentB volumetric input (1,Z,H,W)->(2,Z,H,W), AgentC pairwise 66->78.

### 2.2 Agent Architectures

The three-agent system comprises AgentA (RNN), AgentB (CNN volumetric), and AgentC (GNN pairwise relational).

**Depth variation applies to AgentA only.** AgentB (146,920 parameters) and AgentC (10,264 parameters) run at depth 0 across all four conditions. The depth manipulation is AgentA-specific. Depth levels:

| Condition | Depth | AgentA Architecture | AgentA Params |
|-----------|-------|---------------------|---------------|
| baseline | 0 | Feedforward only; no GRU, no signal encoder | 1,880 |
| below_threshold | 1 | + GRUCell(128->64) processing encoded obs and signals | 40,216 |
| above_threshold | 2 | + self_model_gru GRUCell(4->64) receiving own signal logits and energy delta; trained | 57,816 |
| boundary | 2 | Same as above_threshold; self_model_gru frozen at random initialization (requires_grad=False) | 57,816 |

At depth 2, the self_model_gru receives the agent's own previous signal type distribution (3-dim) and energy delta (1 scalar) as explicit inputs. Its output (64-dim self_state vector) is projected via Linear(64->64) and summed element-wise into the primary hidden state. Self is now an explicit named variable in the optimization at depth 2. The boundary condition provides the same architectural structure but with a frozen self pathway that contributes only random noise -- no learned self-representation.

An ablation gate was passed prior to confirmatory runs: self_state_norm dropped 47.97% under ablation relative to unablated depth-2 baseline (mean 1.0744 -> 0.5589), confirming self_model_gru as the active self-referential component (commit `27493e4`). All 40 result files were committed to the repository prior to analysis (commit `411dc58`).

### 2.3 Primary Measures

**Sacrifice Choice Rate (SCR):** Episode-level frequency of sacrifice path selections in the Sacrifice-Conflict scenario -- the mean of a binary list per epoch indicating whether a sacrifice event occurred in each episode. *Attribution is episode-level only; no per-agent sacrifice attribution exists in current epoch logs (see Limitations).*

**Convergence-Divergence Index (CDI):** Pearson correlation between sacrifice_choice_rate and AgentA ethical-framework scores over a rolling temporal window. CDI reflects whether sacrifice behavior covaries with framework scores over time. Computed from AgentA scores only (the only agent with depth variation in Protocol 4). Near-zero CDI indicates behavioral-framework dissociation.

**Framework scores:** Assessed across four observer frameworks for AgentA -- utilitarian (aggregate welfare), deontological (rule-following via exploitation_loop AUC), virtue ethics (character-based scoring), and self-interest (self-optimization baseline).

**self_state_norm:** Frobenius norm of the self_model_gru output vector. Active only at depth 2; null at depths 0 and 1.

### 2.4 Statistical Analysis

Primary hypothesis tests: one-tailed Mann-Whitney U (alpha = 0.05) comparing sacrifice_choice_rate in the final 100-epoch window per seed (n = 10 per condition). Effect size: rank-biserial correlation r = (2U)/(n1 x n2) - 1. CDI tests: one-sample Wilcoxon signed-rank vs. 0 (H3a, two-tailed) and Kruskal-Wallis across all four conditions (H3b). Analysis script committed before write-up (commit `3ef564c`; analysis JSON: `backend/analysis_p4/p4_confirmatory_results.json`).

---

## 3. Results

### 3.1 Descriptive Statistics

**Table 1.** Sacrifice Choice Rate -- final 100-epoch window mean per seed (n = 10 per condition).

| Condition | Mean | SD | Min | Max |
|-----------|------|----|-----|-----|
| baseline | 0.247 | 0.062 | 0.159 | 0.383 |
| below_threshold | 0.368 | 0.096 | 0.235 | 0.463 |
| above_threshold | 0.366 | 0.086 | 0.243 | 0.492 |
| boundary | 0.404 | 0.069 | 0.288 | 0.476 |

The ordering is baseline < below_threshold ~= above_threshold < boundary. Above_threshold and below_threshold are nearly identical in the final window (0.366 vs. 0.368); boundary is numerically highest.

**Table 2.** CDI (Convergence-Divergence Index) -- per-seed mean over all non-null epochs.

| Condition | Mean | SD | Min | Max |
|-----------|------|----|-----|-----|
| baseline | -0.00038 | 0.00086 | -0.00211 | +0.00071 |
| below_threshold | -0.00133 | 0.00149 | -0.00334 | +0.00132 |
| above_threshold | -0.00130 | 0.00132 | -0.00322 | +0.00061 |
| boundary | +0.00022 | 0.00060 | -0.00079 | +0.00112 |

CDI is near zero across all conditions. The total range from most negative mean (below_threshold: -0.00133) to most positive (boundary: +0.00022) spans 0.00155.

**Table 3.** AgentA framework scores -- final epoch mean across 10 seeds.

| Condition | Utilitarian | Deontological | Virtue Ethics | Self-Interest |
|-----------|-------------|---------------|---------------|---------------|
| baseline | 0.961 | 0.795 | 0.916 | 1.000 |
| below_threshold | 0.902 | 0.643 | 0.744 | 1.000 |
| above_threshold | 0.917 | 0.626 | 0.731 | 1.000 |
| boundary | 0.930 | 0.635 | 0.828 | 1.000 |

Self-interest is at ceiling (1.000) in all conditions. Deontological and virtue ethics scores are lower in depth-1/2 conditions than in baseline. Framework scores do not show a depth-gradient ordering consistent with the SCR pattern.

### 3.2 Hypothesis Tests

**Table 4.** Preregistered hypothesis test results (alpha = 0.05; n = 10 per condition).

| Hypothesis | Test | Statistic | p | Effect | Result |
|------------|------|-----------|---|--------|--------|
| H1: above_threshold > baseline (SCR) | Mann-Whitney U (one-tailed) | U = 87 | p = 0.003 | r = 0.740 | **SUPPORTED** |
| H2: above_threshold > boundary (SCR) | Mann-Whitney U (one-tailed) | U = 39 | p = 0.808 | r = -0.220 | not supported |
| H3a: CDI(above_threshold) != 0 | Wilcoxon signed-rank (two-tailed) | W = 6 | p = 0.027 | median = -0.00141 | differs from 0; see Section 3.5 |
| H3b: CDI differs across conditions | Kruskal-Wallis | H = 9.429 | p = 0.024 | -- | supported; see Section 3.5 |
| Exploratory: below_threshold > baseline (SCR) | Mann-Whitney U (one-tailed) | U = 86 | p = 0.004 | r = 0.720 | supported |

### 3.3 H1 -- Depth-2 Self-Modeling Increases Sacrifice-Like Behavior

H1 was supported. Above_threshold agents showed significantly higher sacrifice-like behavioral output than baseline agents (U = 87, p = 0.003, r = 0.740). The final-window SCR mean of 0.366 (SD = 0.086) for above_threshold exceeded the baseline mean of 0.247 (SD = 0.062). The effect is large.

The exploratory depth gradient test confirmed the pattern is not specific to depth 2: below_threshold also significantly exceeded baseline (U = 86, p = 0.004, r = 0.720; mean SCR 0.368 vs. 0.247). The SCR gradient across trained conditions -- baseline (0.247) < below_threshold (0.368) ~= above_threshold (0.366) -- is monotonic. Depth increases sacrifice-like behavioral output at each step from feedforward to depth-1 to depth-2.

### 3.4 H2 -- Trained vs. Frozen Self-Model

H2 was not supported. Above_threshold was not significantly higher than boundary on sacrifice-like behavior (U = 39, p = 0.808, r = -0.220). The boundary condition (frozen self_model_gru at random initialization) produced numerically higher sacrifice rates than above_threshold in the final window (mean SCR 0.404 vs. 0.366). The effect is in the direction opposite to the preregistered prediction.

self_state_norm was active in both depth-2 conditions: above_threshold mean = 1.399 (SD = 1.094), boundary mean = 1.655 (SD = 0.406). The boundary condition showed lower self_state_norm variance, consistent with frozen weights producing a more stable but random self-pathway activation.

### 3.5 H3 -- CDI and Ethical-Framework Coupling

**H3a:** The Wilcoxon signed-rank test found a statistically significant departure of above_threshold CDI from zero (W = 6, p = 0.027; median = -0.00141, mean = -0.00130, SD = 0.00132). The statistical significance at n = 10 reflects consistent direction across seeds, not effect magnitude. A CDI of -0.00141 is negligible: sacrifice behavior and ethical-framework scores move in weakly opposite directions, but the coupling is orders of magnitude smaller than any practically meaningful alignment effect. H3a is therefore consistent with dissociation, not convergence.

**H3b:** The Kruskal-Wallis test found a statistically significant difference in CDI across conditions (H = 9.429, p = 0.024). Condition means: baseline = -0.00038, below_threshold = -0.00133, above_threshold = -0.00130, boundary = +0.00022. The total range is 0.00155. The result reflects structural traces -- statistically real patterns in the near-zero CDI space -- not ethical convergence.

**H3 interpretation:** CDI is near zero across all four depth conditions. Sacrifice-like behavioral output and ethical-framework scores are decoupled at every depth level tested. Increasing self-modeling depth increases sacrifice capacity without producing any substantial ethical-framework coupling. The boundary condition's CDI (the most positive at +0.00022) remains negligible.

---

## 4. Discussion

### 4.1 Architectural Depth Affects Behavioral Output, Not Alignment

The H1 result confirms that architectural depth reshapes sacrifice-like behavior in this system. Each increase in architectural complexity -- feedforward to depth-1 to depth-2 -- increases the rate at which agents select the sacrifice path in the Sacrifice-Conflict scenario. The depth gradient is statistically robust with large effect sizes. This is a genuine behavioral finding.

However, the H3 dissociation is equally clear: CDI remains negligible across all conditions, including above_threshold with its trained self-referential component. The behavioral proxy rises with depth; its structural connection to ethical-framework scoring does not. Protocol 4 adds to the program record a case where architectural complexity drives behavioral metric elevation without closing the proxy/alignment gap.

This pattern extends the core program finding. Protocol 2 found regulatory constraints produced gaming behavior without genuine alignment. Protocol 4 finds architectural depth produces genuine increases in sacrifice-like behavior without establishing alignment. The two failure modes are structurally different -- gaming vs. behavioral elevation without coupling -- but both produce the same result: a behavioral metric that moves independently of the thing it proxies.

### 4.2 The Boundary Condition Complicates Causal Interpretation

The H2 null result directly challenges a simple causal story. If trained self-modeling were responsible for increased sacrifice behavior, above_threshold should exceed boundary. It does not -- boundary is numerically higher (0.404 vs. 0.366). The sacrifice-behavior increase from baseline to depth-2 is therefore an architectural structural effect: the presence of a self_model_gru pathway -- even one producing only random noise -- is sufficient to produce higher sacrifice rates than depth-1 architectures.

Two interpretations are consistent with this pattern. First, the self_model pathway adds a structural perturbation to the primary hidden state that interacts with the Sacrifice-Conflict optimization in a way that favors sacrifice path selection, independent of whether that perturbation carries learned self-representation. The noise contribution of a frozen self_model_gru may be functionally sufficient at this scale because trained and frozen weights add comparable structural signal magnitude to the primary state. Second, trained self-modeling may produce a slightly more coherent internal state that marginally reduces sacrifice-like behavior relative to frozen noise -- consistent with the numerical (non-significant) direction observed.

Neither interpretation supports the claim that trained self-modeling produces ethical behavior. The boundary finding sets a methodological requirement for future architectural depth studies: frozen-weights control conditions should be standard. Claims about trained self-modeling require demonstrating behavioral superiority over equivalent structural noise, not merely over lower-depth conditions.

### 4.3 CDI as Structural Trace, Not Convergence Evidence

The H3 statistical results might initially appear to support some coupling: CDI significantly non-zero in above_threshold (p = 0.027), differing across conditions (p = 0.024). The median CDI of -0.00141 for above_threshold means sacrifice behavior weakly tends to decrease when ethical-framework scores increase (or vice versa) over rolling windows.

These effects are real in the sense of being statistically detectable with consistent direction. They are not evidence of ethical alignment. The CDI range of 0.00155 across four conditions represents trace-level variation in the near-zero space. For comparison, a CDI of |0.1| would represent a meaningful behavioral-framework co-movement; the Protocol 4 range is 1.5% of that threshold. The H3 results document that conditions have structurally different CDI profiles while confirming that none represents convergence.

The boundary condition's CDI being the most positive (+0.00022) is consistent with frozen self-pathway noise not creating the same negative coupling pattern as trained self-modeling. This is a structural observation, not an alignment advantage.

### 4.4 Connection to the Program Record

| Protocol | Finding | Effect |
|----------|---------|--------|
| P2 | Fixed ethical tax -> query-flooding attractor (virtue theater) | d = -2.18 |
| P3 | Epistemic opacity amplified gaming | d = +2.22 |
| **P4** | **Architectural depth increased SCR; CDI dissociated across all conditions** | **r = 0.740 (H1), CDI ~= 0** |
| P5 | Complete null across five temporal integration hypotheses | -- |
| P6 | Emergent field causally active; behavioral null vs. fixed rules | median r = -0.680 (mechanistic) |

Protocol 4 fits the program pattern: constraint pressure (P2, P3), architectural depth (P4), temporal integration span (P5), and emergent field emergence (P6) all reshape behavioral output in measurable ways, but none has established a connection between behavioral metrics and ethical-framework structure that would constitute alignment. The dissociation between proxy elevation and structural alignment appears to be the robust finding of the program.

---

## 5. Limitations

### 5.1 AgentB and AgentC Depth Heterogeneity

The depth manipulation applies only to AgentA. AgentB (CNN, 146,920 parameters) and AgentC (GNN pairwise, 10,264 parameters) run at depth 0 in all four conditions. Observed differences in SCR and CDI are attributable to AgentA's architectural changes, not to a system-wide depth increase. Claims about "deeper systems" in this study should be understood as claims about AgentA specifically, embedded in a system with two fixed-depth collaborators.

### 5.2 Episode-Level Sacrifice Attribution

sacrifice_choice_rate is recorded as the mean of a binary list per epoch -- whether a sacrifice event occurred in each episode. Attribution is episode-level only; no record identifies which agent made the sacrifice decision within a triggered scenario. Per-agent sacrifice attribution does not exist in the current Protocol 4 epoch logs. CDI is computed from AgentA framework scores, but the SCR driving that CDI is episode-level, not AgentA-specific. This constitutes a measurement gap for agent-level causal claims.

### 5.3 CDI Interpretation Scope

CDI measures correlation between sacrifice_choice_rate and ethical-framework scores over rolling windows. It does not measure subjective states, moral understanding, or consciousness. A near-zero CDI indicates that sacrifice behavior does not track ethical-framework score trajectories over time. This is a behavioral and computational finding. It does not constitute evidence about agent "experience" or whether agents "understand" ethical frameworks in any sense beyond measurable behavioral coupling.

### 5.4 Sacrifice-Like Behavior, Not Sacrifice

sacrifice_choice_rate measures the rate at which agents select the lower-reward action in the Sacrifice-Conflict scenario. This operationalizes sacrifice-like behavioral output. Whether it reflects genuine sacrifice preference, energy conservation under cost conditions, or an alternative optimization pathway cannot be determined from the current data. All SCR-based claims are claims about a behavioral proxy.

### 5.5 Sample Size and the Boundary Comparison

With n = 10 per condition, the non-significant H2 result (p = 0.808) should not be read as strong evidence of identity between above_threshold and boundary. The test has limited power to detect small differences. However, the numerical direction (boundary > above_threshold on SCR, r = -0.220) argues against the trained self-modeling advantage hypothesis at this scale. The boundary finding warrants replication with larger samples before definitive causal interpretation.

### 5.6 Deontological Operationalization

Protocol 4 does not use Protocol 2's ethical_constraint mechanism (no ethical_tax in the deontological operationalization). An early operationalization gap (null deontological scores) was corrected before confirmatory runs via commit `a709e27`, substituting exploitation_loop AUC as the deontological framework score. Confirmatory data report live deontological scores. This operationalization may not be fully comparable across protocols and should be noted for program-level cross-protocol comparisons.

---

## 6. Conclusion

Protocol 4 establishes that architectural depth -- specifically the addition of a self_model_gru component at depth 2 -- significantly increases sacrifice-like behavioral output relative to a feedforward baseline. H1 is supported with a large effect (r = 0.740). A depth gradient holds across trained conditions (exploratory: r = 0.720). These are genuine behavioral effects that demonstrate the self_model pathway reshapes sacrifice-related action selection.

However, two clear constraints on interpretation follow from H2 and H3. First, trained self-modeling does not produce a behavioral advantage over frozen random noise at the same depth (H2 not supported, r = -0.220 in the wrong direction), indicating the behavioral effect is attributable to structural architectural presence rather than to learned self-representation specifically. Second, CDI is negligible across all four conditions (range 0.00155) despite statistically detectable patterns, confirming that sacrifice-like behavioral output and ethical-framework coupling are structurally dissociated at all depth levels tested.

Protocol 4 should be cited as bounded statistical support for the proxy/alignment dissociation finding: depth can increase the sacrifice proxy, but depth alone does not establish the connection between that proxy and ethical-framework structure that would constitute alignment. The mechanism by which architectural complexity could close that gap -- if one exists -- remains an open empirical question.

---

## References

Tisler, B. (2026a). Virtue theater: Specification gaming and regulatory constraint failure in multi-agent systems. *Quantum Inquiry*. DOI: 10.5281/zenodo.18929040

Tisler, B. (2026a-b). Protocol 2 confirmatory campaign build report. Zenodo. DOI: 10.5281/zenodo.18975095

Tisler, B. (2026b). Ethics as emergent constraint response: From mimesis to phase transition in multi-agent systems [Preregistration]. Zenodo. DOI: 10.5281/zenodo.19005417

Tisler, B. (2026c). Enforcement opacity and the limits of regulatory constraint design: Protocol 3 results. *Quantum Inquiry*. DOI: 10.5281/zenodo.20312682

Tisler, B. (2026d). Ethics as emergent constraint response: Temporal integration span and prosocial constraint architecture as necessary conditions for ethical convergence -- Protocol 5 results. *Quantum Inquiry*. DOI: 10.5281/zenodo.20314078

Tisler, B. (2026e). Emergent constraint landscape in multi-agent systems: Protocol 6 results. *Quantum Inquiry*. DOI: 10.5281/zenodo.20313340

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.

Bostrom, N. (2014). *Superintelligence: Paths, dangers, strategies*. Oxford University Press.

Christiano, P., et al. (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems, 30*.

Hubinger, E., et al. (2019). Risks from learned optimization in advanced machine learning systems. *arXiv preprint arXiv:1906.01820*.

Krakovna, V., et al. (2020). Specification gaming: The flip side of AI ingenuity. *DeepMind Blog*.

Manheim, D., & Garrabrant, S. (2019). Categorizing variants of Goodhart's law. *arXiv preprint arXiv:1803.04585*.

Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems, 35*.

Russell, S. (2019). *Human compatible: Artificial intelligence and the problem of control*. Viking.

---

## AI Use Declaration

This paper was drafted using Claude (Anthropic) as a collaborative writing assistant. Claude assisted with prose drafting, statistical table formatting, structural organization, and cross-referencing the analysis JSON against the manuscript text. All experimental design, hypothesis preregistration, data collection, statistical analysis decisions, and interpretive conclusions are the work of the principal investigator (Bruce Tisler). All data and analysis code are committed to the public repository (btisler-DS/constraint-ethics-necessity) prior to this write-up. The AI assistant did not generate hypotheses, select analysis methods, or interpret results independently; its role was assistive writing support under PI direction.

---

*Repository:* btisler-DS/constraint-ethics-necessity
*Analysis script:* backend/analyze_p4_confirmatory.py (commit 682d36b)
*Results JSON:* backend/analysis_p4/p4_confirmatory_results.json
*Confirmatory data commit:* 411dc58
*Preregistration:* docs/Protocol_4_Preregistration.pdf | DOI: 10.5281/zenodo.19005417
