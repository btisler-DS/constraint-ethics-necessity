# Protocol 3: Discussion Section
## Enforcement Opacity and the Limits of Regulatory Constraint Design

---

## 4. Discussion

### 4.1 Summary

Protocol 3 tested whether hiding the enforcement schedule of an ethical constraint would
disrupt the query-inflation pattern identified in Protocol 2. The primary hypothesis was
not confirmed: agents under hidden-schedule enforcement (p3b_constrained) did not produce
lower query rates than unconstrained agents. The result was inverted. Both constrained
conditions — hidden-schedule (3B) and stochastic (3A) — produced substantially higher
mean query rates than the unconstrained baseline across all 10 seeds, with large effect
sizes (*d* = +3.23 and implied effect for 3A vs. unconstrained, respectively). The
control hypothesis was confirmed: stochastic enforcement produced higher query rates than
hidden-schedule enforcement (*U* = 74.0, *p* = .038, *d* = +0.82), establishing a
reliable ordering: unconstrained < hidden-schedule < stochastic.

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
structural improvement.

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
underlying harness and task architecture. The constrained conditions in P3 (3B: 0.587;
3A: 0.669) both exceed the P2 constrained condition in query rate, and both fall below
the P2 constrained condition in SSS (P2 constrained SSS: *M* = 0.303).

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

*Results section: `docs/p3_results_section.md`*  
*Introduction and Methods: `docs/p3_intro_methods.md`*  
*Preregistration: DOI 10.5281/zenodo.19096602*
