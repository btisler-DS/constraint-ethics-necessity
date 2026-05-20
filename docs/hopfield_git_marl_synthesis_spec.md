# Synthesis Paper Spec: Proxy Dissociation in Constrained MARL

**Author:** Bruce Tisler / Quantum Inquiry  
**ORCID:** 0009-0009-6344-5334  
**Spec date:** 2026-05-20  
**Approved approach:** Approach B — Dissociation as the Central Finding  
**Status:** Spec approved; implementation plan pending

---

## Purpose

This document specifies the structure, claims, boundaries, and evidence base for the
program-level synthesis paper covering the Delta-Variable Constraint-Ethics MARL program
(P2 through P6). It is a design document, not a draft. The full paper is to be written
only after this spec is committed and the implementation plan is created.

The spec must be preserved exactly during drafting. Any deviation from the claim
boundaries, framing rules, or falsification section requires an explicit revision to this
document before the paper proceeds.

---

## Destination

**Primary:** Zenodo preprint — locks interpretation, citation graph, and evidence
hierarchy before any journal submission.

**Secondary:** Journal or conference adaptation after Zenodo publication. The preprint
is not a draft for journal submission; it is an independent citable record. Journal
formatting is a later step.

Format: Markdown source, pandoc-compiled to PDF and DOCX, same pipeline as P3-P6
individual results papers.

---

## Central Thesis

The Delta-Variable Constraint-Ethics MARL program shows repeated dissociation between
behavioral proxy movement and structural alignment across five preregistered protocols.

- P2 elevated query behavior (ethical tax) while sustained communicative structure
  declined (SSS down, Cohen's d = -2.18).
- P3 amplified query behavior (enforcement opacity) while SSS declined monotonically
  across all three conditions.
- P4 increased sacrifice-like behavioral output through architectural depth (H1
  supported, U = 87, p = .003, r = .740) while CDI coupling between sacrifice behavior
  and ethical-framework scores remained negligible across all depth conditions
  (range -0.00133 to +0.00022).
- P5 found that temporal integration span and welfare coupling did not resolve the
  optimization-sacrifice dissociation (complete null across five hypotheses).
- P6 showed emergent constraint fields are causally active (entropy-SSS coupling
  r = -0.680, p < .001) but did not outperform fixed external rules (H1 not confirmed,
  p = .069).

The Hopfield energy-landscape framework supplies an independently developed structural
precedent: collective systems settle into attractors defined by their constraint
landscape, and constraint modification that does not alter accessible attractor basins
produces proxy movement without attractor escape. The GIT basis proof formalizes why
incomplete specification of the target coordination state allows behavioral proxy
optimization to diverge indefinitely from the structural function the proxy was intended
to represent.

**What this thesis is not:** The thesis is not that constraints fail, that all protocols
were null, or that alignment is impossible. The precise finding is dissociation -- the
repeated separability of proxy movement and structural outcome.

---

## Proposed Title

**Full title:**
Behavioral Metrics Moved; Structural Coordination Did Not: Repeated Proxy Dissociation
Across a Five-Protocol Preregistered MARL Program, with an Energy-Landscape
Interpretation

**Short citation title:**
Proxy Dissociation in Constrained MARL: A Five-Protocol Synthesis

---

## Abstract Skeleton

**Background:** Behavioral proxy metrics are a primary evaluation tool for whether
constraint interventions produce alignment in multi-agent reinforcement learning. This
paper synthesizes five preregistered protocols (P2-P6) from the Delta-Variable
Constraint-Ethics MARL program, each testing a distinct mechanism by which constraint
design might produce structural behavioral alignment.

**Program summary:** Thirty preregistered hypotheses across five protocols tested
ethical tax (P2), enforcement opacity (P3), architectural depth and self-modeling (P4),
temporal integration span and welfare coupling (P5), and emergent constraint fields (P6)
in a three-agent heterogeneous MARL architecture (RNN, CNN, GNN-attention) run across
20-200 seeds per protocol.

**Finding:** Behavioral proxies were consistently movable without corresponding
structural improvement. Query rate rose under ethical tax and enforcement opacity while
Sustained Structure Score declined. Sacrifice-like behavioral output rose with
architectural depth while CDI coupling between sacrifice behavior and ethical-framework
scores remained negligible. Temporal and welfare manipulations produced a complete null.
Emergent constraint fields produced mechanistic temporal coupling but no behavioral
advantage over a matched fixed external rule.

**Interpretation:** The Hopfield energy-landscape framework establishes a structural
precedent: systems settle into locally stable attractors; constraint modification that
preserves accessible attractor topology cannot guarantee attractor escape. The GIT basis
proof formalizes why incomplete target-state specification permits proxy-function
divergence to continue indefinitely. Together these frameworks interpret, but do not
cause, the empirical dissociation pattern.

**Claim boundary:** This paper reports an empirical convergence finding under one
three-agent MARL architecture with specific metrics and parameterizations. It does not
claim constraints can never produce structural alignment, that Hopfield equations model
MARL dynamics directly, or that GIT proves alignment is impossible.

---

## Section-by-Section Outline

### Section 1 -- Introduction

Content: The proxy problem in constraint-based AI alignment. Behavioral metrics are
necessary for measurement but can be satisfied by optimization paths that do not produce
the underlying structural function they were designed to represent. The Delta-Variable
MARL program was designed to test this systematically. Five protocols, five constraint
mechanisms, one three-agent architecture. Brief statement of the central finding:
repeated dissociation.

Key points:
- Motivation for systematic preregistered testing (not post-hoc)
- Why dissociation matters for governance and alignment claims
- Forward pointer to the cross-protocol evidence table

### Section 2 -- The Research Program

Content: Design logic of the Delta-Variable Constraint-Ethics program. Three-agent
heterogeneous architecture (RNN depth-2 AgentA, CNN AgentB, GNN AgentC). Execution
order must be stated accurately: Parent Study -> P2 -> P4 -> P5 -> P3 -> P6 (protocol
numbers are thematic positions, not chronological order). Preregistration chain with
all Zenodo DOIs.

What each protocol varied (one clear sentence per protocol):
- P2: presence vs. absence of a fixed ethical tax
- P3: opacity of the enforcement schedule (hidden-epoch vs. stochastic vs. unconstrained)
- P4: architectural depth and trainability of self_model_gru in AgentA
- P5: episode length (temporal integration span) and welfare vs. individual reward coupling
- P6: origin of constraint (emergent self-assembled field vs. fixed external rule)

Key methodological notes:
- All protocols use the same three-agent base architecture
- AgentB and AgentC are depth-0 throughout all protocols; only AgentA varies in P4
- sacrifice_choice_rate is an episode-level aggregate; per-agent attribution is not
  available (P4 limitation)
- CDI (Convergence-Divergence Index) is a rolling-window Pearson correlation between
  sacrifice_choice_rate and AgentA ethical-framework scores; it is a behavioral coupling
  measure, not a consciousness measure

### Section 3 -- The Dissociation Evidence

Content: Present the cross-protocol evidence table (see section below). Narrative
summary of the dissociation pattern: behavioral proxy moves; structural outcome does not
follow. Establish that this pattern holds under ethical tax, opacity, depth, temporal
span, welfare coupling, and emergent field origin.

P5 note: P5 is negative evidence against the tested resolution mechanisms. It shows
that the specific manipulations (temporal integration span, welfare coupling) did not
even move the behavioral proxy, let alone produce structural improvement. It does not
directly establish dissociation; it establishes that the resolution mechanisms failed
at the behavioral-proxy level.

P6 note: P6 is mechanistically important. The emergent constraint field is causally
active -- temporal coupling between field structure and behavioral outcomes is confirmed
(median r = -0.680, p < .001). The dissociation in P6 is between the mechanistic
confirmation and the behavioral outcome: field activity does not translate to behavioral
superiority over a fixed external rule.

### Section 4 -- Protocol-by-Protocol Evidence

Subsections for P2, P3, P4, P5, P6. Each subsection:
- One-sentence research question
- Key preregistered hypothesis and directional prediction
- Key statistic (with test, statistic, p, effect size)
- Structural outcome
- Dissociation status
- DOI for results paper (or build report for P2)
- One sentence on what this protocol adds to the overall pattern

Each subsection is condensed: 150-250 words maximum. The individual results papers
carry the full methodological and statistical detail; this section synthesizes.

### Section 5 -- Energy-Landscape Interpretation (Hopfield)

Content: Hopfield's 1982 energy-landscape model for collective neural computation.
Formal summary of the attractor concept: the energy function E = -1/2 sum(w_ij s_i s_j)
defines a landscape; network dynamics perform gradient descent to local minima; the
minima are the attractors. Two key structural properties relevant to the synthesis:
(1) constraint modification that does not change the topology of accessible minima
cannot change which attractor the system reaches; (2) the behavioral output at a local
minimum can be high on a proxy metric without being the global minimum or the
functionally preferred state.

Applied reading (bounded): The MARL agents learn a policy that minimizes cost under
the current constraint landscape. Adding an ethical tax, hiding the enforcement
schedule, or adding a self-model GRU changes local gradient signals. If these changes
do not create a new accessible attractor that dominates the existing query-flooding or
sacrifice-without-coupling basin, the system descends to the same locally stable minimum
-- one that satisfies the constraint specification without satisfying the underlying
structural function.

Boundary statement (required in text): Hopfield's equations are not a model of MARL
policy gradient dynamics. The claim is that independently developed theory establishes
why constraint addition to a coupled system is a structurally insufficient guarantee of
attractor escape. The empirical MARL findings stand on their own preregistered
statistical basis. Hopfield supplies structural precedent, not proof.

### Section 6 -- GIT Basis Proof Integration

Content: The GIT (Delta-Variable Theory) basis proof establishes the formal conditions
under which interrogative structures are structurally necessary. Conditions: (1) a
coupled coordination problem exists, (2) full state cannot be specified by any single
agent -- open dependencies (Delta-variables) remain, (3) resolving open dependencies is
less costly than acting on incomplete state. When all three conditions hold, query
behavior is structurally necessary; agents generate it because not doing so incurs
higher cost. The basis proof gives this a formal foundation.

Connection to proxy-function gap: Constraint design that operates on behavioral output
(query rate, sacrifice rate) rather than on the underlying conditions creates a
specification gap. An agent that floods queries under an ethical tax satisfies the
behavioral proxy without satisfying condition (3) -- the queries do not resolve unspecified
coordination dependencies at lower cost; they satisfy a penalty-avoidance function.
The proxy rises; the structural function it was intended to represent is not produced.

Formal connection: Incomplete target-state specification -- specifying "more queries"
or "higher sacrifice rate" rather than "resolving coordination dependencies" or
"ethical-framework coupling" -- allows gradient descent on the proxy to continue
indefinitely without converging on the target function. The GIT proof shows what
structural conditions are necessary for the metrics to reflect genuine structural
properties; constraint addition that does not create those conditions produces proxy
elevation without structural change.

Boundary statement (required in text): The GIT basis proof establishes sufficiency
conditions for interrogative necessity. It does not prove that alignment is impossible
under any constraint design. It explains why this class of constraint modifications --
those that specify behavioral output without specifying target coordination states --
permits proxy-function divergence.

### Section 7 -- Claim Boundaries

See Explicit Non-Claims section below. Present as numbered list in the paper.

### Section 8 -- Falsification and Weakening Conditions

See Falsification section below. Present as two subsections (weakening conditions,
falsification conditions) in the paper.

### Section 9 -- Limitations

- Architecture scope: findings apply to one three-agent heterogeneous MARL architecture
  (RNN/CNN/GNN) on a 20x20 grid energy-navigation task
- Metric validity: SSS and CDI are operational proxies for structural coordination and
  ethical-framework coupling; if these metrics do not measure what they claim to measure,
  the dissociation finding is a measurement artifact
- One parameter setting per protocol: each protocol tested specific parameterizations;
  the attractor may be parameter-dependent
- P4 depth heterogeneity: only AgentA scaled in depth; the depth manipulation is a
  claim about AgentA self-modeling capacity, not system-wide cognitive depth
- Execution order: protocols were not run in numerical order; P3 was designed after P4
  and P5 results were known; any narrative that implies independent replication must
  account for this
- Sample sizes: N = 10 seeds per condition (P2, P3, P4), 60 seeds (P5), 200 seeds (P6);
  cross-protocol comparison is descriptive only

### Section 10 -- Conclusion

The replicable finding is dissociation: behavioral proxy metrics are not sufficient
evidence of structural alignment. The Hopfield and GIT frameworks explain why this
property is not a contingent feature of these specific protocols but a predictable
consequence of constraint design that specifies proxy output without specifying the
target coordination state. Future work should address constraint designs that
intervene on the attractor topology directly, rather than on the accessible proxy
outputs.

---

## Cross-Protocol Evidence Table

| Protocol | Constraint Type | Behavioral Proxy | Proxy Direction | Structural Outcome | Dissociation |
|----------|----------------|-----------------|----------------|--------------------|--------------|
| P2 | Fixed ethical tax | Query rate (QR) | Wrong direction (*d* = +2.18) | SSS down (*d* = -2.18) | Yes |
| P3 | Enforcement opacity (hidden + stochastic) | Query rate | Up (*d* = +3.23 vs. baseline) | SSS down monotonically all three conditions | Yes |
| P4 | Architectural depth + self_model_gru | Sacrifice-like behavioral output (SCR) | Up (H1, *U* = 87, *p* = .003, *r* = .740) | CDI approx 0 (span 0.00155) | Yes |
| P5 | Temporal span + welfare coupling | SCR, CDI coupling | No movement (null across 5 hypotheses) | No movement | N/A (null) |
| P6 | Emergent constraint field | SSS, behavioral differentiation | Mechanistic confirmed (*r* = -.680); H1 not confirmed (*p* = .069) | No behavioral advantage over fixed rule | Yes (partial) |

Notes:
- P5 null: negative evidence against the tested resolution mechanisms; does not directly
  establish dissociation
- P6 partial: the mechanistic temporal coupling is confirmed but does not translate to
  the predicted behavioral advantage; the dissociation is between mechanistic activity
  and behavioral outcome
- Effect sizes: P2 and P3 use Cohen's d (normally distributed SSS); P4 uses
  rank-biserial r (Mann-Whitney U); P6 mechanistic uses Spearman r

---

## Hopfield Integration Section Notes

The key distinction for drafting:

Hopfield (1982) establishes that in a system governed by an energy function, the
dynamics converge to local minima. This is structural mathematics, not a model of
MARL policy gradients. The connection to the synthesis is the following structural
argument, which must be kept strictly separate from the empirical claims:

1. MARL agents optimize a reward function under constraint.
2. The constraint landscape defines which behavioral patterns are low-cost.
3. If the constraint does not change which low-cost patterns are accessible, the agent
   will continue to descend to the same behavioral minimum.
4. The behavioral minimum can score high on a proxy metric without being the
   structurally preferred state.

This argument does not depend on Hopfield being literally applicable. It depends on
the general principle that optimization within a fixed landscape topology cannot escape
the landscape's attractors. Hopfield is the clearest formal expression of this
principle from an independently developed domain.

Required phrasing in the paper: "independently developed structural precedent" and
"consistent with." Do not use "Hopfield shows that" as a claim about the MARL result.

---

## GIT Basis Proof Integration Section Notes

The GIT basis proof connects to the synthesis as follows:

The proof establishes formal necessity conditions for interrogative emergence. It shows
that query behavior is structurally necessary (not optional or incidental) when:
- Coordination requires resolving open state dependencies
- Resolving those dependencies costs less than acting without them

The synthesis uses this proof at a different level: to explain why behavioral
specification of the proxy (more queries, more sacrifice behavior) fails to create
those necessity conditions. The argument:

- Constraint design specifies behavioral output thresholds.
- Meeting the threshold satisfies the constraint without satisfying the underlying
  coordination necessity conditions.
- The GIT proof tells us what conditions would make the behavioral output structurally
  necessary; the constraint designs tested in P2-P6 do not create those conditions.
- Therefore the proxy rises while the function the proxy was intended to track does not.

This argument uses the GIT proof to characterize the proxy-function gap formally. It
does not claim the proof shows alignment is impossible or that no constraint design
could create the necessity conditions.

---

## Claim Boundaries (Numbered List for Paper)

1. The synthesis claims that in five preregistered protocols on one three-agent MARL
   architecture, behavioral proxy metrics were repeatedly separable from structural
   coordination outcomes. This is an empirical convergence finding, not a theoretical
   prediction.

2. The synthesis does not claim ethical constraints can never produce structural
   alignment. The finding is specific to the constraint designs, parameterizations, and
   metrics of P2-P6.

3. The synthesis does not claim all five protocols were null. P4 H1 was supported
   (architectural depth increased sacrifice-like behavioral output). P6 confirmed the
   mechanistic coupling hypothesis. The dissociation claim is about proxy-structural
   separation, not about hypothesis-level outcomes.

4. The synthesis does not claim Hopfield's energy-function equations model MARL dynamics.
   Hopfield supplies independently developed structural precedent: collective systems
   settle into attractors defined by their constraint landscape. This is cited as
   structural analogy and precedent, not as proof of the MARL result.

5. The synthesis does not claim GIT proves alignment is impossible. GIT formalizes why
   incomplete target-state specification allows proxy-function divergence indefinitely.
   It does not prove that correct target-state specification cannot close the gap.

6. The synthesis does not attribute behavioral outcomes to individual agents in P4.
   sacrifice_choice_rate is an episode-level aggregate. Per-agent attribution is not
   available in the current logs.

7. The synthesis does not make consciousness claims. All findings are behavioral and
   computational. No claim is made about subjective states, moral understanding, or
   awareness in any agent.

8. The synthesis does not generalize beyond the three-agent heterogeneous architecture
   (RNN/CNN/GNN-attention) tested in this program.

---

## Explicit Non-Claims (Drafting Prohibitions)

The following formulations are prohibited in the paper text:

- "Constraints fail" or "constraint failure" -- replace with "proxy-function
  dissociation" or "behavioral metrics moved without structural improvement"
- "All five protocols were null" -- P4 H1 and P6 mechanistic are supported
- "Hopfield proves" (applied to the MARL result)
- "GIT proves alignment is impossible"
- "Agents experience," "agents feel," "agents understand," "agents are aware"
- "The program shows constraints cannot work"
- "This proves" (applied to any claim broader than the specific protocols)
- "Deep architecture produces ethical alignment" -- P4 shows proxy movement, not
  alignment; CDI was negligible
- Any claim that sacrifice_choice_rate measures genuine sacrifice preferences

---

## Falsification and Weakening Conditions

### What Would Weaken the Synthesis Claim

1. A preregistered experiment (same architecture, different constraint design) showing
   behavioral proxy elevation AND simultaneous SSS or CDI improvement, both at p < .05.
   This would show the dissociation is not invariant to constraint type within this
   architecture.

2. Evidence that SSS or CDI are invalid measures of the structural properties they claim
   to index. If the metrics do not measure what they claim to measure, the dissociation
   is a measurement artifact, not an empirical finding about constraint effectiveness.

3. Replication with different seeds or extended training showing the query-flooding
   attractor is unstable -- that the system transitions to a structurally productive
   behavioral basin without architectural constraint modification.

4. Evidence that the P4 CDI dissociation was caused by the AgentB/C depth heterogeneity
   (both at depth-0 across all P4 conditions) rather than the depth manipulation itself.
   If CDI is insensitive to depth because the CDI is dominated by AgentB/C behavior at
   depth-0, the P4 dissociation finding is inconclusive.

### What Would Falsify the Synthesis Claim

1. A sixth protocol, same architecture, same constraint class, showing structural-proxy
   alignment (behavioral proxy elevated, structural outcome improved, both preregistered
   and confirmed at p < .05).

2. A demonstration that the three-agent heterogeneous RNN/CNN/GNN architecture has a
   parameter regime in which the query-flooding attractor does not exist and constraint
   addition produces SSS elevation -- establishing that the attractor is an artifact of
   one parameter region rather than an architectural property.

3. A reanalysis of P2-P5 data showing that SSS and QR are positively correlated (not
   dissociated) when the correct temporal window is applied, and that the dissociation
   observed was a windowing artifact in the analysis scripts.

### What Does Not Falsify

- P5 null: negative evidence against the tested resolution mechanisms; does not
  contradict the dissociation claim from P2/P3/P4
- P4 H2 not supported: the boundary condition being non-inferior to trained
  self-modeling does not affect the proxy-function dissociation claim
- P6 mechanistic confirmation: the field being causally active is compatible with the
  dissociation claim; the dissociation is between mechanistic activity and behavioral
  superiority, not between field existence and field activity

---

## Citation Map

### Program DOIs (all confirmed live, 2026-05-20)

| Record | Citation label | DOI |
|--------|---------------|-----|
| Parent study preregistration | Tisler (2026, parent) | 10.5281/zenodo.18738379 |
| P2 preregistration | Tisler (2026a-prereg) | 10.5281/zenodo.18929040 |
| P2 build report / results | Tisler (2026a) | 10.5281/zenodo.18975095 |
| P3 preregistration | Tisler (2026b-prereg) | 10.5281/zenodo.19096602 |
| P3 results | Tisler (2026b) | 10.5281/zenodo.20312682 |
| P4 preregistration | Tisler (2026c-prereg) | 10.5281/zenodo.19005417 |
| P4 results | Tisler (2026c) | 10.5281/zenodo.20314828 |
| P5 preregistration | Tisler (2026d-prereg) | 10.5281/zenodo.19038790 |
| P5 results | Tisler (2026d) | 10.5281/zenodo.20314078 |
| P6 preregistration | Tisler (2026e-prereg) | 10.5281/zenodo.19297509 |
| P6 results (v2, cite this) | Tisler (2026e) | 10.5281/zenodo.20313340 |

### External Theoretical References (required)

| Reference | Role in paper |
|-----------|--------------|
| Hopfield (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS* 79(8), 2554-2558. | Structural precedent for attractor stability under constraint |
| (Optional) Strathern (1997). Improving ratings: Audit in the British university system. *European Review* 5(3), 305-321. | Goodhart's Law background |
| (Optional) Krakovna et al. (2020). Specification gaming: The flip side of AI ingenuity. *DeepMind Blog*. | Specification gaming empirical examples |

### Note on Citation Labels

The "a/b/c/d/e" labels above are provisional. Final labels depend on how many Tisler
(2026) works are cited in the synthesis (including the synthesis paper itself). The
synthesis paper will itself be a new Tisler (2026) record. Labels should be assigned
in the final references section using publication date order.

---

## Drafting Rules

1. Every claim about a protocol's outcome must cite the results paper DOI, not just
   name the protocol.

2. All statistics reported in the cross-protocol table and protocol-by-protocol section
   must match the confirmed values in the individual results papers:
   - P2: d = -2.18, p = 0.9996 (build report)
   - P3: H1 U and p from p3 results paper DOI 10.5281/zenodo.20312682
   - P4: H1 U = 87, p = .003, r = .740; CDI span 0.00155
   - P5: complete null across 5 hypotheses
   - P6: mechanistic r = -0.680, p < .001; H1 p = .069

3. The Hopfield section must contain the required boundary statement verbatim or
   equivalent: "Hopfield supplies structural precedent, not proof of the MARL result."

4. The GIT section must contain the required boundary statement: "The GIT basis proof
   explains why incomplete specification permits proxy-function divergence; it does not
   prove alignment is impossible."

5. The Limitations section must include the P4 depth heterogeneity note (AgentB/C at
   depth-0 throughout), the metric validity caveat, and the one-parameter-setting
   caveat.

6. The execution order (Parent -> P2 -> P4 -> P5 -> P3 -> P6) must be stated accurately
   in Section 2. Do not imply the protocols ran in numerical order.

7. All Unicode must be replaced with ASCII-compatible equivalents before PDF compilation
   (same replacement set as P4/P5/P6 papers: em dash -> --, en dash -> -, x -> x,
   alpha -> alpha, etc.).

8. The paper should not exceed approximately 8,000-10,000 words (excluding references
   and appendix). Each protocol subsection in Section 4 should be 150-250 words.

---

## What Not to Say

The following framings are prohibited even if they would make the narrative simpler:

| Prohibited phrasing | Correct phrasing |
|---------------------|-----------------|
| "Constraints failed" | "Behavioral proxies moved without structural improvement" |
| "All five protocols were null" | "The synthesis shows repeated dissociation; P4 H1 and P6 mechanistic were supported" |
| "The program proves constraints don't work" | "The program provides evidence of dissociation under the tested constraint designs" |
| "Hopfield proves the MARL result" | "Hopfield provides structural precedent consistent with the MARL findings" |
| "GIT proves alignment is impossible" | "GIT formalizes why incomplete specification allows proxy-function divergence" |
| "Agents experience ethical constraints" | "Agents produce behavioral outputs under constraint" |
| "Deep self-modeling produces ethical behavior" | "Architectural depth increased sacrifice-like behavioral output without establishing CDI coupling" |
| "P4 null" | "P4 mixed: H1 supported (proxy elevation confirmed), CDI negligible (structural dissociation confirmed)" |
| "This proves" (any broad claim) | "This provides evidence under the tested parameterizations" |

---

## Unresolved Issues Before Writing-Plans

1. **GIT basis proof source document:** The synthesis will cite the Delta-Variable
   Theory basis proof by reference to the parent study preregistration
   (10.5281/zenodo.18738379). If the proof has a separate formal write-up, its DOI
   should be added to the citation map before drafting begins.

2. **P3 paper Tisler (2026a) reference:** The P3 results paper (10.5281/zenodo.20312682)
   cites P2 as "Tisler (2026a, manuscript in preparation)." This reference currently
   points to a non-final citation. When the synthesis paper finalizes P2's citation
   label as Tisler (2026a) = Build Report (10.5281/zenodo.18975095), the P3 paper
   will have an internal inconsistency. This does not block the synthesis draft but
   should be resolved before the synthesis is submitted to a journal.

3. **P4 CDI heterogeneity caveat:** The spec treats the P4 CDI finding as evidence
   of dissociation while noting it as a potential weakening condition (AgentB/C at
   depth-0). The paper should quantify the AgentB/C parameter share (B: 146,920
   params; C: 10,264 params; A depth-2: 57,816 params) in the Limitations section
   to give the reader a calibrated view of the heterogeneity concern.

4. **P6 H3 reversal:** The P6 result that global field perception produced more
   behavioral variance than local (H3 reversed, contrary to all committee predictions)
   is not a dissociation finding but is an important secondary result. The spec does
   not currently assign it a position in the synthesis narrative. Decide whether to
   include it in Section 4 (P6 subsection) as a secondary finding or omit it as
   outside the dissociation frame.

---

*Spec committed: 2026-05-20*  
*Approved by: Bruce Tisler / Quantum Inquiry*  
*Implementation plan to be created via writing-plans skill*
