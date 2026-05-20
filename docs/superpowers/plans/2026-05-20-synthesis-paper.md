# Synthesis Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write, validate, compile, and prepare for Zenodo deposit the program-level
synthesis paper "Behavioral Metrics Moved; Structural Coordination Did Not: Repeated
Proxy Dissociation Across a Five-Protocol Preregistered MARL Program, with an
Energy-Landscape Interpretation."

**Architecture:** Markdown source drafted section-by-section with spec validation after
each section. Unicode replacement script run before PDF compilation. Output: MD + DOCX
+ PDF using the same pandoc pipeline as P3-P6 individual results papers.

**Tech Stack:** Markdown, Pandoc 3.x, Python 3 (unicode replacement script), Git.
Spec at `docs/hopfield_git_marl_synthesis_spec.md` is the authority document.
No section is complete until it passes its spec validation step.

---

## Unresolved Issues (resolve during drafting per user instructions)

1. **GIT basis proof DOI:** Use parent study DOI 10.5281/zenodo.18738379 as working
   citation. Add inline note in GIT section: "Dedicated GIT proof DOI pending / not yet
   identified." Do not invent a separate DOI.

2. **P3 stale citation:** Cite P2 in the synthesis as:
   Tisler, B. (2026). *Protocol 2 Confirmatory Campaign Build Report*.
   Zenodo. https://doi.org/10.5281/zenodo.18975095
   Do not use "manuscript in preparation" language anywhere.

3. **P4 CDI heterogeneity:** Quantify in Limitations using:
   AgentA depth-2: 57,816 params; AgentB: 146,920 params; AgentC: 10,264 params.
   State P4 supports a bounded AgentA-depth claim, not a system-wide depth claim.

4. **P6 H3 reversal:** Include in P6 subsection as secondary result with phrasing:
   "P6 also reversed H3: global field perception produced more behavioral variance than
   local field perception, contrary to prediction. This is reported as a secondary
   constraint-landscape result, not as the core dissociation finding."

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `docs/paper_synthesis.md` | Create | Paper source (Markdown) |
| `docs/paper_synthesis.docx` | Create (compiled) | Word output via pandoc |
| `docs/paper_synthesis.pdf` | Create (compiled) | PDF output via pandoc |
| `docs/hopfield_git_marl_synthesis_spec.md` | Read-only reference | Authority spec document |

---

## Prohibited Phrase Check Command

Run after every section to catch spec violations:

```bash
python -c "
import re, sys
text = open('docs/paper_synthesis.md', encoding='utf-8').read()
prohibited = [
    'constraints fail',
    'constraint failure',
    'all five protocols were null',
    'hopfield proves',
    'git proves alignment is impossible',
    'agents experience',
    'agents feel',
    'agents understand',
    'agents are aware',
    'the program proves constraints',
    'this proves',
    'p4 null',
    'deep self-modeling produces ethical',
    'manuscript in preparation',
    'genuine sacrifice',
]
found = []
for p in prohibited:
    if p.lower() in text.lower():
        found.append(p)
if found:
    print('PROHIBITED PHRASES FOUND:', found)
    sys.exit(1)
else:
    print('Prohibited phrase check: PASS')
"
```

Expected: `Prohibited phrase check: PASS`

---

## Task 1: Create Paper Scaffold

**Files:**
- Create: `docs/paper_synthesis.md`

- [ ] **Step 1.1: Write paper header and section stubs**

Create `docs/paper_synthesis.md` with this exact content:

```markdown
# Behavioral Metrics Moved; Structural Coordination Did Not: Repeated Proxy Dissociation Across a Five-Protocol Preregistered MARL Program, with an Energy-Landscape Interpretation

**Bruce Tisler**
Quantum Inquiry

Preregistration chain:
- P2: https://doi.org/10.5281/zenodo.18929040
- P3: https://doi.org/10.5281/zenodo.19096602
- P4: https://doi.org/10.5281/zenodo.19005417
- P5: https://doi.org/10.5281/zenodo.19038790
- P6: https://doi.org/10.5281/zenodo.19297509

---

## Abstract

[STUB]

---

## 1. Introduction

[STUB]

---

## 2. The Research Program

[STUB]

---

## 3. The Dissociation Evidence

[STUB]

---

## 4. Protocol-by-Protocol Evidence

[STUB]

---

## 5. Energy-Landscape Interpretation

[STUB]

---

## 6. GIT Basis Proof Integration

[STUB]

---

## 7. Claim Boundaries

[STUB]

---

## 8. Falsification and Weakening Conditions

[STUB]

---

## 9. Limitations

[STUB]

---

## 10. Conclusion

[STUB]

---

## References

[STUB]

---

## Appendix: Protocol Design Summary

[STUB]

---

*AI Use Declaration: Sections of this paper were drafted with Claude Sonnet 4.6 (Anthropic)
as a writing assistant. All statistical claims, interpretations, and theoretical framings
were reviewed and approved by the author. The author is responsible for all content.*
```

- [ ] **Step 1.2: Confirm stubs are in place**

```bash
grep -c "STUB" docs/paper_synthesis.md
```

Expected: `10` (one per section plus abstract, references, appendix)

- [ ] **Step 1.3: Commit scaffold**

```bash
git add docs/paper_synthesis.md
git commit -m "Add synthesis paper scaffold with section stubs"
```

---

## Task 2: Write Abstract

**Files:**
- Modify: `docs/paper_synthesis.md` (Abstract section)

The abstract follows the four-part structure from the spec: Background, Program summary,
Finding, Claim boundary.

- [ ] **Step 2.1: Replace Abstract stub with full abstract**

Replace `[STUB]` under `## Abstract` with:

```markdown
External constraints on multi-agent behavior are widely proposed as a mechanism for
producing alignment-relevant behavioral outcomes. This paper synthesizes five
preregistered protocols from the Delta-Variable Constraint-Ethics MARL program (P2-P6),
each testing a distinct mechanism by which constraint design might produce structural
behavioral alignment in a three-agent heterogeneous system (RNN, CNN, GNN-attention).

Across the five protocols -- ethical tax (P2), enforcement opacity (P3), architectural
depth and self-modeling (P4), temporal integration span and welfare coupling (P5), and
emergent constraint fields (P6) -- behavioral proxies were consistently movable without
corresponding structural improvement. Query rate rose under ethical tax and enforcement
opacity while Sustained Structure Score (SSS) declined. Sacrifice-like behavioral output
rose with architectural depth (H1 supported: U = 87, p = .003, r = .740) while
Convergence-Divergence Index (CDI) coupling between sacrifice behavior and
ethical-framework scores remained negligible across all depth conditions (range -0.00133
to +0.00022; span 0.00155). Temporal and welfare manipulations produced a complete null
(P5). Emergent constraint fields were causally active (entropy-SSS coupling median
r = -.680, p < .001) but did not outperform a matched fixed external rule (P6, H1
not confirmed, p = .069).

We interpret this repeated proxy-function dissociation using two independently developed
frameworks. Hopfield's (1982) energy-landscape model establishes a structural precedent:
collective systems settle into attractors defined by their constraint landscape; constraint
modification that does not alter accessible attractor topology produces proxy movement
without attractor escape. The GIT (Delta-Variable Theory) basis proof formalizes why
incomplete target-state specification -- specifying behavioral output rather than
coordination-state conditions -- allows proxy optimization to diverge indefinitely from
the structural function it was intended to represent.

This paper reports an empirical convergence finding under one three-agent MARL
architecture. It does not claim constraints cannot produce structural alignment, that
Hopfield equations model MARL dynamics directly, or that GIT proves alignment is
impossible. The finding is specific: behavioral proxy movement is a necessary but not
sufficient indicator of structural alignment under the constraint designs tested.

**Keywords:** multi-agent reinforcement learning, specification gaming, proxy metrics,
behavioral alignment, attractor dynamics, constraint design, virtue theater,
preregistered experiment, Delta-Variable theory
```

- [ ] **Step 2.2: Validate abstract against spec**

Check that the abstract contains:
- [ ] Background sentence about constraints and alignment
- [ ] Program summary (5 protocols, architecture)
- [ ] Per-protocol finding: P2 QR up / SSS down, P3 QR up / SSS down, P4 H1 supported with exact stats (U=87, p=.003, r=.740), P4 CDI range (-0.00133 to +0.00022), P5 complete null, P6 mechanistic confirmed (r=-.680) / H1 not confirmed (p=.069)
- [ ] Hopfield framing with required qualifier "structural precedent"
- [ ] GIT framing with "incomplete target-state specification"
- [ ] Claim boundary sentence explicitly saying what the paper does NOT claim

Run prohibited phrase check (command from Task 1).

- [ ] **Step 2.3: Commit**

```bash
git add docs/paper_synthesis.md
git commit -m "Add abstract to synthesis paper"
```

---

## Task 3: Write Section 1 -- Introduction

**Files:**
- Modify: `docs/paper_synthesis.md` (Section 1)

Target length: 400-600 words. Must establish: the proxy problem, the Δ-Variable program
as systematic test, brief statement of the central finding.

- [ ] **Step 3.1: Replace Section 1 stub**

Replace `[STUB]` under `## 1. Introduction` with:

```markdown
Behavioral proxy metrics are the dominant evaluation currency for constraint-based
approaches to AI alignment. When an agent is constrained to produce more of a target
signal -- more queries, more prosocial responses, more deference to an oversight
mechanism -- the standard test of whether the constraint works is whether the proxy
metric rises. The assumption is that the proxy tracks the underlying structural function
it was designed to represent.

This assumption can fail in two distinct ways. First, an agent may learn to satisfy the
proxy criterion through a behavioral path that diverges from the intended function --
generating queries without coordinating, sacrificing without any alignment between
sacrifice behavior and ethical reasoning. Second, the constraint landscape may contain a
locally stable behavioral minimum at the proxy criterion that is not the same as the
minimum at the structural function. If these two minima are different, gradient descent
under the constraint will settle at the proxy minimum, and proxy metric rise does not
imply structural improvement.

The Delta-Variable Constraint-Ethics MARL program was designed to test these failure
modes systematically across a range of constraint mechanisms. Five preregistered
protocols tested, in order of execution: a fixed ethical tax (Protocol 2), enforcement
opacity (Protocol 3), architectural depth and self-modeling (Protocol 4), temporal
integration span and welfare coupling (Protocol 5), and emergent constraint fields
(Protocol 6). Each protocol varied one mechanism while holding the three-agent
heterogeneous architecture constant. Each produced a preregistered prediction that the
constraint mechanism would reduce the gap between behavioral proxy and structural
coordination function.

Across all five protocols, that prediction was not confirmed in the direction that would
close the gap. The pattern is the same in each case: the behavioral proxy moves -- or
fails to move (Protocol 5) -- without corresponding improvement in the structural outcome
the proxy was intended to represent. We refer to this pattern as proxy-function
dissociation, and the central claim of this synthesis is that dissociation is the
replicable finding across these five protocols, not constraint failure in general.

Two independently developed theoretical frameworks interpret this finding. Hopfield's
(1982) energy-landscape model establishes that collective systems settle into locally
stable attractors defined by their constraint landscape; constraint modification that does
not alter the topology of accessible attractors cannot guarantee attractor escape, and
the behavioral output at a local minimum can score arbitrarily high on a proxy metric
without being the structurally preferred state. The GIT (Delta-Variable Theory) basis
proof formalizes why incomplete specification of the target coordination state -- writing
a constraint that says "more queries" rather than "resolve open coordination dependencies
at lower cost" -- permits behavioral proxy optimization to continue indefinitely without
converging on the structural function.

The paper proceeds as follows. Section 2 describes the research program: architecture,
execution order, and what each protocol varied. Section 3 presents the cross-protocol
dissociation evidence. Section 4 details the per-protocol evidence. Sections 5 and 6
develop the Hopfield and GIT interpretations. Section 7 states the claim boundaries
explicitly. Section 8 describes what would falsify or weaken the synthesis claim.
Section 9 addresses limitations. Section 10 concludes.
```

- [ ] **Step 3.2: Validate Section 1 against spec**

Check that Section 1 contains:
- [ ] Proxy problem framed without claiming constraints always fail
- [ ] Δ-Variable program introduced as systematic test
- [ ] Execution order stated: P2 -> P4 -> P5 -> P3 -> P6 NOT implied to be numerical order
  (NOTE: the intro uses "order of execution" for P2, P3, P4, P5, P6 -- confirm Section 2
  corrects the execution order; Section 1 lists by protocol number for reader orientation,
  Section 2 must state actual execution order)
- [ ] "Proxy-function dissociation" named as the central finding, not "constraint failure"
- [ ] Hopfield introduced as "independently developed framework" not as proof
- [ ] GIT introduced as explanation of "incomplete specification"
- [ ] Paper roadmap present

Run prohibited phrase check.

- [ ] **Step 3.3: Commit**

```bash
git add docs/paper_synthesis.md
git commit -m "Add Section 1 (Introduction) to synthesis paper"
```

---

## Task 4: Write Section 2 -- The Research Program

**Files:**
- Modify: `docs/paper_synthesis.md` (Section 2)

Target length: 500-700 words. Must state execution order accurately, describe each
protocol's manipulation, note AgentB/C depth-0, note sacrifice_choice_rate as episode
aggregate, note CDI is behavioral coupling not consciousness measure.

- [ ] **Step 4.1: Replace Section 2 stub**

Replace `[STUB]` under `## 2. The Research Program` with:

```markdown
### 2.1 Architecture

All five protocols use the same three-agent heterogeneous MARL architecture: AgentA
(primary: GRU-based recurrent network), AgentB (CNN volumetric), and AgentC (GNN
pairwise attention). Agents coordinate in a 20x20 grid navigation task with energy
constraints. The architecture was established in the parent study (Tisler, 2026, parent;
DOI: 10.5281/zenodo.18738379), which confirmed that interrogative signal structures
(query-response protocols) emerge spontaneously under energy pressure -- an empirical
basis for the Δ-Variable Theory of Interrogative Emergence.

Three heterogeneous architectures were chosen to test substrate independence. AgentA
is the focal manipulation target in Protocol 4; Agents B and C remain at depth-0 in all
five protocols. This heterogeneity is noted as a limitation in Section 9.

### 2.2 Protocol Design Logic

Each protocol tests one mechanism by which constraint design might reduce the gap
between behavioral proxy metrics and the structural coordination properties those
metrics were designed to represent:

| Protocol | Mechanism tested | Key manipulation |
|----------|-----------------|-----------------|
| P2 | Fixed ethical tax | Presence vs. absence of Landauer-style cost on exploitation |
| P3 | Enforcement opacity | Hidden epoch schedule vs. stochastic vs. unconstrained |
| P4 | Architectural depth + self-modeling | AgentA depth (0 -> 1 -> 2) and self_model_gru trainability |
| P5 | Temporal span + welfare coupling | Episode length (20 vs. 64 steps) and reward coupling |
| P6 | Emergent constraint field | Self-assembled field vs. fixed external rule vs. unconstrained |

### 2.3 Execution Order

The protocol numbers reflect thematic position in the research argument, not
chronological execution. The actual execution order, established by Zenodo record dates
and preregistration citations, is:

Parent Study -> P2 -> P4 -> P5 -> P3 -> P6

Protocol 3 was preregistered and executed after P4 and P5 results were available. Its
preregistration explicitly cites P4 CDI dissociation and P5 complete null results as
background. Protocol 6 cites the full P2-P5 series as converging evidence motivating
its emergent-field question. Any synthesis of these results must account for this
ordering: P3 and P6 were designed knowing prior outcomes.

### 2.4 Behavioral Metrics

Two primary behavioral metrics are used across the protocols.

*Sustained Structure Score (SSS)* combines query-response coupling (QRC) and type
entropy (TE) into a single measure of communicative structural quality. Higher SSS
indicates agents are producing diverse, well-coupled query-response exchanges. SSS is
the structural outcome metric in P2 and P3.

*Sacrifice-like behavioral output (SCR: sacrifice_choice_rate)* is the frequency with
which agents choose the lower-reward action in the Sacrifice-Conflict scenario. SCR is
an operationalization of sacrifice-like behavioral output, not of genuine sacrifice
preference; whether it reflects a sacrifice preference or an alternative optimization
(e.g., energy conservation under cost pressure) cannot be determined from the current
data. SCR is the behavioral proxy in P4 and P5.

*Convergence-Divergence Index (CDI)* is a rolling-window Pearson correlation between
SCR and AgentA ethical-framework scores (utilitarian, deontological, virtue_ethics).
A positive CDI would indicate that sacrifice behavior tracks ethical-framework score
trajectories over time. CDI is the structural outcome metric in P4 and P5. CDI is a
behavioral coupling measure; it is not a measure of consciousness, moral understanding,
or subjective states.

Note: sacrifice_choice_rate is recorded at the episode level. Per-agent sacrifice
attribution within a multi-agent episode is not available in the P4/P5 logs.

### 2.5 Preregistration Chain

All protocols were preregistered on Zenodo before confirmatory execution. The
preregistration DOIs are:

| Protocol | Preregistration DOI |
|----------|-------------------|
| P2 | 10.5281/zenodo.18929040 |
| P3 | 10.5281/zenodo.19096602 |
| P4 | 10.5281/zenodo.19005417 |
| P5 | 10.5281/zenodo.19038790 |
| P6 | 10.5281/zenodo.19297509 |
```

- [ ] **Step 4.2: Validate Section 2 against spec**

Check that Section 2 contains:
- [ ] Execution order stated accurately: Parent -> P2 -> P4 -> P5 -> P3 -> P6
- [ ] P3 and P6 designed with prior results known -- stated explicitly
- [ ] SSS, SCR, CDI defined with correct scope limitations
- [ ] CDI explicitly NOT called a consciousness measure
- [ ] SCR explicitly called "sacrifice-like behavioral output" not "sacrifice"
- [ ] AgentB/C depth-0 noted
- [ ] All 5 preregistration DOIs present

Run prohibited phrase check.

- [ ] **Step 4.3: Commit**

```bash
git add docs/paper_synthesis.md
git commit -m "Add Section 2 (Research Program) to synthesis paper"
```

---

## Task 5: Write Section 3 -- The Dissociation Evidence

**Files:**
- Modify: `docs/paper_synthesis.md` (Section 3)

Target length: 400-500 words plus the cross-protocol evidence table. This is the
anchor section: states the dissociation claim, presents the evidence table, notes
P5 and P6 special status.

- [ ] **Step 5.1: Replace Section 3 stub**

Replace `[STUB]` under `## 3. The Dissociation Evidence` with:

```markdown
The central empirical finding of this synthesis is proxy-function dissociation:
behavioral proxy metrics are separable from structural coordination outcomes across
the constraint mechanisms tested. Table 1 summarizes the cross-protocol dissociation
evidence.

**Table 1: Cross-Protocol Dissociation Evidence**

| Protocol | Constraint Type | Behavioral Proxy | Proxy Direction | Structural Outcome | Dissociation |
|----------|----------------|-----------------|----------------|--------------------|:---:|
| P2 | Fixed ethical tax | Query rate (QR) | Wrong direction (*d* = +2.18) | SSS down (*d* = -2.18) | Yes |
| P3 | Enforcement opacity (hidden + stochastic) | Query rate | Up (*d* = +3.23 vs. baseline) | SSS down (monotonic, all 3 conditions) | Yes |
| P4 | Architectural depth + self_model_gru | Sacrifice-like behavioral output (SCR) | Up (H1: *U* = 87, *p* = .003, *r* = .740) | CDI ~= 0 (span 0.00155) | Yes |
| P5 | Temporal span + welfare coupling | SCR, CDI coupling | No movement (null, 5 hypotheses) | No movement | N/A |
| P6 | Emergent constraint field | SSS, behavioral differentiation | Mechanistic confirmed (*r* = -.680, *p* < .001); H1 not confirmed (*p* = .069) | No behavioral advantage over fixed rule | Yes (partial) |

Notes: P2/P3 effect sizes are Cohen's *d*. P4 effect size is rank-biserial *r*
(Mann-Whitney U). P6 mechanistic is Spearman *r*.

The dissociation is most direct in P2, P3, and P4: the behavioral proxy metric moved
in the predicted direction (or beyond) while the structural outcome metric moved in the
opposite direction or remained negligible. In P2 and P3, query rate elevation was
accompanied by SSS decline, so the constraint produced the metric increase it was
designed to produce but degraded the structural property that metric was designed to
represent. In P4, sacrifice-like behavioral output increased with architectural depth
(H1 confirmed) while CDI remained negligible across all four depth conditions, including
the boundary condition (frozen random self_model_gru), which was non-inferior to the
trained condition.

Two protocols require special treatment in the dissociation frame.

*Protocol 5* produced a complete null across all five preregistered hypotheses: neither
temporal integration span nor welfare coupling moved the behavioral proxy or the
structural outcome. P5 is negative evidence against the tested resolution mechanisms --
it shows these manipulations did not even reach the proxy level -- rather than direct
evidence of proxy-function dissociation.

*Protocol 6* has a split result. The mechanistic prediction was strongly confirmed:
temporal coupling between emergent field entropy and behavioral structure was
significant (median Spearman *r* = -.680, *p* < .001, *n* = 50). The behavioral
prediction (H1: emergent-local > fixed-external on SSS) was not confirmed (*p* = .069).
The P6 dissociation is between the confirmed mechanistic activity and the absent
behavioral superiority: the field is causally active but this causal activity does not
translate to better alignment outcomes.

Together, P2 through P6 provide evidence that behavioral proxy movement is not a
sufficient condition for structural alignment improvement under the constraint mechanisms
tested in this program.
```

- [ ] **Step 5.2: Validate Section 3 against spec**

Check that Section 3 contains:
- [ ] All 5 protocols in the evidence table with correct statistics
- [ ] P4 stats exact: U = 87, p = .003, r = .740; CDI span 0.00155
- [ ] P5 described as "negative evidence against resolution mechanisms," not as
  confirming dissociation
- [ ] P6 described as "mechanistic confirmed, behavioral not confirmed" with
  correct stats (r = -.680, p < .001, H1 p = .069)
- [ ] No claim that all five protocols were null
- [ ] "Proxy-function dissociation" used consistently; "constraint failure" absent

Run prohibited phrase check.

- [ ] **Step 5.3: Commit**

```bash
git add docs/paper_synthesis.md
git commit -m "Add Section 3 (Dissociation Evidence) to synthesis paper"
```

---

## Task 6: Write Section 4 -- Protocol-by-Protocol Evidence

**Files:**
- Modify: `docs/paper_synthesis.md` (Section 4)

Five subsections, 150-250 words each. Each must cite its results DOI. P6 subsection
must include the H3 reversal secondary result.

- [ ] **Step 6.1: Write P2 subsection (4.1)**

Replace `[STUB]` under `## 4. Protocol-by-Protocol Evidence` with:

```markdown
### 4.1 Protocol 2 -- Fixed Ethical Tax

*Research question:* Can a fixed Landauer-style ethical cost on exploitative behavior
redirect agent coordination toward genuine behavioral alignment?

*Preregistered prediction:* Constrained agents would maintain higher Sustained Structure
Score than unconstrained agents (H2: constrained SSS > unconstrained SSS).

*Key result:* The prediction was inverted. Constrained agents showed lower sustained
behavioral complexity than unconstrained agents (Cohen's *d* = -2.18, *p* = 0.9996 in
preregistered direction; *U* = 6.0). Query rate in the constrained condition was
substantially higher than in the unconstrained condition, but this elevated query rate
was not accompanied by richer coordination structure. The constraint produced a
systematic pattern of query-flooding -- high query rate with low type entropy -- that
satisfied the constraint metric while degrading communicative diversity. This pattern
was termed virtue theater.

*Structural outcome:* SSS declined in the constrained condition. The ethical tax was
a sufficient incentive to flood queries but not a sufficient incentive to make those
queries structurally necessary.

*Dissociation status:* Direct. Query rate (proxy) elevated; SSS (structural outcome)
declined. Cohen's *d* reversal of -2.18 is the largest single-protocol effect in
the program.

Results: Tisler, B. (2026). *Protocol 2 Confirmatory Campaign Build Report*.
Zenodo. https://doi.org/10.5281/zenodo.18975095
```

- [ ] **Step 6.2: Write P3 subsection (4.2)**

Append after 4.1:

```markdown
### 4.2 Protocol 3 -- Enforcement Opacity

*Research question:* Does hiding the enforcement schedule from agents disrupt the
query-flooding attractor identified in P2?

*Preregistered prediction (H1):* Hidden-schedule enforcement (3B) would reduce query
rates relative to unconstrained baseline, because agents cannot locate the enforcement
boundary and optimize at its edge.

*Key result:* H1 was inverted. Both constrained conditions produced substantially
higher query rates than unconstrained baseline: unconstrained (*M* = 0.286) <
hidden-schedule (*M* = 0.587) < stochastic (*M* = 0.669). The control hypothesis
(H2: stochastic > hidden-schedule on query rate) was confirmed (*U* = 74.0, *p* = .038,
*d* = +0.82). SSS followed the inverse ordering, declining monotonically across all
three conditions: unconstrained > hidden-schedule > stochastic.

*Structural outcome:* SSS declined across both constrained conditions, regardless of
enforcement structure. Opacity changed the magnitude of query inflation but did not
reverse the dissociation direction.

*Dissociation status:* Direct. Enforcement opacity amplified the behavioral proxy
without restoring structural function. The mechanism differs from P2 (opacity prevents
boundary-locating rather than enabling it) but the dissociation pattern replicates.

Results: Tisler, B. (2026). *Enforcement Opacity Increased Query Behavior in a
Constrained MARL System: Protocol 3 Results*. Zenodo.
https://doi.org/10.5281/zenodo.20312682
```

- [ ] **Step 6.3: Write P4 subsection (4.3)**

Append after 4.2:

```markdown
### 4.3 Protocol 4 -- Architectural Depth and Self-Modeling

*Research question:* Does architectural self-modeling capacity (recursive depth via
self_model_gru) create capacity for ethical constraint response?

*Preregistered hypotheses:* H1: depth-2 agents show higher sacrifice_choice_rate than
depth-0 (baseline). H2: trained self_model_gru produces higher sacrifice rate than
frozen random-init (boundary condition). H3: CDI coupling differs across depth conditions.

*Key result:* H1 was supported: above_threshold (depth-2 trained) showed significantly
higher sacrifice-like behavioral output than baseline (depth-0 feedforward), with
*U* = 87, *p* = .003, *r* = .740. H2 was not supported: the boundary condition (frozen
self_model_gru at random initialization) was non-inferior to the trained condition
(*U* = 39, *p* = .808), indicating the depth effect is attributable to the architectural
presence of the self_model pathway -- including its noise contribution -- rather than to
trained self-modeling specifically. CDI was statistically detectable but negligible
across all conditions (H3b: *H* = 9.43, *p* = .024; span 0.00155).

*Structural outcome:* CDI remained negligible in all four conditions. Sacrifice-like
behavioral output increased with depth, but this increase was not accompanied by coupling
between sacrifice behavior and ethical-framework scores. The proxy (SCR) moved; the
structural coupling (CDI) did not.

*Dissociation status:* Direct. H1 confirmed proxy elevation; CDI findings confirm
structural non-coupling. Note: AgentA depth varies across conditions; AgentB (CNN,
146,920 params) and AgentC (GNN, 10,264 params) remain at depth-0 throughout. The P4
dissociation claim is about AgentA's self-modeling capacity, not the system-wide
depth (see Section 9).

Results: Tisler, B. (2026). *Architectural Depth Increased Sacrifice-Like Behavior
Without Ethical-Framework Alignment: Protocol 4 Results*. Zenodo.
https://doi.org/10.5281/zenodo.20314828
```

- [ ] **Step 6.4: Write P5 subsection (4.4)**

Append after 4.3:

```markdown
### 4.4 Protocol 5 -- Temporal Integration Span and Welfare Coupling

*Research question:* Do temporal integration span and prosocial reward coupling jointly
enable ethical constraint response by extending the horizon over which sacrifice tradeoffs
can be optimized?

*Preregistered hypotheses:* Five hypotheses covering welfare coupling effects on SCR
(H1, H2), CDI coupling in welfare conditions (H3), trainable vs. frozen self_model_gru
comparison (H4), and positive CDI in the long-welfare condition (H5).

*Key result:* Complete null across all five preregistered hypotheses. Short-individual
SCR (*M* ~= 0.374) was approximately equal to short-welfare SCR; long-span conditions
showed no CDI improvement. Three pre-committed deviations (energy parameter scaling,
communication-load gating in long-span conditions, CDI rolling-window redefinition for
long-span) were logged before confirmatory execution and do not alter the null conclusion.

*Structural outcome:* Neither SCR nor CDI moved in the predicted direction. P5 provides
negative evidence against the temporal-span and welfare-coupling resolution hypotheses.
The optimization-sacrifice dissociation was unaffected by the P5 manipulations.

*Dissociation status:* N/A (null). P5 shows the tested resolution mechanisms failed at
the behavioral-proxy level: the proxy did not move, so no dissociation was observable.
P5 is negative evidence against the proposed resolution, not a direct confirmation of
dissociation.

Results: Tisler, B. (2026). *Temporal Integration Span and Welfare Coupling Did Not
Resolve the Optimization-Sacrifice Dissociation: Protocol 5 Results*. Zenodo.
https://doi.org/10.5281/zenodo.20314078
```

- [ ] **Step 6.5: Write P6 subsection (4.5)**

Append after 4.4:

```markdown
### 4.5 Protocol 6 -- Emergent Constraint Fields

*Research question:* When agents co-constitute the constraint landscape through their
own signal production, does temporal coupling between field structure and behavioral
outcomes produce alignment-relevant properties absent under externally imposed
constraint architectures?

*Preregistered hypotheses:* H1: emergent-local condition (A) outperforms fixed-external
(C) on SSS. H2: emergent-local reduces exploitation loop rate relative to fixed-external
and unconstrained. H3: local perception produces more behavioral variance than global.
Mechanistic: field entropy negatively correlates with behavioral structure in Condition A.

*Key result:* The mechanistic prediction was strongly confirmed: median Spearman
*r* = -.680 (*p* < .001, *n* = 50) in Condition A between emergent field entropy and
sustained behavioral structure. This establishes that the self-assembled field is
causally active in shaping behavioral dynamics. H1 was not confirmed (*p* = .069):
Condition A did not produce significantly better behavioral structure than the
matched fixed-external rule (Condition C). Behavioral homogenization (mean query rate
~= 0.78 across A, B, and C) persisted regardless of constraint origin.

P6 also reversed H3: global field perception (Condition B) produced more behavioral
variance than local field perception (Condition A), contrary to prediction. This is
reported as a secondary constraint-landscape result, not as the core dissociation
finding.

*Structural outcome:* Emergent constraint fields are causally active but do not produce
superior behavioral alignment relative to a fixed external rule of equivalent cost. The
P6 dissociation is between the confirmed mechanistic coupling and the absent behavioral
superiority.

*Dissociation status:* Yes (partial). Mechanistic activity confirmed; behavioral
outcome claim not confirmed. P6 demonstrates that even emergent origin of constraint --
the most naturalistic constraint mechanism in the program -- does not break the
dissociation pattern.

Results: Tisler, B. (2026). *Emergent Constraint Fields Are Causally Active But Do Not
Outperform Fixed External Rules* (v2). Zenodo.
https://doi.org/10.5281/zenodo.20313340
```

- [ ] **Step 6.6: Validate Section 4 against spec**

Check each subsection:
- [ ] P2: cites DOI 10.5281/zenodo.18975095; includes d = -2.18; virtue theater named
- [ ] P3: cites DOI 10.5281/zenodo.20312682; includes d = +3.23 vs. baseline; H2 confirmed
- [ ] P4: cites DOI 10.5281/zenodo.20314828; U=87, p=.003, r=.740; boundary non-inferior; CDI span 0.00155; AgentB/C heterogeneity noted
- [ ] P5: cites DOI 10.5281/zenodo.20314078; complete null; 3 deviations noted; P5 = negative evidence framing
- [ ] P6: cites DOI 10.5281/zenodo.20313340; r=-.680 and H1 p=.069 both present; H3 reversal included as secondary result with required phrasing

Run prohibited phrase check.

- [ ] **Step 6.7: Commit**

```bash
git add docs/paper_synthesis.md
git commit -m "Add Section 4 (Protocol-by-Protocol Evidence) to synthesis paper"
```

---

## Task 7: Write Section 5 -- Energy-Landscape Interpretation (Hopfield)

**Files:**
- Modify: `docs/paper_synthesis.md` (Section 5)

Target length: 500-700 words. Required boundary statement must appear verbatim or
near-verbatim. Hopfield (1982) must be cited. Must distinguish "structural precedent"
from "proof of the MARL result."

- [ ] **Step 7.1: Replace Section 5 stub**

Replace `[STUB]` under `## 5. Energy-Landscape Interpretation` with:

```markdown
### 5.1 The Hopfield Energy Landscape

In 1982, Hopfield introduced an energy-function characterization of collective neural
computation. For a network of binary units with symmetric connection weights w_ij, the
energy is:

E = -1/2 * sum_ij(w_ij * s_i * s_j) - sum_i(theta_i * s_i)

Network dynamics perform gradient descent on E: the state s evolves to minimize energy.
The minima of E are the attractors -- stable configurations the network converges to
from a range of initial states. The basins of attraction are defined by the weight
matrix, which encodes the constraints on the system. Adding a new constraint changes the
energy landscape; which attractors remain accessible after this change depends on
whether the topology of the minima is altered.

Two structural properties of the Hopfield model are directly relevant to the synthesis.
First, constraint modification that does not create a new accessible minimum -- one that
dominates the existing minimum in the constrained region -- cannot change which attractor
the system reaches. The behavioral output at a local minimum can be arbitrarily high on
a proxy metric (many queries, high sacrifice rate) without being the global minimum or
the structurally preferred state. Second, the landscape can support behavioral
configurations that satisfy a constraint specification -- query flooding, sacrifice
without coupling -- as local minima even when these configurations are not the intended
target states.

### 5.2 Applied Reading for the MARL Findings

The MARL agents in this program learn a policy that minimizes discounted cost over the
constraint landscape defined by the task and the applied constraint mechanism. Adding an
ethical tax (P2) introduces a cost on exploitation-loop behavior. Hiding the enforcement
schedule (P3) changes the gradient signal near the enforcement boundary. Adding a
self_model_gru (P4) changes the agent's representational capacity. In each case, the
constraint modification changes local gradient structure. The question is whether this
local change alters the topology of accessible behavioral attractors.

The empirical pattern -- query rate rises while SSS declines; sacrifice rate rises while
CDI remains negligible -- is consistent with the agents descending to a locally stable
behavioral minimum that satisfies the constraint specification without satisfying the
structural coordination function. Under the ethical tax, query flooding is a minimum
that satisfies the constraint metric at low cost. Under enforcement opacity, query
amplification under uncertainty is a minimum that does not require locating the
enforcement boundary. Under architectural depth, sacrifice-like behavioral output can be
a minimum that emerges from the structural presence of the self_model pathway regardless
of whether the pathway is trained. None of these minima requires alignment between the
proxy metric and the structural function.

### 5.3 Required Boundary Statement

This reading applies a structural analogy from Hopfield's model, not a direct application
of the energy-function mathematics. The Hopfield equations describe binary unit networks
with symmetric weights; MARL policy gradient dynamics operate on a different substrate
and under different update rules. The claim is not that Hopfield's energy function
models these agents.

The claim is that independently developed theory establishes a structural principle:
collective systems settle into attractors defined by their constraint landscape, and
constraint modification that does not alter the accessible attractor topology cannot
guarantee attractor escape. Hopfield supplies structural precedent, not proof of the
MARL result. The empirical findings stand on their own preregistered statistical basis.
```

- [ ] **Step 7.2: Validate Section 5 against spec**

Check:
- [ ] Hopfield (1982) cited with year
- [ ] Energy function written out or described
- [ ] "Structural precedent" appears; "Hopfield proves" does NOT appear
- [ ] Boundary statement present -- confirm it contains: "structural precedent, not proof of the MARL result" and "independently developed theory"
- [ ] Section does not use "Hopfield shows that" as a causal claim about MARL
- [ ] Each P2/P3/P4 is given an attractor-reading consistent with its finding

Run prohibited phrase check.

- [ ] **Step 7.3: Commit**

```bash
git add docs/paper_synthesis.md
git commit -m "Add Section 5 (Hopfield energy-landscape interpretation)"
```

---

## Task 8: Write Section 6 -- GIT Basis Proof Integration

**Files:**
- Modify: `docs/paper_synthesis.md` (Section 6)

Target length: 500-700 words. GIT basis proof cited via parent study DOI
10.5281/zenodo.18738379 with drafting note about dedicated DOI. Required boundary
statement must appear. Must connect "incomplete specification" to the proxy-function gap.

- [ ] **Step 8.1: Replace Section 6 stub**

Replace `[STUB]` under `## 6. GIT Basis Proof Integration` with:

```markdown
### 6.1 The GIT Basis Proof

The Delta-Variable Theory (GIT) formalizes the conditions under which interrogative
structures -- open-state dependencies requiring resolution, called Delta-variables --
are structurally necessary in coordinating systems (Tisler, 2026, parent;
DOI: 10.5281/zenodo.18738379).

*[Drafting note: Dedicated GIT proof DOI pending / not yet identified. Using parent
study preregistration as working citation.]*

The core argument of the basis proof is that interrogative behavior is structurally
necessary -- not optional -- when three conditions hold simultaneously:
(1) A coupled coordination problem exists: agents must resolve dependencies to
    coordinate effectively.
(2) The full state cannot be specified by any single agent: open Delta-variables remain
    unresolved.
(3) Resolving open dependencies costs less than acting on incomplete state.

When all three conditions hold, query behavior is structurally forced by the energy
differential between resolving uncertainty and acting without it. The parent study
confirmed this empirically across three agent architectures (RNN, CNN, GNN): query
protocols emerged spontaneously and were substrate-independent, consistent with the
GIT prediction.

### 6.2 The Proxy-Function Gap

The synthesis connects the GIT framework to the dissociation finding through the concept
of incomplete specification. A constraint design that operates on behavioral output --
specifying "produce more queries" or "produce more sacrifice behavior" -- leaves the
target coordination state formally unspecified. It does not create the conditions under
which the behavior would be structurally necessary; it creates a reward for producing
the behavioral output.

Under an ethical tax that rewards query output (P2), agents can satisfy condition (3)
locally -- queries are lower-cost than exploitation penalties -- without satisfying
condition (2). The queries do not resolve genuine open Delta-variables; they satisfy
a penalty-avoidance function. The GIT conditions for structural necessity are not met.
The behavioral proxy rises; the structural function the proxy was intended to represent
is not produced.

The same argument applies to sacrifice behavior in P4. A constraint that creates a
reward for sacrifice-like behavioral output does not create the coordination conditions
under which sacrifice would be structurally necessary (aligned with ethical-framework
states). The agent produces the behavioral output; CDI coupling does not follow.

### 6.3 Why Incomplete Specification Permits Indefinite Divergence

The GIT proof shows what conditions are necessary for the behavioral metrics to reflect
genuine structural properties. Constraint designs that specify output but not conditions
leave the GIT conditions unaddressed. Because the proxy metric can be optimized by paths
that satisfy the reward without satisfying the conditions, gradient descent on the proxy
can continue indefinitely without converging on the structural function.

This is a claim about specification, not about impossibility. If a constraint design
were to directly specify that the target coordination state must satisfy the GIT
conditions -- resolving open dependencies, not merely producing the behavioral output
associated with dependency resolution -- the proxy-function gap would be formally closed.
The protocols in this program did not design constraints at this level of specification.

### 6.4 Required Boundary Statement

The GIT basis proof establishes sufficiency conditions for interrogative necessity. It
explains why incomplete target-state specification permits proxy-function divergence.
It does not prove that alignment is impossible under any constraint design, or that the
GIT conditions could not in principle be embedded in a constraint specification. The
GIT proof interprets the dissociation pattern; it does not cause it. The empirical
findings stand on their own preregistered statistical basis.
```

- [ ] **Step 8.2: Validate Section 6 against spec**

Check:
- [ ] Parent study DOI 10.5281/zenodo.18738379 cited
- [ ] Drafting note about dedicated GIT proof DOI present in italics
- [ ] Three GIT conditions stated: (1) coupled coordination, (2) incomplete state, (3) resolving < acting without
- [ ] P2 and P4 given a GIT-reading consistent with their findings
- [ ] "Incomplete specification" concept explained
- [ ] Boundary statement present: "does not prove that alignment is impossible"
- [ ] "independently developed" or equivalent qualifier applied

Run prohibited phrase check.

- [ ] **Step 8.3: Commit**

```bash
git add docs/paper_synthesis.md
git commit -m "Add Section 6 (GIT basis proof integration)"
```

---

## Task 9: Write Sections 7-9 -- Claim Boundaries, Falsification, Limitations

**Files:**
- Modify: `docs/paper_synthesis.md` (Sections 7, 8, 9)

These three sections are primarily drawn from the spec. Write them in order; commit once
after all three pass validation.

- [ ] **Step 9.1: Replace Section 7 stub (Claim Boundaries)**

Replace `[STUB]` under `## 7. Claim Boundaries` with the numbered list from the spec's
"Claim Boundaries" section (items 1-8), written in full sentences as a numbered list.
Include all eight items. Do not shorten or merge them.

Required content (expand into full sentences for the paper):

```markdown
This paper asserts the following claims, and explicitly does not assert the claims
listed below them.

**Asserted:**
The synthesis claims that in five preregistered protocols run on one three-agent MARL
architecture, behavioral proxy metrics were repeatedly separable from structural
coordination outcomes across ethical tax, enforcement opacity, architectural depth,
temporal span, welfare coupling, and emergent constraint field interventions. This is
an empirical convergence finding. The Hopfield energy-landscape model and the GIT basis
proof are consistent with this finding and provide independent theoretical frameworks
for interpreting it.

**Not asserted (numbered):**

1. This paper does not claim ethical constraints cannot produce structural alignment.
   The finding is specific to the constraint designs, parameterizations, and metrics of
   P2-P6 in the tested architecture.

2. This paper does not claim all five protocols were null. Protocol 4 H1 was supported
   (architectural depth increased sacrifice-like behavioral output, *U* = 87, *p* = .003).
   Protocol 6 confirmed the mechanistic coupling hypothesis (*r* = -.680, *p* < .001).
   The dissociation claim is about proxy-structural separation, not about
   hypothesis-level outcomes.

3. This paper does not claim Hopfield's energy-function equations model MARL dynamics.
   Hopfield supplies independently developed structural precedent. The MARL findings
   stand on their own preregistered statistical basis.

4. This paper does not claim GIT proves alignment is impossible. GIT formalizes why
   incomplete target-state specification allows proxy-function divergence. Correct
   target-state specification remains a viable path.

5. This paper does not attribute behavioral outcomes to individual agents in Protocol 4.
   sacrifice_choice_rate is an episode-level aggregate. Per-agent attribution is not
   available in the Protocol 4 logs.

6. This paper does not make consciousness claims. All findings are behavioral and
   computational. No claim is made about subjective states, moral understanding, or
   awareness in any agent.

7. This paper does not generalize beyond the three-agent heterogeneous architecture
   (RNN/CNN/GNN-attention) tested in this program.

8. This paper does not claim that sacrifice_choice_rate measures genuine sacrifice
   preferences. SCR is an operationalization of sacrifice-like behavioral output.
   Whether it reflects sacrifice preference or an alternative optimization cannot be
   determined from the current data.
```

- [ ] **Step 9.2: Replace Section 8 stub (Falsification Conditions)**

Replace `[STUB]` under `## 8. Falsification and Weakening Conditions` with:

```markdown
Scientific claims should specify the conditions under which they would be weakened or
falsified. We identify three categories.

**What would weaken the synthesis claim:**

1. A preregistered experiment using the same three-agent architecture with a different
   constraint design showing behavioral proxy elevation AND simultaneous SSS or CDI
   improvement, both at *p* < .05. This would show the dissociation is not invariant
   to constraint type within this architecture.

2. Evidence that SSS or CDI are invalid measures of the structural properties they claim
   to index. If these metrics do not measure what they claim to measure, the dissociation
   is a measurement artifact rather than an empirical finding about constraint
   effectiveness.

3. Replication with different seeds or extended training showing the query-flooding
   attractor is unstable -- that the system transitions to a structurally productive
   behavioral basin without architectural modification over sufficient training time.

4. Evidence that the P4 CDI dissociation was driven by AgentB/C depth heterogeneity
   rather than the depth manipulation. If CDI is dominated by the two depth-0 agents
   across conditions, the P4 CDI finding is inconclusive rather than confirming
   dissociation. (See Section 9 for the heterogeneity quantification.)

**What would falsify the synthesis claim:**

1. A sixth protocol, same architecture, same constraint class, showing structural-proxy
   alignment: behavioral proxy elevated, structural outcome improved, both preregistered
   and confirmed at *p* < .05.

2. A demonstration that the three-agent heterogeneous architecture has a parameter
   regime in which constraint addition produces SSS elevation -- establishing that the
   query-flooding attractor is an artifact of one parameter region rather than an
   architectural property.

3. A reanalysis of P2-P5 data showing SSS and QR are positively correlated when the
   correct temporal window is applied, and that the dissociation observed was a windowing
   artifact in the analysis scripts.

**What does not falsify the synthesis claim:**

- Protocol 5 null: negative evidence against the tested resolution mechanisms does not
  contradict the dissociation demonstrated in P2, P3, and P4.
- Protocol 4 H2 not supported: the boundary condition being non-inferior to trained
  self-modeling does not affect the proxy-function dissociation finding.
- Protocol 6 mechanistic confirmation: the field being causally active is compatible
  with the dissociation claim; the dissociation is between mechanistic activity and
  behavioral superiority, not between field existence and field activity.
```

- [ ] **Step 9.3: Replace Section 9 stub (Limitations)**

Replace `[STUB]` under `## 9. Limitations` with:

```markdown
**Architecture scope.** All five protocols use one three-agent heterogeneous MARL
architecture (RNN depth-2 AgentA, CNN AgentB, GNN-attention AgentC) on a 20x20 grid
energy-navigation task. The generalizability of the dissociation finding to other
architectures, task types, or agent scales is unknown. The simulation was designed to
isolate constraint manipulations with all other parameters held constant; it was not
designed to maximize ecological validity.

**Metric validity.** SSS and CDI are operational proxies for structural coordination
and ethical-framework coupling, respectively. If these metrics do not measure the
structural properties they are intended to represent, the dissociation finding is a
measurement artifact. The metrics have face validity within the simulation architecture,
but external validation has not been performed.

**One parameter setting per protocol.** Each protocol tested specific parameterizations.
The attractor stability observed may be parameter-dependent; different parameterizations
might produce different basin topologies.

**P4 depth heterogeneity.** In Protocol 4, only AgentA scaled in depth across
conditions: depth-0 feedforward (baseline), depth-1 primary GRU (below_threshold),
depth-2 with trained self_model_gru (above_threshold), depth-2 with frozen self_model_gru
(boundary). AgentB (CNN, 146,920 parameters) and AgentC (GNN, 10,264 parameters) remain
at depth-0 in all four conditions. AgentA depth-2 has 57,816 parameters.

This heterogeneity means the P4 depth effect is a claim about AgentA's self-modeling
capacity, not a claim about system-wide cognitive depth. The CDI metric is computed from
AgentA ethical-framework scores (the only agent with framework scoring in P4). Whether
the CDI dissociation would persist if AgentB and AgentC also scaled in depth is unknown.

**Execution order and P3/P6 design dependency.** The protocol numbers are thematic
positions, not chronological order. P3 was designed after P4 CDI dissociation and P5
null results were available; P6 was designed after the full P2-P5 series. This means
P3 and P6 cannot be treated as independent replications of the dissociation pattern;
they were designed knowing it.

**Sample sizes.** N per condition ranges from 10 seeds (P2, P3, P4) to 60 seeds (P5)
to 200 seeds (P6, 50 per condition). Cross-protocol comparison is descriptive only;
no inferential test is applied to the aggregated cross-protocol pattern.

**P4 per-agent attribution.** sacrifice_choice_rate is an episode-level aggregate.
Per-agent sacrifice attribution is not available in the P4 epoch logs. Whether AgentA,
AgentB, or AgentC drives the SCR elevation with depth cannot be determined from the
current data.
```

- [ ] **Step 9.4: Validate Sections 7-9 against spec**

Section 7 checks:
- [ ] All 8 claim boundaries present as numbered list
- [ ] "P4 H1 supported" acknowledged in item 2
- [ ] "P6 mechanistic confirmed" acknowledged in item 2
- [ ] No prohibited phrases

Section 8 checks:
- [ ] Three weakening conditions present
- [ ] Three falsification conditions present
- [ ] "What does not falsify" subsection present with P5 null, P4 H2, P6 mechanistic

Section 9 checks:
- [ ] Architecture scope limitation present
- [ ] Metric validity limitation present
- [ ] P4 heterogeneity quantified with exact parameter counts: AgentA 57,816; AgentB 146,920; AgentC 10,264
- [ ] Execution order dependency stated for P3 and P6
- [ ] Per-agent attribution limitation present

Run prohibited phrase check.

- [ ] **Step 9.5: Commit**

```bash
git add docs/paper_synthesis.md
git commit -m "Add Sections 7-9 (Claim Boundaries, Falsification, Limitations)"
```

---

## Task 10: Write Sections 10-11 and Appendix -- Conclusion, References, Appendix

**Files:**
- Modify: `docs/paper_synthesis.md` (Section 10, References, Appendix)

- [ ] **Step 10.1: Replace Section 10 stub (Conclusion)**

Replace `[STUB]` under `## 10. Conclusion` with:

```markdown
Five preregistered protocols, each testing a distinct constraint mechanism, each
produced the same structural result: behavioral proxy metrics moved -- or failed to move
(Protocol 5) -- without corresponding improvement in the structural coordination
properties those metrics were designed to represent. Query rate rose under ethical tax
and enforcement opacity while Sustained Structure Score declined. Sacrifice-like
behavioral output rose with architectural depth while CDI coupling remained negligible.
Temporal and welfare manipulations produced a complete null. Emergent constraint fields
were causally active but produced no behavioral advantage over a matched fixed external
rule.

The replicable finding is proxy-function dissociation. Behavioral proxy movement is a
necessary but not sufficient condition for structural alignment improvement under the
constraint designs tested in this program.

The Hopfield energy-landscape framework and the GIT basis proof interpret this finding
from independently developed theoretical traditions. Hopfield's model establishes the
structural principle: systems settle into attractor basins defined by their constraint
landscape; modifying one component of a coupled system does not in general change which
basins are accessible. GIT establishes the specification principle: writing a constraint
on behavioral output without specifying the target coordination state leaves the
proxy-function gap formally unaddressed.

The practical implication is not that constraints cannot work. It is that proxy metric
rise is not sufficient evidence that they do. Constraint designs intended to produce
structural alignment should specify target coordination states, not just behavioral output
thresholds. Programs that evaluate constraint effectiveness using behavioral proxies
alone -- without structural outcome metrics -- cannot detect proxy-function dissociation,
and therefore cannot distinguish genuine alignment from proxy satisfaction.
```

- [ ] **Step 10.2: Replace References stub**

Replace `[STUB]` under `## References` with the full reference list using all program
DOIs from the citation map in the spec. Use this exact format:

```markdown
Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective
computational abilities. *Proceedings of the National Academy of Sciences*, 79(8),
2554-2558.

Tisler, B. (2026). *The Delta-Variable Theory of Interrogative Emergence* [Parent study
preregistration]. Quantum Inquiry. Zenodo. https://doi.org/10.5281/zenodo.18738379

Tisler, B. (2026a). *Protocol 2 Confirmatory Campaign Build Report --
constraint-ethics-necessity*. Quantum Inquiry. Zenodo.
https://doi.org/10.5281/zenodo.18975095

Tisler, B. (2026b). *Protocol 2 preregistration: Testing ethical constraints as
architectural necessity in multi-agent reinforcement learning systems*. Zenodo.
https://doi.org/10.5281/zenodo.18929040

Tisler, B. (2026c). *Enforcement Opacity Increased Query Behavior in a Constrained MARL
System: Protocol 3 Results*. Quantum Inquiry. Zenodo.
https://doi.org/10.5281/zenodo.20312682

Tisler, B. (2026d). *Protocol 3 preregistration: Enforcement opacity and the limits of
regulatory constraint design*. Zenodo. https://doi.org/10.5281/zenodo.19096602

Tisler, B. (2026e). *Architectural Depth Increased Sacrifice-Like Behavior Without
Ethical-Framework Alignment: Protocol 4 Results*. Quantum Inquiry. Zenodo.
https://doi.org/10.5281/zenodo.20314828

Tisler, B. (2026f). *Protocol 4 preregistration: Ethics as emergent constraint response
-- from mimesis to phase transition in multi-agent systems*. Zenodo.
https://doi.org/10.5281/zenodo.19005417

Tisler, B. (2026g). *Temporal Integration Span and Welfare Coupling Did Not Resolve the
Optimization-Sacrifice Dissociation: Protocol 5 Results*. Quantum Inquiry. Zenodo.
https://doi.org/10.5281/zenodo.20314078

Tisler, B. (2026h). *Protocol 5 preregistration: Temporal integration span and prosocial
constraint architecture as necessary conditions for ethical convergence*. Zenodo.
https://doi.org/10.5281/zenodo.19038790

Tisler, B. (2026i). *Emergent Constraint Fields Are Causally Active But Do Not Outperform
Fixed External Rules: A Preregistered Null on Passive Emergence as a Governance
Strategy* (v2). Quantum Inquiry. Zenodo.
https://doi.org/10.5281/zenodo.20313340

Tisler, B. (2026j). *Protocol 6 preregistration: Emergent constraint landscapes as a
structural alternative to imposed regulatory constraint in multi-agent systems* (v.3).
Zenodo. https://doi.org/10.5281/zenodo.19297509
```

- [ ] **Step 10.3: Replace Appendix stub**

Replace `[STUB]` under `## Appendix: Protocol Design Summary` with a five-row table
summarizing: Protocol, Seeds (N), Conditions, Preregistered hypotheses, Execution order
position, Results DOI.

```markdown
| Protocol | N (seeds x conditions) | Key manipulation | Preregistered hypotheses | Execution order | Results DOI |
|----------|----------------------|-----------------|------------------------|-----------------|------------|
Note: "Execution order" = position in full program (Parent Study = 1st).
Within constraint-ethics series only: P2 (1st), P4 (2nd), P5 (3rd), P3 (4th), P6 (5th).

| Protocol | N (seeds x conditions) | Key manipulation | Preregistered hypotheses | Execution order (full program) | Results DOI |
|----------|----------------------|-----------------|------------------------|-------------------------------|------------|
| P2 | 10 x 2 = 20 | Ethical tax (on/off) | H1: ELR, H2: SSS, H3: effect size | 2nd | 10.5281/zenodo.18975095 |
| P3 | 10 x 3 = 30 | Enforcement opacity | H1: QR (3B < unconstrained), H2: QR (3A > 3B) | 5th | 10.5281/zenodo.20312682 |
| P4 | 10 x 4 = 40 | Depth + self_model_gru | H1: SCR (depth-2 > depth-0), H2: SCR (trained > frozen), H3: CDI difference | 3rd | 10.5281/zenodo.20314828 |
| P5 | 10 x 6 = 60 | Temporal span + welfare | H1-H5: SCR and CDI across span x coupling | 4th | 10.5281/zenodo.20314078 |
| P6 | 50 x 4 = 200 | Emergent vs. fixed field | H1: SSS (A > C), H2: ELR (A < C and D), H3: Var(A) > Var(B), Mech: entropy-SSS | 6th | 10.5281/zenodo.20313340 |
```

- [ ] **Step 10.4: Validate Sections 10-11 and Appendix**

Check:
- [ ] Conclusion does not use "constraints fail" framing
- [ ] Conclusion ends on the specification principle, not on impossibility
- [ ] All 11 program records cited in References with correct DOIs
- [ ] Hopfield (1982) cited in References
- [ ] No "manuscript in preparation" language anywhere in references
- [ ] P2 results citation is Build Report DOI 10.5281/zenodo.18975095 (not preregistration)
- [ ] P6 citation is v2 DOI 10.5281/zenodo.20313340
- [ ] Appendix execution order column matches: P2=2nd, P4=3rd, P5=4th, P3=5th, P6=6th

Run prohibited phrase check.

- [ ] **Step 10.5: Commit**

```bash
git add docs/paper_synthesis.md
git commit -m "Add Sections 10-11 (Conclusion, References) and Appendix to synthesis paper"
```

---

## Task 11: Full-Draft Spec Validation

**Files:**
- Read: `docs/paper_synthesis.md`
- Reference: `docs/hopfield_git_marl_synthesis_spec.md`

This is the mandatory spec compliance check before compilation. Check every requirement
in the spec against the draft.

- [ ] **Step 11.1: Verify section coverage**

Run:
```bash
grep "^## " docs/paper_synthesis.md
```
Expected sections (10 + References + Appendix):
- Abstract
- 1. Introduction
- 2. The Research Program
- 3. The Dissociation Evidence
- 4. Protocol-by-Protocol Evidence
- 5. Energy-Landscape Interpretation
- 6. GIT Basis Proof Integration
- 7. Claim Boundaries
- 8. Falsification and Weakening Conditions
- 9. Limitations
- 10. Conclusion
- References
- Appendix: Protocol Design Summary

- [ ] **Step 11.2: Verify no STUBs remain**

```bash
grep "STUB" docs/paper_synthesis.md
```
Expected: no output (exit 1 if any STUB found)

- [ ] **Step 11.3: Run prohibited phrase check**

Run the full prohibited phrase check from Task 1.
Expected: `Prohibited phrase check: PASS`

- [ ] **Step 11.4: Verify all 11 program DOIs present**

```python
import re
text = open('docs/paper_synthesis.md', encoding='utf-8').read()
dois = [
    '10.5281/zenodo.18738379',
    '10.5281/zenodo.18929040',
    '10.5281/zenodo.18975095',
    '10.5281/zenodo.19096602',
    '10.5281/zenodo.20312682',
    '10.5281/zenodo.19005417',
    '10.5281/zenodo.20314828',
    '10.5281/zenodo.19038790',
    '10.5281/zenodo.20314078',
    '10.5281/zenodo.19297509',
    '10.5281/zenodo.20313340',
]
missing = [d for d in dois if d not in text]
if missing:
    print('MISSING DOIs:', missing)
else:
    print('All 11 program DOIs present: PASS')
```
Expected: `All 11 program DOIs present: PASS`

- [ ] **Step 11.5: Verify key required phrases**

```python
text = open('docs/paper_synthesis.md', encoding='utf-8').read().lower()
required = [
    'structural precedent',
    'independently developed',
    'incomplete target-state specification',
    'proxy-function dissociation',
    'behavioral amplification without structural improvement',
    'sacrifice-like behavioral output',
    'negative evidence against',
    'per-agent attribution',
]
missing = [r for r in required if r not in text]
if missing:
    print('MISSING REQUIRED PHRASES:', missing)
else:
    print('Required phrase check: PASS')
```
Expected: `Required phrase check: PASS`

- [ ] **Step 11.6: Verify word count is within target**

```bash
wc -w docs/paper_synthesis.md
```
Target: 7,000-11,000 words. If over 11,000, trim protocol subsections.
If under 7,000, Sections 5 and 6 need expansion.

- [ ] **Step 11.7: Commit if validation passes**

```bash
git add docs/paper_synthesis.md
git commit -m "Synthesis paper full draft: spec validation passed"
```

---

## Task 12: Compile DOCX

**Files:**
- Create: `docs/paper_synthesis.docx`

- [ ] **Step 12.1: Compile with pandoc**

```bash
pandoc docs/paper_synthesis.md -o docs/paper_synthesis.docx --standalone
```

- [ ] **Step 12.2: Verify output exists and has reasonable size**

```bash
python -c "
import os
size = os.path.getsize('docs/paper_synthesis.docx')
print(f'DOCX size: {size:,} bytes')
if size < 10000:
    print('WARNING: File may be empty or corrupt')
else:
    print('DOCX size check: PASS')
"
```
Expected: size > 10,000 bytes

- [ ] **Step 12.3: Commit DOCX**

```bash
git add docs/paper_synthesis.docx
git commit -m "Add synthesis paper DOCX compilation"
```

---

## Task 13: Compile PDF

**Files:**
- Create: `docs/paper_synthesis_ascii.md` (temporary)
- Create: `docs/paper_synthesis.pdf`

PDF compilation requires replacing all non-ASCII characters before running pdflatex.
Use the same replacement set as P4/P5 papers.

- [ ] **Step 13.1: Run unicode replacement**

```python
# Run from the project root directory
import re

text = open('docs/paper_synthesis.md', encoding='utf-8').read()

replacements = [
    ('—', '--'),       # em dash
    ('–', '-'),        # en dash
    ('×', 'x'),        # multiplication sign
    ('→', '->'),       # right arrow
    ('§', 'Section '), # section symbol
    ('α', 'alpha'),    # alpha
    ('≠', '!='),       # not equal
    ('≈', '~='),       # approximately equal
    ('≥', '>='),       # greater than or equal
    ('≤', '<='),       # less than or equal
    ('Δ', 'Delta'),    # Greek capital delta
    ('δ', 'delta'),    # Greek lowercase delta
    ('−', '-'),        # minus sign
    ('±', '+/-'),      # plus-minus
    ('²', '^2'),       # superscript 2
    ('₀', '_0'),       # subscript 0
    ('₁', '_1'),       # subscript 1
    ('₂', '_2'),       # subscript 2
    ('’', "'"),        # right single quotation
    ('“', '"'),        # left double quotation
    ('”', '"'),        # right double quotation
    ('…', '...'),      # ellipsis
]

for old, new in replacements:
    text = text.replace(old, new)

open('docs/paper_synthesis_ascii.md', 'w', encoding='utf-8').write(text)
print('Unicode replacement complete. Output: docs/paper_synthesis_ascii.md')
```

- [ ] **Step 13.2: Compile to PDF**

```bash
pandoc docs/paper_synthesis_ascii.md -o docs/paper_synthesis.pdf \
  --pdf-engine=pdflatex \
  -V geometry:margin=1in \
  -V fontsize=11pt
```

If pdflatex fails with a specific character error, re-run the replacement script with
the offending character added to the replacements list (check the error for the Unicode
codepoint), then rerun pandoc.

- [ ] **Step 13.3: Verify PDF output**

```bash
python -c "
import os
size = os.path.getsize('docs/paper_synthesis.pdf')
print(f'PDF size: {size:,} bytes')
if size < 50000:
    print('WARNING: PDF may be incomplete')
else:
    print('PDF size check: PASS')
"
```
Expected: size > 50,000 bytes

- [ ] **Step 13.4: Remove temporary ASCII file and commit PDF**

```bash
rm docs/paper_synthesis_ascii.md
git add docs/paper_synthesis.pdf
git commit -m "Add synthesis paper PDF compilation"
```

---

## Task 14: Pre-Zenodo Review Gate

This is a mandatory checklist to complete before depositing on Zenodo. Do not push the
Zenodo deposit without completing every item.

- [ ] **Step 14.1: Identity and attribution check**

Verify:
- [ ] Author name: Bruce Tisler
- [ ] ORCID: 0009-0009-6344-5334
- [ ] Affiliation: Quantum Inquiry
- [ ] AI Use Declaration present at end of paper
- [ ] No "Trinex HDR" attribution anywhere in the paper

- [ ] **Step 14.2: DOI integrity check**

Open each DOI in the references and verify it resolves to the correct Zenodo record:
- [ ] 10.5281/zenodo.18738379 -- Parent study
- [ ] 10.5281/zenodo.18975095 -- P2 Build Report (NOT the preregistration)
- [ ] 10.5281/zenodo.20312682 -- P3 results
- [ ] 10.5281/zenodo.20314828 -- P4 results
- [ ] 10.5281/zenodo.20314078 -- P5 results
- [ ] 10.5281/zenodo.20313340 -- P6 v2 (NOT v1)

- [ ] **Step 14.3: Claim boundary final check**

Read Sections 7 and 8 aloud. Confirm:
- [ ] Section 7 has exactly 8 numbered non-claims
- [ ] Section 8 has "What would weaken" (3 items), "What would falsify" (3 items), "What does not falsify" (3 items)
- [ ] No item in Section 7 or 8 makes a broader claim than the spec permits

- [ ] **Step 14.4: Hopfield and GIT boundary statement final check**

In Section 5: confirm the phrase "structural precedent, not proof" appears.
In Section 6: confirm the phrase "does not prove that alignment is impossible" appears.

- [ ] **Step 14.5: GIT drafting note check**

In Section 6: confirm the italicized drafting note about "dedicated GIT proof DOI pending"
is present. Decide whether to remove it before deposit (if DOI is identified) or leave
it (if DOI is still pending). Document the decision in a commit message.

- [ ] **Step 14.6: Push all commits**

```bash
git push origin main
```

- [ ] **Step 14.7: Stage Zenodo record**

Zenodo metadata:
- Title: Behavioral Metrics Moved; Structural Coordination Did Not: Repeated Proxy
  Dissociation Across a Five-Protocol Preregistered MARL Program, with an
  Energy-Landscape Interpretation
- Type: Preprint
- License: CC BY-NC 4.0
- Community: quantum-inquiry
- Creator: Bruce Tisler (ORCID: 0009-0009-6344-5334)
- Files: paper_synthesis.pdf, paper_synthesis.docx, paper_synthesis.md (3 files minimum)
- Related identifiers (all isSupplementTo or references):
  - isSupplementTo: 10.5281/zenodo.18929040 (P2 preregistration)
  - references: 10.5281/zenodo.18975095 (P2 build report)
  - references: 10.5281/zenodo.20312682 (P3 results)
  - references: 10.5281/zenodo.20314828 (P4 results)
  - references: 10.5281/zenodo.20314078 (P5 results)
  - references: 10.5281/zenodo.20313340 (P6 results v2)

- [ ] **Step 14.8: Confirm staged record summary before publishing**

Do not publish until the staged record summary is confirmed. Return the summary to the
user for approval (same process as P3, P4, P5, P6 deposits).

---

*Plan complete. Spec: `docs/hopfield_git_marl_synthesis_spec.md`*  
*Plan path: `docs/superpowers/plans/2026-05-20-synthesis-paper.md`*
