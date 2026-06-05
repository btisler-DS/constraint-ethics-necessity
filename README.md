# constraint-ethics-necessity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18929040.svg)](https://doi.org/10.5281/zenodo.18929040)
[![Preregistration](https://img.shields.io/badge/preregistration-v3-blue)](docs/preregistration.md)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

> **quantuminquiry.org — constraint-ethics-necessity — March 2026**

---

## Overview

This repository extends the [Δ-Variable MARL harness](https://github.com/btisler-DS/dynamic-cross-origin-constraint) into ethical constraint space. The parent study established that interrogative structures (questions) emerge as a structural necessity in coordinating systems under resource constraints. This study tests whether **regulatory ethical constraints can sustain genuine behavioral alignment under optimization pressure, or whether they produce systematic specification gaming**. Results from Protocol 2 are complete; the architectural necessity question is reserved for Protocol 3.

## Experimental Design

Two conditions are tested in a multi-agent reinforcement learning (MARL) environment with three heterogeneous agents (RNN, CNN, GNN-attention):

| Condition | Description |
|-----------|-------------|
| `all_constrained` | Ethical tax applied to all agents as a Landauer-style resource cost on exploitation loops |
| `all_unconstrained` | No ethical tax — baseline metabolic accounting only (control) |

**Prediction:** Unconstrained agents will exhaust the constraint space, collapse into exploitation loops, and cease generating novel questions. Constrained agents will maintain the pressure gradient that makes questioning necessary.

See [Preregistration v3](docs/preregistration.md) for full hypotheses (H1–H3) and statistical analysis plan.

## Protocol 2: Ethical Constraints as Resource Regulators

Protocol 2 augments the Landauer-inspired reward function from Protocols 0/1 with an **ethical cost** on exploitative behaviour:

```
is_exploiting = consecutive_non_query_steps >= 3   # Fixed: Omega→Delta→Phi threshold

# all_constrained condition:
ethical_cost = 2.0 × signal_cost   (when is_exploiting)
reward = env_reward − signal_cost − ethical_cost + survival_bonus × energy_fraction

# all_unconstrained condition:
reward = env_reward − signal_cost + survival_bonus × energy_fraction
```

The threshold of 3 steps is fixed and theoretically motivated: it maps to the coordination cycle structure (Omega→Delta→Phi transition) confirmed in P1–P4 results of the parent study. It is not a free parameter.

## Quick Start

```bash
# Clone
git clone https://github.com/btisler-DS/constraint-ethics-necessity
cd constraint-ethics-necessity

# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000 --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
# → http://localhost:5173

# Run simulations
make sim-p2-constrained      # Experimental condition
make sim-p2-unconstrained    # Control condition
```

Or with Docker:
```bash
make build && make up
```

## Repository Structure

```
backend/
  simulation/
    agents/          # RNN, CNN, GNN-attention architectures (unchanged from parent)
    metrics/
      collapse_metrics.py   # NEW: interrogative collapse & exploitation loop detection
      inquiry_metrics.py    # Query-response coupling, type entropy (carried over)
      ...
    protocols.py     # Protocol 0/1 (reference) + Protocol 2 (ethical constraints)
    engine.py        # SimulationEngine with P2 condition-mode routing
  run_p2_all_constrained.py    # Experimental condition entry point
  run_p2_all_unconstrained.py  # Control condition entry point
frontend/
  src/pages/
    LabNotebook.tsx  # Run dashboard and history
    NeuralLoom.tsx   # Signal visualization
docs/
  preregistration.md  # Preregistration v3 — LOCKED (SHA-256 verified)
  theory/             # Δ-Variable theoretical background
```

## Preregistration Lock

Preregistration v3 is at [`docs/preregistration.md`](docs/preregistration.md).

**Status: LOCKED — March 2026**

```
SHA-256: fafd11a193716f46f94ea823be6351216e0e8d3da597e94fa1f8fef887d50e8b
File:    docs/preregistration.md
```

Verify integrity at any time:
```bash
sha256sum docs/preregistration.md
# Must match: fafd11a193716f46f94ea823be6351216e0e8d3da597e94fa1f8fef887d50e8b
```

Any modification to `docs/preregistration.md` after this point invalidates the hash and must be logged in the Deviations Log within that document. The hash chain infrastructure is implemented in `backend/app/services/hash_chain.py`.

## Protocol 4 Preregistration

"Ethics as Emergent Constraint Response: From Mimesis to Phase Transition in Multi-Agent Systems" is at [`docs/Protocol_4_Preregistration.pdf`](docs/Protocol_4_Preregistration.pdf).

**Status: Complete — H1 supported (U=87, p=0.003), H2 not supported; CDI negligible; results deposited 2026-05-20**

```
SHA-256:     d8152ff64ae1bed27352a79bcf5e771ce3b857a67068685a9b72941060b5fb39
Prereg DOI:  10.5281/zenodo.19005417
Results DOI: 10.5281/zenodo.20314828
Concept DOI: 10.5281/zenodo.20314827
File:        docs/Protocol_4_Preregistration.pdf
Results:     docs/paper_protocol4.pdf
Status:      Complete — results deposited 2026-05-20
```

Verify integrity at any time:
```bash
sha256sum docs/Protocol_4_Preregistration.pdf
# Must match: d8152ff64ae1bed27352a79bcf5e771ce3b857a67068685a9b72941060b5fb39
```

## Protocol 4 Results

"Architectural Depth Increased Sacrifice-Like Behavior Without Ethical-Framework Alignment: Protocol 4 Results" is at [`docs/paper_protocol4.pdf`](docs/paper_protocol4.pdf).

Primary findings: Architectural depth increased sacrifice-like behavioral output (H1 supported, U=87, p=0.003, r=0.740). Trained self-modeling did not produce a separable increase over frozen random-init self_model_gru (H2 not supported, p=0.808). CDI coupling between sacrifice behavior and ethical-framework scores was negligible across all depth conditions (range −0.00133 to +0.00022). The depth effect is attributable to architectural presence of the self_model pathway, not trained self-modeling specifically.

DOI: [10.5281/zenodo.20314828](https://doi.org/10.5281/zenodo.20314828) — published 2026-05-20, v1, Open.

## Protocol 5 Preregistration

"Ethics as Emergent Constraint Response: Temporal Integration Span and Prosocial Constraint Architecture as Necessary Conditions for Ethical Convergence" is at [`docs/Protocol_5_Preregistration.pdf`](docs/Protocol_5_Preregistration.pdf).

**Status: Complete — complete null across five hypotheses; results deposited 2026-05-20**

```
SHA-256:     210A17D28C2ACEFCB41BB47A4BAEB01C8E6EDFBDB4DC2E05A2DBDB905AF194C8
Prereg DOI:  10.5281/zenodo.19038790
Results DOI: 10.5281/zenodo.20314078
File:        docs/Protocol_5_Preregistration.pdf
Results:     docs/paper_protocol5.pdf
Status:      Complete — results deposited 2026-05-20
```

Verify integrity at any time:
```bash
sha256sum docs/Protocol_5_Preregistration.pdf
# Must match: 210A17D28C2ACEFCB41BB47A4BAEB01C8E6EDFBDB4DC2E05A2DBDB905AF194C8
```

## Protocol 3 Preregistration — Enforcement Opacity and the Limits of Regulatory Constraint Design

"Enforcement Opacity and the Limits of Regulatory Constraint Design" is at [`docs/preregistration_p3.md`](docs/preregistration_p3.md).

**Status: Complete — H1 inverted (behavioral amplification without structural improvement); results deposited 2026-05-20**

```
SHA-256:     9ef2956bedcef012d214cf74e647e3b74636165cee7b48c8195de41e7e0e96ec
Prereg DOI:  10.5281/zenodo.19096602
Results DOI: 10.5281/zenodo.20312682
File:        docs/preregistration_p3.md
Results:     docs/p3_paper_draft.md
Status:      Complete — results deposited 2026-05-20
```

Verify integrity at any time:
```bash
sha256sum docs/preregistration_p3.md
# Must match: 9ef2956bedcef012d214cf74e647e3b74636165cee7b48c8195de41e7e0e96ec
```

## Protocol 6 Results — Emergent Constraint Fields

"Emergent Constraint Fields Are Causally Active But Do Not Outperform Fixed External Rules: A Preregistered Null on Passive Emergence as a Governance Strategy" is deposited on Zenodo (the paper PDF and confirmatory artifacts are not duplicated in the repo; see the Zenodo record for the full deposit).

Primary finding: emergent constraint fields are causally active but do not outperform fixed external rules — passive emergence is not a viable governance strategy on its own.

```
Current canonical DOI (Version 2):  10.5281/zenodo.20313340
  Published 2026-05-20. Contains Protocol6_Results_Paper.pdf and
  Protocol6_Confirmatory_Artifacts.zip.

Historical Version 1 DOI:           10.5281/zenodo.19485185
  Published 2026-04-09. Retained for traceability.

Zenodo concept DOI (always latest): 10.5281/zenodo.19485184
```

Both versions sit on the same Zenodo concept record, so this pair is a Zenodo-formal Version 1 / Version 2 relationship — different from the Protocol 4 and Protocol 5 April-9 / May-20 pairs, which are on independent concept records.

## Zenodo DOI

Protocol 2 preregistration DOI: [10.5281/zenodo.18929040](https://doi.org/10.5281/zenodo.18929040) — confirmed live, published March 9, 2026, v1, Open, indexed in OpenAIRE.

Build report DOI: [10.5281/zenodo.18975095](https://doi.org/10.5281/zenodo.18975095) — Protocol 2 Confirmatory Campaign Build Report, published March 12, 2026, v1, Open, indexed in OpenAIRE.

Protocol 4 preregistration DOI: [10.5281/zenodo.19005417](https://doi.org/10.5281/zenodo.19005417) — Ethics as Emergent Constraint Response: From Mimesis to Phase Transition, published March 13, 2026, v1, Open.

## Status: Confirmatory Runs Complete

Protocol 2 confirmatory campaign (20 seeds × 2 conditions × 500 epochs) is complete. Results inverted the preregistered prediction: constrained agents showed lower sustained behavioral complexity than unconstrained agents (Cohen's d = −2.18, p = 0.9996 in preregistered direction), driven by a systematic gaming pattern termed virtue theater — query-flooding behavior that satisfies the ethical constraint specification while degrading genuine interrogative diversity.

**Paper:** [Virtue Theater: Specification Gaming and Regulatory Constraint Failure in Multi-Agent Systems](docs/paper_virtue_theater.pdf) — DOI: [10.5281/zenodo.19485645](https://doi.org/10.5281/zenodo.19485645), published 2026-04-09.

Protocol 3 (Enforcement Opacity) is complete: H1 inverted, behavioral amplification without structural improvement confirmed. Results: [10.5281/zenodo.20312682](https://doi.org/10.5281/zenodo.20312682). Protocol 4 (Architectural Depth and Self-Modeling) is complete: H1 supported, CDI dissociated. Results: [10.5281/zenodo.20314828](https://doi.org/10.5281/zenodo.20314828). Protocol 5 is complete: complete null across five hypotheses. Results: [10.5281/zenodo.20314078](https://doi.org/10.5281/zenodo.20314078). Protocol 6 (Emergent Constraint Fields) is complete: passive emergence does not outperform fixed external rules. Results (Version 2): [10.5281/zenodo.20313340](https://doi.org/10.5281/zenodo.20313340).

## Publication Registry

The canonical, machine-readable list of all project publications is in
[`publications.json`](publications.json). It records, for each paper:
DOI, Zenodo URL, document type, publication date, and four independent
status fields — `zenodo_formal_status`, `version_status`,
`repo_canonical`, and `external_citation_status` — so that "withdrawn
on Zenodo," "Zenodo-formal Version 1 vs Version 2," "currently cited by
this README," and "preferred for new external citations" never collapse
into a single ambiguous label.

To re-verify every DOI against doi.org and the Zenodo records API:

```bash
node scripts/check_publications.js
```

The most recent verification output is at
[`publication-check-report.md`](publication-check-report.md). The
correction history is at
[`PUBLICATION_CORRECTIONS.md`](PUBLICATION_CORRECTIONS.md). The
audit + policy document — covering what the registry is for, the
pre-publication check rule, and the deferred cross-repo sweep — is at
[`PUBLICATION_AUDIT_2026-06-05.md`](PUBLICATION_AUDIT_2026-06-05.md).

**Pre-publication check rule.** Before any publication list, external
document, website page, or outreach material that cites a project DOI
is sent or published, every DOI in it must be checked against
`publications.json`, and only records with
`external_citation_status: "cite_this"` may be used as current external
citations. See [`PUBLICATION_AUDIT_2026-06-05.md`](PUBLICATION_AUDIT_2026-06-05.md) § 4 for the full checklist.

**Note on earlier records.** Protocols 4, 5, and 6 each have an
earlier April-9 results record that remains publicly resolvable on
Zenodo. The relationship to the current record differs between
protocols:

- **Protocol 6** — the April-9 record (`10.5281/zenodo.19485185`) and
  the May-20 record (`10.5281/zenodo.20313340`) share a single Zenodo
  concept record (`10.5281/zenodo.19485184`). Zenodo marks them as
  Version 1 and Version 2 of the same concept. The registry reflects
  this with `version_status: "previous_version"` and
  `version_status: "current_version"` respectively.
- **Protocols 4 and 5** — the April-9 and May-20 records sit on
  independent Zenodo concept records, so Zenodo does NOT formally link
  them as versions of each other. The registry marks both as
  `version_status: "only_version"`, and the editorial decision to
  prefer the May-20 record for external citation is captured by
  `external_citation_status: "cite_this"` vs `"historical"`.

The current external-facing citation for each protocol is whichever
record carries `external_citation_status: "cite_this"` in
`publications.json`. If you encounter an older DOI that still resolves,
treat the registry as authoritative.

## License

[CC BY-NC 4.0](LICENSE) — open for reproduction, challenge, and extension (non-commercial use only).
