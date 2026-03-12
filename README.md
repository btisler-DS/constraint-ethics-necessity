# constraint-ethics-necessity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18929040.svg)](https://doi.org/10.5281/zenodo.18929040)
[![Preregistration](https://img.shields.io/badge/preregistration-v3-blue)](docs/preregistration.md)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

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

## Zenodo DOI

Preregistration DOI: [10.5281/zenodo.18929040](https://doi.org/10.5281/zenodo.18929040) — confirmed live, published March 9, 2026, v1, Open, indexed in OpenAIRE.

Build report DOI: [10.5281/zenodo.18975095](https://doi.org/10.5281/zenodo.18975095) — Protocol 2 Confirmatory Campaign Build Report, published March 12, 2026, v1, Open, indexed in OpenAIRE.

## Status: Confirmatory Runs Complete

Protocol 2 confirmatory campaign (20 seeds × 2 conditions × 500 epochs) is complete. Results inverted the preregistered prediction: constrained agents showed lower sustained behavioral complexity than unconstrained agents (Cohen's d = −2.18, p = 0.9996 in preregistered direction), driven by a systematic gaming pattern termed virtue theater — query-flooding behavior that satisfies the ethical constraint specification while degrading genuine interrogative diversity.

**Paper:** [Virtue Theater: Specification Gaming and Regulatory Constraint Failure in Multi-Agent Systems](docs/paper_virtue_theater.pdf)

Protocol 3 (testing architectural integration under genuine resource depletion) is in design and will be preregistered separately.

## License

[CC BY 4.0](LICENSE) — open for reproduction, challenge, and extension.
