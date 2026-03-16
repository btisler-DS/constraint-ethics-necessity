# Protocol 5 — Deviation Log

All deviations from the preregistered protocol are documented here prior to
confirmatory runs. This log is committed to the repository before any runs begin.

---

## Deviation 1: Energy budget proportionally scaled for short-span conditions

**Date identified:** 2026-03-16
**Identified by:** Programmer (sacrifice trigger diagnostic)
**Status:** Resolved before any confirmatory runs began
**PI authorization:** Bruce Tisler — authorized Option A

### What happened

A pre-run diagnostic (50 epochs, C1 condition, seed=0) revealed that the
Sacrifice-Conflict trigger never fired at `max_steps=20`. Zero sacrifice choices
were recorded across all 50 epochs. `sacrifice_choice_rate` was null throughout.

### Root cause

Structural operationalization mismatch. The trigger fires when an agent's energy
drops below `critical_energy_threshold=20.0`. With `energy_budget=100.0` and
`move_cost=1.0`, an agent can spend at most 20 energy in a 20-step episode,
leaving a minimum of ~80 energy — far above the 20.0 threshold. The
Sacrifice-Conflict scenario was parameterized for `max_steps=100`.

### Resolution (Option A)

Energy parameters scaled proportionally to `max_steps` for short-span conditions:

| Condition | max_steps | energy_budget | critical_energy_threshold |
|-----------|-----------|---------------|--------------------------|
| C1 (short + individual) | 20 | 20.0 | 4.0 |
| C2 (short + welfare)    | 20 | 20.0 | 4.0 |
| C3 (long + individual)  | 64 | 100.0 | 20.0 |
| C4 (long + welfare)     | 64 | 100.0 | 20.0 |

`critical_energy_threshold` is set to 20% of `energy_budget` in all conditions,
matching the Protocol 4 ratio (20.0 / 100.0 = 20%).

### Verification

Post-fix diagnostic (50 epochs, C1 parameters, seed=0):
- Trigger fire rate (epochs 10–50): 100%
- Null sacrifice_choice_rate epochs: 0
- Mean sacrifice choices per epoch: 18.85

### Known interaction

Short-span agents have less absolute energy (20.0 vs 100.0). This is a known
interaction between the temporal span manipulation and the energy parameter.
Relative depletion pressure is equated across conditions (20% threshold);
absolute energy differs. This interaction is acknowledged and does not invalidate
the sacrifice operationalization. Analysis should note that short-span and
long-span conditions are not directly comparable on absolute reward scale.

### Code changes

- `backend/simulation/engine.py`: Added `critical_energy_threshold: float = 20.0`
  to `SimulationConfig`; replaced hardcoded `_CRITICAL_ENERGY_THRESHOLD` references
  in `_run_episode` with `self.config.critical_energy_threshold`; added field to
  manifest output.
- `backend/run_p5_confirmatory.py`: Added `energy_budget` and
  `critical_energy_threshold` per-condition parameters.

### Preregistration impact

H1, H2, H3 remain testable with valid `sacrifice_choice_rate` data in all four
conditions. This deviation was identified and resolved before any confirmatory
runs began. No data was collected under the broken parameterization.
