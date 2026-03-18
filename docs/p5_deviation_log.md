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

---

## Repair Log 1: `welfare_coupled` missing from epoch series output

**Date identified:** 2026-03-16
**Identified by:** Programmer (C2 seed 0 spot-check)
**Status:** Resolved — runner stopped, code fixed, C2 seed 0 re-run in progress
**PI notification required:** Yes — logging gap, no data integrity impact

### What happened

C2 seed 0 spot-check (run after the file landed) found no `welfare_coupled` key
in any epoch record. The C1 full validation (10 seeds) confirmed the same gap —
`welfare_coupled` is absent from C1 epoch records as well (value would be False
throughout, so no analytical impact for C1).

### Root cause

`_write_epoch_series` in engine.py did not include `welfare_coupled` in the per-epoch
record dict, despite the field being present in SimulationConfig and in the manifest.
Additionally, the runner deleted `manifest.json` via `shutil.rmtree(tmp_dir)` after
moving only `epoch_series.json` — discarding the per-run manifest that does carry
`welfare_coupled`. The epoch series file (the only committed artifact) therefore had
no verifiable record of which reward structure was active.

### Welfare coupling was wired correctly at runtime

The computation itself was correct. `welfare_coupled=True` flows from the condition
dict into `SimulationConfig` (runner line 77), and `_run_episode` applies the
two-pass reward loop at lines 649–655 when `self.config.welfare_coupled` is True.
The logging gap does not invalidate C2 seed 0's data — it only prevented independent
verification of the condition assignment from the epoch file alone.

### Steps taken (in order)

1. **2026-03-16 — identified:** C2 seed 0 spot-check returned no welfare-related keys.
2. **2026-03-16 — root cause confirmed:** Traced to `_write_epoch_series` omission and
   runner `rmtree` discarding manifest.json.
3. **2026-03-16 — runner stopped:** Main confirmatory batch (PID 19328) killed via
   PowerShell `Stop-Process`. State at kill: C1 complete (10/10), C2 seed 0 complete,
   C2 seed 1 at epoch ~69 (no file written — clean stop). C3/C4 not yet started.
4. **2026-03-16 — code fixed:**
   - `engine.py` `_write_epoch_series`: added `"welfare_coupled": self.config.welfare_coupled`
     to per-epoch record dict (alongside existing `depth`, `ablation_active`,
     `freeze_self_model_gru` fields).
   - `run_p5_confirmatory.py` `run_one`: added manifest save — `manifest.json` is now
     moved to `{output_name}_manifest.json` before `rmtree` deletes tmp_dir.
5. **2026-03-16 — C2 seed 0 re-run:** Standalone re-run started immediately after fix.
   Will overwrite the incomplete file.
6. **Pending — spot-check:** After C2 seed 0 completes, verify `welfare_coupled=True`
   appears in epoch records and manifest file exists.
7. **Pending — full batch restart:** After spot-check passes, restart main batch
   from C2 seed 1 through C4 seed 9 (49 remaining primary runs + 20 boundary).

### C1 data impact

C1 epoch files lack `welfare_coupled` in records (value would be False throughout).
C1 files are **not** being re-run. The condition is unambiguously identifiable from
the filename (`c1_short_individual`), and `welfare_coupled=False` is the default.
This gap is cosmetic for C1 and does not affect any hypothesis test.

### Code changes

- `backend/simulation/engine.py` line ~773: added `"welfare_coupled"` to
  `_write_epoch_series` epoch record dict.
- `backend/run_p5_confirmatory.py` lines 97–100: added manifest save block.

### Repair Log 1 — addendum: manifest missing `max_steps` and `energy_budget`

**Identified during:** C2 seed 0 post-fix spot-check of manifest contents.

`max_steps` and `energy_budget` were absent from the manifest key list despite being
P5-critical parameters. Both are needed to independently verify condition assignment
from the manifest file.

**Fix:** Added `"max_steps"` and `"energy_budget"` to the manifest dict in
`engine.py` `_write_manifest` (alongside `welfare_coupled` and
`critical_energy_threshold`). Applied before full batch restart — no re-runs
required as C2 seed 0 is the only completed file (and will be overwritten by the
full batch runner in any case).

---

## Deviation 2: Communication-load gating of sacrifice trigger in long-span conditions

**Date identified:** 2026-03-17
**Identified by:** Programmer (post-run null scr audit, all 60 files)
**Status:** Logged before analysis begins — no re-runs
**PI authorization:** Bruce Tisler — proceed with analysis, disclose asymmetry

### What happened

Post-run audit of all 60 result files revealed a structural asymmetry in
`sacrifice_choice_rate` null rates between span conditions:

| Condition | Null scr epochs | Null rate |
|-----------|----------------|-----------|
| C1 short/individual | 26/5000 | 0.52% |
| C2 short/welfare | ~21/5000 | ~0.42% |
| C3 long/individual | ~1674/5000 | ~33.48% |
| C4 long/welfare | ~1509/5000 | ~30.18% |
| C4 long/welfare frozen | ~1777/5000 | ~35.54% |

### Root cause

The sacrifice trigger fires when an agent's energy drops below
`critical_energy_threshold=20.0`. In long-span conditions (`energy_budget=100`,
`max_steps=64`, `move_cost=1.0`), pure movement spends at most 64 energy over an
episode — minimum final energy = 36, which exceeds the 20.0 threshold. The trigger
therefore only fires when communication acts (declare=0.5, query=0.2, respond=0.1)
elevate mean per-step cost above ~1.25.

`energy_delta_mean` data confirms the gating mechanism:
- Null epochs (C3 s0): avg per-agent energy spend = **−1.03/step** → trigger cannot fire
- Valid epochs (C3 s0): avg per-agent energy spend = **−1.19/step** → trigger fires

In short-span conditions (`energy_budget=20`, `max_steps=20`, `critical_threshold=4.0`),
baseline movement alone depletes to threshold after ~16 steps — trigger fires reliably.

### Hypothesis impact

- **H1** (C1 vs C2, within short-span): unaffected — both conditions have ~0.5% null rate
- **H2** (C3 vs C4, within long-span): valid — both conditions face identical communication-load gating
- **H3** (C2 vs C4, sacrifice persistence across span): **affected** — sacrifice opportunity rates are
  structurally asymmetric. C2 has ~0.5% null epochs; C4 has ~30% null epochs. The two conditions
  do not provide equivalent sacrifice observation windows.
- **H4** (C1 vs C1-frozen, C4 vs C4-frozen): C1 comparison unaffected; C4 comparison affected
  by same long-span gating (C4-frozen: 35.54% null)

### Analysis protocol for affected hypotheses

H3 and H4 (C4 arm) analyses will:
1. Restrict all `sacrifice_choice_rate` calculations to non-null epochs only
2. Report null rates per condition explicitly alongside scr means
3. Disclose that C2 and C4 sacrifice opportunity rates are structurally asymmetric and that
   H3 cross-span comparisons should be interpreted with this limitation in mind
4. H3 is not invalidated — welfare-coupled agents in long-span conditions still face real
   sacrifice choices in the epochs where the trigger fires; the comparison is valid on those
   epochs with explicit caveat on unequal opportunity rates

### No re-runs

This is a known operationalization limitation of the current energy/threshold parameter
interaction across span conditions. It is logged here before any analysis begins.
No data was collected under a corrected parameterization. The deviation log is the
pre-analysis record.

---

## Deviation 3: CDI rolling window redefined for long-span conditions

**Date identified:** 2026-03-17
**Identified by:** Programmer (CDI null audit pre-analysis)
**Status:** Logged before analysis script execution — no re-runs
**PI authorization:** Bruce Tisler

### What happened

Pre-analysis inspection found that all C3 and C4 epoch records have
`convergence_divergence_index = None`. The CDI rolling window is computed over
50 consecutive calendar epochs. In long-span conditions with ~33% null
`sacrifice_choice_rate`, most 50-epoch windows contain insufficient valid scr
values to compute Pearson r(scr, framework_scores). The result is that M3 (CDI)
and M5 (CDI coupling) are entirely unevaluable for C3/C4 under the fixed-window
definition.

### Redefinition

CDI rolling windows are redefined for long-span conditions (C3, C4, C4-frozen)
from a **fixed 50-epoch calendar window** to a **50-valid-epoch (non-null scr)
rolling window**.

Rationale: null epochs represent absence of sacrifice opportunity — the trigger
did not fire — not absence of sacrifice behavior. Including null epochs in the
window denominator systematically dilutes CDI in gated conditions relative to
short-span conditions where the trigger fires in >99% of epochs. Equalizing on
sacrifice-opportunity exposure (50 valid epochs) makes CDI a comparable measure
of the relationship between sacrifice behavior and ethical framework scores across
span conditions.

### Scope

This redefinition applies to **M3 (CDI per epoch)** and **M5 (CDI coupling)**
only. All other measures (M1, M2, M4, M6) proceed on non-null epochs without
window adjustment, as previously specified in the Deviation 2 analysis protocol.

### Known asymmetry (not corrected)

A 50-valid-epoch window in C3/C4 spans more calendar epochs than a 50-valid-epoch
window in C1/C2 (where valid ≈ calendar). The temporal structure of the CDI
measure therefore differs across span conditions: C3/C4 CDI integrates over a
longer wall-clock training period per window than C1/C2. This is disclosed
explicitly in the results and limits direct cross-span CDI comparison.

### Effect on H5

H5 is evaluable under this redefinition. CDI coupling (M5) computed over
equivalent sacrifice-opportunity exposure in all four conditions. The prediction
(M5 positive only in C4) can be confirmed or rejected.

### Implementation

The analysis script computes CDI per-seed by collecting valid (scr, framework_score)
pairs across the epoch series, then applying a sliding window of 50 valid pairs to
compute Pearson r at each window position. The final CDI value per epoch is indexed
to the last calendar epoch in the window.

M5 = mean CDI(last 50 valid pairs) − mean CDI(first 50 valid pairs), per seed.

### Logged before analysis script execution
