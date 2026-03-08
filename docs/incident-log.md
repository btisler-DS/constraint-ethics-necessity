# Incident Log — Execution Environment Issues

**Project**: Dynamic Cross-Origin Constraint (Δ-Variable Theory MARL campaign)
**Last updated**: 2026-03-07

This document records all execution environment failures encountered during the
confirmatory and post-confirmatory campaign. Each incident includes the symptom,
root cause, diagnosis method, fix, and prevention note.

---

## Incident 1 — P2 Rerun: Wrong Python Interpreter

**Date**: 2026-03-05 (root cause identified); 2026-03-07 (resolved)
**Experiment**: P2 cost-sensitivity rerun (`run_p2_rerun.py`)
**Severity**: Campaign blocker — 0 runs produced after two full launch attempts

### Symptom

`run_p2_rerun.py` launched via orchestrator (`python run_p2_rerun.py --workers 5`).
Script exited with code 1 after ~236 minutes. Zero manifests produced. No useful
error message in orchestrator.log — the failure was silent at the orchestrator level.

### Root Cause

The system `python` command resolved to **Python 3.14** (a separate installation),
which has no `torch` package. The script calls `import torch as _torch` inside
`run_new_seeds()` (not at module top-level), so the error occurred only when the
function was invoked — after the argument parsing and logging had already started,
making it appear to run before failing.

```
ModuleNotFoundError: No module named 'torch'
```

The correct interpreter is `C:/Users/btisl/miniconda3/python.exe`
(Python 3.11, torch 2.6.0+cu124).

### Diagnosis

Ran `python run_p2_rerun.py --seeds 0 1 --workers 1` with stdout visible. Saw the
`ModuleNotFoundError` immediately.

### Fix

All campaign scripts must be invoked with the full miniconda path:
```
C:/Users/btisl/miniconda3/python.exe run_p2_rerun.py --workers 5
```

All internal runner scripts use `sys.executable` to spawn child processes, so the
interpreter propagates correctly once the parent is launched correctly.

### Prevention

- Never use bare `python` on this machine. Always use full miniconda path.
- The orchestrator (`run_all_experiments.py`) also uses `sys.executable` — it must
  itself be launched with the miniconda interpreter.
- A pre-flight check (`which python && python --version`) should be run before any
  campaign launch.

---

## Incident 2 — P2 Rerun: Stale `p2_summary.json` During Run

**Date**: 2026-03-07
**Experiment**: P2 cost-sensitivity rerun
**Severity**: Monitoring confusion — no actual data loss

### Symptom

During the 13-hour P2 run, `p2_summary.json` showed `r=0.1955, n=21` for most of
the run despite 30+ manifests being present in `data/p2_rerun/`.

### Root Cause

`p2_summary.json` is only written at the end of the script's analysis phase, after
all manifests are collected. The stale `n=21` was from an earlier failed partial run
(smoke test data). The file was not updated incrementally.

### Fix

None needed. Monitor manifest count (`data/p2_rerun/**/manifest.json`), not the
summary file, as the live progress indicator during a run.

### Prevention

Document that `p2_summary.json` (and equivalent `*_summary.json` files) are only
valid after the script has completed. Do not read them as live status.

---

## Incident 3 — Orchestrator Re-queued Completed P2

**Date**: 2026-03-07
**Experiment**: Post-confirmatory orchestrator startup
**Severity**: Near-miss — would have wasted ~13 hours re-running P2

### Symptom

After P2 completed (45/45 manifests), the post-confirmatory orchestrator
(`run_all_experiments.py`) was launched without `--skip`. It began re-running P2
from scratch despite all manifests being present.

### Root Cause

`run_experiment()` in `run_all_experiments.py` has no skip-if-complete logic. It
records `n_before` and `n_after` to report progress, but does not check whether the
target count is already met before launching the script.

### Fix

Killed the orchestrator and relaunched with `--skip 1`:
```
C:/Users/btisl/miniconda3/python.exe run_all_experiments.py --skip 1
```

### Prevention

Always use `--skip N` when relaunching the orchestrator mid-campaign. Count the
completed experiments (not seeds) to determine the correct skip value.

The orchestrator queue order is fixed:
1. P2 (45 runs) — COMPLETE
2. P5 (30 runs)
3. Exp 3, 11, 12, 4, 7, 8, 9, 13, 14, 15, 1 (remaining)

---

## Incident 4 — P5 Rerun: CUDA Deadlock (8-Hour Silent Hang)

**Date**: 2026-03-07
**Experiment**: P5 coordination advantage rerun (`run_p5_rerun.py`)
**Severity**: Full day lost — 0 runs produced after 7h 42m

### Symptom

`run_p5_rerun.py` was launched (via orchestrator) at 07:33. By 15:15 (7h 42m later),
zero manifests existed in `data/p5_rerun/`. The directory had never been created.
28 Python processes were running with status `running` in psutil, all showing
`spawn_main` in their command line — the standard appearance of
`ProcessPoolExecutor` worker processes on Windows.

No error messages were produced. The orchestrator log showed only the "STARTING: P5"
line — no progress lines and no completion line.

### Root Cause

`run_new_seeds()` in `run_p5_rerun.py` (lines 168–171) detects available GPUs and
assigns `cuda:0` / `cuda:1` to worker processes:

```python
import torch as _torch
_n_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
for _i, _j in enumerate(jobs):
    _j["device"] = f"cuda:{_i % _n_gpus}" if _n_gpus > 0 else "cpu"
```

The machine has **two RTX 3060 GPUs**. On Windows, `ProcessPoolExecutor` uses the
`spawn` start method (fork is not available). Each spawned worker process attempts
to initialize a CUDA context for its assigned device. With multiple workers
competing to initialize CUDA contexts across two GPUs in spawned (not forked)
processes, the CUDA runtime deadlocks — all workers freeze at context init and
never proceed to run the simulation.

This is a known Windows + CUDA + `multiprocessing.spawn` incompatibility. The P2
script did not exhibit this because it has no GPU assignment logic at the
dispatcher level (workers default to `cpu`).

### Diagnosis

1. Confirmed symptom: 28 frozen `spawn_main` processes, 0 manifests after 7h 42m.
2. Confirmed seed 15 runs fine in isolation (no multiprocessing):
   `SimulationEngine(seed=15, device='cpu').run()` completed in <2s (2-epoch test).
3. Identified the GPU assignment block in `run_p5_rerun.py` lines 168–171.
4. Confirmed this block is absent in `run_p2_rerun.py`.

### Fix

Patched `run_p5_rerun.py` to force CPU for all workers:

```python
# Before (deadlocks on Windows with dual CUDA GPUs):
import torch as _torch
_n_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
for _i, _j in enumerate(jobs):
    _j["device"] = f"cuda:{_i % _n_gpus}" if _n_gpus > 0 else "cpu"

# After:
# Force CPU: CUDA multi-process spawning deadlocks on Windows with dual GPUs
for _j in jobs:
    _j["device"] = "cpu"
```

Smoke test (seeds 15–16, workers=2) launched after patch — running as of 15:30.

### Prevention

- All `ProcessPoolExecutor`-based scripts on this machine must use `device='cpu'`
  for worker jobs. Do not attempt GPU assignment in spawned workers on Windows.
- If GPU acceleration is needed, use `torch.multiprocessing` with `forkserver`
  start method, or run single-process with GPU (no executor).
- Audit all remaining experiment scripts (`run_exp*.py`) for the same GPU
  assignment pattern before launching.

### Impact on Results

P5 is exploratory (not confirmatory). Seeds 0–14 reuse existing campaign data.
Seeds 15–29 must be run fresh. With the CPU fix, each 500-epoch run takes
approximately the same wall time as the original campaign runs (~86 min/run).
At 5 workers: 30 runs × 2 conditions ÷ 5 workers ≈ 12–14 hours total.

---

## Audit: GPU Assignment in Remaining Experiment Scripts

The following scripts must be checked for the CUDA deadlock pattern before launch:

| Script | GPU assignment present? | Status |
|---|---|---|
| `run_p2_rerun.py` | No | Safe |
| `run_p5_rerun.py` | Yes → **patched** | Fixed 2026-03-07 |
| `run_exp3_metastability.py` | Needs audit | Pending |
| `run_exp11_saturation.py` | Needs audit | Pending |
| `run_exp12_irreversibility.py` | Needs audit | Pending |
| `run_exp4_noise.py` | Needs audit | Pending |
| `run_exp7_topology.py` | Needs audit | Pending |
| `run_exp8_memory.py` | Needs audit | Pending |
| `run_exp9_observability.py` | Needs audit | Pending |
| `run_exp13_redistribution.py` | Needs audit | Pending |
| `run_exp14_coupling.py` | Needs audit | Pending |
| `run_exp15_ablation.py` | Needs audit | Pending |
| `run_exp1_tax_sweep.py` | Needs audit | Pending |

**Action required**: Before launching any experiment after P5, audit and patch
GPU assignment blocks in all remaining scripts.

---

## Environment Reference

| Item | Value |
|---|---|
| OS | Windows 10 (10.0.19045) |
| GPUs | 2× RTX 3060 |
| Correct interpreter | `C:/Users/btisl/miniconda3/python.exe` |
| Python version | 3.11 |
| PyTorch | 2.6.0+cu124 |
| System `python` | Python 3.14 (no torch — do not use) |
| Multiprocessing method | `spawn` (Windows default — no fork) |
| ProcessPoolExecutor worker behaviour | CUDA init deadlocks in spawned workers |


**Update 2026-03-07**: All 13 scripts patched. grep returns empty.
