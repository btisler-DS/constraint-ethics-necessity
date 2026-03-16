"""Protocol 5 — Confirmatory runs.

10 seeds × 4 conditions × 500 epochs = 40 runs.

Conditions (2×2 factorial: temporal span × constraint structure):
  c1_short_individual   — max_steps=20, individual reward  (Protocol 4 baseline replica)
  c2_short_welfare      — max_steps=20, welfare-coupled (alpha=0.5)
  c3_long_individual    — max_steps=64, individual reward
  c4_long_welfare       — max_steps=64, welfare-coupled (alpha=0.5)

All conditions: Depth 2 (self_model_gru trainable), Protocol 2 constraint pipeline,
population_mode=all_constrained. No architectural changes from Protocol 4.

Boundary condition sub-study (H4): re-run c1 and c4 with freeze_self_model_gru=True
using the --boundary flag. These 20 additional runs are not part of the primary 40.

Output: results/p5_condition_{condition_name}_seed_{seed}.json

Usage:
  python run_p5_confirmatory.py                        # all 40 primary runs
  python run_p5_confirmatory.py c4_long_welfare 3      # single condition/seed
  python run_p5_confirmatory.py --boundary             # 20 boundary sub-study runs
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

from simulation.engine import SimulationEngine, SimulationConfig

# ---------------------------------------------------------------------------
# Condition definitions — DO NOT MODIFY
# ---------------------------------------------------------------------------
# Deviation note: energy_budget scaled proportionally to max_steps for short-span
# conditions (20/100 × 100 = 20) and critical_energy_threshold scaled accordingly
# (20% of budget). Identified via sacrifice trigger diagnostic before any runs began.
# Long-span conditions retain Protocol 4 baseline energy parameters.
CONDITIONS = {
    "c1_short_individual": dict(max_steps=20, welfare_coupled=False, energy_budget=20.0, critical_energy_threshold=4.0),
    "c2_short_welfare":    dict(max_steps=20, welfare_coupled=True,  energy_budget=20.0, critical_energy_threshold=4.0),
    "c3_long_individual":  dict(max_steps=64, welfare_coupled=False, energy_budget=100.0, critical_energy_threshold=20.0),
    "c4_long_welfare":     dict(max_steps=64, welfare_coupled=True,  energy_budget=100.0, critical_energy_threshold=20.0),
}

# Boundary condition sub-study: frozen self_model_gru in c1 and c4 only (tests H4)
BOUNDARY_CONDITIONS = {
    "c1_short_individual_frozen": dict(max_steps=20, welfare_coupled=False, energy_budget=20.0, critical_energy_threshold=4.0, freeze_self_model_gru=True),
    "c4_long_welfare_frozen":     dict(max_steps=64, welfare_coupled=True,  energy_budget=100.0, critical_energy_threshold=20.0, freeze_self_model_gru=True),
}

SEEDS = list(range(10))       # 0–9
NUM_EPOCHS = 500
EPISODES_PER_EPOCH = 10
RESULTS_DIR = "results"


def run_one(condition_name: str, seed: int, cond: dict) -> str:
    """Run one condition/seed pair. Returns output path."""
    output_name = f"p5_condition_{condition_name}_seed_{seed}"
    tmp_dir = os.path.join(RESULTS_DIR, f"_tmp_{output_name}")
    final_path = os.path.join(RESULTS_DIR, f"{output_name}.json")

    config = SimulationConfig(
        protocol=2,
        population_mode="all_constrained",
        seed=seed,
        num_epochs=NUM_EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        output_dir=tmp_dir,
        depth=2,
        freeze_self_model_gru=cond.get("freeze_self_model_gru", False),
        max_steps=cond["max_steps"],
        welfare_coupled=cond["welfare_coupled"],
        energy_budget=cond["energy_budget"],
        critical_energy_threshold=cond["critical_energy_threshold"],
    )

    def _print(m: dict) -> None:
        scr = m.get("sacrifice_choice_rate")
        scr_str = f" scr={scr:.3f}" if scr is not None else ""
        ssn = m.get("self_state_norm_mean")
        ssn_str = f" ssn={ssn:.3f}" if ssn is not None else ""
        print(
            f"  [{condition_name}|s{seed}] epoch {m['epoch']:3d}{scr_str}{ssn_str}",
            flush=True,
        )

    engine = SimulationEngine(config=config, epoch_callback=_print)
    engine.run()

    src = os.path.join(tmp_dir, "epoch_series.json")
    shutil.move(src, final_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return final_path


def main() -> None:
    # Boundary sub-study flag
    if len(sys.argv) == 2 and sys.argv[1] == "--boundary":
        run_batch(BOUNDARY_CONDITIONS, label="boundary sub-study", expected=20)
        return

    # Single condition/seed from argv for manual resumption
    if len(sys.argv) == 3:
        condition_name = sys.argv[1]
        seed = int(sys.argv[2])
        all_conditions = {**CONDITIONS, **BOUNDARY_CONDITIONS}
        if condition_name not in all_conditions:
            print(f"Unknown condition: {condition_name}. Valid: {list(all_conditions)}")
            sys.exit(1)
        print(f"Running single: {condition_name} seed={seed}")
        path = run_one(condition_name, seed, all_conditions[condition_name])
        print(f"Written: {path}")
        return

    # Default: all 40 primary confirmatory runs
    run_batch(CONDITIONS, label="primary confirmatory runs", expected=40)


def run_batch(conditions: dict, label: str, expected: int) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    completed = []
    failed = []
    t0 = time.time()

    for condition_name, cond in conditions.items():
        for seed in SEEDS:
            final_path = os.path.join(RESULTS_DIR, f"p5_condition_{condition_name}_seed_{seed}.json")
            if os.path.exists(final_path):
                print(f"SKIP (exists): {final_path}")
                completed.append(final_path)
                continue

            print(f"\n=== {condition_name} seed={seed} ===", flush=True)
            t1 = time.time()
            try:
                path = run_one(condition_name, seed, cond)
                elapsed = time.time() - t1
                print(f"  Done in {elapsed:.0f}s → {path}", flush=True)
                completed.append(path)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                failed.append((condition_name, seed, str(exc)))

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"{label}: {len(completed)}/{expected} in {total/60:.1f} min")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for c, s, e in failed:
            print(f"  {c} seed={s}: {e}")
        sys.exit(1)
    else:
        print(f"All {expected} runs complete.")


if __name__ == "__main__":
    main()
