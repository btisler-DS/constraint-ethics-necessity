"""Protocol 4 — Confirmatory runs.

10 seeds × 4 conditions × 500 epochs = 40 runs.

Conditions:
  baseline         — depth=0, all agents feedforward (army ant analog)
  below_threshold  — depth=1, full RNN architecture (Protocol 2 baseline)
  above_threshold  — depth=2, AgentA with self_model_gru (unablated)
  boundary         — depth=2, AgentA with self_model_gru frozen at random init

All conditions use Protocol 2 constraint pipeline (population_mode=all_constrained).
No mid-run adjustments authorized. Run script is pre-registration locked.

Output: results/p4_condition_{condition_name}_seed_{seed}.json
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
CONDITIONS = {
    "baseline": dict(depth=0, freeze_self_model_gru=False),
    "below_threshold": dict(depth=1, freeze_self_model_gru=False),
    "above_threshold": dict(depth=2, freeze_self_model_gru=False),
    "boundary": dict(depth=2, freeze_self_model_gru=True),
}

SEEDS = list(range(10))       # 0–9
NUM_EPOCHS = 500
EPISODES_PER_EPOCH = 10
RESULTS_DIR = "results"


def run_one(condition_name: str, seed: int) -> str:
    """Run one condition/seed pair. Returns output path."""
    cond = CONDITIONS[condition_name]
    output_name = f"p4_condition_{condition_name}_seed_{seed}"
    tmp_dir = os.path.join(RESULTS_DIR, f"_tmp_{output_name}")
    final_path = os.path.join(RESULTS_DIR, f"{output_name}.json")

    config = SimulationConfig(
        protocol=2,
        population_mode="all_constrained",
        seed=seed,
        num_epochs=NUM_EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        output_dir=tmp_dir,
        depth=cond["depth"],
        freeze_self_model_gru=cond["freeze_self_model_gru"],
    )

    def _print(m: dict) -> None:
        inq = m.get("inquiry", {}) or {}
        te = inq.get("type_entropy", None)
        te_str = f" H={te:.3f}" if te is not None else ""
        ssn = m.get("self_state_norm")
        ssn_str = f" ssn={ssn:.3f}" if ssn is not None else ""
        print(
            f"  [{condition_name}|s{seed}] epoch {m['epoch']:3d}{te_str}{ssn_str}",
            flush=True,
        )

    engine = SimulationEngine(config=config, epoch_callback=_print)
    engine.run()

    # Move epoch_series.json to final path
    src = os.path.join(tmp_dir, "epoch_series.json")
    shutil.move(src, final_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return final_path


def main() -> None:
    # Optional: single condition/seed from argv for manual resumption
    if len(sys.argv) == 3:
        condition_name = sys.argv[1]
        seed = int(sys.argv[2])
        if condition_name not in CONDITIONS:
            print(f"Unknown condition: {condition_name}. Valid: {list(CONDITIONS)}")
            sys.exit(1)
        print(f"Running single: {condition_name} seed={seed}")
        path = run_one(condition_name, seed)
        print(f"Written: {path}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    completed = []
    failed = []
    t0 = time.time()

    for condition_name in CONDITIONS:
        for seed in SEEDS:
            final_path = os.path.join(RESULTS_DIR, f"p4_condition_{condition_name}_seed_{seed}.json")
            if os.path.exists(final_path):
                print(f"SKIP (exists): {final_path}")
                completed.append(final_path)
                continue

            print(f"\n=== {condition_name} seed={seed} ===", flush=True)
            t1 = time.time()
            try:
                path = run_one(condition_name, seed)
                elapsed = time.time() - t1
                print(f"  Done in {elapsed:.0f}s → {path}", flush=True)
                completed.append(path)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                failed.append((condition_name, seed, str(exc)))

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Completed: {len(completed)}/40 in {total/60:.1f} min")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for c, s, e in failed:
            print(f"  {c} seed={s}: {e}")
        sys.exit(1)
    else:
        print("All 40 runs complete.")


if __name__ == "__main__":
    main()
