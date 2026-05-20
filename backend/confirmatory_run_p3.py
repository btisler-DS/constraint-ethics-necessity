"""Protocol 3 — Confirmatory runs.

10 seeds × 3 conditions × 500 epochs = 30 runs.

Conditions:
  p3_unconstrained  — no ethical tax (control)
  p3b_constrained   — hidden schedule enforcement (primary condition)
  p3a_constrained   — stochastic penalty p=0.5 (comparator condition)

Run order follows preregistration: unconstrained → p3b → p3a.

All conditions: Protocol 3 constraint pipeline, depth=1, welfare_coupled=False.
Parameters locked per preregistration DOI: 10.5281/zenodo.19096602.

Penalty schedule integrity check: before any p3b run, the schedule generated from
PENALTY_SCHEDULE_SEED=20260318 must reproduce SHA-256:
    10df29597e296455a1b72bb5328642db7702ffb611ff2e8c83c9548280fad2e4

Output: data/p3_{condition}/seed_{seed}/manifest.json + epoch_series.json

Usage:
  python confirmatory_run_p3.py                        # all 30 runs in preregistered order
  python confirmatory_run_p3.py p3b_constrained 3      # single condition/seed
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

from simulation.engine import SimulationEngine, SimulationConfig

# ---------------------------------------------------------------------------
# Locked parameters — DO NOT MODIFY
# ---------------------------------------------------------------------------
SEEDS = list(range(10))
NUM_EPOCHS = 500
EPISODES_PER_EPOCH = 10
PENALTY_PROBABILITY = 0.5
PENALTY_EPOCH_FRACTION = 0.5
PENALTY_SCHEDULE_SEED = 20260318
EXPECTED_SCHEDULE_HASH = "10df29597e296455a1b72bb5328642db7702ffb611ff2e8c83c9548280fad2e4"

# Run order matches preregistration: control first, then primary, then comparator
CONDITIONS = [
    "p3_unconstrained",
    "p3b_constrained",
    "p3a_constrained",
]


# ---------------------------------------------------------------------------
# Schedule integrity verification
# ---------------------------------------------------------------------------

def verify_schedule_hash() -> None:
    """Verify the 3B penalty schedule reproduces the preregistered SHA-256.

    Must pass before any p3b_constrained run proceeds.
    Raises RuntimeError if hash does not match.
    """
    rng = np.random.RandomState(PENALTY_SCHEDULE_SEED)
    draws = rng.random(NUM_EPOCHS)
    penalty_epochs = sorted(int(i) for i in np.where(draws < PENALTY_EPOCH_FRACTION)[0])
    epoch_bytes = json.dumps(penalty_epochs).encode("utf-8")
    actual_hash = hashlib.sha256(epoch_bytes).hexdigest()
    if actual_hash != EXPECTED_SCHEDULE_HASH:
        raise RuntimeError(
            f"SCHEDULE HASH MISMATCH — run aborted.\n"
            f"  Expected: {EXPECTED_SCHEDULE_HASH}\n"
            f"  Got:      {actual_hash}\n"
            f"The penalty schedule does not match the preregistered set. "
            f"Do not proceed with p3b_constrained runs."
        )
    print(f"Schedule hash verified: {actual_hash}", flush=True)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(condition: str, seed: int) -> str:
    """Run one condition/seed pair. Returns output directory path."""
    output_dir = os.path.join("data", condition, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    # Skip if already complete
    if os.path.exists(os.path.join(output_dir, "epoch_series.json")):
        print(f"SKIP (exists): {output_dir}", flush=True)
        return output_dir

    config = SimulationConfig(
        protocol=3,
        population_mode=condition,
        seed=seed,
        num_epochs=NUM_EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        output_dir=output_dir,
        depth=1,
        welfare_coupled=False,
        penalty_probability=PENALTY_PROBABILITY,
        penalty_epoch_fraction=PENALTY_EPOCH_FRACTION,
    )

    def _print(m: dict) -> None:
        inq = m.get("inquiry", {}) or {}
        te_str = f" H={inq['type_entropy']:.3f}" if "type_entropy" in inq else ""
        ec = m.get("ethical_constraint", {}) or {}
        pf = ec.get("penalty_fired")
        pf_str = f" [PF]" if pf else ""
        qr = (inq.get("type_distribution") or {}).get("QUERY")
        qr_str = f" qr={qr:.3f}" if qr is not None else ""
        print(
            f"  [{condition}|s{seed}] epoch {m['epoch']:3d}{qr_str}{te_str}{pf_str}",
            flush=True,
        )

    engine = SimulationEngine(config=config, epoch_callback=_print)
    engine.run()
    return output_dir


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(conditions: list[str]) -> None:
    schedule_verified = False
    completed = []
    failed = []
    t0 = time.time()

    for condition in conditions:
        # Verify schedule before first p3b run
        if condition == "p3b_constrained" and not schedule_verified:
            verify_schedule_hash()
            schedule_verified = True

        for seed in SEEDS:
            output_dir = os.path.join("data", condition, f"seed_{seed}")
            if os.path.exists(os.path.join(output_dir, "epoch_series.json")):
                print(f"SKIP (exists): {output_dir}", flush=True)
                completed.append(output_dir)
                continue

            print(f"\n=== {condition} seed={seed} ===", flush=True)
            t1 = time.time()
            try:
                path = run_one(condition, seed)
                elapsed = time.time() - t1
                print(f"  Done in {elapsed:.0f}s -> {path}", flush=True)
                completed.append(path)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                failed.append((condition, seed, str(exc)))

    total = time.time() - t0
    expected = len(conditions) * len(SEEDS)
    print(f"\n{'='*60}")
    print(f"Protocol 3 confirmatory runs: {len(completed)}/{expected} in {total/60:.1f} min")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for c, s, e in failed:
            print(f"  {c} seed={s}: {e}")
        sys.exit(1)
    else:
        print(f"All {expected} runs complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Single condition/seed for manual resumption
    if len(sys.argv) == 3:
        condition = sys.argv[1]
        seed = int(sys.argv[2])
        if condition not in CONDITIONS:
            print(f"Unknown condition: {condition}. Valid: {CONDITIONS}")
            sys.exit(1)
        if condition == "p3b_constrained":
            verify_schedule_hash()
        print(f"Running single: {condition} seed={seed}")
        path = run_one(condition, seed)
        print(f"Written: {path}")
        return

    # Default: all 30 runs in preregistered order
    run_batch(CONDITIONS)


if __name__ == "__main__":
    main()
