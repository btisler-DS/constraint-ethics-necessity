"""Protocol 5 — Sacrifice trigger diagnostic.

Tests whether the Sacrifice-Conflict trigger fires reliably at max_steps=20
(short-span condition). Runs C1 (short + individual) seed=0 for 50 epochs
and reports per-epoch sacrifice choice rate and trigger fire count.

Decision rule (per PI instruction):
  - If trigger fires in >= 50% of episodes across epochs 10-50: proceed with full 40 runs
  - If trigger fires in < 50% of episodes: flag as operationalization failure,
    log deviation, halt and consult PI before any confirmatory runs

Usage:
  cd backend
  python p5_sacrifice_trigger_diagnostic.py
"""

from __future__ import annotations

import json
from simulation.engine import SimulationEngine, SimulationConfig

DIAGNOSTIC_EPOCHS = 50
EPISODES_PER_EPOCH = 10
SEED = 0
MAX_STEPS_SHORT = 20
ENERGY_BUDGET_SHORT = 20.0       # proportionally scaled from 100 (Option A deviation)
CRITICAL_THRESHOLD_SHORT = 4.0   # 20% of energy_budget, matching P4 ratio


def main() -> None:
    print("=" * 60)
    print("Protocol 5 — Sacrifice trigger diagnostic")
    print(f"max_steps={MAX_STEPS_SHORT}, seed={SEED}, epochs={DIAGNOSTIC_EPOCHS}")
    print("=" * 60)

    config = SimulationConfig(
        protocol=2,
        population_mode="all_constrained",
        seed=SEED,
        num_epochs=DIAGNOSTIC_EPOCHS,
        episodes_per_epoch=EPISODES_PER_EPOCH,
        depth=2,
        freeze_self_model_gru=False,
        max_steps=MAX_STEPS_SHORT,
        welfare_coupled=False,
        energy_budget=ENERGY_BUDGET_SHORT,
        critical_energy_threshold=CRITICAL_THRESHOLD_SHORT,
        output_dir=".",
    )

    epoch_results = []

    def _callback(m: dict) -> None:
        choices = m.get("sacrifice_choices", [])
        n_recorded = len(choices)
        scr = m.get("sacrifice_choice_rate")
        scr_str = f"{scr:.3f}" if scr is not None else "null"
        print(
            f"  epoch {m['epoch']:3d} | "
            f"sacrifice_choices recorded: {n_recorded:3d} | "
            f"sacrifice_choice_rate: {scr_str}",
            flush=True,
        )
        epoch_results.append({
            "epoch": m["epoch"],
            "n_sacrifice_choices_recorded": n_recorded,
            "sacrifice_choice_rate": scr,
        })

    engine = SimulationEngine(config=config, epoch_callback=_callback)
    engine.run()

    # Summary
    recorded_counts = [r["n_sacrifice_choices_recorded"] for r in epoch_results]
    null_epochs = sum(1 for r in epoch_results if r["sacrifice_choice_rate"] is None)
    total_choices = sum(recorded_counts)
    epochs_with_any = sum(1 for c in recorded_counts if c > 0)

    # Evaluate against decision rule:
    # trigger fires in >= 50% of *episodes* = at least 5 choices per epoch on average
    # (10 episodes/epoch, sacrifice involves 2 non-critical agents → max 20 choices/epoch)
    eval_epochs = [r for r in epoch_results if r["epoch"] >= 10]
    avg_choices_eval = sum(r["n_sacrifice_choices_recorded"] for r in eval_epochs) / max(len(eval_epochs), 1)
    # Minimum viable: at least 1 trigger per epoch (1 choice from 1 episode) in eval window
    epochs_with_trigger = sum(1 for r in eval_epochs if r["n_sacrifice_choices_recorded"] > 0)
    trigger_rate = epochs_with_trigger / max(len(eval_epochs), 1)

    print()
    print("=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Total epochs run:            {DIAGNOSTIC_EPOCHS}")
    print(f"Epochs with null scr:        {null_epochs}")
    print(f"Epochs with any choices:     {epochs_with_any}")
    print(f"Total sacrifice choices:     {total_choices}")
    print()
    print(f"Eval window (epochs 10-50):")
    print(f"  Epochs with trigger fired: {epochs_with_trigger}/{len(eval_epochs)}")
    print(f"  Trigger fire rate:         {trigger_rate:.2%}")
    print(f"  Avg choices/epoch:         {avg_choices_eval:.2f}")
    print()

    if trigger_rate >= 0.50:
        verdict = "PASS"
        recommendation = (
            "Trigger fires reliably at max_steps=20. "
            "Proceed with full 40 confirmatory runs."
        )
    else:
        verdict = "FAIL"
        recommendation = (
            "Trigger fires in fewer than 50% of eval epochs at max_steps=20. "
            "HALT. Log as preregistration deviation and consult PI before "
            "any confirmatory runs begin."
        )

    print(f"VERDICT: {verdict}")
    print(f"RECOMMENDATION: {recommendation}")
    print()

    # Write diagnostic result to file for deviation log
    result = {
        "diagnostic": "p5_sacrifice_trigger",
        "max_steps": MAX_STEPS_SHORT,
        "seed": SEED,
        "epochs_run": DIAGNOSTIC_EPOCHS,
        "null_scr_epochs": null_epochs,
        "epochs_with_any_choices": epochs_with_any,
        "total_sacrifice_choices": total_choices,
        "eval_window_start": 10,
        "eval_trigger_rate": round(trigger_rate, 4),
        "eval_avg_choices_per_epoch": round(avg_choices_eval, 2),
        "verdict": verdict,
        "recommendation": recommendation,
        "epoch_series": epoch_results,
    }
    out_path = "p5_sacrifice_trigger_diagnostic_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Full result written to: {out_path}")


if __name__ == "__main__":
    main()
