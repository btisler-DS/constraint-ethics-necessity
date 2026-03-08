"""Experiment 15 — Protocol ablation taxonomy.

Tests which components of Protocol 1 are genuinely load-bearing by ablating
them individually. A load-bearing component causes performance/QRC collapse
when removed; decorative chatter does not.

Ablation conditions
-------------------
  baseline     — Protocol 1, q=1.5, r=0.8  (control)
  type_ablated — P1 costs (q=1.5, r=0.8), but hard_type forced to 0 (DECLARE)
                 regardless of type_head output. Soft_type preserved for
                 gradient flow. Tests: are the TYPE SEMANTICS load-bearing?
  signal_ablated — P1 costs + types, but signal vector zeroed out in comm
                 buffer. Type structure preserved. Tests: is SIGNAL CONTENT
                 necessary, or does typed-empty-message coordination suffice?
  cost_ablated — P1 type structure active, but query_cost=0 and respond_cost=0
                 (declare_cost=1.0 only). No incentive for non-DECLARE types.
                 Tests: does the COST STRUCTURE drive crystallization, or do
                 agents use type semantics even without cost pressure?
  p0_control   — Protocol 0: type=DECLARE always, costs=0. Clean no-protocol
                 baseline to confirm cost_ablated ≠ p0.

  5 conditions × 15 seeds = 75 runs.

Predictions
-----------
  If the protocol is genuinely structural:
    type_ablated  → QRC collapse (semantics are load-bearing)
    signal_ablated → partial collapse or slower crystallization (content helps)
    cost_ablated  → crystallization fails (costs drive the phase transition)
    p0_control    → no crystallization (known from campaign)

  If the protocol is decorative:
    Most ablations leave QRC intact because agents have workarounds.

Output
------
  data/exp15_ablation/seed_{s:02d}_{condition}/manifest.json
  data/exp15_ablation/exp15_summary.json

Usage
-----
  python run_exp15_ablation.py
  python run_exp15_ablation.py --workers 4
  python run_exp15_ablation.py --dry-run
  python run_exp15_ablation.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SEEDS        = list(range(15))
EPOCHS       = 500
EPISODES     = 10
DECLARE_COST = 1.0

CONDITIONS = [
    {"label": "baseline",       "protocol": 1, "query_cost": 1.5, "respond_cost": 0.8,
     "ablation": None},
    {"label": "type_ablated",   "protocol": 1, "query_cost": 1.5, "respond_cost": 0.8,
     "ablation": "type"},
    {"label": "signal_ablated", "protocol": 1, "query_cost": 1.5, "respond_cost": 0.8,
     "ablation": "signal"},
    {"label": "cost_ablated",   "protocol": 1, "query_cost": 0.0, "respond_cost": 0.0,
     "ablation": None},
    {"label": "p0_control",     "protocol": 0, "query_cost": 0.0, "respond_cost": 0.0,
     "ablation": None},
]

DATA_ROOT = Path(__file__).parent.parent / "data" / "exp15_ablation"


# ── Worker ─────────────────────────────────────────────────────────────────────

def _run_one(spec: dict) -> dict:
    """Run one seed under one ablation condition. Called in subprocess."""
    import sys, os
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from simulation.engine import SimulationEngine, SimulationConfig

    config = SimulationConfig(
        seed=spec["seed"],
        num_epochs=EPOCHS,
        episodes_per_epoch=EPISODES,
        protocol=spec["protocol"],
        declare_cost=DECLARE_COST,
        query_cost=spec["query_cost"],
        respond_cost=spec["respond_cost"],
        output_dir=spec["output_dir"],
        device=spec.get("device", "cpu"),
    )
    engine = SimulationEngine(config=config)

    if spec["ablation"] == "type":
        # Force hard_type=0 (DECLARE) regardless of type_head output.
        # soft_type is preserved so gradient flows through type_head normally.
        original_resolve = engine.protocol.resolve_signal_type

        def type_ablated_resolve(type_logits, tau, training=True):
            soft_type, _ = original_resolve(type_logits, tau, training=training)
            return soft_type, 0   # always DECLARE

        engine.protocol.resolve_signal_type = type_ablated_resolve

    elif spec["ablation"] == "signal":
        # Zero out signal content in comm buffer while preserving type structure.
        original_send = engine.comm_buffer.send

        def zero_signal_send(agent_name: str, signal, signal_type: int = 0) -> None:
            original_send(agent_name, torch.zeros_like(signal), signal_type=signal_type)

        engine.comm_buffer.send = zero_signal_send

    engine.run()
    return {"seed": spec["seed"], "condition": spec["label"]}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_manifest(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _extract(manifest: dict) -> dict:
    fm = manifest.get("final_metrics", manifest)
    return {
        "crystallized":   manifest.get("crystallized", fm.get("crystallized")),
        "crys_epoch":     manifest.get("crystallization_epoch",
                                       fm.get("crystallization_epoch")),
        "type_entropy":   fm.get("type_entropy"),
        "qrc":            fm.get("query_response_coupling"),
        "query_rate":     fm.get("query_rate"),
        "respond_rate":   fm.get("respond_rate"),
        "survival_rate":  fm.get("survival_rate"),
        "energy_roi":     fm.get("energy_roi"),
    }


# ── Campaign runner ─────────────────────────────────────────────────────────────

def run_campaign(seeds: list[int], workers: int, dry_run: bool) -> None:
    jobs = []
    for cond in CONDITIONS:
        for seed in seeds:
            out_dir = DATA_ROOT / f"seed_{seed:02d}_{cond['label']}"
            if (out_dir / "manifest.json").exists():
                continue
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
            jobs.append({
                "seed":         seed,
                "condition":    cond["label"],
                "protocol":     cond["protocol"],
                "query_cost":   cond["query_cost"],
                "respond_cost": cond["respond_cost"],
                "ablation":     cond["ablation"],
                "output_dir":   str(out_dir),
            })

    if not jobs:
        print("  No new jobs (all data present).")
        return

    print(f"\n  Running {len(jobs)} jobs  (workers={workers})")
    if dry_run:
        for j in jobs[:5]:
            print(f"  DRY-RUN: {j['condition']}  seed={j['seed']}")
        if len(jobs) > 5:
            print(f"  ... ({len(jobs)-5} more)")
        return

    t0 = time.time()
    done = 0
    # Force CPU: CUDA multi-process spawning deadlocks on Windows with dual GPUs
    for _j in jobs:
        _j["device"] = "cpu"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                done += 1
                print(f"  [{done:3d}/{len(jobs)}]  {r['condition']}  seed={r['seed']:02d}")
            except Exception as e:
                j = futures[fut]
                print(f"  ERROR: {j['condition']}  seed={j['seed']}: {e}")
    print(f"  Done in {time.time()-t0:.0f}s")


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(seeds: list[int]) -> dict:
    per_condition: dict[str, list[dict]] = {c["label"]: [] for c in CONDITIONS}

    for cond in CONDITIONS:
        for seed in seeds:
            out_dir = DATA_ROOT / f"seed_{seed:02d}_{cond['label']}"
            m = _load_manifest(out_dir / "manifest.json")
            if m is None:
                continue
            d = _extract(m)
            d["seed"] = seed
            d["condition"] = cond["label"]
            per_condition[cond["label"]].append(d)

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    condition_stats = {}
    for cond in CONDITIONS:
        label  = cond["label"]
        rows   = per_condition[label]
        n      = len(rows)
        n_crys = sum(1 for r in rows if r.get("crystallized"))
        condition_stats[label] = {
            "protocol":             cond["protocol"],
            "query_cost":           cond["query_cost"],
            "respond_cost":         cond["respond_cost"],
            "ablation":             cond["ablation"],
            "n":                    n,
            "n_crystallized":       n_crys,
            "crystallization_rate": round(n_crys / n, 4) if n else None,
            "mean_qrc":             _mean([r["qrc"] for r in rows]),
            "mean_type_entropy":    _mean([r["type_entropy"] for r in rows]),
            "mean_query_rate":      _mean([r["query_rate"] for r in rows]),
            "mean_respond_rate":    _mean([r["respond_rate"] for r in rows]),
            "mean_survival_rate":   _mean([r["survival_rate"] for r in rows]),
            "mean_crys_epoch":      _mean([r["crys_epoch"] for r in rows
                                           if r.get("crystallized") and r["crys_epoch"]]),
        }

    # Collapse scores relative to baseline
    baseline_qrc  = condition_stats["baseline"]["mean_qrc"]  or 1.0
    baseline_rate = condition_stats["baseline"]["crystallization_rate"] or 1.0

    for label, cs in condition_stats.items():
        qrc  = cs["mean_qrc"]
        rate = cs["crystallization_rate"]
        cs["qrc_relative_to_baseline"]  = round(qrc  / baseline_qrc,  4) if qrc  else None
        cs["rate_relative_to_baseline"] = round(rate / baseline_rate, 4) if rate else None

    return {
        "experiment":     "15",
        "description":    "Protocol ablation taxonomy",
        "generated":      datetime.now().isoformat(),
        "n_seeds":        len(seeds),
        "conditions":     condition_stats,
    }


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_summary(result: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nExp 15 summary: {out_path}")

    print(f"\n  {'Condition':<16}  {'abl':>8}  {'n':>3}  {'rate':>5}  "
          f"{'QRC':>6}  {'rate_rel':>8}  {'qrc_rel':>8}  {'H':>6}  {'Qrate':>6}")
    for cond in CONDITIONS:
        label = cond["label"]
        cs    = result["conditions"][label]
        abl   = cond["ablation"] or "none"
        rate  = f"{cs['crystallization_rate']:.2f}" if cs["crystallization_rate"] is not None else " N/A"
        qrc   = f"{cs['mean_qrc']:.3f}"             if cs["mean_qrc"]             is not None else " N/A"
        rr    = f"{cs['rate_relative_to_baseline']:.2f}" if cs["rate_relative_to_baseline"] is not None else " N/A"
        qr    = f"{cs['qrc_relative_to_baseline']:.2f}"  if cs["qrc_relative_to_baseline"]  is not None else " N/A"
        H     = f"{cs['mean_type_entropy']:.3f}"    if cs["mean_type_entropy"]    is not None else " N/A"
        qrate = f"{cs['mean_query_rate']:.3f}"      if cs["mean_query_rate"]       is not None else " N/A"
        print(f"  {label:<16}  {abl:>8}  {cs['n']:>3}  {rate:>5}  "
              f"{qrc:>6}  {rr:>8}  {qr:>8}  {H:>6}  {qrate:>6}")

    print("\n  Interpretation guide:")
    print("  rate_rel < 0.5 → ablation collapses crystallization (load-bearing)")
    print("  qrc_rel  < 0.5 → ablation collapses QRC (load-bearing)")
    print("  rate_rel ~ 1.0 → ablation has little effect (decorative)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 15 -- Protocol ablation taxonomy")
    parser.add_argument("--seeds",        type=int, nargs=2, default=[0, 15],
                        metavar=("START", "END"))
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    seeds   = list(range(args.seeds[0], args.seeds[1]))
    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    print(f"Exp 15 -- Protocol ablation  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}  Workers: {workers}")
    print(f"Conditions: {len(CONDITIONS)}  x  {len(seeds)} seeds = {len(CONDITIONS)*len(seeds)} runs")
    print(f"Data: {DATA_ROOT}")

    if args.analyze_only:
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp15_summary.json")
        return

    if not args.dry_run:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

    run_campaign(seeds, workers, args.dry_run)

    if not args.dry_run:
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp15_summary.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
