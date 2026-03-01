"""Harvest script — aggregates completion status across all Protocol 2 experiments.

Reads all available manifest.json files across all experiment directories and
produces a single status report showing which experiments have completed, which
are still running, and what key results are available so far.

Usage
-----
  python run_harvest.py             # print status report
  python run_harvest.py --watch     # refresh every 60 seconds until all done
  python run_harvest.py --out FILE  # also write JSON summary

Output
------
  Console: per-experiment completion table + partial results
  Optional: data/harvest_status.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

# ── Experiment registry ────────────────────────────────────────────────────────
# Each entry: {name, dir, expected_runs, conditions, metric_keys}

EXPERIMENTS = [
    # Confirmatory reruns
    {
        "name":          "P2 rerun",
        "dir":           "p2_rerun",
        "manifest_glob": "**/seed_*/manifest.json",
        "expected_runs": 45,   # 3 conditions × 15 seeds
        "key":           "p2_rerun",
        "note":          "P2 cost-sensitivity rerun; need r < -0.70 and p < 0.01",
    },
    {
        "name":          "P5 rerun",
        "dir":           "p5_rerun",
        "manifest_glob": "*/seed_*/manifest.json",
        "expected_runs": 30,   # 2 conditions × 15 seeds (seeds 15-29)
        "key":           "p5_rerun",
        "note":          "P5 survival diff rerun; exploratory n=30",
    },
    # High priority
    {
        "name":          "Exp 3 metastability",
        "dir":           "exp3_metastability",
        "manifest_glob": "*/seed_*/manifest.json",
        "expected_runs": 45,
        "key":           "exp3",
        "note":          "Square-wave tax; 3 cond x 15 seeds",
    },
    {
        "name":          "Exp 11 saturation",
        "dir":           "exp11_saturation",
        "manifest_glob": "*/seed_*/manifest.json",
        "expected_runs": 60,
        "key":           "exp11",
        "note":          "Saturation forcing; 4 cond x 15 seeds",
    },
    {
        "name":          "Exp 12 irreversibility",
        "dir":           "exp12_irreversibility",
        "manifest_glob": "seed_*/manifest.json",
        "expected_runs": 15,
        "key":           "exp12",
        "note":          "Three-phase irreversibility; 15 seeds",
    },
    # Medium priority
    {
        "name":          "Exp 1 tax sweep",
        "dir":           "exp1_tax_sweep",
        "manifest_glob": "seed_*/manifest.json",
        "expected_runs": 150,
        "key":           "exp1",
        "note":          "Fine q sweep 0.1-1.0; 10 cond x 15 seeds",
    },
    {
        "name":          "Exp 4 noise",
        "dir":           "exp4_noise",
        "manifest_glob": "seed_*/manifest.json",
        "expected_runs": 120,
        "key":           "exp4",
        "note":          "Noise at boundary; 8 cond x 15 seeds",
    },
    {
        "name":          "Exp 7 topology",
        "dir":           "exp7_topology",
        "manifest_glob": "seed_*/manifest.json",
        "expected_runs": 45,
        "key":           "exp7",
        "note":          "Topology (broadcast/chain/star); 3 cond x 15 seeds",
    },
    {
        "name":          "Exp 8 memory depth",
        "dir":           "exp8_memory_depth",
        "manifest_glob": "seed_*/manifest.json",
        "expected_runs": 75,
        "key":           "exp8",
        "note":          "hidden_dim sweep 4-64; 5 cond x 15 seeds",
    },
    {
        "name":          "Exp 9 observability",
        "dir":           "exp9_observability",
        "manifest_glob": "seed_*/manifest.json",
        "expected_runs": 75,
        "key":           "exp9",
        "note":          "Obs mask 0-90%; 5 cond x 15 seeds",
    },
    {
        "name":          "Exp 13 redistribution",
        "dir":           "exp13_redistribution",
        "manifest_glob": "seed_*/manifest.json",
        "expected_runs": 75,
        "key":           "exp13",
        "note":          "Cost redistribution (total=3.3); 5 cond x 15 seeds",
    },
    {
        "name":          "Exp 14 coupling window",
        "dir":           "exp14_coupling_window",
        "manifest_glob": "seed_*/manifest.json",
        "expected_runs": 135,
        "key":           "exp14",
        "note":          "Query/respond cost sweep; 9 cond x 15 seeds",
    },
    {
        "name":          "Exp 15 ablation",
        "dir":           "exp15_ablation",
        "manifest_glob": "seed_*/manifest.json",
        "expected_runs": 75,
        "key":           "exp15",
        "note":          "Protocol ablation; 5 cond x 15 seeds",
    },
    # Ant modules (already complete)
    {
        "name":          "Exp A (ant delta)",
        "dir":           "ant_experiments/exp_a",
        "manifest_glob": "*/*/manifest.json",
        "expected_runs": 210,
        "key":           "exp_a",
        "note":          "Ant pheromone foraging delta sweep; COMPLETE",
    },
    {
        "name":          "Exp B (ant bridge)",
        "dir":           "ant_experiments/exp_b",
        "manifest_glob": "*/manifest.json",
        "expected_runs": 30,
        "key":           "exp_b",
        "note":          "Ant bridge hysteresis; COMPLETE",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_manifests(exp_dir: Path, glob: str) -> int:
    return len(list(exp_dir.glob(glob)))


def _is_crystallized(m: dict) -> bool | None:
    val = m.get("crystallized")
    if val is not None:
        return bool(val)
    fm = m.get("final_metrics", {})
    val = fm.get("crystallized")
    if val is not None:
        return bool(val)
    crys_epoch = m.get("crystallization_epoch")
    if crys_epoch is not None:
        return True
    if "crystallization_epoch" in m:
        return False
    return None


def _quick_stats(exp_dir: Path, glob: str) -> dict:
    """Read all available manifests and compute quick stats."""
    manifests = list(exp_dir.glob(glob))
    if not manifests:
        return {}

    crys_vals = []
    qrc_vals  = []
    for p in manifests:
        try:
            with open(p) as f:
                m = json.load(f)
        except Exception:
            continue
        c = _is_crystallized(m)
        if c is not None:
            crys_vals.append(float(c))
        fm = m.get("final_metrics", {})
        qrc = fm.get("query_response_coupling") or fm.get("qrc")
        if qrc is not None:
            qrc_vals.append(float(qrc))

    def _m(xs):
        return round(sum(xs)/len(xs), 4) if xs else None

    return {
        "n_manifests":  len(manifests),
        "crys_rate":    _m(crys_vals),
        "mean_qrc":     _m(qrc_vals),
    }


# ── Main report ────────────────────────────────────────────────────────────────

def harvest(watch_interval: int | None = None, out_file: Path | None = None) -> None:
    while True:
        now = datetime.now().isoformat(timespec="seconds")
        print(f"\n{'='*70}")
        print(f"Harvest report  {now}")
        print(f"{'='*70}")

        rows = []
        total_done = 0
        total_expected = 0

        for exp in EXPERIMENTS:
            exp_dir  = DATA / exp["dir"]
            done     = _count_manifests(exp_dir, exp["manifest_glob"]) if exp_dir.exists() else 0
            expected = exp["expected_runs"]
            pct      = round(100 * done / expected) if expected else 0
            stats    = _quick_stats(exp_dir, exp["manifest_glob"]) if done > 0 else {}

            total_done     += done
            total_expected += expected

            status = "COMPLETE" if done >= expected else (f"{pct:>3}%" if done > 0 else "WAITING")
            rows.append({
                "name":     exp["name"],
                "done":     done,
                "expected": expected,
                "status":   status,
                "stats":    stats,
                "note":     exp["note"],
            })

        # Print table
        print(f"\n  {'Experiment':<26}  {'done':>5}  {'of':>5}  {'status':>8}  "
              f"{'crys_rate':>9}  {'QRC':>6}")
        for r in rows:
            cr  = f"{r['stats']['crys_rate']:.3f}" if r['stats'].get('crys_rate') is not None else "   N/A"
            qrc = f"{r['stats']['mean_qrc']:.3f}"  if r['stats'].get('mean_qrc')  is not None else "   N/A"
            print(f"  {r['name']:<26}  {r['done']:>5}  {r['expected']:>5}  "
                  f"{r['status']:>8}  {cr:>9}  {qrc:>6}")

        pct_total = round(100 * total_done / total_expected) if total_expected else 0
        print(f"\n  TOTAL: {total_done}/{total_expected} manifests ({pct_total}%)")

        if out_file:
            with open(out_file, "w") as f:
                json.dump({
                    "generated": now,
                    "total_done": total_done,
                    "total_expected": total_expected,
                    "experiments": rows,
                }, f, indent=2, default=str)
            print(f"  Written: {out_file}")

        if watch_interval is None:
            break
        all_complete = all(r["done"] >= r["expected"] for r in rows)
        if all_complete:
            print("\n  All experiments complete!")
            break
        print(f"\n  Refreshing in {watch_interval}s ...  (Ctrl-C to stop)")
        time.sleep(watch_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest: status report across all Protocol 2 experiments")
    parser.add_argument("--watch", type=int, nargs="?", const=60, default=None,
                        metavar="SECS",
                        help="Auto-refresh interval in seconds (default 60)")
    parser.add_argument("--out",   type=str, default=None,
                        help="Write JSON summary to this file")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else DATA / "harvest_status.json"
    harvest(watch_interval=args.watch, out_file=out_path)


if __name__ == "__main__":
    main()
