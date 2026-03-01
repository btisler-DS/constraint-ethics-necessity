"""Experiment 13 — Redistribution does not equal resolution.

Tests whether redistributing protocol costs among action types — without
changing total cost pressure — resolves the underlying demand for structure,
or whether the structural deficit persists and resurfaces elsewhere.

Theory prediction
-----------------
  Total cost pressure (declare + query + respond summed) is the binding
  constraint. Redistributing the same total budget differently (e.g.,
  cheaper queries but more expensive declares, keeping total ~ constant)
  should not fundamentally alter whether crystallization occurs — the
  deficit is in structural capacity, not in cost allocation alone.

  If crystallization rate and QRC are insensitive to redistribution (holding
  total cost constant), this supports the claim that TOTAL PRESSURE drives
  the phase transition, not its allocation.

  If redistribution produces qualitatively different outcomes, it would
  suggest cost allocation (not just total cost) is the relevant variable.

Design
------
  Baseline total "pressure": d=1.0, q=1.5, r=0.8 → sum=3.3

  Redistribute while keeping approximately the same total:
    baseline:         d=1.0, q=1.5, r=0.8   (sum=3.3)
    declare_heavy:    d=2.5, q=0.5, r=0.3   (sum=3.3 — query/respond cheaper)
    query_heavy:      d=0.3, q=2.5, r=0.5   (sum=3.3 — declare/respond cheaper)
    respond_heavy:    d=0.5, q=0.5, r=2.3   (sum=3.3 — declare/query cheaper)
    flat_equal:       d=1.1, q=1.1, r=1.1   (sum=3.3 — all equal)

  5 conditions × 15 seeds = 75 runs.

Output
------
  data/exp13_redistribution/seed_{s:02d}_{condition}/manifest.json
  data/exp13_redistribution/exp13_summary.json

Usage
-----
  python run_exp13_redistribution.py
  python run_exp13_redistribution.py --workers 4
  python run_exp13_redistribution.py --dry-run
  python run_exp13_redistribution.py --analyze-only
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

SEEDS  = list(range(15))
EPOCHS = 500
EPISODES = 10

# All conditions sum declare+query+respond to approximately 3.3
CONDITIONS = [
    {"label": "baseline",      "declare_cost": 1.0, "query_cost": 1.5, "respond_cost": 0.8},
    {"label": "declare_heavy", "declare_cost": 2.5, "query_cost": 0.5, "respond_cost": 0.3},
    {"label": "query_heavy",   "declare_cost": 0.3, "query_cost": 2.5, "respond_cost": 0.5},
    {"label": "respond_heavy", "declare_cost": 0.5, "query_cost": 0.5, "respond_cost": 2.3},
    {"label": "flat_equal",    "declare_cost": 1.1, "query_cost": 1.1, "respond_cost": 1.1},
]

DATA_ROOT = Path(__file__).parent.parent / "data" / "exp13_redistribution"


# ── Worker ─────────────────────────────────────────────────────────────────────

def _run_one(spec: dict) -> dict:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from simulation.engine import SimulationEngine, SimulationConfig

    config = SimulationConfig(
        seed=spec["seed"],
        num_epochs=EPOCHS,
        episodes_per_epoch=EPISODES,
        protocol=1,
        declare_cost=spec["declare_cost"],
        query_cost=spec["query_cost"],
        respond_cost=spec["respond_cost"],
        output_dir=spec["output_dir"],
    )
    engine = SimulationEngine(config=config)
    engine.run()
    return {"seed": spec["seed"], "condition": spec["label"]}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_manifest(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


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


def _extract(m: dict) -> dict:
    fm = m.get("final_metrics", m)
    return {
        "crystallized":   _is_crystallized(m),
        "crys_epoch":     m.get("crystallization_epoch"),
        "type_entropy":   fm.get("type_entropy"),
        "qrc":            fm.get("query_response_coupling") or fm.get("qrc"),
        "query_rate":     fm.get("query_rate"),
        "respond_rate":   fm.get("respond_rate"),
        "survival_rate":  fm.get("survival_rate"),
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
                "declare_cost": cond["declare_cost"],
                "query_cost":   cond["query_cost"],
                "respond_cost": cond["respond_cost"],
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
        label = cond["label"]
        rows  = per_condition[label]
        n     = len(rows)
        n_crys = sum(1 for r in rows if r.get("crystallized"))
        total_cost = cond["declare_cost"] + cond["query_cost"] + cond["respond_cost"]
        condition_stats[label] = {
            "declare_cost":         cond["declare_cost"],
            "query_cost":           cond["query_cost"],
            "respond_cost":         cond["respond_cost"],
            "total_cost":           round(total_cost, 2),
            "n":                    n,
            "n_crystallized":       n_crys,
            "crystallization_rate": round(n_crys / n, 4) if n else None,
            "mean_qrc":             _mean([r["qrc"] for r in rows]),
            "mean_type_entropy":    _mean([r["type_entropy"] for r in rows]),
            "mean_query_rate":      _mean([r["query_rate"] for r in rows]),
            "mean_respond_rate":    _mean([r["respond_rate"] for r in rows]),
            "mean_survival_rate":   _mean([r["survival_rate"] for r in rows]),
        }

    # Variance in crystallization rate across redistribution conditions
    crys_rates = [
        condition_stats[c["label"]]["crystallization_rate"]
        for c in CONDITIONS
        if condition_stats[c["label"]]["crystallization_rate"] is not None
    ]
    mean_rate = sum(crys_rates) / len(crys_rates) if crys_rates else None
    var_rate  = (sum((r - mean_rate)**2 for r in crys_rates) / len(crys_rates)
                 if mean_rate is not None else None)

    return {
        "experiment":     "13",
        "description":    "Redistribution does not equal resolution",
        "generated":      datetime.now().isoformat(),
        "note":           "All conditions have total cost sum ~ 3.3 (same as baseline). "
                          "Crystallization insensitive to redistribution = total pressure is binding.",
        "n_seeds":        len(seeds),
        "conditions":     condition_stats,
        "cross_condition_stats": {
            "crys_rates":   crys_rates,
            "mean_rate":    round(mean_rate, 4) if mean_rate else None,
            "var_rate":     round(var_rate, 4)  if var_rate  else None,
            "redistribution_matters": var_rate is not None and var_rate > 0.01,
        },
    }


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_summary(result: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nExp 13 summary: {out_path}")

    print(f"\n  {'Condition':<14}  {'d':>4}  {'q':>4}  {'r':>4}  {'sum':>4}  "
          f"{'n':>3}  {'rate':>5}  {'QRC':>6}  {'Qrate':>6}")
    for cond in CONDITIONS:
        label = cond["label"]
        cs    = result["conditions"][label]
        rate  = f"{cs['crystallization_rate']:.2f}" if cs["crystallization_rate"] is not None else " N/A"
        qrc   = f"{cs['mean_qrc']:.3f}"             if cs["mean_qrc"]             is not None else " N/A"
        qr    = f"{cs['mean_query_rate']:.3f}"       if cs["mean_query_rate"]       is not None else " N/A"
        print(f"  {label:<14}  {cond['declare_cost']:>4.1f}  {cond['query_cost']:>4.1f}  "
              f"{cond['respond_cost']:>4.1f}  {cs['total_cost']:>4.1f}  "
              f"  {cs['n']:>2}  {rate:>5}  {qrc:>6}  {qr:>6}")

    cs_stats = result["cross_condition_stats"]
    if cs_stats["mean_rate"] is not None:
        print(f"\n  Mean crys rate across conditions: {cs_stats['mean_rate']:.4f}")
        print(f"  Variance: {cs_stats['var_rate']:.4f}")
        matters = cs_stats["redistribution_matters"]
        print(f"  Redistribution matters (var > 0.01): {'YES' if matters else 'NO'}")
        if not matters:
            print(f"  Interpretation: total cost pressure is the binding constraint,")
            print(f"  not its allocation across action types.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 13 -- Redistribution does not equal resolution")
    parser.add_argument("--seeds",        type=int, nargs=2, default=[0, 15],
                        metavar=("START", "END"))
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    seeds   = list(range(args.seeds[0], args.seeds[1]))
    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    print(f"Exp 13 -- Redistribution  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}  Workers: {workers}")
    print(f"Conditions: {len(CONDITIONS)}  x  {len(seeds)} seeds = {len(CONDITIONS)*len(seeds)} runs")
    print(f"Data: {DATA_ROOT}")

    if args.analyze_only:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp13_summary.json")
        return

    if not args.dry_run:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

    run_campaign(seeds, workers, args.dry_run)

    if not args.dry_run:
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp13_summary.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
