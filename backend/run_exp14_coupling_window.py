"""Experiment 14 — Query cost as coupling primitive (QUERY→RESPONSE lock).

Tests whether there is an intermediate query cost window where asking is rare
but decisive, producing maximal QUERY→RESPONSE coupling. Sweeps query_cost
and respond_cost independently.

Theory prediction
-----------------
  QRC is maximised at intermediate query_cost (not zero, not extreme).
  - At zero cost: agents query freely, signal selectivity collapses, coupling
    weakens (too much noise).
  - At extreme cost: agents never query, QRC is undefined or near zero.
  - At intermediate cost: queries are selective, responses are reliable, and
    coupling is maximised.

  Independently, respond_cost should modulate QRC strength:
  - Low respond_cost: responders over-respond, weakening selectivity.
  - High respond_cost: responders under-respond, reducing coupling.
  - Intermediate: selective responding maximises coupling.

Design
------
  Primary sweep — query_cost (respond_cost fixed at 0.8 baseline):
    qcost_0.5  qcost_1.0  qcost_1.5  qcost_2.0  qcost_3.0  qcost_5.0

  Secondary sweep — respond_cost (query_cost fixed at 1.5 baseline):
    rcost_0.4  rcost_0.8 (already in primary)  rcost_1.5  rcost_3.0

  Note: qcost_1.5/rcost_0.8 is the baseline condition shared between sweeps.

  Deduplication: 9 unique conditions × 15 seeds = 135 runs.

  Full condition list:
    qcost_0.5_rcost_0.8   (primary sweep)
    qcost_1.0_rcost_0.8   (primary sweep)
    qcost_1.5_rcost_0.8   (primary + secondary baseline)
    qcost_2.0_rcost_0.8   (primary sweep)
    qcost_3.0_rcost_0.8   (primary sweep)
    qcost_5.0_rcost_0.8   (primary sweep)
    qcost_1.5_rcost_0.4   (secondary sweep)
    qcost_1.5_rcost_1.5   (secondary sweep)
    qcost_1.5_rcost_3.0   (secondary sweep)

Output
------
  data/exp14_coupling_window/seed_{s:02d}_{condition}/manifest.json
  data/exp14_coupling_window/exp14_summary.json

Usage
-----
  python run_exp14_coupling_window.py
  python run_exp14_coupling_window.py --workers 4
  python run_exp14_coupling_window.py --dry-run
  python run_exp14_coupling_window.py --analyze-only
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

# ── Parameters ─────────────────────────────────────────────────────────────────

SEEDS        = list(range(15))
EPOCHS       = 500
EPISODES     = 10
DECLARE_COST = 1.0

# Primary sweep: vary query_cost, fix respond_cost=0.8
# Secondary sweep: fix query_cost=1.5, vary respond_cost
CONDITIONS = [
    # primary sweep (6 q-cost levels)
    {"label": "qcost_0.5_rcost_0.8",  "query_cost": 0.5, "respond_cost": 0.8},
    {"label": "qcost_1.0_rcost_0.8",  "query_cost": 1.0, "respond_cost": 0.8},
    {"label": "qcost_1.5_rcost_0.8",  "query_cost": 1.5, "respond_cost": 0.8},  # baseline
    {"label": "qcost_2.0_rcost_0.8",  "query_cost": 2.0, "respond_cost": 0.8},
    {"label": "qcost_3.0_rcost_0.8",  "query_cost": 3.0, "respond_cost": 0.8},
    {"label": "qcost_5.0_rcost_0.8",  "query_cost": 5.0, "respond_cost": 0.8},
    # secondary sweep (3 r-cost levels, not including 0.8 which is above)
    {"label": "qcost_1.5_rcost_0.4",  "query_cost": 1.5, "respond_cost": 0.4},
    {"label": "qcost_1.5_rcost_1.5",  "query_cost": 1.5, "respond_cost": 1.5},
    {"label": "qcost_1.5_rcost_3.0",  "query_cost": 1.5, "respond_cost": 3.0},
]

DATA_ROOT = Path(__file__).parent.parent / "data" / "exp14_coupling_window"


# ── Worker ─────────────────────────────────────────────────────────────────────

def _run_one(spec: dict) -> dict:
    """Run one seed under one cost condition. Called in subprocess."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from simulation.engine import SimulationEngine, SimulationConfig

    config = SimulationConfig(
        seed=spec["seed"],
        num_epochs=EPOCHS,
        episodes_per_epoch=EPISODES,
        protocol=1,
        declare_cost=DECLARE_COST,
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


def _extract(manifest: dict) -> dict:
    fm = manifest.get("final_metrics", manifest)
    iq = fm.get("inquiry", fm)   # some manifests nest metrics under "inquiry"
    return {
        "crystallized":        manifest.get("crystallized", fm.get("crystallized")),
        "crys_epoch":          manifest.get("crystallization_epoch",
                                            fm.get("crystallization_epoch")),
        "type_entropy":        fm.get("type_entropy") or iq.get("type_entropy"),
        "qrc":                 fm.get("query_response_coupling") or
                               iq.get("query_response_coupling"),
        "query_rate":          fm.get("query_rate") or iq.get("query_rate"),
        "respond_rate":        fm.get("respond_rate") or iq.get("respond_rate"),
        "survival_rate":       fm.get("survival_rate"),
        "query_selectivity":   fm.get("query_selectivity") or iq.get("query_selectivity"),
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
        condition_stats[label] = {
            "query_cost":           cond["query_cost"],
            "respond_cost":         cond["respond_cost"],
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

    # Primary sweep: QRC by query_cost (respond_cost=0.8)
    primary = [
        (cond["query_cost"], condition_stats[cond["label"]]["mean_qrc"])
        for cond in CONDITIONS if cond["respond_cost"] == 0.8
    ]
    primary.sort()
    qrc_vals = [v for _, v in primary if v is not None]
    peak_qcost = None
    if qrc_vals:
        peak_idx = max(range(len(primary)), key=lambda i: primary[i][1] or 0.0)
        peak_qcost = primary[peak_idx][0]

    # Secondary sweep: QRC by respond_cost (query_cost=1.5)
    secondary = [
        (cond["respond_cost"], condition_stats[cond["label"]]["mean_qrc"])
        for cond in CONDITIONS if cond["query_cost"] == 1.5
    ]
    secondary.sort()

    return {
        "experiment":          "14",
        "description":         "Query cost as coupling primitive (QUERY→RESPONSE lock)",
        "generated":           datetime.now().isoformat(),
        "n_seeds":             len(seeds),
        "conditions":          condition_stats,
        "primary_sweep": {
            "variable":        "query_cost",
            "fixed":           {"respond_cost": 0.8},
            "qrc_by_qcost":    primary,
            "peak_query_cost": peak_qcost,
        },
        "secondary_sweep": {
            "variable":        "respond_cost",
            "fixed":           {"query_cost": 1.5},
            "qrc_by_rcost":    secondary,
        },
    }


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_summary(result: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nExp 14 summary: {out_path}")

    print(f"\n  {'Condition':<25}  {'qcost':>6}  {'rcost':>6}  "
          f"{'n':>3}  {'rate':>5}  {'QRC':>6}  {'Qrate':>6}  {'Rrate':>6}")
    for cond in CONDITIONS:
        label = cond["label"]
        cs    = result["conditions"][label]
        rate  = f"{cs['crystallization_rate']:.2f}" if cs["crystallization_rate"] is not None else " N/A"
        qrc   = f"{cs['mean_qrc']:.3f}"             if cs["mean_qrc"]             is not None else " N/A"
        qr    = f"{cs['mean_query_rate']:.3f}"       if cs["mean_query_rate"]       is not None else " N/A"
        rr    = f"{cs['mean_respond_rate']:.3f}"     if cs["mean_respond_rate"]     is not None else " N/A"
        print(f"  {label:<25}  {cond['query_cost']:>6.1f}  {cond['respond_cost']:>6.1f}  "
              f"  {cs['n']:>2}  {rate:>5}  {qrc:>6}  {qr:>6}  {rr:>6}")

    ps = result["primary_sweep"]
    print(f"\n  Primary QRC sweep (vary query_cost, rcost=0.8):")
    print("  " + "  ".join(f"q={q}→{v:.3f}" if v else f"q={q}→None"
                            for q, v in ps["qrc_by_qcost"]))
    if ps["peak_query_cost"] is not None:
        print(f"  Peak QRC at query_cost = {ps['peak_query_cost']}")

    ss = result["secondary_sweep"]
    print(f"\n  Secondary QRC sweep (vary respond_cost, qcost=1.5):")
    print("  " + "  ".join(f"r={r}→{v:.3f}" if v else f"r={r}→None"
                            for r, v in ss["qrc_by_rcost"]))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 14 -- Query cost as coupling primitive")
    parser.add_argument("--seeds",        type=int, nargs=2, default=[0, 15],
                        metavar=("START", "END"))
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    seeds   = list(range(args.seeds[0], args.seeds[1]))
    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    print(f"Exp 14 -- Coupling window  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}  Workers: {workers}")
    print(f"Conditions: {len(CONDITIONS)}  x  {len(seeds)} seeds = {len(CONDITIONS)*len(seeds)} runs")
    print(f"Data: {DATA_ROOT}")

    if args.analyze_only:
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp14_summary.json")
        return

    if not args.dry_run:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

    run_campaign(seeds, workers, args.dry_run)

    if not args.dry_run:
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp14_summary.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
