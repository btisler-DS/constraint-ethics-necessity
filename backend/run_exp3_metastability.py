"""Experiment 3 — Metastability under intermittent pressure.

Tests whether a crystallized interrogative protocol survives temporary pressure
relaxation and reacquires rapidly when pressure is restored.

Design
------
  Square-wave tax schedule: alternating HIGH (q=3.0) / LOW (q=0.5) phases,
  each wave_period epochs long. If the protocol maintains or rapidly reacquires
  crystallization after a LOW phase, the structure is stored in weights — not
  merely maintained by ongoing cost pressure.

  Conditions:
    square_wave  : q alternates 3.0 / 0.5 every wave_period epochs
    high_constant: q=3.0 constant (reference: pressure-only baseline)
    low_constant : q=0.5 constant (reference: does low pressure ever crystallize?)

  Fixed: DECLARE=1.0, RESPOND=0.8, epsilon=0.01, tau=20, seeds 0..14.

Key outcome measures (per seed, per condition):
  1. crystallization_epoch: first epoch of 5-streak with H < 0.95
  2. n_high_phases_crystallized: number of HIGH phases where H < 0.95 streak holds
  3. reacquisition_latency: epochs from start of HIGH phase to H < 0.95 (post-first-crystallization)
  4. mean_entropy_high / mean_entropy_low: entropy in HIGH vs LOW phases
  5. entropy_sustained: fraction of ALL epochs (high+low) where H < 0.95
  6. collapse_events: number of times H crosses back above 0.95 after crystallization

Usage
-----
  python run_exp3_metastability.py
  python run_exp3_metastability.py --conditions square_wave high_constant
  python run_exp3_metastability.py --workers 4
  python run_exp3_metastability.py --dry-run
  python run_exp3_metastability.py --analyze-only

Output
------
  data/exp3_metastability/{condition}/seed_{s:02d}/manifest.json
  data/exp3_metastability/exp3_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Preregistered parameters ───────────────────────────────────────────────────

SEEDS        = list(range(15))
EPOCHS       = 500
EPISODES     = 10
DECLARE_COST = 1.0
RESPOND_COST = 0.8
WAVE_PERIOD  = 50    # epochs per phase (50 high + 50 low = 100-epoch cycle)

CONDITIONS = {
    "square_wave": {
        "label":           "Square-wave q=3.0/0.5 (period=50)",
        "mode":            "square_wave",
        "query_cost_high": 3.0,
        "query_cost_low":  0.5,
        "wave_period":     WAVE_PERIOD,
    },
    "high_constant": {
        "label":           "Constant q=3.0 (pressure reference)",
        "mode":            "constant",
        "query_cost":      3.0,
    },
    "low_constant": {
        "label":           "Constant q=0.5 (relaxed reference)",
        "mode":            "constant",
        "query_cost":      0.5,
    },
}

ENTROPY_CRYS_THRESHOLD = 0.95
ENTROPY_STREAK         = 5


# ── Square-wave engine ─────────────────────────────────────────────────────────

def _run_square_wave(seed: int, output_dir: str, query_cost_high: float,
                     query_cost_low: float, wave_period: int) -> dict:
    """Run one seed with a square-wave query cost schedule."""
    from simulation.engine import SimulationEngine, SimulationConfig

    # Start in HIGH phase (epoch 0 = high, epoch wave_period = low, ...)
    phase_log = []

    config = SimulationConfig(
        seed=seed,
        num_epochs=EPOCHS,
        episodes_per_epoch=EPISODES,
        protocol=1,
        declare_cost=DECLARE_COST,
        query_cost=query_cost_high,   # initial phase = HIGH
        respond_cost=RESPOND_COST,
        output_dir=output_dir,
    )

    engine = SimulationEngine(config=config)

    # Override run() via epoch_callback isn't sufficient since callback fires
    # AFTER the epoch. Instead we wrap _run_epoch to mutate cost before each epoch.
    original_run_epoch = engine._run_epoch

    def patched_run_epoch(epoch: int) -> dict:
        phase_idx = (epoch // wave_period) % 2
        is_high   = (phase_idx == 0)
        engine.protocol.query_cost = query_cost_high if is_high else query_cost_low
        phase_log.append({"epoch": epoch, "phase": "HIGH" if is_high else "LOW",
                          "query_cost": engine.protocol.query_cost})
        metrics = original_run_epoch(epoch)
        metrics["square_wave_phase"] = "HIGH" if is_high else "LOW"
        metrics["query_cost_this_epoch"] = engine.protocol.query_cost
        return metrics

    engine._run_epoch = patched_run_epoch
    engine.run()

    return _analyze_run(engine.epoch_metrics, phase_log, wave_period, seed)


def _run_constant(seed: int, output_dir: str, query_cost: float) -> dict:
    """Run one seed with constant query cost."""
    from simulation.engine import SimulationEngine, SimulationConfig

    config = SimulationConfig(
        seed=seed,
        num_epochs=EPOCHS,
        episodes_per_epoch=EPISODES,
        protocol=1,
        declare_cost=DECLARE_COST,
        query_cost=query_cost,
        respond_cost=RESPOND_COST,
        output_dir=output_dir,
    )
    engine = SimulationEngine(config=config)
    engine.run()

    phase_log = [{"epoch": e, "phase": "CONST", "query_cost": query_cost}
                 for e in range(EPOCHS)]
    return _analyze_run(engine.epoch_metrics, phase_log, None, seed)


# ── Per-run analysis ───────────────────────────────────────────────────────────

def _analyze_run(epoch_metrics: list[dict], phase_log: list[dict],
                 wave_period: int | None, seed: int) -> dict:
    """Extract metastability metrics from a completed run."""

    entropies = []
    for m in epoch_metrics:
        inq = m.get("inquiry", {})
        te  = inq.get("type_entropy")
        entropies.append(te)

    n = len(entropies)

    # Crystallization epoch (standard: first epoch of 5-streak H < threshold)
    crys_epoch = None
    streak = 0
    for i, h in enumerate(entropies):
        if h is not None and h < ENTROPY_CRYS_THRESHOLD:
            streak += 1
            if streak >= ENTROPY_STREAK and crys_epoch is None:
                crys_epoch = i - ENTROPY_STREAK + 1
        else:
            streak = 0

    # Fraction of epochs below threshold (crystallization persistence)
    below = [h for h in entropies if h is not None and h < ENTROPY_CRYS_THRESHOLD]
    entropy_sustained = round(len(below) / n, 4) if n > 0 else 0.0

    # Collapse events: H crosses above threshold after first crystallization
    collapse_events = 0
    if crys_epoch is not None:
        was_crystallized = False
        for h in entropies[crys_epoch:]:
            if h is None:
                continue
            if h < ENTROPY_CRYS_THRESHOLD:
                was_crystallized = True
            elif was_crystallized:
                collapse_events += 1
                was_crystallized = False

    # Per-phase statistics (square wave only)
    phase_entropy: dict[str, list[float]] = {}
    for i, pl in enumerate(phase_log):
        if i < len(entropies) and entropies[i] is not None:
            phase = pl["phase"]
            phase_entropy.setdefault(phase, []).append(entropies[i])

    def _mean(xs): return round(sum(xs) / len(xs), 4) if xs else None

    mean_entropy_by_phase = {ph: _mean(vals) for ph, vals in phase_entropy.items()}

    # Reacquisition latency (square wave only):
    # For each HIGH phase that begins AFTER first crystallization, measure how many
    # epochs from HIGH phase start until H < threshold again (after prior LOW phase).
    reacquisition_latencies = []
    if wave_period and crys_epoch is not None:
        for phase_start in range(0, n, wave_period):
            phase_idx = (phase_start // wave_period) % 2
            if phase_idx != 0:  # not HIGH phase
                continue
            if phase_start <= crys_epoch:  # before first crystallization
                continue
            # Check previous phase was LOW
            prev_phase_start = phase_start - wave_period
            if prev_phase_start < 0:
                continue
            # Find how quickly H drops below threshold in this HIGH phase
            for offset in range(min(wave_period, n - phase_start)):
                h = entropies[phase_start + offset]
                if h is not None and h < ENTROPY_CRYS_THRESHOLD:
                    reacquisition_latencies.append(offset)
                    break
            else:
                reacquisition_latencies.append(wave_period)  # never reacquired

    mean_reacq = _mean(reacquisition_latencies) if reacquisition_latencies else None
    n_high_phases_crystallized = sum(1 for lat in reacquisition_latencies if lat < wave_period) \
                                 if reacquisition_latencies else None

    return {
        "seed":                       seed,
        "crystallization_epoch":      crys_epoch,
        "crystallized":               crys_epoch is not None,
        "entropy_sustained":          entropy_sustained,
        "collapse_events":            collapse_events,
        "mean_entropy_by_phase":      mean_entropy_by_phase,
        "reacquisition_latencies":    reacquisition_latencies,
        "mean_reacquisition_latency": mean_reacq,
        "n_high_phases_crystallized": n_high_phases_crystallized,
    }


# ── Worker (called in subprocess) ─────────────────────────────────────────────

def _worker(spec: dict) -> dict:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    cond   = CONDITIONS[spec["condition"]]
    seed   = spec["seed"]
    outdir = spec["output_dir"]

    if cond["mode"] == "square_wave":
        result = _run_square_wave(
            seed=seed, output_dir=outdir,
            query_cost_high=cond["query_cost_high"],
            query_cost_low=cond["query_cost_low"],
            wave_period=cond["wave_period"],
        )
    else:
        result = _run_constant(
            seed=seed, output_dir=outdir,
            query_cost=cond["query_cost"],
        )

    result["condition"] = spec["condition"]

    # Save extended manifest alongside engine manifest
    ext_path = os.path.join(outdir, "exp3_metrics.json")
    with open(ext_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Campaign runner ────────────────────────────────────────────────────────────

def run_campaign(conditions: list[str], seeds: list[int], workers: int,
                 dry_run: bool, data_root: Path) -> list[dict]:
    jobs = []
    for cond in conditions:
        for seed in seeds:
            out_dir = data_root / cond / f"seed_{seed:02d}"
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
            jobs.append({"condition": cond, "seed": seed, "output_dir": str(out_dir)})

    print(f"\nExp 3 -- Metastability under intermittent pressure")
    print(f"  conditions: {conditions}")
    print(f"  seeds:      {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
    print(f"  total:      {len(jobs)} runs  |  workers: {workers}")
    print(f"  wave:       period={WAVE_PERIOD} epochs  q_high=3.0  q_low=0.5")
    print(f"  output:     {data_root}")

    if dry_run:
        for j in jobs[:5]:
            print(f"  DRY-RUN: {j['condition']} seed={j['seed']} -> {j['output_dir']}")
        if len(jobs) > 5:
            print(f"  ... ({len(jobs)-5} more)")
        return []

    results = []
    t0   = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                done += 1
                crys_str = f"crys@{r['crystallization_epoch']:4d}" \
                           if r["crystallized"] else "crys=None"
                reacq = (f"reacq={r['mean_reacquisition_latency']:.1f}"
                         if r["mean_reacquisition_latency"] is not None else "reacq=N/A")
                print(f"  [{done:2d}/{len(jobs)}]  {r['condition']:15s}  seed={r['seed']:02d}"
                      f"  {crys_str}  sustain={r['entropy_sustained']:.3f}"
                      f"  collapse={r['collapse_events']}  {reacq}")
            except Exception as e:
                j = futures[fut]
                print(f"  ERROR: {j['condition']} seed={j['seed']}: {e}")

    print(f"\nExp 3 done: {done}/{len(jobs)} runs in {time.time()-t0:.0f}s")
    return results


# ── Summary ────────────────────────────────────────────────────────────────────

def write_summary(results: list[dict], data_root: Path) -> None:
    if not results:
        return

    from collections import defaultdict
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    def _mean(xs): return round(sum(xs)/len(xs), 4) if xs else None
    def _sd(xs):
        if len(xs) < 2: return None
        mu = sum(xs)/len(xs)
        return round((sum((x-mu)**2 for x in xs)/len(xs))**0.5, 4)

    cond_summaries = []
    for cond in CONDITIONS:
        runs = by_cond.get(cond, [])
        if not runs:
            continue
        n = len(runs)
        crys      = [r for r in runs if r["crystallized"]]
        sustains  = [r["entropy_sustained"] for r in runs]
        collapses = [r["collapse_events"] for r in runs]
        reacqs    = [r["mean_reacquisition_latency"] for r in runs
                     if r["mean_reacquisition_latency"] is not None]

        cond_summaries.append({
            "condition":             cond,
            "label":                 CONDITIONS[cond]["label"],
            "n_seeds":               n,
            "crystallization_rate":  round(len(crys)/n, 4),
            "n_crystallized":        len(crys),
            "mean_crystallization_epoch": _mean([r["crystallization_epoch"] for r in crys]),
            "mean_entropy_sustained":     _mean(sustains),
            "sd_entropy_sustained":       _sd(sustains),
            "mean_collapse_events":       _mean(collapses),
            "mean_reacquisition_latency": _mean(reacqs),
            "sd_reacquisition_latency":   _sd(reacqs),
        })

    summary = {
        "experiment":    "3",
        "description":   "Metastability under intermittent pressure (square-wave tax)",
        "generated":     datetime.now().isoformat(),
        "wave_period":   WAVE_PERIOD,
        "query_cost_high": 3.0,
        "query_cost_low":  0.5,
        "n_seeds":       len(by_cond.get("square_wave", [])),
        "conditions":    cond_summaries,
        "per_seed":      sorted(results, key=lambda r: (r["condition"], r["seed"])),
    }

    out = data_root / "exp3_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nExp 3 summary: {out}")

    print(f"\n  {'condition':20s}  {'crys_rate':>9}  {'mean_epoch':>10}  "
          f"{'sustained':>9}  {'collapses':>9}  {'reacq_lat':>9}")
    for c in cond_summaries:
        ep = f"{c['mean_crystallization_epoch']:.0f}" \
             if c["mean_crystallization_epoch"] else "   None"
        rq = f"{c['mean_reacquisition_latency']:.1f}" \
             if c["mean_reacquisition_latency"] else "   N/A"
        print(f"  {c['condition']:20s}  {c['crystallization_rate']:>9.3f}  "
              f"{ep:>10}  {c['mean_entropy_sustained']:>9.3f}  "
              f"{c['mean_collapse_events']:>9.2f}  {rq:>9}")


# ── Analyze-only ───────────────────────────────────────────────────────────────

def analyze_only(conditions: list[str], seeds: list[int], data_root: Path) -> None:
    results = []
    for cond in conditions:
        for seed in seeds:
            p = data_root / cond / f"seed_{seed:02d}" / "exp3_metrics.json"
            if not p.exists():
                print(f"  WARNING: missing {p}")
                continue
            with open(p) as f:
                r = json.load(f)
            r["condition"] = cond
            results.append(r)
    if results:
        write_summary(results, data_root)
    else:
        print("  No data found.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 3 -- Metastability under intermittent pressure")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()),
                        choices=list(CONDITIONS.keys()))
    parser.add_argument("--seeds",       type=int, nargs=2, default=[0, 15],
                        metavar=("START", "END"))
    parser.add_argument("--workers",     type=int, default=None)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--data-dir",    type=str,
                        default=str(Path(__file__).parent.parent / "data" / "exp3_metastability"))
    args = parser.parse_args()

    seeds     = list(range(args.seeds[0], args.seeds[1]))
    workers   = args.workers or max(1, (os.cpu_count() or 4) - 1)
    data_root = Path(args.data_dir)

    print(f"Exp 3 -- Metastability  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}  Workers: {workers}  Data: {data_root}")

    if args.analyze_only:
        analyze_only(args.conditions, seeds, data_root)
        return

    if not args.dry_run:
        data_root.mkdir(parents=True, exist_ok=True)

    results = run_campaign(args.conditions, seeds, workers, args.dry_run, data_root)
    if results:
        write_summary(results, data_root)
    print("\nDone.")


if __name__ == "__main__":
    main()
