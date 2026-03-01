"""Experiment 9 — Partial observability ladder.

Systematically reduces each agent's observation quality by masking a random
fraction of observation dimensions to zero. Tests whether agents compensate
for missing state by increasing communication (query rate) — predicting that
memory/communication substitutes for direct observation.

Theory prediction
-----------------
  As observability drops (mask_prob rises):
    - query_rate increases (agents ask because they can't see)
    - QRC may strengthen (queries become more informative relative to obs)
    - survival_rate drops (communication partially compensates but cannot
      fully replace direct observation at high mask rates)

  Signature: a monotone increase in query_rate with mask_prob would be
  direct evidence that communication substitutes for missing state.

Design
------
  Noise model: with probability mask_prob, each observation dimension is
  zeroed before passing to the agent. The mask is applied per agent per step,
  independently drawn.

  Hook: agent.forward is patched to apply masking before the normal forward
  pass. Mask indices are randomized per call (random partial masking, not a
  fixed channel mask).

  Conditions:
    obs_mask_0.00  (no masking, baseline)
    obs_mask_0.25  (25% of obs dims zeroed)
    obs_mask_0.50  (50% zeroed)
    obs_mask_0.75  (75% zeroed)
    obs_mask_0.90  (90% zeroed — near total blindness)

  5 conditions × 15 seeds = 75 runs.

Output
------
  data/exp9_observability/seed_{s:02d}_{condition}/manifest.json
  data/exp9_observability/exp9_summary.json

Usage
-----
  python run_exp9_observability.py
  python run_exp9_observability.py --workers 4
  python run_exp9_observability.py --dry-run
  python run_exp9_observability.py --analyze-only
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
QUERY_COST   = 1.5
RESPOND_COST = 0.8

CONDITIONS = [
    {"label": "obs_mask_0.00", "mask_prob": 0.00},
    {"label": "obs_mask_0.25", "mask_prob": 0.25},
    {"label": "obs_mask_0.50", "mask_prob": 0.50},
    {"label": "obs_mask_0.75", "mask_prob": 0.75},
    {"label": "obs_mask_0.90", "mask_prob": 0.90},
]

DATA_ROOT = Path(__file__).parent.parent / "data" / "exp9_observability"


# ── Worker ─────────────────────────────────────────────────────────────────────

def _run_one(spec: dict) -> dict:
    """Run one seed under one observability condition. Called in subprocess."""
    import sys, os, random as _random
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from simulation.engine import SimulationEngine, SimulationConfig

    seed      = spec["seed"]
    mask_prob = spec["mask_prob"]

    config = SimulationConfig(
        seed=seed,
        num_epochs=EPOCHS,
        episodes_per_epoch=EPISODES,
        protocol=1,
        declare_cost=DECLARE_COST,
        query_cost=QUERY_COST,
        respond_cost=RESPOND_COST,
        output_dir=spec["output_dir"],
        device=spec.get("device", "cpu"),
    )
    engine = SimulationEngine(config=config)

    if mask_prob > 0.0:
        # Patch each agent's forward to mask a fraction of the observation.
        # Uses a per-seed RNG so masking is deterministic given seed.
        rng = _random.Random(seed + 555_555)

        for agent in engine.agents:
            original_forward = agent.forward

            def _make_patched(orig, _rng=rng, _mp=mask_prob):
                def patched_forward(obs, incoming):
                    masked_obs = obs.clone()
                    n = masked_obs.numel()
                    mask_size = int(n * _mp)
                    if mask_size > 0:
                        indices = list(range(n))
                        _rng.shuffle(indices)
                        for i in indices[:mask_size]:
                            masked_obs.view(-1)[i] = 0.0
                    return orig(masked_obs, incoming)
                return patched_forward

            agent.forward = _make_patched(original_forward)

    engine.run()
    return {"seed": seed, "condition": spec["label"]}


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
                "seed":       seed,
                "condition":  cond["label"],
                "mask_prob":  cond["mask_prob"],
                "output_dir": str(out_dir),
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
    import torch as _torch
    _n_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
    for _i, _j in enumerate(jobs):
        _j["device"] = f"cuda:{_i % _n_gpus}" if _n_gpus > 0 else "cpu"
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
            "mask_prob":            cond["mask_prob"],
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

    # Monotonicity: does query_rate increase with mask_prob?
    mask_vs_qrate = [
        (cond["mask_prob"], condition_stats[cond["label"]]["mean_query_rate"])
        for cond in CONDITIONS
    ]
    mask_vs_survival = [
        (cond["mask_prob"], condition_stats[cond["label"]]["mean_survival_rate"])
        for cond in CONDITIONS
    ]
    qrate_mono = all(
        (mask_vs_qrate[i][1] or 0) <= (mask_vs_qrate[i+1][1] or 0)
        for i in range(len(mask_vs_qrate)-1)
    )
    survival_mono = all(
        (mask_vs_survival[i][1] or 0) >= (mask_vs_survival[i+1][1] or 0)
        for i in range(len(mask_vs_survival)-1)
    )

    return {
        "experiment":      "9",
        "description":     "Partial observability ladder",
        "generated":       datetime.now().isoformat(),
        "n_seeds":         len(seeds),
        "conditions":      condition_stats,
        "monotonicity": {
            "query_rate_increases_with_mask": qrate_mono,
            "survival_decreases_with_mask":   survival_mono,
            "mask_vs_query_rate":             mask_vs_qrate,
            "mask_vs_survival_rate":          mask_vs_survival,
        },
    }


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_summary(result: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nExp 9 summary: {out_path}")

    print(f"\n  {'Condition':<18}  {'mask':>5}  {'n':>3}  "
          f"{'rate':>5}  {'QRC':>6}  {'Qrate':>6}  {'surv':>6}")
    for cond in CONDITIONS:
        label = cond["label"]
        cs    = result["conditions"][label]
        rate  = f"{cs['crystallization_rate']:.2f}" if cs["crystallization_rate"] is not None else " N/A"
        qrc   = f"{cs['mean_qrc']:.3f}"             if cs["mean_qrc"]             is not None else " N/A"
        qr    = f"{cs['mean_query_rate']:.3f}"       if cs["mean_query_rate"]       is not None else " N/A"
        sv    = f"{cs['mean_survival_rate']:.3f}"    if cs["mean_survival_rate"]    is not None else " N/A"
        print(f"  {label:<18}  {cond['mask_prob']:>5.2f}  {cs['n']:>3}  "
              f"{rate:>5}  {qrc:>6}  {qr:>6}  {sv:>6}")

    mn = result["monotonicity"]
    print(f"\n  Query rate increases with mask: {mn['query_rate_increases_with_mask']}")
    print(f"  Survival decreases with mask:   {mn['survival_decreases_with_mask']}")
    print(f"  Mask → Qrate: " +
          "  ".join(f"{m:.2f}→{q:.3f}" if q else f"{m:.2f}→None"
                    for m, q in mn["mask_vs_query_rate"]))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 9 -- Partial observability ladder")
    parser.add_argument("--seeds",        type=int, nargs=2, default=[0, 15],
                        metavar=("START", "END"))
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    seeds   = list(range(args.seeds[0], args.seeds[1]))
    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    print(f"Exp 9 -- Partial observability  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}  Workers: {workers}")
    print(f"Conditions: {len(CONDITIONS)}  x  {len(seeds)} seeds = {len(CONDITIONS)*len(seeds)} runs")
    print(f"Data: {DATA_ROOT}")

    if args.analyze_only:
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp9_summary.json")
        return

    if not args.dry_run:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

    run_campaign(seeds, workers, args.dry_run)

    if not args.dry_run:
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp9_summary.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
