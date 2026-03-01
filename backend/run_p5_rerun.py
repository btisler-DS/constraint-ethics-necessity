"""P5 Coordination Advantage rerun -- n=30, survival rate differential metric.

EXPLORATORY — labeled as such throughout. Not a confirmatory test.

The original P5 test used energy ROI (target_reached_rate / avg_energy_spent)
at n=15. It failed formal significance (p=0.227, Cohen's d=0.28, underpowered
for preregistered d>=0.8). This rerun addresses both problems:

  Primary metric (exploratory): survival_rate differential
    = P1_survival_rate - P0_survival_rate  (per paired seed)
    Less zero-inflated than energy ROI; directly measures the coordination
    advantage conferred by query-enabled agents.

  Secondary metric: energy_roi differential (original preregistered metric)
    Retained for comparison and continuity.

Design
------
  Protocol 1 (query-enabled): DECLARE=1.0x, QUERY=1.5x, RESPOND=0.8x  (baseline)
  Protocol 0 (query-disabled): DECLARE=1.0x only  (control)
  Seeds: 0..29 (n=30 per group)
  Epochs: 500, Episodes: 10 per epoch

Existing data reuse
-------------------
  Seeds 0-14 for baseline and control exist in data/campaign/.
  Only seeds 15-29 are run fresh.

Output
------
  data/p5_rerun/baseline/seed_{s:02d}/manifest.json  (seeds 15-29 only)
  data/p5_rerun/control/seed_{s:02d}/manifest.json   (seeds 15-29 only)
  data/p5_rerun/p5_summary.json

Usage
-----
  python run_p5_rerun.py                  # seeds 15-29 (extends existing)
  python run_p5_rerun.py --seeds 0 30     # all 30 seeds (re-runs 0-14 too)
  python run_p5_rerun.py --workers 4
  python run_p5_rerun.py --analyze-only   # reread existing data
  python run_p5_rerun.py --dry-run
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

EPOCHS   = 500
EPISODES = 10

CAMPAIGN_DIR = Path(__file__).parent.parent / "data" / "campaign"
DATA_DIR     = Path(__file__).parent.parent / "data" / "p5_rerun"


# ── Worker ─────────────────────────────────────────────────────────────────────

def _run_one(spec: dict) -> dict:
    """Run one seed under one protocol. Called in subprocess."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from simulation.engine import SimulationEngine, SimulationConfig

    cfg = SimulationConfig(
        seed=spec["seed"],
        num_epochs=spec["epochs"],
        episodes_per_epoch=spec["episodes"],
        protocol=spec["protocol"],
        declare_cost=spec["declare_cost"],
        query_cost=spec["query_cost"],
        respond_cost=spec["respond_cost"],
        output_dir=spec["output_dir"],
        device=spec.get("device", "cpu"),
    )
    engine = SimulationEngine(config=cfg)
    engine.run()
    return {"seed": spec["seed"], "condition": spec["condition"]}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_manifest(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _extract_metrics(manifest: dict) -> dict:
    fm = manifest.get("final_metrics", manifest)
    return {
        "survival_rate":      fm.get("survival_rate"),
        "target_reached_rate": fm.get("target_reached_rate"),
        "energy_roi":         fm.get("energy_roi"),
    }


def _load_seed_metrics(seed: int, condition: str) -> dict | None:
    """Try rerun dir first, fall back to original campaign dir."""
    # Rerun dir (seeds 15-29 or re-run seeds)
    rerun_path = DATA_DIR / condition / f"seed_{seed:02d}" / "manifest.json"
    if rerun_path.exists():
        m = _load_manifest(rerun_path)
        if m:
            return _extract_metrics(m)

    # Original campaign dir (seeds 0-14)
    orig_path = CAMPAIGN_DIR / condition / f"seed_{seed:02d}" / "manifest.json"
    if orig_path.exists():
        m = _load_manifest(orig_path)
        if m:
            return _extract_metrics(m)

    return None


# ── Campaign runner ────────────────────────────────────────────────────────────

def run_new_seeds(seeds: list[int], workers: int, dry_run: bool) -> None:
    """Run only seeds that don't exist in either location."""
    jobs = []
    for seed in seeds:
        for condition, proto, qcost, rcost in [
            ("baseline", 1, 1.5, 0.8),
            ("control",  0, 0.0, 0.0),
        ]:
            # Skip if data already exists
            if _load_seed_metrics(seed, condition) is not None:
                continue
            out_dir = DATA_DIR / condition / f"seed_{seed:02d}"
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
            jobs.append({
                "seed":        seed,
                "condition":   condition,
                "protocol":    proto,
                "declare_cost": 1.0,
                "query_cost":  qcost,
                "respond_cost": rcost,
                "epochs":      EPOCHS,
                "episodes":    EPISODES,
                "output_dir":  str(out_dir),
            })

    if not jobs:
        print("  No new seeds to run (all data present).")
        return

    print(f"\n  Running {len(jobs)} new seed/condition pairs  (workers={workers})")
    if dry_run:
        for j in jobs[:5]:
            print(f"  DRY-RUN: {j['condition']} seed={j['seed']}")
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
                print(f"  [{done:2d}/{len(jobs)}] {r['condition']} seed={r['seed']:02d}")
            except Exception as e:
                j = futures[fut]
                print(f"  ERROR: {j['condition']} seed={j['seed']}: {e}")
    print(f"  Done in {time.time()-t0:.0f}s")


# ── Statistics ─────────────────────────────────────────────────────────────────

def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 6) if xs else None


def _sd(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    mu = sum(xs) / len(xs)
    return round((sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5, 6)


def _ttest_one_sample(vals: list[float], null: float = 0.0) -> dict:
    """One-sample t-test (two-tailed, then extract one-sided p for H1: mean > null)."""
    n = len(vals)
    if n < 2:
        return {"t": None, "df": None, "p_one_sided": None, "cohens_d": None}
    mu = sum(vals) / n
    sd = (sum((x - mu) ** 2 for x in vals) / (n - 1)) ** 0.5
    if sd == 0:
        return {"t": None, "df": n - 1, "p_one_sided": 0.0 if mu > null else 1.0,
                "cohens_d": None}
    t = (mu - null) / (sd / math.sqrt(n))
    df = n - 1
    # One-sided p (H1: mean > null): p = P(T > t | df)
    # Normal approximation for large df
    z = t
    if z <= 0:
        p = 1.0
    else:
        tt = 1 / (1 + 0.2316419 * z)
        poly = tt * (0.319381530 + tt * (-0.356563782 + tt * (1.781477937
                     + tt * (-1.821255978 + tt * 1.330274429))))
        p = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
        p = max(p, 1e-10)
    cohens_d = round((mu - null) / sd, 4)
    return {
        "t": round(t, 4),
        "df": df,
        "p_one_sided": round(p, 6),
        "cohens_d": cohens_d,
    }


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(seeds: list[int]) -> dict:
    per_seed = []
    missing = []
    for seed in seeds:
        b = _load_seed_metrics(seed, "baseline")
        c = _load_seed_metrics(seed, "control")
        if b is None or c is None:
            missing.append(seed)
            continue
        per_seed.append({
            "seed": seed,
            "baseline_survival":  b["survival_rate"],
            "control_survival":   c["survival_rate"],
            "survival_diff":      round((b["survival_rate"] or 0) - (c["survival_rate"] or 0), 6),
            "baseline_roi":       b["energy_roi"],
            "control_roi":        c["energy_roi"],
            "roi_diff":           round((b["energy_roi"] or 0) - (c["energy_roi"] or 0), 6),
        })

    n = len(per_seed)
    surv_diffs = [r["survival_diff"] for r in per_seed]
    roi_diffs  = [r["roi_diff"]      for r in per_seed]

    surv_test = _ttest_one_sample(surv_diffs, null=0.0)
    roi_test  = _ttest_one_sample(roi_diffs,  null=0.0)

    # P5 original criterion: ROI >= 25% higher (ratio >= 1.25)
    baseline_rois = [r["baseline_roi"] for r in per_seed if r["baseline_roi"]]
    control_rois  = [r["control_roi"]  for r in per_seed if r["control_roi"]]
    mean_b_roi = _mean(baseline_rois)
    mean_c_roi = _mean(control_rois)
    roi_ratio  = round(mean_b_roi / mean_c_roi, 4) if (mean_b_roi and mean_c_roi and mean_c_roi > 0) else None

    return {
        "experiment":    "P5_rerun",
        "label":         "EXPLORATORY — survival rate differential, n=30",
        "generated":     datetime.now().isoformat(),
        "note":          ("This is an exploratory rerun addressing underpowered original P5 "
                          "test (d=0.28, n=15). Survival_rate_differential is the new primary "
                          "metric. Energy ROI is retained as secondary for continuity. "
                          "Results must be labeled exploratory in the paper."),
        "n_seeds":       n,
        "missing_seeds": missing,

        "primary_metric_exploratory": {
            "metric":          "survival_rate differential (P1 - P0, per paired seed)",
            "mean_diff":       _mean(surv_diffs),
            "sd_diff":         _sd(surv_diffs),
            "mean_baseline":   _mean([r["baseline_survival"] for r in per_seed]),
            "mean_control":    _mean([r["control_survival"]  for r in per_seed]),
            "t_test":          surv_test,
            "H1_direction_met": (_mean(surv_diffs) or 0) > 0,
            "p_one_sided":     surv_test["p_one_sided"],
            "significant_at_0_05": (surv_test["p_one_sided"] or 1.0) < 0.05,
        },

        "secondary_metric_original": {
            "metric":         "energy ROI differential (P1 - P0, per paired seed)",
            "mean_diff":      _mean(roi_diffs),
            "sd_diff":        _sd(roi_diffs),
            "mean_baseline_roi": mean_b_roi,
            "mean_control_roi":  mean_c_roi,
            "roi_ratio":      roi_ratio,
            "t_test":         roi_test,
            "original_p5_criterion_met": roi_ratio is not None and roi_ratio >= 1.25,
        },

        "per_seed": sorted(per_seed, key=lambda r: r["seed"]),
    }


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_summary(result: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nP5 rerun summary: {out_path}")

    pm = result["primary_metric_exploratory"]
    sm = result["secondary_metric_original"]
    tt = pm["t_test"]

    print(f"\n  [EXPLORATORY] Survival rate differential (P1 - P0), n={result['n_seeds']}")
    print(f"  Mean diff:     {pm['mean_diff']:.4f}  (sd={pm['sd_diff']:.4f})")
    print(f"  Baseline mean: {pm['mean_baseline']:.4f}")
    print(f"  Control mean:  {pm['mean_control']:.4f}")
    print(f"  t({tt['df']}) = {tt['t']}  p(one-sided) = {tt['p_one_sided']}  "
          f"d = {tt['cohens_d']}")
    print(f"  Significant at 0.05: {'YES' if pm['significant_at_0_05'] else 'NO'}")

    print(f"\n  [Secondary] Energy ROI differential")
    print(f"  Mean ROI baseline: {sm['mean_baseline_roi']}")
    print(f"  Mean ROI control:  {sm['mean_control_roi']}")
    print(f"  ROI ratio: {sm['roi_ratio']}x  "
          f"(original P5 criterion >=1.25: {'MET' if sm['original_p5_criterion_met'] else 'NOT MET'})")

    if result["missing_seeds"]:
        print(f"\n  WARNING: missing seeds: {result['missing_seeds']}")

    print(f"\n  Per-seed survival differential:")
    print(f"  {'seed':>4}  {'P1_surv':>8}  {'P0_surv':>8}  {'diff':>8}  {'P1_roi':>8}  {'P0_roi':>8}")
    for r in result["per_seed"]:
        print(f"  {r['seed']:>4d}  {r['baseline_survival']:>8.4f}  "
              f"{r['control_survival']:>8.4f}  {r['survival_diff']:>+8.4f}  "
              f"{(r['baseline_roi'] or 0):>8.6f}  {(r['control_roi'] or 0):>8.6f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="P5 rerun -- survival rate differential, n=30 (EXPLORATORY)")
    parser.add_argument("--seeds",        type=int, nargs=2, default=[15, 30],
                        metavar=("START", "END"),
                        help="Seeds to run (default: 15 30, extending existing 0-14)")
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Load existing data and regenerate summary")
    args = parser.parse_args()

    seeds_to_run = list(range(args.seeds[0], args.seeds[1]))
    all_seeds    = list(range(30))
    workers      = args.workers or max(1, (os.cpu_count() or 4) - 1)

    print(f"P5 rerun (EXPLORATORY)  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds to run: {seeds_to_run}  Workers: {workers}")
    print(f"Existing data: {CAMPAIGN_DIR}")
    print(f"New data:      {DATA_DIR}")

    if not args.analyze_only and not args.dry_run:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        run_new_seeds(seeds_to_run, workers, args.dry_run)

    if args.dry_run:
        print("\nDry run complete.")
        return

    result = analyze(all_seeds)
    out_path = DATA_DIR / "p5_summary.json"
    write_summary(result, out_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
