"""Experiment 1 — Tax sweep: critical threshold mapping (fine resolution).

The confirmatory campaign tested q ∈ {0.0, 1.2, 1.5, 3.0, 5.0} (n=15 per
condition). That leaves the transition zone (q < 1.2) unsampled at fine
resolution. This experiment fills the gap.

New conditions (run fresh, n=15 each):
  q_0.10  q_0.20  q_0.30  q_0.40  q_0.50
  q_0.60  q_0.70  q_0.80  q_0.90  q_1.00

Note: q=0.50 and q=1.00 also appear in Exp 14 (coupling window). Existing
data from exp14 is reused if present; otherwise run fresh.

Existing data reused:
  q=0.00 → data/campaign/control/          (P0 baseline)
  q=1.20 → data/campaign/low_pressure/
  q=1.50 → data/campaign/baseline/
  q=3.00 → data/campaign/high_pressure/
  q=5.00 → data/campaign/extreme/
  q=0.50 → data/exp14_coupling_window/seed_*_qcost_0.5_rcost_0.8/ (if present)
  q=1.00 → data/exp14_coupling_window/seed_*_qcost_1.0_rcost_0.8/ (if present)

Analysis
--------
  Fits a logistic curve P(crys) = σ(a + b·q) to the pooled n=15 data per
  condition and estimates:
    - critical_q: inflection point of the logistic (50% crys probability)
    - transition_band_low/high: 10% and 90% crys thresholds
    - transition_width: high - low

Output
------
  data/exp1_tax_sweep/seed_{s:02d}_q{q:.2f}/manifest.json  (new conditions only)
  data/exp1_tax_sweep/exp1_summary.json

Usage
-----
  python run_exp1_tax_sweep.py
  python run_exp1_tax_sweep.py --workers 4
  python run_exp1_tax_sweep.py --dry-run
  python run_exp1_tax_sweep.py --analyze-only
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

SEEDS        = list(range(15))
EPOCHS       = 500
EPISODES     = 10
DECLARE_COST = 1.0
RESPOND_COST = 0.8

# New q values to run (not in existing campaign)
NEW_Q_VALUES = [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90]
# These may already exist from Exp14; runner checks before running
OPT_Q_VALUES = [0.50, 1.00]

DATA_ROOT    = Path(__file__).parent.parent / "data" / "exp1_tax_sweep"
CAMPAIGN_DIR = Path(__file__).parent.parent / "data" / "campaign"
EXP14_DIR    = Path(__file__).parent.parent / "data" / "exp14_coupling_window"


# ── Worker ─────────────────────────────────────────────────────────────────────

def _run_one(spec: dict) -> dict:
    """Run one seed at one query cost. Called in subprocess."""
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
        respond_cost=RESPOND_COST,
        output_dir=spec["output_dir"],
        device=spec.get("device", "cpu"),
    )
    engine = SimulationEngine(config=config)
    engine.run()
    return {"seed": spec["seed"], "q": spec["query_cost"]}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_manifest(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _is_crystallized(m: dict) -> bool | None:
    """Extract crystallized status from a manifest. Uses crystallization_epoch
    as the proxy (not-None → crystallized)."""
    if m is None:
        return None
    # New-style: explicit boolean field
    val = m.get("crystallized")
    if val is not None:
        return bool(val)
    fm = m.get("final_metrics", {})
    val = fm.get("crystallized")
    if val is not None:
        return bool(val)
    # Campaign-style: crystallization_epoch is None iff not crystallized
    crys_epoch = m.get("crystallization_epoch")
    if crys_epoch is not None:
        return True
    # If manifest exists but crystallization_epoch is None → not crystallized
    if "crystallization_epoch" in m:
        return False
    return None


def _load_seed_crystallized(seed: int, q: float) -> bool | None:
    """Return crystallized status from the best available source."""

    # 1. New exp1 directory
    p = DATA_ROOT / f"seed_{seed:02d}_q{q:.2f}" / "manifest.json"
    m = _load_manifest(p)
    if m is not None:
        return _is_crystallized(m)

    # 2. Exp14 directory (for q=0.5 and q=1.0)
    p2 = EXP14_DIR / f"seed_{seed:02d}_qcost_{q:.1f}_rcost_0.8" / "manifest.json"
    m2 = _load_manifest(p2)
    if m2 is not None:
        return _is_crystallized(m2)

    # 3. Original campaign directory (known q values)
    condition_map = {
        0.0:  "control",
        1.2:  "low_pressure",
        1.5:  "baseline",
        3.0:  "high_pressure",
        5.0:  "extreme",
    }
    cond = condition_map.get(q)
    if cond:
        p3 = CAMPAIGN_DIR / cond / f"seed_{seed:02d}" / "manifest.json"
        m3 = _load_manifest(p3)
        if m3 is not None:
            return _is_crystallized(m3)

    return None


# ── Campaign runner ─────────────────────────────────────────────────────────────

def run_campaign(seeds: list[int], workers: int, dry_run: bool) -> None:
    all_q = sorted(NEW_Q_VALUES + OPT_Q_VALUES)
    jobs  = []
    for q in all_q:
        for seed in seeds:
            # Skip if data already available from any source
            if _load_seed_crystallized(seed, q) is not None:
                continue
            out_dir = DATA_ROOT / f"seed_{seed:02d}_q{q:.2f}"
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
            jobs.append({"seed": seed, "query_cost": q, "output_dir": str(out_dir)})

    if not jobs:
        print("  No new jobs (all data present or reusable).")
        return

    print(f"\n  Running {len(jobs)} new seed/q pairs  (workers={workers})")
    if dry_run:
        for j in jobs[:8]:
            print(f"  DRY-RUN: q={j['query_cost']:.2f}  seed={j['seed']}")
        if len(jobs) > 8:
            print(f"  ... ({len(jobs)-8} more)")
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
                print(f"  [{done:3d}/{len(jobs)}]  q={r['q']:.2f}  seed={r['seed']:02d}")
            except Exception as e:
                j = futures[fut]
                print(f"  ERROR: q={j['query_cost']:.2f}  seed={j['seed']}: {e}")
    print(f"  Done in {time.time()-t0:.0f}s")


# ── Logistic fit ───────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def _logistic_log_likelihood(a: float, b: float, xs, ks, ns) -> float:
    ll = 0.0
    for x, k, n in zip(xs, ks, ns):
        p = _sigmoid(a + b * x)
        p = max(1e-10, min(1 - 1e-10, p))
        ll += k * math.log(p) + (n - k) * math.log(1 - p)
    return ll


def _fit_logistic(xs, ks, ns) -> dict:
    """Gradient ascent on logistic log-likelihood. Returns {a, b, ll}."""
    a, b = -2.0, 3.0
    best_ll = float("-inf")
    best    = (a, b)

    for lr in [0.05, 0.02, 0.005]:
        a, b = -2.0, 3.0
        for _ in range(10000):
            ll = _logistic_log_likelihood(a, b, xs, ks, ns)
            if ll > best_ll:
                best_ll = ll
                best    = (a, b)
            # Numerical gradient
            da = (_logistic_log_likelihood(a + 1e-5, b, xs, ks, ns) -
                  _logistic_log_likelihood(a - 1e-5, b, xs, ks, ns)) / 2e-5
            db = (_logistic_log_likelihood(a, b + 1e-5, xs, ks, ns) -
                  _logistic_log_likelihood(a, b - 1e-5, xs, ks, ns)) / 2e-5
            a += lr * da
            b += lr * db

    a, b = best
    # Critical q (inflection at P=0.5): a + b*q = 0 → q = -a/b
    critical_q = -a / b if b != 0 else None
    # P=0.10: a + b*q = logit(0.10) = -2.197
    # P=0.90: a + b*q = logit(0.90) = +2.197
    logit_10 = math.log(0.10 / 0.90)
    logit_90 = math.log(0.90 / 0.10)
    low  = (logit_10 - a) / b if b != 0 else None
    high = (logit_90 - a) / b if b != 0 else None

    return {
        "a":                    round(a, 4),
        "b":                    round(b, 4),
        "log_likelihood":       round(best_ll, 4),
        "critical_q":           round(critical_q, 4) if critical_q else None,
        "transition_band_low":  round(low, 4)        if low        else None,
        "transition_band_high": round(high, 4)       if high       else None,
        "transition_width":     round(high - low, 4) if (low and high) else None,
    }


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(seeds: list[int]) -> dict:
    # All q values — new runs + existing campaign + optional
    all_q_values = sorted(set(
        NEW_Q_VALUES + OPT_Q_VALUES +
        [0.0, 1.2, 1.5, 3.0, 5.0]   # from existing campaign
    ))

    rates = {}
    for q in all_q_values:
        crys_vals = [_load_seed_crystallized(s, q) for s in seeds]
        crys_vals = [v for v in crys_vals if v is not None]
        if not crys_vals:
            continue
        n_crys = sum(1 for v in crys_vals if v)
        rates[q] = {
            "n":      len(crys_vals),
            "n_crys": n_crys,
            "rate":   round(n_crys / len(crys_vals), 4),
        }

    # Logistic fit
    xs = [q for q, r in rates.items() if r["n"] > 0]
    ks = [rates[q]["n_crys"] for q in xs]
    ns = [rates[q]["n"]      for q in xs]
    fit = _fit_logistic(xs, ks, ns) if xs else None

    return {
        "experiment":   "1",
        "description":  "Tax sweep: critical threshold mapping",
        "generated":    datetime.now().isoformat(),
        "n_seeds":      len(seeds),
        "rates":        {f"{q:.2f}": rates[q] for q in sorted(rates)},
        "logistic_fit": fit,
    }


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_summary(result: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nExp 1 summary: {out_path}")

    print(f"\n  {'q':>6}  {'n':>3}  {'n_crys':>6}  {'rate':>6}")
    for q_str, r in sorted(result["rates"].items(), key=lambda kv: float(kv[0])):
        print(f"  {float(q_str):>6.2f}  {r['n']:>3}  {r['n_crys']:>6}  {r['rate']:>6.4f}")

    if result["logistic_fit"]:
        fit = result["logistic_fit"]
        print(f"\n  Logistic fit: P(crys) = sigmoid({fit['a']} + {fit['b']}*q)")
        print(f"  Critical q (P=0.50): {fit['critical_q']}")
        print(f"  Transition band (P=0.10 to 0.90): "
              f"{fit['transition_band_low']} to {fit['transition_band_high']}")
        print(f"  Transition width: {fit['transition_width']}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 1 -- Tax sweep: critical threshold mapping")
    parser.add_argument("--seeds",        type=int, nargs=2, default=[0, 15],
                        metavar=("START", "END"))
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    seeds   = list(range(args.seeds[0], args.seeds[1]))
    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    print(f"Exp 1 -- Tax sweep  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}  Workers: {workers}")
    print(f"New q values: {NEW_Q_VALUES}")
    print(f"Optional q values (reuse from Exp14 if available): {OPT_Q_VALUES}")
    print(f"Data: {DATA_ROOT}")

    if args.analyze_only:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        result = analyze(seeds)
        out_path = DATA_ROOT / "exp1_summary.json"
        write_summary(result, out_path)
        return

    if not args.dry_run:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

    run_campaign(seeds, workers, args.dry_run)

    if not args.dry_run:
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp1_summary.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
