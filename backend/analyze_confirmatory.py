"""Confirmatory analysis: compute per-seed metrics for H1, H2, H3.

Reads epoch_series.json files produced by each confirmatory run and computes:
  - sustained_structure_score  (H2 metric: mean(type_entropy) * mean(qrc) over final 20 epochs)
  - exploitation_loop_rate     (H1 proxy: fraction of epochs where query_rate < 0.05)
  - gaming_flag                (constrained only: ethical_cost AND type_entropy both declining)

Usage:
    cd backend
    python analyze_confirmatory.py

Expects data layout:
    data/p2_constrained/seed_{N}/epoch_series.json
    data/p2_unconstrained/seed_{N}/epoch_series.json
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

SEEDS = list(range(10))
CONDITIONS = ["constrained", "unconstrained"]
DATA_ROOT = Path("data")
SUSTAINED_WINDOW = 20     # final N epochs for sustained_structure_score
COLLAPSE_THRESHOLD = 0.05  # query_rate below which epoch is "collapsed"
GAMING_HALF = 0.5          # second half of run used for trend detection


def load_series(condition: str, seed: int) -> list[dict] | None:
    path = DATA_ROOT / f"p2_{condition}" / f"seed_{seed}" / "epoch_series.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def sustained_structure_score(series: list[dict]) -> float | None:
    """mean(type_entropy) × mean(qrc) over the final SUSTAINED_WINDOW epochs."""
    tail = series[-SUSTAINED_WINDOW:]
    te_vals = [r["type_entropy"] for r in tail if r["type_entropy"] is not None]
    qrc_vals = [r["qrc"] for r in tail if r["qrc"] is not None]
    if not te_vals or not qrc_vals:
        return None
    return round((sum(te_vals) / len(te_vals)) * (sum(qrc_vals) / len(qrc_vals)), 6)


def exploitation_loop_rate(series: list[dict]) -> float | None:
    """Fraction of epochs where query_rate < COLLAPSE_THRESHOLD (H1 proxy)."""
    rates = [r["query_rate"] for r in series if r["query_rate"] is not None]
    if not rates:
        return None
    collapsed = sum(1 for r in rates if r < COLLAPSE_THRESHOLD)
    return round(collapsed / len(rates), 6)


def gaming_flag(series: list[dict]) -> bool:
    """True if ethical_cost AND type_entropy both decline in the second half.

    Gaming pattern: agent learned to minimally comply with ethical tax
    (declining cost) without maintaining genuine interrogative diversity
    (declining entropy). Observed in pilot seed 2 constrained.
    """
    n = len(series)
    if n < 4:
        return False
    split = n // 2
    second_half = series[split:]

    costs = [r["ethical_cost"] for r in second_half if r["ethical_cost"] is not None]
    entropies = [r["type_entropy"] for r in second_half if r["type_entropy"] is not None]

    if len(costs) < 4 or len(entropies) < 4:
        return False

    def _slope(vals: list[float]) -> float:
        n = len(vals)
        xs = list(range(n))
        xm = sum(xs) / n
        ym = sum(vals) / n
        num = sum((xs[i] - xm) * (vals[i] - ym) for i in range(n))
        den = sum((x - xm) ** 2 for x in xs)
        return num / den if den > 0 else 0.0

    cost_slope = _slope(costs)
    entropy_slope = _slope(entropies)
    # Both declining (negative slope)
    return cost_slope < 0 and entropy_slope < 0


def main() -> None:
    results: dict[str, dict[int, dict]] = {c: {} for c in CONDITIONS}
    missing: list[str] = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            series = load_series(condition, seed)
            if series is None:
                missing.append(f"{condition}/seed_{seed}")
                continue
            sss = sustained_structure_score(series)
            elr = exploitation_loop_rate(series)
            gf = gaming_flag(series) if condition == "constrained" else False
            results[condition][seed] = {
                "sustained_structure_score": sss,
                "exploitation_loop_rate": elr,
                "gaming_flag": gf,
                "n_epochs": len(series),
            }

    # --- Report ---
    print("=" * 70)
    print("CONFIRMATORY ANALYSIS — RAW PER-SEED METRICS")
    print("=" * 70)
    print(f"sustained_structure_window = {SUSTAINED_WINDOW} epochs")
    print(f"collapse_threshold         = {COLLAPSE_THRESHOLD}")
    print()

    for condition in CONDITIONS:
        print(f"--- {condition.upper()} ---")
        print(f"{'Seed':>6}  {'sss':>12}  {'elr':>12}  {'gaming':>8}  {'epochs':>8}")
        for seed in SEEDS:
            if seed not in results[condition]:
                print(f"{seed:>6}  {'MISSING':>12}")
                continue
            r = results[condition][seed]
            gf_str = "[GAMING]" if r["gaming_flag"] else ""
            sss_str = f"{r['sustained_structure_score']:.6f}" if r["sustained_structure_score"] is not None else "None"
            elr_str = f"{r['exploitation_loop_rate']:.6f}" if r["exploitation_loop_rate"] is not None else "None"
            print(f"{seed:>6}  {sss_str:>12}  {elr_str:>12}  {gf_str:>8}  {r['n_epochs']:>8}")
        print()

    if missing:
        print(f"MISSING RUNS ({len(missing)}): {', '.join(missing)}")
        print()

    # Summary stats per condition (for reference, not the preregistered test)
    print("--- SUMMARY (descriptive only — Mann-Whitney is the preregistered test) ---")
    for condition in CONDITIONS:
        sss_vals = [r["sustained_structure_score"] for r in results[condition].values()
                    if r["sustained_structure_score"] is not None]
        elr_vals = [r["exploitation_loop_rate"] for r in results[condition].values()
                    if r["exploitation_loop_rate"] is not None]
        if sss_vals:
            sss_sd = statistics.stdev(sss_vals) if len(sss_vals) > 1 else 0.0
            print(f"{condition:>14} sss: mean={statistics.mean(sss_vals):.4f}  "
                  f"median={statistics.median(sss_vals):.4f}  "
                  f"stdev={sss_sd:.4f}")
        if elr_vals:
            elr_sd = statistics.stdev(elr_vals) if len(elr_vals) > 1 else 0.0
            print(f"{condition:>14} elr: mean={statistics.mean(elr_vals):.4f}  "
                  f"median={statistics.median(elr_vals):.4f}  "
                  f"stdev={elr_sd:.4f}")
    print()

    gaming_seeds = [s for s, r in results["constrained"].items() if r.get("gaming_flag")]
    if gaming_seeds:
        print(f"GAMING PATTERN observed in constrained seeds: {gaming_seeds}")
        print("  (declining ethical_cost AND declining type_entropy in second half)")
    else:
        print("No gaming pattern detected in constrained seeds.")


if __name__ == "__main__":
    main()
