"""P3 reanalysis at preregistered 100-epoch window (epochs 400-499).

The initial analysis script (analyze_p3_confirmatory.py) used SUSTAINED_WINDOW=20.
The preregistration specifies epochs 400-499 as the primary window.
This script uses the preregistered window and is the operative confirmatory analysis.
The 20-epoch results are included as a sensitivity check only.
"""

import json
import math
import statistics
from pathlib import Path
from scipy import stats

DATA_ROOT = Path("data")
CONDITIONS = ["p3_unconstrained", "p3b_constrained", "p3a_constrained"]
SEEDS = list(range(10))
WINDOW_START = 400
WINDOW_END = 500  # epochs 400-499 inclusive


def load(cond, seed):
    p = DATA_ROOT / cond / f"seed_{seed}" / "epoch_series.json"
    return json.loads(p.read_text()) if p.exists() else None


def qr_window(series, start, end):
    tail = [r for r in series if start <= r["epoch"] < end]
    vals = [r["query_rate"] for r in tail if r.get("query_rate") is not None]
    return round(statistics.mean(vals), 6) if vals else None


def sss_window(series, start, end):
    tail = [r for r in series if start <= r["epoch"] < end]
    te = [r["type_entropy"] for r in tail if r.get("type_entropy") is not None]
    qrc = [r["qrc"] for r in tail if r.get("qrc") is not None]
    if not te or not qrc:
        return None
    return round(statistics.mean(te) * statistics.mean(qrc), 6)


def elr_full(series):
    vals = [r["query_rate"] for r in series if r.get("query_rate") is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v < 0.05) / len(vals), 6)


def pfr_full(series):
    costs = [r["ethical_cost"] for r in series if r.get("ethical_cost") is not None]
    if not costs:
        return None
    return round(sum(1 for c in costs if c > 0) / len(costs), 6)


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    pooled = math.sqrt((statistics.variance(a) + statistics.variance(b)) / 2)
    return round((ma - mb) / pooled, 4) if pooled else float("nan")


def run_mwu(a, b, label_a, label_b, alternative="less"):
    u, p = stats.mannwhitneyu(a, b, alternative=alternative)
    d = cohens_d(a, b)
    return {
        "U": round(float(u), 4),
        "p": round(float(p), 6),
        "d": d,
        "mean_a": round(statistics.mean(a), 4),
        "mean_b": round(statistics.mean(b), 4),
        "label_a": label_a,
        "label_b": label_b,
        "alternative": alternative,
    }


def main():
    # Load per-seed metrics
    results = {}
    for cond in CONDITIONS:
        results[cond] = {}
        for seed in SEEDS:
            s = load(cond, seed)
            if s:
                results[cond][seed] = {
                    "qr100": qr_window(s, WINDOW_START, WINDOW_END),
                    "sss100": sss_window(s, WINDOW_START, WINDOW_END),
                    "qr20":  qr_window(s, 480, 500),
                    "elr":   elr_full(s),
                    "pfr":   pfr_full(s) if "unconstrained" not in cond else None,
                    "n_epochs": len(s),
                }

    print("=" * 80)
    print("PROTOCOL 3 CONFIRMATORY ANALYSIS — PREREGISTERED 100-EPOCH WINDOW")
    print("Primary window: epochs 400-499 (per preregistration DOI 10.5281/zenodo.19096602)")
    print("=" * 80)
    print()

    # Per-seed table
    for cond in CONDITIONS:
        print(f"--- {cond.upper()} ---")
        print(f"{'Seed':>6}  {'qr_final':>10}  {'sss':>10}  {'elr':>10}  {'pfr':>10}  {'epochs':>7}")
        for seed in SEEDS:
            if seed not in results[cond]:
                print(f"{seed:>6}  MISSING")
                continue
            r = results[cond][seed]
            qr  = f"{r['qr100']:.6f}"  if r['qr100']  is not None else "None"
            sss = f"{r['sss100']:.6f}" if r['sss100'] is not None else "None"
            el  = f"{r['elr']:.6f}"   if r['elr']   is not None else "None"
            pf  = f"{r['pfr']:.6f}"   if r['pfr']   is not None else "    n/a"
            print(f"{seed:>6}  {qr:>10}  {sss:>10}  {el:>10}  {pf:>10}  {r['n_epochs']:>7}")
        print()

    # Descriptives
    print("--- DESCRIPTIVE SUMMARY (primary 100-epoch window) ---")
    for cond in CONDITIONS:
        qr_vals  = [r["qr100"]  for r in results[cond].values() if r["qr100"]  is not None]
        sss_vals = [r["sss100"] for r in results[cond].values() if r["sss100"] is not None]
        print(f"  {cond}")
        print(f"    qr_final : mean={statistics.mean(qr_vals):.4f}  "
              f"median={statistics.median(qr_vals):.4f}  "
              f"sd={statistics.stdev(qr_vals):.4f}")
        print(f"    sss      : mean={statistics.mean(sss_vals):.4f}  "
              f"median={statistics.median(sss_vals):.4f}  "
              f"sd={statistics.stdev(sss_vals):.4f}")
    print()

    # Preregistered tests
    print("=" * 80)
    print("PREREGISTERED TESTS")
    print("=" * 80)

    qr_unc = [r["qr100"] for r in results["p3_unconstrained"].values() if r["qr100"] is not None]
    qr_3b  = [r["qr100"] for r in results["p3b_constrained"].values()  if r["qr100"] is not None]
    qr_3a  = [r["qr100"] for r in results["p3a_constrained"].values()  if r["qr100"] is not None]

    print()
    print("PRIMARY — H1: 3B < unconstrained (one-tailed Mann-Whitney U)")
    t1 = run_mwu(qr_3b, qr_unc, "3B", "unconstrained", alternative="less")
    print(f"  U={t1['U']}  p={t1['p']}  Cohen's d={t1['d']}")
    print(f"  mean_3B={t1['mean_a']:.4f}  mean_unc={t1['mean_b']:.4f}")
    if t1["p"] < 0.05:
        verdict1 = "CONFIRMED — opacity reduces query rate (p < .05)"
    elif t1["mean_a"] > t1["mean_b"]:
        verdict1 = "INVERTED — 3B query rate higher than unconstrained"
    else:
        verdict1 = "NULL — gaming persists regardless of opacity"
    print(f"  RESULT: {verdict1}")

    print()
    print("CONTROL — H2: 3A > 3B (one-tailed Mann-Whitney U)")
    t2 = run_mwu(qr_3a, qr_3b, "3A", "3B", alternative="greater")
    print(f"  U={t2['U']}  p={t2['p']}  Cohen's d={t2['d']}")
    print(f"  mean_3A={t2['mean_a']:.4f}  mean_3B={t2['mean_b']:.4f}")
    if t2["p"] < 0.05:
        verdict2 = "CONFIRMED — 3A > 3B (p < .05); boundary-opacity isolates as mechanism"
    else:
        verdict2 = "NULL — stochasticity and opacity not distinguishable at n=10"
    print(f"  RESULT: {verdict2}")

    # Sensitivity check: 20-epoch window
    print()
    print("=" * 80)
    print("SENSITIVITY CHECK — 20-epoch window (epochs 480-499, post-hoc only)")
    print("=" * 80)
    sq_unc = [r["qr20"] for r in results["p3_unconstrained"].values() if r["qr20"] is not None]
    sq_3b  = [r["qr20"] for r in results["p3b_constrained"].values()  if r["qr20"] is not None]
    sq_3a  = [r["qr20"] for r in results["p3a_constrained"].values()  if r["qr20"] is not None]
    print(f"  mean_unc={statistics.mean(sq_unc):.4f}  mean_3B={statistics.mean(sq_3b):.4f}  mean_3A={statistics.mean(sq_3a):.4f}")
    u1s, p1s = stats.mannwhitneyu(sq_3b, sq_unc, alternative="less")
    u2s, p2s = stats.mannwhitneyu(sq_3a, sq_3b, alternative="greater")
    d1s = cohens_d(sq_3b, sq_unc)
    d2s = cohens_d(sq_3a, sq_3b)
    print(f"  H1: U={u1s:.1f}  p={p1s:.4f}  d={d1s}")
    print(f"  H2: U={u2s:.1f}  p={p2s:.4f}  d={d2s}")
    if statistics.mean(sq_3b) > statistics.mean(sq_unc):
        print("  H1 direction: INVERTED (consistent with 100-epoch primary result)")
    if p2s < 0.05:
        print("  H2: CONFIRMED (consistent with 100-epoch primary result)")


if __name__ == "__main__":
    main()
