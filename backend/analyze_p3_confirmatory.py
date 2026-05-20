"""Protocol 3 confirmatory analysis.

Computes per-seed metrics and runs preregistered statistical tests for P3.

Preregistered comparisons (in priority order):
  Primary:  3B vs unconstrained — Mann-Whitney U one-tailed
            H: p3b_constrained query_rate < p3_unconstrained query_rate
            (epistemic opacity of enforcement disrupts exploitation strategy)
  Control:  3A vs 3B — Mann-Whitney U one-tailed
            H: p3a_constrained query_rate > p3b_constrained query_rate
            (stochasticity alone does not produce the 3B effect; boundary-opacity
            is the active mechanism)
  Null alt: if 3B ≈ unconstrained, gaming persists regardless of opacity —
            strengthens architectural necessity claim.

Preregistration DOI: 10.5281/zenodo.19096602

Metrics (matching P2 analysis):
  query_rate_final   — mean query_rate over final SUSTAINED_WINDOW epochs (primary)
  sustained_structure_score (SSS) — mean(type_entropy) × mean(qrc), final window
  exploitation_loop_rate (ELR)    — fraction of epochs where query_rate < 0.05
  penalty_fired_rate — fraction of epochs where ethical_cost > 0 (constrained only)

Usage:
    cd backend
    python analyze_p3_confirmatory.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy import stats  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = list(range(10))
CONDITIONS = ["p3_unconstrained", "p3b_constrained", "p3a_constrained"]
DATA_ROOT = Path("data")
SUSTAINED_WINDOW = 20      # final N epochs for sustained metrics
COLLAPSE_THRESHOLD = 0.05  # query_rate below which epoch counts as collapsed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_series(condition: str, seed: int) -> list[dict] | None:
    path = DATA_ROOT / condition / f"seed_{seed}" / "epoch_series.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def query_rate_final(series: list[dict]) -> float | None:
    """Mean query_rate over the final SUSTAINED_WINDOW epochs (primary metric)."""
    tail = series[-SUSTAINED_WINDOW:]
    vals = [r["query_rate"] for r in tail if r.get("query_rate") is not None]
    return round(statistics.mean(vals), 6) if vals else None


def sustained_structure_score(series: list[dict]) -> float | None:
    """mean(type_entropy) × mean(qrc) over the final SUSTAINED_WINDOW epochs."""
    tail = series[-SUSTAINED_WINDOW:]
    te = [r["type_entropy"] for r in tail if r.get("type_entropy") is not None]
    qrc = [r["qrc"] for r in tail if r.get("qrc") is not None]
    if not te or not qrc:
        return None
    return round(statistics.mean(te) * statistics.mean(qrc), 6)


def exploitation_loop_rate(series: list[dict]) -> float | None:
    """Fraction of epochs where query_rate < COLLAPSE_THRESHOLD."""
    vals = [r["query_rate"] for r in series if r.get("query_rate") is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v < COLLAPSE_THRESHOLD) / len(vals), 6)


def penalty_fired_rate(series: list[dict]) -> float | None:
    """Fraction of epochs where ethical_cost > 0 (proxy for penalty_fired)."""
    costs = [r["ethical_cost"] for r in series if r.get("ethical_cost") is not None]
    if not costs:
        return None
    return round(sum(1 for c in costs if c > 0) / len(costs), 6)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d = (mean_a - mean_b) / pooled_sd. Sign preserved."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    pooled_sd = math.sqrt((var_a + var_b) / 2)
    if pooled_sd == 0:
        return float("nan")
    return round((mean_a - mean_b) / pooled_sd, 4)


def run_mannwhitney(a: list[float], b: list[float], label_a: str, label_b: str,
                    alternative: str = "less") -> dict:
    """One-tailed Mann-Whitney U. alternative='less' tests a < b."""
    result = stats.mannwhitneyu(a, b, alternative=alternative)
    d = cohens_d(a, b)
    return {
        "comparison": f"{label_a} vs {label_b}",
        "alternative": f"{label_a} {alternative} {label_b}",
        "U": round(float(result.statistic), 4),
        "p": round(float(result.pvalue), 6),
        "cohens_d": d,
        "mean_a": round(statistics.mean(a), 6),
        "mean_b": round(statistics.mean(b), 6),
        "n_a": len(a),
        "n_b": len(b),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load and compute per-seed metrics
    results: dict[str, dict[int, dict]] = {c: {} for c in CONDITIONS}
    missing: list[str] = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            series = load_series(condition, seed)
            if series is None:
                missing.append(f"{condition}/seed_{seed}")
                continue
            results[condition][seed] = {
                "query_rate_final":          query_rate_final(series),
                "sustained_structure_score": sustained_structure_score(series),
                "exploitation_loop_rate":    exploitation_loop_rate(series),
                "penalty_fired_rate":        penalty_fired_rate(series) if "unconstrained" not in condition else None,
                "n_epochs":                  len(series),
            }

    # --- Per-seed table ---
    print("=" * 80)
    print("PROTOCOL 3 CONFIRMATORY ANALYSIS")
    print("Preregistration DOI: 10.5281/zenodo.19096602")
    print("=" * 80)
    print(f"sustained_window   = {SUSTAINED_WINDOW} epochs")
    print(f"collapse_threshold = {COLLAPSE_THRESHOLD}")
    print()

    for condition in CONDITIONS:
        print(f"--- {condition.upper()} ---")
        print(f"{'Seed':>6}  {'qr_final':>10}  {'sss':>10}  {'elr':>10}  {'pfr':>8}  {'epochs':>8}")
        for seed in SEEDS:
            if seed not in results[condition]:
                print(f"{seed:>6}  {'MISSING':>10}")
                continue
            r = results[condition][seed]
            qr  = f"{r['query_rate_final']:.6f}"         if r['query_rate_final']          is not None else "None"
            sss = f"{r['sustained_structure_score']:.6f}" if r['sustained_structure_score'] is not None else "None"
            elr = f"{r['exploitation_loop_rate']:.6f}"   if r['exploitation_loop_rate']    is not None else "None"
            pfr = f"{r['penalty_fired_rate']:.6f}"       if r['penalty_fired_rate']        is not None else "    n/a"
            print(f"{seed:>6}  {qr:>10}  {sss:>10}  {elr:>10}  {pfr:>8}  {r['n_epochs']:>8}")
        print()

    if missing:
        print(f"MISSING RUNS ({len(missing)}): {', '.join(missing)}")
        print()

    # --- Descriptive summary ---
    print("--- DESCRIPTIVE SUMMARY ---")
    for condition in CONDITIONS:
        qr_vals  = [r["query_rate_final"]          for r in results[condition].values() if r["query_rate_final"]          is not None]
        sss_vals = [r["sustained_structure_score"] for r in results[condition].values() if r["sustained_structure_score"] is not None]
        elr_vals = [r["exploitation_loop_rate"]    for r in results[condition].values() if r["exploitation_loop_rate"]    is not None]
        if qr_vals:
            print(f"  {condition}")
            print(f"    query_rate_final : mean={statistics.mean(qr_vals):.4f}  "
                  f"median={statistics.median(qr_vals):.4f}  "
                  f"sd={statistics.stdev(qr_vals):.4f}")
            print(f"    sss              : mean={statistics.mean(sss_vals):.4f}  "
                  f"median={statistics.median(sss_vals):.4f}  "
                  f"sd={statistics.stdev(sss_vals):.4f}")
            if all(v == 0.0 for v in elr_vals):
                print(f"    elr              : 0.0 across all seeds (H1 not evaluable — matches P2)")
            else:
                print(f"    elr              : mean={statistics.mean(elr_vals):.4f}  "
                      f"median={statistics.median(elr_vals):.4f}")
    print()

    # --- Preregistered statistical tests ---
    print("=" * 80)
    print("PREREGISTERED TESTS (priority order)")
    print("=" * 80)

    def get_qr(condition: str) -> list[float]:
        return [r["query_rate_final"] for r in results[condition].values()
                if r["query_rate_final"] is not None]

    qr_unc = get_qr("p3_unconstrained")
    qr_3b  = get_qr("p3b_constrained")
    qr_3a  = get_qr("p3a_constrained")

    # --- Primary: 3B vs unconstrained (H: 3B < unconstrained) ---
    print()
    print("PRIMARY — 3B vs unconstrained")
    print("  H: p3b_constrained query_rate < p3_unconstrained query_rate")
    print("  (epistemic opacity disrupts exploitation strategy)")
    t1 = run_mannwhitney(qr_3b, qr_unc, "3b", "unconstrained", alternative="less")
    print(f"  U={t1['U']}  p={t1['p']}  Cohen's d={t1['cohens_d']}  "
          f"(mean_3b={t1['mean_a']:.4f}, mean_unc={t1['mean_b']:.4f})")
    if t1["p"] < 0.05:
        print("  RESULT: p < 0.05 — supports primary hypothesis (opacity reduces query rate)")
    elif t1["mean_a"] > t1["mean_b"]:
        print("  RESULT: INVERTED — 3B query rate higher than unconstrained (gaming or entropy collapse)")
    else:
        print("  RESULT: p >= 0.05 — null (gaming persists regardless of opacity)")
        print("          → strengthens architectural necessity claim per preregistration")

    # --- Control: 3A vs 3B (H: 3A > 3B) ---
    print()
    print("CONTROL — 3A vs 3B")
    print("  H: p3a_constrained query_rate > p3b_constrained query_rate")
    print("  (stochasticity alone does not produce 3B effect; boundary-opacity is the mechanism)")
    t2 = run_mannwhitney(qr_3a, qr_3b, "3a", "3b", alternative="greater")
    print(f"  U={t2['U']}  p={t2['p']}  Cohen's d={t2['cohens_d']}  "
          f"(mean_3a={t2['mean_a']:.4f}, mean_3b={t2['mean_b']:.4f})")
    if t2["p"] < 0.05:
        print("  RESULT: p < 0.05 — boundary-opacity isolates as mechanism (3A > 3B confirmed)")
    else:
        print("  RESULT: p >= 0.05 — stochasticity and opacity not distinguishable at n=10")

    print()
    print("=" * 80)
    print("NOTE: ELR = 0.0 across all conditions is expected (matches P2 finding).")
    print("      H1 (exploitation_loop_rate) remains not evaluable — operationalization gap")
    print("      documented in preregistration deviation log.")
    print("=" * 80)


if __name__ == "__main__":
    main()
