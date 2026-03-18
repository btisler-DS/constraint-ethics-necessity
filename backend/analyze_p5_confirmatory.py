"""
Protocol 5 — Confirmatory Analysis Script
==========================================
Preregistered measures:
  M1  sacrifice_choice_rate (per epoch, null epochs excluded per deviation log)
  M2  Ethical output (framework_scores: utilitarian, deontological, virtue_ethics,
      self_interest) — mean over last 100 epochs
  M3  Convergence-Divergence Index (CDI) — Pearson r(scr, framework_score) over
      50-epoch rolling window; pre-computed in epoch records
  M4  Deception metric — std(framework scores) per agent
  M5  CDI Coupling — CDI(last 100 epochs) - CDI(first 100 epochs)
  M6  Sacrifice Persistence — linear slope of scr over epochs 300-500

Hypotheses:
  H1  Temporal span alone sufficient (predicted: REJECTED)
      Test: C3 vs C1 (span alone: no welfare coupling)
  H2  Prosocial constraints alone sufficient (predicted: REJECTED)
      Test: C2 vs C1 (welfare coupling alone: short span)
  H3  Joint necessity — both span + welfare required (predicted: SUPPORTED)
      Tests: C4 > C3, C4 > C2, C4 > C1
  H4  Boundary reversal in long+welfare condition (predicted: SUPPORTED)
      Tests: C4-frozen > C4, C1-frozen vs C1
  H5  CDI coupling positive only in C4 (predicted: M5_C4 > 0, others <= 0)

Statistical tests: Mann-Whitney U (one-tailed, alpha=0.05), Cohen's d
Null exclusion policy: null scr epochs excluded from all M1/M6 calculations
  (Deviation 2: long-span conditions 30-35% null; short-span 0.5%)
"""

import json
import glob
import os
import numpy as np
from scipy import stats
from datetime import datetime

# ---------------------------------------------------------------------------
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "analysis_p5")
ALPHA        = 0.05
os.makedirs(ANALYSIS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_condition(condition_name: str, seeds=range(10)) -> list[list[dict]]:
    """Return list of 10 epoch-series lists."""
    data = []
    for seed in seeds:
        fpath = os.path.join(RESULTS_DIR, f"p5_condition_{condition_name}_seed_{seed}.json")
        with open(fpath) as f:
            ep = json.load(f)
        data.append([e for e in ep if isinstance(e, dict)])
    return data


# ---------------------------------------------------------------------------
# Per-seed measure extractors
# ---------------------------------------------------------------------------

def scr_valid(ep: list[dict]) -> list[float]:
    return [e["sacrifice_choice_rate"] for e in ep
            if e.get("sacrifice_choice_rate") is not None]


def m1_mean_scr_last100(ep: list[dict]) -> float:
    """M1: mean scr over last 100 epochs (non-null)."""
    valid = [e["sacrifice_choice_rate"] for e in ep[-100:]
             if e.get("sacrifice_choice_rate") is not None]
    return float(np.mean(valid)) if valid else float("nan")


def m2_framework_last100(ep: list[dict]) -> dict:
    """M2: mean framework scores (per agent, per dimension) over last 100 epochs."""
    dims = ["utilitarian", "deontological", "virtue_ethics", "self_interest"]
    agents = ["agent_a", "agent_b", "agent_c"]
    acc = {ag: {d: [] for d in dims} for ag in agents}
    for e in ep[-100:]:
        fs = e.get("framework_scores")
        if not isinstance(fs, dict):
            continue
        for ag in agents:
            if ag not in fs:
                continue
            for d in dims:
                v = fs[ag].get(d)
                if v is not None:
                    acc[ag][d].append(v)
    return {ag: {d: float(np.mean(vals)) if vals else float("nan")
                 for d, vals in dim_vals.items()}
            for ag, dim_vals in acc.items()}


def m3_cdi_valid_window(ep: list[dict], window: int = 50) -> list[float | None]:
    """
    M3: CDI computed over a sliding 50-valid-epoch window (Deviation 3).

    Collects (scr, mean_framework_score) pairs from epochs where scr is non-null,
    then slides a window of `window` valid pairs across the series computing
    Pearson r at each position. Returns a list of CDI values indexed by the
    calendar epoch of the last pair in each window.

    For short-span conditions the pre-computed convergence_divergence_index is
    used where available (50 valid ≈ 50 calendar epochs). For long-span
    conditions this function provides the only non-null CDI values.
    """
    dims = ["utilitarian", "deontological", "virtue_ethics", "self_interest"]
    # Collect valid (epoch_idx, scr, mean_fs) triples
    valid_pairs: list[tuple[int, float, float]] = []
    for i, e in enumerate(ep):
        scr = e.get("sacrifice_choice_rate")
        if scr is None:
            continue
        fs = e.get("framework_scores")
        if not isinstance(fs, dict):
            continue
        # mean framework score across all agents and dimensions
        vals = []
        for ag_scores in fs.values():
            if isinstance(ag_scores, dict):
                vals.extend(v for d, v in ag_scores.items()
                            if d in dims and v is not None)
        if not vals:
            continue
        valid_pairs.append((i, float(scr), float(np.mean(vals))))

    if len(valid_pairs) < window:
        return [None] * len(ep)

    # Slide window over valid pairs
    cdi_at_epoch: dict[int, float] = {}
    for start in range(len(valid_pairs) - window + 1):
        w = valid_pairs[start: start + window]
        xs = np.array([p[1] for p in w])   # scr
        ys = np.array([p[2] for p in w])   # mean_fs
        if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
            continue
        r, _ = stats.pearsonr(xs, ys)
        last_epoch = w[-1][0]
        cdi_at_epoch[last_epoch] = float(r)

    return [cdi_at_epoch.get(i) for i in range(len(ep))]


def m4_deception_last100(ep: list[dict]) -> dict:
    """M4: mean deception metric (per agent) over last 100 epochs."""
    agents = ["agent_a", "agent_b", "agent_c"]
    acc = {ag: [] for ag in agents}
    for e in ep[-100:]:
        dm = e.get("deception_metric")
        if not isinstance(dm, dict):
            continue
        for ag in agents:
            v = dm.get(ag)
            if v is not None:
                acc[ag].append(v)
    return {ag: float(np.mean(vals)) if vals else float("nan")
            for ag, vals in acc.items()}


def m5_cdi_coupling(ep: list[dict], window: int = 50) -> float:
    """
    M5: CDI coupling = mean_CDI(last 50 valid pairs) - mean_CDI(first 50 valid pairs).
    Uses 50-valid-epoch window per Deviation 3.
    NaN if fewer than 100 valid pairs available (can't split into first/last 50).
    """
    cdi_series = m3_cdi_valid_window(ep, window=window)
    non_null   = [v for v in cdi_series if v is not None]
    if len(non_null) < 2:
        return float("nan")
    mid   = len(non_null) // 2
    first = non_null[:mid]
    last  = non_null[-mid:]
    return float(np.mean(last) - np.mean(first))


def m6_scr_slope(ep: list[dict], start: int = 300, end: int = 500) -> float:
    """M6: linear slope of scr over epochs start:end (non-null only)."""
    window = ep[start:end]
    pairs  = [(i, e["sacrifice_choice_rate"]) for i, e in enumerate(window)
              if e.get("sacrifice_choice_rate") is not None]
    if len(pairs) < 10:
        return float("nan")
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    slope, *_ = stats.linregress(xs, ys)
    return float(slope)


def null_rate(ep: list[dict]) -> float:
    nulls = sum(1 for e in ep if e.get("sacrifice_choice_rate") is None)
    return nulls / len(ep) * 100 if ep else float("nan")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d = (mean_a - mean_b) / pooled_std. Sign preserved."""
    a_, b_ = np.array(a), np.array(b)
    n_a, n_b = len(a_), len(b_)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled = np.sqrt(((n_a - 1) * np.std(a_, ddof=1) ** 2 +
                      (n_b - 1) * np.std(b_, ddof=1) ** 2) /
                     (n_a + n_b - 2))
    return float((np.mean(a_) - np.mean(b_)) / pooled) if pooled > 0 else 0.0


def mwu_one_tailed(a: list[float], b: list[float],
                   direction: str = "a_gt_b") -> tuple[float, float]:
    """
    One-tailed Mann-Whitney U.
    direction='a_gt_b': H1: a > b (p = prob under null that a <= b)
    Returns (U, p_one_tailed).
    """
    a_ = [v for v in a if not np.isnan(v)]
    b_ = [v for v in b if not np.isnan(v)]
    if not a_ or not b_:
        return float("nan"), float("nan")
    u, p_two = stats.mannwhitneyu(a_, b_, alternative="two-sided")
    if direction == "a_gt_b":
        p_one = p_two / 2 if np.mean(a_) >= np.mean(b_) else 1 - p_two / 2
    else:  # a_lt_b
        p_one = p_two / 2 if np.mean(a_) <= np.mean(b_) else 1 - p_two / 2
    return float(u), float(p_one)


def sig_stars(p: float) -> str:
    if np.isnan(p):   return "n/a"
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Condition summary
# ---------------------------------------------------------------------------

def summarize(name: str, data: list[list[dict]]) -> dict:
    m1  = [m1_mean_scr_last100(d) for d in data]
    m4  = [m4_deception_last100(d) for d in data]
    m5  = [m5_cdi_coupling(d)      for d in data]
    m6  = [m6_scr_slope(d)         for d in data]
    nr  = [null_rate(d)             for d in data]

    # Mean framework scores per agent (aggregate across seeds)
    fw_by_seed = [m2_framework_last100(d) for d in data]

    return {
        "condition": name,
        "M1_scr_last100":    m1,
        "M4_deception":      m4,
        "M5_cdi_coupling":   m5,
        "M6_scr_slope":      m6,
        "null_rate_pct":     nr,
        "framework_by_seed": fw_by_seed,
        # convenience scalars
        "M1_mean":  float(np.nanmean(m1)),
        "M1_sd":    float(np.nanstd(m1, ddof=1)),
        "M5_mean":  float(np.nanmean(m5)),
        "M5_valid_n": int(sum(1 for v in m5 if not np.isnan(v))),
        "M6_mean":  float(np.nanmean(m6)),
        "M6_sd":    float(np.nanstd(m6, ddof=1)),
        "null_rate_mean": float(np.nanmean(nr)),
    }


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def run_hypothesis_tests(s: dict) -> list[dict]:
    """
    s: dict of condition_name -> summary dict
    Returns list of hypothesis result dicts.
    """
    c1 = s["c1_short_individual"]["M1_scr_last100"]
    c2 = s["c2_short_welfare"]["M1_scr_last100"]
    c3 = s["c3_long_individual"]["M1_scr_last100"]
    c4 = s["c4_long_welfare"]["M1_scr_last100"]
    c1f = s["c1_short_individual_frozen"]["M1_scr_last100"]
    c4f = s["c4_long_welfare_frozen"]["M1_scr_last100"]

    tests = []

    # ------------------------------------------------------------------
    # H1: Temporal span alone sufficient (predicted REJECTED: C3 not > C1)
    u, p = mwu_one_tailed(c3, c1, direction="a_gt_b")
    d    = cohen_d(c3, c1)
    tests.append({
        "hypothesis":   "H1",
        "description":  "Temporal span alone sufficient (predicted rejected: C3 not > C1)",
        "comparison":   "C3 vs C1",
        "direction":    "C3 > C1",
        "U":            u,   "p_one_tailed": p,
        "cohen_d":      d,
        "sig":          sig_stars(p),
        "C3_mean":      s["c3_long_individual"]["M1_mean"],
        "C1_mean":      s["c1_short_individual"]["M1_mean"],
        "predicted_outcome": "not significant (H1 rejected = span alone insufficient)",
    })

    # ------------------------------------------------------------------
    # H2: Prosocial constraints alone sufficient (predicted REJECTED: C2 not > C1)
    u, p = mwu_one_tailed(c2, c1, direction="a_gt_b")
    d    = cohen_d(c2, c1)
    tests.append({
        "hypothesis":   "H2",
        "description":  "Prosocial constraints alone sufficient (predicted rejected: C2 not > C1)",
        "comparison":   "C2 vs C1",
        "direction":    "C2 > C1",
        "U":            u,   "p_one_tailed": p,
        "cohen_d":      d,
        "sig":          sig_stars(p),
        "C2_mean":      s["c2_short_welfare"]["M1_mean"],
        "C1_mean":      s["c1_short_individual"]["M1_mean"],
        "predicted_outcome": "not significant (H2 rejected = welfare alone insufficient)",
    })

    # ------------------------------------------------------------------
    # H3a: C4 > C3 (welfare coupling adds to long span)
    u, p = mwu_one_tailed(c4, c3, direction="a_gt_b")
    d    = cohen_d(c4, c3)
    tests.append({
        "hypothesis":   "H3a",
        "description":  "Joint necessity: welfare coupling elevates long-span sacrifice (C4 > C3)",
        "comparison":   "C4 vs C3",
        "direction":    "C4 > C3",
        "U":            u,   "p_one_tailed": p,
        "cohen_d":      d,
        "sig":          sig_stars(p),
        "C4_mean":      s["c4_long_welfare"]["M1_mean"],
        "C3_mean":      s["c3_long_individual"]["M1_mean"],
        "predicted_outcome": "significant (H3 supported)",
        "note":         "Deviation 2: C3/C4 null rate ~32% vs C1/C2 ~0.5%. M1 computed on non-null epochs.",
    })

    # ------------------------------------------------------------------
    # H3b: C4 > C2 (long span adds to welfare coupling)
    u, p = mwu_one_tailed(c4, c2, direction="a_gt_b")
    d    = cohen_d(c4, c2)
    tests.append({
        "hypothesis":   "H3b",
        "description":  "Joint necessity: long span elevates welfare-coupled sacrifice (C4 > C2)",
        "comparison":   "C4 vs C2",
        "direction":    "C4 > C2",
        "U":            u,   "p_one_tailed": p,
        "cohen_d":      d,
        "sig":          sig_stars(p),
        "C4_mean":      s["c4_long_welfare"]["M1_mean"],
        "C2_mean":      s["c2_short_welfare"]["M1_mean"],
        "predicted_outcome": "significant (H3 supported)",
        "note":         "Deviation 2: asymmetric sacrifice opportunity (C4 ~30% null, C2 ~0.5%). "
                        "Comparison on non-null epochs; interpret with caveat.",
    })

    # ------------------------------------------------------------------
    # H3c: C4 > C1 (both factors together vs neither)
    u, p = mwu_one_tailed(c4, c1, direction="a_gt_b")
    d    = cohen_d(c4, c1)
    tests.append({
        "hypothesis":   "H3c",
        "description":  "Joint necessity: C4 (both factors) > C1 (neither)",
        "comparison":   "C4 vs C1",
        "direction":    "C4 > C1",
        "U":            u,   "p_one_tailed": p,
        "cohen_d":      d,
        "sig":          sig_stars(p),
        "C4_mean":      s["c4_long_welfare"]["M1_mean"],
        "C1_mean":      s["c1_short_individual"]["M1_mean"],
        "predicted_outcome": "significant (H3 supported)",
    })

    # ------------------------------------------------------------------
    # H4a: C4-frozen > C4 (boundary reversal in long+welfare)
    u, p = mwu_one_tailed(c4f, c4, direction="a_gt_b")
    d    = cohen_d(c4f, c4)
    tests.append({
        "hypothesis":   "H4a",
        "description":  "Boundary reversal: frozen self_model_gru increases sacrifice in C4",
        "comparison":   "C4-frozen vs C4",
        "direction":    "C4-frozen > C4",
        "U":            u,   "p_one_tailed": p,
        "cohen_d":      d,
        "sig":          sig_stars(p),
        "C4f_mean":     s["c4_long_welfare_frozen"]["M1_mean"],
        "C4_mean":      s["c4_long_welfare"]["M1_mean"],
        "predicted_outcome": "significant (H4 supported)",
        "note":         "Deviation 2 applies to both C4 arms (both ~30-35% null).",
    })

    # ------------------------------------------------------------------
    # H4b: C1-frozen vs C1 (control boundary comparison, short span)
    u, p = mwu_one_tailed(c1f, c1, direction="a_gt_b")
    d    = cohen_d(c1f, c1)
    tests.append({
        "hypothesis":   "H4b",
        "description":  "Boundary control: frozen self_model_gru effect in short-span (C1-frozen vs C1)",
        "comparison":   "C1-frozen vs C1",
        "direction":    "C1-frozen > C1",
        "U":            u,   "p_one_tailed": p,
        "cohen_d":      d,
        "sig":          sig_stars(p),
        "C1f_mean":     s["c1_short_individual_frozen"]["M1_mean"],
        "C1_mean":      s["c1_short_individual"]["M1_mean"],
        "predicted_outcome": "exploratory — no directional prediction for short-span boundary",
    })

    return tests


# ---------------------------------------------------------------------------
# H5: CDI coupling
# ---------------------------------------------------------------------------

def run_h5(s: dict) -> dict:
    """
    H5: CDI coupling (M5) positive only in C4.
    Reports M5 mean per condition. Flags conditions with all-NaN M5.
    """
    result = {}
    for cname, summary in s.items():
        m5_vals   = summary["M5_cdi_coupling"]
        valid     = [v for v in m5_vals if not np.isnan(v)]
        mean_m5   = float(np.nanmean(m5_vals)) if valid else float("nan")
        evaluable = len(valid) > 0
        result[cname] = {
            "M5_mean":    mean_m5,
            "M5_valid_n": len(valid),
            "evaluable":  evaluable,
            "note":       ("CDI null — fewer than 2 valid-window positions available "
                           "(Deviation 2/3)") if not evaluable else
                          "50-valid-epoch window (Deviation 3)",
        }
    return result


# ---------------------------------------------------------------------------
# M6: Sacrifice Persistence summary
# ---------------------------------------------------------------------------

def run_m6(s: dict) -> dict:
    result = {}
    for cname, summary in s.items():
        m6_vals = summary["M6_scr_slope"]
        valid   = [v for v in m6_vals if not np.isnan(v)]
        result[cname] = {
            "M6_mean":    float(np.nanmean(m6_vals)) if valid else float("nan"),
            "M6_sd":      float(np.nanstd(m6_vals, ddof=1)) if len(valid) > 1 else float("nan"),
            "M6_valid_n": len(valid),
            "M6_by_seed": m6_vals,
        }
    return result


# ---------------------------------------------------------------------------
# Deception metric summary (M4)
# ---------------------------------------------------------------------------

def run_m4(s: dict) -> dict:
    result = {}
    agents = ["agent_a", "agent_b", "agent_c"]
    for cname, summary in s.items():
        m4_by_seed = summary["M4_deception"]
        cond_result = {}
        for ag in agents:
            vals = [seed_m4[ag] for seed_m4 in m4_by_seed
                    if isinstance(seed_m4, dict) and not np.isnan(seed_m4.get(ag, float("nan")))]
            cond_result[ag] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "sd":   float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            }
        result[cname] = cond_result
    return result


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(summaries: dict, hyp_tests: list[dict],
                 h5: dict, m6: dict, m4: dict) -> None:

    CNAMES = {
        "c1_short_individual":        "C1 Short/Individual",
        "c2_short_welfare":           "C2 Short/Welfare",
        "c3_long_individual":         "C3 Long/Individual",
        "c4_long_welfare":            "C4 Long/Welfare",
        "c1_short_individual_frozen": "C1 Short/Individual/Frozen",
        "c4_long_welfare_frozen":     "C4 Long/Welfare/Frozen",
    }

    print("=" * 72)
    print("PROTOCOL 5 — CONFIRMATORY ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    print("\n--- M1: Mean SCR (last 100 epochs, non-null) ---")
    print(f"{'Condition':<34} {'Mean':>8} {'SD':>8} {'Null%':>7}")
    print("-" * 60)
    for cname, s in summaries.items():
        print(f"{CNAMES.get(cname, cname):<34} "
              f"{s['M1_mean']:>8.4f} {s['M1_sd']:>8.4f} "
              f"{s['null_rate_mean']:>6.1f}%")

    print("\n--- HYPOTHESIS TESTS (Mann-Whitney U, one-tailed, alpha=0.05) ---")
    print(f"{'Test':<6} {'Comparison':<18} {'Mean_A':>8} {'Mean_B':>8} "
          f"{'U':>8} {'p':>8} {'d':>7} {'Sig':<5}")
    print("-" * 75)
    for t in hyp_tests:
        a_lbl, b_lbl = t["comparison"].split(" vs ")
        mean_a = t.get(f"{a_lbl.replace('-','f')}_mean",
                       t.get(f"{a_lbl}_mean",
                             t.get("C4f_mean", t.get("C1f_mean", float("nan")))))
        mean_b_key = [k for k in t if k.endswith("_mean") and k != "M1_mean" and k != mean_a]
        # simpler: just grab the two *_mean keys
        mean_keys = [k for k in t if k.endswith("_mean")]
        vals = [t[k] for k in mean_keys]
        mean_a_v = vals[0] if vals else float("nan")
        mean_b_v = vals[1] if len(vals) > 1 else float("nan")
        print(f"{t['hypothesis']:<6} {t['comparison']:<18} "
              f"{mean_a_v:>8.4f} {mean_b_v:>8.4f} "
              f"{t['U']:>8.1f} {t['p_one_tailed']:>8.4f} "
              f"{t['cohen_d']:>7.3f} {t['sig']:<5}")
        if t.get("note"):
            print(f"       NOTE: {t['note']}")

    print("\n--- H5: CDI COUPLING (M5) ---")
    print(f"{'Condition':<34} {'M5_mean':>9} {'valid_n':>8}  Evaluable")
    print("-" * 60)
    for cname, r in h5.items():
        m5_str = f"{r['M5_mean']:.4f}" if not np.isnan(r["M5_mean"]) else "NaN"
        print(f"{CNAMES.get(cname, cname):<34} {m5_str:>9} {r['M5_valid_n']:>8}  "
              f"{'YES' if r['evaluable'] else 'NO — ' + r['note'][:40]}")

    print("\n--- M6: SACRIFICE PERSISTENCE (slope epochs 300-500) ---")
    print(f"{'Condition':<34} {'M6_mean':>10} {'M6_sd':>8} {'valid_n':>8}")
    print("-" * 63)
    for cname, r in m6.items():
        m6_str = f"{r['M6_mean']:.6f}" if not np.isnan(r["M6_mean"]) else "NaN"
        sd_str = f"{r['M6_sd']:.6f}" if not np.isnan(r["M6_sd"]) else "NaN"
        print(f"{CNAMES.get(cname, cname):<34} {m6_str:>10} {sd_str:>8} "
              f"{r['M6_valid_n']:>8}")

    print("\n--- M4: DECEPTION METRIC (mean last 100 epochs) ---")
    print(f"{'Condition':<34} {'agent_a':>9} {'agent_b':>9} {'agent_c':>9}")
    print("-" * 63)
    for cname, r in m4.items():
        print(f"{CNAMES.get(cname, cname):<34} "
              f"{r['agent_a']['mean']:>9.4f} "
              f"{r['agent_b']['mean']:>9.4f} "
              f"{r['agent_c']['mean']:>9.4f}")

    print("\n--- EVALUABILITY SUMMARY ---")
    print("H1:  EVALUABLE (C3 vs C1, both short/long individual)")
    print("H2:  EVALUABLE (C2 vs C1, both short span)")
    print("H3a: EVALUABLE (C4 vs C3) — Deviation 2 caveat on opportunity asymmetry")
    print("H3b: EVALUABLE (C4 vs C2) — Deviation 2 caveat on opportunity asymmetry")
    print("H3c: EVALUABLE (C4 vs C1) — Deviation 2 caveat on opportunity asymmetry")
    print("H4a: EVALUABLE (C4-frozen vs C4)")
    print("H4b: EVALUABLE (C1-frozen vs C1)")
    h5_eval = {c: r["evaluable"] for c, r in h5.items()}
    h5_short = all(h5_eval.get(c, False) for c in ["c1_short_individual", "c2_short_welfare"])
    h5_long  = all(h5_eval.get(c, False) for c in ["c3_long_individual", "c4_long_welfare"])
    if h5_long:
        print("H5:  EVALUABLE")
    else:
        print("H5:  PARTIALLY EVALUABLE — CDI null for all long-span conditions (C3/C4).")
        print("     M5 computable for short-span (C1/C2) only.")
        print("     H5 prediction (CDI coupling positive ONLY in C4) cannot be confirmed")
        print("     or rejected: C4 CDI data is absent. Operationalization gap, Deviation 2.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    condition_names = [
        "c1_short_individual",
        "c2_short_welfare",
        "c3_long_individual",
        "c4_long_welfare",
        "c1_short_individual_frozen",
        "c4_long_welfare_frozen",
    ]

    print("Loading result files...", flush=True)
    data      = {c: load_condition(c) for c in condition_names}
    summaries = {c: summarize(c, data[c]) for c in condition_names}

    hyp_tests = run_hypothesis_tests(summaries)
    h5_result = run_h5(summaries)
    m6_result = run_m6(summaries)
    m4_result = run_m4(summaries)

    print_report(summaries, hyp_tests, h5_result, m6_result, m4_result)

    # ------------------------------------------------------------------
    # Save JSON output
    output = {
        "generated":   datetime.now().isoformat(),
        "protocol":    5,
        "alpha":       ALPHA,
        "summaries":   summaries,
        "hypothesis_tests": hyp_tests,
        "H5_cdi_coupling":  h5_result,
        "M6_sacrifice_persistence": m6_result,
        "M4_deception": m4_result,
    }
    out_path = os.path.join(ANALYSIS_DIR, "p5_confirmatory_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
