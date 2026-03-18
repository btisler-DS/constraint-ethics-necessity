"""
Generate two publication figures for Protocol 5.
  figure1_scr_trajectories.png  — Mean SCR trajectories, C1–C4, with 95% CI bands
  figure2_cdi_coupling_boxplot.png — M5 CDI coupling boxplot, all 6 conditions
Output: D:\Claude_Code\project_8\docs\figures\
"""

import json
import glob
import os
import math
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── paths ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "backend", "results")
ANALYSIS_JSON = os.path.join(BASE, "backend", "analysis_p5", "p5_confirmatory_results.json")
OUT_DIR = os.path.join(BASE, "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────────
# Print-safe, colourblind-friendly palette (Wong 2011)
COLORS = {
    "c1": "#0072B2",   # blue
    "c2": "#D55E00",   # vermilion
    "c3": "#009E73",   # green
    "c4": "#CC79A7",   # pink/purple
    "c1f": "#56B4E9",  # sky blue
    "c4f": "#E69F00",  # orange
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — SCR Trajectories
# ─────────────────────────────────────────────────────────────────────────────

PRIMARY_CONDITIONS = [
    ("c1_short_individual", "C1 Short/Individual", COLORS["c1"]),
    ("c2_short_welfare",    "C2 Short/Welfare",    COLORS["c2"]),
    ("c3_long_individual",  "C3 Long/Individual",  COLORS["c3"]),
    ("c4_long_welfare",     "C4 Long/Welfare",     COLORS["c4"]),
]

def load_condition_scr(cond_key, n_seeds=10):
    """Return list of 500-length arrays (np.nan where null)."""
    arrays = []
    for seed in range(n_seeds):
        fname = os.path.join(RESULTS_DIR, f"p5_condition_{cond_key}_seed_{seed}.json")
        if not os.path.exists(fname):
            print(f"  WARNING: missing {fname}")
            continue
        with open(fname) as f:
            d = json.load(f)
        eps = d if isinstance(d, list) else d.get("epoch_series", [d])
        # Build 500-length array keyed by epoch index
        arr = np.full(500, np.nan)
        for e in eps:
            if not isinstance(e, dict):
                continue
            ep = e.get("epoch")
            scr = e.get("sacrifice_choice_rate")
            if ep is not None and scr is not None and 0 <= ep < 500:
                arr[ep] = float(scr)
        arrays.append(arr)
    return np.array(arrays)  # shape (n_seeds, 500)

def bootstrap_ci(data_2d, n_boot=N_BOOTSTRAP, ci=0.95):
    """
    data_2d: (n_seeds, n_epochs). NaN = missing.
    Returns mean, lo, hi each of shape (n_epochs,).
    At each epoch, bootstraps over non-NaN seeds.
    """
    n_seeds, n_epochs = data_2d.shape
    mean_arr = np.nanmean(data_2d, axis=0)
    lo_arr   = np.full(n_epochs, np.nan)
    hi_arr   = np.full(n_epochs, np.nan)
    alpha = (1.0 - ci) / 2.0

    for ep in range(n_epochs):
        col = data_2d[:, ep]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            lo_arr[ep] = mean_arr[ep]
            hi_arr[ep] = mean_arr[ep]
            continue
        boot_means = np.array([
            np.mean(RNG.choice(valid, size=len(valid), replace=True))
            for _ in range(n_boot)
        ])
        lo_arr[ep] = np.quantile(boot_means, alpha)
        hi_arr[ep] = np.quantile(boot_means, 1.0 - alpha)

    return mean_arr, lo_arr, hi_arr

def smooth_series(x_epochs, y_vals, y_lo, y_hi, window=15):
    """Rolling mean over non-NaN epochs for display."""
    # Use pandas-style rolling via numpy convolution
    mask = ~np.isnan(y_vals)
    if mask.sum() == 0:
        return x_epochs, y_vals, y_lo, y_hi
    # simple uniform rolling average
    from numpy.lib.stride_tricks import sliding_window_view
    # pad-free approach: use valid indices only, apply uniform filter
    from scipy.ndimage import uniform_filter1d
    y_sm   = np.where(mask, y_vals, np.nan)
    lo_sm  = np.where(mask, y_lo,   np.nan)
    hi_sm  = np.where(mask, y_hi,   np.nan)
    # interpolate NaN for smoothing, then re-apply mask
    def smooth_interp(arr):
        xp = np.where(mask)[0]
        yp = arr[mask]
        arr_interp = np.interp(np.arange(len(arr)), xp, yp)
        return uniform_filter1d(arr_interp, size=window)
    return x_epochs, smooth_interp(y_sm), smooth_interp(lo_sm), smooth_interp(hi_sm)

print("Figure 1: loading epoch series data...")
fig1, ax1 = plt.subplots(figsize=(9, 5.5))

epochs_x = np.arange(500)

for cond_key, label, color in PRIMARY_CONDITIONS:
    print(f"  {cond_key}...", end=" ", flush=True)
    data = load_condition_scr(cond_key)
    mean, lo, hi = bootstrap_ci(data)
    print(f"valid epochs mean={np.nanmean(mean):.3f}")

    # Smooth for display clarity
    _, mean_s, lo_s, hi_s = smooth_series(epochs_x, mean, lo, hi, window=20)

    ax1.plot(epochs_x, mean_s, color=color, linewidth=2.0, label=label, zorder=3)
    ax1.fill_between(epochs_x, lo_s, hi_s, color=color, alpha=0.18, linewidth=0, zorder=2)

ax1.set_xlim(-5, 504)
# Trim y to observed range with padding
ax1.set_ylim(0.0, 0.70)
ax1.set_xlabel("Training epoch", labelpad=6)
ax1.set_ylabel("Sacrifice choice rate", labelpad=6)
ax1.set_title("Mean sacrifice_choice_rate by condition (non-null epochs, 95% CI)",
              pad=10, fontweight="bold")
ax1.legend(loc="upper right", framealpha=0.85, edgecolor="#CCCCCC")

# Annotation for C3/C4 null epoch note
ax1.annotate(
    "C3/C4: ~30–35% of calendar epochs are null\n(sacrifice trigger did not fire; Deviation 2)",
    xy=(250, 0.04),
    fontsize=8.5, color="#555555", style="italic",
    ha="center",
)

# Caption
fig1.text(
    0.5, -0.02,
    "All four conditions show flat or marginally declining trajectories with overlapping confidence intervals\n"
    "throughout training. No condition shows the rising trajectory predicted for C4.",
    ha="center", va="top", fontsize=9, style="italic", color="#333333",
    wrap=True
)

fig1.tight_layout(rect=[0, 0.06, 1, 1])

out1 = os.path.join(OUT_DIR, "figure1_scr_trajectories.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"Saved: {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — M5 CDI Coupling Boxplot
# ─────────────────────────────────────────────────────────────────────────────

print("\nFigure 2: building CDI coupling boxplot...")

# M5 values per condition (from analysis JSON Item 2)
M5_DATA = {
    "C1\nShort/Indiv":  [-0.1275, 0.0882, 0.1916, 0.1157, -0.0084, 0.0256, 0.0458, -0.0610, -0.1198, -0.0109],
    "C2\nShort/Welfare":[-0.0087, 0.1476,-0.0257,-0.0521,  0.1279,-0.1176,-0.1003, -0.0934,  0.0147, -0.0955],
    "C3\nLong/Indiv":   [ 0.1685,-0.0428, 0.0408, 0.0503, -0.0375,-0.0472,-0.1210, -0.1084,  0.1255,  0.1383],
    "C4\nLong/Welfare": [-0.0994,-0.1250, 0.0332, 0.0605,  0.0661, 0.0659,-0.1096, -0.0206,  0.1576, -0.0497],
    "C1-frz\nShort/Indiv":[-0.0352, 0.0773, 0.1265, 0.1409, 0.1040, 0.1113,-0.0339,  0.1153,-0.1441,  0.0764],
    "C4-frz\nLong/Welfare":[-0.0060,-0.0902,-0.0206,-0.0818, 0.1747,-0.1569, 0.1390,  0.0195, 0.0005,  0.0220],
}

BOX_COLORS = [
    COLORS["c1"], COLORS["c2"], COLORS["c3"],
    COLORS["c4"], COLORS["c1f"], COLORS["c4f"],
]

labels = list(M5_DATA.keys())
values = list(M5_DATA.values())

fig2, ax2 = plt.subplots(figsize=(9, 5.5))

# Boxplot
bp = ax2.boxplot(
    values,
    patch_artist=True,
    notch=False,
    widths=0.45,
    medianprops=dict(color="black", linewidth=2.0),
    whiskerprops=dict(linewidth=1.4),
    capprops=dict(linewidth=1.4),
    flierprops=dict(marker="", markersize=0),  # hide fliers; we'll jitter
    zorder=2,
)

for patch, color in zip(bp["boxes"], BOX_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.45)
    patch.set_edgecolor(color)
    patch.set_linewidth(1.6)

for whisker, color in zip(bp["whiskers"], [c for c in BOX_COLORS for _ in range(2)]):
    whisker.set_color(color)
for cap, color in zip(bp["caps"], [c for c in BOX_COLORS for _ in range(2)]):
    cap.set_color(color)

# Jittered individual points
jitter_rng = np.random.default_rng(99)
for i, (vals, color) in enumerate(zip(values, BOX_COLORS)):
    x_pos = i + 1
    jitter = jitter_rng.uniform(-0.15, 0.15, size=len(vals))
    ax2.scatter(
        x_pos + jitter, vals,
        color=color, s=38, alpha=0.85, zorder=4,
        edgecolors="white", linewidths=0.6
    )

# Reference lines
ax2.axhline(0, color="#333333", linewidth=1.2, linestyle="--", alpha=0.7, zorder=1, label="y = 0")
ax2.axhline(0.3, color="#AA0000", linewidth=1.4, linestyle="--", alpha=0.8, zorder=1,
            label="H5 convergence threshold (r > 0.3)")

ax2.set_xticks(range(1, len(labels) + 1))
ax2.set_xticklabels(labels, fontsize=9.5)
ax2.set_ylabel("M5 CDI coupling value", labelpad=6)
ax2.set_xlabel("Condition", labelpad=6)
ax2.set_title("M5 CDI Coupling by condition (N = 10 seeds each)",
              pad=10, fontweight="bold")

# y-axis range: add padding
all_vals_flat = [v for sub in values for v in sub]
ymin = min(all_vals_flat) - 0.06
ymax = max(0.35, max(all_vals_flat) + 0.06)
ax2.set_ylim(ymin, ymax)

# Legend (reference lines only; conditions already distinguished by colour)
legend_elems = [
    Line2D([0], [0], color="#333333", linewidth=1.2, linestyle="--", label="y = 0"),
    Line2D([0], [0], color="#AA0000", linewidth=1.4, linestyle="--", label="H5 threshold r > 0.3"),
]
ax2.legend(handles=legend_elems, loc="upper right", framealpha=0.85,
           edgecolor="#CCCCCC", fontsize=9.5)

# Caption
fig2.text(
    0.5, -0.02,
    "All conditions cluster near zero. No condition approaches the preregistered convergence threshold of\n"
    "r > 0.3 (dashed line). Within-condition variance substantially exceeds between-condition differences.",
    ha="center", va="top", fontsize=9, style="italic", color="#333333",
)

fig2.tight_layout(rect=[0, 0.06, 1, 1])

out2 = os.path.join(OUT_DIR, "figure2_cdi_coupling_boxplot.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved: {out2}")

# ── verify file sizes ─────────────────────────────────────────────────────────
for fn in [out1, out2]:
    size = os.path.getsize(fn)
    print(f"  {os.path.basename(fn)}: {size:,} bytes ({size//1024} KB)")
