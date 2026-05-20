"""
Generate two publication figures for Protocol 3.

  figure1_p3_query_trajectories.png
      Mean query rate per epoch across all 500 epochs, three conditions,
      with 95% bootstrap CI bands. Shows full temporal dynamics.

  figure2_p3_final_window.png
      Two-panel figure (query rate | SSS) for epochs 400-499.
      Boxplots with individual seed points, visualising the primary result
      (H1 inverted) and the dissociation (rising query rate, falling SSS).

Output: docs/figures/
Usage:
    cd backend
    python generate_p3_figures.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE, "backend", "data")
OUT_DIR   = os.path.join(BASE, "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS      = list(range(10))
CONDITIONS = ["p3_unconstrained", "p3b_constrained", "p3a_constrained"]
LABELS     = {
    "p3_unconstrained": "Unconstrained",
    "p3b_constrained":  "3B — Hidden schedule",
    "p3a_constrained":  "3A — Stochastic (p = 0.50)",
}
WINDOW_START = 400
WINDOW_END   = 500

# Wong (2011) colourblind-friendly palette
COLORS = {
    "p3_unconstrained": "#0072B2",  # blue
    "p3b_constrained":  "#D55E00",  # vermilion
    "p3a_constrained":  "#009E73",  # green
}

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)

# ── global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
})


# ── data loading ──────────────────────────────────────────────────────────────

def load_series(condition: str, seed: int) -> list[dict] | None:
    path = os.path.join(DATA_ROOT, condition, f"seed_{seed}", "epoch_series.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_trajectory(condition: str, field: str) -> np.ndarray:
    """Return (n_seeds, 500) array for a given field, NaN where missing."""
    arrays = []
    for seed in SEEDS:
        series = load_series(condition, seed)
        arr = np.full(500, np.nan)
        if series:
            for r in series:
                ep = r.get("epoch")
                v  = r.get(field)
                if ep is not None and v is not None and 0 <= ep < 500:
                    arr[ep] = float(v)
        arrays.append(arr)
    return np.array(arrays)  # (n_seeds, 500)


def final_window_values(condition: str, field: str) -> list[float]:
    """Per-seed mean of field over epochs 400-499."""
    vals = []
    for seed in SEEDS:
        series = load_series(condition, seed)
        if not series:
            continue
        tail = [r[field] for r in series
                if WINDOW_START <= r.get("epoch", -1) < WINDOW_END
                and r.get(field) is not None]
        if tail:
            vals.append(float(np.mean(tail)))
    return vals


# ── bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(data_2d: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 ci: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-epoch bootstrap CI. Returns mean, lo, hi of shape (500,)."""
    n_epochs = data_2d.shape[1]
    mean_arr = np.nanmean(data_2d, axis=0)
    lo_arr   = np.full(n_epochs, np.nan)
    hi_arr   = np.full(n_epochs, np.nan)
    alpha    = (1.0 - ci) / 2.0

    for ep in range(n_epochs):
        col   = data_2d[:, ep]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            lo_arr[ep] = mean_arr[ep]
            hi_arr[ep] = mean_arr[ep]
            continue
        boot = np.array([
            np.mean(RNG.choice(valid, size=len(valid), replace=True))
            for _ in range(n_boot)
        ])
        lo_arr[ep] = np.quantile(boot, alpha)
        hi_arr[ep] = np.quantile(boot, 1.0 - alpha)

    return mean_arr, lo_arr, hi_arr


def smooth(arr: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling mean, forward-padded at edges."""
    from scipy.ndimage import uniform_filter1d
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return arr
    xp = np.where(mask)[0]
    yp = arr[mask]
    interp = np.interp(np.arange(len(arr)), xp, yp)
    return uniform_filter1d(interp, size=window)


# ── Figure 1: Query rate trajectories ─────────────────────────────────────────

print("Figure 1: loading query rate trajectories...")

fig1, ax1 = plt.subplots(figsize=(9, 5.5))
epochs_x   = np.arange(500)

for cond in CONDITIONS:
    color  = COLORS[cond]
    label  = LABELS[cond]
    data   = build_trajectory(cond, "query_rate")
    mean, lo, hi = bootstrap_ci(data)
    print(f"  {cond}: mean qr={np.nanmean(mean):.3f}")

    mean_s = smooth(mean)
    lo_s   = smooth(lo)
    hi_s   = smooth(hi)

    ax1.plot(epochs_x, mean_s, color=color, linewidth=2.2,
             label=label, zorder=3)
    ax1.fill_between(epochs_x, lo_s, hi_s, color=color,
                     alpha=0.18, linewidth=0, zorder=2)

# Confirmation window annotation
ax1.axvspan(WINDOW_START, WINDOW_END - 1, alpha=0.07, color="#888888",
            zorder=1, label="Confirmatory window (epochs 400–499)")
ax1.axvline(WINDOW_START, color="#888888", linewidth=1.0,
            linestyle=":", zorder=2)

ax1.set_xlim(-5, 504)
ax1.set_ylim(0.0, 1.0)
ax1.set_xlabel("Training epoch", labelpad=6)
ax1.set_ylabel("QUERY signal rate", labelpad=6)
ax1.set_title(
    "Mean QUERY rate by condition across 500 epochs (95% CI, 10 seeds)",
    pad=10, fontweight="bold"
)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax1.legend(loc="upper left", framealpha=0.88, edgecolor="#CCCCCC")

fig1.text(
    0.5, -0.02,
    "Both constrained conditions (3B, 3A) show elevated query rates relative to unconstrained baseline.\n"
    "The ordering unconstrained < hidden-schedule < stochastic is stable across the confirmatory window.",
    ha="center", va="top", fontsize=9, style="italic", color="#333333",
)

fig1.tight_layout(rect=[0, 0.07, 1, 1])
out1 = os.path.join(OUT_DIR, "figure1_p3_query_trajectories.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"Saved: {out1}")


# ── Figure 2: Final-window comparison (query rate | SSS) ──────────────────────

print("\nFigure 2: building final-window comparison panels...")

fig2, (ax_qr, ax_sss) = plt.subplots(1, 2, figsize=(11, 5.5),
                                      gridspec_kw={"wspace": 0.38})

condition_order = ["p3_unconstrained", "p3b_constrained", "p3a_constrained"]
box_labels      = ["Unconstrained", "3B\nHidden schedule", "3A\nStochastic"]
box_colors      = [COLORS[c] for c in condition_order]

jitter_rng = np.random.default_rng(77)

for ax, field, ylabel, title_suffix in [
    (ax_qr,  "query_rate", "Mean QUERY rate (epochs 400–499)",
     "Query rate — primary metric"),
    (ax_sss, None,         "Mean SSS (epochs 400–499)",
     "Sustained Structure Score"),
]:
    all_vals = []
    for cond in condition_order:
        if field == "query_rate":
            vals = final_window_values(cond, "query_rate")
        else:
            # SSS = mean(type_entropy) * mean(qrc) per seed over window
            sss_vals = []
            for seed in SEEDS:
                series = load_series(cond, seed)
                if not series:
                    continue
                tail = [r for r in series
                        if WINDOW_START <= r.get("epoch", -1) < WINDOW_END]
                te_  = [r["type_entropy"] for r in tail
                        if r.get("type_entropy") is not None]
                qrc_ = [r["qrc"] for r in tail
                        if r.get("qrc") is not None]
                if te_ and qrc_:
                    sss_vals.append(float(np.mean(te_)) * float(np.mean(qrc_)))
            vals = sss_vals
        all_vals.append(vals)

    bp = ax.boxplot(
        all_vals,
        patch_artist=True,
        notch=False,
        widths=0.42,
        medianprops=dict(color="black", linewidth=2.2),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker=""),
        zorder=2,
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.6)
    for whisker, color in zip(bp["whiskers"],
                               [c for c in box_colors for _ in range(2)]):
        whisker.set_color(color)
    for cap, color in zip(bp["caps"],
                          [c for c in box_colors for _ in range(2)]):
        cap.set_color(color)

    # Individual seed points (jittered)
    for i, (vals, color) in enumerate(zip(all_vals, box_colors)):
        jitter = jitter_rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(np.array([i + 1] * len(vals)) + jitter, vals,
                   color=color, s=42, alpha=0.85, zorder=4,
                   edgecolors="white", linewidths=0.6)

    ax.set_xticks(range(1, len(box_labels) + 1))
    ax.set_xticklabels(box_labels, fontsize=10)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title_suffix, pad=8, fontweight="bold", fontsize=12)
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# Annotate the dissociation direction with arrows
ax_qr.annotate(
    "", xy=(3.15, np.mean(final_window_values("p3a_constrained", "query_rate"))),
    xytext=(3.15, np.mean(final_window_values("p3_unconstrained", "query_rate"))),
    arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.4),
)
ax_qr.text(3.32, 0.46, "↑ QR", fontsize=9, color="#444444", va="center")

# Shared caption
fig2.text(
    0.5, -0.02,
    "Left: Query rate rises monotonically — unconstrained < 3B < 3A (H1 inverted, H2 confirmed).\n"
    "Right: SSS falls monotonically in the same ordering. More QUERY output did not produce stronger\n"
    "communicative structure. Each point is one seed; boxes show IQR and median.",
    ha="center", va="top", fontsize=9, style="italic", color="#333333",
)

fig2.suptitle(
    "Protocol 3: Final-window metrics by condition (epochs 400–499, N = 10 seeds)",
    fontsize=13, fontweight="bold", y=1.01,
)

fig2.tight_layout(rect=[0, 0.09, 1, 1])
out2 = os.path.join(OUT_DIR, "figure2_p3_final_window.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved: {out2}")

# ── verify output ──────────────────────────────────────────────────────────────
print()
for fn in [out1, out2]:
    size = os.path.getsize(fn)
    print(f"  {os.path.basename(fn)}: {size:,} bytes ({size // 1024} KB)")
