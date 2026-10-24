#!/usr/bin/env python3
"""Forest plots of observed vs projection-preserving null (beta = 0).
Render-only: reads results/final/summary.csv (written by diagnose_final.py).

Main figure (default): two columns, partial measure only --
  left  R^m_part  (multiplicity-aware, multihypergraph start, filter k3-25)
  right R^M_part  (M-weighted support, dedup start E*, filter k3-25-dedup)
One row per dataset (communication / collaboration separated by a gap).
Gray segment = central 95% interval of the null distribution (2.5th-97.5th
percentiles over all retained null states), gray open circle = null
mean, colored filled circle = observed score. LOG x-axis, so the horizontal
gap between the two circles is log(observed / null mean); the ratio is
printed at the right margin of every row.

--appendix: the 2 x 3 version (rows: multiplicity-aware, support; columns:
any, partial, all). Null quantiles that are exactly 0 (R_all, collaboration
datasets) are drawn to the axis floor with a left-pointing tick.
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from make_main_outputs import ORDER, SETS, INK, BODY, MUTED, GRID, SURF  # noqa: E402

NULL_C = "#8a8a85"
COLS = {"any": ("s1", r"R_{\rm any}", "#eb6834"),
        "part": ("r2", r"R_{\rm part}", "#2a78d6"),
        "all": ("r1", r"R_{\rm all}", "#009490")}
FAMS = {"multi": dict(filt="k3-25", suf="_multi", sup=r"\rm m", name="multiplicity-aware"),
        "support": dict(filt="k3-25-dedup", suf="_support_M", sup="", name="support")}


def load(path):
    s = pd.read_csv(path)
    # beta = 0 main chains only: drop the phase-I (33 independent chains) combos
    return s[(s.beta == 0) & ~s.combo.str.contains("indep")]


def rows_y():
    """dataset -> y position, with a gap between the two domains."""
    y, out, prev = 0.0, {}, None
    for key, name, dom in ORDER:
        if prev is not None and dom != prev:
            y += 0.8
        out[key] = y; y += 1.0; prev = dom
    return out


def panel(ax, S, fam, col, ypos, floor, show_labels):
    f, c = FAMS[fam], COLS[col]
    meas = c[0] + f["suf"]
    d = S[(S["filter"] == f["filt"]) & (S.measure == meas)].set_index("dataset")
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):                  # axes and numbers in ink
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.9)
    ax.grid(False)
    ax.tick_params(length=3, width=0.9, labelsize=10, colors=INK)
    lo_all = []
    for key, name, dom in ORDER:
        if key not in d.index:
            continue
        r = d.loc[key]; y = ypos[key]
        q025, q975 = r.q025, r.q975
        clipped = q025 <= 0
        q025 = max(q025, floor)
        ax.plot([q025, q975], [y, y], color=NULL_C, lw=2.2, solid_capstyle="butt", zorder=2)
        if clipped:
            ax.plot([floor], [y], marker="<", ms=5, color=NULL_C, zorder=3)
        ax.plot([max(r.null_mean, floor)], [y], "o", ms=6.5, mfc=SURF, mec=NULL_C, mew=1.5, zorder=4)
        ax.plot([r.observed], [y], "o", ms=6.5, color=c[2], mec=c[2], zorder=5)
        ratio = r.observed / r.null_mean if r.null_mean > 0 else np.inf
        ax.text(1.01, y, f"×{ratio:.1f}" if np.isfinite(ratio) and ratio < 100 else
                (f"×{ratio:.0f}" if np.isfinite(ratio) else "×∞"),
                transform=ax.get_yaxis_transform(), fontsize=9, color=INK, va="center")
        lo_all.append(max(min(q025, r.null_mean, r.observed), floor))
    ax.set_xscale("log")
    ax.set_xlim(min(lo_all) / 1.6, 1.0)
    ax.set_ylim(max(ypos.values()) + 0.8, -0.8)
    if show_labels:                       # axes share y: set the labels once
        ax.set_yticks([ypos[k] for k, *_ in ORDER])
        ax.set_yticklabels([n for _, n, _ in ORDER], fontsize=10.5, color=INK)
    ax.set_title(rf"${c[1]}^{{{f['sup']}}}$  ({f['name']})", fontsize=11.5, color=INK, pad=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "final"))
    p.add_argument("--appendix", action="store_true")
    p.add_argument("--family", choices=list(FAMS), default=None,
                   help="one family, three panels (any / part / all), own legend")
    p.add_argument("--summary", default="summary.csv",
                   help="summary file inside --out (e.g. summary_s1000.csv)")
    p.add_argument("--floor", type=float, default=1e-5, help="log-axis floor for zero quantiles")
    a = p.parse_args()
    root = os.path.abspath(a.out)
    S = load(os.path.join(root, a.summary))
    tag = a.summary.replace("summary", "").replace(".csv", "")
    ypos = rows_y()
    if a.family:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
        for j, col in enumerate(("any", "part", "all")):
            panel(axes[j], S, a.family, col, ypos, a.floor, show_labels=(j == 0))
        stem = f"fig_forest_{a.family}{tag}"
    elif a.appendix:
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), sharey=True)
        for i, fam in enumerate(("multi", "support")):
            for j, col in enumerate(("any", "part", "all")):
                panel(axes[i, j], S, fam, col, ypos, a.floor, show_labels=(j == 0))
        stem = f"fig_forest_appendix{tag}"
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True)
        panel(axes[0], S, "multi", "part", ypos, a.floor, show_labels=True)
        panel(axes[1], S, "support", "part", ypos, a.floor, show_labels=False)
        stem = f"fig_forest_main{tag}"
    fig.patch.set_facecolor(SURF)
    for ax in np.ravel(axes)[-(3 if (a.appendix or a.family) else 2):]:
        ax.set_xlabel("reciprocity (log scale)", fontsize=10.5, color=INK)
    # domain separators + labels
    for ax in np.ravel(axes):
        ysep = (ypos["twitter"] + ypos["congress"]) / 2
        ax.axhline(ysep, color=GRID, lw=0.8, ls=(0, (3, 3)))
    handles = [Line2D([], [], color=NULL_C, lw=2.2, label="null distribution: central 95% interval"),
               Line2D([], [], marker="o", mfc=SURF, mec=NULL_C, mew=1.5, lw=0, ms=6.5, label="null mean"),
               Line2D([], [], marker="o", color=BODY, lw=0, ms=6.5, label="observed"),
               Line2D([], [], lw=0, label="×r = observed / null mean")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=10,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0), handletextpad=0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        out = os.path.join(root, f"{stem}.{ext}")
        fig.savefig(out, facecolor=SURF, dpi=180)
        print("[saved]", out)


if __name__ == "__main__":
    main()

# How this was run (2026-10-13):
#   <venv-python> make_forest_fig.py              # main: R_part, multi | support
#   <venv-python> make_forest_fig.py --appendix   # 2 x 3 (family x any/part/all)
#   per-family 1 x 3 figures from the common 1000-sweep budget (2026-10-24):
#     <venv-python> diagnose_final.py --max-sweeps 1000 --tag _s1000
#     <venv-python> make_forest_fig.py --family multi   --summary summary_s1000.csv
#     <venv-python> make_forest_fig.py --family support --summary summary_s1000.csv
# Render-only: reads results/final/summary.csv; writes fig_forest_main.{pdf,png}
# and fig_forest_appendix.{pdf,png}.
