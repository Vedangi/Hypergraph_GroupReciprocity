#!/usr/bin/env python3
"""Four-panel horizon figure for the prediction experiments.

One panel per dataset; x = prediction horizon (each dataset's own horizons,
evenly spaced); y = Delta-AP (graph+hypergraph minus graph), common scale;
three series (any response / exact-team response / group-response mode) with
paired event-clustered bootstrap 95% intervals; zero reference line.
Render-only: reads <results-dir>/<dataset>_horizon_tables.csv.
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- edit here: datasets, their horizons (days), panel titles ---------------
PANELS = [("emaileu", "email-Eu", [1, 2, 3, 5, 7, 10, 15, 20, 30]),
          ("enron", "Enron", [1, 2, 3, 5, 7, 10, 15, 20, 30]),
          ("dnc", "DNC", [1, 2, 3, 4, 5]),
          ("twitter", "Twitter (Congress)", [1, 3, 5, 7, 10, 15, 20, 30, 60])]
SERIES = [("y_dyad", "any response", "#009490", "^", -0.18),
          ("y_group_exact", "exact-team response", "#2a78d6", "o", 0.0),
          ("y_mode", "group-response mode", "#eb6834", "s", +0.18)]
# palette validated (dataviz validator: all checks pass, worst CVD dE 11.9);
# marker shape is the secondary encoding.
INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68",
                                "#d9d9d4", "#ffffff")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=os.path.join(
        HERE, "..", "results", "split_50_20_30_tuned"))
    p.add_argument("--model", default="LightGBM", choices=["LightGBM", "LogReg"])
    p.add_argument("--out", default=None)
    a = p.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.4), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, (key, title, horizons) in zip(axes, PANELS):
        df = pd.read_csv(os.path.join(a.results_dir, f"{key}_horizon_tables.csv"))
        df = df[(df.model == a.model) & df.horizon.isin(horizons)]
        ax.set_facecolor(SURF)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.axhline(0, color=MUTED, lw=1.1, ls="--", zorder=1)
        pos = {h: i for i, h in enumerate(horizons)}
        for label, _name, col, mk, dx in SERIES:
            s = df[df.label == label].sort_values("horizon")
            x = [pos[h] + dx for h in s.horizon]
            y = s.dPR.to_numpy()
            err = [y - s.dPR_ci_low.to_numpy(), s.dPR_ci_high.to_numpy() - y]
            ax.errorbar(x, y, yerr=err, color=col, marker=mk, ms=5.5, lw=1.8,
                        elinewidth=1.2, capsize=2.5, markeredgecolor=SURF,
                        markeredgewidth=1.0, zorder=3)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlim(-0.45, len(horizons) - 0.55)
        ax.tick_params(labelsize=10.5, colors=MUTED, length=0)
        ax.set_title(title, fontsize=14, color=INK, pad=6)
        ax.set_xlabel("horizon (days)", fontsize=12.5, color=BODY)
    axes[0].set_ylabel(r"$\Delta$AP  (graph+hypergraph $-$ graph)",
                       fontsize=12.5, color=BODY)
    handles = [Line2D([], [], color=c, marker=m, ms=7, lw=2,
                      markeredgecolor=SURF, label=n)
               for _l, n, c, m, _d in SERIES]
    fig.legend(handles=handles, fontsize=12.5, frameon=False, labelcolor=BODY,
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
               handlelength=2.2, columnspacing=2.4)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = a.out or os.path.join(HERE, "..", "results", "figures",
                                f"horizon_deltaAP_{a.model}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", facecolor=SURF, dpi=170)
        print("[saved]", f"{out}.{ext}")


if __name__ == "__main__":
    main()

# How this was run (2026-09-21):
#   cd Hypergraph_GroupReciprocity/prediction/code
#   <venv-python> make_horizon_figure.py                      # LightGBM, tuned 50-20-30
#   <venv-python> make_horizon_figure.py --model LogReg
#   other protocol: --results-dir ../results/split_70_30
# Render-only. Edit PANELS at the top to change datasets/horizons/titles.
# Outputs: ../results/figures/horizon_deltaAP_<model>.{pdf,png}
