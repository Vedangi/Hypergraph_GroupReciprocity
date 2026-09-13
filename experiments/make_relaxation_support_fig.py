#!/usr/bin/env python3
"""Figures for the support-measure relaxation traces (dedup-start chains).
One small-multiples figure per measure (R_any, R_part, R_all): M-weighted
solid, p-weighted dashed; band = stationary tail mean +- 2 sd (M-weighted,
both chains, sweeps >= 100)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "relaxation_support_results")
INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68",
                                "#d9d9d4", "#ffffff")
C_REAL, C_FILL = "#2a78d6", "#eb6834"
LABEL = {"fauci": "Fauci email", "wiki": "Wiki talk", "dnc": "DNC email",
         "radoslaw": "Manufacturing email", "higgs": "Higgs Twitter",
         "twitter": "Twitter (senators)", "enron": "Enron"}
ORDER = ["fauci", "wiki", "dnc", "radoslaw", "higgs", "twitter", "enron"]
MEASURES = {"R_any": 0, "R_part": 1, "R_all": 2}   # column offsets; +3 = p-weighted
NICE = {"R_any": r"$R^{\rm any}_{\rm team}$", "R_part": r"$R^{\rm part}_{\rm team}$",
        "R_all": r"$R^{\rm all}_{\rm team}$"}

for mname, off in MEASURES.items():
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 4.6))
    fig.patch.set_facecolor(SURF)
    for ax, key in zip(axes.flat, ORDER):
        ax.set_facecolor(SURF)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(labelsize=7.5, colors=MUTED, length=0, which="both")
        trs = {k: np.load(os.path.join(RES, f"relax_{key}_{k}.npy"))
               for k in ("real", "fill")}
        tail = np.concatenate([trs[k][100:, off] for k in ("real", "fill")])
        mu, sd = tail.mean(), tail.std()
        ax.axhspan(mu - 2 * sd, mu + 2 * sd, color=GRID, alpha=0.55, lw=0)
        for kind, col in (("real", C_REAL), ("fill", C_FILL)):
            x = np.arange(trs[kind].shape[0])
            ax.plot(x, trs[kind][:, off], color=col, lw=1.5)
            ax.plot(x, trs[kind][:, off + 3], color=col, lw=1.0,
                    ls="--", alpha=0.65)
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xlim(0, 300)
        ax.set_xticks([0, 2, 10, 50, 300])
        ax.set_xticklabels(["0", "2", "10", "50", "300"])
        ax.set_title(LABEL[key], fontsize=9, color=INK, pad=4)
    axes.flat[-1].axis("off")
    handles = [
        Line2D([], [], color=C_REAL, lw=2, label="start: dedup observed"),
        Line2D([], [], color=C_FILL, lw=2, label="start: randomized fill"),
        Line2D([], [], color=BODY, lw=1.6, label="M-weighted (fixed |E|)"),
        Line2D([], [], color=BODY, lw=1.1, ls="--", alpha=0.7,
               label=r"p-weighted (support of state)"),
        plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.55,
                      label="tail mean ± 2 sd")]
    axes.flat[-1].legend(handles=handles, fontsize=8, frameon=False,
                         labelcolor=BODY, loc="center left")
    for ax in axes[1, :3]:
        ax.set_xlabel("sweeps", fontsize=8, color=BODY)
    for ax in axes[:, 0]:
        ax.set_ylabel(NICE[mname], fontsize=9, color=BODY)
    fig.suptitle(f"Support-measure relaxation from the deduplicated "
                 f"hypergraph ({NICE[mname]}, θ = 0, 3 ≤ k ≤ 25)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        out = os.path.join(RES, f"relaxation_support_{mname}.{ext}")
        fig.savefig(out, facecolor=SURF, dpi=170)
        print("[saved]", out)
    plt.close(fig)

# How this was run (2026-09-13):
#   <venv-python> make_relaxation_support_fig.py
# Re-renders relaxation_support_{R_any,R_part,R_all}.{pdf,png} from
# relaxation_support_results/*.npy (run relaxation_support.py first).
