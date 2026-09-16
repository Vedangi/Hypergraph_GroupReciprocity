#!/usr/bin/env python3
"""Support-measure relaxation figure for email-Eu -- M-weighted only
(1x3: R_any / R_part / R_all; dedup-observed start vs randomized fill;
band = stationary tail mean +- 2 sd). Render-only: reads the traces
produced by relaxation_support_emaileu.py."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "relaxation_support_results")
KEY, LABEL = "emaileu", "email-Eu"
INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68",
                                "#d9d9d4", "#ffffff")
C_REAL, C_FILL = "#2a78d6", "#eb6834"
NICE = {"R_any": r"$R^{\rm any}_{\rm team}$",
        "R_part": r"$R^{\rm part}_{\rm team}$",
        "R_all": r"$R^{\rm all}_{\rm team}$"}

trs = {k: np.load(os.path.join(RES, f"relax_{KEY}_{k}.npy"))
       for k in ("real", "fill")}
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5))
fig.patch.set_facecolor(SURF)
for ax, (mname, off) in zip(axes, [("R_any", 0), ("R_part", 1),
                                   ("R_all", 2)]):
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=11, colors=MUTED, length=0, which="both")
    tail = np.concatenate([trs[k][100:, off] for k in ("real", "fill")])
    ax.axhspan(tail.mean() - 2 * tail.std(), tail.mean() + 2 * tail.std(),
               color=GRID, alpha=0.55, lw=0)
    for kind, col in (("real", C_REAL), ("fill", C_FILL)):
        x = np.arange(trs[kind].shape[0])
        ax.plot(x, trs[kind][:, off], color=col, lw=1.6)
    obs = trs["real"][0, off]
    ax.text(0.97, 0.9, f"observed (real network): {obs:.3f}",
            transform=ax.transAxes, fontsize=11.5, color=MUTED,
            ha="right", va="center")
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xlim(0, trs["real"].shape[0] - 1)
    ax.set_xticks([0, 2, 10, 50, 300])
    ax.set_xticklabels(["0", "2", "10", "50", "300"])
    ax.set_title(f"{NICE[mname]} — {LABEL}", fontsize=15, color=INK, pad=4)
    ax.set_xlabel("sweeps", fontsize=12, color=BODY)
handles = [Line2D([], [], color=C_REAL, lw=2, label="start: dedup observed"),
           Line2D([], [], color=C_FILL, lw=2, label="start: randomized fill"),
           plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.55,
                         label="null mean ± 2 sd")]
axes[1].legend(handles=handles, fontsize=9.5, frameon=False,
                   labelcolor=BODY, loc="center", handlelength=1.5)
fig.suptitle("email-Eu: support-measure relaxation from the deduplicated hypergraph",
             fontsize=15, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ("pdf", "png"):
    out = os.path.join(RES, f"relaxation_support_{KEY}_M.{ext}")
    fig.savefig(out, facecolor=SURF, dpi=170)
    print("[saved]", out)

# How this was run (2026-09-15):
#   cd Hypergraph_GroupReciprocity/experiments
#   <venv-python> make_relax_support_emaileu_M.py
# Render-only (no chains): needs relaxation_support_results/relax_emaileu_*.npy
# from relaxation_support_emaileu.py. Output: relaxation_support_emaileu_M.{pdf,png}.
