#!/usr/bin/env python3
"""Relaxation figure: R_some per sweep at theta=0, real start (decays) vs
projection-fill start (begins at the null), null band from the ESS runs."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "relaxation_results")
INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68",
                                "#d9d9d4", "#ffffff")
C_REAL, C_FILL = "#2a78d6", "#eb6834"
LABEL = {"fauci": "Fauci email", "wiki": "Wiki talk", "dnc": "DNC email",
         "radoslaw": "Manufacturing email", "higgs": "Higgs Twitter",
         "twitter": "Twitter (Congress)", "enron": "Enron"}

meta = pd.read_csv(os.path.join(RES, "meta.csv")).set_index("dataset")
ess = pd.read_csv(os.path.join(HERE, "ess_pilot_results", "ess_pilot.csv")
                  ).set_index("dataset")
order = ["fauci", "wiki", "dnc", "radoslaw", "higgs", "twitter", "enron"]

fig, axes = plt.subplots(2, 4, figsize=(10.4, 4.6))
fig.patch.set_facecolor(SURF)
for ax, key in zip(axes.flat, order):
    E = meta.loc[key, "n_events"]
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    mu, sd = ess.loc[key, "null_r2"], ess.loc[key, "null_sd"]
    ax.axhspan(mu - 2 * sd, mu + 2 * sd, color=GRID, alpha=0.55, lw=0)
    for kind, col in (("real", C_REAL), ("fill", C_FILL)):
        tr = np.load(os.path.join(RES, f"relax_{key}_{kind}.npy")) / E
        ax.plot(np.arange(len(tr)), tr, color=col, lw=1.6)
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xlim(0, 300)
    ax.set_xticks([0, 2, 10, 50, 300])
    ax.set_xticklabels(["0", "2", "10", "50", "300"])
    ax.set_title(LABEL[key], fontsize=9, color=INK, pad=4)
    ax.text(295, meta.loc[key, "real_r2"], f"{meta.loc[key,'real_r2']:.3f}",
            fontsize=7, color=MUTED, ha="right", va="bottom")
axes.flat[-1].axis("off")
handles = [Line2D([], [], color=C_REAL, lw=2, label="start: observed hypergraph"),
           Line2D([], [], color=C_FILL, lw=2, label="start: randomized fill"),
           plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.55,
                         label="null mean ± 2 sd (ESS runs)")]
axes.flat[-1].legend(handles=handles, fontsize=8.5, frameon=False,
                     labelcolor=BODY, loc="center left")
for ax in axes[1, :3]:
    ax.set_xlabel("sweeps", fontsize=8, color=BODY)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$R_{\mathrm{some}}^{\mathrm{multi}}$",
                  fontsize=8.5, color=BODY)
fig.suptitle("Relaxation to the projection-fixed null at θ = 0",
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ("pdf", "png"):
    out = os.path.join(RES, f"relaxation.{ext}")
    fig.savefig(out, facecolor=SURF, dpi=170)
    print("[saved]", out)

# How this was run (2026-09-13):
#   <venv-python> make_relaxation_fig.py
# Re-renders relaxation.{pdf,png} from relaxation_results/*.npy; needs
# ess_pilot_results/ess_pilot.csv for the null band. No chains re-run.
