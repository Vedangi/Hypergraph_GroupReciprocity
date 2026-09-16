#!/usr/bin/env python3
"""Figure for the support-family beta sweep: E_beta[R] vs beta for the three
M-weighted support measures, observed values as dashed horizontals, the
R_part moment-match beta-hat marked. Render-only."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "support_beta_results")
KEY = sys.argv[1] if len(sys.argv) > 1 else "emaileu"
LABEL = {"emaileu": "email-Eu", "enron": "Enron"}.get(KEY, KEY)
INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68",
                                "#d9d9d4", "#ffffff")
C = {"R_any": "#eb6834", "R_part": "#2a78d6", "R_all": "#009490"}
MK = {"R_any": "s", "R_part": "o", "R_all": "^"}
NICE = {"R_any": r"$R^{\rm any}_{\rm team}$",
        "R_part": r"$R^{\rm part}_{\rm team}$",
        "R_all": r"$R^{\rm all}_{\rm team}$"}

df = pd.read_csv(os.path.join(RES, f"beta_sweep_{KEY}.csv")).sort_values("beta")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax.tick_params(labelsize=12, colors=MUTED, length=0)
for k in ("R_any", "R_part", "R_all"):
    ax.plot(df.beta, df[k], marker=MK[k], ms=5.5, lw=2, color=C[k],
            label=NICE[k])
    ax.axhline(df[f"obs_{k}"].iloc[0], color=C[k], lw=1.2, ls="--", alpha=0.6)
# moment-match beta-hat for R_part (linear interpolation of the crossing)
obs = df["obs_R_part"].iloc[0]
b, v = df.beta.to_numpy(), df.R_part.to_numpy()
ix = np.where((v[:-1] <= obs) & (v[1:] >= obs))[0]
if len(ix):
    i = ix[0]
    bh = b[i] + (obs - v[i]) / (v[i + 1] - v[i]) * (b[i + 1] - b[i])
    ax.axvline(bh, color=C["R_part"], lw=1.2, ls=":", alpha=0.8)
    ax.annotate(rf"$\hat\beta$ = {bh:.2f}", (bh, obs), xytext=(10, -16),
                textcoords="offset points", fontsize=12.5,
                color=C["R_part"])
ax.text(df.beta.iloc[-1], df["obs_R_any"].iloc[0], " observed",
        fontsize=10.5, color=MUTED, va="bottom", ha="right")
ax.set_xlabel(r"$\beta$", fontsize=14, color=BODY)
ax.set_ylabel("support measure (M-weighted)", fontsize=13, color=BODY)
ax.legend(fontsize=12.5, frameon=False, labelcolor=BODY, loc="upper left")
ax.set_title(f"{LABEL}: support measures under "
             r"$P_\beta \propto e^{\beta\,\Phi}$"
             "  (dedup start)",
             fontsize=12.5, color=INK, pad=8)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(RES, f"support_beta_{KEY}.{ext}")
    fig.savefig(out, facecolor=SURF, dpi=170)
    print("[saved]", out)

# How this was run (2026-09-16):
#   <venv-python> make_support_beta_fig.py emaileu
# Render-only: reads support_beta_results/beta_sweep_<dataset>.csv
# (produced by support_beta_sweep.py).
