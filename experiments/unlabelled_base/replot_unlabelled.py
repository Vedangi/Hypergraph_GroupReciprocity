#!/usr/bin/env python3
"""Appendix figure: the labelled vs unlabelled null on RAW email-Eu.
LEFT   how we know each chain converged (and that they need very different
       budgets: labelled flat from 5.6M, unlabelled only settles at 89.6M
       where the high and low starts meet).
RIGHT  the converged theta=0 gap for each measure, both base measures.
Reads multi_unlabelled_probe.csv, multi_unlabelled_mixing.csv and
multi_unlabelled_null_final.json -- no resampling."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg"); matplotlib.rcdefaults()
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, BODY, MUTED, GRID, SURF = ("#1a1a19","#3d3d3a","#6e6d68","#d9d9d4","#ffffff")
CB = {"labelled": "#2a78d6", "unlabelled": "#eb6834"}
MKB = {"labelled": "o", "unlabelled": "s"}
NICE = {"s1_multi": r"$R_{\mathrm{any}}^{\mathrm{multi}}$",
        "r2_multi": r"$R_{\mathrm{some}}^{\mathrm{multi}}$",
        "r1_multi": r"$R_{\mathrm{all}}^{\mathrm{multi}}$",
        "s1_support": r"$R_{\mathrm{any}}$ (support)"}

pr = pd.read_csv(os.path.join(HERE, "multi_unlabelled_probe.csv"))
mx = pd.read_csv(os.path.join(HERE, "multi_unlabelled_mixing.csv"))
fin = json.load(open(os.path.join(HERE, "multi_unlabelled_null_final.json")))
E = fin["real"]["n_events"]

def _style(a):
    a.set_facecolor(SURF); a.grid(False)
    for s in ("top","right"): a.spines[s].set_visible(False)
    for s in ("left","bottom"): a.spines[s].set_color(GRID)
    a.tick_params(labelsize=8.5, colors=MUTED, length=0)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 3.9), facecolor=SURF,
                       gridspec_kw={"width_ratios": [1, 1.2]})
fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.16, wspace=0.30)

# ---- A: convergence ---------------------------------------------------
lab = pr[pr.base == "labelled"]
lx = list(lab.n_steps) + [89_600_000]
ly = list(lab.r2_multi) + [fin["labelled"]["Phi"] / E]
ax[0].plot(lx, ly, marker=MKB["labelled"], ms=6, lw=2, color=CB["labelled"],
           markeredgecolor=SURF, markeredgewidth=1.2,
           label="labelled, from real")
unl = pr[pr.base == "unlabelled"]
hx = list(unl.n_steps) + list(mx.n_steps[1:])
hy = list(unl.r2_multi) + list(mx.r2_from_high[1:])
ax[0].plot(hx, hy, marker=MKB["unlabelled"], ms=6, lw=2, color=CB["unlabelled"],
           markeredgecolor=SURF, markeredgewidth=1.2,
           label="unlabelled, from real")
ax[0].plot(mx.n_steps, mx.r2_from_low, marker=MKB["unlabelled"], ms=6, lw=2,
           ls="--", color=CB["unlabelled"], markeredgecolor=SURF,
           markeredgewidth=1.2, alpha=0.75,
           label="unlabelled, from randomised")
ax[0].set_xscale("log")
ax[0].set_xlabel("MCMC steps", color=BODY)
ax[0].set_ylabel(r"null $R_{\mathrm{some}}^{\mathrm{multi}}$  at $\theta=0$",
                 color=BODY)
ax[0].set_title("Convergence of the two base measures", fontsize=10.5, color=INK)
ax[0].text(0.02, 0.03, "note the axis span: 0.072-0.084", transform=ax[0].transAxes,
           fontsize=7.5, color=MUTED)
ax[0].legend(fontsize=8, frameon=False, labelcolor=BODY, loc="upper right")

# ---- B: converged gaps ------------------------------------------------
keys = ["s1_support", "s1_multi", "r2_multi", "r1_multi"]
rows, y = [], 0.0
for k in keys:
    for base in ("labelled", "unlabelled"):
        rows.append((y, k, base, fin[base][k], fin["real"][k])); y -= 1.0
    y -= 0.55
for yy, k, base, null, real in rows:
    ax[1].plot([null, real], [yy, yy], lw=2, color=GRID, zorder=1,
               solid_capstyle="round")
    ax[1].plot([null], [yy], marker=MKB[base], ms=8, color=CB[base], zorder=3,
               markeredgecolor=SURF, markeredgewidth=1.2)
    ax[1].plot([real], [yy], marker="o", ms=8, color=INK, zorder=3,
               markeredgecolor=SURF, markeredgewidth=1.2)
    ax[1].text(null - 0.008, yy, f"{null:.3f}", ha="right", va="center",
               fontsize=8, color=BODY)
    ax[1].text(real + 0.011, yy, f"{real/null:.2f}×", ha="left", va="center",
               fontsize=8.5, color=INK)
ticks = [(rows[i][0] + rows[i+1][0]) / 2 for i in range(0, len(rows), 2)]
ax[1].set_yticks(ticks, [NICE[k] for k in keys], color=BODY, fontsize=10.5)
ax[1].set_ylim(rows[-1][0] - 0.9, rows[0][0] + 0.9)
ax[1].set_xlim(-0.02, 0.46)
ax[1].set_xlabel(r"group reciprocity  (null $\rightarrow$ observed)", color=BODY)
ax[1].set_title(r"Converged null at $\theta=0$ (89.6M steps)", fontsize=10.5,
                color=INK)
h = [plt.Line2D([], [], ls="none", marker=MKB[b], ms=8, color=CB[b],
                markeredgecolor=SURF, label=f"{b} null") for b in CB]
h.append(plt.Line2D([], [], ls="none", marker="o", ms=8, color=INK,
                    markeredgecolor=SURF, label="observed"))
ax[1].legend(handles=h, fontsize=8.5, frameon=False, labelcolor=BODY,
             loc="lower right")

for a in ax: _style(a)
fig.suptitle("Event-labelled vs unlabelled null on RAW email-Eu "
             "(duplicates kept, 3 ≤ k ≤ 25)", fontsize=12, color=INK)
out = os.path.join(HERE, "multi_unlabelled_emaileu.pdf")
fig.savefig(out, facecolor=SURF); print("[saved]", out)
