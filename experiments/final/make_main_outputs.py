#!/usr/bin/env python3
"""Main-paper table and figure for the beta=0 null experiment (plan sec. 10).
Render-only: reads results/final/summary.csv (written by diagnose_final.py).

Table  : observed, null mean, observed - null for the three multiplicity-aware
         measures; '*' marks observed outside the 95% null reference interval
         (here always: observed exceeds every retained draw). -> main_table.{md,tex}
Figure : dot plot, datasets on the y-axis grouped by domain; left panel
         observed - E0[R] (absolute), right panel observed / E0[R] (log);
         bars span the null 95% reference interval mapped onto each quantity.
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = [("emaileu", "Email-Eu", "Communication"), ("enron", "Enron", "Communication"),
         ("dnc", "DNC", "Communication"), ("fauci", "Fauci", "Communication"),
         ("radoslaw", "Radoslaw", "Communication"), ("twitter", "Twitter", "Communication"),
         ("congress", "Congress-bills", "Collaboration"), ("dblp", "DBLP", "Collaboration"),
         ("music", "Music", "Collaboration")]
MEASURES = [("s1_multi", r"$R^{\rm m}_{\rm any}$", "#eb6834", "s"),
            ("r2_multi", r"$R^{\rm m}_{\rm part}$", "#2a78d6", "o"),
            ("r1_multi", r"$R^{\rm m}_{\rm all}$", "#009490", "^")]
TEX_NAME = {"s1_multi": r"R^{\mathrm m}_{\mathrm{any}}", "r2_multi": r"R^{\mathrm m}_{\mathrm{part}}",
            "r1_multi": r"R^{\mathrm m}_{\mathrm{all}}"}
INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68", "#d9d9d4", "#ffffff")


def load(path, filt="k3-25"):
    s = pd.read_csv(path)
    s = s[(s.beta == 0) & (s["filter"] == filt) & s.measure.isin([m for m, *_ in MEASURES])]
    return {(r.dataset, r.measure): r for r in s.itertuples()}


def fmt(x, nd=3):
    if x >= 0.01:
        return f"{x:.{nd}f}"
    if x >= 0.0005:
        return f"{x:.4f}"
    return f"{x:.1e}"            # e.g. Congress R_all null mean 2.4e-05


def make_table(S, out_md, out_tex):
    hdr = "| Domain | Dataset | " + " | ".join(
        f"{lab} obs | null | obs−null" for _, lab, *_ in MEASURES) + " |"
    md = ["# Observed vs projection-preserving null (β = 0, 3 ≤ k ≤ 25)", "",
          hdr, "|" + "---|" * (2 + 3 * len(MEASURES))]
    tex = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Observed group reciprocity versus its expectation under the "
           r"projection-preserving null model ($\beta=0$; $3\le k\le 25$; nine datasets, "
           r"six chains each, all post-burn-in states). Null means are expectations under "
           r"the fixed-weighted-projection, fixed-event-size null; $^{*}$ marks observed "
           r"values outside the 95\% null reference interval (in every case the observed "
           r"value exceeds all retained null draws). Null reference intervals, "
           r"$\widehat R$, ESS and Monte Carlo errors are reported in the appendix.}",
           r"\label{tab:null-main}", r"\small", r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
           r"Domain & Dataset & " + " & ".join(
               rf"\multicolumn{{3}}{{c}}{{${TEX_NAME[m]}$}}" for m, *_ in MEASURES) + r" \\",
           r" & & " + " & ".join("obs. & null & diff." for _ in MEASURES) + r" \\", r"\midrule"]
    prev = None
    for key, name, dom in ORDER:
        cells_md, cells_tex = [], []
        for m, *_ in MEASURES:
            r = S[(key, m)]
            star = "*" if r.obs_exceeds_all_draws else ""
            cells_md += [fmt(r.observed) + star, fmt(r.null_mean), f"+{fmt(r.diff)}"]
            cells_tex += [fmt(r.observed) + (r"$^{*}$" if star else ""), fmt(r.null_mean),
                          f"+{fmt(r.diff)}"]
        if prev and dom != prev:
            tex.append(r"\midrule")
        md.append(f"| {dom if dom != prev else ''} | {name} | " + " | ".join(cells_md) + " |")
        tex.append(f"{dom if dom != prev else ''} & {name} & " + " & ".join(cells_tex) + r" \\")
        prev = dom
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    md.append("\n`*` observed value exceeds every retained null draw (outside the 95% null reference interval).")
    open(out_md, "w").write("\n".join(md) + "\n")
    open(out_tex, "w").write("\n".join(tex) + "\n")


def make_figure(S, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1]))
    fig.patch.set_facecolor(SURF)
    ypos = {}
    y = 0
    labels, prev = [], None
    for key, name, dom in ORDER:
        if prev and dom != prev:
            y += 0.8                                   # gap between domains
        ypos[key] = y; labels.append((y, name)); y += 1; prev = dom
    for ax, kind in zip(axes, ("diff", "ratio")):
        ax.set_facecolor(SURF)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.xaxis.grid(True, color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
        ax.tick_params(labelsize=11, colors=MUTED, length=0)
        for m, lab, col, mk in MEASURES:
            off = {"s1_multi": -0.22, "r2_multi": 0.0, "r1_multi": 0.22}[m]
            for key, *_ in ORDER:
                r = S[(key, m)]
                if kind == "diff":
                    x, lo, hi = r.diff, r.observed - r.q975, r.observed - r.q025
                else:   # q025 can be exactly 0 for R_all (no rotation under the null)
                    x = r.ratio
                    lo = r.observed / r.q975 if r.q975 > 0 else x
                    hi = r.observed / r.q025 if r.q025 > 0 else x   # bar ends at the dot
                ax.plot([lo, hi], [ypos[key] + off] * 2, color=col, lw=1.6, alpha=0.6)
                ax.plot(x, ypos[key] + off, marker=mk, ms=7, color=col,
                        markeredgecolor=SURF, markeredgewidth=0.9, ls="", zorder=3)
        if kind == "diff":
            ax.axvline(0, color=MUTED, lw=1, ls="--")
            ax.set_xlabel(r"$R_{\rm obs} - \mathbb{E}_0[R]$", fontsize=12.5, color=BODY)
            ax.set_title("absolute excess over the null", fontsize=12, color=INK)
        else:
            ax.set_xscale("log"); ax.axvline(1, color=MUTED, lw=1, ls="--")
            ax.set_xlabel(r"$R_{\rm obs}\,/\,\mathbb{E}_0[R]$", fontsize=12.5, color=BODY)
            ax.set_title("relative enrichment", fontsize=12, color=INK)
            ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    axes[0].set_yticks([p for p, _ in labels]); axes[0].set_yticklabels([n for _, n in labels],
                                                                        fontsize=11.5, color=BODY)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axhline(ypos["congress"] - 0.9, color=GRID, lw=0.8)
    axes[0].text(axes[0].get_xlim()[1], ypos["emaileu"] - 0.55, "communication",
                 fontsize=10, color=MUTED, ha="right", va="bottom")
    axes[0].text(axes[0].get_xlim()[1], ypos["congress"] - 0.55, "collaboration",
                 fontsize=10, color=MUTED, ha="right", va="bottom")
    handles = [Line2D([], [], marker=mk, ms=7, color=col, markeredgecolor=SURF, ls="", label=lab)
               for _, lab, col, mk in MEASURES]
    fig.legend(handles=handles, fontsize=12, frameon=False, labelcolor=BODY, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.01), handletextpad=0.4, columnspacing=2.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", facecolor=SURF, dpi=170); print("[saved]", f"{out}.{ext}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(HERE, "..", "..", "results", "final"))
    a = p.parse_args()
    S = load(os.path.join(a.results, "summary.csv"))
    make_table(S, os.path.join(a.results, "main_table.md"), os.path.join(a.results, "main_table.tex"))
    print("[saved]", os.path.join(a.results, "main_table.{md,tex}"))
    make_figure(S, os.path.join(a.results, "fig_null_separation"))


if __name__ == "__main__":
    main()

# How to run:  python make_main_outputs.py   (after diagnose_final.py)
# Outputs in results/final/: main_table.md, main_table.tex, fig_null_separation.{pdf,png}
