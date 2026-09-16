#!/usr/bin/env python3
"""Support-measure relaxation for email-Eu (single-dataset companion to
relaxation_support.py: dedup start + randomized fill, theta=0, 3<=k<=25),
rendered as one 1x3 figure (R_any / R_part / R_all, M solid, p dashed)."""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import relaxation_support as RS          # reuse support_trace / support_vals
import run_extended_sweeps as R
import ess_pilot as EP

OUT = RS.OUT                             # same results dir
KEY = "emaileu"


def _job(task):
    kind, start, seed = task
    return kind, RS.support_trace(start, RS.N_SWEEPS, seed)


def main():
    ev = R.load_raw(KEY)
    dd = sorted({(s, tuple(Rr)) for s, Rr in ev})
    dd = [(s, list(Rr)) for s, Rr in dd]
    fill = EP.projection_random_start(dd, seed=4242)
    assert R.MT.proj_sig(fill) == R.MT.proj_sig(dd)
    print(f"emaileu dedup {len(dd):,}/{len(ev):,} events", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(2) as ex:
        for kind, tr in ex.map(_job, [("real", dd, 17), ("fill", fill, 71)]):
            np.save(os.path.join(OUT, f"relax_{KEY}_{kind}.npy"), tr)
            print(f"  {kind:4s} R_part^M {tr[0,1]:.4f} -> {tr[-1,1]:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68",
                                    "#d9d9d4", "#ffffff")
    C_REAL, C_FILL = "#2a78d6", "#eb6834"
    NICE = {"R_any": r"$R^{\rm any}_{\rm team}$",
            "R_part": r"$R^{\rm part}_{\rm team}$",
            "R_all": r"$R^{\rm all}_{\rm team}$"}
    trs = {k: np.load(os.path.join(OUT, f"relax_{KEY}_{k}.npy"))
           for k in ("real", "fill")}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    fig.patch.set_facecolor(SURF)
    for ax, (mname, off) in zip(axes, [("R_any", 0), ("R_part", 1),
                                       ("R_all", 2)]):
        ax.set_facecolor(SURF)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(labelsize=7.5, colors=MUTED, length=0, which="both")
        tail = np.concatenate([trs[k][100:, off] for k in ("real", "fill")])
        ax.axhspan(tail.mean() - 2 * tail.std(), tail.mean() + 2 * tail.std(),
                   color=GRID, alpha=0.55, lw=0)
        for kind, col in (("real", C_REAL), ("fill", C_FILL)):
            x = np.arange(trs[kind].shape[0])
            ax.plot(x, trs[kind][:, off], color=col, lw=1.5)
            ax.plot(x, trs[kind][:, off + 3], color=col, lw=1.0, ls="--",
                    alpha=0.65)
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xlim(0, RS.N_SWEEPS)
        ax.set_xticks([0, 2, 10, 50, 300])
        ax.set_xticklabels(["0", "2", "10", "50", "300"])
        ax.set_title(NICE[mname], fontsize=10, color=INK, pad=4)
        ax.set_xlabel("sweeps", fontsize=8, color=BODY)
    handles = [Line2D([], [], color=C_REAL, lw=2, label="start: dedup observed"),
               Line2D([], [], color=C_FILL, lw=2, label="start: randomized fill"),
               Line2D([], [], color=BODY, lw=1.6, label="M-weighted"),
               Line2D([], [], color=BODY, lw=1.1, ls="--", alpha=0.7,
                      label="p-weighted"),
               plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.55,
                             label="null mean ± 2 sd (post burn-in)")]
    axes[2].legend(handles=handles, fontsize=6.8, frameon=False,
                   labelcolor=BODY, loc="upper right", handlelength=1.6)
    fig.suptitle("email-Eu: support-measure relaxation from the deduplicated "
                 "hypergraph (θ = 0, 3 ≤ k ≤ 25)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        out = os.path.join(OUT, f"relaxation_support_{KEY}.{ext}")
        fig.savefig(out, facecolor=SURF, dpi=170)
        print("[saved]", out, flush=True)


if __name__ == "__main__":
    main()

# How this was run (2026-09-15):
#   cd Hypergraph_GroupReciprocity/experiments
#   caffeinate -i <venv-python> relaxation_support_emaileu.py \
#       >> relaxation_support_results/run.log 2>&1
# (no CLI arguments; reuses relaxation_support.support_trace; ~10 min.
#  Outputs relax_emaileu_{real,fill}.npy + relaxation_support_emaileu.{pdf,png}
#  into relaxation_support_results/.)
