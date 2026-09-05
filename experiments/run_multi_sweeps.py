#!/usr/bin/env python3
"""run_multi_sweeps.py -- multi-hypergraph tilt sweeps on the remaining datasets,
plus a mixing diagnostic at high theta.

PART 1  theta-sweep on RAW (non-deduplicated) group hypergraphs for
        enron, congress, dblp, twitter  -> multi_tilt_<key>.{pdf,csv}
PART 2  mixing check on email-Eu: two far-apart starts in the same component
        (the real data, and a long theta=0 randomisation of it) run at three
        increasing step budgets.  If Phi converges to the same value from both
        starts, the sweep values are equilibrium values.
        -> multi_mixing_emaileu.{pdf,csv}

Everything is written incrementally so a partial run is still usable.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in ("src", "src/multihypergraph", "src/dedup"):
    sys.path.insert(0, os.path.join(_ROOT, _p))

import multi_reciprocity as MR
import multi_tilt as MT
import reciprocity_dedup as rf

# optional: path to the DBLP .mat (only needed for the dblp sweep)
DBLP_MAT = os.environ.get("DBLP_MAT", "")

KMAX = 25
MAX_STEPS = 5_000_000          # cap so Congress stays tractable
SWEEP_MULT = 20
REPS = 2
THETAS = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

# congress needs CONGRESS_DIR, dblp needs DBLP_MAT (data not bundled);
# missing ones are skipped with a traceback by the per-dataset guard.
DATASETS = [("enron", "datasets", None), ("congress", "datasets", None),
            ("dblp", "dblp_mat", None), ("twitter", "datasets", None)]

INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68",
                                "#ececE8", "#ffffff")
C = {"r2_multi": "#2a78d6", "s1_multi": "#eb6834", "r1_multi": "#1baf7a",
     "s1_support": "#4a3aa7"}
MK = {"r2_multi": "o", "s1_multi": "s", "r1_multi": "^", "s1_support": "D"}
LB = {"r2_multi": r"$R_{\mathrm{some}}^{\mathrm{multi}}$ (tilted)",
      "s1_multi": r"$R_{\mathrm{any}}^{\mathrm{multi}}$",
      "r1_multi": r"$R_{\mathrm{all}}^{\mathrm{multi}}$",
      "s1_support": r"$R_{\mathrm{any}}$ (support)"}


def load_raw(key, source, max_size=None):
    """RAW group stream: duplicates KEPT, k>=3, size-capped."""
    if source == "dblp_mat":
        df = rf.df_events_from_mat(DBLP_MAT)
        it = [(int(r.sender), list(r.recipients))
              for r in df.itertuples(index=False)]
    else:
        import datasets
        it = [(int(s), list(R))
              for s, R, _t in datasets.load_dataset(key, max_size=max_size)]
    out = []
    for s, R in it:
        R = sorted(set(int(x) for x in R) - {s})
        if len(R) >= 2 and 1 + len(R) <= KMAX:
            out.append((s, R))
    return out


def _style(a):
    a.set_facecolor(SURF)
    a.grid(False)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_color(GRID)
    a.tick_params(labelsize=8.5, colors=MUTED, length=0)


def plot_sweep(label, df, real, path):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.7), facecolor=SURF)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.15,
                        wspace=0.25)
    for col in ("r2_multi", "s1_multi", "r1_multi", "s1_support"):
        ax[0].plot(df.theta, df[col], marker=MK[col], ms=5, lw=1.8,
                   color=C[col], label=LB[col], markeredgecolor=SURF,
                   markeredgewidth=0.8)
        ax[0].axhline(real[col], ls="--", lw=1, color=C[col], alpha=0.5)
    ax[0].axvline(0, ls=":", lw=1, color=MUTED)
    ax[0].set_xlabel(r"tilt $\theta$", color=BODY)
    ax[0].set_ylabel("group reciprocity", color=BODY)
    ax[0].set_title("multiplicity-aware measures (dashed = real data)",
                    fontsize=10.5, color=INK)
    ax[0].legend(fontsize=8, frameon=False, labelcolor=BODY)

    ax[1].plot(df.theta, df.graph_binary, "o-", ms=4.5, lw=1.6,
               color="#d62728", label="binary", markeredgecolor=SURF)
    ax[1].plot(df.theta, df.graph_weighted, "s-", ms=4.5, lw=1.6,
               color=MUTED, label="weighted (Squartini)",
               markeredgecolor=SURF)
    lo = min(real["graph_binary"], real["graph_weighted"])
    hi = max(real["graph_binary"], real["graph_weighted"])
    for v in (real["graph_binary"], real["graph_weighted"]):
        ax[1].axhline(v, ls="--", lw=1, color=INK, alpha=0.5)
    ax[1].set_ylim(lo - 0.08, hi + 0.08)
    ax[1].axvline(0, ls=":", lw=1, color=MUTED)
    ax[1].set_xlabel(r"tilt $\theta$", color=BODY)
    ax[1].set_title("graph reciprocity (invariant)", fontsize=10.5, color=INK)
    ax[1].legend(fontsize=8, frameon=False, labelcolor=BODY)
    for a in ax:
        _style(a)
    fig.suptitle(f"Multi-hypergraph tilt on RAW {label} "
                 f"(duplicates kept, 3 ≤ k ≤ {KMAX})",
                 fontsize=12, color=INK)
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)
    print(f"  [saved] {path}", flush=True)


# ------------------------------------------------------------------- PART 1
def part1():
    allrows = []
    for key, src, ms in DATASETS:
        print(f"\n===== {key} =====", flush=True)
        try:
            ev = load_raw(key, src, ms)
            nd = len({(s, frozenset(R)) for s, R in ev})
            tokens = sum(len(R) for _s, R in ev)
            n_steps = min(SWEEP_MULT * tokens, MAX_STEPS)
            real = MR.measures(ev, min_k=3)
            print(f"  {len(ev):,} events ({nd:,} distinct, "
                  f"{100*(len(ev)-nd)/len(ev):.1f}% dup), {tokens:,} tokens, "
                  f"n_steps={n_steps:,} ({n_steps/tokens:.1f} sweeps)",
                  flush=True)
            print("  real: " + "  ".join(f"{k}={real[k]:.4f}" for k in
                  ("r2_multi", "s1_multi", "r1_multi", "s1_support",
                   "graph_binary", "graph_weighted")), flush=True)
            t0 = time.time()
            df = MT.theta_sweep(ev, THETAS, reps=REPS, n_steps=n_steps,
                                verbose=False)
            print(f"  swept in {time.time()-t0:.0f}s", flush=True)
        except Exception as exc:
            print(f"  SKIPPED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=2)
            continue
        df.insert(0, "dataset", key)
        for k, v in real.items():
            if isinstance(v, (int, float)):
                df[f"real_{k}"] = v
        allrows.append(df)
        print(df[["theta", "Phi", "r2_multi", "s1_multi", "r1_multi",
                  "graph_binary", "graph_weighted", "accept",
                  "projection_fixed"]].round(4).to_string(index=False),
              flush=True)
        plot_sweep(key, df, real, os.path.join(HERE, f"multi_tilt_{key}.pdf"))
        pd.concat(allrows, ignore_index=True).to_csv(
            os.path.join(HERE, "multi_tilt_all.csv"), index=False)


# ------------------------------------------------------------------- PART 2
def part2():
    print("\n\n===== MIXING TEST (email-Eu, raw) =====", flush=True)
    ev = load_raw("emaileu", "datasets", None)
    tokens = sum(len(R) for _s, R in ev)
    print(f"  {len(ev):,} events, {tokens:,} tokens; "
          f"Phi(real) = {MT._build_state(ev,'r2')[2]:,.1f}", flush=True)

    # far-away start in the SAME component: long theta=0 randomisation
    B, _ = MT.tilt_sample(ev, 0.0, n_steps=8_000_000, seed=4242)
    print(f"  start B (long theta=0 run): Phi = "
          f"{MT._build_state(B,'r2')[2]:,.1f}", flush=True)

    rows = []
    for th in (1.5, 2.0, 2.5):
        for ns in (1_400_000, 5_600_000, 22_400_000):
            vals = {}
            for nm, st in (("A_real", ev), ("B_random", B)):
                ph = [MT.tilt_sample(st, th, n_steps=ns, seed=s)[1]
                      for s in (1, 2)]
                vals[nm] = float(np.mean(ph))
            gap = abs(vals["A_real"] - vals["B_random"]) / max(vals.values())
            rows.append({"theta": th, "n_steps": ns,
                         "Phi_from_real": vals["A_real"],
                         "Phi_from_random": vals["B_random"],
                         "rel_gap": gap})
            print(f"  theta={th:+.1f} n_steps={ns:>10,} | "
                  f"from_real={vals['A_real']:9,.0f} "
                  f"from_random={vals['B_random']:9,.0f} gap={gap:6.2%}",
                  flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(HERE, "multi_mixing_emaileu.csv"), index=False)

    mix = pd.DataFrame(rows)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.6), facecolor=SURF)
    for i, th in enumerate(sorted(mix.theta.unique())):
        d = mix[mix.theta == th]
        ax.plot(d.n_steps, d.rel_gap * 100, "o-", ms=5, lw=1.8,
                color=["#2a78d6", "#eb6834", "#1baf7a"][i],
                label=rf"$\theta={th}$", markeredgecolor=SURF)
    ax.set_xscale("log")
    ax.set_xlabel("MCMC steps", color=BODY)
    ax.set_ylabel(r"relative gap in $\Phi$ between starts (%)", color=BODY)
    ax.set_title("Mixing: two far-apart starts converge?", fontsize=10.5,
                 color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=BODY)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "multi_mixing_emaileu.pdf"),
                facecolor=SURF)
    fig.savefig(os.path.join(HERE, "multi_mixing_emaileu.png"),
                facecolor=SURF, dpi=200)
    print("  [saved] multi_mixing_emaileu.{pdf,png,csv}", flush=True)


if __name__ == "__main__":
    part1()
    part2()
    print("\nDONE", flush=True)
