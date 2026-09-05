#!/usr/bin/env python3
"""run_extended_sweeps.py -- extended theta sweep on the COMMUNICATION datasets,
in the labelled multi-hypergraph model, with a built-in two-start mixing check.

WHY EXTENDED:  the earlier grid stopped at theta=2 and every dataset still had
E_theta[r2_multi] < observed there, so theta-hat was only ever a lower bound.
This grid runs to theta=5 so the MLE is bracketed.

WHY TWO STARTS:  a single chain started at the real data and read off at its
terminal state can look converged when it is not (that is exactly how the
unlabelled email-Eu null misreported itself by ~10%).  Here rep 0 starts from
the REAL hypergraph and rep 1 starts from a long randomisation of it, so the
per-theta spread between reps IS the mixing diagnostic -- at no extra cost.

BUDGET:  40 sweeps (= 40 x #recipient-tokens), the point at which the email-Eu
two-start test agreed to ~1%; floor of 2M steps for the small datasets.
No MAX_STEPS cap, so Congress and Enron get their full 40 sweeps this time.

    -> extended_sweeps.csv, ext_tilt_<key>.pdf, ext_theta_hat.pdf
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MH = os.path.abspath(os.path.join(HERE, ".."))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in ("src", "src/multihypergraph", "src/dedup"):
    sys.path.insert(0, os.path.join(_ROOT, _p))

import multi_reciprocity as MR
import multi_tilt as MT

KMAX = 25
SWEEP_MULT = 40
MIN_STEPS = 2_000_000
THETAS = [0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
# (key, label, n_chains).  Chains alternate start: even -> REAL hypergraph,
# odd -> a long randomisation of it, so mean(real starts) vs mean(random
# starts) is a two-start mixing diagnostic at every theta.  Small datasets get
# more chains because their Phi is small and MC noise, not mixing, dominates.
# Smallest first, so partial results are useful early.
# Communication networks only -- Congress bills (cosponsorship) is out of scope.
DATASETS = [("fauci", "Fauci email", 6), ("dnc", "DNC email", 6),
            ("twitter", "Twitter", 6), ("enron", "Enron", 2)]

INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68",
                                "#d9d9d4", "#ffffff")
C = {"r2_multi": "#2a78d6", "s1_multi": "#eb6834",
     "r1_multi": "#009490", "s1_support": "#8a2f8a"}   # validated palette
MK = {"r2_multi": "o", "s1_multi": "s", "r1_multi": "^", "s1_support": "D"}
LB = {"r2_multi": r"$R_{\mathrm{some}}^{\mathrm{multi}}$",
      "s1_multi": r"$R_{\mathrm{any}}^{\mathrm{multi}}$",
      "r1_multi": r"$R_{\mathrm{all}}^{\mathrm{multi}}$",
      "s1_support": r"$R_{\mathrm{any}}$ (support)"}
KEYS = ("r2_multi", "s1_multi", "r1_multi", "s1_support",
        "graph_binary", "graph_weighted")


def load_raw(key):
    """RAW group stream: duplicates KEPT, 3 <= k <= KMAX."""
    import datasets
    it = [(int(s), list(R)) for s, R, _t in datasets.load_dataset(key)]
    out = []
    for s, R in it:
        R = sorted(set(int(x) for x in R) - {s})
        if len(R) >= 2 and 1 + len(R) <= KMAX:
            out.append((s, R))
    return out


def theta_hat(thetas, vals, target):
    """Linear interpolation of the theta at which the curve crosses target."""
    t, v = np.asarray(thetas, float), np.asarray(vals, float)
    o = np.argsort(t); t, v = t[o], v[o]
    for i in range(len(t) - 1):
        if (v[i] - target) * (v[i + 1] - target) <= 0 and v[i + 1] != v[i]:
            return float(t[i] + (target - v[i]) * (t[i + 1] - t[i])
                         / (v[i + 1] - v[i]))
    return float("nan")


def sweep_one(key, label, n_chains):
    ev = load_raw(key)
    nd = len({(s, frozenset(R)) for s, R in ev})
    tokens = sum(len(R) for _s, R in ev)
    n_steps = max(SWEEP_MULT * tokens, MIN_STEPS)
    real = MR.measures(ev, min_k=3)
    print(f"\n===== {label} ({key}) =====", flush=True)
    print(f"  {len(ev):,} events ({nd:,} distinct, "
          f"{100*(len(ev)-nd)/len(ev):.1f}% dup), {tokens:,} tokens, "
          f"{real['n_teams']:,} teams", flush=True)
    print(f"  n_steps={n_steps:,} ({n_steps/tokens:.0f} sweeps), "
          f"{n_chains} chains/theta", flush=True)
    print("  real: " + "  ".join(f"{k}={real[k]:.4f}" for k in KEYS),
          flush=True)

    t0 = time.time()
    rand_start, _ = MT.tilt_sample(ev, 0.0, n_steps=n_steps, seed=8888)
    print(f"  randomised start built in {time.time()-t0:.0f}s "
          f"(Phi {MT._build_state(ev,'r2')[2]:,.0f} -> "
          f"{MT._build_state(rand_start,'r2')[2]:,.0f})", flush=True)

    rows = []
    for ti, th in enumerate(THETAS):
        acc = {k: [] for k in (*KEYS, "Phi", "accept")}
        phi_by_start = {"real": [], "random": []}
        ok = True
        for r in range(n_chains):
            which = "real" if r % 2 == 0 else "random"
            start = ev if which == "real" else rand_start
            t0 = time.time()
            out, Phi, st = MT.tilt_sample(start, th, n_steps=n_steps,
                                          seed=997 * r + 13 * ti,
                                          return_stats=True, base="labelled")
            ok &= (MT.proj_sig(out) == MT.proj_sig(ev))
            m = MR.measures(out, min_k=3)
            for k in KEYS:
                acc[k].append(m[k])
            acc["Phi"].append(Phi)
            acc["accept"].append(st["accept_rate"])
            phi_by_start[which].append(Phi)
            print(f"    theta={th:+.2f} start={which:6s} "
                  f"Phi={Phi:10,.1f} r2={m['r2_multi']:.4f} "
                  f"acc={st['accept_rate']:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
        mr_, mn_ = np.mean(phi_by_start["real"]), np.mean(phi_by_start["random"])
        gap = abs(mr_ - mn_) / max(mr_, mn_) if max(mr_, mn_) else 0.0
        rows.append({"dataset": key, "label": label, "theta": th,
                     **{k: float(np.mean(v)) for k, v in acc.items()},
                     "Phi_real_start": float(mr_),
                     "Phi_random_start": float(mn_),
                     "Phi_sd": float(np.std(acc["Phi"])),
                     "two_start_gap": gap, "n_chains": n_chains,
                     "n_steps": n_steps, "sweeps": n_steps / tokens,
                     "projection_fixed": bool(ok),
                     **{f"real_{k}": real[k] for k in KEYS},
                     "n_events": len(ev), "n_teams": real["n_teams"],
                     "pct_dup": 100 * (len(ev) - nd) / len(ev)})
        print(f"    -> two-start gap in Phi: {gap:.2%}"
              f"{'   <-- CHECK MIXING' if gap > 0.03 else ''}", flush=True)
    df = pd.DataFrame(rows)
    th = theta_hat(df.theta, df.r2_multi, real["r2_multi"])
    df["theta_hat"] = th
    print(f"  theta_hat (r2_multi) = "
          f"{th if np.isfinite(th) else float('nan'):.3f}"
          if np.isfinite(th) else "  theta_hat = NOT BRACKETED (> 5)",
          flush=True)
    return df, real


def _style(a):
    a.set_facecolor(SURF); a.grid(False)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(labelsize=8.5, colors=MUTED, length=0)


def plot_one(df, real, label, path):
    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcdefaults()
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.9), facecolor=SURF,
                           gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.155,
                        wspace=0.26)
    for col in ("r2_multi", "s1_multi", "r1_multi", "s1_support"):
        ax[0].plot(df.theta, df[col], marker=MK[col], ms=5.5, lw=1.9,
                   color=C[col], label=LB[col], markeredgecolor=SURF,
                   markeredgewidth=1.0)
        ax[0].axhline(real[col], ls="--", lw=1, color=C[col], alpha=0.45)
    th = df.theta_hat.iloc[0]
    if np.isfinite(th):
        ax[0].axvline(th, ls="-", lw=1.2, color=INK, alpha=0.55)
        ax[0].text(th, ax[0].get_ylim()[1], rf"  $\hat\theta={th:.2f}$",
                   va="top", ha="left", fontsize=9, color=INK)
    ax[0].set_xlabel(r"tilt $\theta$", color=BODY)
    ax[0].set_ylabel("group reciprocity", color=BODY)
    ax[0].set_title("Extended tilt sweep (dashed = observed)", fontsize=10.5,
                    color=INK)
    ax[0].legend(fontsize=8, frameon=False, labelcolor=BODY, loc="upper left")

    ax[1].plot(df.theta, df.two_start_gap * 100, "o-", ms=5.5, lw=1.9,
               color="#2a78d6", markeredgecolor=SURF, markeredgewidth=1.0)
    ax[1].axhline(3, ls="--", lw=1, color=MUTED)
    ax[1].text(df.theta.max(), 3.15, "3% guide", ha="right", va="bottom",
               fontsize=8, color=MUTED)
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel(r"tilt $\theta$", color=BODY)
    ax[1].set_ylabel(r"two-start gap in $\Phi$ (%)", color=BODY)
    ax[1].set_title("Mixing check: real vs randomised start", fontsize=10.5,
                    color=INK)
    for a in ax: _style(a)
    fig.suptitle(f"{label}: extended $\\theta$ sweep, labelled multi-hypergraph "
                 f"({df.sweeps.iloc[0]:.0f} sweeps, 3 ≤ k ≤ {KMAX})",
                 fontsize=12, color=INK)
    fig.savefig(path, facecolor=SURF); plt.close(fig)
    print(f"  [saved] {os.path.basename(path)}", flush=True)


def plot_summary(all_df, path):
    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcdefaults()
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    g = all_df.groupby(["dataset", "label"], sort=False).first().reset_index()
    g = g.sort_values("theta_hat")
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.5), facecolor=SURF)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.17,
                        wspace=0.42)
    y = np.arange(len(g))[::-1]
    ax[0].hlines(y, 0, g.theta_hat, lw=2, color=GRID)
    ax[0].plot(g.theta_hat, y, "o", ms=9, color="#2a78d6",
               markeredgecolor=SURF, markeredgewidth=1.4)
    for yy, v in zip(y, g.theta_hat):
        ax[0].text(v + 0.1, yy, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax[0].set_yticks(y, g.label, color=BODY, fontsize=9.5)
    ax[0].set_xlim(0, float(np.nanmax(g.theta_hat)) * 1.22)
    ax[0].set_xlabel(r"$\hat\theta$  (moment-matched)", color=BODY)
    ax[0].set_title(r"Fitted tilt $\hat\theta$", fontsize=10.5, color=INK)

    nulls = all_df[all_df.theta == 0.0].set_index("dataset")
    ratio = (nulls.real_r2_multi / nulls.r2_multi).reindex(g.dataset)
    ax[1].hlines(y, 1, ratio.values, lw=2, color=GRID)
    ax[1].plot(ratio.values, y, "o", ms=9, color="#eb6834",
               markeredgecolor=SURF, markeredgewidth=1.4)
    for yy, v in zip(y, ratio.values):
        ax[1].text(v * 1.04, yy, f"{v:.1f}×", va="center", fontsize=9,
                   color=INK)
    ax[1].set_yticks(y, ["" for _ in y])
    ax[1].set_xscale("log")
    ax[1].set_xlim(1, float(np.nanmax(ratio.values)) * 1.5)
    ax[1].set_xlabel(r"observed / null   ($R_{\mathrm{some}}^{\mathrm{multi}}$)",
                     color=BODY)
    ax[1].set_title("Excess over the projection-fixed null", fontsize=10.5,
                    color=INK)
    for a in ax: _style(a)
    fig.suptitle("Communication datasets, labelled multi-hypergraph model",
                 fontsize=12, color=INK)
    fig.savefig(path, facecolor=SURF); plt.close(fig)
    print(f"[saved] {os.path.basename(path)}", flush=True)


def main():
    parts = []
    for key, label, n_chains in DATASETS:
        try:
            df, real = sweep_one(key, label, n_chains)
        except Exception as exc:
            print(f"  SKIPPED {key}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=3)
            continue
        parts.append(df)
        pd.concat(parts, ignore_index=True).to_csv(
            os.path.join(HERE, "extended_sweeps.csv"), index=False)
        plot_one(df, real, label, os.path.join(HERE, f"ext_tilt_{key}.pdf"))
    if parts:
        allp = pd.concat(parts, ignore_index=True)
        plot_summary(allp, os.path.join(HERE, "ext_theta_hat.pdf"))
        print("\n===== SUMMARY =====", flush=True)
        for key, g in allp.groupby("dataset", sort=False):
            n0 = g[g.theta == 0.0].iloc[0]
            print(f"  {g.label.iloc[0]:16s} theta_hat={g.theta_hat.iloc[0]:5.2f}"
                  f"  null={n0.r2_multi:.4f} real={n0.real_r2_multi:.4f}"
                  f"  ({n0.real_r2_multi/n0.r2_multi:5.2f}x)"
                  f"  worst two-start gap={g.two_start_gap.max():.2%}",
                  flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
