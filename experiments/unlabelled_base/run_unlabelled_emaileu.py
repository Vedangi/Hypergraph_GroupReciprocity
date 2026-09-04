#!/usr/bin/env python3
"""run_unlabelled_emaileu.py -- how much of the real-vs-null reciprocity gap on
email-Eu survives when the null is made uniform over UNLABELLED multi-
hypergraphs instead of event-labelled ones.

The sampler moves on event-labelled states (fillings of each sender's fixed-size
event slots).  An unlabelled multi-hypergraph H has
    N(H) = const / Prod_{s,e} m_{s,e}!
labelled representations, so a labelled-uniform null DOWN-weights states with
repeated hyperedges by 1/Prod m!.  base="unlabelled" cancels that with the
Metropolis factor Prod m_new!/Prod m_old!, giving a null that is uniform over
distinct multi-hypergraphs.

Budget: 5.6M steps = 40 sweeps, the point at which the two-start mixing
diagnostic put the gap at ~1%.
    -> multi_unlabelled_emaileu.{pdf,csv}
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "multihypergraph"))

import multi_reciprocity as MR
import multi_tilt as MT

KMAX, N_STEPS, REPS = 25, 5_600_000, 2
THETAS = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
KEYS = ("r2_multi", "s1_multi", "r1_multi", "s1_support",
        "graph_binary", "graph_weighted")

INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68",
                                "#ececE8", "#ffffff")
CB = {"labelled": "#2a78d6", "unlabelled": "#eb6834"}   # validated pair
MKB = {"labelled": "o", "unlabelled": "s"}
NICE = {"r2_multi": r"$R_{\mathrm{some}}^{\mathrm{multi}}$",
        "s1_multi": r"$R_{\mathrm{any}}^{\mathrm{multi}}$",
        "r1_multi": r"$R_{\mathrm{all}}^{\mathrm{multi}}$"}


def load_raw():
    import datasets as lpf
    ev = []
    for s, R, _t in lpf.load_dataset("emaileu"):
        R = sorted(set(int(x) for x in R) - {int(s)})
        if len(R) >= 2 and 1 + len(R) <= KMAX:
            ev.append((int(s), R))
    return ev


def sweep(ev, base):
    rows = []
    for ti, th in enumerate(THETAS):
        acc, ok = {k: [] for k in (*KEYS, "Phi", "accept", "move_rate")}, True
        for r in range(REPS):
            t0 = time.time()
            out, Phi, st = MT.tilt_sample(
                ev, th, n_steps=N_STEPS, seed=997 * r + 13 * ti,
                return_stats=True, base=base)
            ok &= (MT.proj_sig(out) == MT.proj_sig(ev))
            m = MR.measures(out, min_k=3)
            for k in KEYS:
                acc[k].append(m[k])
            acc["Phi"].append(Phi)
            acc["accept"].append(st["accept_rate"])
            acc["move_rate"].append(st["move_rate"])
            print(f"  [{base:10s}] theta={th:+.2f} rep={r+1}/{REPS} "
                  f"Phi={Phi:9,.1f} r2_multi={m['r2_multi']:.4f} "
                  f"acc={st['accept_rate']:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
        rows.append({"base": base, "theta": th,
                     **{k: float(np.mean(v)) for k, v in acc.items()},
                     "r2_sd": float(np.std(acc["r2_multi"])),
                     "projection_fixed": bool(ok)})
    return pd.DataFrame(rows)


def _style(a):
    a.set_facecolor(SURF)
    a.grid(False)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(GRID)
    a.tick_params(labelsize=8.5, colors=MUTED, length=0)


def plot(df, real, path):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.9), facecolor=SURF)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.16,
                        wspace=0.28)

    # --- A: tilt curve under each base measure -------------------------
    for base in ("labelled", "unlabelled"):
        d = df[df.base == base]
        ax[0].plot(d.theta, d.r2_multi, marker=MKB[base], ms=6, lw=2,
                   color=CB[base], label=f"{base} null",
                   markeredgecolor=SURF, markeredgewidth=1.2)
    ax[0].axhline(real["r2_multi"], ls="--", lw=1.4, color=INK, alpha=0.75)
    ax[0].annotate(f"observed  {real['r2_multi']:.3f}",
                   xy=(THETAS[0], real["r2_multi"]), xytext=(0, 5),
                   textcoords="offset points", fontsize=8.5, color=INK)
    ax[0].axvline(0, ls=":", lw=1, color=MUTED)
    ax[0].set_xlabel(r"tilt $\theta$", color=BODY)
    ax[0].set_ylabel(r"$R_{\mathrm{some}}^{\mathrm{multi}}$", color=BODY)
    ax[0].set_title("Tilt curve under each base measure", fontsize=10.5,
                    color=INK)
    ax[0].legend(fontsize=8.5, frameon=False, labelcolor=BODY, loc="upper left")

    # --- B: the theta = 0 null, per measure ----------------------------
    keys = ["r2_multi", "s1_multi", "r1_multi"]
    n0 = {b: df[(df.base == b) & (df.theta == 0.0)].iloc[0] for b in CB}
    y = np.arange(len(keys))[::-1]
    h = 0.30
    for off, base in ((+0.17, "labelled"), (-0.17, "unlabelled")):
        vals = [n0[base][k] for k in keys]
        ax[1].barh(y + off, vals, height=h, color=CB[base], linewidth=0,
                   label=f"{base} null")
        for yy, v in zip(y + off, vals):
            ax[1].text(v + 0.006, yy, f"{v:.3f}", va="center", fontsize=8,
                       color=BODY)
    for yy, k in zip(y, keys):
        ax[1].plot([real[k]], [yy], marker="|", ms=22, mew=2.2, color=INK,
                   zorder=5)
        ax[1].text(real[k], yy + 0.42, f"{real[k]:.3f}", ha="center",
                   fontsize=8, color=INK)
    ax[1].plot([], [], marker="|", ms=12, mew=2.2, ls="none", color=INK,
               label="observed")
    ax[1].set_yticks(y, [NICE[k] for k in keys], color=BODY, fontsize=10)
    ax[1].set_xlim(0, max(real[k] for k in keys) * 1.28)
    ax[1].set_xlabel("group reciprocity", color=BODY)
    ax[1].set_title(r"The null at $\theta=0$", fontsize=10.5, color=INK)
    ax[1].legend(fontsize=8.5, frameon=False, labelcolor=BODY,
                 loc="lower right")

    for a in ax:
        _style(a)
    fig.suptitle("Event-labelled vs unlabelled null on RAW email-Eu "
                 f"(duplicates kept, 3 ≤ k ≤ {KMAX}, {N_STEPS/1e6:.1f}M steps)",
                 fontsize=12, color=INK)
    fig.savefig(path, facecolor=SURF)
    print(f"[saved] {path}", flush=True)


def main():
    ev = load_raw()
    nd = len({(s, frozenset(R)) for s, R in ev})
    real = MR.measures(ev, min_k=3)
    print(f"RAW email-Eu: {len(ev):,} events ({nd:,} distinct, "
          f"{100*(len(ev)-nd)/len(ev):.1f}% duplicates), "
          f"{real['n_teams']:,} teams", flush=True)
    print("real: " + "  ".join(f"{k}={real[k]:.4f}" for k in KEYS), flush=True)

    parts = []
    for base in ("labelled", "unlabelled"):
        parts.append(sweep(ev, base))
        pd.concat(parts, ignore_index=True).to_csv(
            os.path.join(HERE, "multi_unlabelled_emaileu.csv"), index=False)
    df = pd.concat(parts, ignore_index=True)
    print("\n" + df.round(4).to_string(index=False), flush=True)

    print("\n===== null (theta = 0) vs observed =====", flush=True)
    for k in ("r2_multi", "s1_multi", "r1_multi"):
        lab = df[(df.base == "labelled") & (df.theta == 0.0)].iloc[0][k]
        unl = df[(df.base == "unlabelled") & (df.theta == 0.0)].iloc[0][k]
        print(f"  {k:12s} observed={real[k]:.4f}  "
              f"labelled null={lab:.4f} ({real[k]/lab:5.2f}x)  "
              f"unlabelled null={unl:.4f} ({real[k]/unl:5.2f}x)  "
              f"null shifts {100*(unl-lab)/lab:+6.1f}%", flush=True)
    plot(df, real, os.path.join(HERE, "multi_unlabelled_emaileu.pdf"))

    # --- convergence probe at theta = 0 --------------------------------
    # the unlabelled chain accepts less often, so verify 5.6M steps is
    # enough for BOTH base measures before the numbers above are trusted.
    print("\n===== convergence probe at theta = 0 =====", flush=True)
    probe = []
    for base in ("labelled", "unlabelled"):
        for ns in (5_600_000, 11_200_000, 22_400_000):
            v = [MT.tilt_sample(ev, 0.0, n_steps=ns, seed=s_, base=base)[1]
                 for s_ in (31, 32)]
            probe.append({"base": base, "n_steps": ns,
                          "Phi_mean": float(np.mean(v)),
                          "r2_multi": float(np.mean(v)) / real["n_events"],
                          "Phi_spread": float(abs(v[0] - v[1]))})
            print(f"  {base:10s} n_steps={ns:>10,}  Phi={np.mean(v):9,.1f}  "
                  f"r2_multi={np.mean(v)/real['n_events']:.4f}  "
                  f"between-seed spread={abs(v[0]-v[1]):6.1f}", flush=True)
            pd.DataFrame(probe).to_csv(
                os.path.join(HERE, "multi_unlabelled_probe.csv"), index=False)


if __name__ == "__main__":
    main()
