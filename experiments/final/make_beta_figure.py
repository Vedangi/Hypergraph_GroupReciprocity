#!/usr/bin/env python3
"""Beta-control figure + table for the exponential tilt P_beta ∝ exp(beta*Phi),
Phi = |E| * R_part^m (plan sec. 6.4 / sec. 10). Render-only: reads the final
phase-B traces (results/final/<ds>/beta<b>/chain*/trace.csv) and reuses the
phase-A beta=0 chains; recomputes every number from the traces.

Per (dataset, beta, measure): pooled post-burn-in mean, sd across states,
Sokal tau_int, pooled ESS, MCSE = sd*sqrt(tau/N) and split-Rhat (same
estimators as diagnose_final.py). A combo is used only if the manifest says
all its chains reached the target sweeps.

--set multi    : R^m_any, R^m_part, R^m_all (the tilt acts on R^m_part)
--set support  : M-weighted support measures evaluated on the SUPPORT of each
                 occurrence-level tilted state (plan sec. 7 wording; this is
                 not a support-only state space and the tilt is still Phi^m)
--set support_tilt : phase-T chains: dedup start E*, tilt Phi = |E*| R_part^M
                 (the M-weighted support partial reciprocity); beta=0 = phase D

Figure: one facet per dataset, beta on x, E_beta[R] on y, one curve per
measure, error bars = ±2 MCSE (usually smaller than the marker), observed
values as dashed horizontals, R_part moment-match beta-hat marked. Markers
are filled when the point passes the mixing rule (ESS(Phi) >= --ess-min and
Rhat <= --rhat-max for all plotted measures) and hollow otherwise.
"""
import argparse, glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from diagnose_final import tau_int, split_rhat                  # noqa: E402
from make_main_outputs import SETS, INK, BODY, MUTED, GRID, SURF  # noqa: E402

DATASETS = [("emaileu", "Email-Eu"), ("enron", "Enron"),
            ("twitter", "Twitter"), ("congress", "Congress-bills")]
PRIMARY = {"multi": "r2_multi", "support": "r2_support_M", "support_tilt": "r2_support_M"}
# support_tilt: phase T chains -- dedup start E*, tilt Phi = |E*| R_part^M
# (phi="r2M"); beta=0 is phase D. Same measures as "support", own tag.
SETS["support_tilt"] = dict(SETS["support"], tag="_support_tilt", dedup=True,
                            title="support measures (M-weighted), support tilt, dedup start")


def combo_dirs(root, ds, dedup=False):
    """{beta: combo_dir} for the k3-25 runs of one dataset: the
    multihypergraph-start chains (dedup=False) or the dedup-start chains."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, ds, "beta*"))):
        tag = os.path.basename(d)
        if "_k" in tag or tag.endswith("_dedup") != dedup:
            continue
        out[float(tag.replace("beta", "").replace("_dedup", ""))] = d
    return out


def load_combo(cdir, manifest, root, burn):
    """Post-burn-in per-chain arrays (dict col -> list of arrays) or None."""
    combo = os.path.relpath(cdir, root)
    m = manifest.get(combo)
    chains = sorted(glob.glob(os.path.join(cdir, "chain*", "trace.csv")))
    if m is None or len(chains) < m["n_chains"]:
        return None, f"{combo}: {len(chains)} chains on disk, manifest expects {m and m['n_chains']}"
    per, obs = [], None
    for tr in chains:
        meta = json.load(open(os.path.join(os.path.dirname(tr), "meta.json")))
        if meta.get("status") != "complete" or meta["sweeps"] < m["sweeps"]:
            return None, f"{combo}: {os.path.basename(os.path.dirname(tr))} at {meta['sweeps']}/{m['sweeps']} sweeps"
        t = pd.read_csv(tr)
        per.append(t.iloc[int(burn * len(t)):].reset_index(drop=True))
        obs = meta["observed"]
    return (per, obs), None


def summarize(per, col):
    xs = [p[col].to_numpy(float) for p in per]
    pooled = np.concatenate(xs)
    tau = float(np.mean([tau_int(x) for x in xs]))
    n = len(pooled)
    sd = float(pooled.std(ddof=1))
    return dict(mean=float(pooled.mean()), sd=sd, tau=tau, ess=n / tau,
                mcse=sd * np.sqrt(tau / n), rhat=float(split_rhat(xs)), n_states=n)


def beta_hat(betas, means, obs):
    """Linear interpolation of the first crossing E_beta[R] = obs."""
    b, v = np.asarray(betas, float), np.asarray(means, float)
    for i in range(len(b) - 1):
        lo, hi = sorted((v[i], v[i + 1]))
        if lo <= obs <= hi and v[i + 1] != v[i]:
            return b[i] + (obs - v[i]) / (v[i + 1] - v[i]) * (b[i + 1] - b[i])
    return np.nan


def style(ax):
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax.tick_params(labelsize=10.5, colors=MUTED, length=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "final"))
    p.add_argument("--set", choices=list(SETS), default="multi")
    p.add_argument("--burn", type=float, default=0.25)
    p.add_argument("--ess-min", type=float, default=400,
                   help="Vehtari et al. (2021) minimum pooled ESS for Phi")
    p.add_argument("--rhat-max", type=float, default=1.05)
    p.add_argument("--beta-max", type=float, default=np.inf,
                   help="drop beta points above this value from the figure")
    a = p.parse_args()
    root = os.path.abspath(a.out)
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    S = SETS[a.set]
    measures = S["measures"]
    cols = [m for m, *_ in measures]
    prim = PRIMARY[a.set]

    rows = []
    for ds, _ in DATASETS:
        for beta, cdir in sorted(combo_dirs(root, ds, S.get("dedup", False)).items()):
            loaded, why = load_combo(cdir, manifest, root, a.burn)
            if loaded is None:
                print("  SKIP", why)
                continue
            per, obs = loaded
            phi = summarize(per, "Phi")
            acc = np.mean([json.load(open(os.path.join(os.path.dirname(tr), "meta.json")))["accept_rate"]
                           for tr in sorted(glob.glob(os.path.join(cdir, "chain*", "trace.csv")))])
            base = dict(dataset=ds, beta=beta, n_chains=len(per), n_states=phi["n_states"],
                        accept_rate=acc, Phi_mean=phi["mean"], Phi_tau=phi["tau"],
                        Phi_ess=phi["ess"], Phi_rhat=phi["rhat"])
            for c in cols:
                s = summarize(per, c)
                rows.append({**base, "measure": c, "observed": obs[c], **s})
    T = pd.DataFrame(rows).sort_values(["dataset", "beta", "measure"])
    rh = T.groupby(["dataset", "beta"]).rhat.max().rename("rhat_max_measures")
    T = T.merge(rh, on=["dataset", "beta"])
    T["mixed"] = (T.Phi_ess >= a.ess_min) & (T.rhat_max_measures <= a.rhat_max)
    tab = os.path.join(root, f"beta_sweep_summary{S['tag']}.csv")
    T.to_csv(tab, index=False, float_format="%.6g")
    print("[saved]", tab)

    # ---- figure
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.6 * len(DATASETS), 3.7))
    fig.patch.set_facecolor(SURF)
    F = T[T.beta <= a.beta_max]
    for ax, (ds, name) in zip(axes, DATASETS):
        style(ax)
        d = F[F.dataset == ds]
        if d.empty:
            ax.set_title(name + " (no complete runs)", color=INK); continue
        for c, lab, col, mk in measures:
            g = d[d.measure == c].sort_values("beta")
            ax.plot(g.beta, g["mean"], color=col, lw=1.8, zorder=2)
            ax.errorbar(g.beta, g["mean"], yerr=2 * g.mcse, fmt="none", ecolor=col,
                        elinewidth=1, capsize=2, zorder=2)
            ok = g.mixed.to_numpy()
            ax.plot(g.beta[ok], g["mean"][ok], mk, ms=6, color=col, mec=col, zorder=3)
            ax.plot(g.beta[~ok], g["mean"][~ok], mk, ms=6, mfc=SURF, mec=col, mew=1.4, zorder=3)
            ax.axhline(g.observed.iloc[0], color=col, lw=1, ls="--", alpha=0.65, zorder=1)
        g = d[d.measure == prim].sort_values("beta")
        bh = beta_hat(g.beta, g["mean"], g.observed.iloc[0])
        if np.isfinite(bh):
            pc = dict((m, col) for m, _, col, _ in measures)[prim]
            ax.axvline(bh, color=pc, lw=1, ls=":", alpha=0.9, zorder=1)
            ax.annotate(rf"$\hat\beta$={bh:.1f}", (bh, ax.get_ylim()[0]), xytext=(4, 4),
                        textcoords="offset points", fontsize=10, color=pc, va="bottom")
        ax.set_title(name, fontsize=12, color=INK)
        ax.set_xlabel(r"$\beta$", fontsize=12, color=BODY)
        ax.set_xticks(sorted(F.beta.unique()))
    axes[0].set_ylabel(r"$\mathbb{E}_\beta[R]$" +
                       ("  (support, $M$-weighted)" if a.set != "multi" else ""),
                       fontsize=11.5, color=BODY)
    handles = [Line2D([], [], color=col, marker=mk, lw=1.8, ms=6, label=lab)
               for _, lab, col, mk in measures]
    handles += [Line2D([], [], color=MUTED, ls="--", lw=1, label="observed"),
                Line2D([], [], color=MUTED, marker="o", mfc=SURF, lw=0, ms=6, mew=1.4,
                       label=rf"ESS($\Phi$)<{a.ess_min:g} or $\widehat R$>{a.rhat_max}")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=10.5, labelcolor=BODY, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        out = os.path.join(root, f"fig_beta_sweep{S['tag']}.{ext}")
        fig.savefig(out, facecolor=SURF, dpi=180)
        print("[saved]", out)

    # ---- console digest
    piv = T[T.measure == prim].pivot(index="beta", columns="dataset", values="mean")
    print(f"\nE_beta[{prim}] (filled = mixed by ESS>={a.ess_min:g}, Rhat<={a.rhat_max}):")
    print(piv.round(4).to_string())
    for ds, name in DATASETS:
        g = T[(T.dataset == ds) & (T.measure == prim)].sort_values("beta")
        if len(g):
            print(f"  {name:15s} beta-hat({prim}) = {beta_hat(g.beta, g['mean'], g.observed.iloc[0]):.2f}"
                  f"   unmixed betas: {sorted(g.beta[~g.mixed].tolist())}")


if __name__ == "__main__":
    main()

# How this was run (2026-10-13):
#   <venv-python> make_beta_figure.py --set multi            # main figure
#   <venv-python> make_beta_figure.py --set support          # appendix (plan sec. 7)
#   <venv-python> make_beta_figure.py --set support_tilt     # phase T (support-family tilt)
#   add --beta-max 4 to drop the diagnostic-only beta=5 points from the plot
# Render-only: reads results/final/<ds>/beta*/chain*/{trace.csv,meta.json} and
# results/final/manifest.json; writes results/final/beta_sweep_summary[_support].csv
# and fig_beta_sweep[_support].{pdf,png}.
