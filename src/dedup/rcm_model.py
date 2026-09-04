"""rcm_model.py  --  Reciprocal Coalition Model (RCM): the Benson-style analytic
generative model for directed-hypergraph group reciprocity.

MODEL (fixed coalition size k):
  * N nodes, coalition size k, reciprocity parameter theta in [0,1].
  * We generate `n_coalitions` DISTINCT k-node coalitions C (random k-subsets).
  * Given a coalition C, each member v in C independently "leads" with prob theta,
    emitting the event (sender=v, recipients=C\\{v}) -- a full-team broadcast.
  * A coalition with >=1 leader appears in the data; one with 0 leaders is invisible.

So every event of C has the EXACT same node set C (an exact team recurring under
different leaders) -- which is why RCM is the natural model for the exact-team
measures. NAMING: grant convention -- r2 = pairwise_group; r2_weighted = the
k-weighted "partial participation" (rl.partial_participation).

KEY RANDOM VARIABLE: p_e = # members of C that led ~ Binomial(k, theta).

CLOSED FORMS (ratio-of-expectations; exact as n_coalitions -> infinity):
  * E[r2] = theta                                   (= pairwise_group; independent of
                                                      k; r2 DIRECTLY estimates theta)
  * E[r1] = theta**(k-1)                             (full participation) = E[s2]
  * E[r2_weighted] = (k/(k-1)) * (1 - (1-(1-theta)**k)/(k*theta))   (k-weighted partial)
  * E[s1] = 1 - (1-theta)**(k-1)                     (>=1 recipient reconvenes the
                                                      exact team; strict to_team OR)
All are monotone increasing in theta and span [0,1] as theta: 0 -> 1.

NOTE: RCM does NOT hold the directed projection fixed -- graph reciprocity also
rises with theta (reported empirically below). Pinning graph reciprocity while
varying group reciprocity is the job of the projection-fixed tilt model (Model B),
not RCM. RCM's job is analytic interpretability + a ground-truth testbed.

Usage:
    python rcm_model.py                 # runs the validation sweep + prints tables
    from rcm_model import simulate_rcm, theta_sweep, analytic_r1, ...
"""

from __future__ import annotations

from collections import defaultdict
from math import comb

import numpy as np
import pandas as pd

import reciprocity_dedup as rl


# ------------------------------------------------------------------ simulator
def simulate_rcm(N, k, n_coalitions, theta, rng):
    """Return a list of (sender, [recipients]) events from one RCM draw.
    Coalitions are sampled WITHOUT replacement (distinct node sets), so each is a
    genuine exact team with an independent Binomial(k, theta) leader count."""
    if k < 2:
        raise ValueError("k must be >= 2")
    n_coalitions = min(int(n_coalitions), comb(N, k))
    coalitions = set()
    guard = 0
    while len(coalitions) < n_coalitions:
        c = frozenset(int(x) for x in rng.choice(N, size=k, replace=False))
        coalitions.add(c)
        guard += 1
        if guard > 50 * n_coalitions + 1000:      # safety for tiny N (near-full space)
            break
    events = []
    for C in coalitions:
        members = list(C)
        leaders = [v for v in members if rng.random() < theta]
        for v in leaders:
            R = [u for u in members if u != v]
            events.append((v, R))
    return events


# ------------------------------------------------------------------ closed forms
# Grant naming: r2 = pairwise_group (E=theta); r2_weighted = k-weighted partial.
def analytic_r2(k, theta):
    """E[r2] = E[pairwise_group] = theta (grant r2; independent of k)."""
    return float(theta)


def analytic_r1(k, theta):
    return float(theta ** (k - 1))


def analytic_r2_weighted(k, theta):
    """E[r2_weighted] = E[partial_participation] = (k/(k-1))(1-(1-(1-theta)^k)/(k*theta))."""
    if theta <= 0.0:
        return 0.0                                  # limit as theta->0 is 0
    q = 1.0 - (1.0 - theta) ** k                     # P(p_e >= 1)
    return float((k / (k - 1)) * (1.0 - q / (k * theta)))


def analytic_s1(k, theta):
    """E[s1] = P(>=1 of the other k-1 team members reconvenes the exact team) =
    1 - (1-theta)**(k-1). (strict to_team s1; needs k>=3 -- group emails only.)"""
    return float(1.0 - (1.0 - theta) ** (k - 1))


def mbar_fixed_k(N, k, n_coalitions):
    """Expected number of coalitions per node-pair (density) for fixed-k RCM."""
    return n_coalitions * k * (k - 1) / (N * (N - 1))


def analytic_graph_recip(theta, mbar):
    """Expected graph (dyadic) reciprocity of RCM. A node pair co-occurs in
    m ~ Poisson(mbar) coalitions; each direction exists w.p. pi(m)=1-(1-theta)^m,
    INDEPENDENTLY, so  r_g = E[pi(m)^2] / E[pi(m)]  with, using E[(1-theta)^m]=e^{-mbar*theta},
        E[pi]  = 1 - e^{-mbar*theta}
        E[pi^2]= 1 - 2 e^{-mbar*theta} + e^{-mbar*(2*theta - theta^2)}.
    Rises with BOTH theta and density mbar, and saturates to 1 when dense (the naive
    1-(1-theta)^mbar is the constant-m approximation -- only accurate when dense).
    NOTE: unlike the group measures, this needs mbar (N, n_coalitions) -- it is a
    projection quantity, and RCM's projection is a RANDOM output, not fixed."""
    e1 = np.exp(-mbar * theta)
    e2 = np.exp(-mbar * (2 * theta - theta * theta))
    Epi = 1.0 - e1
    Epi2 = 1.0 - 2.0 * e1 + e2
    return float(Epi2 / Epi) if Epi > 0 else 0.0


# ------------------------------------------------------------------ measurement
def rcm_measures(events, maxk=10 ** 9):
    """Empirical s1/r1/r2/pairwise/graph_recip on an RCM event list via reciprocity_lib_faster."""
    s1 = rl.compute_group_reciprocity_measures(
        rl._events_to_df(events), dedup=True, mode="to_team")["s1_any"]
    return {
        "s1": s1,
        "r1": rl.full_participation(events)[1],
        "r2_weighted": rl.partial_participation(events)[1],
        "r2": rl.grant_r2(events)[1],                        # grant r2 (partial participation, base = sum p_e)
        "graph_recip": rl.graph_reciprocity(events, maxk=maxk),
        "n_events": len(events),
        "n_teams": len({frozenset([s] + R) for s, R in events}),
    }


# ------------------------------------------------------------------ sweeps
def theta_sweep(N, k, n_coalitions, thetas, reps=5, seed=0):
    """Empirical (mean over `reps` draws) vs analytic measures across a theta grid."""
    mbar = mbar_fixed_k(N, k, n_coalitions)
    rows = []
    for ti, theta in enumerate(thetas):
        emp = defaultdict(list)
        for r in range(reps):
            rng = np.random.default_rng(seed + 9973 * r + 101 * ti)
            m = rcm_measures(simulate_rcm(N, k, n_coalitions, theta, rng))
            for key, val in m.items():
                emp[key].append(val)
        rows.append({
            "k": k, "theta": round(float(theta), 3),
            "s1_emp": np.mean(emp["s1"]),          "s1_analytic": analytic_s1(k, theta),
            "r1_emp": np.mean(emp["r1"]),          "r1_analytic": analytic_r1(k, theta),
            # grant naming: "r2" = pairwise_group (= theta), "r2_weighted" = k-weighted partial
            "r2_emp": np.mean(emp["r2"]),          "r2_analytic": analytic_r2(k, theta),
            "r2_weighted_emp": np.mean(emp["r2_weighted"]), "r2_weighted_analytic": analytic_r2_weighted(k, theta),
            "graph_recip_emp": np.mean(emp["graph_recip"]),
            "graph_recip_analytic": analytic_graph_recip(theta, mbar),
            "r1_emp_std": np.std(emp["r1"]),
            "n_teams": int(np.mean(emp["n_teams"])),
        })
    return pd.DataFrame(rows)


def max_abs_error(df):
    """Largest |empirical - analytic| across the measures in a sweep frame."""
    e = np.concatenate([
        (df.s1_emp - df.s1_analytic).abs().values,
        (df.r1_emp - df.r1_analytic).abs().values,
        (df.r2_emp - df.r2_analytic).abs().values,
        (df.r2_weighted_emp - df.r2_weighted_analytic).abs().values,
        (df.graph_recip_emp - df.graph_recip_analytic).abs().values,
    ])
    return float(np.nanmax(e))


# ------------------------------------------------------------------ plotting
def plot_theta_sweep(N, k_values, n_coalitions, thetas, reps=5, seed=0,
                     outpath="figures/rcm_validation.png"):
    """Empirical (markers) vs analytic (smooth lines) r1/r2/pairwise across theta, one
    color per k, plus a graph-reciprocity panel showing RCM does NOT fix the projection.
    Saves a PNG (and returns the per-k sweep DataFrames). Mirrors Benson's Fig. 4:
    one knob (theta) sweeps the measure across its full [0,1] range."""
    import os
    import matplotlib
    matplotlib.use("Agg")                            # no display; save to file
    import matplotlib.pyplot as plt

    fine = np.linspace(0.0, 1.0, 201)                # smooth grid for analytic lines
    cmap = plt.get_cmap("viridis")
    colors = {k: cmap(i / max(1, len(k_values) - 1)) for i, k in enumerate(k_values)}

    sweeps = {k: theta_sweep(N, k, n_coalitions, thetas, reps=reps, seed=seed)
              for k in k_values}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("s1",       "s1_emp",   lambda k: [analytic_s1(k, t) for t in fine],
         r"$s_1$ (any reconvenes)  $=1-(1-\theta)^{k-1}$"),
        ("r1",       "r1_emp",   lambda k: [analytic_r1(k, t) for t in fine],
         r"$r_1$ (full participation)  $=\theta^{\,k-1}$"),
        ("r2",       "r2_emp",   lambda k: [analytic_r2(k, t) for t in fine],
         r"$r_2$ (= pairwise_group)  $=\theta$"),
        ("r2_weighted", "r2_weighted_emp", lambda k: [analytic_r2_weighted(k, t) for t in fine],
         r"$r_2^{\mathrm{weighted}}$ (partial participation)"),
    ]
    for ax, (_, empcol, afun, title) in zip(axes.flat[:4], panels):
        for k in k_values:
            df = sweeps[k]
            ax.plot(fine, afun(k), "-", color=colors[k], lw=1.8, alpha=0.9,
                    label=f"analytic k={k}")
            ax.plot(df.theta, df[empcol], "o", color=colors[k], ms=5,
                    mec="black", mew=0.4, label=f"sim k={k}")
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=1, alpha=0.6)   # y=x guide
        ax.set_xlabel(r"reciprocity parameter $\theta$")
        ax.set_ylabel("measure value")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7, ncol=2, loc="upper left")

    # 5th panel: graph reciprocity -- empirical (markers) vs analytic (lines).
    # RISES with theta (RCM does NOT fix the projection); analytic = E[pi^2]/E[pi].
    axg = axes.flat[4]
    for k in k_values:
        df = sweeps[k]
        axg.plot(fine, [analytic_graph_recip(t, mbar_fixed_k(N, k, n_coalitions)) for t in fine],
                 "-", color=colors[k], lw=1.6, alpha=0.9)
        axg.plot(df.theta, df.graph_recip_emp, "s", color=colors[k], ms=5,
                 mec="black", mew=0.4, label=f"k={k}")
    axg.set_xlabel(r"reciprocity parameter $\theta$")
    axg.set_ylabel("graph (dyadic) reciprocity")
    axg.set_title("graph reciprocity RISES with " + r"$\theta$" +
                  "\n(RCM does NOT fix the projection; analytic $E[\\pi^2]/E[\\pi]$)", fontsize=10)
    axg.set_xlim(0, 1); axg.set_ylim(-0.02, 1.02)
    axg.grid(alpha=0.25); axg.legend(fontsize=8)
    axes.flat[5].axis("off")                          # 6th cell unused

    fig.suptitle(f"RCM: empirical (markers) vs closed-form (lines)   "
                 f"[N={N}, n_coalitions={n_coalitions}, reps={reps}]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[saved] {os.path.abspath(outpath)}")
    return sweeps


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)

    N = 500
    NC = 30000            # number of distinct coalitions per draw
    REPS = 5
    THETAS = np.round(np.arange(0.0, 1.0001, 0.1), 3)

    for k in (3, 4, 5, 6):
        print(f"\n================= RCM validation: N={N}, k={k}, "
              f"n_coalitions={NC}, reps={REPS} =================")
        df = theta_sweep(N, k, NC, THETAS, reps=REPS, seed=0)
        show = df[["theta", "n_teams",
                   "s1_emp", "s1_analytic", "r1_emp", "r1_analytic",
                   "r2_emp", "r2_analytic", "r2_weighted_emp", "r2_weighted_analytic",
                   "graph_recip_emp", "graph_recip_analytic"]].round(4)
        print(show.to_string(index=False))
        print(f"  >> max |empirical - analytic| over s1,r1,r2,r2_weighted,graph_recip: {max_abs_error(df):.4f}")

    print("\nNote: graph_recip_emp RISES with theta -> RCM does NOT fix the projection "
          "(that is Model B's job). Here it just confirms group AND graph reciprocity\n"
          "are distinct quantities that need not move together.")

    print("\n[plotting] rendering validation figure ...")
    plot_theta_sweep(N, (3, 4, 5, 6), NC, THETAS, reps=REPS, seed=0,
                     outpath="figures/rcm_validation_500.png")
