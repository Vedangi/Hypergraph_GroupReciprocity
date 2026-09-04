"""model_b_tilt.py  --  Projection-fixed exponential-tilt generative model (Model B):
the thesis-carrying generator. Dial theta -> group reciprocity moves, while the
directed projection (hence graph/dyadic reciprocity) stays EXACTLY fixed.

IDEA
  * Fix a directed projection pi (each sender's recipient multiset). Every hypergraph
    we consider is a regrouping of the SAME pi -> graph reciprocity is constant.
  * Put a Gibbs / exponential-family distribution over these regroupings:
        P_theta(H)  ∝  exp( theta * Phi(H) )
    with reciprocity sufficient statistic
        Phi(H) = sum over exact teams N of p_e(N) * (p_e(N) - 1)      (r2/pairwise numerator)
    (p_e(N) = # DISTINCT senders whose event has node set exactly N.)
  * theta = 0  -> uniform over same-projection regroupings = the projection-preserving
    NULL. theta > 0 favors reciprocal groupings; theta < 0 anti-reciprocal.

SAMPLING (Metropolis on 2-switches -- generalizes shuffle_pairwise_by_sender):
  propose a within-sender 2-switch (swap a recipient between two of a sender's events;
  preserves that sender's recipient multiset = pi), accept with prob min(1, exp(theta*dPhi)).
  The unknown normalizer Z cancels (only the ratio exp(theta*dPhi) is used). theta=0
  -> always accept -> the uniform null shuffle.

GUARANTEES
  * graph reciprocity is CONSTANT for all theta (every state shares pi).
  * E_theta[Phi] is monotone non-decreasing in theta  (d/dtheta E[Phi] = Var(Phi) >= 0).

Phi is maintained INCREMENTALLY: a 2-switch touches only 4 teams, so dPhi is O(1).

Usage:
    python model_b_tilt.py                      # demo sweep + figure
    from model_b_tilt import tilt_sample, theta_sweep_B
"""

from __future__ import annotations

import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import reciprocity_dedup as rl
from rcm_model import simulate_rcm


# ------------------------------------------------------------------ state
# ------------------------------------------------------------------ statistic forms
def _team_term(pe, form):
    """Raw (unweighted) per-team contribution to Phi as a function of p_e.
    form='quad' -> p_e(p_e-1)   (dyadic pair count; the original statistic)
    form='lin'  -> max(p_e-1, 0)   (contribution count; numerator of r2 / R_some)
    Both give 0 when p_e <= 1 (a lone sender is not reciprocal / adds no extra
    contributor), so single-sender teams never contribute."""
    if form == "quad":
        return pe * (pe - 1)
    elif form == "lin":
        return pe - 1 if pe >= 1 else 0      # p_e>=1 always for a realized team
    else:
        raise ValueError("form must be 'quad' or 'lin'")

def _parse_stat(stat):
    """Map the public `stat` string to (form, norm).
    'pairs'   -> quadratic, raw weight        (original dyadic statistic)
    'norm'    -> quadratic, (k-1)-normalized  (dyadic, k-normalized)
    'contrib' -> linear,    (k-1)-normalized  (numerator of r2 -> tilts r2 itself)"""
    table = {
        "pairs":   ("quad", False),
        "norm":    ("quad", True),
        "contrib": ("lin",  True),
    }
    if stat not in table:
        raise ValueError(f"stat must be one of {sorted(table)}")
    return table[stat]
   
def _team_w(team, norm):
    """Weight of one team in Phi. norm=False -> 1 (raw pair count). norm=True ->
    1/(k-1) with k = |team|: the (k-1)-NORMALIZED statistic
        Phi'(H) = sum_N p_e(N)(p_e(N)-1)/(|N|-1),
    i.e. exactly the numerator of grant_r2 / R_some, so the tilt is sufficient for
    the reported measure and large-k teams no longer dominate the raw pair count.
    (A 2-switch swaps one recipient for another, so every event keeps its size and
    the weight of each touched team is well-defined and constant.)"""
    return 1.0 / (len(team) - 1) if norm else 1.0


def _build_state(events, norm=False, form = 'quad'):
    """sender_events[s] = list of recipient-sets; tsc[N][s] = # of s's events with node
    set N; Phi = sum_N w(N) p_e(N)(p_e(N)-1) with w from _team_w."""
    sender_events = defaultdict(list)
    tsc = defaultdict(dict)                       # frozenset(nodeset) -> {sender: count}
    for s, R in events:
        recs = set(int(x) for x in R)
        s = int(s)
        sender_events[s].append(recs)
        N = frozenset(recs | {s})
        tsc[N][s] = tsc[N].get(s, 0) + 1
    # Phi = sum(_team_w(N, norm) * len(d) * (len(d) - 1) for N, d in tsc.items())
    Phi = sum(_team_w(N, norm) * _team_term(len(d), form) for N, d in tsc.items())

    return sender_events, tsc, Phi


def _chg(tsc, team, s, delta, norm=False, form='quad'):
    """Change sender s's membership count in `team` by delta (+1/-1); return the change
    in Phi (= change in w(N) p_e(p_e-1) for that team). Maintains tsc."""
    d = tsc[team]                                 # defaultdict creates {} if new
    pe_old = len(d)
    c = d.get(s, 0) + delta
    if c <= 0:
        d.pop(s, None)
    else:
        d[s] = c
    pe_new = len(d)
    if not d:
        del tsc[team]
    # return _team_w(team, norm) * (pe_new * (pe_new - 1) - pe_old * (pe_old - 1))
    return _team_w(team, norm) * (_team_term(pe_new, form) - _team_term(pe_old, form))


def _apply_switch(sender_events, tsc, s, i, j, a, b, norm=False, form='quad'):
    """Swap recipient a (in event i) with b (in event j) for sender s; update tsc/Phi.
    Returns dPhi. Its own inverse is _apply_switch(..., s, i, j, b, a)."""
    Ei = sender_events[s][i]; Ej = sender_events[s][j]
    Ni_old = frozenset(Ei | {s}); Nj_old = frozenset(Ej | {s})
    Ei.discard(a); Ei.add(b)
    Ej.discard(b); Ej.add(a)
    Ni_new = frozenset(Ei | {s}); Nj_new = frozenset(Ej | {s})
    # dPhi = (_chg(tsc, Ni_old, s, -1, norm) + _chg(tsc, Ni_new, s, +1, norm)
    #         + _chg(tsc, Nj_old, s, -1, norm) + _chg(tsc, Nj_new, s, +1, norm))
    dPhi = (_chg(tsc, Ni_old, s, -1, norm, form) + _chg(tsc, Ni_new, s, +1, norm, form) 
            + _chg(tsc, Nj_old, s, -1, norm, form) + _chg(tsc, Nj_new, s, +1, norm, form))
    return dPhi


# ------------------------------------------------------------------ sampler
def tilt_sample(base_events, theta, n_steps=None, sweep_multiplier=50, seed=0,
                return_trace=False, stat="pairs",form='quad'):
    """Metropolis sample from P_theta starting at base_events. Returns (events, Phi).
    n_steps defaults to sweep_multiplier * (total recipient tokens).
    stat: 'pairs' -> Phi = sum p(p-1)   (raw pair count, the original statistic)
          'norm'  -> Phi = sum p(p-1)/(k-1)   (grant_r2 / R_some numerator).
          'contrib -> Phi = sum (p-1)/(k-1) grant-r2 itself """
    
    if stat not in ("pairs", "norm", "contrib"):
        raise ValueError("stat must be 'pairs', 'norm', or 'contrib'")
    # norm = stat == "norm"
    form, norm = _parse_stat(stat)
    rng = np.random.default_rng(seed)
    # sender_events, tsc, Phi = _build_state(base_events, norm=norm)
    sender_events, tsc, Phi = _build_state(base_events, norm=norm, form=form)
    # senders that can actually be switched (>=2 events, some size>0)
    swap_senders = [s for s, evs in sender_events.items()
                    if len(evs) >= 2 and sum(len(e) for e in evs) > 0]
    if not swap_senders:
        ev = [(s, sorted(e)) for s, evs in sender_events.items() for e in evs]
        return (ev, Phi) if not return_trace else (ev, Phi, [Phi])
    tokens = sum(len(e) for evs in sender_events.values() for e in evs)
    if n_steps is None:
        n_steps = sweep_multiplier * tokens
    trace = []
    accepts = 0
    for step in range(n_steps):
        s = swap_senders[rng.integers(len(swap_senders))]
        evs = sender_events[s]
        m = len(evs)
        i = int(rng.integers(m)); j = int(rng.integers(m))
        if i == j:
            continue
        Ei, Ej = evs[i], evs[j]
        A = list(Ei - Ej); B = list(Ej - Ei)
        if not A or not B:
            continue
        a = A[rng.integers(len(A))]; b = B[rng.integers(len(B))]
        # dPhi = _apply_switch(sender_events, tsc, s, i, j, a, b, norm)
        dPhi = _apply_switch(sender_events, tsc, s, i, j, a, b, norm, form) 
        # Metropolis: accept with prob min(1, e^{theta*dPhi}). The shortcut must be
        # on theta*dPhi >= 0, not dPhi >= 0 -- the latter always accepted uphill-Phi
        # moves even when theta < 0, biasing negative-theta samples upward.
        if theta * dPhi >= 0 or rng.random() < math.exp(theta * dPhi):
            Phi += dPhi; accepts += 1
        else:
            # _apply_switch(sender_events, tsc, s, i, j, b, a, norm)   # revert
            _apply_switch(sender_events, tsc, s, i, j, b, a, norm, form)
        if return_trace and (step % max(1, n_steps // 200) == 0):
            trace.append(Phi)
    events = [(s, sorted(e)) for s, evs in sender_events.items() for e in evs]
    if return_trace:
        return events, Phi, trace
    return events, Phi


# ------------------------------------------------------------------ helpers
def _proj_sig(events):
    w = defaultdict(int)
    for s, R in events:
        for r in R:
            w[(s, r)] += 1
    return tuple(sorted(w.items()))


def _measures(events, maxk=10 ** 9):
    s1 = rl.compute_group_reciprocity_measures(
        rl._events_to_df(events), dedup=True, mode="to_team")["s1_any"]
    return {
        "s1": s1,                                             # strict to_team, any reconvenes
        "r1": rl.full_participation(events)[1],
        "r2": rl.grant_r2(events)[1],                         # grant r2 (partial participation, base = sum p_e)
        "r2_weighted": rl.partial_participation(events)[1],   # k-weighted partial
        "graph_recip": rl.graph_reciprocity(events, maxk=maxk),
    }


# ------------------------------------------------------------------ sweep
def theta_sweep_B(base_events, thetas, sweep_multiplier=20, reps=3, seed=0,
                  stat="pairs"):
    """For each theta: sample (reps times), report mean group measures + graph
    reciprocity + a check that the projection is invariant vs the base.
    stat: 'pairs' (raw Phi) or 'norm' ((k-1)-normalized Phi) -- see tilt_sample."""
    base_sig = _proj_sig(base_events)
    base_graph = rl.graph_reciprocity(base_events, maxk=10 ** 9)
    rows = []
    for ti, theta in enumerate(thetas):
        acc = defaultdict(list); proj_ok = True
        for r in range(reps):
            ev, Phi = tilt_sample(base_events, theta, sweep_multiplier=sweep_multiplier,
                                  seed=seed + 131 * r + 7 * ti, stat=stat)
            proj_ok &= (_proj_sig(ev) == base_sig)
            m = _measures(ev)
            for key, val in m.items():
                acc[key].append(val)
            acc["Phi"].append(Phi)
        rows.append({
            "theta": round(float(theta), 3),
            "s1": np.mean(acc["s1"]),                    # strict to_team (any reconvenes)
            "r2": np.mean(acc["r2"]),                    # = pairwise_group
            "r1": np.mean(acc["r1"]),
            "r2_weighted": np.mean(acc["r2_weighted"]),  # partial participation
            "graph_recip": np.mean(acc["graph_recip"]),
            "graph_recip_base": base_graph,
            "Phi": np.mean(acc["Phi"]),
            "projection_fixed": proj_ok,
            "stat": stat,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plot
_MEAS_COLOR = {"s1": "#ff7f0e", "r2": "#1f77b4", "r1": "#2ca02c", "r2_weighted": "#9467bd"}


def plot_sweep_B(df, outpath="figures/model_b_tilt_linear.png", base_graph=None,
                 real_values=None, title=None):
    """Left: group measures vs theta. Right: graph reciprocity (fixed). If real_values
    (dict measure->value, e.g. from the real base) is given, draw dashed reference lines
    on the left panel and note where the real data sits on the family."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(df.theta, df.s1, "D-", label="s1 (any reconvenes)", color=_MEAS_COLOR["s1"])
    ax[0].plot(df.theta, df.r2, "o-", label="r2 (= pairwise_group)", color=_MEAS_COLOR["r2"])
    ax[0].plot(df.theta, df.r1, "s-", label="r1 (full particip.)", color=_MEAS_COLOR["r1"])
    ax[0].plot(df.theta, df.r2_weighted, "^-", label="r2_weighted (partial)", color=_MEAS_COLOR["r2_weighted"])
    if real_values:
        for meas, val in real_values.items():
            c = _MEAS_COLOR.get(meas, "gray")
            ax[0].axhline(val, ls="--", color=c, lw=1, alpha=0.7)
            ax[0].annotate(f"real {meas}={val:.2f}", xy=(df.theta.min(), val),
                           xytext=(df.theta.min(), val + 0.02), fontsize=7, color=c)
    ax[0].set_xlabel(r"tilt parameter $\theta$"); ax[0].set_ylabel("group reciprocity")
    ax[0].set_title("group reciprocity RISES with " + r"$\theta$"); ax[0].grid(alpha=0.25)
    ax[0].axvline(0, ls=":", color="gray", lw=1); ax[0].legend(fontsize=8)

    ax[1].plot(df.theta, df.graph_recip, "o-", color="#d62728", label="graph reciprocity (sampled)")
    if base_graph is not None:
        ax[1].axhline(base_graph, ls="--", color="black", lw=1, label="base projection value")
    ax[1].set_xlabel(r"tilt parameter $\theta$"); ax[1].set_ylabel("graph (dyadic) reciprocity")
    ax[1].set_title("graph reciprocity is FIXED (projection preserved)")
    ax[1].set_ylim(max(0, df.graph_recip.min() - 0.1), df.graph_recip.max() + 0.1)
    ax[1].grid(alpha=0.25); ax[1].legend(fontsize=8)

    fig.suptitle(title or "Model B: exponential tilt at fixed projection", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if os.path.dirname(outpath):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150); plt.close(fig)
    print(f"[saved] {os.path.abspath(outpath)}")


# --------------------------------------------- run Model B on a REAL email network
def real_base_from_triples(path, sizes=(3,), sep=None):
    """Build a Model-B base from pairwise temporal triples '(sender recipient ... time)'.
    Group emails are formed by grouping triples with the SAME (sender, time); the result
    is deduped (each distinct (sender, recipient-set) once) and restricted to team sizes
    in `sizes` (k = 1 + #recipients). Works for SNAP email-Eu and enron out.enron
    (first col = sender, second = recipient, LAST = time; middle cols ignored). Small
    sizes (k=3) are recommended -- the exact-team Phi is degenerate on large groups."""
    by_event = defaultdict(set)                     # (sender, time) -> recipients
    with open(path) as fh:
        for line in fh:
            parts = line.split() if sep is None else line.rstrip("\n").split(sep)
            if len(parts) < 3:
                continue
            s = int(parts[0]); r = int(parts[1]); t = int(parts[-1])
            if r != s:
                by_event[(s, t)].add(r)
    seen = set(); base = []
    for (s, _t), R in by_event.items():
        if (1 + len(R)) not in sizes:
            continue
        key = (s, frozenset(R))
        if key in seen:                              # dedup distinct directed hyperedges
            continue
        seen.add(key)
        base.append((s, sorted(R)))
    return base


def run_on_real_network(path, name="email-Eu", sizes=(3,), thetas=None,
                        sweep_multiplier=25, reps=3, seed=0, outpath=None, stat = 'contrib'):
    """Load a real email network (email-Eu / enron triples), take its deduped size-`sizes`
    group hypergraph as the fixed-projection base, sweep theta, and plot -- marking where
    the REAL data's r2/s1 sit on the tiltx family. Returns the sweep DataFrame."""
    base = real_base_from_triples(path, sizes=sizes)
    if not base:
        print(f"[{name}] no group events of sizes {sizes} at {path}")
        return None
    real = _measures(base)
    print(f"[{name}] base: {len(base)} events (sizes {sizes})  "
          f"graph_recip={real['graph_recip']:.4f}  real r2={real['r2']:.4f}  s1={real['s1']:.4f}")
    if thetas is None:
        # thetas = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        thetas = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 5.0, 7.0, 10.0, 15.0]
    df = theta_sweep_B(base, thetas, sweep_multiplier=sweep_multiplier, reps=reps, seed=seed, stat=stat)
    print(df[["theta", "s1", "r2", "r1", "r2_weighted",
              "graph_recip", "projection_fixed"]].round(4).to_string(index=False))
    if outpath is None:
        outpath = f"figures/model_b_real_{name}_linear.png"
    plot_sweep_B(df, outpath=outpath, base_graph=real["graph_recip"],
                 real_values={"r2": real["r2"], "s1": real["s1"]},
                 title=f"Model B on REAL {name} (sizes {sizes}): projection held fixed at "
                       f"{real['graph_recip']:.3f}")
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 160)

    # --- base hypergraph: fixes the projection (graph reciprocity) ----------
    # Denser base so the theta=0 NULL sits above 0 -> a smooth S-curve. (For a paper
    # figure, pass a REAL dataset's deduped events here instead: its projection has a
    # natural intermediate graph reciprocity, and the sweep dials group reciprocity
    # up/down around it while graph reciprocity stays pinned at the real value.)


    rng = np.random.default_rng(1)
    # base = simulate_rcm(N=400, k=4, n_coalitions=10000, theta=0.4, rng=rng)
    # print(f"base: {len(base)} events, "
    #       f"graph_recip={rl.graph_reciprocity(base, maxk=10**9):.4f}, "
    #       f"r2(=pairwise)={rl.pairwise_group(base)[1]:.4f}")

    # THETAS = [-1.0, -0.5, -0.2, 0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5]
    # print("\n[sampling] Metropolis tilt sweep (theta<0 below null, theta>0 above) ...")
    # df = theta_sweep_B(base, THETAS, sweep_multiplier=30, reps=3, seed=0)
    # print(df.round(4).to_string(index=False))

    # base_graph = rl.graph_reciprocity(base, maxk=10 ** 9)
    # print(f"\nprojection fixed for every theta: {bool(df.projection_fixed.all())}")
    # print(f"graph reciprocity spread across theta: "
    #       f"{df.graph_recip.max() - df.graph_recip.min():.2e}  (should be ~0)")
    # print(f"r2(=pairwise) range across theta: {df.r2.min():.3f} -> {df.r2.max():.3f}")

    # plot_sweep_B(df, outpath="figures/model_b_tilt.png", base_graph=base_graph)

    # --- OPTIONAL: run Model B on a REAL email network (email-Eu / enron) ----------
    # Uses the deduped size-3 group hypergraph as the fixed-projection base; the sweep
    # dials group reciprocity while graph reciprocity stays pinned at the REAL value,
    # and the plot marks where the real data sits on the family. (Paths are relative to
    # code/. Switch the file / uncomment enron as desired.)


    _DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    EMAIL_EU = os.path.join(_DATA, "emailEu", "email-Eu-core-temporal.txt")
    ENRON = os.path.join(_DATA, "enron", "out.enron")
    if os.path.exists(ENRON):
        print("\n[real] Model B on enron ...")
        run_on_real_network(ENRON, name="enron", sizes=(3,4,5,6,7,8,9,10), reps=2, seed=0)
    # if os.path.exists(EMAIL_EU):
    #     print("\n[real] Model B on email-Eu (size-3 to 10 group hypergraph) ...")
    #     run_on_real_network(EMAIL_EU, name="emailEu", sizes=(3,4,5,6,7,8,9,10), reps=3, seed=0)
    