#!/usr/bin/env python3
"""multi_tilt.py -- projection-fixed exponential tilt over single-sender
directed MULTI-hypergraphs (duplicates allowed; NOTHING is deduplicated).

STATE SPACE
    All regroupings of each sender's recipient tokens into events of the
    observed sizes (the full per-sender Ryser space).  Duplicate hyperedges are
    legal states, so no rejection step is needed and irreducibility follows
    from Ryser's interchange theorem (edge-labelled convention; cf. Fosdick et
    al. 2018 on stub- vs vertex-labelled spaces).

THE STATISTIC IS THE MEASURE
    Phi(H) = Sum_T (2/(k_T - 1)) Sum_{i<j in T} min(m_{T,i}, m_{T,j})
           = |E| * r2_multi(H),
    the exact numerator of the multiplicity-aware partial-participation score
    (multi_reciprocity.r2_multi).  |E| (total events) is invariant under every
    switch, so tilting Phi tilts r2_multi itself and
        E_theta[r2_multi] = E_theta[Phi] / |E|
    can be read straight off the chain -- no separate translation between the
    tilted statistic and the reported measure.

    stat="pairs" gives the smooth alternative Sum_T Sum_{i<j} m_i m_j (the
    multiplicity Holland-Leinhardt pair count; = p(p-1)/2 form on 0/1 states).

INCREMENTAL UPDATE
    A switch changes at most 4 (team, sender) cells by +/-1.  For the min-based
    Phi, changing m_{T,s}: c -> c+1 adds (2/(k-1)) * #{t != s active: m_t >= c+1},
    and c -> c-1 subtracts (2/(k-1)) * #{t != s active: m_t >= c}; each touched
    team costs O(#active senders of that team), typically 1-3.

ACCEPTANCE
    Plain Metropolis, accept with min(1, exp(theta * dPhi)) via the sign-safe
    shortcut theta*dPhi >= 0.  Failed/identity proposals consume a step
    (self-loop); |A|=|B|=1 swaps merely exchange two events and are skipped as
    no-ops on the unordered object.

Run as a script: sweeps theta on the RAW (non-deduplicated) email-Eu group
hypergraph and writes multi_tilt_emaileu.{pdf,png,csv} into this folder.
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import multi_reciprocity as MR


# ------------------------------------------------------------------ statistic
def _w(team):
    return 2.0 / (len(team) - 1)


def _phi_team(team, cnt, stat):
    m = list(cnt.values())
    if stat == "pairs":
        return float(sum(a * b for i, a in enumerate(m) for b in m[i + 1:]))
    return _w(team) * sum(min(a, b) for i, a in enumerate(m) for b in m[i + 1:])


def _dphi_change(tsc, team, s, delta, stat):
    """Apply m_{team,s} += delta (delta = +1/-1) to tsc and return dPhi."""
    d = tsc[team]
    c_old = d.get(s, 0)
    c_new = c_old + delta
    if stat == "pairs":
        others = sum(v for t, v in d.items() if t != s)
        dphi = float(delta * others)
    else:
        if delta == +1:
            dphi = _w(team) * sum(1 for t, v in d.items()
                                  if t != s and v >= c_new)
        else:
            dphi = -_w(team) * sum(1 for t, v in d.items()
                                   if t != s and v >= c_old)
    if c_new <= 0:
        d.pop(s, None)
        if not d:
            del tsc[team]
    else:
        d[s] = c_new
    return dphi


def _mult_change(d, key, delta):
    """Apply m_key += delta to the multiplicity dict d and return
    log( (new m_key)! / (old m_key)! )."""
    c = d.get(key, 0)
    if delta == +1:
        d[key] = c + 1
        return math.log(c + 1.0)
    if c <= 1:
        d.pop(key, None)
    else:
        d[key] = c - 1
    return -math.log(float(c))


def _logW(mlt):
    """log W(H) = Sum_{s,e} log m_{s,e}!  (W = 1 / #labelled representations,
    up to a state-independent constant)."""
    return sum(math.lgamma(c + 1.0) for d in mlt.values() for c in d.values())


def _build_state(events, stat):
    evs = defaultdict(list)
    tsc = defaultdict(dict)
    mlt = defaultdict(dict)          # sender -> {frozenset(recipients): mult}
    for s, R in events:
        s = int(s)
        recs = set(int(x) for x in R)
        evs[s].append(recs)
        T = frozenset(recs | {s})
        tsc[T][s] = tsc[T].get(s, 0) + 1
        key = frozenset(recs)
        mlt[s][key] = mlt[s].get(key, 0) + 1
    Phi = sum(_phi_team(T, d, stat) for T, d in tsc.items())
    return evs, tsc, Phi, mlt


def _apply_switch(evs, tsc, s, i, j, a, b, stat, mlt=None):
    """Swap recipient a (event i) <-> b (event j) for sender s.
    Returns (dPhi, dLogW).  dLogW is log W(new)/W(old) with
    W(H) = Prod_{s,e} m_{s,e}!; it is 0.0 when mlt is None (labelled base
    measure).  Inverse: _apply_switch(..., b, a, stat, mlt)."""
    Ei, Ej = evs[s][i], evs[s][j]
    Ti_old = frozenset(Ei | {s}); Tj_old = frozenset(Ej | {s})
    dlw = 0.0
    if mlt is not None:
        d = mlt[s]
        dlw += _mult_change(d, frozenset(Ei), -1)
        dlw += _mult_change(d, frozenset(Ej), -1)
    Ei.discard(a); Ei.add(b)
    Ej.discard(b); Ej.add(a)
    Ti_new = frozenset(Ei | {s}); Tj_new = frozenset(Ej | {s})
    if mlt is not None:
        d = mlt[s]
        dlw += _mult_change(d, frozenset(Ei), +1)
        dlw += _mult_change(d, frozenset(Ej), +1)
    dphi = (_dphi_change(tsc, Ti_old, s, -1, stat)
            + _dphi_change(tsc, Ti_new, s, +1, stat)
            + _dphi_change(tsc, Tj_old, s, -1, stat)
            + _dphi_change(tsc, Tj_new, s, +1, stat))
    return dphi, dlw


# ------------------------------------------------------------------- sampler
def tilt_sample(events, theta, n_steps=None, sweep_multiplier=20, seed=0,
                stat="r2", return_stats=False, base="labelled"):
    """Metropolis sample from P_theta(H) ~ exp(theta * Phi(H)).  Returns
    (events, Phi[, stats]).

    base="labelled"   : uniform (at theta=0) over EVENT-LABELLED regroupings,
                        i.e. over fillings of each sender's fixed-size event
                        slots.  An unlabelled multi-hypergraph H then carries
                        weight proportional to its number of labelled
                        representations, 1/Prod_{s,e} m_{s,e}!, so states with
                        repeated hyperedges are DOWN-weighted.
    base="unlabelled" : uniform over distinct multi-hypergraphs.  Obtained by
                        targeting exp(theta*Phi) * Prod m! on the labelled
                        space, i.e. the extra Metropolis factor
                        Prod m_new! / Prod m_old!  (the same factorial
                        reweighting DiNgHy uses for its edge-unordered space).
    """
    if stat not in ("r2", "pairs"):
        raise ValueError("stat must be 'r2' or 'pairs'")
    if base not in ("labelled", "unlabelled"):
        raise ValueError("base must be 'labelled' or 'unlabelled'")
    rng = np.random.default_rng(seed)
    evs, tsc, Phi, mlt = _build_state(events, stat)
    if base == "labelled":
        mlt = None
    swap = [s for s, e in evs.items() if len(e) >= 2]
    tokens = sum(len(x) for e in evs.values() for x in e)
    if n_steps is None:
        n_steps = sweep_multiplier * tokens
    n_acc = n_prop = n_id = 0
    if swap:
        for _ in range(n_steps):
            s = swap[rng.integers(len(swap))]
            lst = evs[s]
            m = len(lst)
            i = int(rng.integers(m)); j = int(rng.integers(m))
            if i == j:
                continue
            A = list(lst[i] - lst[j]); B = list(lst[j] - lst[i])
            if not A or not B:
                continue
            if len(A) == 1 and len(B) == 1:      # exchanges the two events: no-op
                n_id += 1
                continue
            a = A[rng.integers(len(A))]; b = B[rng.integers(len(B))]
            n_prop += 1
            dPhi, dLogW = _apply_switch(evs, tsc, s, i, j, a, b, stat, mlt)
            arg = theta * dPhi + dLogW
            if arg >= 0 or rng.random() < math.exp(arg):
                Phi += dPhi
                n_acc += 1
            else:
                _apply_switch(evs, tsc, s, i, j, b, a, stat, mlt)
    out = [(s, sorted(e)) for s, lst in evs.items() for e in lst]
    stats = {"n_steps": n_steps, "n_proposed": n_prop, "n_accepted": n_acc,
             "n_identity": n_id, "base": base,
             "accept_rate": n_acc / n_prop if n_prop else float("nan"),
             "move_rate": n_acc / n_steps if n_steps else float("nan")}
    return (out, Phi, stats) if return_stats else (out, Phi)


def proj_sig(events):
    w = defaultdict(int)
    for s, R in events:
        for r in R:
            w[(int(s), int(r))] += 1
    return tuple(sorted(w.items()))


# --------------------------------------------------------------------- sweep
def theta_sweep(events, thetas, reps=2, sweep_multiplier=20, n_steps=None,
                stat="r2", seed=0, verbose=True, base="labelled"):
    import pandas as pd
    base_sig = proj_sig(events)
    rows = []
    for ti, th in enumerate(thetas):
        acc = defaultdict(list)
        ok = True
        for r in range(reps):
            ev, Phi, st = tilt_sample(events, th, n_steps=n_steps,
                                      sweep_multiplier=sweep_multiplier,
                                      seed=seed + 997 * r + 13 * ti,
                                      stat=stat, return_stats=True, base=base)
            ok &= (proj_sig(ev) == base_sig)
            m = MR.measures(ev, min_k=3)
            for key in ("r2_multi", "s1_multi", "r1_multi", "s1_support",
                        "graph_binary", "graph_weighted"):
                acc[key].append(m[key])
            acc["Phi"].append(Phi)
            acc["accept"].append(st["accept_rate"])
            acc["move_rate"].append(st["move_rate"])
            if verbose:
                print(f"  theta={th:+.2f} rep={r+1}/{reps} Phi={Phi:,.1f} "
                      f"r2_multi={m['r2_multi']:.4f}", flush=True)
        rows.append({"theta": th, "stat": stat, "base": base,
                     **{k: float(np.mean(v)) for k, v in acc.items()},
                     "projection_fixed": bool(ok)})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- driver
def _load_raw_emaileu(kmax=25):
    """RAW email-Eu group stream: duplicates KEPT, k>=3, size cap."""
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
    import datasets as lpf
    ev = []
    for s, R, _t in lpf.load_dataset("emaileu"):
        R = sorted(set(int(x) for x in R) - {int(s)})
        if len(R) >= 2 and 1 + len(R) <= kmax:
            ev.append((int(s), R))
    return ev


def main():
    import pandas as pd
    ev = _load_raw_emaileu()
    n_distinct = len({(s, frozenset(R)) for s, R in ev})
    real = MR.measures(ev, min_k=3)
    print(f"RAW email-Eu: {len(ev):,} events ({n_distinct:,} distinct; "
          f"{len(ev) - n_distinct:,} duplicates), {real['n_teams']:,} teams")
    print("real:  " + "  ".join(f"{k}={real[k]:.4f}" for k in
          ("r2_multi", "s1_multi", "r1_multi", "s1_support",
           "graph_binary", "graph_weighted")))

    thetas = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    df = theta_sweep(ev, thetas, reps=2, sweep_multiplier=20)
    df.to_csv(os.path.join(HERE, "multi_tilt_emaileu.csv"), index=False)
    print(df.round(4).to_string(index=False))
    assert bool(df.projection_fixed.all()), "projection drifted!"

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    INK, BODY, MUTED, GRID, SURF = ("#1a1a19", "#3d3d3a", "#6e6d68",
                                    "#ececE8", "#ffffff")
    C = {"r2_multi": "#2a78d6", "s1_multi": "#eb6834", "r1_multi": "#1baf7a",
         "s1_support": "#4a3aa7"}
    MK = {"r2_multi": "o", "s1_multi": "s", "r1_multi": "^", "s1_support": "D"}
    LB = {"r2_multi": r"$R_{\mathrm{some}}^{\mathrm{multi}}$ (tilted)",
          "s1_multi": r"$R_{\mathrm{any}}^{\mathrm{multi}}$",
          "r1_multi": r"$R_{\mathrm{all}}^{\mathrm{multi}}$",
          "s1_support": r"$R_{\mathrm{any}}$ (support)"}

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
               color=MUTED, label="weighted (Squartini)", markeredgecolor=SURF)
    for v in (real["graph_binary"], real["graph_weighted"]):
        ax[1].axhline(v, ls="--", lw=1, color=INK, alpha=0.5)
    pad = 0.08
    ax[1].set_ylim(min(real["graph_weighted"], real["graph_binary"]) - pad,
                   max(real["graph_weighted"], real["graph_binary"]) + pad)
    ax[1].axvline(0, ls=":", lw=1, color=MUTED)
    ax[1].set_xlabel(r"tilt $\theta$", color=BODY)
    ax[1].set_title("graph reciprocity (invariant)", fontsize=10.5, color=INK)
    ax[1].legend(fontsize=8, frameon=False, labelcolor=BODY)
    for a in ax:
        a.set_facecolor(SURF)
        a.grid(False)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(labelsize=8.5, colors=MUTED, length=0)
    fig.suptitle("Multi-hypergraph tilt on RAW email-Eu "
                 "(duplicates kept, 3 ≤ k ≤ 25)",
                 fontsize=12, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"multi_tilt_emaileu.{ext}"),
                    facecolor=SURF, dpi=200)
    print("saved multi_tilt_emaileu.{pdf,png,csv}")


if __name__ == "__main__":
    main()
