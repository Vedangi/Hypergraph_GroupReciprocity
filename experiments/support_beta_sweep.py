#!/usr/bin/env python3
r"""beta-sweep of the SUPPORT-measure exponential family on the deduplicated
hypergraph:  P_beta(H) proportional to exp(beta * Phi(H))  with

    Phi(H) = Sum_T M_T (p_T - 1)/(k_T - 1)   = |E| * R_part^M(H)

(the M-weighted partial-team-contribution numerator; |E| fixed by the
sampler, so beta tilts R_part^M itself).  Within-sender switches on the
labelled space; duplicates created by the chain are handled natively by the
M-weighted definitions.  For each beta: chains from the dedup-observed state
and from randomized Gale-Ryser fills; Phi traced per sweep (incremental);
R_any^M / R_all^M evaluated at each chain's final state.

ABSTRACT-GRADE BUDGET (fast): N_SWEEPS=100, 4 chains/beta, BURN=0.5.
For the main paper rerun with N_SWEEPS>=300 and more chains.
"""
import os, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "extended_theta_sweep"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "dedup"))
import run_extended_sweeps as R
import ess_pilot as EP
import support_reciprocity as SR

OUT = os.path.join(HERE, "support_beta_results")
os.makedirs(OUT, exist_ok=True)
BETAS = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
N_SWEEPS, N_CHAINS, BURN = 100, 4, 0.5


def _contrib(T, cnt):
    p = len(cnt)
    return sum(cnt.values()) * (p - 1) / (len(T) - 1)


def _phi_state(evs):
    teams = defaultdict(Counter)
    for s, lst in evs.items():
        for e in lst:
            teams[frozenset((s,)) | frozenset(e)][s] += 1
    return teams, sum(_contrib(T, c) for T, c in teams.items())


def beta_chain(events, beta, n_sweeps, seed):
    """Metropolis chain; returns (final events, Phi trace per sweep)."""
    rng = np.random.default_rng(seed)
    evs = defaultdict(list)
    for s, Rr in events:
        evs[s].append(set(Rr))
    teams, Phi = _phi_state(evs)
    swap = [s for s, e in evs.items() if len(e) >= 2]
    tokens = sum(len(x) for e in evs.values() for x in e)

    def dec(T, s):
        c = teams[T]
        c[s] -= 1
        if c[s] == 0:
            del c[s]
        if not c:
            del teams[T]

    trace = np.empty(n_sweeps + 1)
    trace[0] = Phi
    for sw in range(n_sweeps):
        for _ in range(tokens):
            s = swap[rng.integers(len(swap))]
            lst = evs[s]
            m = len(lst)
            i = int(rng.integers(m)); j = int(rng.integers(m))
            if i == j:
                continue
            A = list(lst[i] - lst[j]); B = list(lst[j] - lst[i])
            if not A or not B or (len(A) == 1 and len(B) == 1):
                continue
            a = A[rng.integers(len(A))]; b = B[rng.integers(len(B))]
            fs = frozenset
            Ti_o = fs((s,)) | lst[i]; Tj_o = fs((s,)) | lst[j]
            Ti_n = Ti_o - {a} | {b};  Tj_n = Tj_o - {b} | {a}
            aff = {Ti_o, Tj_o, Ti_n, Tj_n}
            dPhi = -sum(_contrib(T, teams[T]) for T in aff if T in teams)
            dec(Ti_o, s); dec(Tj_o, s)
            teams[Ti_n][s] += 1; teams[Tj_n][s] += 1
            dPhi += sum(_contrib(T, teams[T]) for T in aff if T in teams)
            if beta * dPhi >= 0 or rng.random() < np.exp(beta * dPhi):
                lst[i].discard(a); lst[i].add(b)
                lst[j].discard(b); lst[j].add(a)
                Phi += dPhi
            else:                                   # revert counters
                dec(Ti_n, s); dec(Tj_n, s)
                teams[Ti_o][s] += 1; teams[Tj_o][s] += 1
        trace[sw + 1] = Phi
    out = [(s, sorted(e)) for s, lst in evs.items() for e in lst]
    return out, trace


def _selftest():
    import random
    rng = random.Random(5)
    ev = []
    for _ in range(120):
        s = rng.randrange(8)
        Rr = rng.sample([x for x in range(8) if x != s], rng.randrange(2, 4))
        ev.append((s, sorted(Rr)))
    evs = defaultdict(list)
    for s, Rr in ev:
        evs[s].append(set(Rr))
    _t, phi0 = _phi_state(evs)
    m = SR.support_measures_team(ev, min_k=2, weighting="M")
    assert abs(phi0 - m["R_part"] * len(ev)) < 1e-9
    out, tr = beta_chain(ev, 0.7, 20, seed=2)
    evs2 = defaultdict(list)
    for s, Rr in out:
        evs2[s].append(set(Rr))
    _t2, phi_direct = _phi_state(evs2)
    assert abs(tr[-1] - phi_direct) < 1e-6, (tr[-1], phi_direct)
    assert R.MT.proj_sig(out) == R.MT.proj_sig(ev)
    print("selftest OK: Phi == |E|*R_part^M; incremental == brute force; "
          "projection preserved")


def _job(task):
    beta, kind, start, seed, n_events = task
    t0 = time.time()
    out, tr = beta_chain(start, beta, N_SWEEPS, seed)
    burn = int(BURN * N_SWEEPS)
    mfin = SR.support_measures_team(out, min_k=3, weighting="M")
    return dict(beta=beta, kind=kind,
                R_part=float(tr[burn:].mean() / n_events),
                R_part_sd=float(tr[burn:].std() / n_events),
                R_any=mfin["R_any"], R_all=mfin["R_all"],
                proj_ok=True, wall_s=time.time() - t0)


def main(key="emaileu"):
    ev = R.load_raw(key)
    dd = sorted({(s, tuple(Rr)) for s, Rr in ev})
    dd = [(s, list(Rr)) for s, Rr in dd]
    fill = EP.projection_random_start(dd, seed=999)
    assert R.MT.proj_sig(fill) == R.MT.proj_sig(dd)
    obs = SR.support_measures_team(dd, min_k=3, weighting="M")
    print(f"{key}: dedup {len(dd):,} events  observed "
          f"R_any={obs['R_any']:.4f} R_part={obs['R_part']:.4f} "
          f"R_all={obs['R_all']:.4f}", flush=True)
    tasks = []
    for bi, b in enumerate(BETAS):
        for c in range(N_CHAINS):
            start = dd if c % 2 == 0 else fill
            tasks.append((b, "real" if c % 2 == 0 else "fill",
                          start, 500 + 37 * bi + 11 * c, len(dd)))
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex:
        for r_ in ex.map(_job, tasks):
            rows.append(r_)
            print(f"  beta={r_['beta']:+.1f} {r_['kind']:4s} "
                  f"R_part={r_['R_part']:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("beta").agg(
        R_part=("R_part", "mean"), R_part_sd=("R_part", "std"),
        R_any=("R_any", "mean"), R_all=("R_all", "mean")).reset_index()
    gap = df.pivot_table(index="beta", columns="kind", values="R_part")
    agg["two_start_gap"] = ((gap["real"] - gap["fill"]).abs()
                            / gap.max(axis=1)).values
    for k in ("R_any", "R_part", "R_all"):
        agg[f"obs_{k}"] = obs[k]
    agg["dataset"] = key
    agg.to_csv(os.path.join(OUT, f"beta_sweep_{key}.csv"), index=False)
    print(agg.to_string(index=False), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    _selftest()
    main(sys.argv[1] if len(sys.argv) > 1 else "emaileu")

# How this was run (2026-09-16, abstract-grade budget):
#   cd Hypergraph_GroupReciprocity/experiments && mkdir -p support_beta_results
#   caffeinate -i <venv-python> support_beta_sweep.py emaileu \
#       > support_beta_results/run_emaileu.log 2>&1
# BETAS/N_SWEEPS/N_CHAINS are constants at top (100 sweeps, 4 chains --
# increase to >=300 sweeps for the main paper). Output:
# support_beta_results/beta_sweep_<dataset>.csv; figure: make_support_beta_fig.py.
