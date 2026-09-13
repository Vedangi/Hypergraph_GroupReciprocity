#!/usr/bin/env python3
"""Relaxation traces for the SUPPORT measures, chains started from the
DEDUPLICATED hypergraph (distinct (s,R) events, 3 <= k <= 25).

Same theta=0 within-sender chain as everywhere else (labelled base); per
sweep we snapshot all three support measures R_any/R_part/R_all in BOTH
weightings of support_reciprocity.py:
  M-weighted  (denominator |E| fixed by the sampler; = per-event summation)
  p-weighted  (the paper's support-hypergraph formula; denominator Sum p_T
               varies from state to state as duplicates form)
At sweep 0 the state is duplicate-free, so M == p and the two coincide;
they separate exactly when the chain creates duplicate hyperedges -- the
figure therefore doubles as the "support measures act threshold-y on
multisets" exhibit.  Two chains per dataset: dedup-real start and a
randomized Gale-Ryser fill of the dedup projection.
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

OUT = os.path.join(HERE, "relaxation_support_results")
os.makedirs(OUT, exist_ok=True)
N_SWEEPS = 300
DATASETS = ("fauci", "wiki", "dnc", "radoslaw", "higgs", "twitter", "enron")
COLS = [f"{k}_{w}" for w in ("M", "p") for k in SR.F]   # 6 columns


def support_vals(evs):
    """All six support values from the sampler's state dict."""
    m = defaultdict(Counter)
    for s, lst in evs.items():
        for e in lst:
            m[frozenset({s}) | frozenset(e)][s] += 1
    st = [(sum(c.values()), len(c), len(T))
          for T, c in m.items() if len(T) >= 3]
    out = []
    for w in ("M", "p"):
        den = sum((M if w == "M" else p) for M, p, _k in st)
        for key, f in SR.F.items():
            out.append(sum((M if w == "M" else p) * f(p, k)
                           for M, p, k in st) / den if den else 0.0)
    return out


def support_trace(events, n_sweeps, seed):
    MT = R.MT
    rng = np.random.default_rng(seed)
    evs, tsc, Phi, _ = MT._build_state(events, "r2")
    swap = [s for s, e in evs.items() if len(e) >= 2]
    tokens = sum(len(x) for e in evs.values() for x in e)
    snaps = [support_vals(evs)]
    for _sw in range(n_sweeps):
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
            MT._apply_switch(evs, tsc, s, i, j, a, b, "r2", None)
        snaps.append(support_vals(evs))
    return np.array(snaps)                          # (n_sweeps+1, 6)


def _job(task):
    key, kind, start, seed = task
    return key, kind, support_trace(start, N_SWEEPS, seed)


def main():
    tasks, meta = [], []
    for key in DATASETS:
        ev = R.load_raw(key)
        dd = sorted({(s, tuple(Rr)) for s, Rr in ev})
        dd = [(s, list(Rr)) for s, Rr in dd]
        fill = EP.projection_random_start(dd, seed=4242)
        assert R.MT.proj_sig(fill) == R.MT.proj_sig(dd)
        tasks += [(key, "real", dd, 17), (key, "fill", fill, 71)]
        meta.append(dict(dataset=key, n_events_multi=len(ev),
                         n_events_dedup=len(dd)))
        print(f"  {key:9s} dedup {len(dd):,}/{len(ev):,} events", flush=True)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "meta.csv"), index=False)
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex:
        for key, kind, tr in ex.map(_job, tasks):
            np.save(os.path.join(OUT, f"relax_{key}_{kind}.npy"), tr)
            print(f"  {key:9s} {kind:4s} R_part^M {tr[0,1]:.4f} -> "
                  f"{tr[-1,1]:.4f}  (p-w drift {tr[-1,4]-tr[-1,1]:+.4f})  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

# How this was run (2026-09-13):
#   cd Hypergraph_GroupReciprocity/experiments
#   mkdir -p relaxation_support_results
#   caffeinate -i <venv-python> relaxation_support.py \
#       > relaxation_support_results/run.log 2>&1
# (no CLI arguments; venv needs numpy/pandas. Datasets/sweep count are the
#  constants at the top. Outputs: relax_<dataset>_<real|fill>.npy with
#  columns R_any_M,R_part_M,R_all_M,R_any_p,R_part_p,R_all_p per sweep,
#  plus meta.csv. Figure: make_relaxation_support_fig.py.)
