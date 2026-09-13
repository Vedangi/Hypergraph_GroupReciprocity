#!/usr/bin/env python3
"""Relaxation traces for the burn-in/decay figure: R_some per sweep from
sweep 0, theta=0, two chains per dataset (real start: decays to the null;
projection-fill start: begins at the null), 300 sweeps."""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "extended_theta_sweep"))
import run_extended_sweeps as R
import ess_pilot as EP

OUT = os.path.join(HERE, "relaxation_results")
os.makedirs(OUT, exist_ok=True)
N_SWEEPS = 300
DATASETS = ("fauci", "wiki", "dnc", "radoslaw", "higgs", "twitter", "enron")


def _job(task):
    key, kind, start, seed = task
    phi0 = R.MT._build_state(start, "r2")[2]
    tr = EP.tilt_trace(start, N_SWEEPS, seed)
    return key, kind, np.concatenate([[phi0], tr])


def main():
    tasks, meta = [], []
    for key in DATASETS:
        ev = R.load_raw(key)
        fill = EP.projection_random_start(ev, seed=4242)
        assert R.MT.proj_sig(fill) == R.MT.proj_sig(ev)
        tasks += [(key, "real", ev, 17), (key, "fill", fill, 71)]
        meta.append(dict(dataset=key, n_events=len(ev),
                         real_r2=R.MR.measures(ev, min_k=3)["r2_multi"]))
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "meta.csv"), index=False)
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex:
        for key, kind, tr in ex.map(_job, tasks):
            np.save(os.path.join(OUT, f"relax_{key}_{kind}.npy"), tr)
            print(f"  {key:9s} {kind:4s} Phi {tr[0]:,.0f} -> {tr[-1]:,.0f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

# How this was run (2026-09-13):
#   cd Hypergraph_GroupReciprocity/experiments && mkdir -p relaxation_results
#   caffeinate -i <venv-python> relaxation_traces.py > relaxation_results/run.log 2>&1
# No CLI arguments (datasets/N_SWEEPS are constants at top). ~35 min.
# Outputs: relaxation_results/relax_<dataset>_<real|fill>.npy + meta.csv.
