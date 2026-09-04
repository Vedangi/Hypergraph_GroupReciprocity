#!/usr/bin/env python3
"""The unlabelled theta=0 chain was still drifting at 22.4M steps, so bracket it:
run from a HIGH start (the real data, Phi~6950) and a LOW start (a theta=-1
unlabelled sample, Phi~1270) at increasing budgets.  Convergence of the two is
the diagnostic; a persistent gap means neither is an equilibrium value.
    -> multi_unlabelled_mixing.csv
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "multihypergraph"))
import multi_reciprocity as MR, multi_tilt as MT
from run_unlabelled_emaileu import load_raw

ev = load_raw()
real = MR.measures(ev, min_k=3)
E = real["n_events"]
print(f"email-Eu: {len(ev):,} events, Phi(real)={MT._build_state(ev,'r2')[2]:,.1f}",
      flush=True)

lo, _ = MT.tilt_sample(ev, -1.0, n_steps=11_200_000, seed=77, base="unlabelled")
print(f"LOW start (theta=-1, unlabelled): Phi={MT._build_state(lo,'r2')[2]:,.1f}",
      flush=True)

rows = []
for ns in (22_400_000, 44_800_000, 89_600_000):
    vals = {}
    for nm, st in (("high_real", ev), ("low_tilted", lo)):
        t0 = time.time()
        _, Phi = MT.tilt_sample(st, 0.0, n_steps=ns, seed=5, base="unlabelled")
        vals[nm] = Phi
        print(f"  n_steps={ns:>11,}  {nm:11s} Phi={Phi:9,.1f} "
              f"r2_multi={Phi/E:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    gap = abs(vals["high_real"] - vals["low_tilted"]) / max(vals.values())
    rows.append({"n_steps": ns, "Phi_from_high": vals["high_real"],
                 "Phi_from_low": vals["low_tilted"],
                 "r2_from_high": vals["high_real"] / E,
                 "r2_from_low": vals["low_tilted"] / E, "rel_gap": gap})
    print(f"  --> relative gap {gap:.2%}", flush=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(HERE, "multi_unlabelled_mixing.csv"), index=False)
print("\nDONE", flush=True)
