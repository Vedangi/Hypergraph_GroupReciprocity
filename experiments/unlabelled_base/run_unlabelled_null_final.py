#!/usr/bin/env python3
"""One converged unlabelled theta=0 sample (89.6M steps -- the budget at which
the high/low bracketing test agreed to 0.06%), reporting ALL measures so the
appendix row is equilibrium-grade rather than terminal-state."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_R = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_R, "src"))
sys.path.insert(0, os.path.join(_R, "src", "multihypergraph"))
import multi_reciprocity as MR, multi_tilt as MT
from run_unlabelled_emaileu import load_raw

HERE = os.path.dirname(os.path.abspath(__file__))
ev = load_raw(); real = MR.measures(ev, min_k=3)
out = {"real": {k: real[k] for k in real if isinstance(real[k], (int, float))}}
for base in ("unlabelled", "labelled"):
    t0 = time.time()
    s, Phi = MT.tilt_sample(ev, 0.0, n_steps=89_600_000, seed=5, base=base)
    m = MR.measures(s, min_k=3)
    out[base] = {k: m[k] for k in m if isinstance(m[k], (int, float))}
    out[base]["Phi"] = Phi
    print(f"{base:11s} Phi={Phi:,.1f}  " + "  ".join(
        f"{k}={m[k]:.4f}" for k in ("r2_multi","s1_multi","r1_multi",
        "s1_support","graph_binary","graph_weighted")) +
        f"  ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(os.path.join(HERE,"multi_unlabelled_null_final.json"),"w"),
          indent=2)
print("\nratios (real / null):", flush=True)
for k in ("r2_multi","s1_multi","r1_multi","s1_support"):
    print(f"  {k:12s} real={real[k]:.4f}  "
          f"labelled={out['labelled'][k]:.4f} ({real[k]/out['labelled'][k]:.2f}x)  "
          f"unlabelled={out['unlabelled'][k]:.4f} ({real[k]/out['unlabelled'][k]:.2f}x)",
          flush=True)
print("DONE", flush=True)
