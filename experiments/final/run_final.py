#!/usr/bin/env python3
"""Orchestrator for the final experiments (results/final_expt_plan.md).

Builds the chain task list, schedules longest-first over a process pool, and
skips chains whose checkpoint already reaches the target sweeps -- so the
plan's adaptive-extension rule is: raise --sweeps for the failing
(dataset, beta) combos (via --extend) and rerun; completed chains are
untouched, others resume from their checkpoint.

Phases:
  A  beta=0 null: 9 datasets x 6 chains (3 observed + 3 fill), 1000 sweeps
  B  beta sweep: {emaileu, enron, twitter, congress} x beta in {-1,1,2,3,4,5}
     x 4 chains (2+2); 500 sweeps for |beta|<=2, 1000 for beta>=3
  S  size-cap sensitivity: {enron, congress} x {k<=10, uncapped} x 4 chains
     (the k<=25 runs are phase A), 1000 sweeps
  D  support analysis: beta=0 null started from the DEDUPLICATED hypergraph
     E* (each distinct hyperedge once), 9 datasets x 6 chains, 1000 sweeps;
     report the M-weighted support measures (*_support_M), denominator |E*|
"""
import argparse, json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_chains as FC

DATASETS = ["congress", "music", "dblp", "enron", "emaileu", "twitter",
            "radoslaw", "dnc", "fauci"]
BETA_DATASETS = ["emaileu", "enron", "twitter", "congress"]
BETAS = [-1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
TOKENS = {"congress": 661297, "music": 431689, "enron": 336454,
          "dblp": 221926, "emaileu": 139453, "twitter": 51481,
          "radoslaw": 30318, "dnc": 16040, "fauci": 1288,
          ("enron", 3, 10): 171413, ("enron", 3, 10**9): 965447,
          ("congress", 3, 10): 205249, ("congress", 3, 10**9): 2127636}


def tasks_for(phase, out_root, sweeps_override=None, smoke=False):
    T = []
    def add(ds, beta, cid, init, sweeps, min_k=3, max_k=25, tag=None, dedup=False):
        tag = tag or f"beta{beta:+.1f}"
        out = os.path.join(out_root, ds, tag, f"chain{cid}_{init}")
        T.append(dict(dataset=ds, beta=beta, chain_id=cid, init=init,
                      sweeps=sweeps_override or sweeps, out_dir=out,
                      min_k=min_k, max_k=max_k, dedup=dedup))
    if "A" in phase:
        for ds in (["fauci", "dnc"] if smoke else DATASETS):
            for c in range(2 if smoke else 6):
                add(ds, 0.0, c, "observed" if c % 2 == 0 else "fill",
                    8 if smoke else 1000)
    if "B" in phase:
        for ds in (["fauci"] if smoke else BETA_DATASETS):
            for b in ([1.0, 3.0] if smoke else BETAS):
                for c in range(2 if smoke else 4):
                    add(ds, b, c, "observed" if c % 2 == 0 else "fill",
                        8 if smoke else (500 if abs(b) <= 2 else 1000))
    if "S" in phase:
        for ds in (["fauci"] if smoke else ["enron", "congress"]):
            for mk in ((10,) if smoke else (10, 10**9)):
                for c in range(2 if smoke else 4):
                    add(ds, 0.0, c, "observed" if c % 2 == 0 else "fill",
                        8 if smoke else 1000, 3, mk,
                        tag=f"beta+0.0_k3-{mk if mk < 10**9 else 'inf'}")
    if "D" in phase:
        for ds in (["fauci", "dnc"] if smoke else DATASETS):
            for c in range(2 if smoke else 6):
                add(ds, 0.0, c, "observed" if c % 2 == 0 else "fill",
                    8 if smoke else 1000, tag="beta+0.0_dedup", dedup=True)
    return T


def cost(t):
    key = (t["dataset"], t["min_k"], t["max_k"]) if t["max_k"] != 25 else t["dataset"]
    return TOKENS.get(key, 100000) * t["sweeps"] * (0.6 if t.get("dedup") else 1.0)


def done(t):
    ck = os.path.join(t["out_dir"], "checkpoint.pkl")
    mj = os.path.join(t["out_dir"], "meta.json")
    if not (os.path.exists(ck) and os.path.exists(mj)):
        return False
    with open(mj) as f:
        return json.load(f).get("sweeps", 0) >= t["sweeps"]


def _run(t):
    try:
        m = FC.run_chain(t)
        return t, m, None
    except Exception:
        return t, None, traceback.format_exc()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="ABSD", help="any of A, B, S, D")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "final"))
    p.add_argument("--sweeps", type=int, default=None,
                   help="override target sweeps for ALL selected tasks")
    p.add_argument("--extend", nargs="*", default=[],
                   help="dataset[:beta] combos to extend, e.g. enron:3.0 "
                        "congress  (used with --sweeps)")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out_root = os.path.abspath(a.out if not a.smoke else a.out + "_smoke")
    T = tasks_for(a.phase, out_root, a.sweeps, a.smoke)
    if a.extend:
        want = set(a.extend)
        T = [t for t in T if t["dataset"] in want
             or f"{t['dataset']}:{t['beta']:g}" in want]
    pending = [t for t in T if not done(t)]
    pending.sort(key=cost, reverse=True)                     # longest first
    print(f"{len(T)} tasks, {len(T)-len(pending)} already complete, "
          f"{len(pending)} to run on {a.workers} workers -> {out_root}", flush=True)
    os.makedirs(out_root, exist_ok=True)
    errlog = open(os.path.join(out_root, "errors.log"), "a")
    t0 = time.time(); n_ok = n_err = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_run, t) for t in pending]
        for fu in as_completed(futs):
            t, m, err = fu.result()
            tag = (f"{t['dataset']} beta={t['beta']:+.1f} k{t['min_k']}-{t['max_k']}"
                   f"{'-dedup' if t.get('dedup') else ''} chain{t['chain_id']}/{t['init']}")
            if err:
                n_err += 1
                errlog.write(f"[{time.strftime('%F %T')}] {tag}\n{err}\n"); errlog.flush()
                print(f"  FAILED {tag}", flush=True)
            else:
                n_ok += 1
                print(f"  done {tag}: {m['sweeps']} sweeps, acc={m['accept_rate']:.3f}, "
                      f"{m['wall_s_last_segment']/60:.1f} min  [{n_ok+n_err}/{len(pending)}, "
                      f"{(time.time()-t0)/3600:.2f} h]", flush=True)
    print(f"ALL DONE: {n_ok} ok, {n_err} failed in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()

# Phase D (support analysis, dedup start), run on its own:
#   nohup python run_final.py --phase D --workers 54 > ../../results/final/run_D.log 2>&1 &
# How to run on the 64-core server (2026-10-11):
#   cd Hypergraph_GroupReciprocity/experiments/final
#   nohup python run_final.py --phase ABS --workers 60 > ../../results/final/run.log 2>&1 &
# Extension after diagnostics (example: enron beta=3 and all congress tasks to 2000 sweeps):
#   python run_final.py --phase AB --extend enron:3.0 congress --sweeps 2000 --workers 60
# Smoke test (minutes, separate results/final_smoke dir):
#   python run_final.py --phase ABS --smoke --workers 4
# Then: python diagnose_final.py [--out ../../results/final]
