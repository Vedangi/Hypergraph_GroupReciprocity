#!/usr/bin/env python3
"""Overnight driver: cheapest experiments first, incremental saves throughout.

Stage A  duplicate-%% of theta=0 null states, all 8 datasets (small -> large)
Stage B  extended theta sweep for the new datasets: wiki, radoslaw, higgs
Stage C  Twitter re-run at 160 sweeps, 6 chains  (removes lower-bound dagger)
Stage D  Enron   re-run at 120 sweeps, 4 chains  (longest; may not finish
         on battery -- existing enron rows are only replaced on completion)

Chains run in a process pool (6 workers).  After every dataset the CSV and
plots are rewritten and synced to ../../results/, so a forced sleep loses at
most the dataset in flight.  STAGES=ABCD / SMOKE=1 env vars control scope.
"""
import os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_extended_sweeps as R          # brings MT, MR, loaders, plotting

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
STAGES = os.environ.get("STAGES", "ABCD").upper()
WORKERS = int(os.environ.get("WORKERS", "6"))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results",
                                       "extended_theta_sweep"))
SWEEPS_CSV = os.path.join(HERE, ("smoke_" if SMOKE else "") + "extended_sweeps.csv")
DUP_CSV = os.path.join(HERE, ("smoke_" if SMOKE else "") + "null_dup_stats.csv")
THETAS = [0.0, 2.0] if SMOKE else list(R.THETAS)

_G = {}

def _init(ev, rand, sig):
    _G.update(ev=ev, rand=rand, sig=sig)

def _chain(task):
    theta, r, n_steps, seed = task
    which = "real" if r % 2 == 0 else "random"
    start = _G["ev"] if which == "real" else _G["rand"]
    t0 = time.time()
    out, Phi, st = R.MT.tilt_sample(start, theta, n_steps=n_steps, seed=seed,
                                    return_stats=True, base="labelled")
    ok = R.MT.proj_sig(out) == _G["sig"]
    m = R.MR.measures(out, min_k=3)
    return theta, r, which, Phi, st["accept_rate"], m, ok, time.time() - t0

def _dup_chain(task):
    n_steps, seed = task
    out, _ = R.MT.tilt_sample(_G["ev"], 0.0, n_steps=n_steps, seed=seed)
    nd = len({(s, frozenset(Rr)) for s, Rr in out})
    return 100.0 * (len(out) - nd) / len(out)

def merge_rows(df_new, key):
    # fall back to the shipped results copy so a fresh checkout never
    # starts the merge from an empty table (lost fauci/dnc once)
    src = SWEEPS_CSV if os.path.exists(SWEEPS_CSV) else os.path.join(
        RESULTS, "extended_sweeps.csv")
    old = pd.read_csv(src) if os.path.exists(src) else pd.DataFrame()
    if len(old):
        old = old[old.dataset != key]
    allp = pd.concat([old, df_new], ignore_index=True)
    allp.to_csv(SWEEPS_CSV, index=False)
    return allp

def sync_results():
    if SMOKE:
        return
    os.makedirs(RESULTS, exist_ok=True)
    for f in os.listdir(HERE):
        if f.endswith((".csv", ".pdf")) and not f.startswith("smoke_"):
            src = os.path.join(HERE, f)
            dst = os.path.join(RESULTS, f)
            with open(src, "rb") as a, open(dst, "wb") as b:
                b.write(a.read())

def sweep_parallel(key, label, n_chains, sweep_mult, seed_base):
    ev = R.load_raw(key)
    nd = len({(s, frozenset(Rr)) for s, Rr in ev})
    tokens = sum(len(Rr) for _s, Rr in ev)
    floor = 20_000 if SMOKE else R.MIN_STEPS
    n_steps = max(sweep_mult * tokens, floor)
    real = R.MR.measures(ev, min_k=3)
    print(f"\n===== {label} ({key}) ===== {len(ev):,} events "
          f"({100*(len(ev)-nd)/len(ev):.1f}% dup), {tokens:,} tokens, "
          f"n_steps={n_steps:,} ({n_steps/tokens:.0f} sweeps), "
          f"{n_chains} chains x {len(THETAS)} thetas", flush=True)
    t0 = time.time()
    rand_steps = min(n_steps, max(40 * tokens, floor))
    rand, _ = R.MT.tilt_sample(ev, 0.0, n_steps=rand_steps, seed=8888)
    sig = R.MT.proj_sig(ev)
    print(f"  randomised start built in {time.time()-t0:.0f}s", flush=True)

    tasks = [(th, r, n_steps, seed_base + 997 * r + 13 * ti)
             for ti, th in enumerate(THETAS) for r in range(n_chains)]
    res = []
    with ProcessPoolExecutor(WORKERS, initializer=_init,
                             initargs=(ev, rand, sig)) as ex:
        for out in ex.map(_chain, tasks):
            th, r, which, Phi, acc_rate, m, ok, dt = out
            print(f"    theta={th:+.2f} chain={r} start={which:6s} "
                  f"Phi={Phi:10,.1f} r2={m['r2_multi']:.4f} ({dt:.0f}s)",
                  flush=True)
            res.append(out)

    rows = []
    for ti, th in enumerate(THETAS):
        sub = [x for x in res if x[0] == th]
        acc = {k: [x[5][k] for x in sub] for k in R.KEYS}
        phis = {"real": [x[3] for x in sub if x[2] == "real"],
                "random": [x[3] for x in sub if x[2] == "random"]}
        mr_, mn_ = np.mean(phis["real"]), np.mean(phis["random"])
        gap = abs(mr_ - mn_) / max(mr_, mn_) if max(mr_, mn_) else 0.0
        rows.append({"dataset": key, "label": label, "theta": th,
                     **{k: float(np.mean(v)) for k, v in acc.items()},
                     "Phi": float(np.mean([x[3] for x in sub])),
                     "accept": float(np.mean([x[4] for x in sub])),
                     "Phi_real_start": float(mr_), "Phi_random_start": float(mn_),
                     "Phi_sd": float(np.std([x[3] for x in sub])),
                     "two_start_gap": gap, "n_chains": n_chains,
                     "n_steps": n_steps, "sweeps": n_steps / tokens,
                     "projection_fixed": bool(all(x[6] for x in sub)),
                     **{f"real_{k}": real[k] for k in R.KEYS},
                     "n_events": len(ev), "n_teams": real["n_teams"],
                     "pct_dup": 100 * (len(ev) - nd) / len(ev)})
        print(f"    theta={th:+.2f} two-start gap {gap:.2%}"
              f"{'  <-- CHECK MIXING' if gap > 0.03 else ''}", flush=True)
    df = pd.DataFrame(rows)
    df["theta_hat"] = R.theta_hat(df.theta, df.r2_multi, real["r2_multi"])
    allp = merge_rows(df, key)
    R.plot_one(df, real, label, os.path.join(
        HERE, ("smoke_" if SMOKE else "") + f"ext_tilt_{key}.pdf"))
    R.plot_summary(allp, os.path.join(
        HERE, ("smoke_" if SMOKE else "") + "ext_theta_hat.pdf"))
    sync_results()
    n0 = df[df.theta == 0.0].iloc[0]
    print(f"  DONE {label}: theta_hat={df.theta_hat.iloc[0]:.2f} "
          f"null={n0.r2_multi:.4f} real={real['r2_multi']:.4f} "
          f"({real['r2_multi']/n0.r2_multi:.2f}x) "
          f"worst gap={df.two_start_gap.max():.2%}", flush=True)

def stage_A():
    order = [("fauci", 3), ("wiki", 3), ("dnc", 3), ("radoslaw", 3),
             ("higgs", 3), ("twitter", 3), ("emaileu", 3), ("enron", 3)]
    if SMOKE:
        order = [("fauci", 2)]
    rows = []
    for key, reps in order:
        try:
            ev = R.load_raw(key)
            nd = len({(s, frozenset(Rr)) for s, Rr in ev})
            tokens = sum(len(Rr) for _s, Rr in ev)
            n_steps = max(40 * tokens, 20_000 if SMOKE else 2_000_000)
            t0 = time.time()
            with ProcessPoolExecutor(min(WORKERS, reps), initializer=_init,
                                     initargs=(ev, None, None)) as ex:
                dups = list(ex.map(_dup_chain,
                                   [(n_steps, 777 + i) for i in range(reps)]))
            real_dup = 100 * (len(ev) - nd) / len(ev)
            rows.append({"dataset": key, "n_events": len(ev),
                         "real_pct_dup": real_dup,
                         "null_pct_dup_mean": float(np.mean(dups)),
                         "null_pct_dup_sd": float(np.std(dups)),
                         "reps": reps, "n_steps": n_steps})
            print(f"  [A] {key:9s} real dup {real_dup:5.1f}%  ->  null dup "
                  f"{np.mean(dups):5.2f}% +- {np.std(dups):.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(DUP_CSV, index=False)
        except Exception as exc:
            print(f"  [A] {key} FAILED: {exc}", flush=True)
            traceback.print_exc(limit=2)
    sync_results()

def main():
    t00 = time.time()
    print(f"overnight run: STAGES={STAGES} SMOKE={SMOKE} workers={WORKERS}",
          flush=True)
    if "A" in STAGES:
        print("\n########## STAGE A: null duplicate fractions ##########",
              flush=True)
        stage_A()
    for stage, cfg in (("B", [("wiki", "Wiki talk", 6, 40),
                              ("radoslaw", "Manufacturing email", 6, 40),
                              ("higgs", "Higgs Twitter", 6, 40)]),
                       ("C", [("twitter", "Twitter", 6, 160)]),
                       ("D", [("enron", "Enron", 4, 120)])):
        if stage not in STAGES:
            continue
        if SMOKE and stage != "B":
            continue
        cfg2 = [c for c in cfg if not SMOKE or c[0] == "wiki"]
        print(f"\n########## STAGE {stage} ##########", flush=True)
        for i, (key, label, nch, mult) in enumerate(cfg2):
            try:
                sweep_parallel(key, label, 2 if SMOKE else nch, mult,
                               {"B": 200000, "C": 300000, "D": 400000}[stage])
            except Exception as exc:
                print(f"  {key} FAILED: {exc}", flush=True)
                traceback.print_exc(limit=3)
    print(f"\nALL REQUESTED STAGES DONE in {(time.time()-t00)/3600:.2f}h",
          flush=True)

if __name__ == "__main__":
    main()
