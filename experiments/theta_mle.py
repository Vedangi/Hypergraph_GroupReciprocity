#!/usr/bin/env python3
"""theta-hat inference pass (X.4 completion): for each communication dataset,
run chains AT the fitted theta-hat and report

  1. moment-match validation: E_th[Phi] +- MCSE vs Phi_obs, plus a one-step
     Newton refinement  theta' = theta + (Phi_obs - E[Phi]) / Var[Phi]
     (valid because dE/dtheta = Var[Phi] in the exponential family);
  2. SE(theta-hat) = 1 / sqrt(Var_th[Phi])   (Fisher information);
  3. goodness-of-fit on statistics NOT used in fitting: R_any, R_all,
     R_any-support -- model mean +- sd over sampled states vs observed,
     discrepancy in sd units.

4 chains per dataset (2 from the observed hypergraph, 2 from randomized
Gale-Ryser fills), 1000 sweeps, Phi traced every sweep, full measures
snapshotted every 5 sweeps after 25% burn-in.
"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "extended_theta_sweep"))
import run_extended_sweeps as R
import ess_pilot as EP

OUT = os.path.join(HERE, "theta_mle_results")
os.makedirs(OUT, exist_ok=True)
N_CHAINS, N_SWEEPS, BURN, SNAP = 4, 1000, 0.25, 5
GOF_KEYS = ("s1_multi", "r1_multi", "s1_support")


def tilt_trace_snap(events, theta, n_sweeps, seed):
    """Phi per sweep + full measures every SNAP sweeps after burn-in."""
    MT, MR = R.MT, R.MR
    rng = np.random.default_rng(seed)
    evs, tsc, Phi, _ = MT._build_state(events, "r2")
    swap = [s for s, e in evs.items() if len(e) >= 2]
    tokens = sum(len(x) for e in evs.values() for x in e)
    burn = int(BURN * n_sweeps)
    trace = np.empty(n_sweeps)
    snaps = []
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
            dPhi, _ = MT._apply_switch(evs, tsc, s, i, j, a, b, "r2", None)
            if theta * dPhi >= 0 or rng.random() < np.exp(theta * dPhi):
                Phi += dPhi
            else:
                MT._apply_switch(evs, tsc, s, i, j, b, a, "r2", None)
        trace[sw] = Phi
        if sw >= burn and (sw - burn) % SNAP == 0:
            state = [(s, sorted(e)) for s, lst in evs.items() for e in lst]
            snaps.append(MR.measures(state, min_k=3))
    return trace, snaps


def _job(task):
    start, theta, seed = task
    return tilt_trace_snap(start, theta, N_SWEEPS, seed)


def main():
    sweeps = pd.read_csv(os.path.join(
        HERE, "..", "results", "extended_theta_sweep", "extended_sweeps.csv"))
    rows = []
    for key in ("wiki", "fauci", "radoslaw", "dnc", "higgs",
                "twitter", "enron"):
        g = sweeps[sweeps.dataset == key]
        if not len(g):
            print(f"-- {key}: no theta_hat, skipping", flush=True)
            continue
        th = float(g.theta_hat.iloc[0])
        ev = R.load_raw(key)
        real = R.MR.measures(ev, min_k=3)
        Phi_obs = len(ev) * real["r2_multi"]
        print(f"\n===== {key} ===== theta_hat={th:.3f}  "
              f"Phi_obs={Phi_obs:,.1f}  ({len(ev):,} events)", flush=True)
        fills = [EP.projection_random_start(ev, seed=1000 + i)
                 for i in range(N_CHAINS // 2)]
        starts = [ev if i % 2 == 0 else fills[i // 2]
                  for i in range(N_CHAINS)]
        t0 = time.time()
        with ProcessPoolExecutor(N_CHAINS) as ex:
            res = list(ex.map(_job, [(starts[i], th, 91 * i + 3)
                                     for i in range(N_CHAINS)]))
        dt = time.time() - t0
        burn = int(BURN * N_SWEEPS)
        kept = [tr[burn:] for tr, _sn in res]
        taus = [EP.tau_int(k_) for k_ in kept]
        ess = sum(len(k_) / t for k_, t in zip(kept, taus))
        rhat = EP.split_rhat(kept)
        allk = np.concatenate(kept)
        mu, var = allk.mean(), allk.var()
        mcse = np.sqrt(var / ess)
        se_th = 1.0 / np.sqrt(var)
        newton = th + (Phi_obs - mu) / var
        resid_sd = (Phi_obs - mu) / np.sqrt(var)
        row = dict(dataset=key, theta_hat=th, Phi_obs=Phi_obs,
                   E_Phi=mu, MCSE_Phi=mcse, Var_Phi=var,
                   resid_sd=resid_sd, theta_newton=newton,
                   SE_theta=se_th, tau_int=float(np.mean(taus)),
                   ESS=ess, rhat=rhat, wall_s=dt)
        print(f"  E[Phi]={mu:,.1f}+-{mcse:.1f}  resid={resid_sd:+.2f} sd  "
              f"theta_newton={newton:.3f}  SE(theta)={se_th:.3f}  "
              f"ESS={ess:,.0f} Rhat={rhat:.4f}  [{dt:.0f}s]", flush=True)
        snaps = [s for _tr, sn in res for s in sn]
        for k_ in GOF_KEYS:
            vals = np.array([s[k_] for s in snaps])
            z = (real[k_] - vals.mean()) / vals.std() if vals.std() else np.nan
            row[f"gof_{k_}_model"] = vals.mean()
            row[f"gof_{k_}_sd"] = vals.std()
            row[f"gof_{k_}_obs"] = real[k_]
            row[f"gof_{k_}_z"] = z
            print(f"  GOF {k_:11s} model={vals.mean():.4f}+-{vals.std():.4f} "
                  f" obs={real[k_]:.4f}  z={z:+.1f}", flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "theta_mle.csv"),
                                  index=False)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
