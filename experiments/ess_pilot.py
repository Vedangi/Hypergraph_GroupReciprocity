#!/usr/bin/env python3
"""ESS pilot (standalone -- touches nothing in the existing setup).

For each (small) dataset, run 4 theta=0 chains: 2 from the REAL hypergraph
and 2 from a chain-free randomized start (projection + per-event sizes fixed,
recipients refilled by a randomized Gale-Ryser construction).  Record the
Phi trace every sweep, then report per dataset:

  tau_int (Sokal windowing, c=5), combined ESS, split-Rhat over the 4 chains,
  null mean +- MCSE for R_some = Phi/|E|, and the sweeps/chain needed for a
  combined ESS of 2000.

Also validates the randomized start: identical projection signature and
per-sender event-size multisets, and reports its initial Phi.
"""
import os, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "extended_theta_sweep"))
import run_extended_sweeps as R          # load_raw + MT/MR, unchanged

DATASETS = os.environ.get(
    "ESS_DATASETS", "fauci,wiki,dnc,radoslaw,higgs").split(",")
N_CHAINS = int(os.environ.get("ESS_CHAINS", "4"))   # half real-, half fill-start
SWEEPS_OVERRIDE = int(os.environ.get("ESS_SWEEPS", "0"))
BURN_FRAC = 0.25
OUT = os.path.join(HERE, "ess_pilot_results")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- random start
def projection_random_start(events, seed):
    """Chain-free randomized state with the SAME senders, per-event sizes and
    weighted projection: per sender, refill each event's recipient set by
    weighted sampling without replacement from the remaining recipient
    multiset (retry on dead-ends, largest-remaining greedy as fallback)."""
    rng = np.random.default_rng(seed)
    by_s = defaultdict(list)
    for s, Rr in events:
        by_s[s].append(list(Rr))
    out = []
    for s, evlist in by_s.items():
        sizes = [len(x) for x in evlist]
        pool = Counter(r for x in evlist for r in x)
        fill = _fill_random(sizes, pool, rng)
        out.extend((s, sorted(f)) for f in fill)
    return out


def _fill_random(sizes, pool, rng, tries=25):
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
    for _ in range(tries):
        rem = dict(pool)
        res = {}
        ok = True
        for i in order:
            k = sizes[i]
            cand = [r for r, c in rem.items() if c > 0]
            if len(cand) < k:
                ok = False
                break
            w = np.array([rem[r] for r in cand], float)
            idx = rng.choice(len(cand), size=k, replace=False, p=w / w.sum())
            take = [cand[t] for t in idx]
            for r in take:
                rem[r] -= 1
            res[i] = take
        if ok:
            return [set(res[i]) for i in range(len(sizes))]
    # deterministic largest-remaining-multiplicity greedy: always feasible
    rem = dict(pool)
    res = {}
    for i in order:
        k = sizes[i]
        cand = sorted((r for r, c in rem.items() if c > 0),
                      key=lambda r: (-rem[r], rng.random()))
        take = cand[:k]
        for r in take:
            rem[r] -= 1
        res[i] = take
    return [set(res[i]) for i in range(len(sizes))]


# ---------------------------------------------------------------- traced chain
def tilt_trace(events, n_sweeps, seed, theta=0.0):
    """theta=0 labelled chain identical to multi_tilt.tilt_sample, but
    recording Phi once per sweep (uses MT's own state/switch helpers)."""
    MT = R.MT
    rng = np.random.default_rng(seed)
    evs, tsc, Phi, _m = MT._build_state(events, "r2")
    swap = [s for s, e in evs.items() if len(e) >= 2]
    tokens = sum(len(x) for e in evs.values() for x in e)
    trace = np.empty(n_sweeps)
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
    return trace


# ---------------------------------------------------------------- diagnostics
def tau_int(x, c=5.0):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    v = np.dot(x, x) / n
    if v == 0:
        return 1.0
    rho = np.array([np.dot(x[:n - t], x[t:]) / ((n - t) * v)
                    for t in range(1, n // 3)])
    tau = 1.0
    for M in range(1, len(rho) + 1):
        tau = 1.0 + 2.0 * rho[:M].sum()
        if M >= c * tau:
            break
    return max(tau, 1.0)


def split_rhat(chains):
    halves = [h for ch in chains for h in (ch[:len(ch)//2], ch[len(ch)//2:])]
    m = len(halves); n = min(len(h) for h in halves)
    halves = np.array([h[:n] for h in halves])
    W = halves.var(axis=1, ddof=1).mean()
    B = n * halves.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W)) if W > 0 else 1.0


def _job(task):
    kind, ev_or_none, n_sweeps, seed = task
    return tilt_trace(ev_or_none, n_sweeps, seed)


def main():
    rows = []
    for key in DATASETS:
        ev = R.load_raw(key)
        tokens = sum(len(Rr) for _s, Rr in ev)
        n_sweeps = SWEEPS_OVERRIDE or max(1000, int(np.ceil(2_000_000 / tokens)))
        MT, MR = R.MT, R.MR
        phi_real = MT._build_state(ev, "r2")[2]

        t0 = time.time()
        fills = [projection_random_start(ev, seed=1000 + i)
                 for i in range(max(2, N_CHAINS // 2))]
        for f in fills:
            assert MT.proj_sig(f) == MT.proj_sig(ev), "projection changed!"
            sz = lambda e: {s: sorted(len(x[1]) for x in g)
                            for s, g in pd.DataFrame(e, columns=["s", "R"])
                            .groupby("s")}
        phi_fill = [MT._build_state(f, "r2")[2] for f in fills]
        print(f"\n===== {key} ===== {len(ev):,} events, {tokens:,} tokens, "
              f"{n_sweeps} sweeps/chain", flush=True)
        print(f"  Phi real={phi_real:,.0f}  fill-starts={phi_fill[0]:,.0f}/"
              f"{phi_fill[1]:,.0f}  (fill built in {time.time()-t0:.1f}s)",
              flush=True)

        starts = [ev if i % 2 == 0 else fills[i // 2] for i in range(N_CHAINS)]
        tasks = [("c", starts[i], n_sweeps, 31 * i + 7) for i in range(N_CHAINS)]
        t0 = time.time()
        with ProcessPoolExecutor(min(6, N_CHAINS)) as ex:
            traces = list(ex.map(_job, tasks))
        dt = time.time() - t0

        burn = int(BURN_FRAC * n_sweeps)
        kept = [tr[burn:] for tr in traces]
        taus = [tau_int(k) for k in kept]
        tau = float(np.mean(taus))
        ess = sum(len(k) / t for k, t in zip(kept, taus))
        rhat = split_rhat(kept)
        allk = np.concatenate(kept)
        mu, sd = allk.mean() / len(ev), allk.std() / len(ev)
        mcse = sd / np.sqrt(ess)
        need = tau * 2000 / N_CHAINS + burn
        secs_per_sweep = dt / (N_CHAINS * n_sweeps)
        rows.append(dict(dataset=key, n_events=len(ev), tokens=tokens,
                         n_sweeps=n_sweeps, tau_int=tau,
                         tau_by_chain=";".join(f"{t:.1f}" for t in taus),
                         ESS=ess, rhat=rhat, null_r2=mu, null_sd=sd,
                         mcse=mcse, rel_mcse=mcse / mu,
                         sweeps_per_chain_for_ESS2000=need,
                         est_minutes_for_ESS2000=need * N_CHAINS
                         * secs_per_sweep / 60,
                         phi_real=phi_real, phi_fill=float(np.mean(phi_fill)),
                         wall_s=dt))
        print(f"  tau_int={tau:.1f} (per chain: {rows[-1]['tau_by_chain']})  "
              f"ESS={ess:,.0f}  split-Rhat={rhat:.4f}", flush=True)
        print(f"  null r2 = {mu:.5f} +- {mcse:.5f} (rel {mcse/mu:.2%}), "
              f"sd={sd:.5f}", flush=True)
        print(f"  for ESS>=2000: ~{need:,.0f} sweeps/chain "
              f"(~{rows[-1]['est_minutes_for_ESS2000']:.1f} min at "
              f"{N_CHAINS} chains)   [pilot took {dt:.0f}s]", flush=True)
        np.save(os.path.join(OUT, f"trace_{key}.npy"), np.array(kept))
        csvp = os.path.join(OUT, "ess_pilot.csv")
        older = pd.read_csv(csvp) if os.path.exists(csvp) else pd.DataFrame()
        if len(older):
            older = older[older.dataset != key]
        pd.concat([older, pd.DataFrame(rows[-1:])],
                  ignore_index=True).to_csv(csvp, index=False)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
