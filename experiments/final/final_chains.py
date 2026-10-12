#!/usr/bin/env python3
r"""Traced, checkpointed Metropolis chains for the final structural experiments
(results/final_expt_plan.md).

State space: event-labelled regroupings with every sender's recipient
multiset (the weighted projection) and every event's size fixed; proposal =
within-sender 2x2 switch; one sweep = L = sum_i |R_i| proposal attempts;
target P_beta(H) ~ exp(beta * Phi(H)), Phi = |E| * R_part^m (= r2_multi).

Every completed sweep (including sweeps with no accepted move) records a
state: Phi plus all three multiplicity-aware and all three support measures,
overall and within the size bins k=3-5 / 6-10 / 11-25.  Chains checkpoint
(state, RNG, counters) so a run can be extended to more sweeps later.
"""
from __future__ import annotations
import json, os, pickle, sys, time, hashlib
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in ("src", "src/multihypergraph"):
    sys.path.insert(0, os.path.join(ROOT, p))
import datasets                      # noqa: E402
import multi_reciprocity as MR       # noqa: E402
import multi_tilt as MT              # noqa: E402

BINS = (("k3_5", 3, 5), ("k6_10", 6, 10), ("k11_25", 11, 10**9))
MULTI = ("s1_multi", "r1_multi", "r2_multi")          # R_any^m, R_all^m, R_part^m
SUPP = ("s1_support", "r1_support", "r2_support")  # p-weighted: support of the state
# M-weighted support measures (the paper's definition): same per-team credit
# f(p_T), weighted by M_T with denominator |E|; identical on duplicate-free states.
SUPP_M = ("s1_support_M", "r1_support_M", "r2_support_M")
ALL_M = MULTI + SUPP + SUPP_M
MEASURE_COLS = ([f"{m}" for m in ALL_M]
                + [f"{m}_{b}" for b, _, _ in BINS for m in ALL_M])
CHECKPOINT_EVERY = 25


# ------------------------------------------------------------------ data
def load_events(dataset, min_k=3, max_k=25, dedup=False):
    """Group events (sender, sorted recipients) with min_k <= k <= max_k;
    self-loops and recipient repeats removed. Dyadic events are excluded.
    dedup=True keeps each distinct (sender, recipient set) once -- the support
    hypergraph E*, the start state of the support analysis. The chain itself
    never deduplicates: repeats that form during sampling are kept and the
    M-weighted support measures count them (denominator stays |E*|)."""
    out = []
    for s, R, _t in datasets.load_dataset(dataset):
        R = sorted(set(int(x) for x in R) - {int(s)})
        if min_k <= 1 + len(R) <= max_k:
            out.append((int(s), R))
    if dedup:
        out = sorted({(s, tuple(R)) for s, R in out})
        out = [(s, list(R)) for s, R in out]
    return out


def projection_random_start(events, seed):
    """Chain-free randomized state with the same senders, per-event sizes and
    weighted projection (randomized Gale-Ryser fill, greedy fallback)."""
    rng = np.random.default_rng(seed)
    by_s = defaultdict(list)
    for s, R in events:
        by_s[s].append(list(R))
    out = []
    for s, evl in by_s.items():
        sizes = [len(x) for x in evl]
        pool = Counter(r for x in evl for r in x)
        order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
        fill = None
        for _ in range(25):
            rem, res, ok = dict(pool), {}, True
            for i in order:
                cand = [r for r, c in rem.items() if c > 0]
                if len(cand) < sizes[i]:
                    ok = False; break
                w = np.array([rem[r] for r in cand], float)
                take = [cand[t] for t in rng.choice(len(cand), size=sizes[i],
                                                    replace=False, p=w / w.sum())]
                for r in take:
                    rem[r] -= 1
                res[i] = take
            if ok:
                fill = [set(res[i]) for i in range(len(sizes))]; break
        if fill is None:                                  # greedy, always feasible
            rem, res = dict(pool), {}
            for i in order:
                cand = sorted((r for r, c in rem.items() if c > 0),
                              key=lambda r: (-rem[r], rng.random()))
                take = cand[:sizes[i]]
                for r in take:
                    rem[r] -= 1
                res[i] = take
            fill = [set(res[i]) for i in range(len(sizes))]
        out.extend((s, sorted(f)) for f in fill)
    return out


# ------------------------------------------------------------------ measures
def snapshot(evs):
    """All measures from the sampler's state dict, overall and per size bin
    (event-weighted multi, support-edge-weighted support; = MR.measures)."""
    events = [(s, sorted(e)) for s, lst in evs.items() for e in lst]
    prof = MR.team_profiles(events)
    num = defaultdict(float); den = defaultdict(float)
    for T, cnt in prof.items():
        k = len(T)
        if k < 3:
            continue
        ts = MR.team_scores(T, cnt)
        M, q = ts["M"], ts["q"]
        tags = ["", *[f"_{b}" for b, lo, hi in BINS if lo <= k <= hi]]
        for tag in tags:
            den["M" + tag] += M; den["q" + tag] += q
            for m in MULTI:
                num[m + tag] += ts[m] * M
            for m in SUPP:
                num[m + tag] += ts[m] * q               # p-weighted
                num[m + "_M" + tag] += ts[m] * M        # M-weighted (paper)
    row = {}
    for col in MEASURE_COLS:
        base, tag = (col.split("_k", 1) + [""])[:2]
        tag = ("_k" + tag) if tag else ""
        w = "q" if (base.endswith("_support")) else "M"   # multi and *_support_M use |E|
        d = den[w + tag]
        row[col] = num[col] / d if d else float("nan")
    return row


# ------------------------------------------------------------------ chain
def chain_seed(dataset, beta, chain_id, filt):
    h = hashlib.sha256(f"{dataset}|{beta:+.3f}|{chain_id}|{filt}".encode()).hexdigest()
    return int(h[:8], 16)


def run_chain(task):
    """task: dict(dataset, beta, chain_id, init, sweeps, out_dir, min_k, max_k)
    Resumes from <out_dir>/checkpoint.pkl if present; extends to task['sweeps'].
    Writes trace.csv (one row per sweep incl. sweep 0), meta.json, checkpoint."""
    d = task; out = d["out_dir"]; os.makedirs(out, exist_ok=True)
    filt = f"k{d['min_k']}-{d['max_k']}" + ("-dedup" if d.get("dedup") else "")
    seed = chain_seed(d["dataset"], d["beta"], d["chain_id"], filt)
    ck = os.path.join(out, "checkpoint.pkl")
    t0 = time.time()
    if os.path.exists(ck):
        with open(ck, "rb") as f:
            st = pickle.load(f)
        if st["sweep"] >= d["sweeps"]:
            return _finish(out, st, t0, resumed=True)
        evs, tsc, Phi = st["evs"], st["tsc"], st["Phi"]
        rng = np.random.default_rng(); rng.bit_generator.state = st["rng"]
        sweep0, n_prop, n_acc, n_id = st["sweep"], st["n_prop"], st["n_acc"], st["n_id"]
        tokens = st["tokens"]; n_events = st["n_events"]; obs = st["obs"]
        mode = "a"
    else:
        ev = load_events(d["dataset"], d["min_k"], d["max_k"], d.get("dedup", False))
        obs = snapshot(_evs_of(ev))
        start = ev if d["init"] == "observed" else projection_random_start(ev, seed ^ 0xA5A5)
        assert MT.proj_sig(start) == MT.proj_sig(ev)
        evs, tsc, Phi, _ = MT._build_state(start, "r2")
        rng = np.random.default_rng(seed)
        sweep0 = 0; n_prop = n_acc = n_id = 0
        tokens = sum(len(x) for e in evs.values() for x in e); n_events = len(ev)
        mode = "w"
    swap = [s for s, e in evs.items() if len(e) >= 2]
    beta = d["beta"]
    with open(os.path.join(out, "trace.csv"), mode) as tr:
        if mode == "w":
            tr.write("sweep,Phi," + ",".join(MEASURE_COLS) + "\n")
            _write_row(tr, 0, Phi, snapshot(evs))
        for sw in range(sweep0 + 1, d["sweeps"] + 1):
            for _ in range(tokens):
                s = swap[rng.integers(len(swap))]
                lst = evs[s]; m = len(lst)
                i = int(rng.integers(m)); j = int(rng.integers(m))
                if i == j:
                    continue
                A = sorted(lst[i] - lst[j]); B = sorted(lst[j] - lst[i])  # deterministic across pickling
                if not A or not B:
                    continue
                if len(A) == 1 and len(B) == 1:
                    n_id += 1; continue
                a = A[rng.integers(len(A))]; b = B[rng.integers(len(B))]
                n_prop += 1
                dPhi, _ = MT._apply_switch(evs, tsc, s, i, j, a, b, "r2", None)
                if beta * dPhi >= 0 or rng.random() < np.exp(beta * dPhi):
                    Phi += dPhi; n_acc += 1
                else:
                    MT._apply_switch(evs, tsc, s, i, j, b, a, "r2", None)
            _write_row(tr, sw, Phi, snapshot(evs))
            if sw % CHECKPOINT_EVERY == 0 or sw == d["sweeps"]:
                tr.flush()
                st = dict(evs=evs, tsc=tsc, Phi=Phi, rng=rng.bit_generator.state,
                          sweep=sw, n_prop=n_prop, n_acc=n_acc, n_id=n_id,
                          tokens=tokens, n_events=n_events, obs=obs, seed=seed,
                          task=d)
                with open(ck + ".tmp", "wb") as f:
                    pickle.dump(st, f)
                os.replace(ck + ".tmp", ck)
    return _finish(out, st, t0, resumed=(mode == "a"))


def _evs_of(ev):
    evs = defaultdict(list)
    for s, R in ev:
        evs[s].append(set(R))
    return evs


def _write_row(tr, sw, Phi, snap):
    tr.write(f"{sw},{Phi:.6f}," + ",".join(f"{snap[c]:.8f}" for c in MEASURE_COLS) + "\n")


def _finish(out, st, t0, resumed):
    d = st["task"]
    meta = dict(dataset=d["dataset"], beta=d["beta"], chain_id=d["chain_id"],
                init=d["init"], filter=f"k{d['min_k']}-{d['max_k']}"
                + ("-dedup" if d.get("dedup") else ""), dedup_start=bool(d.get("dedup")),
                seed=st["seed"], sweeps=st["sweep"], tokens=st["tokens"],
                n_events=st["n_events"], n_proposed=st["n_prop"],
                n_accepted=st["n_acc"], n_identity_skips=st["n_id"],
                accept_rate=(st["n_acc"] / st["n_prop"] if st["n_prop"] else None),
                move_rate=st["n_acc"] / (st["sweep"] * st["tokens"]) if st["sweep"] else None,
                observed=st["obs"], wall_s_last_segment=time.time() - t0,
                resumed=resumed, status="complete")
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    import tempfile, filecmp, shutil
    tmp = tempfile.mkdtemp()
    base = dict(dataset="fauci", beta=0.0, chain_id=0, init="observed",
                min_k=3, max_k=25)
    # 1) fixed seed -> identical traces
    run_chain({**base, "sweeps": 12, "out_dir": f"{tmp}/a"})
    run_chain({**base, "sweeps": 12, "out_dir": f"{tmp}/b"})
    assert filecmp.cmp(f"{tmp}/a/trace.csv", f"{tmp}/b/trace.csv", shallow=False)
    # 2) resume: 6 then extend to 12 == straight 12
    run_chain({**base, "sweeps": 6, "out_dir": f"{tmp}/c"})
    run_chain({**base, "sweeps": 12, "out_dir": f"{tmp}/c"})
    assert filecmp.cmp(f"{tmp}/a/trace.csv", f"{tmp}/c/trace.csv", shallow=False)
    # 3) snapshot == MR.measures on the observed state
    ev = load_events("fauci")
    snap = snapshot(_evs_of(ev)); ref = MR.measures(ev, min_k=3)
    for m in MULTI + SUPP:
        assert abs(snap[m] - ref[m]) < 1e-12, m
    sys.path.insert(0, os.path.join(ROOT, "src", "dedup"))
    import support_reciprocity as SR
    sm = SR.support_measures_team(ev, min_k=3, weighting="M")
    for m, k in (("s1_support_M", "R_any"), ("r2_support_M", "R_part"), ("r1_support_M", "R_all")):
        assert abs(snap[m] - sm[k]) < 1e-12, m
    # 4) dedup start: at sweep 0 every M_T = 1, so M- and p-weighted support agree
    dd = load_events("fauci", dedup=True)
    sd = snapshot(_evs_of(dd))
    assert len(dd) == len({(s, tuple(R)) for s, R in dd}) < len(ev)
    for m in SUPP:
        assert abs(sd[m] - sd[m + "_M"]) < 1e-12, m
    run_chain({**base, "dedup": True, "sweeps": 3, "out_dir": f"{tmp}/e"})
    # 5) fill start preserves projection; beta != 0 runs
    run_chain({**base, "beta": 2.0, "init": "fill", "sweeps": 3, "out_dir": f"{tmp}/d"})
    shutil.rmtree(tmp)
    print("final_chains selftest OK: reproducible seeds, resume == straight run, "
          "snapshot == MR.measures, dedup start (M==p at sweep 0), fill start + beta>0 run")
