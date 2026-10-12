#!/usr/bin/env python3
"""Diagnostics + summary tables for the final chains, implementing the
acceptance rules of results/final_expt_plan.md.

Per (dataset, beta, filter): for every measure -- split-Rhat over chains
(each split in half), Sokal tau_int, pooled ESS, MCSE, start-state agreement
(observed vs fill post-burn-in means vs combined MCSE) -- and the summary:
observed value, null mean, 2.5/97.5 percentiles of retained draws, diff,
ratio, and whether the observed value exceeds every retained draw.
Burn-in: 25% of each chain. Writes diagnostics.csv, summary.csv,
chains.csv (metadata) and prints the extension recommendations.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# reported measures: multiplicity-aware for the main (multihypergraph-start)
# chains; M-weighted support measures for the dedup-start support analysis
SETS = {"multi": ("r2_multi", ("s1_multi", "r1_multi", "r2_multi")),
        "support": ("r2_support_M", ("s1_support_M", "r1_support_M", "r2_support_M"))}
RHAT_MAX, ESS_MIN_A, ESS_MIN_B = 1.01, 2000, 1000


def tau_int(x, c=5.0):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    v = np.dot(x, x) / n
    if v == 0 or n < 10:
        return 1.0
    tau = 1.0
    for M in range(1, n // 3):
        rho = np.dot(x[:n - M], x[M:]) / ((n - M) * v)
        tau += 2.0 * rho
        if M >= c * tau:
            break
    return max(tau, 1.0)


def split_rhat(chains):
    halves = [h for ch in chains for h in (ch[:len(ch)//2], ch[len(ch)//2:])]
    n = min(len(h) for h in halves)
    H = np.array([h[:n] for h in halves])
    W = H.var(axis=1, ddof=1).mean(); B = n * H.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W)) if W > 0 else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "final"))
    p.add_argument("--burn", type=float, default=0.25)
    a = p.parse_args()
    root = os.path.abspath(a.out)
    metas, diag, summ = [], [], []
    for mj in sorted(glob.glob(os.path.join(root, "*", "*", "chain*", "meta.json"))):
        m = json.load(open(mj)); m["dir"] = os.path.dirname(mj); metas.append(m)
    cm = pd.DataFrame(metas)
    if not len(cm):
        raise SystemExit("no completed chains found")
    cm["combo"] = cm.dir.map(lambda d: os.path.relpath(os.path.dirname(d), root))
    cm.drop(columns=["observed"]).to_csv(os.path.join(root, "chains.csv"), index=False)
    mpath = os.path.join(root, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    recs = []
    for combo in sorted(set(cm.combo) | set(manifest)):
        g = cm[cm.combo == combo]
        exp = manifest.get(combo)
        if exp is None:
            print(f"  WARNING {combo}: no manifest entry (not launched by run_final?)")
        else:
            ok = g[(g.status == "complete") & (g.sweeps >= exp["sweeps"])]
            if len(ok) < exp["n_chains"]:
                diag.append(dict(combo=combo, n_chains=len(ok), accepted=False,
                                 fail_reasons=f"INCOMPLETE {len(ok)}/{exp['n_chains']} "
                                              f"chains at >= {exp['sweeps']} sweeps"))
                recs.append((combo, [diag[-1]["fail_reasons"]]))
                continue
            g = ok
        ds, beta, filt = g.dataset.iloc[0], g.beta.iloc[0], g["filter"].iloc[0]
        obs = g.observed.iloc[0]
        PRIMARY, REPORTED = SETS["support" if "dedup" in filt else "multi"]
        traces = {r.init: [] for r in g.itertuples()}
        all_tr = []
        for r in g.itertuples():
            tr = pd.read_csv(os.path.join(r.dir, "trace.csv"))
            tr = tr.iloc[int(a.burn * len(tr)):]
            traces[r.init].append(tr); all_tr.append(tr)
        cols = [c for c in all_tr[0].columns if c not in ("sweep",)]
        row_d = dict(combo=combo, dataset=ds, beta=beta, filter=filt,
                     n_chains=len(all_tr), sweeps=int(g.sweeps.min()),
                     accept_rate=float(g.accept_rate.mean()))
        fails = []
        for col in cols:
            ch = [t[col].to_numpy() for t in all_tr]
            taus = [tau_int(c_) for c_ in ch]
            ess = sum(len(c_) / t for c_, t in zip(ch, taus))
            rhat = split_rhat(ch)
            pooled = np.concatenate(ch)
            mcse = pooled.std() / np.sqrt(ess) if ess else np.nan
            row_d.update({f"{col}_rhat": rhat, f"{col}_tau": float(np.mean(taus)),
                          f"{col}_ess": ess, f"{col}_mcse": mcse})
            # start-state agreement
            if "observed" in traces and "fill" in traces:
                mo = np.mean([t[col].mean() for t in traces["observed"]])
                mf = np.mean([t[col].mean() for t in traces["fill"]])
                row_d[f"{col}_start_gap_in_mcse"] = (abs(mo - mf) / (2 * mcse)
                                                     if mcse else np.nan)
            if col in REPORTED or col == "Phi":
                oval = obs.get(col) if col != "Phi" else None
                r = dict(combo=combo, dataset=ds, beta=beta, filter=filt, measure=col,
                         observed=oval, null_mean=pooled.mean(), null_sd=pooled.std(),
                         q025=np.percentile(pooled, 2.5), q975=np.percentile(pooled, 97.5),
                         rhat=rhat, ess=ess, mcse=mcse)
                if oval is not None:
                    r["diff"] = oval - r["null_mean"]
                    r["ratio"] = oval / r["null_mean"] if r["null_mean"] else np.nan
                    r["obs_exceeds_all_draws"] = bool(oval > pooled.max())
                summ.append(r)
            if col in REPORTED and rhat > RHAT_MAX:
                fails.append(f"{col} Rhat={rhat:.3f}")
        ess_min = ESS_MIN_A if beta == 0 else ESS_MIN_B
        key = PRIMARY if beta == 0 else "Phi"
        if row_d[f"{key}_ess"] < ess_min:
            fails.append(f"{key} ESS={row_d[f'{key}_ess']:.0f}<{ess_min}")
        sg = row_d.get(f"{key}_start_gap_in_mcse")
        if sg is not None and sg > 1:
            fails.append(f"start-state gap {sg:.1f}x MCSE")
        row_d["accepted"] = not fails; row_d["fail_reasons"] = "; ".join(fails)
        diag.append(row_d)
        recs.append((combo, fails))
    pd.DataFrame(diag).to_csv(os.path.join(root, "diagnostics.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(root, "summary.csv"), index=False)
    print(f"{len(diag)} combos; accepted {sum(1 for _, f in recs if not f)}")
    for combo, fails in recs:
        if fails:
            print(f"  EXTEND {combo}: " + "; ".join(fails))
    s = pd.DataFrame(summ)
    s0 = s[(s.beta == 0) & s.measure.isin(["r2_multi", "r2_support_M"])
           & ((s.measure == "r2_multi") != s["filter"].str.contains("dedup"))]
    if len(s0):
        print("\nbeta=0 partial measure -- R_part^m (multi start) or R_part^M (dedup start):"
              "\n  observed / null mean / ratio / exceeds all draws")
        for r in s0.itertuples():
            print(f"  {r.dataset:9s} {r.filter:12s} {r.observed:.4f} / {r.null_mean:.5f} "
                  f"/ {r.ratio:6.2f}x / {r.obs_exceeds_all_draws}  "
                  f"(Rhat {r.rhat:.4f}, ESS {r.ess:,.0f})")


if __name__ == "__main__":
    main()

# How to run:  python diagnose_final.py --out ../../results/final
# Outputs in that dir: chains.csv, diagnostics.csv, summary.csv; EXTEND lines
# name the (dataset/beta/filter) combos failing the plan's acceptance rules.
