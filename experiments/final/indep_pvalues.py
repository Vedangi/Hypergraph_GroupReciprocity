#!/usr/bin/env python3
"""DiNgHy/NuDHy-style summary of the phase-I chains (33 independent chains
from the observed hypergraph, 20 x bipartite-edge proposals each, terminal
state only). For each dataset and measure: observed value, median / min /
max / mean / sd of the 33 terminal states, the median-centred two-sided
empirical p-value used by DiNgHy (plot_p_values.py):
    p = #{ samples : |x - median| >= |observed - median| } / 33,
the one-sided p = #{ samples : x >= observed } / 33, and, for comparison,
the phase-A null mean from summary.csv. -> results/final/indep33.{csv,md}
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from make_main_outputs import ORDER  # noqa: E402
MEAS = [("s1_multi", "R_any"), ("r2_multi", "R_part"), ("r1_multi", "R_all")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "final"))
    a = p.parse_args()
    root = os.path.abspath(a.out)
    sp = os.path.join(root, "summary.csv")          # phase-A null means, if available
    S = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame(columns=["dataset", "measure", "beta", "filter", "null_mean"])
    S = S[(S.beta == 0) & (S["filter"] == "k3-25")].set_index(["dataset", "measure"])
    rows = []
    for key, name, dom in ORDER:
        chains = sorted(glob.glob(os.path.join(root, key, "beta+0.0_indep33", "chain*", "trace.csv")))
        if not chains:
            continue
        term, obs, sweeps = [], None, set()
        for tr in chains:
            meta = json.load(open(os.path.join(os.path.dirname(tr), "meta.json")))
            if meta["status"] != "complete":
                continue
            t = pd.read_csv(tr)
            term.append(t.iloc[-1]); obs = meta["observed"]; sweeps.add(int(t.sweep.iloc[-1]))
        T = pd.DataFrame(term)
        for m, lab in MEAS:
            x = T[m].to_numpy(float); o = obs[m]; med = np.median(x)
            rows.append(dict(dataset=name, measure=lab, n_chains=len(x), sweeps=max(sweeps),
                             observed=o, median=med, min=x.min(), max=x.max(), mean=x.mean(),
                             sd=x.std(ddof=1), p_two_sided=np.mean(np.abs(x - med) >= abs(o - med)),
                             p_one_sided=np.mean(x >= o),
                             phaseA_null_mean=S.loc[(key, m), "null_mean"] if (key, m) in S.index else np.nan))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(root, "indep33.csv"), index=False, float_format="%.6g")
    md = ["# DiNgHy-style replication: 33 independent chains from the observed state, terminal states only", "",
          "| dataset | measure | chains | sweeps | observed | median | min | max | mean | sd | p (two-sided) | p (one-sided) | phase-A null mean |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in D.itertuples():
        md.append(f"| {r.dataset} | {r.measure} | {r.n_chains} | {r.sweeps} | {r.observed:.4g} | {r.median:.4g} | "
                  f"{r.min:.4g} | {r.max:.4g} | {r.mean:.4g} | {r.sd:.2g} | {r.p_two_sided:.2f} | "
                  f"{r.p_one_sided:.2f} | {r.phaseA_null_mean:.4g} |")
    md.append("\np (two-sided): median-centred empirical p-value as in DiNgHy; resolution 1/33 ≈ 0.03, "
              "so 0.00 means the observed value is farther from the median than every one of the 33 samples.")
    open(os.path.join(root, "indep33.md"), "w").write("\n".join(md) + "\n")
    print(D[["dataset", "measure", "n_chains", "sweeps", "observed", "median", "max", "p_two_sided", "phaseA_null_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("[saved]", os.path.join(root, "indep33.{csv,md}"))


if __name__ == "__main__":
    main()

# How this was run (2026-10-16):
#   <venv-python> run_final.py --phase I --workers 6     (chains; ~2 h on a laptop)
#   <venv-python> indep_pvalues.py
# Render-only summary: reads results/final/<ds>/beta+0.0_indep33/chain*/ and summary.csv.
