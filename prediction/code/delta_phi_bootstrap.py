#!/usr/bin/env python3
r"""Delta-Phi as a zero-parameter predictor, with uncertainty.

Score every causal row (s -> v at t) by dPhi = 2/(k-1) * #{j in T\{v}: m_j > m_v}
(team sender counts strictly before t) and evaluate on the SAME embargoed
test block as run_horizon_tables.py (default 50-20-30). Per horizon x label:

  AP(dPhi), prevalence, permutation no-skill AP (mean over 50 score shuffles),
  lift = AP - prevalence with a 95% event-clustered paired bootstrap CI
  (300 resamples of focal events), pooled and per size bucket; plus the
  event-level rotation label (any member leads the exact team within h).

Significant := CI of (AP - prevalence) excludes 0.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import link_pred_features as LPF
import link_prediction_expt as LPE
from delta_phi_scorer import delta_phi_frame

BUCKETS = [("k=3-5", 3, 5), ("k=6-10", 6, 10), ("k>=11", 11, 10**9)]
N_BOOT, N_PERM = 300, 50


def ap(y, s):
    return average_precision_score(y, s) if len(np.unique(y)) == 2 else np.nan


def boot(y, s, eid, k=None, n=N_BOOT, seed=11):
    """CI of (AP - prevalence), pooled and per bucket, same event resamples."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(eid, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    masks = ({b[0]: (k >= b[1]) & (k <= b[2]) for b in BUCKETS}
             if k is not None else {})
    d = {"pooled": [], **{b: [] for b in masks}}
    for _ in range(n):
        idx = np.concatenate([groups[g] for g in
                              rng.integers(len(groups), size=len(groups))])
        if len(np.unique(y[idx])) == 2:
            d["pooled"].append(ap(y[idx], s[idx]) - y[idx].mean())
        for b, m in masks.items():
            j = idx[m[idx]]
            if len(j) and len(np.unique(y[j])) == 2:
                d[b].append(ap(y[j], s[j]) - y[j].mean())
    return {b: (tuple(np.percentile(v, [2.5, 97.5])) if len(v) >= 20
                else (np.nan, np.nan)) for b, v in d.items()}


def perm_ap(y, s, n=N_PERM, seed=3):
    rng = np.random.default_rng(seed)
    return float(np.mean([ap(y, rng.permutation(s)) for _ in range(n)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizons", type=float, nargs="+", required=True)
    p.add_argument("--train-q", type=float, default=0.50)
    p.add_argument("--val-q", type=float, default=0.70)
    p.add_argument("--max-size", type=int, default=25)
    p.add_argument("--min-recipients", type=int, default=2)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--out-dir", default=os.path.join(HERE, "..", "results",
                                                     "delta_phi"))
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    events = LPF.load_dataset(a.dataset, max_size=a.max_size)
    unit = LPF.DATASETS[a.dataset]["step"]
    rows = pd.DataFrame(LPF.build_causal_rows(events, theta=a.theta))
    rows = rows.merge(delta_phi_frame(events), on=["eid", "v"], how="left",
                      validate="one_to_one")
    if a.min_recipients > 1:
        rows = rows[rows.k >= 1 + a.min_recipients].reset_index(drop=True)
    train_cut, val_cut = LPE.temporal_cutoffs(rows["t"].to_numpy(),
                                             a.train_q, a.val_q)
    t_end = events[-1][2]
    print(f"{a.dataset}: {len(rows):,} rows (built {time.time()-t0:.0f}s)",
          flush=True)
    out = []
    for h in a.horizons:
        fr = LPE.add_labels(rows, h * unit, t_end)
        tr, va, te = LPE.temporal_masks(fr, h * unit, train_cut, val_cut)
        for label in ("y_dyad", "y_group_exact", "y_mode"):
            _, _, te_l = LPE.label_population(fr, label, tr, va, te)
            sub = fr[te_l]
            y, s = sub[label].to_numpy(), sub["dphi"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            k, eid = sub["k"].to_numpy(), sub["eid"].to_numpy()
            cis = boot(y, s, eid, k=k)
            row = dict(dataset=a.dataset, horizon=h, label=label, unit="recipient",
                       n_test=len(sub), prevalence=float(y.mean()),
                       ap=ap(y, s), roc=roc_auc_score(y, s),
                       perm_ap=perm_ap(y, s))
            row["lift"] = row["ap"] - row["prevalence"]
            row["lift_ci_low"], row["lift_ci_high"] = cis["pooled"]
            for b, lo_k, hi_k in BUCKETS:
                m = (k >= lo_k) & (k <= hi_k)
                row[f"lift_{b}"] = (ap(y[m], s[m]) - y[m].mean()
                                    if m.sum() and len(np.unique(y[m])) == 2 else np.nan)
                row[f"lift_{b}_ci_low"], row[f"lift_{b}_ci_high"] = cis[b]
            out.append(row)
            star = "*" if row["lift_ci_low"] > 0 else " "
            print(f"  h={h:g} {label:14s} n={len(sub):7,d} prev={row['prevalence']:.3f} "
                  f"perm={row['perm_ap']:.3f} AP={row['ap']:.3f} lift={row['lift']:+.3f}{star} "
                  f"[{row['lift_ci_low']:+.3f},{row['lift_ci_high']:+.3f}] ROC={row['roc']:.3f}  "
                  + " ".join(f"{b[0]}:{row['lift_'+b[0]]:+.3f}"
                             + ("*" if row['lift_'+b[0]+'_ci_low'] > 0 else "")
                             for b in BUCKETS), flush=True)
        # event-level rotation label
        sub = fr[te]
        ev = sub.groupby("eid").agg(y=("y_group_exact", "max"),
                                    s=("dphi", "max"), k=("k", "first"))
        y, s, k = ev.y.to_numpy(), ev.s.to_numpy(), ev.k.to_numpy()
        if len(np.unique(y)) == 2:
            cis = boot(y, s, ev.index.to_numpy(), k=k)
            row = dict(dataset=a.dataset, horizon=h, label="y_rotation",
                       unit="event", n_test=len(ev), prevalence=float(y.mean()),
                       ap=ap(y, s), roc=roc_auc_score(y, s), perm_ap=perm_ap(y, s))
            row["lift"] = row["ap"] - row["prevalence"]
            row["lift_ci_low"], row["lift_ci_high"] = cis["pooled"]
            for b, lo_k, hi_k in BUCKETS:
                m = (k >= lo_k) & (k <= hi_k)
                row[f"lift_{b}"] = (ap(y[m], s[m]) - y[m].mean()
                                    if m.sum() and len(np.unique(y[m])) == 2 else np.nan)
                row[f"lift_{b}_ci_low"], row[f"lift_{b}_ci_high"] = cis[b]
            out.append(row)
            star = "*" if row["lift_ci_low"] > 0 else " "
            print(f"  h={h:g} {'y_rotation':14s} n={len(ev):7,d} prev={row['prevalence']:.3f} "
                  f"perm={row['perm_ap']:.3f} AP={row['ap']:.3f} lift={row['lift']:+.3f}{star} "
                  f"[{row['lift_ci_low']:+.3f},{row['lift_ci_high']:+.3f}] ROC={row['roc']:.3f}",
                  flush=True)
        pd.DataFrame(out).to_csv(os.path.join(a.out_dir, f"{a.dataset}_delta_phi.csv"),
                                 index=False)
    print(f"DONE {a.dataset} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

# How this was run (2026-09-21):
#   cd Hypergraph_GroupReciprocity/prediction/code
#   <venv-python> delta_phi_bootstrap.py --dataset emaileu --horizons 1 7 30
#   <venv-python> delta_phi_bootstrap.py --dataset enron   --horizons 1 7 30
#   <venv-python> delta_phi_bootstrap.py --dataset dnc     --horizons 1 3 5 7
#   <venv-python> delta_phi_bootstrap.py --dataset twitter --horizons 7 30 60
# (defaults: 50-20-30 split, k<=25, min 2 recipients; plain numpy/pandas/sklearn venv)
# Output: ../results/delta_phi/<dataset>_delta_phi.csv
