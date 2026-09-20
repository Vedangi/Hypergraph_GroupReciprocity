#!/usr/bin/env python3
"""Horizon x size-bucket prediction tables (Table-14 style).

For one dataset and a list of horizons: build the causal rows once, then for
each horizon, label (y_dyad / y_group_exact / y_mode) and model (LogReg,
LightGBM) fit tier A (20 graph features) and tier C (A + 5 hypergraph
features) on train+val and score the test block once. Reports pooled PR-A,
PR-C, dPR with an event-clustered paired bootstrap CI, and dPR within team-
size buckets k = 3-5, 6-10, >= 11. Split is configurable (default 50-20-30,
both boundaries embargoed by the horizon). Fixed curated features -- no
validation-based selection -- so rows are comparable across horizons.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import link_pred_features as LPF
import link_prediction_expt as LPE

BUCKETS = [("k=3-5", 3, 5), ("k=6-10", 6, 10), ("k>=11", 11, 10**9)]
LABELS = ("y_dyad", "y_group_exact", "y_mode")
MODELS = ("LogReg", "LightGBM")
N_BOOT = 300


def fit_score(Xtr, ytr, Xte, model_name):
    model, scale = (LPE.make_logreg() if model_name == "LogReg"
                    else LPE.make_lgbm())
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def ap(y, s):
    return average_precision_score(y, s) if len(np.unique(y)) == 2 else np.nan


def cluster_boot_ci(y, sA, sC, eid, n=N_BOOT, seed=11):
    """Paired bootstrap resampling whole focal events."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(eid, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(uniq))]
    deltas = []
    for _ in range(n):
        pick = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(ap(y[idx], sC[idx]) - ap(y[idx], sA[idx]))
    return (np.percentile(deltas, [2.5, 97.5]) if deltas
            else (np.nan, np.nan))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizons", type=float, nargs="+", required=True)
    p.add_argument("--train-q", type=float, default=0.50)
    p.add_argument("--val-q", type=float, default=0.70)
    p.add_argument("--max-size", type=int, default=25)
    p.add_argument("--min-recipients", type=int, default=2)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    t0 = time.time()
    events = LPF.load_dataset(a.dataset, max_size=a.max_size)
    unit = LPF.DATASETS[a.dataset]["step"]
    rows = pd.DataFrame(LPF.build_causal_rows(events, theta=a.theta))
    if a.min_recipients > 1:
        rows = rows[rows.k >= 1 + a.min_recipients].reset_index(drop=True)
    train_cut, val_cut = LPE.temporal_cutoffs(rows["t"].to_numpy(),
                                             a.train_q, a.val_q)
    t_end = events[-1][2]
    A = list(LPF.GRAPH_FEATURES)
    C = A + list(LPF.HYPERGRAPH_FEATURES)
    print(f"{a.dataset}: {len(events):,} events, {len(rows):,} rows, "
          f"split {a.train_q:.2f}/{a.val_q-a.train_q:.2f}/{1-a.val_q:.2f}, "
          f"|A|={len(A)} |C|={len(C)}  (built in {time.time()-t0:.0f}s)",
          flush=True)

    out = []
    for h in a.horizons:
        horizon = h * unit
        frame = LPE.add_labels(rows, horizon, t_end)
        tr, va, te = LPE.temporal_masks(frame, horizon, train_cut, val_cut)
        for label in LABELS:
            tr_l, va_l, te_l = LPE.label_population(frame, label, tr, va, te)
            fit = tr_l | va_l                    # untuned: refit on train+val
            y_tr = frame.loc[fit, label].to_numpy()
            sub = frame[te_l]
            y_te = sub[label].to_numpy()
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            k_te = sub["k"].to_numpy()
            eid = sub["eid"].to_numpy()
            for mname in MODELS:
                t1 = time.time()
                sA = fit_score(frame.loc[fit, A].to_numpy(), y_tr,
                               sub[A].to_numpy(), mname)
                sC = fit_score(frame.loc[fit, C].to_numpy(), y_tr,
                               sub[C].to_numpy(), mname)
                apA, apC = ap(y_te, sA), ap(y_te, sC)
                lo, hi = cluster_boot_ci(y_te, sA, sC, eid)
                row = dict(dataset=a.dataset, horizon=h, label=label,
                           model=mname, n_train=int(fit.sum()),
                           n_test=len(sub), rate=float(y_te.mean()),
                           PR_A=apA, PR_C=apC, dPR=apC - apA,
                           dPR_ci_low=lo, dPR_ci_high=hi,
                           ROC_A=roc_auc_score(y_te, sA),
                           ROC_C=roc_auc_score(y_te, sC))
                for bname, lo_k, hi_k in BUCKETS:
                    m = (k_te >= lo_k) & (k_te <= hi_k)
                    row[f"n_{bname}"] = int(m.sum())
                    row[f"dPR_{bname}"] = (ap(y_te[m], sC[m]) - ap(y_te[m], sA[m])
                                           if m.sum() and len(np.unique(y_te[m])) == 2
                                           else np.nan)
                out.append(row)
                print(f"  h={h:g} {label:14s} {mname:8s} n={len(sub):7,d} "
                      f"rate={row['rate']:.3f} PR-A={apA:.3f} PR-C={apC:.3f} "
                      f"dPR={apC-apA:+.3f} [{lo:+.3f},{hi:+.3f}]  "
                      + "  ".join(f"{b[0]}:{row['dPR_'+b[0]]:+.3f}"
                                  for b in BUCKETS)
                      + f"  ({time.time()-t1:.0f}s)", flush=True)
        pd.DataFrame(out).to_csv(
            os.path.join(a.out_dir, f"{a.dataset}_horizon_tables.csv"),
            index=False)

    # markdown tables, one per model, Table-14 layout
    df = pd.DataFrame(out)
    md = [f"# {a.dataset}: split {a.train_q:.2f}/{a.val_q-a.train_q:.2f}/"
          f"{1-a.val_q:.2f}, fixed features (|A|={len(A)}, |C|={len(C)})\n"]
    for mname in MODELS:
        md.append(f"\n## {mname}\n")
        for label in LABELS:
            s = df[(df.model == mname) & (df.label == label)]
            if not len(s):
                continue
            md.append(f"\n**{label}**\n\n| h | n_test | rate | PR-A | PR-C | ΔPR "
                      "[95% CI] | k=3–5 | k=6–10 | k≥11 |\n|---|---|---|---|---|---|---|---|---|")
            for _, r in s.iterrows():
                md.append(f"| {r.horizon:g} | {r.n_test:,} | {r.rate:.3f} | "
                          f"{r.PR_A:.3f} | {r.PR_C:.3f} | {r.dPR:+.3f} "
                          f"[{r.dPR_ci_low:+.3f}, {r.dPR_ci_high:+.3f}] | "
                          f"{r['dPR_k=3-5']:+.3f} | {r['dPR_k=6-10']:+.3f} | "
                          f"{r['dPR_k>=11']:+.3f} |")
    with open(os.path.join(a.out_dir, f"{a.dataset}_horizon_tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nDONE {a.dataset} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

# How this was run (2026-09-20) -- needs a venv with lightgbm
# (locally Reciprocity/code/.venv/bin/python):
#   cd Hypergraph_GroupReciprocity/prediction/code
#   <lgbm-python> run_horizon_tables.py --dataset emaileu --horizons 1 7 30 \
#       --train-q 0.5 --val-q 0.7 --out-dir ../results/split_50_20_30
#   <lgbm-python> run_horizon_tables.py --dataset enron   --horizons 1 7 30  (same flags)
#   <lgbm-python> run_horizon_tables.py --dataset dnc     --horizons 1 3 7   (same flags)
#   twitter: --horizons 7 30 60  (pending hypergraph-construction decision)
# Outputs: <out-dir>/<dataset>_horizon_tables.{csv,md}
