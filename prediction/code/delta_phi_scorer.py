#!/usr/bin/env python3
r"""Delta-Phi scorer (repo copy; used for delta_phi_frame): the paper's exponential model used directly as a predictor.

For each causal row (s -> v at t) the score is
    dPhi(v | T, t) = 2/(k-1) * #{ j in T \ {v} : m_{T,j} > m_{T,v} }
computed from team sender-counts over events STRICTLY before t (same-timestamp
batching identical to build_causal_rows), i.e. the increase in Phi = |E|*R_some
if v led a reply to team T.  Zero training; ranking is theta-free.

Evaluated on the identical test rows / labels / embargoed split as
link_prediction_expt.py, and compared against the stored lp_6feat model APs.
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import link_pred_features as LPF
import link_prediction_expt as LPE


def delta_phi_frame(events):
    """Emit (eid, v, dphi, team_seen) in build_causal_rows' emission order."""
    counts = defaultdict(dict)          # frozenset(team) -> {node: m}
    out = []
    i = 0
    while i < len(events):
        t = events[i][2]
        j = i + 1
        while j < len(events) and events[j][2] == t:
            j += 1
        for eid in range(i, j):
            s, R, _ = events[eid]
            team = frozenset([s] + R)
            k = len(team)
            m = counts.get(team, {})
            seen = 1 if m else 0
            for v in R:
                m_v = m.get(v, 0)
                greater = sum(1 for x in team
                              if x != v and m.get(x, 0) > m_v)
                out.append((eid, v, 2.0 * greater / (k - 1), seen))
        for eid in range(i, j):
            s, R, _ = events[eid]
            team = frozenset([s] + R)
            counts[team][s] = counts[team].get(s, 0) + 1
        i = j
    return pd.DataFrame(out, columns=["eid", "v", "dphi", "team_seen"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="emaileu")
    ap.add_argument("--max-size", type=int, default=25)
    ap.add_argument("--min-recipients", type=int, default=2)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--horizons", type=float, nargs="+", default=[7.0, 30.0])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "results",
                                                      "delta_phi"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    events = LPF.load_dataset(args.dataset, max_size=args.max_size)
    unit = LPF.DATASETS[args.dataset]["step"]
    print(f"{args.dataset}: {len(events):,} events", flush=True)

    rows = pd.DataFrame(LPF.build_causal_rows(events, theta=args.theta))
    dp = delta_phi_frame(events)
    rows = rows.merge(dp, on=["eid", "v"], how="left", validate="one_to_one")
    assert rows.dphi.notna().all()
    if args.min_recipients > 1:
        rows = rows[rows.k >= 1 + args.min_recipients].reset_index(drop=True)
    print(f"samples: {len(rows):,}  "
          f"(dphi>0 on {(rows.dphi>0).mean():.1%}, "
          f"recurring-team rows {rows.team_seen.mean():.1%})", flush=True)

    t_end = events[-1][2]
    # cutoffs on the UNCENSORED stream, once -- identical to the pipeline
    train_cut, val_cut = LPE.temporal_cutoffs(rows["t"].to_numpy())
    results = []
    for h in args.horizons:
        horizon = h * unit
        frame = LPE.add_labels(rows, horizon, t_end)
        tr, va, te = LPE.temporal_masks(frame, horizon, train_cut, val_cut)
        print(f"\n== horizon {h:g} ({args.dataset})  train_cut={train_cut:.0f} "
              f"test rows={te.sum():,}", flush=True)
        for label in ("y_dyad", "y_mode", "y_group_exact"):
            _, _, test = LPE.label_population(frame, label, tr, va, te)
            sub = frame[test]
            y = sub[label].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            score = sub["dphi"].to_numpy()
            ap_ = average_precision_score(y, score)
            roc = roc_auc_score(y, score)
            prev = y.mean()
            rec = sub.team_seen.to_numpy() == 1
            ap_rec = (average_precision_score(y[rec], score[rec])
                      if rec.sum() and len(np.unique(y[rec])) == 2 else np.nan)
            prev_rec = y[rec].mean() if rec.sum() else np.nan
            ap_cold = (average_precision_score(y[~rec], score[~rec])
                       if (~rec).sum() and len(np.unique(y[~rec])) == 2
                       else np.nan)
            results.append(dict(dataset=args.dataset, horizon=h, label=label,
                                n_test=len(sub), prevalence=prev, ap=ap_,
                                rel_ap=ap_ / prev, roc=roc,
                                pct_recurring=rec.mean(),
                                ap_recurring=ap_rec, prev_recurring=prev_rec,
                                ap_cold=ap_cold))
            print(f"  {label:14s} prev={prev:.4f}  dPhi AP={ap_:.4f} "
                  f"(rel {ap_/prev:4.2f}x, ROC {roc:.3f})  "
                  f"recurring: AP={ap_rec:.4f} (prev {prev_rec:.4f}, "
                  f"{rec.mean():.1%} of rows)", flush=True)
        # --- rotation-label tryout (event level): does ANY other member
        # lead the exact team within h?  score = max row dPhi of the event.
        sub = frame[te]
        ev = (sub.groupby("eid")
                 .agg(y=("y_group_exact", "max"), score=("dphi", "max"),
                      seen=("team_seen", "max"), t=("t", "first")))
        y, sc = ev.y.to_numpy(), ev.score.to_numpy()
        if len(np.unique(y)) == 2:
            results.append(dict(dataset=args.dataset, horizon=h,
                                label="y_rotation_event", n_test=len(ev),
                                prevalence=y.mean(),
                                ap=average_precision_score(y, sc),
                                rel_ap=average_precision_score(y, sc) / y.mean(),
                                roc=roc_auc_score(y, sc),
                                pct_recurring=(ev.seen == 1).mean(),
                                ap_recurring=np.nan, prev_recurring=np.nan,
                                ap_cold=np.nan))
            r = results[-1]
            print(f"  {'y_rotation_ev':14s} prev={r['prevalence']:.4f}  "
                  f"dPhi AP={r['ap']:.4f} (rel {r['rel_ap']:4.2f}x, "
                  f"ROC {r['roc']:.3f})  events={len(ev):,}", flush=True)
    out = os.path.join(args.out_dir, f"{args.dataset}_delta_phi.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print("\n[saved]", out, flush=True)


if __name__ == "__main__":
    main()

# How this was run (2026-09-08):
#   cd Reciprocity/prediction_final/code
#   <venv-python> delta_phi_scorer.py --dataset emaileu     # defaults: --max-size 25 --min-recipients 2 --theta 0.5 --horizons 7 30
# All other datasets were run by run_all_delta_phi.py, which pulls each
# domain's exact config (horizons/max_size/min_recipients/theta) from
# ../results/lp_6feat/<ds>_diagnostics.json. Music needs music_feature_to_df.py
# beside this file (copied from Reciprocity/code). Plain numpy/pandas/sklearn venv.
# Outputs: ../results/delta_phi/<dataset>_delta_phi.csv
