#!/usr/bin/env python3
r"""Candidate higher-order features, tested against the curated tier C.

Candidates (all causal, count-based, clock-free; team history strictly < t):
  h_team_pairmatch      = 2 Sum_{i<j} min(m_i,m_j) / ((k-1) Sum_i m_i)
                          (causal per-team R_some^multi; 0 on unseen teams)
  h_team_participation  = |{u in T\{s}: m_u > 0}| / (k-1)
                          (the focal event's causal R_part credit)
  dphi                  = 2/(k-1) #{j in T\{v}: m_j > m_v}  (v-specific)

Variants (tuned 50-20-30, same protocol as run_horizon_tables.py --tune):
  C5        curated five
  C5swap    h_team_leaders -> h_team_participation
  C5+pm     C5 + h_team_pairmatch
  C5+pm+dphi
Reported: dPR vs tier A, and the paired event-clustered CI of each variant
vs C5 (the decision quantity).
"""
import argparse, os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import link_pred_features as LPF
import link_prediction_expt as LPE
import run_horizon_tables as RH


def candidate_frame(events):
    counts = defaultdict(dict)
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
            M = sum(m.values())
            if M:
                c = sorted(m.values())
                p = len(c)
                pm = 2.0 * sum(ci * (p - 1 - a) for a, ci in enumerate(c)) / ((k - 1) * M)
            else:
                pm = 0.0
            part = sum(1 for u in m if u != s) / (k - 1)
            for v in R:
                m_v = m.get(v, 0)
                dphi = 2.0 * sum(1 for x in team if x != v and m.get(x, 0) > m_v) / (k - 1)
                out.append((eid, v, pm, part, dphi))
        for eid in range(i, j):
            s, R, _ = events[eid]
            team = frozenset([s] + R)
            counts[team][s] = counts[team].get(s, 0) + 1
        i = j
    return pd.DataFrame(out, columns=["eid", "v", "h_team_pairmatch",
                                      "h_team_participation", "dphi"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="emaileu")
    p.add_argument("--horizons", type=float, nargs="+", default=[1, 7, 30])
    p.add_argument("--out-dir", default=os.path.join(HERE, "..", "results",
                                                     "feature_candidates"))
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    events = LPF.load_dataset(a.dataset, max_size=25)
    unit = LPF.DATASETS[a.dataset]["step"]
    rows = pd.DataFrame(LPF.build_causal_rows(events, theta=0.5))
    rows = rows.merge(candidate_frame(events), on=["eid", "v"], how="left",
                      validate="one_to_one")
    rows = rows[rows.k >= 3].reset_index(drop=True)
    train_cut, val_cut = LPE.temporal_cutoffs(rows["t"].to_numpy(), 0.5, 0.7)
    t_end = events[-1][2]
    A = list(LPF.GRAPH_FEATURES)
    H = list(LPF.HYPERGRAPH_FEATURES)
    V = {"C5": A + H,
         "C5swap": A + [f for f in H if f != "h_team_leaders"] + ["h_team_participation"],
         "C5+pm": A + H + ["h_team_pairmatch"],
         "C5+pm+dphi": A + H + ["h_team_pairmatch", "dphi"]}
    print(f"{a.dataset}: {len(rows):,} rows; pairmatch>0 on "
          f"{(rows.h_team_pairmatch>0).mean():.1%}; corr(leaders,participation)="
          f"{rows.h_team_leaders.corr(rows.h_team_participation):.3f}, "
          f"corr(leaders,pairmatch)={rows.h_team_leaders.corr(rows.h_team_pairmatch):.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    out = []
    for h in a.horizons:
        fr = LPE.add_labels(rows, h * unit, t_end)
        tr, va, te = LPE.temporal_masks(fr, h * unit, train_cut, val_cut)
        for label in RH.LABELS:
            tr_l, va_l, te_l = LPE.label_population(fr, label, tr, va, te)
            fit = tr_l | va_l
            sub = fr[te_l]
            y_te = sub[label].to_numpy()
            eid = sub["eid"].to_numpy()
            ytr, yva, yfit = (fr.loc[tr_l, label].to_numpy(),
                              fr.loc[va_l, label].to_numpy(),
                              fr.loc[fit, label].to_numpy())
            for mname in RH.MODELS:
                def run(cols):
                    s, _ = RH.fit_score_tuned(
                        fr.loc[tr_l, cols].to_numpy(), ytr,
                        fr.loc[va_l, cols].to_numpy(), yva,
                        fr.loc[fit, cols].to_numpy(), yfit,
                        sub[cols].to_numpy(), mname)
                    return s
                sA = run(A)
                scores = {name: run(cols) for name, cols in V.items()}
                apA = RH.ap(y_te, sA)
                base = scores["C5"]
                line = f"  h={h:g} {label:14s} {mname:8s} PR-A={apA:.3f}"
                for name, s in scores.items():
                    d_vs_A = RH.ap(y_te, s) - apA
                    row = dict(dataset=a.dataset, horizon=h, label=label,
                               model=mname, variant=name, PR=RH.ap(y_te, s),
                               dPR_vs_A=d_vs_A)
                    if name != "C5":
                        lo, hi = RH.cluster_boot_ci(y_te, base, s, eid)["pooled"]
                        row.update(d_vs_C5=RH.ap(y_te, s) - RH.ap(y_te, base),
                                   ci_low=lo, ci_high=hi)
                        star = "*" if lo > 0 else ("-" if hi < 0 else " ")
                        line += f" | {name} {row['d_vs_C5']:+.3f}{star}"
                    else:
                        line += f" | C5 dPR={d_vs_A:+.3f}"
                    out.append(row)
                print(line, flush=True)
        pd.DataFrame(out).to_csv(os.path.join(
            a.out_dir, f"{a.dataset}_feature_candidates.csv"), index=False)
    print(f"DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

# How this was run (2026-09-20) -- lightgbm venv:
#   cd Hypergraph_GroupReciprocity/prediction/code
#   mkdir -p ../results/feature_candidates
#   <lgbm-python> feature_candidates_test.py --dataset emaileu --horizons 1 7 30 \
#       > ../results/feature_candidates/run_emaileu.log 2>&1
