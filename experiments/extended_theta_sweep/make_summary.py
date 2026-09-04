#!/usr/bin/env python3
"""Rebuild ext_theta_hat.pdf + the summary table from extended_sweeps.csv,
so the summary survives a run that was stopped early."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_extended_sweeps as R

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "extended_sweeps.csv"))
R.plot_summary(df, os.path.join(HERE, "ext_theta_hat.pdf"))
print(f"\n{'dataset':16s} {'events':>8s} {'%dup':>6s} {'sweeps':>7s} "
      f"{'theta_hat':>9s} {'null':>8s} {'real':>8s} {'ratio':>7s} {'max 2-start gap':>16s}")
for key, g in df.groupby("dataset", sort=False):
    n0 = g[g.theta == 0.0].iloc[0]
    th = g.theta_hat.iloc[0]
    print(f"{g.label.iloc[0]:16s} {n0.n_events:8,.0f} {n0.pct_dup:5.1f}% "
          f"{n0.sweeps:7.0f} {th:9.2f} {n0.r2_multi:8.4f} "
          f"{n0.real_r2_multi:8.4f} {n0.real_r2_multi/n0.r2_multi:6.2f}x "
          f"{g.two_start_gap.max():15.2%}")
