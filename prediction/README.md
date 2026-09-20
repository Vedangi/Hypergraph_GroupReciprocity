# prediction — reciprocation-mode prediction experiments

Temporal datasets only: **email-Eu, Enron, DNC, Twitter (senators)**.

```
code/
  link_pred_features.py   loaders (repo data/), causal rows, 20 graph + 5 hypergraph features
  link_prediction_expt.py original consolidated pipeline (validation-based selection etc.)
  run_horizon_tables.py   Table-14-style driver: pooled PR-A / PR-C / ΔPR (+ event-clustered
                          bootstrap CI) and ΔPR per team-size bucket (k=3–5, 6–10, ≥11),
                          LogReg + LightGBM, configurable split, fixed curated features
results/
  split_50_25_25/         frozen results of the original pipeline (train_q=.50, val_q=.75)
  split_50_20_30/         horizon tables at the 50-20-30 split (this README's driver)
report_prediction_experiments.md   protocol, labels, and findings write-up
```

Horizons (days): email-Eu & Enron {1, 7, 30}; DNC {1, 3, 7}; Twitter {7, 30, 60}.
Labels: `y_dyad` (control), `y_group_exact` (exact team reconvenes), `y_mode`
(group vs dyadic reply, among reciprocators). Population: recipients of
group events, 3 ≤ k ≤ 25; feature history uses the full stream.

Run with a venv that has lightgbm (see the footer of each script).
