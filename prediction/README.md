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
  split_50_20_30/         horizon tables, 50-20-30 split, untuned defaults (LightGBM 300 trees)
  split_50_20_30_tuned/   same, hyperparameters selected on the validation block (--tune):
                          LightGBM early stopping on val AP + num_leaves {15,31,63};
                          LogReg C {0.1,1,10}; per tier x label x horizon. PRIMARY tables.
                          (DNC h=7: validation block empty after the double embargo ->
                          rows fall back to untuned defaults, tuned=False, n_val=0)
  sensitivity_maxsize/    Enron h=7 at k<=50 and uncapped: cap removes broadcast
                          dilution, not signal
  sensitivity_earlystop/  Enron h=30 with logloss early stopping (never stops at 2000
                          trees -> overfits both tiers). Enron exact-team dPR across
                          protocols: +0.094 (AP-stop) / +0.113 (logloss) / +0.138
                          (untuned) -- robust in sign and magnitude
report_prediction_experiments.md   protocol, labels, and findings write-up
```

Horizons (days): email-Eu & Enron {1, 7, 30}; DNC {1, 3, 7}; Twitter {7, 30, 60}.
Labels: `y_dyad` (control), `y_group_exact` (exact team reconvenes), `y_mode`
(group vs dyadic reply, among reciprocators). Population: recipients of
group events, 3 ≤ k ≤ 25; feature history uses the full stream.

Run with a venv that has lightgbm (see the footer of each script).
