#!/bin/bash
# Phase-B extension after the first diagnostics pass (plan sec. 6.3).
#   |beta| <= 2 : extend to 3000 total sweeps  (48 chains; tau_Phi <= 14 -> ESS >= 1000)
#   beta  =  3  : extend to 8000 total sweeps  (16 chains; tau_Phi 22-47 -> ESS ~ 500-1100)
#   beta  = 4,5 : NOT extended -- tau_Phi 37-270 sweeps would need 12k-90k sweeps;
#                 kept as diagnostic-only points (open markers in the beta figure).
# Both tiers run concurrently (64 chains on 64 physical cores); completed chains
# resume from checkpoint.pkl, so re-running this script is safe.
# Wall time is bounded by congress beta=3 (~14.4 s/sweep * 7000 sweeps ~ 28 h);
# everything else finishes within ~12 h.

source /home/vedangibengali/ml_projects/testenv/bin/activate
cd /home/vedangibengali/ml_projects/Reciprocity/Hypergraph_GroupReciprocity/experiments/final
mkdir -p ../../results/final

python run_final.py --phase B --workers 48 --sweeps 3000 \
  --extend emaileu:-1 emaileu:1 emaileu:2 enron:-1 enron:1 enron:2 \
           twitter:-1 twitter:1 twitter:2 congress:-1 congress:1 congress:2 \
  > ../../results/final/runB_ext_lo.log 2>&1 &

sleep 60   # let the first run write manifest.json before the second updates it

python run_final.py --phase B --workers 16 --sweeps 8000 \
  --extend emaileu:3 enron:3 twitter:3 congress:3 \
  > ../../results/final/runB_ext_b3.log 2>&1 &

wait
