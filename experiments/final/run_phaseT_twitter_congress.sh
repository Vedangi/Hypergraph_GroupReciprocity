#!/bin/bash
# Phase T (support-family tilt, dedup start, Phi = |E*| R_part^M) for the two
# remaining beta-sweep datasets, twitter and congress, at the budget the
# extension gives the others: beta in {1,2,3} -> 3000 sweeps, beta in {-1,4,5}
# -> 1000 sweeps; 4 chains each (2 dedup-observed + 2 fill). 48 chains.
# Speeds (dedup start): twitter 0.76 s/sweep, congress 13.5 s/sweep, so wall
# time is ~11 h (congress beta=3). Resumes from checkpoints; safe to re-run.
# Launch AFTER run_extend_BT.sh has finished (or alongside with fewer cores).

source /home/vedangibengali/ml_projects/testenv/bin/activate
cd /home/vedangibengali/ml_projects/Reciprocity/Hypergraph_GroupReciprocity/experiments/final
mkdir -p ../../results/final
L=../../results/final

python run_final.py --phase T --workers 24 --sweeps 1000 \
  --extend twitter:-1 twitter:4 twitter:5 congress:-1 congress:4 congress:5 \
  > $L/runT_tc_1000.log 2>&1 &
sleep 60
python run_final.py --phase T --workers 24 --sweeps 3000 \
  --extend twitter:1 twitter:2 twitter:3 congress:1 congress:2 congress:3 \
  > $L/runT_tc_3000.log 2>&1 &

wait
echo "PHASE T twitter/congress DONE" >> $L/runT_tc_3000.log
