#!/bin/bash
# Extension of the beta sweeps so that every plotted point (beta <= 3; Twitter
# <= 4) meets Rhat <= 1.05 and ESS(Phi) >= 400 (Vehtari et al. 2021):
#   phase B (multi tilt)   beta in {1,2,3} -> 3000 sweeps; enron:3 -> 6000;
#                          twitter:4 -> 8000
#   phase T (support tilt) beta in {1,2,3} -> 3000 sweeps; enron:3 -> 6000
# beta = -1 points already pass; beta = 4, 5 (except twitter:4) stay as
# diagnostic-only rows. All runs resume from checkpoint.pkl; re-running is safe.
# ~104 chains on 128 threads; wall time ~11 h (congress beta=3 at ~14 s/sweep).
# Launches are staggered so each run writes manifest.json before the next starts.

source /home/vedangibengali/ml_projects/testenv/bin/activate
cd /home/vedangibengali/ml_projects/Reciprocity/Hypergraph_GroupReciprocity/experiments/final
mkdir -p ../../results/final
L=../../results/final

python run_final.py --phase B --workers 48 --sweeps 3000 \
  --extend emaileu:1 emaileu:2 emaileu:3 enron:1 enron:2 \
           twitter:1 twitter:2 twitter:3 congress:1 congress:2 congress:3 \
  > $L/runB_ext3000.log 2>&1 &
sleep 60
python run_final.py --phase B --workers 4 --sweeps 6000 --extend enron:3 \
  > $L/runB_ext_enron3.log 2>&1 &
sleep 60
python run_final.py --phase B --workers 4 --sweeps 8000 --extend twitter:4 \
  > $L/runB_ext_twitter4.log 2>&1 &
sleep 60
python run_final.py --phase T --workers 36 --sweeps 3000 \
  --extend emaileu:1 emaileu:2 emaileu:3 enron:1 enron:2 \
  > $L/runT_ext3000.log 2>&1 &
sleep 60
python run_final.py --phase T --workers 4 --sweeps 6000 --extend enron:3 \
  > $L/runT_ext_enron3.log 2>&1 &

wait
echo "ALL EXTENSIONS DONE" >> $L/runB_ext3000.log
