#!/bin/bash
# Phase T: support-family beta sweep (dedup start E*, Phi = |E*| * R_part^M,
# phi="r2M"), {emaileu, enron} x beta in {-1,1,2,3,4,5} x 4 chains (2 observed
# + 2 fill); 500 sweeps for |beta|<=2, 1000 for beta>=3. beta=0 is phase D.
# Output: results/final/<ds>/beta<b>_dedup/chain*/ ; then on the laptop:
#   python diagnose_final.py && python make_beta_figure.py --set support_tilt

source /home/vedangibengali/ml_projects/testenv/bin/activate
cd /home/vedangibengali/ml_projects/Reciprocity/Hypergraph_GroupReciprocity/experiments/final
mkdir -p ../../results/final
python final_chains.py
python run_final.py --phase T --workers 48 > ../../results/final/runT.log 2>&1
