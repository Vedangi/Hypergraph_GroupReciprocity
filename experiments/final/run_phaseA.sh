#!/bin/bash

source /home/vedangibengali/ml_projects/testenv/bin/activate

cd /home/vedangibengali/ml_projects/Reciprocity/Hypergraph_GroupReciprocity
pip install -q -r requirements.txt
cd experiments/final
python final_chains.py

mkdir -p ../../results/final
python run_final.py --phase A --workers 54 > ../../results/final/runA.log 2>&1
~
