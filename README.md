# Hypergraph Group Reciprocity

Reciprocity measures, projection-preserving null models, and exponential
random models for **single-sender directed hypergraphs** (email-like group
communication: one sender, many recipients, repeated events).

Code accompanying the paper *[title TBD]*.

## What is here

Each event is `(sender s, recipient set R)`; the **team** is `T = {s} ∪ R`.
Repeated events on the same team give each member a sender multiplicity
`m_{T,i}`. We provide three multiplicity-aware reciprocity measures —

- **R_any** (`s1_multi`) — any-other-sender capacity,
- **R_some** (`r2_multi`) — pairwise sender-role overlap (primary measure),
- **R_all** (`r1_multi`) — strict full-rotation units,

which all reduce **exactly** to weighted directed-graph reciprocity
(Squartini et al. 2013) when every team is dyadic, and to classical binary
reciprocity (Newman 2002; Garlaschelli & Loffredo 2004) when additionally
deduplicated. Formal definitions and reduction proofs: `docs/measures.md`.

The statistical machinery:

- a **projection-preserving null**: within-sender Ryser 2×2 switches that hold
  every dyadic count `w_ij` fixed, so any real-vs-null gap is attributable
  purely to *grouping* (irreducibility proof: `docs/irreducibility.pdf`);
- an **exponential tilt** `P(H) ∝ exp(θ·Φ(H))` with `Φ = |E|·R_some`, the
  hypergraph analog of the Holland–Leinhardt mutual-statistic model, with
  moment-matching MLE for `θ`;
- **labelled and unlabelled** base measures (Fosdick et al. 2018 style) via a
  Metropolis factorial reweighting.

## Layout

```
src/
  datasets.py                 unified data loaders (see data/README.md)
  multihypergraph/            duplicate-aware (multigraph) setting
    multi_reciprocity.py        measures + graph-reduction checks
    multi_tilt.py               projection-fixed null + exponential tilt MCMC
  dedup/                      deduplicated (support) setting
    reciprocity_dedup.py        support measures r1-r4/s1-s3 + shuffle nulls
    rcm_model.py                Reciprocal Coalition Model (closed forms + validation)
    model_b_tilt.py             projection-fixed tilt on deduplicated hypergraphs
experiments/
  run_multi_sweeps.py         θ-sweeps + email-Eu mixing bracket
  extended_theta_sweep/       communication datasets, two-start mixing diagnostic
  unlabelled_base/            labelled vs unlabelled base measure study (appendix)
results/                      CSVs + figures as reported in the paper
data/                         bundled datasets (provenance in data/README.md)
docs/                         measure definitions + irreducibility proof
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# observed measures on one dataset
python - <<'EOF'
import sys; sys.path += ["src", "src/multihypergraph"]
import datasets, multi_reciprocity as MR
events = [(s, R) for s, R, _t in datasets.load_dataset("dnc", max_size=25)]
print(MR.measures(events))
EOF

# full θ-sweep + null comparison on the communication datasets
cd experiments/extended_theta_sweep && python run_extended_sweeps.py
```

Chains write incrementally, so partial runs are usable; `make_summary.py`
rebuilds the summary figure/table from the CSV at any point.

## Citation

Citation entry will be added on publication.
