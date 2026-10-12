# Final Experimental Plan for Structural Reciprocity Experiments

## 1. Purpose and scope

This document specifies the final experiments for the structural-results section of the paper. The experiments should answer two main questions:

1. **Reciprocity beyond the graph projection:** Are the observed group-reciprocity scores larger than those expected when the weighted directed graph projection and event sizes are fixed, but recipients are regrouped into hyperedges?
2. **Controlled variation at a fixed graph projection:** Can group reciprocity be varied systematically through the exponential tilt parameter \(\beta\), even though the weighted directed graph projection—and therefore graph reciprocity—remains unchanged?

The main analysis should use the **multiplicity-aware measures**. Support-based results should be presented as a secondary robustness analysis, preferably in the appendix. No parameter fitting or maximum-likelihood estimation is required for this paper.

Unless a sensitivity analysis explicitly says otherwise, retain only group events with

\[
3 \leq k \leq 25,
\]

where \(k=1+|R|\) is the number of participants in an event. Construct the fixed weighted graph projection from this same filtered collection of group events. Dyadic events are not part of these structural experiments.

## 2. Datasets

Run the final null-model experiment on all nine datasets used in the paper:

### Communication networks

- Email-Eu
- Enron
- DNC
- Fauci
- Radoslaw
- Twitter

### Collaboration networks

- Congress-bills
- DBLP
- Music collaboration

Use exactly these names and dataset versions in the scripts and output tables. Some preliminary result files contain Wiki and Higgs but omit Email-Eu and Music. Those preliminary runs do not constitute the final nine-dataset analysis.

## 3. Quantities to compute

For every observed hypergraph and every retained Markov-chain state, compute all three multiplicity-aware group-reciprocity measures:

- any-participation reciprocity, \(R_{\mathrm{any}}^{\mathrm m}\);
- all-participation reciprocity, \(R_{\mathrm{all}}^{\mathrm m}\);
- partial-participation reciprocity, \(R_{\mathrm{part}}^{\mathrm m}\).

The exponential model is tilted by

\[
\Phi(H)=|E|R_{\mathrm{part}}^{\mathrm m}(H),
\]

so \(\Phi\) or \(R_{\mathrm{part}}^{\mathrm m}\) is the primary diagnostic trace. Nevertheless, final convergence checks should be computed for **all three reported measures**, because good mixing of the tilted statistic does not automatically guarantee good estimation of every derived statistic.

When useful, also compute the three measures within the event-size bins

\[
k=3\text{--}5,\qquad k=6\text{--}10,\qquad k=11\text{--}25.
\]

These size-stratified scores do not require new chains; calculate them on the states already retained from the main runs.

## 4. Definition of one sweep

Use a consistent definition in all code, logs, and paper text:

> One sweep consists of \(L=\sum_i |R_i|\) proposal attempts, where \(L\) is the total number of sender--recipient incidences in the filtered hypergraph.

Record the chain state after every completed sweep, including sweeps during which some or all proposals are rejected. A self-loop is a valid Markov-chain state and must not be discarded.

## 5. Experiment A: projection-preserving null model at \(\beta=0\)

This is the central structural experiment and should be completed for all nine datasets.

### 5.1 Chain design

For each dataset:

- Run **six independent chains**.
- Initialize three chains at the observed hypergraph.
- Initialize three chains at independently randomized feasible states produced by the existing projection-preserving fill/randomization procedure.
- Use independent, recorded random seeds for every chain.
- Set \(\beta=0\), so the target distribution is the projection-preserving null distribution over the edge-labeled state space.
- Begin with **1,000 sweeps per chain**.
- Discard the first **25%** of every chain as burn-in.
- Retain and summarize all post-burn-in states, rather than only the terminal state.

Using two dispersed types of starting state is important: agreement between observed-start and randomized-start chains is direct evidence that the result is not an artifact of initialization.

### 5.2 Adaptive extension rule

After the first 1,000 sweeps, compute split-\(\widehat R\) and effective sample size (ESS) for each of the three reciprocity traces. Then apply this rule:

1. Accept the run if every reported measure has \(\widehat R\leq 1.01\), and the primary trace \(R_{\mathrm{part}}^{\mathrm m}\) has pooled ESS at least 2,000.
2. If the rule fails, continue the same chains to **1,600 total sweeps** and recompute the diagnostics.
3. If it still fails, continue to **3,200 total sweeps**.
4. If convergence still fails at 3,200 sweeps, do not silently average the chains. Inspect chain-specific traces, acceptance rates, starting-state differences, and implementation logs, and report the dataset as requiring special treatment.

For \(R_{\mathrm{any}}^{\mathrm m}\) or \(R_{\mathrm{all}}^{\mathrm m}\), an ESS lower than 2,000 can be acceptable if \(\widehat R\leq1.01\) and its Monte Carlo uncertainty is demonstrably negligible relative to the observed-null separation. Record that justification explicitly.

### 5.3 Null summaries

For each dataset and each measure, report:

- the observed score;
- the post-burn-in null mean;
- the empirical 2.5th and 97.5th percentiles of the retained null draws;
- the absolute difference, observed minus null mean;
- optionally, the ratio of the observed score to the null mean;
- split-\(\widehat R\), pooled ESS, and Monte Carlo standard error.

The 2.5th--97.5th percentile interval is a **null reference interval**, not a confidence interval for the observed hypergraph. Because adjacent MCMC draws are correlated, estimate Monte Carlo error using the ESS rather than treating all saved states as independent.

If reporting an empirical tail probability, account for MCMC dependence and avoid writing an exact zero. A conservative statement such as “the observed score exceeded all retained null draws” is preferable when that is what the simulation shows.

### 5.4 Required outputs

Save, at minimum:

- a chain-level metadata file containing dataset, chain ID, initialization type, seed, number of sweeps, burn-in, proposal count, accepted moves, acceptance rate, wall time, and completion status;
- a state-level trace file containing sweep number and all three reciprocity measures;
- a convergence table with \(\widehat R\), ESS, autocorrelation-time estimate, and Monte Carlo standard error for every measure;
- a dataset-level summary table with observed and null statistics;
- the final state or a reproducible checkpoint for every chain;
- an error log that makes partial or failed runs distinguishable from completed runs.

Do not overwrite shorter pilot runs. Put final results in a new, clearly named directory.

## 6. Experiment B: reciprocity control across \(\beta\)

This experiment demonstrates how the exponential tilt changes group organization while the graph projection remains fixed. It is not a parameter-estimation experiment.

### 6.1 Representative datasets

Use four networks in the main beta-sweep figure:

- Email-Eu;
- Enron;
- Twitter;
- Congress-bills.

This set covers small and large communication networks, a social-media network, and a collaboration network. If page space is tight, show two in the main paper and place the other two in the appendix, but run all four using the same protocol.

### 6.2 Beta grid and chains

Use

\[
\beta\in\{-1,0,1,2,3,4,5\}.
\]

For each dataset and each nonzero \(\beta\):

- run **four independent chains**;
- start two at the observed hypergraph and two at independently randomized feasible states;
- run **500 sweeps per chain** initially;
- discard the first **125 sweeps** as burn-in;
- summarize all remaining states, not only the endpoint;
- use independent, recorded seeds.

Reuse the six-chain \(\beta=0\) output from Experiment A rather than rerunning it. Its larger number of chains does not cause a problem; the uncertainty for each beta should be displayed directly.

### 6.3 Beta-sweep extension rule

For every dataset-beta combination, compute diagnostics for \(\Phi\) and all reported measures. Accept the run if:

- \(\widehat R\leq1.01\) for all reported measures;
- pooled ESS for \(\Phi\) is at least 1,000;
- the observed-start and randomized-start post-burn-in means agree within their Monte Carlo uncertainty.

If the rule fails, continue all four chains for that dataset-beta combination to **1,000 total sweeps**. If it still fails, extend further or label the point as insufficiently mixed; do not present an unconverged terminal value as an estimate.

Acceptance can deteriorate at large positive or negative \(\beta\). Therefore save acceptance rates separately for every beta, and inspect the \(\beta=5\) runs particularly carefully. If \(\beta=5\) remains poorly mixed, retain it only as a diagnostic point and make \(\beta=4\) the plotted endpoint.

### 6.4 Beta-sweep summaries

At each beta, report post-burn-in means and uncertainty for all three multiplicity-aware measures. The primary plot should show beta on the horizontal axis and reciprocity on the vertical axis, with one curve per measure and uncertainty bands or error bars.

The weighted graph projection is identical at every beta, so graph reciprocity is constant. State this in the caption rather than plotting a visually redundant horizontal curve unless that curve materially improves interpretation.

For the tilted statistic,

\[
\frac{d}{d\beta}\mathbb E_{\beta}[\Phi]
=\operatorname{Var}_{\beta}(\Phi)\geq0.
\]

Thus, the beta sweep should be framed as an illustration of the attainable change, its magnitude across domains, and the behavior of the two non-tilted reciprocity measures—not as empirical proof that the expectation of \(\Phi\) is monotone.

## 7. Support-based robustness analysis

The multiplicity-aware results should be the main analysis because the observed systems are multihypergraphs and repeated events carry substantive information.

For a low-cost support robustness analysis, use every retained state \(H_t\) from Experiment A, collapse it to its support \(\operatorname{supp}(H_t)\), and compute the three support-based measures. Compare the observed support score with the distribution of the support statistic induced by the occurrence-level fixed-projection null chain.

This analysis must be described precisely:

> We evaluate support-based statistics on the deduplicated support of each state sampled from the occurrence-level, fixed-weighted-projection null model.

It is **not** a separate Markov chain that is uniform over deduplicated hypergraphs, and deduplication does not generally preserve the support graph projection. Therefore, do not describe these outputs as samples from a support-only fixed-projection state space.

Do not compare the raw numerical size of a support score directly with its multiplicity-aware counterpart as though one were universally an upper bound. Instead, compare each observed statistic with the corresponding statistic under its stated null construction.

A separate support-beta sweep is not necessary unless the \(M_T\)-weighted support extension is promoted to a formal contribution. The existing 100-sweep Email-Eu and Enron runs are preliminary diagnostics, not final evidence.

## 8. Additional robustness analyses

These analyses belong in the appendix unless they reveal a major qualification to the main result.

### 8.1 Hyperedge-size-stratified reciprocity

Using the already retained states from Experiment A, compute observed and null scores for:

- \(k=3\text{--}5\);
- \(k=6\text{--}10\);
- \(k=11\text{--}25\).

Report the observed-null difference or ratio within each bin. Do not launch separate chains for the bins unless the paper asks a different conditional-null question.

### 8.2 Sensitivity to the maximum hyperedge size

Run this sensitivity check on **Enron** and **Congress-bills**, which represent communication and collaboration domains and are computationally substantial enough to be informative.

Use three filters:

- \(3\leq k\leq10\);
- \(3\leq k\leq25\) (the main specification);
- \(k\geq3\) with no upper cutoff.

For each new filter, use four chains, 1,000 initial sweeps, 25% burn-in, and the same \(\widehat R\leq1.01\) rule. Extend if necessary. Compare qualitative conclusions and observed-null separations; exact equality across filters is not expected.

### 8.3 Edge-labeling sensitivity

Use the existing Email-Eu unlabeled-state-space experiment as an appendix sensitivity result, provided its implementation and target distribution are documented clearly. The existing run already shows a similar observed-null separation and does not need to be repeated solely to increase the number of states.

If a second dataset is desired, use DNC with four chains and 1,000 sweeps per chain, extending under the usual diagnostic rule. This is optional and lower priority than finishing the nine-dataset main analysis.

## 9. Convergence and mixing diagnostics

### 9.1 What the diagnostics mean

- **Trace plot:** reciprocity against sweep number for every chain. Stable overlapping traces with no persistent drift support stationarity and mixing.
- **Autocorrelation:** similarity between chain values separated by a given lag. Slow decay means that nearby saved states contain limited new information.
- **Integrated autocorrelation time, \(\tau\):** the approximate number of correlated draws equivalent to one independent draw.
- **Effective sample size, ESS:** the approximate number of independent draws represented by the correlated post-burn-in output. For pooled output, it is roughly the total number of post-burn-in draws divided by \(\tau\).
- **Split-\(\widehat R\):** comparison of within-chain and between-chain variation after splitting chains. Values near one indicate agreement across chains; use \(1.01\) as the final threshold.
- **Starting-state comparison:** comparison between chains initialized at the observed state and dispersed randomized states. A persistent difference is evidence that burn-in or run length is inadequate.
- **Acceptance rate:** fraction of attempted proposals accepted. Very low rates can cause slow movement; unusually high rates together with little change may indicate that proposals are too local.

### 9.2 Diagnostic figures

The appendix should include representative trace and autocorrelation plots for at least:

- one small/fast-mixing dataset;
- one large/slow-mixing dataset;
- one beta-sweep endpoint with the poorest acceptable mixing.

Provide a complete diagnostic table for every final run even if only representative trace plots are shown.

The existing 300-sweep relaxation plots are useful for illustrating movement away from different starting states, but they should not replace diagnostics based on the final chains and should not be used as the sole null estimates.

## 10. Main-paper tables and figures

### Main table: observed versus null

For all nine datasets, report the observed and null values of the three multiplicity-aware measures. To keep the table compact, use one of these layouts:

- observed score, null mean, and observed-minus-null for each measure; put null intervals in the appendix; or
- observed score and null mean with a compact marker indicating whether the observed value lies outside the 95% null reference interval.

Do not label the null mean as a generic “expected score” without saying that it is expectation under the projection-preserving null model.

### Main figure: normalized observed-null separation

Create a dot or interval plot with datasets on one axis and either

\[
R_{\mathrm{obs}}-\mathbb E_0[R]
\]

or

\[
R_{\mathrm{obs}}/\mathbb E_0[R]
\]

on the other. Absolute differences are easier to interpret when null means are near zero; ratios can be placed in the appendix or reported alongside differences. Use color or facets for the three measures.

### Main beta-control figure

Plot post-burn-in mean reciprocity versus beta for the four representative datasets, with uncertainty shown and a consistent beta grid. Faceting by dataset is preferable to placing all networks on a single axis with very different score ranges.

### Appendix material

Include:

- full observed/null means and 95% null reference intervals;
- \(\widehat R\), ESS, autocorrelation-time, acceptance-rate, and runtime tables;
- trace and autocorrelation diagnostics;
- event-size-bin results;
- maximum-size-cutoff sensitivity;
- support-statistic robustness results;
- labeled versus unlabeled state-space sensitivity;
- additional beta-sweep panels if only a subset fits in the main paper.

## 11. Estimated computational budget

The estimates below are planning ranges rather than guarantees; actual time depends on parallelism, proposal acceptance, dataset size, and adaptive extensions.

### Final \(\beta=0\) null analysis

- Six chains for each of nine datasets.
- Initial run: 1,000 sweeps per chain.
- Expected wall-clock time with approximately six workers: **12--18 hours**.
- Allow up to **24 hours** if Enron, Congress-bills, or other slow chains require extension.

Preliminary ESS results suggest that approximately 1,000--1,600 sweeps should be sufficient for most datasets. Congress-bills is the primary computational bottleneck, so it should be launched early and checkpointed.

### Four-dataset beta sweep

- Six nonzero beta values, four chains per point, 500 initial sweeps.
- Reuse \(\beta=0\) from the main null run.
- Expected wall-clock time with parallel workers: **12--18 hours**.
- Allow up to **24 hours** for extensions at extreme beta values.

### Appendix sensitivity experiments

- Maximum-size-cutoff sensitivity: approximately **6--10 hours**.
- Size-bin and support summaries: minimal extra chain time, because they reuse saved states.
- Optional second unlabeled experiment: run only after all primary outputs are complete.

The complete recommended program should fit in roughly **1.5--2 days of wall-clock computation**, or **2--3 days** if several adaptive extensions are needed.

## 12. Execution order

Run experiments in the following order:

1. Freeze and record dataset preprocessing, especially the \(3\leq k\leq25\) rule.
2. Run a short reproducibility test to verify that fixed seeds reproduce identical traces.
3. Launch the six-chain \(\beta=0\) experiment for Congress-bills and Enron first.
4. Launch the remaining seven datasets.
5. Compute diagnostics and perform only the required adaptive extensions.
6. Generate the final nine-dataset null table and plot.
7. Run the four-dataset beta sweep and extend failing beta points.
8. Compute support and size-bin statistics from saved main-chain states.
9. Run the two-dataset hyperedge-size-cutoff sensitivity.
10. Generate appendix diagnostics and verify that all displayed numbers come from runs marked complete.

## 13. Quality-control checklist

Before treating a run as final, verify all of the following:

- [ ] The input contains only events with \(3\leq k\leq25\), unless it is a declared sensitivity run.
- [ ] Every event retains its sender and size throughout the chain.
- [ ] Every sender--recipient multiplicity in the weighted graph projection is invariant.
- [ ] Repeated hyperedges are allowed and edge labels distinguish repeated event occurrences.
- [ ] The number of proposal attempts per sweep equals the total number of recipient incidences.
- [ ] Rejected proposals are represented as self-loops rather than omitted states.
- [ ] All random seeds and initialization types are saved.
- [ ] Burn-in is removed separately from each chain.
- [ ] Estimates use all post-burn-in states, not only chain endpoints.
- [ ] Split-\(\widehat R\) and ESS are computed for every reported measure.
- [ ] Observed-start and randomized-start chains agree after burn-in.
- [ ] Null reference intervals are not mislabeled as confidence intervals for the observed data.
- [ ] Support statistics are described as statistics evaluated on supports of occurrence-level null states.
- [ ] Failed, interrupted, and incomplete runs cannot enter the final aggregation silently.
- [ ] Output tables contain the same nine datasets, filters, and measure definitions used in the paper.

## 14. Experiments not needed for the current paper

Do not spend final-computation time on the following unless the paper's scope changes:

- maximum-likelihood estimation of \(\beta\);
- fitting a domain-specific beta parameter;
- 30 independent chains per dataset;
- a full support-beta sweep on all nine datasets;
- the preliminary Model B experiment with fewer than approximately 16 effective sweeps;
- terminal-state-only beta summaries;
- repeated calculations of duplicate-edge percentages unless those values are actually reported;
- additional datasets such as Wiki or Higgs that are not part of the paper's stated nine-network analysis.

The priority is a clean, reproducible, well-diagnosed answer to the two central questions—not a larger collection of loosely connected preliminary experiments.
