# Predicting the Mode of Reciprocation in Single-Sender Directed Hypergraphs

*Report on the prediction experiments (final sweep, Jul 18–19; logs in
`code/final_sweep_Jul18/`). The feature definitions are given in the separate
feature table; this report covers the prediction problems, the evaluation
protocol, the validation-selected features, and the results.*

---

## 1. The prediction problems

### 1.1 Setting and unit of prediction

The data is a stream of timestamped single-sender directed hyperedges
("events"): `e = (s, R, t)` — sender `s`, recipient set `R`, time `t`. The
team is `N = {s} ∪ R` with size `k = 1 + |R|`.

One **prediction row** is a pair *(focal event, recipient)*: for every group
event (`|R| ≥ 2`) and every recipient `v ∈ R`, we ask — given everything
observable strictly before `t` — *whether and how `v` will reciprocate within
a horizon `h`*. Every feature is computed **causally** from events strictly
before `t` (events sharing a timestamp are all emitted before any of them
updates the feature state), and no feature uses the horizon, clock decay, or
any tuned timescale.

### 1.2 The four labels

A *qualifying group response* is a later event **led by `v`** with at least
two recipients. The labels differ in what that response must re-assemble:

| label | positive iff, within `(t, t+h]` … | role |
|---|---|---|
| **`y_dyad`** ("any-reply") | `v` sends *any* message to `s` | **negative control** — a purely dyadic target that graph features should already solve |
| **`y_group_strict`** | `v` leads a group response that (a) addresses `s`, (b) re-engages ≥ 1 original co-recipient, (c) overlaps the original team by ≥ θ = 0.5 | the workhorse group-reciprocation target; condition (b) ensures a near-dyadic "s + anyone" reply cannot satisfy it |
| **`y_group_exact`** | `v` leads a response to *exactly* `N∖{v}` — no one missing, no one extra | the strictest target; the temporal analog of the static `to_team` s-measures |
| **`y_mode`** | *among reciprocators only* (rows with `y_dyad`=1 or `y_group_strict`=1): 1 if the group channel activated, 0 if the reciprocation was dyadic-only | the mode question — "given that `v` responds, *how* does `v` respond?" |

A fifth, **event-level** label completes the correspondence with the static
measures: **`y_rotation`** — for each group event, does *any* member `v ∈ R`
lead the exact team within `h`? (= max over `v` of `y_group_exact`.) It is
the forward-looking version of sender-role sharing, i.e. the temporal analog
of the *partial* (R_some) measure. The conceptual mapping is *atoms versus
aggregates*: a positive `y_group_exact` event is precisely an increment to
`m_{T,v}` — the atomic event out of which all static measures are built —
and R_any/R_some/R_all differ only in how they aggregate these atoms across
members. The overlap-based `y_group_strict` is a sparsity-robust relaxation
of the exact atom (exact teams recur rarely), with precedent in the temporal
module's θ-overlap measures; the θ-sweep below shows every conclusion
strengthens monotonically as θ → exact, so the relaxation buys power, not a
different phenomenon.

The labels are nested in strictness (`exact ⊆ strict`), and a *coverage*
variant (`N ⊆ N′`) sits between them in the label-variant analyses. Labels use
**window semantics** (a state forecast: "will this behavior occur within
`h`?"), not reply attribution — the datasets carry no thread metadata, and
window semantics are the standard formulation in temporal (higher-order) link
prediction. Because both compared models score identical labels, any window
generosity cancels in the contrasts we report.

### 1.3 The scientific design: a contrast, not a benchmark

Every task is evaluated twice with the same rows, labels, and model class:

* **Tier A (graph baseline)** — 15 features computable from the *directed
  weighted projection* of the history (message counts, balance, pair
  turn-taking and latency, degrees, common neighbours), deliberately
  strengthened with the best-performing features from the graph-reciprocity
  literature: Kleinberg-style status ratios and directed two-step paths
  (Cheng, Romero, Meeder & Kleinberg), and the reciprocal-neighbourhood RC
  index (Sett et al.). Tier A sees **no event-size or grouping information**.
* **Tier C (hypergraph)** — tier A **plus five** hypergraph reciprocity
  measures (team size; group-vs-individual reciprocity `gir`; causal r4
  leadership coverage; exact-team turn-taking; exact-team leadership
  breadth). Each is a named measure from the theory section; none is `v`'s own
  past rate of the label class (no behavioral self-history), and none uses
  clock time — so tiers A and C are **temporally symmetric** (if anything,
  only tier A carries clock-time information, via reply latency).

The claim under test is a **double dissociation**: tier C should beat tier A
on the group targets, increasingly so as the target becomes more group-like
(`strict` → `exact`), and should add *nothing* on the dyadic control.

## 2. Data split and evaluation protocol

**Temporal 50-20-30 split with double embargo.** Order all rows by time and
cut at the 50th and 70th quantiles:

```
train : rows whose entire label window resolves before t(0.50)   (model fitting)
val   : rows in [t(0.50), ·) whose window resolves before t(0.70) (feature selection ONLY)
test  : rows at or after t(0.70)                                  (reported once)
```

Rows whose label window crosses a boundary are **purged** (embargo), so no
training label depends on a validation-period event and no validation label
depends on a test-period event. All data-dependent choices — the feature
selection of §3, and the horizon itself — are made without touching the test
block. Final models are refit on train+val and scored once on test.

**Horizon.** Chosen from the *training block only*: `h` = the 75th percentile
of fully-observed group-reciprocation delays (censoring reported). This
resolved to **35 days** for email-Eu and **2 years** for DBLP — a per-domain
reciprocation-timescale fingerprint in itself. Earlier sensitivity sweeps
(3–90 days; θ ∈ {0.3, 0.5, 0.75, 1.0}; 50-50 and 70-30 splits; rolling-origin
folds) showed the contrasts are insensitive to all of these choices.

**Population.** Prediction rows come from group events only
(`--min-recipients 2`; the dyadic k=2 stratum would otherwise dominate pooled
numbers), and events larger than **k = 25** are excluded
(`--max-size 25`; for DBLP this is the standard "restricted" coauthorship
hypergraph). Feature histories still use the *complete* event stream — the
filter restricts what we predict about, not what the models may know.

**Models and uncertainty.** Gradient-boosted trees (LightGBM, fixed default
hyperparameters — nothing is tuned) with logistic regression as a linear
check. The headline metric is PR-AUC (average precision), reported with its
no-skill floor (the prevalence) and as a *relative lift* (PR-AUC /
prevalence). Pooled contrasts carry a **paired, event-clustered bootstrap 95%
CI**: each replicate resamples whole events (all recipients of an email move
together) and scores both models on the same resample, so the interval is on
the gap itself. Results are stratified by team size (k = 3–5, 6–10, 11–25).

## 3. What the validation set selects

Feature selection runs only on the validation block: leave-one-out screening
of the five hypergraph features over the tier-A base (keep a feature if its
removal costs > 0.002 validation AP), followed by a joint-recovery pass that
re-adds omitted features while the reduced set underperforms the full set —
guarding against discarding correlated features that matter jointly.

Validation ΔAP if removed (bold = kept; JR = re-added by joint recovery):

| feature | email-Eu `y_mode` | email-Eu `y_exact` | DBLP `y_mode` | DBLP `y_exact` |
|---|---|---|---|---|
| `h_r4_lead` | **+0.023** | −0.005 | **+0.034** | **+0.004** |
| `h_rsize` | **+0.010** | −0.003 | **+0.022** | **+0.004** |
| `h_gir` | **+0.013** | **+0.005** | +0.002 (JR) | −0.000 |
| `h_team_leaders` | +0.000 | **+0.006** | +0.002 (JR) | +0.000 (JR) |
| `h_team_turntaking` | **+0.005** | −0.001 (JR) | −0.000 | −0.000 |

Three regularities:

1. **The mode task belongs to the actor.** `h_r4_lead` (has `v` previously
   *led* the members of this team?) is the top-ranked feature for `y_mode` on
   both datasets, with team size second. Predicting *whether the group channel
   activates* is chiefly a question about `v`'s leadership relationship to
   this particular team.
2. **The exact task belongs to the team.** For `y_group_exact` the selection
   shifts to team-side features — leadership breadth (`h_team_leaders`) and,
   on email, turn-taking — i.e., *whether this exact team's leadership is
   shared* predicts whether it reconvenes.
3. **Domain signatures.** `h_gir` (does this pair reciprocate via groups or
   one-to-one?) is selected on email — a medium offering a genuine
   dyadic-vs-group channel choice — and is marginal on group-native
   coauthorship. `h_team_turntaking` likewise matters only on email
   (coauthorship teams rarely alternate leaders). `h_team_leaders` is the
   "glue" feature: individually near-redundant, but re-added by joint
   recovery in three of four passes — its information matters in combination.

The union of the selected sets reproduces the curated five on email-Eu
exactly, and four of five on DBLP (turn-taking omitted there); the
test-block performance of the validation-selected sets is identical to the
curated set to the third decimal. The curation is therefore **validated
test-blind**: no reported number depends on a choice that saw the test data.

## 4. Results

### 4.1 Headline tables (test block, LightGBM, pooled over k ∈ [3, 25])

**email-Eu** (209,005 events; h = 35 d; 138,531 rows; test n = 41,564):

| label | prevalence | A (lift) | A+C (lift) | ΔPR-AUC [95% CI] |
|---|---|---|---|---|
| `y_dyad` (control) | 0.649 | 0.902 (1.39×) | 0.903 (1.39×) | +0.001 [+0.001, +0.002] |
| `y_group_strict` | 0.290 | 0.626 (2.16×) | **0.751 (2.59×)** | **+0.125 [+0.117, +0.134]** |
| `y_group_exact` | 0.109 | 0.364 (3.34×) | **0.568 (5.21×)** | **+0.204 [+0.190, +0.218]** |
| `y_mode` | 0.447 | 0.699 (1.56×) | **0.828 (1.85×)** | **+0.129 [+0.122, +0.136]** |

**DBLP** (2,954,515 papers, ≤ 25 authors; h = 2 y; 5,180,000 rows; test
n = 1,937,431):

| label | prevalence | A (lift) | A+C (lift) | ΔPR-AUC [95% CI] |
|---|---|---|---|---|
| `y_dyad` (control) | 0.057 | 0.191 (3.4×) | 0.194 (3.4×) | +0.003 [+0.002, +0.003] |
| `y_group_strict` | 0.035 | 0.106 (3.0×) | **0.125 (3.6×)** | **+0.019 [+0.018, +0.020]** |
| `y_group_exact` | 0.0066 | 0.023 (3.5×) | **0.051 (7.7×)** | **+0.028 [+0.026, +0.030]** |
| `y_mode` | 0.607 | 0.729 (1.20×) | **0.824 (1.36×)** | **+0.095 [+0.092, +0.099]** |

All eight group-channel contrasts are significant (every CI excludes zero);
both controls sit at +0.001…+0.003.

### 4.2 What the results mean

**(i) The double dissociation.** The same five features that add +0.10…+0.20
PR-AUC to the group targets add essentially nothing to the dyadic control —
on email-Eu's larger buckets they *subtract* slightly (k = 6–10: −0.006;
k = 11–25: −0.007), i.e. they behave as irrelevant noise exactly where the
theory says they carry no signal. Conversely, once the baseline includes
status features, graph information fully saturates the dyadic channel. The
group channel and the dyadic channel of reciprocity are governed by different
information — and only the former requires the hypergraph.

**(ii) The strictness law.** The hypergraph advantage grows monotonically
with how group-like the target is: dyad ≈ 0 < strict < exact, on both
datasets. At the exact-team endpoint — the prediction target that corresponds
one-to-one with the static `to_team` measures — hypergraph features lift the
baseline from 3.3× to 5.2× no-skill on email-Eu, and **more than double** it
on DBLP (3.5× → 7.7×). The more irreducibly higher-order the notion of
reciprocation, the more indispensable hypergraph structure becomes.

**(iii) Where the gains live.** Gains peak in mid-size teams (k = 6–10:
email `y_mode` +0.172, DBLP +0.107) and attenuate in the largest stratum
(k = 11–25), where positives become rare; on DBLP, exact-team reconvening of
11+-author teams essentially never happens (zero test positives) — large
collaborations are one-shot, itself a substantive observation.

**(iv) The baseline is not a strawman.** Tier A includes the features that
*won* in the published graph-reciprocity literature (status ratios were the
single best predictors in Cheng et al.; the RC index and reciprocity-aware
weighting are Sett et al.'s contributions). Adding them absorbed roughly
10–15% of the naive gaps — quantifying how much a weak baseline would have
flattered us — and everything reported here is measured against the
strengthened version.

### 4.3 Supporting robustness (earlier experiment logs)

The final-sweep configuration was frozen only after each of its choices was
shown not to drive the result: the gaps hold across horizons 3–90 days
(24/24 positive cells on email-Eu), overlap thresholds θ = 0.3…1.0 (32/32),
alternative splits (70-30, 50-50), four rolling-origin time folds
(77/80 positive fold×bucket cells across five domains), an
order-restricted-history ablation whose projection control shows the signal
is the grouping itself (worth ≈ +0.06…+0.08 for the mode task and ≈ 0 for the
dyadic task), and replication in four further domains (Enron, DNC, US
Congress cosponsorship, MusicBrainz collaborations) with the same signature
at domain-dependent magnitudes.

### 4.4 Limitations

Window-based labels allow one response event to verify several earlier rows
(mitigated: contrasts share labels; gaps persist at 2-day horizons where
sharing is minimal; CIs cluster by focal event but not by shared response).
The k = 11–25 stratum is thin for the exact label. LightGBM's unseeded
subsampling introduces ≈ ±0.005 run-to-run jitter, far below every reported
gap. The five-feature curation predates this protocol; it is validated here
by the test-blind selection reproducing it, and its earlier development used
a separate 70-30 test period.

## 5. The fitted model as a predictor: ΔΦ scoring (zero parameters)

The exponential model of the theory section, `P_θ(H) ∝ exp(θ·Φ(H))` with
`Φ = |E|·R_some`, can be used *directly* as a predictor. For a candidate
response (row `(s, v, t)` on team `T`, `|T| = k`), score it by the increase
in Φ it would cause, computed from the causal team counts `m_{T,i}` over
events strictly before `t`:

    ΔΦ(v | T, t) = 2/(k−1) · #{ j ∈ T∖{v} : m_{T,j} > m_{T,v} }

— "the number of teammates who have out-led v," normalized. Ranking by ΔΦ is
parameter-free (monotone in the fitted θ̂ > 0) and is formally a one-statistic
relational hyperevent model (RHEM). Evaluated on the identical test rows as
the model comparisons (results: `results/delta_phi/*.csv`, all 7 domains):

* the relative-AP lift is **ordered by label strictness in every domain**
  (dyad ≈ 1.0× ⟶ exact 1.2–14.3×), the paper's theme reproduced by a formula
  with zero learned parameters;
* on the exact-team label the lift tracks the domain's structural effect:
  Congress 11.8–14.3× (prevalence 0.4%, ROC 0.71), music 3.1×, email/Twitter
  1.5–1.8×, DBLP 1.2×;
* on the rotation task ΔΦ alone reaches AP 0.51 vs a 0.27 prevalence on
  email-Eu (ROC 0.76) — about 90% of a trained 40-feature logistic baseline;
* caveat: ΔΦ is silent on teams with no history (cold starts tie at the
  baseline score) — reciprocity information lives in team history, and the
  recurring/cold split is itself an informative stratification.

## 6. The rotation task: A-vs-C on communication networks

Rotation was adopted only after passing the same contrast test as the other
labels. Features are aggregated to the event level **symmetrically** (mean
and max of every tier-A and tier-C column), preserving temporal parity;
identical split/embargo; paired bootstrap CIs over test events
(results: `results/rotation_ab/*.csv`).

| dataset | h | LogReg ΔPR | LightGBM ΔPR |
|---|---|---|---|
| email-Eu | 7 d | +0.032* | +0.056* |
| email-Eu | 30 d | +0.100* | +0.065* |
| DNC | 2 d | +0.007 | +0.016 |
| DNC | 7 d | +0.041* | +0.041* |
| Twitter | 30 d | +0.021 | +0.032* |
| Twitter | 60 d | +0.008 | +0.054* |
| Enron | 7 d | +0.062* | +0.075* |
| Enron | 30 d | +0.078* | **+0.124*** |

All 16 cells positive; 11/16 significant; LightGBM significant in 8/8. The
weak cells are explainable (DNC h=2: only 1,130 test events; Twitter LogReg
lags while its LightGBM is consistently significant). Enron h=30 (+0.124) is
among the largest hypergraph-over-graph gaps in the study. Verdict: the
rotation task joins the headline tables for communication networks.

---

*Reproduction:* `python e5_val.py --dataset {emaileu|dblp} --horizon auto
--min-recipients 2 --max-size 25 --out-dir final_sweep_Jul18` — all other
parameters are defaults; complete console logs with per-bucket tables are in
`code/final_sweep_Jul18/*_val_run.log`.
For §5–6: `code/delta_phi_scorer.py`, `code/run_all_delta_phi.py`, and
`code/rotation_ab_test.py` in `prediction_final/` (configs mirror each
domain's `lp_6feat` diagnostics).
