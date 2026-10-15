#!/usr/bin/env python3
"""Appendix diagnostics table for the beta sweeps. Render-only: reads
results/final/beta_sweep_summary.csv (multiplicity-aware tilt, phase A/B) and
beta_sweep_summary_support_tilt.csv (support-family tilt, phase D/T), written
by make_beta_figure.py, and emits one row per (family, dataset, beta):
chains, sweeps per chain, acceptance rate, tau_int(Phi), pooled ESS(Phi),
max split-Rhat over the three reported measures, E_beta[R_part] and its MCSE.
Rows with beta > --plot-max are marked 'diagnostic only'.
-> results/final/beta_sweep_diagnostics.{md,tex}
"""
import argparse, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"emaileu": "Email-Eu", "enron": "Enron", "twitter": "Twitter", "congress": "Congress-bills"}
FAMS = [("beta_sweep_summary.csv", "r2_multi", r"multiplicity-aware ($\Phi=|E|\,R^{\rm m}_{\rm part}$)"),
        ("beta_sweep_summary_support_tilt.csv", "r2_support_M",
         r"support ($\Phi=|E^*|\,R^{M}_{\rm part}$, dedup start)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "final"))
    p.add_argument("--plot-max", type=float, default=3.0)
    a = p.parse_args()
    root = os.path.abspath(a.out)
    md = ["# Beta-sweep diagnostics (25% burn-in per chain; 4 chains per beta, 6 at beta = 0)", "",
          "| family | dataset | β | chains | sweeps/chain | accept | τ(Φ) | ESS(Φ) | max R̂ | E_β[R_part] | MCSE | plotted |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{table*}[t]", r"\centering",
           r"\caption{Beta-sweep diagnostics. Four chains per $\beta$ (two from the observed state, "
           r"two from independent random realizations of the same projection; six at $\beta=0$), "
           r"25\% burn-in per chain. $\tau$: integrated autocorrelation time of $\Phi$ in sweeps; "
           r"ESS: pooled effective sample size of $\Phi$; $\widehat R$: largest split-$\widehat R$ "
           r"over the three reported measures; MCSE: Monte Carlo standard error of "
           r"$\mathbb E_\beta[R_{\rm part}]$. Points with $\beta>" + f"{a.plot_max:g}" +
           r"$ are retained as diagnostics only and not plotted.}",
           r"\label{tab:beta-diag}", r"\small", r"\begin{tabular}{llrrrrrrrrrc}", r"\toprule",
           r"family & dataset & $\beta$ & chains & sweeps & acc. & $\tau(\Phi)$ & ESS($\Phi$) & "
           r"max $\widehat R$ & $\mathbb E_\beta[R_{\rm part}]$ & MCSE & plotted \\", r"\midrule"]
    for f, prim, label in FAMS:
        path = os.path.join(root, f)
        if not os.path.exists(path):
            continue
        t = pd.read_csv(path)
        t = t[t.measure == prim].sort_values(["dataset", "beta"])
        first_fam = True
        for ds in NAMES:
            g = t[t.dataset == ds]
            first_ds = True
            for r in g.itertuples():
                sw = int(round(r.n_states / r.n_chains / 0.75))
                plotted = r.beta <= a.plot_max
                fam_md = label if first_fam else ""
                ds_md = NAMES[ds] if first_ds else ""
                md.append(f"| {fam_md} | {ds_md} | {r.beta:+.0f} | {r.n_chains} | {sw} | {r.accept_rate:.2f} | "
                          f"{r.Phi_tau:.1f} | {r.Phi_ess:.0f} | {r.rhat_max_measures:.3f} | "
                          f"{r.mean:.4f} | {r.mcse:.1e} | {'yes' if plotted else 'diagnostic only'} |")
                tex.append(f"{fam_md if first_fam else ''} & {ds_md} & ${r.beta:+.0f}$ & {r.n_chains} & {sw} & "
                           f"{r.accept_rate:.2f} & {r.Phi_tau:.1f} & {r.Phi_ess:.0f} & {r.rhat_max_measures:.3f} & "
                           f"{r.mean:.4f} & {r.mcse:.1e} & {'yes' if plotted else 'no'} \\\\")
                first_fam = first_ds = False
            if len(g):
                tex.append(r"\addlinespace[2pt]")
        tex.append(r"\midrule")
    tex = tex[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    for name, body in (("beta_sweep_diagnostics.md", md), ("beta_sweep_diagnostics.tex", tex)):
        out = os.path.join(root, name)
        open(out, "w").write("\n".join(body) + "\n")
        print("[saved]", out)


if __name__ == "__main__":
    main()

# How this was run (2026-10-15):
#   <venv-python> make_beta_diag_table.py --plot-max 3
# Render-only: reads results/final/beta_sweep_summary*.csv (make_beta_figure.py);
# writes results/final/beta_sweep_diagnostics.{md,tex}.
