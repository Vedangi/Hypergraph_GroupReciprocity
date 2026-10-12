#!/usr/bin/env python3
"""Compute publication-ready statistics for the paper's hypergraph datasets.

The analysis sample contains group events whose team size is

    min_size <= k = 1 + |R| <= max_size,

where an event is ``(sender, recipients, timestamp)``.  Exact duplicate
oriented hyperedges are retained as separate events in the multihypergraph;
``n_support_edges`` counts them once after timestamps are ignored.

Examples
--------
From the repository root::

    python src/dataset_statistics.py
    python src/dataset_statistics.py --csv results/data_statistics_k25.csv \
        --latex results/data_statistics_k25.tex

All statistics except ``n_group_events_uncapped`` and ``retained_pct`` are
computed on the filtered analysis sample.  The graph-reciprocity columns use
the directed dyadic projection of those retained group events.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import datasets  # noqa: E402  (local src/datasets.py)


DATASET_ORDER = (
    "emaileu",
    "enron",
    "fauci",
    "dnc",
    "radoslaw",
    "twitter",
    "congress",
    "dblp",
    "music",
)

DATASET_LABELS = {
    "emaileu": "Email-Eu",
    "enron": "Enron",
    "fauci": "Fauci",
    "dnc": "DNC",
    "radoslaw": "Radoslaw",
    "twitter": "Twitter",
    "congress": "Congress-bills",
    "dblp": "DBLP",
    "music": "Music",
}

DOMAINS = {
    "emaileu": "Communication",
    "enron": "Communication",
    "fauci": "Communication",
    "dnc": "Communication",
    "radoslaw": "Communication",
    "twitter": "Communication",
    "congress": "Collaboration",
    "dblp": "Collaboration",
    "music": "Collaboration",
}


def canonicalize_events(
    events: Iterable[tuple[int, Sequence[int], int]],
) -> list[tuple[int, tuple[int, ...], int]]:
    """Remove within-event self-loops/recipient repeats, but keep events."""
    clean = []
    for sender, recipients, timestamp in events:
        sender = int(sender)
        rec = tuple(sorted({int(v) for v in recipients if int(v) != sender}))
        if rec:
            clean.append((sender, rec, int(timestamp)))
    return clean


def graph_reciprocity(
    events: Iterable[tuple[int, Sequence[int], int]],
) -> tuple[float, float, int, int]:
    """Binary and weighted reciprocity of the directed dyadic projection.

    Binary reciprocity is the fraction of distinct directed arcs whose reverse
    arc exists.  Weighted reciprocity is ``sum_ij min(w_ij, w_ji) / sum_ij
    w_ij``.  Both sums are over ordered arcs.
    """
    weights: Counter[tuple[int, int]] = Counter()
    for sender, recipients, _timestamp in events:
        for recipient in recipients:
            weights[(sender, int(recipient))] += 1

    n_arcs = len(weights)
    total_weight = sum(weights.values())
    binary = (
        sum((v, u) in weights for u, v in weights) / n_arcs
        if n_arcs
        else math.nan
    )
    weighted = (
        sum(min(w, weights.get((v, u), 0)) for (u, v), w in weights.items())
        / total_weight
        if total_weight
        else math.nan
    )
    return binary, weighted, n_arcs, total_weight


def summarize_dataset(name: str, min_size: int, max_size: int) -> dict:
    """Load one dataset and summarize its size-filtered group events."""
    raw = canonicalize_events(datasets.load_dataset(name))
    group_uncapped = [event for event in raw if 1 + len(event[1]) >= min_size]
    kept = [
        event
        for event in group_uncapped
        if 1 + len(event[1]) <= max_size
    ]
    if not kept:
        raise ValueError(
            f"{name}: no events satisfy {min_size} <= k <= {max_size}"
        )

    nodes = {node for sender, recipients, _ in kept for node in (sender, *recipients)}
    support_edges = {(sender, recipients) for sender, recipients, _ in kept}
    teams = {frozenset((sender, *recipients)) for sender, recipients, _ in kept}
    team_sizes = [1 + len(recipients) for _sender, recipients, _time in kept]
    graph_binary, graph_weighted, n_arcs, projected_weight = graph_reciprocity(kept)

    n_uncapped = len(group_uncapped)
    n_events = len(kept)
    n_support = len(support_edges)
    return {
        "domain": DOMAINS[name],
        "dataset_key": name,
        "dataset": DATASET_LABELS[name],
        "n_nodes": len(nodes),
        "n_group_events_uncapped": n_uncapped,
        "n_events": n_events,
        "retained_pct": 100.0 * n_events / n_uncapped if n_uncapped else math.nan,
        "n_support_edges": n_support,
        "n_teams": len(teams),
        "duplicate_pct": 100.0 * (n_events - n_support) / n_events,
        "mean_team_size": sum(team_sizes) / n_events,
        "median_team_size": float(median(team_sizes)),
        "n_projected_arcs": n_arcs,
        "projected_weight": projected_weight,
        "graph_binary_reciprocity": graph_binary,
        "graph_weighted_reciprocity": graph_weighted,
        "min_team_size": min(team_sizes),
        "max_team_size": max(team_sizes),
    }


def latex_table(frame: pd.DataFrame, min_size: int, max_size: int) -> str:
    """Return a compact booktabs table for the manuscript."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Dataset statistics for the group interactions used in the "
        rf"structural analysis. We retain events with ${min_size}\leq k\leq {max_size}$. "
        r"$M_{\rm all}$ is the number of group events before the upper-size "
        r"restriction; $M$ and all remaining statistics are computed after "
        r"filtering. $|E^*|$ is the number of distinct oriented hyperedges, "
        r"and $|\mathcal{T}|$ is the number of distinct participating teams.}",
        r"\label{tab:dataset-statistics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrrr}",
        r"\toprule",
        r"Domain & Dataset & $|V|$ & $M_{\rm all}$ & $M$ & Retained (\%) & "
        r"$|E^*|$ & $|\mathcal{T}|$ & Duplicates (\%) & Mean $k$ & "
        r"$r_{\rm bin}$ & $r_{\rm wt}$ \\",
        r"\midrule",
    ]

    previous_domain = None
    for row in frame.itertuples(index=False):
        if previous_domain is not None and row.domain != previous_domain:
            lines.append(r"\midrule")
        domain = row.domain if row.domain != previous_domain else ""
        lines.append(
            f"{domain} & {row.dataset} & "
            f"{row.n_nodes:,} & {row.n_group_events_uncapped:,} & "
            f"{row.n_events:,} & {row.retained_pct:.2f} & "
            f"{row.n_support_edges:,} & {row.n_teams:,} & "
            f"{row.duplicate_pct:.1f} & {row.mean_team_size:.2f} & "
            f"{row.graph_binary_reciprocity:.3f} & "
            f"{row.graph_weighted_reciprocity:.3f} \\\\"
        )
        previous_domain = row.domain

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DATASET_ORDER),
        help="dataset keys to summarize (default: all paper datasets)",
    )
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=25)
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional path for the complete statistics CSV",
    )
    parser.add_argument(
        "--latex",
        type=Path,
        help="optional path for the compact LaTeX table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_size < 2 or args.max_size < args.min_size:
        raise SystemExit("require 2 <= --min-size <= --max-size")

    rows = [
        summarize_dataset(name, args.min_size, args.max_size)
        for name in args.datasets
    ]
    frame = pd.DataFrame(rows)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
    table = latex_table(frame, args.min_size, args.max_size)
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(table, encoding="utf-8")

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(frame.to_string(index=False))
    print("\n" + table)


if __name__ == "__main__":
    main()
