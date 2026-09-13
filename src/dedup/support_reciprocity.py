#!/usr/bin/env python3
r"""support_reciprocity.py -- the support (deduplicated) team-contribution
measures, in BOTH aggregation forms, extended to multihypergraphs.

Notation: for team T let p_T = number of DISTINCT members that appear as the
sender of some event on T ("contributors"), k_T = |T|, and M_T = total number
of event INSTANCES on T (duplicates counted).  On the support hypergraph H*
(duplicates removed) the paper defines, summing over teams:

    R_any (H*) = (1/|E*|) Sum_T  p_T 1[p_T >= 2]
    R_part(H*) = (1/|E*|) Sum_T  p_T (p_T - 1) / (k_T - 1)
    R_all (H*) = (1/|E*|) Sum_T  p_T 1[p_T = k_T]

with |E*| = Sum_T p_T.  Equivalently, summing over DISTINCT hyperedges: each
of the p_T distinct (sender, T) edges contributes f(p_T) with
f in { 1[p>=2],  (p-1)/(k-1),  1[p=k] }.

**Extension to multihypergraphs** (duplicate events kept, as in the MCMC
state space): replace the leading p_T by M_T,

    R_x^{M}(H) = (1/|E|) Sum_T  M_T f(p_T),        |E| = Sum_T M_T,

which is IDENTICALLY equal to the per-event-instance summation
Sum_{events e} f(p_{T(e)}) / |E|  (group the event sum by team).  On a
duplicate-free hypergraph M_T = p_T and both reduce to the paper's formulas.
So the support measures are well defined on every state the sampler visits;
`weighting="p"` gives the support-of-the-state convention used by
multi_reciprocity.measures (the *_support columns in all sweep CSVs), and
`weighting="M"` gives the event-weighted extension.

Caveat (the "acts a bit weird" part): the extension inherits THRESHOLD
semantics -- f depends only on p_T, never on how the M_T instances are
distributed.  The dyad with multiplicities (100, 1) has R_any^{M} = 1.0
(all 101 events sit on a p=2 team) while the multiplicity-aware
s1_multi = 2/101: volume-weighted support credit is not matched credit.
"""
from __future__ import annotations

from collections import Counter, defaultdict

F = {
    "R_any":  lambda p, k: 1.0 if p >= 2 else 0.0,
    "R_part": lambda p, k: (p - 1) / (k - 1) if k > 1 else 0.0,
    "R_all":  lambda p, k: 1.0 if p == k else 0.0,
}
ALIAS = {"R_any": "s1", "R_part": "r2", "R_all": "r1"}


def team_stats(events, min_k=2):
    """{team: (M_T, p_T, k_T)} over teams with k >= min_k."""
    m = defaultdict(Counter)
    for s, R in events:
        m[frozenset({s}) | frozenset(R)][s] += 1
    return {T: (sum(c.values()), len(c), len(T))
            for T, c in m.items() if len(T) >= min_k}


def support_measures_team(events, min_k=2, weighting="M"):
    """Team-summation form.  weighting="M": event-weighted extension
    (identical to per-event summation).  weighting="p": the paper's
    support-hypergraph formula (= multi_reciprocity's *_support)."""
    if weighting not in ("M", "p"):
        raise ValueError("weighting must be 'M' or 'p'")
    st = team_stats(events, min_k)
    den = sum((M if weighting == "M" else p) for M, p, _ in st.values())
    out = {}
    for key, f in F.items():
        num = sum((M if weighting == "M" else p) * f(p, k)
                  for M, p, k in st.values())
        out[key] = num / den if den else 0.0
        out[ALIAS[key] + "_support_" + weighting] = out[key]
    out["denominator"] = den
    out["weighting"] = weighting
    return out


def support_measures_events(events, min_k=2):
    """Per-event-instance summation: each event contributes f(p_T) for its
    team.  Provably equal to support_measures_team(..., weighting="M")."""
    st = team_stats(events, min_k)
    team_of = {}
    for s, R in events:
        T = frozenset({s}) | frozenset(R)
        if T in st:
            team_of.setdefault(T, st[T])
    num = defaultdict(float)
    den = 0
    for s, R in events:
        T = frozenset({s}) | frozenset(R)
        if T not in st:
            continue
        _M, p, k = st[T]
        den += 1
        for key, f in F.items():
            num[key] += f(p, k)
    return {key: (num[key] / den if den else 0.0) for key in F} | {
        "denominator": den}


def support_measures_dedup_edges(events, min_k=2):
    """Per-DISTINCT-hyperedge summation (each distinct (s, T) contributes
    f(p_T)).  Provably equal to weighting="p"."""
    st = team_stats(events, min_k)
    seen = set()
    num = defaultdict(float)
    den = 0
    for s, R in events:
        T = frozenset({s}) | frozenset(R)
        if T not in st or (s, T) in seen:
            continue
        seen.add((s, T))
        _M, p, k = st[T]
        den += 1
        for key, f in F.items():
            num[key] += f(p, k)
    return {key: (num[key] / den if den else 0.0) for key in F} | {
        "denominator": den}


# ------------------------------------------------------------------ self-test
def _selftest():
    import random
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "multihypergraph"))
    import multi_reciprocity as MR
    import reciprocity_dedup as RD
    rng = random.Random(3)

    for trial in range(30):
        ev = []
        for _ in range(rng.randrange(40, 200)):
            s = rng.randrange(10)
            R = rng.sample([x for x in range(10) if x != s],
                           rng.randrange(1, 5))
            ev.append((s, sorted(R)))
            if rng.random() < 0.4:            # inject duplicates
                ev.append((s, sorted(R)))

        tm = support_measures_team(ev, weighting="M")
        pe = support_measures_events(ev)
        for k in F:
            assert abs(tm[k] - pe[k]) < 1e-12, (k, tm[k], pe[k])

        tp = support_measures_team(ev, weighting="p")
        de = support_measures_dedup_edges(ev)
        for k in F:
            assert abs(tp[k] - de[k]) < 1e-12

        # weighting="p" == multi_reciprocity's *_support columns
        m = MR.measures(ev, min_k=2)
        assert abs(tp["R_any"] - m["s1_support"]) < 1e-12
        assert abs(tp["R_all"] - m["r1_support"]) < 1e-12
        assert abs(tp["R_part"] - m["r2_support"]) < 1e-12

        # weighting="p" R_part == reciprocity_dedup.grant_r2 (dedups inside)
        assert abs(tp["R_part"] - RD.grant_r2(ev)[1]) < 1e-9

        # duplicate-free input: M == p, all four coincide
        dd = list({(s, tuple(R)) for s, R in ev})
        dd = [(s, list(R)) for s, R in dd]
        a = support_measures_team(dd, weighting="M")
        b = support_measures_team(dd, weighting="p")
        for k in F:
            assert abs(a[k] - b[k]) < 1e-12

    # the pathology exhibit: dyad with multiplicities (100, 1)
    ev = [(0, [1])] * 100 + [(1, [0])]
    x = support_measures_team(ev, min_k=2, weighting="M")
    m = MR.measures(ev, min_k=2)
    assert abs(x["R_any"] - 1.0) < 1e-12
    assert abs(m["s1_multi"] - 2 / 101) < 1e-12
    print("selftest OK: team(M)==events, team(p)==dedup-edges=="
          "multi_reciprocity *_support==grant_r2; dedup input collapses all; "
          f"(100,1) dyad: R_any^M={x['R_any']:.3f} vs s1_multi={m['s1_multi']:.4f}")


if __name__ == "__main__":
    _selftest()

# How this was run:
#   <venv-python> support_reciprocity.py     # executes _selftest()
# Library file otherwise -- import support_measures_team / _events / _dedup_edges.
