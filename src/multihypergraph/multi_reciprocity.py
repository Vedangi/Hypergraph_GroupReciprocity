#!/usr/bin/env python3
"""multi_reciprocity.py -- reciprocity scores for single-sender directed
MULTI-hypergraphs (duplicate hyperedges allowed), implementing
docs/measures.md.

An event is (sender s, recipient list R); duplicates are kept, timestamps are
ignored.  For each team T = {s} u R let

    m_{T,i} = # events (i -> T\\{i})          (sender-role multiplicity)
    M_T     = sum_i m_{T,i}                   (event occurrences on T)
    q_T     = #{i : m_{T,i} > 0}              (distinct observed senders)
    k_T     = |T|

MEASURES (event-weighted / "micro" unless noted; all in [0,1])

  support (deduplicated view; = the original R_any/R_all/R_some on the support
  hypergraph, event-weighted over its Sum_T q_T distinct directed hyperedges):
      s1_support = Sum_T q 1{q>=2}          / Sum_T q
      r1_support = Sum_T q 1{q=k}           / Sum_T q
      r2_support = Sum_T q (q-1)/(k-1)      / Sum_T q

  multiplicity-aware (md SS10-12; denominator |E| = Sum_T M_T = #events):
      s1_multi = Sum_T Sum_i min(m_i, M_T - m_i)            / |E|
      r1_multi = Sum_T k_T min_i m_i                        / |E|
      r2_multi = Sum_T (2/(k_T-1)) Sum_{i<j} min(m_i, m_j)  / |E|

  Reductions (md SS13, asserted in the self-test):
      * 0/1 multiplicities  -> multi == support, exactly;
      * all teams dyadic    -> all three == directed-multigraph reciprocity
                               2 Sum min(m_ij, m_ji) / Sum (m_ij + m_ji).

  graph projection (both invariant under within-sender switches):
      graph_binary   = |{(i,j): both directions present}| / |{(i,j)}|
      graph_weighted = Sum min(w_ij, w_ji) / Sum w_ij      (Squartini et al.;
                       w = directed contact counts incl. duplicates)

Also provided: macro (per-team) aggregation and a per-team table.
"""
from __future__ import annotations

from collections import Counter, defaultdict


# --------------------------------------------------------------- core profile
def team_profiles(events):
    """events: iterable of (sender, recipients).  Returns
    {frozenset team: Counter{sender: multiplicity}} with duplicates counted."""
    prof = defaultdict(Counter)
    for s, R in events:
        s = int(s)
        recs = frozenset(int(x) for x in R)
        if not recs or s in recs:
            raise ValueError(f"bad event ({s}, {sorted(recs)})")
        prof[frozenset(recs | {s})][s] += 1
    return prof


# ------------------------------------------------------------ per-team scores
def team_scores(team, cnt):
    """Local scores for one team.  cnt: Counter{sender: m>0}.  md SS10-12."""
    k = len(team)
    m = list(cnt.values())
    M = sum(m)
    q = len(m)
    mx = max(m)
    s1m = min(M, 2 * (M - mx)) / M                      # closed form of SS10
    r1m = k * (min(m) if q == k else 0) / M
    pair = sum(min(a, b) for i, a in enumerate(m) for b in m[i + 1:])
    r2m = 2.0 * pair / ((k - 1) * M)
    return {"k": k, "M": M, "q": q, "m_max": mx,
            "s1_multi": s1m, "r1_multi": r1m, "r2_multi": r2m,
            "s1_support": 1.0 if q >= 2 else 0.0,
            "r1_support": 1.0 if q == k else 0.0,
            "r2_support": (q - 1) / (k - 1)}


# --------------------------------------------------------------- projections
def graph_binary_reciprocity(events):
    seen = set()
    for s, R in events:
        for r in R:
            seen.add((int(s), int(r)))
    return sum((b, a) in seen for a, b in seen) / len(seen) if seen else 0.0


def graph_weighted_reciprocity(events):
    """Squartini r_w = Sum min(w_ij, w_ji) / Sum w_ij on the multigraph
    projection (w = directed contact counts, duplicates included)."""
    w = Counter()
    for s, R in events:
        for r in R:
            w[(int(s), int(r))] += 1
    W = sum(w.values())
    Wr = sum(min(c, w.get((b, a), 0)) for (a, b), c in w.items())
    return Wr / W if W else 0.0


def multigraph_dyadic_reciprocity(events):
    """md SS5 on the DYADIC teams only (k=2 slice): 2 Sum min / Sum (m+m)."""
    prof = team_profiles(events)
    num = den = 0
    for T, cnt in prof.items():
        if len(T) != 2:
            continue
        a, b = sorted(T)
        num += 2 * min(cnt.get(a, 0), cnt.get(b, 0))
        den += sum(cnt.values())
    return num / den if den else float("nan")


# ------------------------------------------------------------------ measures
def measures(events, min_k=2):
    """All global scores.  min_k filters TEAMS by size (use 3 for group-only)."""
    prof = {T: c for T, c in team_profiles(events).items() if len(T) >= min_k}
    if not prof:
        raise ValueError("no teams after filtering")
    num = defaultdict(float)
    E = Q = 0
    macro = defaultdict(float)
    for T, cnt in prof.items():
        ts = team_scores(T, cnt)
        M, q = ts["M"], ts["q"]
        E += M
        Q += q
        for key in ("s1_multi", "r1_multi", "r2_multi"):
            num[key] += ts[key] * M                    # event-weighted
            macro[key] += ts[key]
        for key in ("s1_support", "r1_support", "r2_support"):
            num[key] += ts[key] * q                    # support-edge-weighted
    nT = len(prof)
    out = {f"{k}": num[k] / (E if k.endswith("multi") else Q)
           for k in num}
    out.update({f"{k}_macro": macro[k] / nT for k in macro})
    out.update({"n_events": E, "n_support_edges": Q, "n_teams": nT,
                "graph_binary": graph_binary_reciprocity(events),
                "graph_weighted": graph_weighted_reciprocity(events)})
    return out


def per_team_table(events, min_k=2):
    import pandas as pd
    rows = []
    for T, cnt in team_profiles(events).items():
        if len(T) < min_k:
            continue
        rows.append({"team": tuple(sorted(T)), **team_scores(T, cnt)})
    return pd.DataFrame(rows).sort_values(["k", "M"], ascending=[True, False])


# ------------------------------------------------------------------ self-test
def _selftest():
    # md SS15 worked examples ------------------------------------------------
    ev = [(0, [1])] * 100 + [(1, [0])]                     # dyad (100,1)
    ts = team_scores(frozenset({0, 1}), team_profiles(ev)[frozenset({0, 1})])
    assert abs(ts["s1_multi"] - 2 / 101) < 1e-12
    assert abs(ts["r2_multi"] - 2 / 101) < 1e-12

    ev = [(0, [1, 2])] * 100 + [(1, [0, 2])] + [(2, [0, 1])]   # (100,1,1)
    ts = team_scores(frozenset({0, 1, 2}),
                     team_profiles(ev)[frozenset({0, 1, 2})])
    assert abs(ts["s1_multi"] - 4 / 102) < 1e-12
    assert abs(ts["r1_multi"] - 3 / 102) < 1e-12
    assert abs(ts["r2_multi"] - 3 / 102) < 1e-12

    ev = ([(0, [1, 2, 3])] * 8 + [(1, [0, 2, 3])] * 3          # (8,3,1,0)
          + [(2, [0, 1, 3])])
    ts = team_scores(frozenset({0, 1, 2, 3}),
                     team_profiles(ev)[frozenset({0, 1, 2, 3})])
    assert abs(ts["s1_multi"] - 8 / 12) < 1e-12
    assert ts["r1_multi"] == 0.0
    assert abs(ts["r2_multi"] - 10 / 36) < 1e-12

    # SS13.1: 0/1 multiplicities -> multi == support -------------------------
    ev = [(0, [1, 2]), (1, [0, 2]), (0, [2, 3]), (3, [0, 2]), (2, [0, 3]),
          (4, [0, 5, 6])]
    m = measures(ev)
    for a, b in [("s1_multi", "s1_support"), ("r1_multi", "r1_support"),
                 ("r2_multi", "r2_support")]:
        assert abs(m[a] - m[b]) < 1e-12, (a, m[a], m[b])

    # SS13.2: all-dyadic -> all three == multigraph reciprocity --------------
    ev = [(0, [1])] * 5 + [(1, [0])] * 2 + [(2, [3])] * 4 + [(3, [2])] * 4 \
        + [(4, [5])]
    m = measures(ev)
    rmg = multigraph_dyadic_reciprocity(ev)
    for k in ("s1_multi", "r1_multi", "r2_multi"):
        assert abs(m[k] - rmg) < 1e-12, (k, m[k], rmg)
    assert abs(rmg - (2 * (2 + 4 + 0)) / (7 + 8 + 1)) < 1e-12

    print("all self-tests passed")


if __name__ == "__main__":
    _selftest()
