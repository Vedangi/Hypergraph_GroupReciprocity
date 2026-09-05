"""reci_faster.py  --  OPTIMIZED drop-in for reciprocity_lib.

Identical API and results to reciprocity_lib, but much faster on big data:
  * shuffle_pairwise_preserving_by_sender_swaps: O(1)-per-swap 2-switch
    (python RNG + list/set per event; no numpy choice, no set-difference).
  * to_team measure: group emails indexed by [sender][target] + short-circuit
    (scans only a node's group emails that actually contain s).

Use exactly like reciprocity_lib:   from reciprocity_lib_faster import *
Then call the functions on your df_events (columns: sender, recipients[list]).

Contains:
  * Nate's r1-r4 (faithful port)              compute_advisor_reciprocity_measures / _profiles
  * graph reciprocity                          graph_reciprocity / weighted_graph_reciprocity
  * simple group measures s1/s2/s3             compute_group_reciprocity_measures / _profiles
  * same-projection generator                  same_projection_regroup / projection_signature
  * s-vs-null evaluation helpers               all_reciprocity_measures / evaluate_s_vs_null /
                                               summarize_s_vs_null
  * null-model shuffles                         shuffle_pairwise_preserving_by_sender_swaps
                                               (projection-preserving; the headline null),
                                               shuffle_dinghy_hypergraph (degree-preserving),
                                               shuffle_global_random_hypergraph,
                                               shuffle_global_recipients_only_hypergraph,
                                               shuffle_sender_local_hypergraph,
                                               shuffle_group_emails_preserve_sender_size

Example (in Colab):
    from reciprocity_lib import *
    true_s, null_s = evaluate_s_vs_null(
        df_events, shuffle_pairwise_preserving_by_sender_swaps, n_shuffles=30, seed=42)
    print(summarize_s_vs_null(true_s, null_s).to_string(index=False))
"""

from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ helpers
def _events_from_df(df_events):
    return [(int(r.sender), [int(x) for x in r.recipients])
            for r in df_events.itertuples(index=False)]


def _hyp(events):
    hyp = defaultdict(lambda: defaultdict(set))      # hyp[k][frozenset(nodeset)] = {distinct senders}
    for s, R in events:
        node = frozenset([s] + list(R))
        hyp[len(node)][node].add(s)
    return hyp


def _kmax(hyp, maxedgesize):
    return min(max(hyp) if hyp else 1, maxedgesize)


# ------------------------------------------------ Nate's r1-r4 (events-based)
def full_participation(events, maxedgesize=1000):              # r1
    hyp = _hyp(events); num = den = 0.0; recvec = {}
    for k in range(2, _kmax(hyp, maxedgesize) + 1):
        numk = denk = 0.0
        for senders in hyp.get(k, {}).values():
            pe = len(senders); denk += pe
            if pe == k:
                numk += k
        recvec[k] = numk / denk if denk else 0.0
        num += numk; den += denk
    return recvec, (num / den if den else 0.0)


def partial_participation(events, maxedgesize=1000):
    # k-weighted r2 (as in the Julia reference implementation).
    # NOTE: this is NOT the paper's partial-participation measure;
    # use grant_r2 below when regenerating paper tables.
    hyp = _hyp(events); num = den = 0.0; recvec = {}
    for k in range(2, _kmax(hyp, maxedgesize) + 1):
        numk = denk = 0.0
        for senders in hyp.get(k, {}).values():
            pe = len(senders); denk += pe
            numk += k * (pe - 1) / (k - 1)
        recvec[k] = numk / denk if denk else 0.0
        num += numk; den += denk
    return recvec, (num / den if den else 0.0)


def pairwise_group(events, maxedgesize=1000):
    hyp = _hyp(events); num = den = 0.0; recvec = {}
    for k in range(2, _kmax(hyp, maxedgesize) + 1):
        numk = denk = 0.0
        for senders in hyp.get(k, {}).values():
            pe = len(senders); denk += pe * (k - 1); numk += pe * (pe - 1)
        recvec[k] = numk / denk if denk else 0.0
        num += numk; den += denk
    return recvec, (num / den if den else 0.0)


def _supporters_by_size(events, maxk):
    by_size = defaultdict(lambda: defaultdict(set))
    for s, R in events:
        k = 1 + len(R)
        if k <= maxk:
            for r in R:
                by_size[k][r].add(s)
    return by_size


def group_leader(events, maxedgesize=1000, maxk=100):           # r4 (same-size leadership)
    hyp = _hyp(events); sup = _supporters_by_size(events, maxk)
    num = den = 0.0; recvec = {}
    for k in range(2, _kmax(hyp, maxedgesize) + 1):
        supk = sup.get(k, {}); numk = denk = 0.0
        for node, senders in hyp.get(k, {}).items():
            denk += len(senders)
            for sender in senders:
                numk += len(supk.get(sender, set()) & node) / (k - 1)
        recvec[k] = numk / denk if denk else 0.0
        num += numk; den += denk
    return recvec, (num / den if den else 0.0)


def full_group_leader(events, maxedgesize=1000, maxk=100):      # r3
    hyp = _hyp(events); sup = _supporters_by_size(events, maxk)
    num = den = 0.0; recvec = {}
    for k in range(2, _kmax(hyp, maxedgesize) + 1):
        supk = sup.get(k, {}); numk = denk = 0.0
        for node, senders in hyp.get(k, {}).items():
            denk += len(senders)
            for sender in senders:
                if len(supk.get(sender, set()) & node) == k - 1:
                    numk += 1
        recvec[k] = numk / denk if denk else 0.0
        num += numk; den += denk
    return recvec, (num / den if den else 0.0)

def grant_r2(events, maxedgesize=1000):
    # The paper's partial-participation formula (base = sum of p_e).
    # This is the variant reported in the Model-B tables.
    hyp = _hyp(events)
    num = den = 0.0
    recvec = {}

    for k in range(2, _kmax(hyp, maxedgesize) + 1):
        numk = denk = 0.0

        for senders in hyp.get(k, {}).values():
            pe = len(senders)

            # There are pe directed hyperedges for this exact team.
            denk += pe
            numk += pe * (pe - 1) / (k - 1)

        recvec[k] = numk / denk if denk else 0.0
        num += numk
        den += denk

    return recvec, num / den if den else 0.0

# ----------------------------------------------------- graph reciprocity
def _dir_edges(events, maxk):
    w = defaultdict(int)
    for s, R in events:
        if 1 + len(R) <= maxk:
            for r in R:
                if r != s:
                    w[(s, r)] += 1
    return w


def graph_reciprocity(events, maxk=100):
    w = _dir_edges(events, maxk); edges = set(w)
    den = len(edges); num = sum(1 for (a, b) in edges if (b, a) in edges)
    return num / den if den else 0.0


def weighted_graph_reciprocity(events, maxk=100):
    w = _dir_edges(events, maxk); den = sum(w.values())
    num = sum(min(c, w.get((b, a), 0)) for (a, b), c in w.items())
    return num / den if den else 0.0


def graph_project_reciprocities(events, maxedgesize=1000, maxk=100):
    by_k = defaultdict(list)
    for s, R in events:
        by_k[1 + len(R)].append((s, R))
    hyp = _hyp(events); recvec = {}
    for k in range(2, _kmax(hyp, maxedgesize) + 1):
        recvec[k] = graph_reciprocity(by_k.get(k, []), maxk)
    return recvec


def compute_advisor_reciprocity_measures(df_events, maxedgesize=1000, maxk=100):
    ev = _events_from_df(df_events)
    return {
        "r1_participation_strict":  full_participation(ev, maxedgesize)[1],
        "grant_r2":                 grant_r2(ev, maxedgesize)[1],   # grant r2 (partial participation, base = sum p_e)
        "r2_participation_partial": partial_participation(ev, maxedgesize)[1],
        "r3_leadership_strict":     full_group_leader(ev, maxedgesize, maxk)[1],
        "r4_leadership_partial":    group_leader(ev, maxedgesize, maxk)[1],
        "pairwise_group":           pairwise_group(ev, maxedgesize)[1],
        "graph_reciprocity":        graph_reciprocity(ev, maxk),
        "weighted_graph_reciprocity": weighted_graph_reciprocity(ev, maxk),
        "n_events": len(ev),
        "n_nodes": len({n for s, R in ev for n in [s] + R}),
    }


def compute_advisor_reciprocity_profiles(df_events, maxedgesize=1000, maxk=100):
    ev = _events_from_df(df_events)
    return {
        "full_participation":    full_participation(ev, maxedgesize)[0],
        "grant_r2":              grant_r2(ev, maxedgesize)[0],
        "partial_participation": partial_participation(ev, maxedgesize)[0],
        "pairwise_group":        pairwise_group(ev, maxedgesize)[0],
        "group_leader":          group_leader(ev, maxedgesize, maxk)[0],
        "full_group_leader":     full_group_leader(ev, maxedgesize, maxk)[0],
        "graph_reduced":         graph_project_reciprocities(ev, maxedgesize, maxk),
    }


# ------------------------------------------------ simple group measures s1/s2/s3
def _group_sent_sets(events):
    gs = defaultdict(set)                  # gs[v] = nodes v sent to in GROUP emails (|R'|>=2)
    for s, R in events:
        if len(R) >= 2:
            for r in R:
                gs[s].add(r)
    return gs


def _group_emails_by_sender(events):
    gi = defaultdict(list)
    for s, R in events:
        if len(R) >= 2:
            gi[s].append(frozenset(R))
    return gi


def _s_accumulate(events, dedup, mode):
    """to_sender: recipient v reciprocates if v sends ANY group email containing s.
    to_team: STRICT same-group -- v reciprocates only if v sends a group email whose
    recipient set is EXACTLY the rest of the team N\\{v} (the original sender s plus
    all co-recipients), i.e. v replies to the same group, no one missing and no one
    extra."""
    gs = _group_sent_sets(events)
    ge = None
    if mode == "to_team":
        ge = defaultdict(set)            # ge[v] = {frozenset(recipients) of v's GROUP emails}
        for v, R in events:
            if len(R) >= 1:
                ge[v].add(frozenset(R))

    def x_of(v, s, N):
        if mode == "to_sender":
            return 1 if s in gs.get(v, ()) else 0
        # to_team: v must reply to the EXACT same group (recipients == N\{v})
        g = ge.get(v)
        return 1 if (g is not None and (N - {v}) in g) else 0

    if dedup:
        hyp = _hyp(events)
        units = ((s, [x for x in node if x != s], node)
                 for k, d in hyp.items() for node, senders in d.items() for s in senders)
    else:
        units = ((s, R, frozenset([s] + R)) for s, R in events)

    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])    # team size k -> [s1, s2, s3, count]
    for s, R, N in units:
        n = len(R)
        if n == 0:
            continue
        ssum = sum(x_of(v, s, N) for v in R)
        a = acc[1 + n]
        a[0] += 1.0 if ssum >= 1 else 0.0
        a[1] += 1.0 if ssum == n else 0.0
        a[2] += ssum / n
        a[3] += 1
    return acc


def compute_group_reciprocity_measures(df_events, dedup=True, min_recipients=1, mode="to_team"): #to_sender
    """s1 (any/OR), s2 (all/AND), s3 (fraction) over GROUP emails. Returns flat dict.
    mode='to_team' uses the STRICT same-group rule (v must reply to exactly N\\{v})."""
    acc = _s_accumulate(_events_from_df(df_events), dedup, mode)
    tot = [0.0, 0.0, 0.0, 0]
    for k in acc:
        if (k - 1) >= min_recipients:
            a = acc[k]
            tot[0] += a[0]; tot[1] += a[1]; tot[2] += a[2]; tot[3] += a[3]
    n = tot[3]
    return {"s1_any": tot[0] / n if n else 0.0,
            "s2_all": tot[1] / n if n else 0.0,
            "s3_frac": tot[2] / n if n else 0.0,
            "n_group_units": n}


def compute_group_reciprocity_profiles(df_events, dedup=True, mode="to_team"):
    """Per team-size profiles. 'n' = number of group units at each size k (the
    denominator), so sparse large-size points can be judged for reliability."""
    acc = _s_accumulate(_events_from_df(df_events), dedup, mode)
    s1, s2, s3, n = {}, {}, {}, {}
    for k in sorted(acc):
        a1, a2, a3, c = acc[k]
        if c:
            s1[k], s2[k], s3[k] = a1 / c, a2 / c, a3 / c
            n[k] = c
    return {"s1": s1, "s2": s2, "s3": s3, "n": n}


# ------------------------------------------------ same-projection generator
def _fill_events(c, sizes, rng):
    remaining = dict(c)
    events = []
    for k in sorted(sizes, reverse=True):
        cands = [r for r, v in remaining.items() if v > 0]
        if len(cands) < k:
            raise ValueError("infeasible size profile")
        rng.shuffle(cands)
        cands.sort(key=lambda r: -remaining[r])
        chosen = cands[:k]
        for r in chosen:
            remaining[r] -= 1
        events.append(chosen)
    if any(v > 0 for v in remaining.values()):
        raise ValueError("leftover tokens")
    return events


def _size_profile(strategy, c, orig_sizes, target_size):
    T = sum(c.values()); D = len(c); maxmult = max(c.values())
    if strategy == "size_preserving":
        return list(orig_sizes)
    if strategy == "merge":
        E = max(maxmult, math.ceil(T / D)); base, rem = divmod(T, E)
        return [base + 1] * rem + [base] * (E - rem)
    if strategy == "split":
        ts = max(1, int(target_size)); E = max(maxmult, math.ceil(T / ts)); base, rem = divmod(T, E)
        return [base + 1] * rem + [base] * (E - rem)
    raise ValueError(strategy)


def same_projection_regroup(df_events, strategy="size_preserving", target_size=2, seed=0):
    """Hypergraph with the SAME directed projection but a regrouped hyperedge
    structure. strategy in {size_preserving, merge, split}."""
    import pandas as pd
    rng = random.Random(seed)
    df = df_events.copy().reset_index(drop=True)
    out_rows = []; eid = 0
    for sender, block in df.groupby("sender", sort=False):
        sender = int(sender)
        evs = [list(map(int, R)) for R in block["recipients"]]
        c = defaultdict(int)
        for R in evs:
            for r in R:
                c[r] += 1
        c = dict(c)
        if not c:
            for R in evs:
                out_rows.append({"event_id": eid, "time": 0, "sender": sender,
                                 "recipients": [], "size": 1, "nodes": [sender]}); eid += 1
            continue
        sizes = _size_profile(strategy, c, [len(R) for R in evs], target_size)
        try:
            events = _fill_events(c, sizes, rng)
        except ValueError:
            events = evs
        for R in events:
            R = sorted(R)
            out_rows.append({"event_id": eid, "time": 0, "sender": sender,
                             "recipients": R, "size": 1 + len(R), "nodes": [sender] + R}); eid += 1
    return pd.DataFrame(out_rows).sort_values(["sender", "event_id"]).reset_index(drop=True)


def projection_signature(df_events):
    w = defaultdict(int)
    for s, R in _events_from_df(df_events):
        for r in R:
            w[(s, r)] += 1
    return tuple(sorted(w.items()))


def dedup_events(df_events):
    """Collapse each sender's repeated identical recipient-sets to ONE event (the
    deduped directed hypergraph). Use this BEFORE shuffling when computing dedup=True
    measures: the 2-switch null preserves recipient MULTIPLICITIES, and duplicate
    events can't be swapped with each other, so they freeze exact teams in place and
    inflate the null's exact-team recurrence (it can even flip the sign of the r1/r2
    gap -- see congress). Deduping first lets the null scramble the distinct teams
    properly. Graph reciprocity is unaffected (it is set-based, ignores multiplicity).
    e.g.  evaluate_s_vs_null(dedup_events(df), shuffle_..., mode='to_team')."""
    import pandas as pd
    seen = set(); rows = []
    for r in df_events.itertuples(index=False):
        R = sorted(int(x) for x in r.recipients)
        key = (int(r.sender), frozenset(R))
        if key in seen:
            continue
        seen.add(key)
        s = int(r.sender)
        rows.append({"event_id": len(rows), "time": 0, "sender": s,
                     "recipients": R, "size": 1 + len(R), "nodes": [s] + R})
    return pd.DataFrame(rows)


# ------------------------------------------------ s-vs-null evaluation
def all_reciprocity_measures(df, dedup=True, min_recipients=1, mode="to_sender", maxk_graph=10**9):
    ev = _events_from_df(df)
    return {
        **compute_advisor_reciprocity_measures(df),
        **compute_group_reciprocity_measures(df, dedup=dedup, min_recipients=min_recipients, mode=mode),
        "graph_recip_full": graph_reciprocity(ev, maxk=maxk_graph),     # full projection (no maxk cap)
    }


def evaluate_s_vs_null(df_events, shuffle_fn, n_shuffles=30, seed=0,
                       measure_kwargs=None, **shuffle_kwargs):
    """measure_kwargs go to the MEASURES (e.g. {"mode":"to_team", "dedup":True,
    "min_recipients":1}); **shuffle_kwargs go to the shuffle_fn."""
    import pandas as pd
    mk = measure_kwargs or {}
    true_df = pd.DataFrame([all_reciprocity_measures(df_events, **mk)])
    rows = []
    for b in range(n_shuffles):
        df_shuf = shuffle_fn(df_events, seed=seed + b, **shuffle_kwargs)
        rows.append({**all_reciprocity_measures(df_shuf, **mk), "shuffle_id": b})
    return true_df, pd.DataFrame(rows)


S_COLS = ["s1_any", "s2_all", "s3_frac",
          "r1_participation_strict", "grant_r2", "r2_participation_partial",
          "r3_leadership_strict", "r4_leadership_partial", "pairwise_group", "graph_recip_full"]


def summarize_s_vs_null(true_df, null_df, cols=S_COLS):
    import pandas as pd
    import numpy as np
    rows = []
    for c in cols:
        tv = float(true_df.iloc[0][c]); nm = float(null_df[c].mean()); sd = float(null_df[c].std(ddof=1))
        rows.append({"measure": c, "true_value": tv, "null_mean": nm, "null_std": sd,
                     "rel_dev": (tv - nm) / nm if nm else np.nan,
                     "z_score": (tv - nm) / sd if sd > 0 else np.nan,
                     "empirical_p_upper": float((null_df[c] >= tv).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------ size-resolved (per-bin) analysis
DEFAULT_SIZE_BINS = [(2, 2), (3, 3), (4, 5), (6, 10), (11, 25), (26, 10**9)]


def _bin_label(lo, hi):
    if lo == hi:
        return str(lo)
    return f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}"


def _filter_by_size(df, lo, hi):
    """Rows whose team size k = 1 + |recipients| is in [lo, hi]."""
    k = df["recipients"].map(len) + 1
    return df[(k >= lo) & (k <= hi)]


def size_binned_measures(df_events, bins=DEFAULT_SIZE_BINS, dedup=True, mode="to_team",
                         min_recipients=1, maxk_graph=10**9):
    """All measures computed WITHIN each team-size bin. The k=2 bin isolates the
    dyads (= the graph-reciprocity control); group measures live in k>=3 bins. Each
    row reports n_events and n_group_units so sparse large-size bins can be judged.
    NOTE: under the strict same-group rule a reply reconvenes the exact team, so the
    reciprocating event has the SAME size as the focal one -- binning by size keeps
    both inside the same bin (this would NOT hold for mode='to_sender')."""
    import pandas as pd
    rows = []
    for lo, hi in bins:
        sub = _filter_by_size(df_events, lo, hi)
        row = {"size_bin": _bin_label(lo, hi), "n_events": int(len(sub))}
        if len(sub):
            row.update(all_reciprocity_measures(sub, dedup=dedup, min_recipients=min_recipients,
                                                mode=mode, maxk_graph=maxk_graph))
        rows.append(row)
    return pd.DataFrame(rows)


def size_binned_vs_null(df_events, shuffle_fn, bins=DEFAULT_SIZE_BINS, n_shuffles=30, seed=0,
                        dedup=True, mode="to_team", min_recipients=1, maxk_graph=10**9,
                        cols=None, **shuffle_kwargs):
    """Real vs null, per size bin. Each null draw is taken on the FULL hypergraph (so
    the directed projection is preserved globally), then re-binned. Returns a tidy
    (size_bin x measure) table with true / null_mean / null_std / rel_dev / z, plus
    n_units for each bin. **shuffle_kwargs go to shuffle_fn."""
    import pandas as pd
    import numpy as np
    if cols is None:
        cols = S_COLS
    true_tab = size_binned_measures(df_events, bins, dedup, mode, min_recipients, maxk_graph)
    null_tabs = [size_binned_measures(shuffle_fn(df_events, seed=seed + b, **shuffle_kwargs),
                                      bins, dedup, mode, min_recipients, maxk_graph)
                 for b in range(n_shuffles)]
    rows = []
    for i, (lo, hi) in enumerate(bins):
        tr = true_tab.iloc[i]
        for c in cols:
            tv = tr.get(c, np.nan)
            vals = np.array([t.iloc[i].get(c, np.nan) for t in null_tabs], dtype=float)
            nfin = int(np.isfinite(vals).sum())
            nm = np.nanmean(vals) if nfin else np.nan
            sd = np.nanstd(vals, ddof=1) if nfin > 1 else np.nan
            rows.append({"size_bin": _bin_label(lo, hi), "measure": c,
                         "n_units": tr.get("n_group_units", np.nan),
                         "true": tv, "null_mean": nm, "null_std": sd,
                         "rel_dev": (tv - nm) / nm if nm else np.nan,
                         "z": (tv - nm) / sd if (sd and sd > 0) else np.nan})
    return pd.DataFrame(rows)


# =====================================================================
# Null-model shuffle functions (operate on df_events)
# =====================================================================

# ---------------------------
# Null model / shuffling
# ---------------------------

def shuffle_group_emails_preserve_sender_size(
    df_events: pd.DataFrame,
    seed: int = 0,
    mode: str = "recipient_freq",
) -> pd.DataFrame:
    """
    Shuffle hyperedges while preserving:
      - sender per event
      - event size
      - number of events

    mode:
      "uniform"       -> recipients sampled uniformly from all nodes except sender
      "recipient_freq"-> recipients sampled without replacement with probability
                         proportional to empirical recipient frequency
    """
    rng = np.random.default_rng(seed)
    df = df_events.copy()

    all_nodes = sorted(set(df["sender"]).union(*df["recipients"].map(set)))
    node_arr = np.array(all_nodes, dtype=np.int64)

    # empirical recipient popularity
    recip_counts = defaultdict(int)
    for recips in df["recipients"]:
        for v in recips:
            recip_counts[int(v)] += 1

    if mode == "recipient_freq":
        weights = np.array([recip_counts[n] for n in node_arr], dtype=float)
        if weights.sum() == 0:
            weights = np.ones_like(weights, dtype=float)
        weights = weights / weights.sum()
    elif mode == "uniform":
        weights = None
    else:
        raise ValueError("mode must be 'uniform' or 'recipient_freq'")

    shuffled_rows = []
    for row in df.itertuples(index=False):
        s = int(row.sender)
        k = len(row.recipients)

        candidates = node_arr[node_arr != s]
        if len(candidates) < k:
            raise ValueError(f"Not enough candidate recipients for sender={s}, k={k}")

        if mode == "uniform":
            recips = rng.choice(candidates, size=k, replace=False)
        else:
            cand_weights = np.array([recip_counts[int(x)] for x in candidates], dtype=float)
            if cand_weights.sum() == 0:
                cand_weights = np.ones_like(cand_weights, dtype=float)
            cand_weights = cand_weights / cand_weights.sum()
            recips = rng.choice(candidates, size=k, replace=False, p=cand_weights)

        recips = sorted(map(int, recips))
        shuffled_rows.append(
            {
                "event_id": int(row.event_id),
                "time": int(row.time),
                "sender": s,
                "recipients": recips,
                "size": 1 + len(recips),
                "nodes": [s] + recips,
            }
        )

    out = pd.DataFrame(shuffled_rows)
    return out.sort_values(["time", "sender", "event_id"]).reset_index(drop=True)


def shuffle_pairwise_preserving_by_sender_swaps(
    df_events: pd.DataFrame,
    seed: int = 0,
    nswap_multiplier: int = 20,
    max_attempt_factor: int = 50,
) -> pd.DataFrame:
    """
    Pairwise-preserving hypergraph null.

    For each sender separately:
      - preserve sender identity
      - preserve number of events
      - preserve each event's size
      - preserve exact dyadic projection counts sender->recipient
      - randomize which recipients co-occur in the same event

    Mechanism:
      Random 2-switches across events of the SAME sender:
        event i contains a not b
        event j contains b not a
      swap them:
        i: a -> b
        j: b -> a

    This preserves row sums (event sizes) and column sums
    (how many times the sender contacted each recipient).
    """
    rng = random.Random(seed)              # python RNG: fast scalar ops in the hot loop
    df = df_events.copy().reset_index(drop=True)
    out_rows = []

    for sender, block in df.groupby("sender", sort=False):
        sender = int(sender)
        lists = [list(map(int, R)) for R in block["recipients"]]   # random index access
        sets = [set(L) for L in lists]                              # O(1) membership
        meta = [(int(r.event_id), int(r.time)) for r in block.itertuples(index=False)]
        m = len(lists)
        total_ones = sum(len(L) for L in lists)

        if m > 1 and total_ones > 0:
            target = nswap_multiplier * total_ones
            max_attempts = max_attempt_factor * target
            ri = rng.randrange
            done = att = 0
            while done < target and att < max_attempts:
                att += 1
                i = ri(m); j = ri(m)
                if i == j:
                    continue
                Li, Lj = lists[i], lists[j]
                if not Li or not Lj:
                    continue
                ai = ri(len(Li)); bj = ri(len(Lj))
                a = Li[ai]; b = Lj[bj]
                Si, Sj = sets[i], sets[j]
                if a == b or a in Sj or b in Si:    # swap would create a duplicate
                    continue
                Li[ai] = b; Si.discard(a); Si.add(b)
                Lj[bj] = a; Sj.discard(b); Sj.add(a)
                done += 1

        for (eid, t), L in zip(meta, lists):
            Ls = sorted(L)
            out_rows.append({"event_id": eid, "time": t, "sender": sender,
                             "recipients": Ls, "size": 1 + len(Ls),
                             "nodes": [sender] + Ls})

    out = pd.DataFrame(out_rows)
    return out.sort_values(["time", "sender", "event_id"], kind="mergesort").reset_index(drop=True)


def shuffle_sender_local_hypergraph(
    df_events: pd.DataFrame,
    seed: int = 0,
    mode: str = "local_freq",
) -> pd.DataFrame:
    """
    Sender-local hypergraph null.

    For each sender separately:
      - preserve sender identity
      - preserve number of events
      - preserve each event size
      - shuffle recipients only within that sender's historical recipient pool

    mode:
      - "local_uniform": recipients drawn uniformly from sender's recipient pool
      - "local_freq": recipients drawn with probabilities proportional to
                      sender-specific recipient frequencies
    """
    rng = np.random.default_rng(seed)

    df = df_events.copy().reset_index(drop=True)
    out_rows = []

    for sender, block in df.groupby("sender", sort=False):
        block = block.copy().reset_index(drop=True)

        # sender's historical recipient pool and frequencies
        local_counts = defaultdict(int)
        for recips in block["recipients"]:
            for v in recips:
                local_counts[int(v)] += 1

        pool = np.array(sorted(local_counts.keys()), dtype=np.int64)

        if len(pool) == 0:
            for row in block.itertuples(index=False):
                out_rows.append({
                    "event_id": int(row.event_id),
                    "time": int(row.time),
                    "sender": int(row.sender),
                    "recipients": [],
                    "size": 1,
                    "nodes": [int(row.sender)],
                })
            continue

        if mode == "local_uniform":
            base_probs = np.ones(len(pool), dtype=float) / len(pool)
        elif mode == "local_freq":
            base_probs = np.array([local_counts[int(v)] for v in pool], dtype=float)
            base_probs = base_probs / base_probs.sum()
        else:
            raise ValueError("mode must be 'local_uniform' or 'local_freq'")

        for row in block.itertuples(index=False):
            k = len(row.recipients)

            if k > len(pool):
                raise ValueError(
                    f"Sender {sender} has event size {k}, but only {len(pool)} unique local recipients."
                )

            recips = rng.choice(pool, size=k, replace=False, p=base_probs)
            recips = sorted(map(int, recips))

            out_rows.append({
                "event_id": int(row.event_id),
                "time": int(row.time),
                "sender": int(row.sender),
                "recipients": recips,
                "size": 1 + len(recips),
                "nodes": [int(row.sender)] + recips,
            })

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["time", "sender", "event_id"], kind="mergesort").reset_index(drop=True)
    return out


def shuffle_global_random_hypergraph(
    df_events: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Global random directed-hypergraph null.

    Preserves:
      - number of events
      - event times
      - event_id
      - hyperedge sizes (equivalently, number of recipients per event)

    Randomizes:
      - sender
      - recipient set

    For each event:
      - choose 1 sender uniformly from all nodes
      - choose r distinct recipients uniformly from the remaining nodes,
        where r = original number of recipients for that event

    Assumes df_events has columns:
      event_id, time, sender, recipients
    and optionally size, nodes.
    """
    rng = np.random.default_rng(seed)

    df = df_events.copy().reset_index(drop=True)

    # collect all nodes appearing anywhere in the data
    all_nodes = sorted(
        set(df["sender"].astype(int)).union(*df["recipients"].map(lambda x: set(map(int, x))))
    )
    all_nodes = np.array(all_nodes, dtype=np.int64)

    out_rows = []

    for row in df.itertuples(index=False):
        r = len(row.recipients)  # number of recipients to preserve

        if len(all_nodes) < r + 1:
            raise ValueError(
                f"Cannot sample sender + {r} distinct recipients from only {len(all_nodes)} nodes."
            )

        # choose sender
        sender = int(rng.choice(all_nodes))

        # choose recipients from all other nodes
        recipient_pool = all_nodes[all_nodes != sender]
        recipients = rng.choice(recipient_pool, size=r, replace=False)
        recipients = sorted(map(int, recipients))

        out_rows.append({
            "event_id": int(row.event_id),
            "time": int(row.time),
            "sender": sender,
            "recipients": recipients,
            "size": 1 + len(recipients),
            "nodes": [sender] + recipients,
        })

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["time", "sender", "event_id"], kind="mergesort").reset_index(drop=True)
    return out

def shuffle_global_recipients_only_hypergraph(
    df_events: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Global recipient-only random directed-hypergraph null.

    Preserves:
      - number of events
      - event times
      - event_id
      - sender of each event
      - event sizes

    Randomizes:
      - recipient set, drawn uniformly from all nodes except sender
    """
    rng = np.random.default_rng(seed)

    df = df_events.copy().reset_index(drop=True)

    all_nodes = sorted(
        set(df["sender"].astype(int)).union(*df["recipients"].map(lambda x: set(map(int, x))))
    )
    all_nodes = np.array(all_nodes, dtype=np.int64)

    out_rows = []

    for row in df.itertuples(index=False):
        sender = int(row.sender)
        r = len(row.recipients)

        recipient_pool = all_nodes[all_nodes != sender]
        if len(recipient_pool) < r:
            raise ValueError(
                f"Sender {sender} cannot sample {r} distinct recipients from only {len(recipient_pool)} candidates."
            )

        recipients = rng.choice(recipient_pool, size=r, replace=False)
        recipients = sorted(map(int, recipients))

        out_rows.append({
            "event_id": int(row.event_id),
            "time": int(row.time),
            "sender": sender,
            "recipients": recipients,
            "size": 1 + len(recipients),
            "nodes": [sender] + recipients,
        })

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["time", "sender", "event_id"], kind="mergesort").reset_index(drop=True)
    return out


# ---------------------------
# DiNgHy / NuDHy degree-preserving directed-hypergraph nulls
# ---------------------------
#
# Port of the DiNgHy / NuDHy MCMC samplers (ECML-PKDD '25 / Phys. Rev. X 2024),
# operating directly on df_events. Each event is a directed hyperedge with
#   head = {sender}   tail = recipients
#
# Unlike the sender-preserving nulls above, these preserve BOTH:
#   - each node's sender-degree   (# events in which it is the sender)
#   - each node's recipient-degree (# events in which it is a recipient)
#   - every event's size (# recipients)
# while randomizing who co-occurs / who sends. One MCMC step is a degree-
# preserving "swap" of two memberships between two events.
#
# Variants (`sampler=`):
#   "dinghy"     - swaps hyperedge head/tail sets; forbids degeneracy
#                  (a sender never becomes one of its own recipients). [recommended]
#   "dinghy_b"   - same non-degeneracy guarantee, swaps over nodes instead.
#   "nudhy_degs" - degree-preserving only; MAY make a sender appear among its
#                  own recipients (degenerate hyperedge -> a self-loop in the
#                  dyadic projection).
#
# Self-contained (no external package needed) and statistically equivalent to
# the Java reference (uses Python's RNG, so it is not byte-identical).



def _dh_symmetric_difference(a, b):
    """(elements only in a, elements only in b) for two sorted lists."""
    only_a, only_b = [], []
    i = j = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
        elif a[i] < b[j]:
            only_a.append(a[i]); i += 1
        else:
            only_b.append(b[j]); j += 1
    if i < la:
        only_a.extend(a[i:])
    if j < lb:
        only_b.extend(b[j:])
    return only_a, only_b


def _dh_symmetric_difference_nd(a, b, proh_a, proh_b):
    """Symmetric difference excluding values present in the prohibited sets
    (the opposite side of each hyperedge -> prevents degeneracy)."""
    prohibited = set(proh_a)
    prohibited.update(proh_b)
    oa, ob = _dh_symmetric_difference(a, b)
    return ([x for x in oa if x not in prohibited],
            [x for x in ob if x not in prohibited])


def _dh_replace_in_sorted(values, old, new):
    """Replace `old` (present) with `new` (absent), keeping `values` sorted."""
    idx = bisect.bisect_left(values, old)
    del values[idx]
    bisect.insort(values, new)


def shuffle_dinghy_hypergraph(
    df_events: pd.DataFrame,
    seed: int = 0,
    sampler: str = "dinghy",
    swap_factor: int = 20,
) -> pd.DataFrame:
    """
    DiNgHy / NuDHy degree-preserving directed-hypergraph null.

    Preserves number of events, event_id/time, event sizes, and the full
    sender- and recipient-degree sequences; randomizes head/tail membership.

    sampler: "dinghy" (default), "dinghy_b", or "nudhy_degs".
    swap_factor: number of MCMC steps = swap_factor * (#head + #tail memberships).
    """
    if sampler not in ("dinghy", "dinghy_b", "nudhy_degs"):
        raise ValueError("sampler must be 'dinghy', 'dinghy_b', or 'nudhy_degs'")

    # --- build the directed bipartite representation (nodes <-> events) ---
    nodes = set()
    for row in df_events.itertuples(index=False):
        nodes.add(int(row.sender))
        nodes.update(int(x) for x in row.recipients)
    node_of_id = sorted(nodes)
    id_of_node = {lab: i for i, lab in enumerate(node_of_id)}
    n = len(node_of_id)
    m = len(df_events)

    left_P = [[] for _ in range(n)]   # node -> events it is SENDER of   (head membership)
    left_M = [[] for _ in range(n)]   # node -> events it is RECIPIENT of (tail membership)
    right_M = [[] for _ in range(m)]  # event -> head vertices (the sender)
    right_P = [[] for _ in range(m)]  # event -> tail vertices (the recipients)
    meta = []                         # (event_id, time) per event, kept aligned by index

    for h, row in enumerate(df_events.itertuples(index=False)):
        s = id_of_node[int(row.sender)]
        right_M[h].append(s)
        left_P[s].append(h)
        for x in row.recipients:
            v = id_of_node[int(x)]
            right_P[h].append(v)
            left_M[v].append(h)
        meta.append((int(row.event_id), int(row.time)))

    for arr in (left_P, left_M, right_M, right_P):
        for lst in arr:
            lst.sort()

    size_D0 = sum(len(x) for x in right_M)  # total head memberships
    size_D1 = sum(len(x) for x in right_P)  # total tail memberships

    # coin bias: if one direction is empty, only swap on the other side
    bias = 0.5
    if size_D1 == 0:
        bias = 1.0
    elif size_D0 == 0:
        bias = 0.0

    nondegenerate = sampler in ("dinghy", "dinghy_b")

    def branches(head_side):
        # returns (P, opposite, M): P = sets being swapped (indexed by picked
        # entity), M = mirror adjacency (indexed by swapped element),
        # opposite = prohibited values for non-degenerate swaps.
        if sampler == "nudhy_degs":
            return (left_P, None, right_M) if head_side else (right_P, None, left_M)
        if sampler == "dinghy":
            return (right_M, right_P, left_P) if head_side else (right_P, right_M, left_M)
        # dinghy_b
        return (left_P, left_M, right_M) if head_side else (right_P, right_M, left_M)

    rnd = random.Random(seed)
    num_swaps = swap_factor * (size_D0 + size_D1)

    for _ in range(num_swaps):
        head_side = rnd.random() <= bias
        P, opposite, M = branches(head_side)
        L = len(P)
        if L < 2:
            continue
        u = rnd.randrange(L)
        z = rnd.randrange(L)
        while u == z:
            z = rnd.randrange(L)

        if nondegenerate:
            only_u, only_z = _dh_symmetric_difference_nd(P[u], P[z], opposite[u], opposite[z])
        else:
            only_u, only_z = _dh_symmetric_difference(P[u], P[z])

        if not only_u or not only_z:
            continue  # self-loop (rejection)

        v = only_u[rnd.randrange(len(only_u))]
        w = only_z[rnd.randrange(len(only_z))]
        _dh_replace_in_sorted(P[u], v, w)
        _dh_replace_in_sorted(M[v], u, z)
        _dh_replace_in_sorted(P[z], w, v)
        _dh_replace_in_sorted(M[w], z, u)

    # --- reconstruct df_events (head size is 1, so each event keeps one sender) ---
    out_rows = []
    for h in range(m):
        sender = node_of_id[right_M[h][0]]
        recips = sorted(node_of_id[v] for v in right_P[h])
        eid, t = meta[h]
        out_rows.append({
            "event_id": eid,
            "time": t,
            "sender": sender,
            "recipients": recips,
            "size": 1 + len(recips),
            "nodes": [sender] + recips,
        })

    out = pd.DataFrame(out_rows)
    return out.sort_values(["time", "sender", "event_id"], kind="mergesort").reset_index(drop=True)


# =====================================================================
# Data loaders -> df_events (sender, recipients)
# =====================================================================
# Nate's .mat files store H (edge x node incidence) + Senders/Sponsors (one
# sender per hyperedge).  df_events_from_mat handles both classic and v7.3/HDF5.
# df_events_from_tsv reads the dinghy "head<TAB>tail" format (SINGLE-head only:
# congress/enron/email-Eu; NOT multi-head dblp_v9.tsv -> use the .mat for dblp).

def events_from_mat(path):
    """Return list of (sender, recipients) from a Nate .mat (Senders/Sponsors)."""
    try:
        from scipy.io import loadmat
        M = loadmat(path)
        H = M["H"]
        skey = "Sponsors" if "Sponsors" in M else "Senders"
        senders = np.asarray(M[skey]).ravel().astype(int)
        H = H.tocsr()
        ev = []
        for j in range(H.shape[0]):
            s = int(senders[j])
            ev.append((s, [int(x) + 1 for x in H[j].indices if int(x) + 1 != s]))
        return ev
    except NotImplementedError:           # MATLAB v7.3 (HDF5)
        import h5py
        f = h5py.File(path, "r")
        Hg = f["H"]
        ir = np.asarray(Hg["ir"]).astype(int)
        jc = np.asarray(Hg["jc"]).astype(int)
        n_edges = int(Hg.attrs["MATLAB_sparse"])
        skey = "Sponsors" if "Sponsors" in f else "Senders"
        senders = np.asarray(f[skey]).ravel().astype(int)
        edge_nodes = defaultdict(list)    # CSC by column(node) -> edges
        for v in range(len(jc) - 1):
            for idx in range(jc[v], jc[v + 1]):
                edge_nodes[int(ir[idx])].append(v + 1)   # 1-based node id
        return [(int(senders[j]),
                 [x for x in edge_nodes.get(j, []) if x != int(senders[j])])
                for j in range(n_edges)]


def _events_to_df(ev):
    return pd.DataFrame([{"event_id": i, "time": 0, "sender": s, "recipients": R,
                          "size": 1 + len(R), "nodes": [s] + R}
                         for i, (s, R) in enumerate(ev)])


def df_events_from_mat(path):
    """Load a Nate .mat hypergraph into a df_events frame (static; time=0)."""
    return _events_to_df(events_from_mat(path))


def df_events_from_tsv(path):
    """Load a dinghy 'head<TAB>tail' tsv (SINGLE head node = sender)."""
    ev = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            head = [int(x) for x in parts[0].split(",") if x != ""]
            tail = ([int(x) for x in parts[1].split(",") if x != ""]
                    if len(parts) >= 2 else [])
            s = head[0]                    # single-sender; extra head nodes ignored
            ev.append((s, [x for x in tail if x != s]))
    return _events_to_df(ev)


def party_labels_from_mat(path):
    """Load node->party dict from congress-bills-sponsors-hypergraph-parties.mat."""
    import h5py
    f = h5py.File(path, "r")
    parties = np.asarray(f["Parties"]).ravel().astype(int)
    return {i + 1: int(p) for i, p in enumerate(parties)}   # node id (1-based) -> party