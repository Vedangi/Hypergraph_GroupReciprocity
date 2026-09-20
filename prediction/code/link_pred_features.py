"""Causal features and future-response delays for reciprocity prediction.

This module is the single source of truth for:

* dataset loading;
* the 15 graph and 5 hypergraph features;
* future-event indices;
* horizon-independent causal feature rows.

One sample is a focal event-recipient pair ``(e=(s,R,t), v in R)``.  Features
use events at times strictly smaller than ``t``.  The output stores delays,
not fixed-horizon labels, so one feature frame can be reused for every
prediction horizon.

The experiment module derives three labels:

``y_dyad``
    Whether ``v -> s`` occurs in any form.
``y_mode``
    Among rows with a dyadic response, whether ``v`` gives a strict
    overlap-based group response.
``y_group_exact``
    Whether ``v`` leads a response with exactly the original node set.
"""

from __future__ import annotations

import ast
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DAY = 86400.0
ROOT = Path(__file__).resolve().parents[2]  # prediction/code -> repo root

DATASETS = {
    "emaileu": {
        "path": ROOT / "data/emailEu/email-Eu-core-temporal.txt",
        "kind": "triples", "step": DAY, "unit": "day",
        "default_horizons": (1, 7, 30), "default_theta": 0.5,
    },
    "enron": {
        "path": ROOT / "data/enron/out.enron",
        "kind": "triples", "step": DAY, "unit": "day",
        "default_horizons": (1, 7, 30), "default_theta": 0.5,
    },
    "dnc": {
        "path": ROOT / "data/dnc/email-dnc.edges",
        "kind": "triples", "step": DAY, "unit": "day",
        "default_horizons": (1, 3, 7), "default_theta": 0.5,
    },
    "twitter": {
        "path": ROOT / "data/twitter/twitter_observed_min1_events.parquet",
        "kind": "twitter", "step": DAY, "unit": "day",
        "default_horizons": (7, 30, 60), "default_theta": 0.5,
    },
    "twitter_noreply": {
        "path": ROOT / "data/twitter/twitter_noreply_events.parquet",
        "kind": "twitter", "step": DAY, "unit": "day",
        "default_horizons": (7, 30, 60), "default_theta": 0.5,
    },
}


# ---------------------------------------------------------------------------
# Data loading

def load_events(path: Path, limit: int | None = None):
    """Load temporal triples and merge rows with the same sender and time."""
    by_event = defaultdict(set)
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                sender = int(parts[0])
                recipient = int(parts[1])
                timestamp = int(parts[-1])
            except ValueError:
                continue
            if sender != recipient and timestamp >= 0:
                by_event[(sender, timestamp)].add(recipient)

    events = [
        (sender, sorted(recipients), timestamp)
        for (sender, timestamp), recipients in by_event.items()
        if recipients
    ]
    events.sort(key=lambda event: (event[2], event[0]))
    return events[:limit] if limit else events


def load_events_simplices(
    folder: Path,
    sender: str = "first",
    max_size: int | None = 25,
    limit: int | None = None,
):
    """Load Cornell temporal simplices and designate one node as sender."""
    import glob

    def find_file(suffix):
        hits = glob.glob(str(folder / f"*-{suffix}.txt"))
        if not hits:
            raise FileNotFoundError(f"No *-{suffix}.txt in {folder}")
        return hits[0]

    nverts = np.loadtxt(find_file("nverts"), dtype=np.int64)
    times = np.loadtxt(find_file("times"), dtype=np.int64)
    events = []
    with open(find_file("simplices")) as fh:
        for size, timestamp in zip(nverts, times):
            nodes = [int(next(fh)) for _ in range(int(size))]
            if size < 2 or (max_size and size > max_size):
                continue
            leader = nodes[0] if sender == "first" else min(nodes)
            recipients = sorted(set(nodes) - {leader})
            if recipients:
                events.append((leader, recipients, int(timestamp)))
    events.sort(key=lambda event: (event[2], event[0]))
    return events[:limit] if limit else events


def load_events_congress(folder: Path, limit: int | None = None):
    """Load sponsor/cosponsor bill records as directed temporal hyperedges."""

    def read_lines(name):
        with open(folder / name) as fh:
            return [line.strip() for line in fh]

    sponsors = read_lines("sponsors.txt")
    cosponsors = read_lines("Cosponsors.txt")
    dates = read_lines("Dates.txt")
    if not (len(sponsors) == len(cosponsors) == len(dates)):
        raise ValueError("sponsors/Cosponsors/Dates line counts differ")

    parsed_dates = pd.to_datetime(pd.Series(dates), errors="coerce")
    epoch_days = parsed_dates.astype("int64") // 10**9 // 86400
    events = []
    for i, sponsor in enumerate(sponsors):
        if sponsor in ("", "NA") or cosponsors[i] in ("", "NA"):
            continue
        if pd.isna(parsed_dates.iloc[i]):
            continue
        try:
            sender = int(sponsor)
            recipients = sorted({
                int(x) for x in cosponsors[i].split()
                if x not in ("", "NA") and int(x) != sender
            })
        except ValueError:
            continue
        if recipients:
            events.append((sender, recipients, int(epoch_days.iloc[i])))
    events.sort(key=lambda event: (event[2], event[0]))
    return events[:limit] if limit else events


def load_events_twitter(path: Path, limit: int | None = None):
    """Load Twitter mention events from a Parquet event table."""

    def as_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return ast.literal_eval(value)
        return list(value)

    frame = pd.read_parquet(path)
    events = []
    seen = set()
    for row in frame.itertuples(index=False):
        sender = int(row.sender)
        recipients = sorted({
            int(x) for x in as_list(row.recipients) if int(x) != sender
        })
        timestamp = int(row.timestamp)
        key = (sender, tuple(recipients), timestamp)
        if recipients and key not in seen:
            seen.add(key)
            events.append((sender, recipients, timestamp))
    events.sort(key=lambda event: (event[2], event[0]))
    return events[:limit] if limit else events


def load_events_frame(frame):
    """Convert a df_events frame (event_id/time/sender/recipients columns)
    into sorted (sender, recipients, time) triples."""
    events = []
    for row in frame.itertuples(index=False):
        sender = int(row.sender)
        recipients = sorted({
            int(x) for x in row.recipients if int(x) != sender
        })
        if recipients:
            events.append((sender, recipients, int(row.time)))
    events.sort(key=lambda event: (event[2], event[0]))
    return events


def load_dataset(
    name: str,
    limit: int | None = None,
    max_size: int | None = None,
):
    """Load one registered dataset and apply a common maximum team size."""
    cfg = DATASETS[name]
    path = cfg["path"]
    kind = cfg["kind"]
    if kind == "triples":
        events = load_events(path, limit=None)
    elif kind == "simplices":
        events = load_events_simplices(
            path, max_size=max_size, limit=None)
    elif kind == "congress":
        events = load_events_congress(path, limit=None)
    elif kind == "music":
        from music_feature_to_df import load_events_music
        events = load_events_music(
            str(path), limit=None, max_size=max_size)
    elif kind == "twitter":
        events = load_events_twitter(path, limit=None)
    elif kind == "cordis":
        from analyze_cordis import load_cordis_events
        events = load_events_frame(load_cordis_events())
    elif kind == "aact":
        from analyze_aact import load_aact_events
        frame = load_aact_events(min_recipients=1)
        # Registry start dates contain typos (1916) and far-future planned
        # trials (2050); clamp to 1990-01-01 .. 2026-12-31 (epoch days).
        frame = frame[(frame["time"] >= 7305) & (frame["time"] <= 20819)]
        events = load_events_frame(frame)
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

    if max_size:
        events = [
            event for event in events
            if 1 + len(event[1]) <= max_size
        ]
    return events[:limit] if limit else events


# ---------------------------------------------------------------------------
# Active feature definitions

GRAPH_FEATURES = [
    "f_fwd",
    "f_rev",
    "f_recip",
    "f_balance",
    "f_pair_pErec",
    "f_pair_latency",
    "f_outdeg_s",
    "f_indeg_v",
    "f_cn",
    "f_status_s",
    "f_status_v",
    "f_status_gap",
    "f_2step_sv",
    "f_2step_vs",
    "f_rc",
    # Kleinberg link-prediction features (directed neighbourhood overlap).
    # Their own tables rank these WEAKEST for reciprocity; included so the
    # graph baseline cannot be accused of omitting standard LP features.
    "f_mn_in",         # |Gamma^-(s) & Gamma^-(v)|  directed mutual in-neighbours
    "f_mn_out",        # |Gamma^+(s) & Gamma^+(v)|  directed mutual out-neighbours
    "f_jaccard_in",    # in-neighbour Jaccard coefficient
    "f_jaccard_out",   # out-neighbour Jaccard coefficient
    "f_adamic_adar",   # Adamic/Adar over common in-neighbours, 1/log(deg^-(x))
]

HYPERGRAPH_FEATURES = [
    "h_rsize",
    "h_team_turntaking",
    "h_gir",
    "h_r4_lead",
    "h_team_leaders",
]

# Candidate hypergraph features under test -- NOT part of the frozen tier C.
# Offered to the validation-only selection via --extra-candidates so the
# validation block decides whether each earns a place.
CANDIDATE_FEATURES = [
    # peer s1 (ARO effect): fraction of v's CO-RECIPIENTS who have already
    # sent s a group message. Higher-order (group-restricted, not projection)
    # and autocorrelation-free (about u != v, not v's own history).
    "h_corecip_grouprecip",
    # sender's own r4 leadership coverage of the team (the library's original
    # orientation; we otherwise apply r4 only to v).
    "h_team_leadcov_s",
    # ROBUSTNESS bucket (self-history -- expect autocorrelation flags):
    "h_vs_group_count",   # raw group_count(v->s): the h_gir numerator's volume
    "h_s_group_indeg",    # # distinct nodes who have group-led s (normalizer)
]

FEATURE_GROUPS = {
    "A": GRAPH_FEATURES,
    "C": HYPERGRAPH_FEATURES,
    "CAND": CANDIDATE_FEATURES,
}

TIERS = {
    "A": GRAPH_FEATURES,
    "C": GRAPH_FEATURES + HYPERGRAPH_FEATURES,
}

ALL_FEATURES = TIERS["C"]
FEATURE_INDEX = {name: i for i, name in enumerate(ALL_FEATURES)}


def tier_of(name: str):
    """Return ``A`` for graph and ``C`` for hypergraph features."""
    for tier, names in FEATURE_GROUPS.items():
        if name in names:
            return tier
    return "?"


def resolve_cols(spec):
    """Expand tier names and/or raw feature names, preserving unique order."""
    columns = []
    for item in spec:
        columns.extend(TIERS[item] if item in TIERS else [item])
    return list(dict.fromkeys(columns))


# ---------------------------------------------------------------------------
# Future response indices and delays

def build_response_index(events):
    """Build future dyadic, group-sender, and exact-team timestamp indices."""
    dyadic_times = defaultdict(list)
    group_times = defaultdict(list)
    group_teams = defaultdict(list)
    exact_times = defaultdict(list)

    for sender, recipients, timestamp in events:
        for recipient in recipients:
            dyadic_times[(sender, recipient)].append(timestamp)
        if len(recipients) >= 2:
            team = frozenset([sender] + recipients)
            group_times[sender].append(timestamp)
            group_teams[sender].append(team)
            exact_times[(sender, team)].append(timestamp)

    dyadic = {
        key: np.asarray(sorted(values), dtype=np.int64)
        for key, values in dyadic_times.items()
    }
    groups = {}
    for sender, values in group_times.items():
        order = np.argsort(
            np.asarray(values, dtype=np.int64), kind="mergesort")
        groups[sender] = (
            np.asarray(values, dtype=np.int64)[order],
            [group_teams[sender][i] for i in order],
        )
    exact = {
        key: np.asarray(sorted(values), dtype=np.int64)
        for key, values in exact_times.items()
    }
    return dyadic, groups, exact


def _next_delay(times, timestamp):
    if times is None:
        return np.inf
    index = int(np.searchsorted(times, timestamp, side="right"))
    return (
        float(times[index] - timestamp)
        if index < len(times) else np.inf
    )


def delay_dyad(dyadic_index, v, s, timestamp):
    """Delay to the first future ``v -> s`` incidence in any message form."""
    return _next_delay(dyadic_index.get((v, s)), timestamp)


def delay_group_exact(exact_index, v, team, timestamp):
    """Delay to the first future event led by ``v`` with exactly ``team``."""
    return _next_delay(exact_index.get((v, team)), timestamp)


def delay_group_overlap(
    group_index,
    v,
    s,
    team,
    timestamp,
    theta,
):
    """Delay to the first strict overlap-based group response led by ``v``."""
    entry = group_index.get(v)
    if entry is None:
        return np.inf
    times, teams = entry
    start = int(np.searchsorted(times, timestamp, side="right"))
    co_recipients = team - {s, v}
    for i in range(start, len(times)):
        response_team = teams[i]
        if s not in response_team:
            continue
        if co_recipients and not (co_recipients & response_team):
            continue
        if len(team & response_team) / len(team) >= theta:
            return float(times[i] - timestamp)
    return np.inf


# ---------------------------------------------------------------------------
# Causal feature stream

def build_causal_rows(events, theta=0.5):
    """Build one horizon-independent causal row per event-recipient pair.

    All rows at timestamp ``t`` are emitted before any event at ``t`` updates
    state.  Pair and exact-team turn-taking statistics are updated from
    timestamp-level sender sets; simultaneous opposite-direction events are
    treated as unordered and never counted as temporal flips.
    """
    if not events:
        raise ValueError("No events")

    dyadic_index, group_index, exact_index = build_response_index(events)
    count = defaultdict(int)
    group_count = defaultdict(int)
    out_strength = defaultdict(int)
    in_strength = defaultdict(int)
    out_neighbors = defaultdict(set)
    in_neighbors = defaultdict(set)
    undirected_neighbors = defaultdict(set)
    reciprocal_neighbors = defaultdict(set)
    group_in_neighbors = defaultdict(set)  # node -> senders who group-led it

    # [last unique sender, transitions, flips, last unique time, latency sum,
    #  latency count]
    pair_state = defaultdict(lambda: [None, 0, 0, None, 0.0, 0])
    # [last unique sender, transitions, flips]
    team_state = defaultdict(lambda: [None, 0, 0])
    team_leaders = defaultdict(set)
    led_to = defaultdict(set)

    rows = []
    i = 0
    while i < len(events):
        timestamp = events[i][2]
        j = i + 1
        while j < len(events) and events[j][2] == timestamp:
            j += 1
        batch = events[i:j]

        # Emit from history with timestamps strictly smaller than timestamp.
        for eid in range(i, j):
            sender, recipients, _ = events[eid]
            group_size = len(recipients)
            team_size = group_size + 1
            team = frozenset([sender] + recipients)
            sender_neighbors = undirected_neighbors[sender]

            exact_state = team_state[team]
            team_turntaking = (
                exact_state[2] / exact_state[1]
                if exact_state[1] else 0.0
            )

            # ---- candidate features (event level; shared by all v) ----
            recipient_set = set(recipients)
            # sender's own r4 leadership coverage of the team (N\{s} = R).
            team_leadcov_s = (
                len(led_to[sender] & recipient_set) / group_size
                if group_size else 0.0
            )
            # peers who have already group-messaged s (co-recipient reciprocity)
            corecip_grp_total = sum(
                1 for u in recipients if group_count[(u, sender)] > 0
            )
            s_group_indeg = len(group_in_neighbors[sender])

            for v in recipients:
                forward = count[(sender, v)]
                reverse = count[(v, sender)]
                total = forward + reverse
                pair = pair_state[frozenset((sender, v))]
                common_set = (
                    sender_neighbors & undirected_neighbors[v]
                    if sender_neighbors and undirected_neighbors[v]
                    else set()
                )

                status_s = math.log(
                    (out_strength[sender] + 1)
                    / (in_strength[sender] + 1)
                )
                status_v = math.log(
                    (out_strength[v] + 1) / (in_strength[v] + 1)
                )
                two_step_sv = len(
                    out_neighbors[sender] & in_neighbors[v]
                )
                two_step_vs = len(
                    out_neighbors[v] & in_neighbors[sender]
                )
                reciprocal_common = (
                    len(common_set & reciprocal_neighbors[sender])
                    + len(common_set & reciprocal_neighbors[v])
                )

                # Kleinberg directed neighbourhood-overlap features.
                in_s, in_v = in_neighbors[sender], in_neighbors[v]
                out_s, out_v = out_neighbors[sender], out_neighbors[v]
                common_in = in_s & in_v if in_s and in_v else set()
                common_out = out_s & out_v if out_s and out_v else set()
                union_in = len(in_s) + len(in_v) - len(common_in)
                union_out = len(out_s) + len(out_v) - len(common_out)
                mutual_in = len(common_in)
                mutual_out = len(common_out)
                jaccard_in = mutual_in / union_in if union_in else 0.0
                jaccard_out = mutual_out / union_out if union_out else 0.0
                adamic_adar = 0.0
                for x in common_in:
                    degree = len(in_neighbors[x])
                    if degree > 1:
                        adamic_adar += 1.0 / math.log(degree)

                # candidate features (per v)
                vs_group_count = group_count[(v, sender)]
                corecip_grouprecip = (
                    (corecip_grp_total - (1 if vs_group_count > 0 else 0))
                    / (group_size - 1)
                    if group_size > 1 else 0.0
                )

                row = {
                    "t": timestamp,
                    "k": team_size,
                    "eid": eid,
                    "s": sender,
                    "v": v,
                    "f_fwd": forward,
                    "f_rev": reverse,
                    "f_recip": 1.0 if reverse else 0.0,
                    "f_balance": (
                        max(forward, reverse) / total if total else 0.5
                    ),
                    "f_pair_pErec": (
                        pair[2] / pair[1] if pair[1] else 0.0
                    ),
                    "f_pair_latency": (
                        pair[4] / pair[5] if pair[5] else 0.0
                    ),
                    # These are true projection degrees (unique neighbors).
                    "f_outdeg_s": len(out_neighbors[sender]),
                    "f_indeg_v": len(in_neighbors[v]),
                    "f_cn": len(common_set),
                    "f_status_s": status_s,
                    "f_status_v": status_v,
                    "f_status_gap": status_s - status_v,
                    "f_2step_sv": two_step_sv,
                    "f_2step_vs": two_step_vs,
                    "f_rc": reciprocal_common,
                    "f_mn_in": mutual_in,
                    "f_mn_out": mutual_out,
                    "f_jaccard_in": jaccard_in,
                    "f_jaccard_out": jaccard_out,
                    "f_adamic_adar": adamic_adar,
                    "h_rsize": float(team_size),
                    "h_team_turntaking": team_turntaking,
                    "h_gir": (
                        group_count[(v, sender)] / reverse
                        if reverse else 0.0
                    ),
                    "h_r4_lead": (
                        len(led_to[v] & (team - {v})) / group_size
                        if group_size else 0.0
                    ),
                    "h_team_leaders": (
                        len(team_leaders[team]) / team_size
                    ),
                    "h_corecip_grouprecip": corecip_grouprecip,
                    "h_team_leadcov_s": team_leadcov_s,
                    "h_vs_group_count": float(vs_group_count),
                    "h_s_group_indeg": float(s_group_indeg),
                    "d_dyad": delay_dyad(
                        dyadic_index, v, sender, timestamp),
                    "d_overlap": delay_group_overlap(
                        group_index, v, sender, team, timestamp, theta),
                    "d_exact": delay_group_exact(
                        exact_index, v, team, timestamp),
                }
                rows.append(row)

        # Collect unordered timestamp-level sender sets before updating.
        pair_senders = defaultdict(set)
        team_senders = defaultdict(set)
        affected_directions = set()
        for sender, recipients, _ in batch:
            team = frozenset([sender] + recipients)
            team_senders[team].add(sender)
            for recipient in recipients:
                pair_senders[frozenset((sender, recipient))].add(sender)
                affected_directions.add((sender, recipient))

        # Update pair turn-taking only when this timestamp has one direction.
        for pair_key, senders in pair_senders.items():
            state = pair_state[pair_key]
            if len(senders) == 1:
                current_sender = next(iter(senders))
                if state[0] is not None:
                    state[1] += 1
                    if state[0] != current_sender:
                        state[2] += 1
                        if state[3] is not None:
                            state[4] += timestamp - state[3]
                            state[5] += 1
                state[0] = current_sender
                state[3] = timestamp
            else:
                # Simultaneous opposite directions have no temporal order.
                state[0] = None
                state[3] = None

        # Update exact-team turn-taking using the same simultaneity rule.
        for team, senders in team_senders.items():
            state = team_state[team]
            if len(senders) == 1:
                current_sender = next(iter(senders))
                if state[0] is not None:
                    state[1] += 1
                    if state[0] != current_sender:
                        state[2] += 1
                state[0] = current_sender
            else:
                state[0] = None
            team_leaders[team].update(senders)

        # Counts and graph sets are commutative within a timestamp.
        for sender, recipients, _ in batch:
            is_group = len(recipients) >= 2
            if is_group:
                led_to[sender].update(recipients)
            for recipient in recipients:
                count[(sender, recipient)] += 1
                out_strength[sender] += 1
                in_strength[recipient] += 1
                out_neighbors[sender].add(recipient)
                in_neighbors[recipient].add(sender)
                undirected_neighbors[sender].add(recipient)
                undirected_neighbors[recipient].add(sender)
                if is_group:
                    group_count[(sender, recipient)] += 1
                    group_in_neighbors[recipient].add(sender)

        # Reciprocal-neighbor status is evaluated after all batch directions
        # are inserted, so simultaneous opposite edges are represented equally.
        for sender, recipient in affected_directions:
            if count[(recipient, sender)] > 0:
                reciprocal_neighbors[sender].add(recipient)
                reciprocal_neighbors[recipient].add(sender)

        i = j

    return pd.DataFrame(rows)


# Backward-compatible concise alias for callers.
build = build_causal_rows
