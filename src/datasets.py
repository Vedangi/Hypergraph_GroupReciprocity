"""datasets.py -- unified loaders for the communication hypergraph datasets.

Every loader returns a list of temporal single-sender events
    (sender: int, recipients: sorted list[int], timestamp: int)
with self-loops removed inside each event. Duplicate (sender, R) events are
KEPT -- deduplication (when wanted) is done by the measures, not the loader.

Datasets bundled under data/ (see data/README.md for provenance):
    emaileu   email-Eu-core-temporal (SNAP)         -- triples, merged on (sender, time)
    enron     Enron employee emails (KONECT)        -- triples, merged on (sender, time)
    dnc       DNC email network (KONECT)            -- comma triples, merged on (sender, time)
    fauci     Fauci email hyperedges (processed)    -- native hyperedge CSV
    twitter   Twitter mention events (processed)    -- native hyperedge parquet

The triple-merging convention (one event = all recipients a sender addressed
at the same timestamp) follows the loaders used for all experiments in the
paper; the per-dataset parsing quirks below intentionally reproduce them
byte-for-byte so results are reproducible.

Set HGR_DATA to point at a data directory other than <repo>/data.
"""
from __future__ import annotations

import ast
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get(
    "HGR_DATA", Path(__file__).resolve().parents[1] / "data"))

DATASETS = {
    "emaileu": DATA_DIR / "emailEu/email-Eu-core-temporal.txt",
    "enron":   DATA_DIR / "enron/out.enron",
    "dnc":     DATA_DIR / "dnc/email-dnc.edges",
    "fauci":   DATA_DIR / "fauci/fauci_temporal_hyperedges.csv",
    "twitter": DATA_DIR / "twitter/twitter_observed_min1_events.parquet",
}


# ---------------------------------------------------------------------------
# temporal triple files (one directed pairwise edge + timestamp per line)

def load_events_triples(path, limit=None):
    """Whitespace/comma triples; merge rows sharing (sender, time). t >= 0."""
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
    events = [(s, sorted(R), t) for (s, t), R in by_event.items() if R]
    events.sort(key=lambda e: (e[2], e[0]))
    return events[:limit] if limit else events


def load_events_triples_comma(path, limit=None):
    """Strict comma triples variant (used for dnc); requires t > 0."""
    by_event = defaultdict(set)
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            try:
                s, r, t = int(parts[0]), int(parts[1]), int(parts[-1])
            except ValueError:
                continue
            if s != r and t > 0:
                by_event[(s, t)].add(r)
    events = [(s, sorted(R), t) for (s, t), R in by_event.items() if R]
    events.sort(key=lambda e: (e[2], e[0]))
    return events[:limit] if limit else events


# ---------------------------------------------------------------------------
# native hyperedge files

def load_events_fauci(path, limit=None):
    """Fauci hyperedge CSV: sender, ';'-separated recipients, ISO timestamp."""
    df = pd.read_csv(path)
    sec = (
        pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .to_numpy(dtype="datetime64[s]")
        .astype("int64")
    )
    events, seen = [], set()
    for s, rc, t in zip(df["sender"], df["recipients"], sec):
        if pd.isna(s) or pd.isna(rc) or t <= 0:
            continue
        s = int(s)
        R = sorted({int(x) for x in str(rc).split(";")
                    if x != "" and int(x) != s})
        if not R:
            continue
        key = (s, tuple(R), int(t))
        if key in seen:
            continue
        seen.add(key)
        events.append((s, R, int(t)))
    events.sort(key=lambda e: (e[2], e[0]))
    return events[:limit] if limit else events


def load_events_twitter(path, limit=None):
    """Twitter mention events from a Parquet event table."""

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
    events, seen = [], set()
    for row in frame.itertuples(index=False):
        sender = int(row.sender)
        recipients = sorted({
            int(x) for x in as_list(row.recipients) if int(x) != sender})
        timestamp = int(row.timestamp)
        key = (sender, tuple(recipients), timestamp)
        if recipients and key not in seen:
            seen.add(key)
            events.append((sender, recipients, timestamp))
    events.sort(key=lambda e: (e[2], e[0]))
    return events[:limit] if limit else events


# ---------------------------------------------------------------------------
# optional loaders for non-communication data (files not bundled)

def load_events_simplices(folder, sender="first", max_size=25, limit=None):
    """Cornell temporal simplex format (e.g. coauth-DBLP-full)."""
    import glob

    folder = Path(folder)

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
    events.sort(key=lambda e: (e[2], e[0]))
    return events[:limit] if limit else events


def load_events_congress(folder, limit=None):
    """congress-bills sponsors.txt / Cosponsors.txt / Dates.txt."""
    folder = Path(folder)

    def read_lines(name):
        with open(folder / name) as fh:
            return [line.strip() for line in fh]

    sponsors = read_lines("sponsors.txt")
    cosponsors = read_lines("Cosponsors.txt")
    dates = read_lines("Dates.txt")
    if not (len(sponsors) == len(cosponsors) == len(dates)):
        raise ValueError("sponsors/Cosponsors/Dates line counts differ")

    parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
    epoch_days = parsed.astype("int64") // 10**9 // 86400
    events = []
    for i, sponsor in enumerate(sponsors):
        if sponsor in ("", "NA") or cosponsors[i] in ("", "NA"):
            continue
        if pd.isna(parsed.iloc[i]):
            continue
        try:
            sender = int(sponsor)
            recipients = sorted({int(x) for x in cosponsors[i].split()
                                 if x not in ("", "NA") and int(x) != sender})
        except ValueError:
            continue
        if recipients:
            events.append((sender, recipients, int(epoch_days.iloc[i])))
    events.sort(key=lambda e: (e[2], e[0]))
    return events[:limit] if limit else events


# ---------------------------------------------------------------------------

def load_dataset(name, limit=None, max_size=None):
    """Load one bundled dataset by key; optional team-size cap (|T| <= max_size)."""
    path = DATASETS[name]
    if name in ("emaileu", "enron"):
        events = load_events_triples(path)
    elif name == "dnc":
        events = load_events_triples_comma(path)
    elif name == "fauci":
        events = load_events_fauci(path)
    elif name == "twitter":
        events = load_events_twitter(path)
    else:
        raise ValueError(f"Unknown dataset: {name!r}")
    if max_size:
        events = [e for e in events if 1 + len(e[1]) <= max_size]
    return events[:limit] if limit else events
