# Data

All loaders live in `src/datasets.py`; each returns temporal single-sender
events `(sender, recipients, timestamp)` with in-event self-loops removed and
duplicates kept.

| key | file | source | events (3 <= k <= 25, dups kept) |
|---|---|---|---|
| `fauci` | `fauci/fauci_temporal_hyperedges.csv` | processed from the Fauci email corpus (Benson et al., leopold-nih-foia release) | 408 |
| `dnc` | `dnc/email-dnc.edges` | KONECT "DNC emails" | 5,200 |
| `twitter` | `twitter/twitter_observed_min1_events.parquet` | processed mention events from the Twitter follower network crawl | 15,760 |
| `enron` | `enron/out.enron` | KONECT "Enron employees" temporal edges | 57,423 |
| `emaileu` | `emailEu/email-Eu-core-temporal.txt` | SNAP `email-Eu-core-temporal` | (appendix base-measure study) |
| `radoslaw` | `radoslaw/out.radoslaw_email_email` | KONECT "Manufacturing emails" (Radoslaw Michalski et al.) | 7,906 |
| `higgs` | `higgs/higgs-activity_time.txt.gz` | SNAP "Higgs Twitter" mention (MT) layer, July 2012 (De Domenico et al. 2013) | 16,381 |
| `wiki` | `wiki/talk_hyperedges.csv` | processed Simple English Wikipedia talk-page discussions | 927 |

For pairwise-timestamped sources (emaileu, enron, dnc, radoslaw, higgs) an event is formed by
merging all recipients a sender addressed at the same timestamp; fauci and
twitter are native hyperedge tables.

The raw pairwise files are redistributed unchanged from their public sources
(SNAP: https://snap.stanford.edu/data/, KONECT: http://konect.cc/); please
cite the original sources alongside this repository.

Non-bundled extras: `load_events_simplices` (Cornell temporal simplex format,
e.g. coauth-DBLP-full) and `load_events_congress` (congress-bills) support the
non-communication datasets discussed in the appendix; download those from
https://www.cs.cornell.edu/~arb/data/.
