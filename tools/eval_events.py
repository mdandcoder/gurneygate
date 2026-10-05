# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Event-level field scorer (for door-camera recordings).

Input 1: ground-truth CSV (by hand): clip,kind,t
  kind: transport (a patient transport passing the door; t = moment the leading edge crosses the threshold, s)
        cart | pedestrian (a passage that must not open the door; t = passage time)
  Plus one row per clip: clip,duration,<seconds>   (for unmatched opens per hour)
Input 2: the system's event log (logs/events.csv, column 't' is scene time), one per clip:
  --events clipA=logs/a.csv clipB=logs/b.csv

Matching: a door_open belongs to a transport passage when t_open is in [t - max_lead, t].
Report (Clopper-Pearson 95%):
- on-time opening rate (lead >= door_open_time)   - opening rate with any lead
- false-open rate on cart/pedestrian passages      - unmatched opens per hour
- lead median and quartiles

python tools/eval_events.py --gt field_gt.csv --events door1=logs/events.csv --door-open-time 1.5
"""
import argparse, csv, math, statistics
from collections import defaultdict


def _cdf(x, k, n):
    """P(X <= k), X ~ Bin(n, x). In log space so large n does not overflow."""
    if k < 0:
        return 0.0
    if x <= 0.0:
        return 1.0
    if x >= 1.0:
        return 1.0 if k >= n else 0.0
    lx, l1 = math.log(x), math.log1p(-x)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lx + (n - i) * l1
             for i in range(k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial 95% interval (by bisection; no scipy needed)."""
    if n == 0:
        return float("nan"), float("nan")
    lower = 0.0 if k == 0 else _lower(k, n, alpha, _cdf)
    upper = 1.0 if k == n else _upper(k, n, alpha, _cdf)
    return lower, upper


def _lower(k, n, alpha, cdf):
    lo, hi = 0.0, 1.0            # P(X >= k | p) = alpha/2 ; increases with p
    for _ in range(60):
        mid = (lo + hi) / 2
        if 1 - cdf(mid, k - 1, n) < alpha / 2:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _upper(k, n, alpha, cdf):
    lo, hi = 0.0, 1.0            # P(X <= k | p) = alpha/2 ; decreases with p
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid, k, n) > alpha / 2:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def score(gt_rows, opens_by_clip, door_open_time=1.5, max_lead=6.0):
    """gt_rows: [(clip, kind, t)], opens_by_clip: {clip: [t_open,...]}, durations: kind=='duration'."""
    dur = {c: t for c, k, t in gt_rows if k == "duration"}
    trans = [(c, t) for c, k, t in gt_rows if k == "transport"]
    neg = [(c, t) for c, k, t in gt_rows if k in ("cart", "pedestrian")]
    used = defaultdict(set)
    leads = []
    for c, t in sorted(trans, key=lambda x: x[1]):
        cand = [(i, to) for i, to in enumerate(opens_by_clip.get(c, []))
                if t - max_lead <= to <= t and i not in used[c]]
        if cand:
            i, to = max(cand, key=lambda x: x[1])   # the latest opening before the passage
            used[c].add(i)
            leads.append(t - to)
        else:
            leads.append(None)
    on_time = sum(1 for l in leads if l is not None and l >= door_open_time)
    any_open = sum(1 for l in leads if l is not None)
    neg_opened = 0
    for c, t in neg:                               # same window as transports: a cart would trigger ~2.2 s before passing
        if any(t - max_lead <= to <= t + 0.5 and i not in used[c] for i, to in enumerate(opens_by_clip.get(c, []))):
            neg_opened += 1
    unmatched = sum(len(v) - len(used[c]) for c, v in opens_by_clip.items())
    hours = sum(dur.values()) / 3600 if dur else float("nan")
    got = sorted(l for l in leads if l is not None)
    if len(got) >= 2:
        q25, q50, q75 = statistics.quantiles(got, n=4, method="inclusive")
    else:
        q25 = q50 = q75 = got[0] if got else float("nan")
    return dict(n_transport=len(trans), on_time=on_time, any_open=any_open,
                n_neg=len(neg), neg_opened=neg_opened, unmatched_opens=unmatched, hours=hours,
                lead_q25=q25, lead_med=q50, lead_q75=q75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--events", nargs="+", required=True, help="clip=events.csv")
    ap.add_argument("--door-open-time", type=float, default=1.5)
    a = ap.parse_args()
    gt = []
    for r in csv.reader(open(a.gt, encoding="utf-8")):
        if r and not r[0].startswith("#") and r[0] != "clip":
            gt.append((r[0], r[1].strip(), float(r[2])))
    opens = {}
    for spec in a.events:
        clip, path = spec.split("=", 1)
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        last = rows[-1].get("run") if rows else None      # if the file holds several runs, the last one
        opens[clip] = [float(r["t"]) for r in rows if r.get("run") == last
                       and r.get("event") in ("door_open", "door_extend")]
    s = score(gt, opens, a.door_open_time)
    n = s["n_transport"]
    lo, hi = clopper_pearson(s["on_time"], n)
    print(f"transport passages: {n}")
    print(f"  on-time opening (lead >= {a.door_open_time} s): {s['on_time']}/{n}  [95% {lo:.2f}-{hi:.2f}]")
    lo, hi = clopper_pearson(s["any_open"], n)
    print(f"  opened with any lead: {s['any_open']}/{n}  [95% {lo:.2f}-{hi:.2f}]")
    print(f"  lead quartiles: {s['lead_q25']:.2f} / {s['lead_med']:.2f} / {s['lead_q75']:.2f} s")
    lo, hi = clopper_pearson(s["neg_opened"], s["n_neg"])
    print(f"false opens on cart/pedestrian passages: {s['neg_opened']}/{s['n_neg']}  [95% {lo:.2f}-{hi:.2f}]")
    print(f"unmatched opens: {s['unmatched_opens']}  ({s['unmatched_opens'] / s['hours']:.2f} / hour)"
          if s["hours"] == s["hours"] else f"unmatched opens: {s['unmatched_opens']} (no duration row)")


if __name__ == "__main__":
    main()
