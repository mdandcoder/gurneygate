# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from eval_events import clopper_pearson, score


def test_clopper_pearson_known():
    lo, hi = clopper_pearson(8, 9)          # 95% Clopper-Pearson: 51.8-99.7%
    assert abs(lo - 0.518) < 0.002 and abs(hi - 0.997) < 0.002
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0


def test_score_matching():
    gt = [("k", "duration", 3600.0), ("k", "transport", 10.0), ("k", "transport", 30.0),
          ("k", "transport", 50.0), ("k", "cart", 70.0)]
    opens = {"k": [8.0, 29.5, 69.5, 90.0]}     # 2.0 s on time, 0.5 s late, 1 miss, open on a cart, 1 stray
    s = score(gt, opens, door_open_time=1.5)
    assert (s["n_transport"], s["on_time"], s["any_open"]) == (3, 1, 2)
    assert s["neg_opened"] == 1 and s["unmatched_opens"] == 2


def test_clopper_pearson_large_n_no_overflow():
    lo, hi = clopper_pearson(600, 1200)
    assert 0.47 < lo < 0.5 < hi < 0.53


def test_cart_open_before_crossing_counted():
    gt = [("c", "cart", 10.0), ("c", "duration", 60.0)]
    s = score(gt, {"c": [7.5]})          # opened 2.5 s before: TTA threshold is 2.2 s
    assert s["neg_opened"] == 1


def test_quartiles_not_biased():
    gt = [("c", "transport", t) for t in (10, 20, 30, 40)]
    s = score(gt, {"c": [9, 18, 27, 36]})   # leads 1,2,3,4
    assert abs(s["lead_med"] - 2.5) < 1e-9
