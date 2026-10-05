# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from eval_door import wilson, mcnemar, iou


def test_wilson_known_value():
    p, lo, hi = wilson(8, 9)             # 8/9 -> 95% Wilson 0.565-0.980
    assert abs(p - 0.889) < 1e-3 and 0.55 < lo < 0.58 and 0.97 < hi < 1.0


def test_mcnemar_symmetric_and_exact():
    assert mcnemar([1, 1, 0], [1, 1, 0])[2] == 1.0
    b, c, p = mcnemar([1] * 10 + [0] * 5, [0] * 10 + [0] * 5)
    assert (b, c) == (10, 0) and p < 0.01


def test_iou():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert abs(iou((0, 0, 2, 2), (1, 1, 3, 3)) - 1 / 7) < 1e-9


def test_greedy_matching_is_global_not_file_order():
    import eval_door as E
    # GT0 overlaps both predictions, GT1 only P0: file order would give P0 to GT0
    rows = [("stretcher", (0.0, 0.0, 0.5, 0.5)), ("stretcher", (0.05, 0.0, 0.55, 0.5))]
    preds = [((0.05, 0.0, 0.55, 0.5), True), ((0.0, 0.0, 0.45, 0.5), True)]
    got, _ = E.match(rows, preds)
    assert got == {0, 1}


def test_ignore_region_not_false_positive(tmp_path):
    import eval_door as E
    rows = [("ignore", (0.8, 0.3, 0.95, 0.6)), ("stretcher", (0.1, 0.1, 0.4, 0.4))]
    preds = [((0.82, 0.32, 0.93, 0.58), True), ((0.1, 0.1, 0.4, 0.4), True)]
    ign = [gb for g, gb in rows if g == "ignore"]
    real = [r for r in rows if r[0] != "ignore"]
    got, used = E.match(real, preds)
    fp = sum(1 for pj, (pb, o) in enumerate(preds) if o and pj not in used and not any(E._ioa(pb, g) >= 0.5 for g in ign))
    assert got == {0} and fp == 0


def test_min_iou_loose_match():
    import eval_door as E
    rows = [("stretcher", (0.0, 0.0, 0.5, 0.5))]
    preds = [((0.0, 0.0, 0.5, 1.0), True)]          # same object, box twice as tall: IoU 0.5 -> just in
    preds_wide = [((0.0, 0.0, 1.0, 0.7), True)]     # IoU 0.36: a box-shape miss, but the object was seen
    assert E.match(rows, preds)[0] == {0}
    assert E.match(rows, preds_wide)[0] == set()
    assert E.match(rows, preds_wide, min_iou=0.3)[0] == {0}
