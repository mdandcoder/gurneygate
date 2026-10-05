# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Unit tests for the decision logic (no model, no camera)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gurneygate.decision import ApproachDecider
from gurneygate.detector import Detection

CFG = dict(door_line_y=0.85, approach_direction="down", 
           min_speed_per_s=0.06, min_box_area=0.03, trigger_distance=0.45)


def box(tid, cy, size=0.3):
    return Detection(tid, "stretcher", 0.9, 0.35, cy - size / 2, 0.65, cy + size / 2)


def run(seq):
    d = ApproachDecider(CFG)
    opened = []
    for frame in seq:
        o, _ = d.update(frame)
        opened.append(o)
    return opened


def test_approaching_stretcher_opens():
    seq = [[box(1, 0.2 + i * 0.03)] for i in range(15)]
    out = run(seq)
    assert any(out), "a stretcher moving toward the door must open it"
    # must not open during the first 4 frames (debounce)
    assert not any(out[:4])


def test_receding_stretcher_does_not_open():
    seq = [[box(1, 0.8 - i * 0.03)] for i in range(15)]
    assert not any(run(seq)), "a receding stretcher must not open the door"


def test_stationary_stretcher_does_not_open():
    seq = [[box(1, 0.6)] for _ in range(15)]
    assert not any(run(seq)), "a stationary stretcher must not open the door"


def test_far_small_object_ignored():
    seq = [[box(1, 0.2 + i * 0.03, size=0.05)] for i in range(15)]
    assert not any(run(seq)), "a very distant/small box must be ignored"


def test_jitter_does_not_open():
    # box jittering back and forth (typical false-positive behaviour)
    seq = [[box(1, 0.6 + (0.01 if i % 2 else -0.01))] for i in range(20)]
    assert not any(run(seq))


def test_lost_track_cleanup():
    # a single empty frame does not delete the track (short occlusion); dropped after max_lost_seconds
    d = ApproachDecider(CFG, fps=30)
    d.update([box(1, 0.5)])
    d.update([])
    assert 1 in d.tracks
    for _ in range(60):
        d.update([])
    assert 1 not in d.tracks


def test_tta_estimate_reasonable():
    d = ApproachDecider(CFG, fps=30)
    # 0.9 units/s; last box leading edge 0.68 -> 0.17 to the line -> ~0.19 s
    for i in range(12):
        d.update([box(1, 0.2 + i * 0.03)])
    tta = d.tracks[1].last_tta
    assert 0.1 < tta < 0.35, tta


def test_slow_approach_opens_when_near():
    # stretcher approaching very slowly: long TTA but inside trigger_distance -> open
    seq = [[box(1, 0.5 + i * 0.004)] for i in range(40)]
    assert any(run(seq))


def test_fast_approach_opens_early():
    # a fast stretcher must open while still far away (short TTA)
    d = ApproachDecider(dict(CFG, trigger_distance=0.1), fps=30)
    opened_at = None
    for i in range(20):
        o, _ = d.update([box(1, 0.05 + i * 0.04)])
        if o and opened_at is None:
            opened_at = 0.05 + i * 0.04
    assert opened_at is not None and opened_at < 0.75, opened_at


def test_require_occupant_blocks_empty_stretcher():
    cfg = dict(CFG, require_occupant=True)
    d = ApproachDecider(cfg)
    out = [d.update([box(1, 0.2 + i * 0.03)])[0] for i in range(15)]
    assert not any(out)
    d2 = ApproachDecider(cfg)
    out2 = []
    for i in range(15):
        s = box(1, 0.2 + i * 0.03)
        person = Detection(9, "person", 0.9, s.x1 + 0.05, s.y1 + 0.05, s.x2 - 0.05, s.y2 - 0.05)
        out2.append(d2.update([s], [person])[0])
    assert any(out2)


def test_crossing_reported_once():
    d = ApproachDecider(CFG)
    crossed = []
    for i in range(30):
        d.update([box(1, 0.2 + i * 0.03)])
        crossed += d.last_crossed
    assert crossed == [1]


def test_draw_runs():
    import numpy as np
    from gurneygate.main import draw
    from gurneygate.door import DryRunDoor
    d = ApproachDecider(CFG)
    _, infos = d.update([box(1, 0.5)])
    frame = np.zeros((360, 640, 3), np.uint8)
    draw(frame, infos, d, DryRunDoor(1.0, 0.0), 30.0)


def test_born_past_line_not_counted_as_crossing():
    d = ApproachDecider(CFG)
    crossed = []
    for _ in range(5):
        d.update([box(1, 0.9)])  # bottom edge 1.05 > 0.85, past the line from the start
        crossed += d.last_crossed
    assert crossed == []


def test_debounce_and_stability_gates_each_delay_opening():
    """Each gate (consecutive approach, track stability) must delay opening on its own."""
    seq = [[box(1, 0.2 + i * 0.03)] for i in range(20)]
    def first_open(cfg):
        d = ApproachDecider(cfg, fps=30)
        return next(i for i, f in enumerate(seq) if d.update(f)[0])
    base = dict(CFG, min_stable_seconds=0.0, min_consecutive_seconds=0.0)
    f0 = first_open(base)
    assert first_open(dict(base, min_consecutive_seconds=0.4)) > f0
    assert first_open(dict(base, min_stable_seconds=0.5)) > f0


def test_crossing_reported_once_with_edge_jitter():
    d = ApproachDecider(CFG)
    crossed = []
    for i in range(20):
        d.update([box(1, 0.5 + i * 0.03)])
        crossed += d.last_crossed
    for i in range(20):                  # leading edge jitters on both sides of the line
        d.update([box(1, 0.70 + (0.02 if i % 2 else -0.02))])
        crossed += d.last_crossed
    assert crossed == [1], crossed
