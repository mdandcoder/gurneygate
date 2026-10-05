# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Simulator: no camera, no model. Generates fake stretcher/person motion and runs it through
the real decision (Kalman + TTA) and door layers. Counts correct and wrong openings.

python -m gurneygate.sim                 # with a window
python -m gurneygate.sim --headless      # report only
"""
from __future__ import annotations

import argparse
import random
import time

import cv2
import numpy as np
import yaml

from .decision import ApproachDecider
from .detector import Detection
from .door import make_door
from .main import draw

W, H = 960, 540


def scenario_list():
    """(name, should open?, generator). A generator takes t (seconds) and returns (stretchers, persons).
    Speeds are units/second: changing --fps samples the same PHYSICAL scene at a different rate."""
    def approach(speed=0.6, occupied=True, jitter=0.004):
        def gen(t):
            cy = 0.1 + t * speed + random.uniform(-jitter, jitter)
            if cy > 1.1:
                return None
            s = Detection(1, "stretcher", 0.9, 0.35, cy - 0.15, 0.65, cy + 0.15)
            p = [Detection(2, "person", 0.9, 0.42, cy - 0.1, 0.58, cy + 0.1)] if occupied else []
            return [s], p
        return gen

    def recede(speed=0.6):
        def gen(t):
            cy = 0.9 - t * speed
            if cy < -0.1:
                return None
            return [Detection(1, "stretcher", 0.9, 0.35, cy - 0.15, 0.65, cy + 0.15)], []
        return gen

    def crossing():  # stretcher passing across the corridor in front of the door
        def gen(t):
            cx = -0.1 + t * 0.6
            if cx > 1.1:
                return None
            return [Detection(1, "stretcher", 0.9, cx - 0.15, 0.4, cx + 0.15, 0.7)], []
        return gen

    def parked():
        def gen(t):
            if t > 3.0:
                return None
            j = random.uniform(-0.005, 0.005)
            return [Detection(1, "stretcher", 0.9, 0.35, 0.45 + j, 0.65, 0.75 + j)], []
        return gen

    def far_then_stop():  # approaches, stops far from the door
        def gen(t):
            if t > 3.4:
                return None
            cy = min(0.25, 0.05 + t * 0.3)  # stops 0.6 short of the door
            return [Detection(1, "stretcher", 0.9, 0.35, cy - 0.15, 0.65, cy + 0.15)], []
        return gen

    def cart(mixed=False):  # approaching supply cart; mixed: the model says "stretcher" half the time
        def gen(t):
            cy = 0.1 + t * 0.5
            if cy > 1.1:
                return None
            name = random.choice(("stretcher", "cart")) if mixed else "cart"   # coin flip every frame
            return [Detection(1, name, 0.9, 0.35, cy - 0.15, 0.65, cy + 0.15)], []
        return gen

    def door_pass():  # passes across right at the door, drifts slightly toward it, box jitters
        def gen(t):
            cx = -0.1 + t * 0.45
            if cx > 1.1:
                return None
            y2 = 0.76 + 0.02 * t + random.uniform(-0.02, 0.02)
            return [Detection(1, "stretcher", 0.9, cx - 0.15, y2 - 0.3, cx + 0.15, y2)], []
        return gen

    def door_parked():  # stretcher parked right at the door, jittering
        def gen(t):
            if t > 3.0:
                return None
            j = random.uniform(-0.02, 0.02)
            return [Detection(1, "stretcher", 0.9, 0.35, 0.50 + j, 0.65, 0.80 + j)], []
        return gen

    return [
        ("fast approach, occupied stretcher", True, approach(0.9)),
        ("slow approach, occupied stretcher", True, approach(0.24)),
        ("approaching empty stretcher", True, approach(0.6, occupied=False)),  # opens when require_occupant=false
        ("receding stretcher", False, recede()),
        ("stretcher passing across", False, crossing()),
        ("parked stretcher (jitter)", False, parked()),
        ("stretcher stopping far away", False, far_then_stop()),
        ("approaching supply cart", False, cart()),
        ("half-mislabelled cart (stretcher/cart)", False, cart(mixed=True)),
        ("stretcher passing at the door", False, door_pass()),
        ("stretcher parked at the door", False, door_parked()),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--fps", type=float, default=30)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    random.seed(7)
    results = []
    for name, expect, gen in scenario_list():
        decider = ApproachDecider(cfg["decision"], fps=args.fps)
        door = make_door({"backend": "dryrun"}, cfg["decision"]["hold_open_seconds"], 0.0)
        door._open = lambda: None
        door._close = lambda: None
        opened, open_frame, first_dist = False, None, None
        first_seen_t, arrive_t = None, None
        i = 0
        while True:
            out = gen(i / args.fps)
            if out is None:
                break
            dets, persons = out
            t = i / args.fps
            if dets and first_seen_t is None:
                first_seen_t = t
            if dets and arrive_t is None and dets[0].y2 >= decider.door_line:
                arrive_t = t                       # leading edge reached the door line (ground truth)
            should, infos = decider.update(dets, persons)
            if should:
                door.trigger()
                if not opened:
                    opened, open_frame = True, i
                    first_dist = decider._distance(decider._axis(dets[0].cx, dets[0].cy, 0, 0)[0])
            door.tick()
            if not args.headless:
                frame = np.full((H, W, 3), 40, np.uint8)
                for d in dets:
                    cv2.rectangle(frame, (int(d.x1 * W), int(d.y1 * H)), (int(d.x2 * W), int(d.y2 * H)), (120, 120, 120), -1)
                for p in persons:
                    cv2.ellipse(frame, (int(p.cx * W), int(p.cy * H)), (int((p.x2 - p.x1) * W / 2), int((p.y2 - p.y1) * H / 2)), 0, 0, 360, (200, 180, 160), -1)
                cv2.putText(frame, name, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Sim", draw(frame, infos, decider, door, args.fps))
                if cv2.waitKey(int(1000 / args.fps)) & 0xFF in (27, ord("q")):
                    args.headless = True
            i += 1
        door_t = float(cfg["decision"]["door_open_time"])
        lead = (arrive_t - open_frame / args.fps) if (opened and arrive_t is not None) else None
        max_lead = (arrive_t - first_seen_t) if (arrive_t is not None and first_seen_t is not None) else None
        feasible = max_lead is not None and max_lead >= door_t + 0.3
        ok = opened == expect
        # feasible or not: late if the door is still not open when the leading edge reaches the line
        late = expect and opened and (lead is None or lead <= 0 or (feasible and lead < door_t))
        if late:
            ok = False
        results.append((name, expect, opened, open_frame, first_dist, ok, lead, max_lead, feasible, late))
    try:
        cv2.destroyAllWindows()
    except cv2.error:               # headless OpenCV builds have no GUI
        pass
    print(f"\n{args.fps:.0f} fps; lead: how many s before the leading edge reached the line the door opened "
          f"(needed {cfg['decision']['door_open_time']} s); max: upper bound allowed by the field of view")
    print("Scenario                                 expected  opened  frame lead     max    result")
    for name, e, o, f, dist, ok, lead, mx, feas, late in results:
        fd = f"{f:>4}" if f is not None else "   -"
        ld = f"{lead:6.2f}s" if lead is not None else "      -"
        md = f"{mx:5.2f}s" if mx is not None else "     -"
        tag = "OK" if ok else ("LATE" if late else "FAIL")
        if ok and e and o and not feas:
            tag = "OK (view too short)"
        print(f"{name:<40} {str(e):<9} {str(o):<7} {fd}  {ld} {md}  {tag}")
    n_ok = sum(r[5] for r in results)
    print(f"\n{n_ok}/{len(results)} scenarios correct")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
