# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Stretcher-aware automatic door: main loop.

Usage:
  python -m gurneygate.main                 # source from config.yaml (fixed door camera)
  python -m gurneygate.main --source video.mp4 --no-show
  python -m gurneygate.main --source clip.mp4 --moving-camera   # handheld/tracking-shot test clips
  python -m gurneygate.main --source rtsp://... --weights weights/gurneygate-yolo26s.pt
"""
from __future__ import annotations

import argparse
import os
import signal
import threading
import time

import cv2
import numpy as np
import yaml

from .decision import ApproachDecider
from .detector import StretcherDetector
from .door import make_door
from .metrics import EventLog


class Watchdog:
    """Releases the trigger if the main loop stalls (blocked RTSP read, frozen inference).
    The loop calls beat() every frame; if it is not called for timeout_s the door is forced closed."""

    def __init__(self, door, timeout_s: float, on_fire=None, period_s: float = 0.1):
        self.door, self.timeout_s, self.on_fire, self.period_s = door, timeout_s, on_fire, period_s
        self._last = time.monotonic()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def beat(self):
        self._last = time.monotonic()

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)

    def _run(self):
        while not self._stop.wait(self.period_s):
            gap = time.monotonic() - self._last
            if gap > self.timeout_s and self.door.force_close() and self.on_fire:
                self.on_fire(gap)


def draw(frame, infos, decider, door, fps):
    h, w = frame.shape[:2]
    d = decider.direction
    if d in ("down", "up"):
        y = int(decider.door_line * h)
        cv2.line(frame, (0, y), (w, y), (0, 200, 255), 2)
        cv2.putText(frame, "DOOR", (10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    else:
        x = int(decider.door_line * w)
        cv2.line(frame, (x, 0), (x, h), (0, 200, 255), 2)
    for info in infos:
        det = info["det"]
        x1, y1, x2, y2 = int(det.x1 * w), int(det.y1 * h), int(det.x2 * w), int(det.y2 * h)
        hot = info.get("opens", False)
        color = (0, 0, 255) if hot else ((0, 255, 0) if info.get("class_ok", True) else (255, 160, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        occ = " [patient]" if info.get("occupied") else ""
        cname = info.get("cls_smooth") or det.cls_name
        flip = "*" if cname != det.cls_name.lower() else ""
        label = f"#{det.track_id} {cname}{flip} {det.conf:.2f}{occ} | {info['status']}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    egos = ""
    if infos and abs(infos[0].get("ego_scale", 1.0) - 1.0) > 0.004:
        egos = f"  |  CAMERA {'forward' if infos[0]['ego_scale'] > 1 else 'backward'}"
    state = f"DOOR: {'OPEN ' + f'{door.remaining():.1f}s' if door.is_open else 'closed'}  |  {fps:.1f} FPS{egos}"
    cv2.putText(frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if door.is_open else (255, 255, 255), 2)
    return frame


def _is_file(src) -> bool:
    """Is this a video file on disk? rtsps/rtmp/udp streams and /dev/video* are live."""
    return isinstance(src, str) and os.path.isfile(src) and not src.startswith("/dev/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="0, a video path or an rtsp URL")
    ap.add_argument("--weights")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="for tests: stop after n frames")
    ap.add_argument("--save", help="annotated output video path (mp4). Frames are written ONLY if this is given.")
    ap.add_argument("--moving-camera", action="store_true",
                    help="handheld/tracking shot: camera-motion compensation + GMC tracker (keep off on a fixed door camera)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.weights:
        cfg["model"]["weights"] = args.weights
    if args.moving_camera:
        cfg["decision"]["use_egomotion"] = True
        cfg["model"]["tracker"] = "botsort_moving.yaml"
    src = args.source if args.source is not None else cfg["camera"]["source"]
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    show = cfg["display"]["show"] and not args.no_show

    detector = StretcherDetector(cfg["model"])
    if cfg["decision"].get("require_occupant") and not detector.person_ids:
        raise SystemExit("[config] require_occupant: true but the model has no 'person' class; "
                         "the door would never open. Turn the setting off or use a model with a person class.")

    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/video: {src}")

    # time base: for files the source frame rate and frame index; live, the measured processing rate
    is_file = _is_file(src)
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    src_fps = cap_fps if 1.0 < cap_fps < 240.0 else float(cfg["camera"].get("fps", 30))
    n = 0
    clock = (lambda: n / src_fps) if is_file else time.monotonic
    decider = ApproachDecider(cfg["decision"], fps=src_fps, open_classes=cfg["model"]["stretcher_classes"])
    print(f"[time] {'file' if is_file else 'live'}, decision frame rate {src_fps:.1f}")

    from .egomotion import EgoMotion
    ego_est = EgoMotion() if cfg["decision"].get("use_egomotion", False) else None
    if ego_est:
        print("[ego] camera-motion compensation on")
    depth_est = None
    if cfg["decision"].get("use_depth", False):
        from .depth import DepthEstimator
        dcfg = cfg.get("depth", {})
        depth_est = DepthEstimator(size=int(dcfg.get("input_size", 294)))
        depth_every = int(dcfg.get("every_n_frames", 2)); depth_map = None
        print(f"[depth] Depth-Anything-V2-Small, input {depth_est.size}px, every {depth_every} frames, device {depth_est.device}")

    door = make_door(cfg["door"], cfg["decision"]["hold_open_seconds"], cfg["decision"]["cooldown_seconds"])
    log = EventLog(cfg.get("logging", {}).get("events_csv"), clock=clock)
    proc_w = int(cfg["camera"].get("process_width", 0))
    watchdog_s = float(cfg.get("safety", {}).get("max_frame_gap_seconds", 1.0))
    wd = Watchdog(door, watchdog_s, on_fire=lambda gap: log.log("watchdog", note=f"frame gap {gap:.2f}s"))

    def _term(signum, _frame):
        raise SystemExit(f"terminated by signal {signum}")
    signal.signal(signal.SIGTERM, _term)

    t0, fps = time.time(), 0.0
    low_fps_warned = False
    triggers = 0
    writer = None
    if args.save:
        w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w0, h0))
    wd.start()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            wd.beat()
            if cfg["camera"].get("flip"):
                frame = cv2.flip(frame, 1)
            proc = frame
            if proc_w and frame.shape[1] > proc_w:
                # privacy + speed: process at low resolution (faces not recognisable); normalised boxes are unchanged
                proc = cv2.resize(frame, (proc_w, int(frame.shape[0] * proc_w / frame.shape[1])))
            dets, persons = detector.track(proc)
            if ego_est is not None:
                ego_est.update(proc, [(d.x1, d.y1, d.x2, d.y2) for d in dets + persons])
            depths = None
            if depth_est is not None and dets:
                if depth_map is None or n % depth_every == 0:
                    depth_map = depth_est(proc)
                from .depth import box_depth
                ph, pw = proc.shape[:2]
                if decider.direction in ("down", "up"):
                    y = int(decider.door_line * ph); band = depth_map[max(0, y - int(0.03 * ph)):min(ph, y + int(0.03 * ph)), :]
                else:
                    x = int(decider.door_line * pw); band = depth_map[:, max(0, x - int(0.03 * pw)):min(pw, x + int(0.03 * pw))]
                door_depth = float(np.median(band)) if band.size else None
                depths = {d.track_id: (box_depth(depth_map, (d.x1 * pw, d.y1 * ph, d.x2 * pw, d.y2 * ph)), door_depth) for d in dets}
            should_open, infos = decider.update(dets, persons, depths, ego_est, t=clock())
            if should_open:
                opened = door.trigger()
                triggers += opened
                hot = next((i for i in infos if i.get("opens")), None)
                # a second transport arriving while the door is already open is also logged (door_extend);
                # otherwise its crossing would count as a miss
                if hot and (opened or hot["id"] not in log.open_t):
                    log.door_opened(hot["id"], hot["tta"], hot["occupied"], extend=not opened)
            for tid in decider.last_crossed:
                log.crossed_line(tid)
            for tid in decider.last_lost:
                log.track_lost(tid)
            door.tick()
            n += 1
            if n % 10 == 0:
                fps = 10 / max(1e-6, time.time() - t0)
                t0 = time.time()
                if not is_file:      # live: decisions run at the rate frames are actually processed
                    decider.fps = min(60.0, max(2.0, fps))
                    if fps < 5.0 and not low_fps_warned:
                        low_fps_warned = True
                        print(f"[warning] processing rate {fps:.1f} fps: below 5 fps a fast stretcher cannot be "
                              "served in time (simulator: a fast approach is missed at 4 fps)")
            if show or writer is not None:
                vis = draw(frame, infos, decider, door, fps)
                if writer is not None:
                    writer.write(vis)
            if show:
                cv2.imshow("GurneyGate", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        wd.stop()
        door.shutdown()            # the relay is released whatever the exit path
        cap.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:          # headless OpenCV builds have no GUI
            pass
        log.log("shutdown", note=f"{n} frames, {triggers} triggers")
        log.close()
        print(f"[summary] {n} frames, {triggers} door triggers")


if __name__ == "__main__":
    main()
