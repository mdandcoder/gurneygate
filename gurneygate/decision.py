# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Approach decision based on a Kalman filter and time to arrival (TTA) at the door.

Idea: filter the door-side edge of a patient-transport box with a Kalman filter and compute
its time to reach the door line in SECONDS. The door needs T_open seconds to open, so trigger
when TTA <= T_open + margin; the door is then already open when the stretcher arrives.

Rules:
- A track survives short detection gaps on Kalman prediction and is dropped after max_lost_s.
- TTA and line crossing are measured at the same point: the door-side edge of the box.
- The track's VOTED class must be an opening class (carts never open the door).
- With the intent model on, TTA does not trigger before its window fills and it says "entering".
- The only exception is the at-door band (trigger_distance): a transport moving toward the
  door there opens regardless of intent. It is a safety band, not a long-range opening path.
- "Moving toward the door" comes from the least-squares slope of the leading edge over the
  last motion_window_s seconds, not from the Kalman's instantaneous velocity (robust to box
  jitter). The Kalman filter is kept for position and TTA speed.
All positions are normalised (0..1), speeds are units/second. Pure Python + numpy; unit-testable.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .intent import IntentModel
from .kalman import CVKalman

DEFAULT_OPEN_CLASSES = ("transport", "stretcher", "gurney", "bed", "hospital_bed", "wheelchair")


@dataclass
class TrackState:
    kf: CVKalman
    hist: list = field(default_factory=list)   # (t, cx, cy, area) for the intent model, t in seconds
    p_enter: float | None = None
    hits: int = 0
    last_seen: float = 0.0                      # decision time when last seen (s)
    approaching_frames: int = 0
    occupied: bool = False
    last_tta: float = float("inf")
    crossed: bool = False
    prev_edge_dist: float | None = None
    depth_hist: list = field(default_factory=list)   # relative inverse depth (larger = closer)
    depth_vel: float | None = None                    # depth change per second (+ = approaching)
    tta_depth: float = float("inf")
    cls_votes: Counter = field(default_factory=Counter)
    cls_smooth: str = ""
    motion: list = field(default_factory=list)  # (t, x, y) leading edge, camera motion removed
    ego_off: list = field(default_factory=lambda: [0.0, 0.0])   # accumulated camera shift


class ApproachDecider:
    def __init__(self, cfg: dict, fps: float = 30.0, open_classes=None):
        self.door_line = float(cfg.get("door_line_y", 0.85))
        self.direction = cfg.get("approach_direction", "down")
        # debounce thresholds are in seconds; converted to frames at the current rate every frame
        self.min_consec_s = float(cfg.get("min_consecutive_seconds", 4 / 30))
        self.min_speed = float(cfg.get("min_speed_per_s", 0.06))       # minimum speed toward the door (units/s)
        self.min_area = float(cfg.get("min_box_area", 0.03))
        # at-door band: a transport moving toward the door within this distance opens regardless of intent
        self.trigger_distance = float(cfg.get("trigger_distance", 0.12))
        self.max_trigger_distance = float(cfg.get("max_trigger_distance", 0.40))  # no TTA opening beyond this distance
        self.door_open_time = float(cfg.get("door_open_time", 1.5))
        self.lead_margin = float(cfg.get("lead_margin", 0.7))
        # constant-velocity Kalman covariance does not depend on the data; stability is observation time
        self.min_stable_s = float(cfg.get("min_stable_seconds", 8 / 30))
        self.max_lost_s = float(cfg.get("max_lost_seconds", 1.5))
        self.accel_std = float(cfg.get("kalman_accel_std", 28.5))   # units/s^2, Kalman process noise
        # "moving toward the door" uses the least-squares slope over a window; the Kalman's
        # instantaneous velocity jumps with box jitter and makes a parked track look approaching
        self.motion_window_s = float(cfg.get("motion_window_seconds", 0.5))
        self.band_min_speed = float(cfg.get("band_min_speed_per_s", 0.10))
        self.require_occupant = bool(cfg.get("require_occupant", False))
        self.open_classes = {c.lower() for c in (open_classes or cfg.get("open_classes") or DEFAULT_OPEN_CLASSES)}
        self.fps = float(fps)
        self.intent = IntentModel(cfg.get("intent_weights", "weights/intent.npz"))
        self.intent_threshold = float(cfg.get("intent_threshold", 0.5))
        self.use_intent = bool(cfg.get("use_intent", True)) and self.intent.ready
        if cfg.get("use_intent", True) and not self.intent.ready:
            print(f"[decision] WARNING: intent weights {cfg.get('intent_weights', 'weights/intent.npz')} not found "
                  "(paths are relative to the working directory); the intent cue is OFF")
        self.use_class_vote = bool(cfg.get("use_class_vote", True))
        # the lower confidence bound (Wilson, one-sided z) of the opening-class vote share must pass
        # this threshold: a track labelled "cart" half the time cannot open on an early lucky streak
        self.min_open_share = float(cfg.get("min_open_vote_share", 0.6))
        self.vote_z = float(cfg.get("vote_confidence_z", 1.64))
        self.vote_window = int(cfg.get("class_vote_window", 15))
        self.use_ego = bool(cfg.get("use_egomotion", False))
        self.use_depth = bool(cfg.get("use_depth", False))
        self.depth_min_vel = float(cfg.get("depth_min_vel_per_s", 0.12))
        self.depth_window = int(cfg.get("depth_window", 8))
        self.tracks: dict[int, TrackState] = {}
        self.frame = 0
        self._t_in = None                           # timestamp of the last frame passed in
        self.t = 0.0                                # decision clock (s), continuous across frame-rate changes
        self.last_crossed: list[int] = []
        self.last_lost: list[int] = []

    # ----- geometry -----
    def _axis(self, cx, cy, vx, vy):
        """(position, speed) along the door axis. Positive speed = toward the door."""
        if self.direction == "down":
            return cy, vy
        if self.direction == "up":
            return cy, -vy
        if self.direction == "right":
            return cx, vx
        return cx, -vx

    def _distance(self, p: float) -> float:
        if self.direction in ("down", "right"):
            return self.door_line - p
        return p - self.door_line

    def _lead_point(self, d):
        """Midpoint of the door-side box edge: used for TTA, crossing and (later) the ground point."""
        return {"down": (d.cx, d.y2), "up": (d.cx, d.y1),
                "right": (d.x2, d.cy), "left": (d.x1, d.cy)}[self.direction]

    def time_to_arrival(self, px, py, vx, vy) -> float:
        """Seconds; inf if not moving toward the door. (px, py) is the door-side edge, speed in units/s."""
        p, v = self._axis(px, py, vx, vy)
        dist = self._distance(p)
        if dist <= 0:
            return 0.0
        if v <= 1e-9:
            return float("inf")
        return dist / v

    @property
    def min_frames(self) -> int:
        """Consecutive 'approaching' frames required. Heading already comes from a slope over at least
        3 samples, so the floor is 2 (1 lets jitter open a parked stretcher, 3 is late at low fps)."""
        return max(2, round(self.min_consec_s * self.fps))

    @property
    def min_stable_hits(self) -> int:
        return max(3, round(self.min_stable_s * self.fps))

    @property
    def _win(self) -> float:
        """Slope window: long enough for at least 3 samples, so heading is measurable at low fps."""
        return max(self.motion_window_s, 2.5 / max(1e-3, self.fps))

    def _heading(self, st: TrackState):
        """(toward-door, lateral) speed of the leading edge over the last _win seconds, units/s. None if too few samples."""
        h = [m for m in st.motion if m[0] >= self.t - self._win - 1e-9]
        if len(h) < 3:
            return None
        t, x, y = (np.array(c, dtype=float) for c in zip(*h))
        sx, sy = np.polyfit(t, x, 1)[0], np.polyfit(t, y, 1)[0]
        return self._axis(0, 0, sx, sy)[1], (sx if self.direction in ("down", "up") else sy)

    def _class_of(self, st: TrackState, d) -> str:
        return (st.cls_smooth or d.cls_name).lower() if self.use_class_vote else d.cls_name.lower()

    # ----- depth (optional) -----
    def _depth_step(self, st: "TrackState", val, door_val):
        if val is None or not math.isfinite(val):
            return
        st.depth_hist.append(float(val))
        if len(st.depth_hist) > self.depth_window:
            st.depth_hist.pop(0)
        if len(st.depth_hist) >= 4:
            n = len(st.depth_hist); mx = (n - 1) / 2; my = sum(st.depth_hist) / n
            cov = sum((x - mx) * (y - my) for x, y in enumerate(st.depth_hist))
            var = sum((x - mx) ** 2 for x in range(n))
            st.depth_vel = (cov / var if var > 0 else 0.0) * self.fps   # units/s
            if door_val is not None and math.isfinite(door_val) and st.depth_vel > 1e-9 and door_val > val:
                st.tta_depth = (door_val - val) / st.depth_vel
            elif door_val is not None and math.isfinite(door_val) and door_val <= val:
                st.tta_depth = 0.0
            else:
                st.tta_depth = float("inf")

    # ----- main interface -----
    def update(self, detections, persons=None, depths=None, ego=None, t=None) -> tuple[bool, list[dict]]:
        """Called every frame. t: frame time (s); if given, the step is the real interval (so live
        stalls do not inflate speeds), otherwise 1/fps.
        detections: transport and blocking-class (cart) detections with track_id.
        persons: optional person boxes (is there a patient on the stretcher?).
        Returns (should the door open?, info for each visible track)."""
        self.frame += 1
        dt = 1.0 / max(1e-3, self.fps)
        if t is not None:
            if self._t_in is not None:
                dt = min(max(t - self._t_in, 1e-3), self.max_lost_s)
            self._t_in = t
        self.t += dt
        should_open = False
        infos = []
        persons = persons or []
        self.last_crossed = []
        self.last_lost = []
        seen = set()
        for st in self.tracks.values():          # every track advances in time, seen or not
            st.kf.predict(dt)
        for d in detections:
            seen.add(d.track_id)
            px, py = self._lead_point(d)
            st = self.tracks.get(d.track_id)
            if st is None:
                st = TrackState(kf=CVKalman(px, py, accel_std=self.accel_std))
                self.tracks[d.track_id] = st
            st.kf.update(px, py)
            st.hits += 1
            st.last_seen = self.t
            st.hist.append((self.t, d.cx, d.cy, d.area))
            if len(st.hist) > 60:
                st.hist.pop(0)
            st.p_enter = (self.intent.p_enter(st.hist, self.door_line, self.direction)
                          if self.use_intent else None)
            st.occupied = any(_overlap(d, p) > 0.3 for p in persons)
            if self.use_class_vote:
                st.cls_votes[d.cls_name.lower()] += 1
                tot = sum(st.cls_votes.values())
                if tot > self.vote_window:
                    for k in list(st.cls_votes):
                        st.cls_votes[k] *= self.vote_window / tot
                st.cls_smooth = st.cls_votes.most_common(1)[0][0]
            if self.use_depth and depths and d.track_id in depths:
                self._depth_step(st, *depths[d.track_id])

            kx, ky = st.kf.pos
            vx, vy = st.kf.vel
            ego_ix = ego_iy = 0.0
            if self.use_ego and ego is not None and getattr(ego, "ok", False):
                ix, iy = ego.induced(kx, ky)            # camera-induced shift per frame
                ego_ix, ego_iy = ix * self.fps, iy * self.fps
                vx -= ego_ix; vy -= ego_iy
                # limitation: camera shift is not accumulated while the track is unseen (small drift after short gaps)
                st.ego_off[0] += ix; st.ego_off[1] += iy
            st.motion.append((self.t, px - st.ego_off[0], py - st.ego_off[1]))
            while st.motion and st.motion[0][0] < self.t - self._win - 1e-9:
                st.motion.pop(0)
            _, v_axis = self._axis(kx, ky, vx, vy)
            tta = self.time_to_arrival(kx, ky, vx, vy)
            dist = self._distance(self._axis(kx, ky, 0, 0)[0])
            # crossing: raw measurement, same edge point
            edge_dist = self._distance(self._axis(px, py, 0, 0)[0])
            if not st.crossed and st.prev_edge_dist is not None and st.prev_edge_dist > 0 and edge_dist <= 0:
                st.crossed = True
                self.last_crossed.append(d.track_id)
            elif st.prev_edge_dist is None and edge_dist <= 0:
                st.crossed = True  # born past the line; excluded from metrics
            st.prev_edge_dist = edge_dist

            cls = self._class_of(st, d)
            # veto if this frame's raw class is non-opening (cart): no open even if a cart takes over the track id
            class_ok = cls in self.open_classes and d.cls_name.lower() in self.open_classes
            if self.use_class_vote and class_ok:
                tot = sum(st.cls_votes.values())
                k_open = sum(v for k, v in st.cls_votes.items() if k in self.open_classes)
                # no non-opening votes means no label mixing: the confidence bound would only add delay
                class_ok = k_open >= tot - 1e-9 or _wilson_lower(k_open, tot, self.vote_z) >= self.min_open_share
            status = "tracking"
            if d.area < self.min_area:
                st.approaching_frames = 0
                status = "far"
            elif st.hits < 2:
                status = "new"
            else:
                hd = self._heading(st)
                toward = hd is not None and hd[0] >= self.min_speed
                st.approaching_frames = st.approaching_frames + 1 if toward else max(0, st.approaching_frames - 1)
                stable = st.hits >= self.min_stable_hits
                ok_occ = (not self.require_occupant) or st.occupied
                depth_ok = True
                if self.use_depth and st.depth_vel is not None:
                    depth_ok = st.depth_vel >= self.depth_min_vel
                    if tta != float("inf") and st.tta_depth != float("inf"):
                        tta = 0.5 * (tta + st.tta_depth)
                    elif st.tta_depth != float("inf"):
                        tta = st.tta_depth
                will_arrive = tta <= self.door_open_time + self.lead_margin and dist <= self.max_trigger_distance
                # the at-door band ignores intent, so it needs sustained motion TOWARD the door over the
                # window: a parked track or one passing across in front of the door does not trigger it
                at_door = (dist <= self.trigger_distance and toward
                           and hd[0] >= self.band_min_speed and hd[0] > abs(hd[1]))
                if self.use_intent:
                    intent_ok = st.p_enter is not None and st.p_enter >= self.intent_threshold
                else:
                    intent_ok = True
                # a track that has crossed or was born past the line never opens: its edge is already in
                # the doorway and the measured "approach" is box growth/clipping (scene cut, frame-filling box)
                base = (toward and st.approaching_frames >= self.min_frames and stable and ok_occ and depth_ok
                        and not st.crossed)
                if not class_ok:
                    status = f"{cls}: does not open"
                elif base and ((will_arrive and intent_ok) or at_door):
                    should_open = True
                    why = "at door" if at_door and not (will_arrive and intent_ok) else f"TTA {tta:.1f}s"
                    pe = f", intent {st.p_enter:.2f}" if st.p_enter is not None else ""
                    status = f"APPROACHING -> OPEN ({why}{pe})"
                elif base and will_arrive and not intent_ok:
                    status = ("intent window filling" if st.p_enter is None
                              else f"rule says open, intent says no ({st.p_enter:.2f})")
                elif toward and st.approaching_frames >= self.min_frames and not depth_ok:
                    status = f"image says approaching, depth says no (dv {st.depth_vel:+.3f}/s)"
                elif toward and st.approaching_frames > 0:
                    status = f"approaching? (TTA {tta:.1f}s)" if tta != float("inf") else "approaching?"
                elif not ok_occ and toward:
                    status = "empty stretcher, not opening"
                else:
                    status = "receding/stopped"
            st.last_tta = tta
            infos.append({"id": d.track_id, "status": status, "opens": status.startswith("APPROACHING"),
                          "speed": v_axis,
                          "cls_smooth": cls, "class_ok": class_ok,
                          "ego": (ego_ix, ego_iy), "ego_scale": (getattr(ego, "scale", 1.0) if ego is not None else 1.0),
                          "frames": st.approaching_frames, "tta": tta,
                          "occupied": st.occupied, "p_enter": st.p_enter, "det": d,
                          "depth_vel": st.depth_vel, "tta_depth": st.tta_depth})
        for tid in list(self.tracks):
            if tid not in seen and self.t - self.tracks[tid].last_seen >= self.max_lost_s - 1e-9:
                del self.tracks[tid]
                self.last_lost.append(tid)
        return should_open, infos


def _overlap(a, b) -> float:
    """Intersection of b (person) with a (stretcher), as a fraction of b's area."""
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    return inter / b.area if b.area > 0 else 0.0


def _wilson_lower(k: float, n: float, z: float) -> float:
    """Wilson lower confidence bound of a proportion (votes may be fractional)."""
    if n <= 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
