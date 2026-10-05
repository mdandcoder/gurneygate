# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""YOLO-based stretcher detection and tracking (Ultralytics)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    track_id: int
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


class StretcherDetector:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        weights = cfg["weights"]
        if not os.path.exists(weights):
            if not cfg.get("allow_fallback", False):
                raise SystemExit(f"[detector] weights not found: {weights}. Not silently falling back to a generic "
                                 f"COCO model; set model.allow_fallback: true if you really want that.")
            print(f"[detector] WARNING: {weights} not found, using {cfg['fallback_weights']} "
                  f"(COCO 'bed' counts as a stretcher; no accuracy can be claimed)")
            weights = cfg["fallback_weights"]
        self.model = YOLO(weights)
        self.device = pick_device(cfg.get("device", "auto"))
        # low confidence threshold for the tracker: BoT-SORT's second stage keeps tracks alive on weak detections.
        # The new-track threshold lives in the tracker file (new_track_thresh).
        self.track_conf = float(cfg.get("track_conf", 0.10))
        self.iou = float(cfg.get("iou", 0.5))
        self.imgsz = int(cfg.get("imgsz", 640))
        # A box covering almost the whole frame is degenerate: on a door camera a stretcher
        # cannot fill the frame like that. Measured: clinic false alarm 96% of frame, real stretcher 7%.
        self.max_box_area = float(cfg.get("max_box_area", 0.85))
        tr = cfg.get("tracker", "botsort_door.yaml")
        local = os.path.join(os.path.dirname(__file__), "trackers", tr)
        self.tracker = local if os.path.exists(local) else tr
        names = self.model.names  # {id: name}
        open_w = {c.lower() for c in cfg["stretcher_classes"]}
        block_w = {c.lower() for c in cfg.get("block_classes", ["cart"])}
        self.class_ids = [i for i, n in names.items() if n.lower() in open_w]
        # carts etc. are tracked too: "does not open" is decided per track, on the voted class
        self.block_ids = [i for i, n in names.items() if n.lower() in block_w]
        self.person_ids = [i for i, n in names.items() if n.lower() == "person"]
        if not self.class_ids:
            raise RuntimeError(f"Model lacks these classes: {open_w}. Model classes: {names}")
        self.names = names
        print(f"[detector] model={weights} device={self.device} "
              f"opening={[names[i] for i in self.class_ids]} non_opening={[names[i] for i in self.block_ids]} "
              f"tracker={os.path.basename(self.tracker)}")

    def track(self, frame: np.ndarray) -> tuple[list[Detection], list[Detection]]:
        """Detection + BoT-SORT tracking on a frame; returns normalised boxes (transport+blocking classes, persons)."""
        h, w = frame.shape[:2]
        cfg_tracker = self.tracker
        results = self.model.track(
            frame,
            persist=True,
            classes=self.class_ids + self.block_ids + self.person_ids,
            conf=self.track_conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            tracker=cfg_tracker,
        )
        out: list[Detection] = []
        persons: list[Detection] = []
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return out, persons
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), tid, cf, c in zip(xyxy, ids, confs, clss):
            if self.max_box_area < 1.0 and int(c) not in self.person_ids:
                if (x2 - x1) * (y2 - y1) / (w * h) > self.max_box_area:
                    continue          # degenerate box, drop it
            det = Detection(int(tid), self.names[int(c)], float(cf), x1 / w, y1 / h, x2 / w, y2 / h)
            (persons if int(c) in self.person_ids else out).append(det)
        return out, persons
