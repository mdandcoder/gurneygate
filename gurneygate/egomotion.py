# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Camera ego-motion estimation and compensation.

Problem: if the camera moves toward an object, the object grows and shifts down in the image;
the system reads that as "object approaching the door". This cannot happen on a fixed door camera,
but handheld footage, vibration or a moving mount produce false triggers.

Solution: estimate the camera's own motion from background point flow
(similarity transform: translation + rotation + scale) and subtract it from the apparent object motion.
What remains is the object's real motion.
"""
from __future__ import annotations

import cv2
import numpy as np


class EgoMotion:
    def __init__(self, max_corners: int = 300, quality: float = 0.01, min_dist: int = 12,
                 min_points: int = 20, scale_width: int = 480):
        self.prev_gray: np.ndarray | None = None
        self.max_corners = max_corners
        self.quality = quality
        self.min_dist = min_dist
        self.min_points = min_points
        self.scale_width = scale_width
        self.A: np.ndarray | None = None      # 2x3 similarity transform (previous -> current)
        self.ok = False
        self.n_pts = 0

    def _prep(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        s = self.scale_width / w if w > self.scale_width else 1.0
        small = cv2.resize(frame, (int(w * s), int(h * s))) if s != 1.0 else frame
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), s

    def update(self, frame: np.ndarray, boxes_norm: list[tuple[float, float, float, float]] | None = None):
        """Called every frame. boxes_norm: object boxes (0..1); these regions are excluded from the background."""
        gray, s = self._prep(frame)
        h, w = gray.shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        for (x1, y1, x2, y2) in (boxes_norm or []):
            a, b = int(max(0, x1) * w), int(max(0, y1) * h)
            c, d = int(min(1, x2) * w), int(min(1, y2) * h)
            if c > a and d > b:
                mask[b:d, a:c] = 0
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray, self.A, self.ok, self.n_pts = gray, None, False, 0
            return
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, self.max_corners, self.quality, self.min_dist, mask=mask)
        if p0 is None or len(p0) < self.min_points:
            self.prev_gray, self.A, self.ok, self.n_pts = gray, None, False, 0
            return
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                             winSize=(21, 21), maxLevel=3)
        if p1 is None:
            self.prev_gray, self.A, self.ok, self.n_pts = gray, None, False, 0
            return
        st = st.reshape(-1).astype(bool)
        a0, a1 = p0.reshape(-1, 2)[st], p1.reshape(-1, 2)[st]
        self.n_pts = len(a0)
        if self.n_pts < self.min_points:
            self.prev_gray, self.A, self.ok = gray, None, False
            return
        A, inl = cv2.estimateAffinePartial2D(a0, a1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        self.prev_gray = gray
        if A is None or inl is None or int(inl.sum()) < self.min_points // 2:
            self.A, self.ok = None, False
            return
        # convert to normalised coordinates: scale/rotation unchanged, translation pixels -> 0..1
        self.A = A.copy()
        self.A[0, 2] /= w
        self.A[1, 2] /= h
        self.ok = True

    def induced(self, cx: float, cy: float) -> tuple[float, float]:
        """Apparent shift caused by the camera at normalised point (cx, cy)."""
        if not self.ok or self.A is None:
            return 0.0, 0.0
        a, b, tx = self.A[0]
        c, d, ty = self.A[1]
        return (a * cx + b * cy + tx) - cx, (c * cx + d * cy + ty) - cy

    @property
    def scale(self) -> float:
        """>1: camera moving forward (scene grows). 1: pure translation or static."""
        if not self.ok or self.A is None:
            return 1.0
        return float(np.hypot(self.A[0, 0], self.A[1, 0]))

    @property
    def moving(self) -> bool:
        """Is the camera clearly moving (translation or forward/backward)?"""
        if not self.ok or self.A is None:
            return False
        return abs(self.scale - 1.0) > 0.004 or np.hypot(self.A[0, 2], self.A[1, 2]) > 0.004
