# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Constant-velocity 2D Kalman filter (x, y, vx, vy); pure numpy, time base in SECONDS.

Position in normalised units (0..1), velocity in units/second. Process noise follows the continuous
white-noise acceleration model (CWNA): the same physical motion gets the same uncertainty at any frame rate.
Defaults were tuned at 30 fps.
"""
from __future__ import annotations

import numpy as np


class CVKalman:
    def __init__(self, x: float, y: float, accel_std: float = 28.5,
                 meas_std: float = 0.02, init_vel_std: float = 9.5):
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.diag([meas_std ** 2 * 25, meas_std ** 2 * 25, init_vel_std ** 2, init_vel_std ** 2])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2) * meas_std ** 2
        self.q = accel_std ** 2
        self.age = 0

    def predict(self, dt: float):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = self.q
        a, b, c = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
        Q = q * np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x

    def update(self, zx: float, zy: float):
        z = np.array([zx, zy])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.age += 1
        return self.x

    @property
    def pos(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self):
        """units/second"""
        return float(self.x[2]), float(self.x[3])
