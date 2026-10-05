# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Learned intent classifier: last W frames of a track -> {entering, passing, stopped/receding}.

A door-specific reduction of pedestrian intent prediction (trajectory-based crossing/not-crossing
classification). A small MLP trained on synthetic trajectories (see train()); retraining on field recordings is
future work.
Used together with the rule-based decision (two cues).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

W = 10          # track window (frames)
REF_FPS = 30.0  # implicit frame rate of the synthetic training tracks; other rates are resampled to it
N_FEAT = W * 3 + 4
LABELS = ["enter", "pass", "other"]


def features(hist, door_line: float, direction: str) -> np.ndarray | None:
    """hist: [(t, cx, cy, area), ...] timestamped track (t in seconds), normalised coordinates.
    The track is resampled to REF_FPS (last W/REF_FPS seconds, linear interpolation), so at any frame
    rate the model sees the 10-sample, 30 fps window it was trained on.
    At 30 fps the samples already sit on those points, so no interpolation happens."""
    if len(hist) < 3:
        return None
    h = np.array(hist[-60:], dtype=np.float64)
    t, x, y, a = h[:, 0], h[:, 1], h[:, 2], h[:, 3]
    span = (W - 1) / REF_FPS
    if t[-1] - t[0] < span - 1e-6:
        return None
    T = t[-1] - np.arange(W - 1, -1, -1) / REF_FPS
    x, y, a = np.interp(T, t, x), np.interp(T, t, y), np.interp(T, t, a)
    # rotate to the door axis: the door is always "down" (+y)
    if direction == "up":
        y = 1 - y
    elif direction == "right":
        x, y = y, x
    elif direction == "left":
        x, y = y, 1 - x
    door = door_line if direction in ("down", "right") else 1 - door_line
    dx, dy = np.diff(x), np.diff(y)
    da = np.diff(a) / (a[:-1] + 1e-6)
    dist = door - y[-1]
    vy = dy[-3:].mean()
    vx = dx[-3:].mean()
    heading = math.atan2(vy, vx + 1e-9)          # +pi/2 = straight at the door
    tta = dist / vy if vy > 1e-4 else 10.0
    f = np.concatenate([dx, dy, da, [dist, vy, math.sin(heading), min(tta, 10.0) / 10.0]])
    # pad: dx,dy,da have length W-1 -> total 3(W-1)+4 = 3W+1; pad up to N_FEAT
    return np.pad(f, (0, N_FEAT - len(f))).astype(np.float32)


class IntentModel:
    def __init__(self, path: str | None = "weights/intent.npz"):
        self.params = None
        if path and Path(path).exists():
            z = np.load(path)
            self.params = [z[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
            self.mu, self.sd = z["mu"], z["sd"]

    @property
    def ready(self):
        return self.params is not None

    def predict_proba(self, f: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2, W3, b3 = self.params
        x = (f - self.mu) / self.sd
        h = np.maximum(0, x @ W1 + b1)
        h = np.maximum(0, h @ W2 + b2)
        z = h @ W3 + b3
        e = np.exp(z - z.max())
        return e / e.sum()

    def p_enter(self, hist, door_line, direction) -> float | None:
        f = features(hist, door_line, direction)
        if f is None or not self.ready:
            return None
        return float(self.predict_proba(f)[0])


# --------------------------- training ---------------------------
def synth_trajectories(n: int, rng: np.random.Generator):
    """Synthetic tracks: enter / pass / other. The door is at the bottom (y=0.85)."""
    X, Y = [], []
    door = 0.85
    for _ in range(n):
        kind = rng.integers(0, 3)
        L = W + rng.integers(0, 10)
        noise = rng.uniform(0.002, 0.012)
        x0 = rng.uniform(0.15, 0.85)
        if kind == 0:   # enter: toward the door, slight curve, speed change
            speed = rng.uniform(0.006, 0.04)
            y0 = rng.uniform(0.05, door - 0.15)
            drift = rng.uniform(-0.01, 0.01)
            acc = rng.uniform(-0.0008, 0.0008)
            ys = y0 + np.cumsum(np.maximum(0.002, speed + acc * np.arange(L)))
            xs = x0 + drift * np.arange(L)
        elif kind == 1:  # pass: horizontal pass along the corridor, slight vertical drift
            speed = rng.uniform(0.01, 0.04) * rng.choice([-1, 1])
            y0 = rng.uniform(0.2, 0.7)
            xs = x0 + speed * np.arange(L)
            ys = y0 + rng.uniform(-0.004, 0.004) * np.arange(L)
        else:            # other: stopped, receding, or approached then stopped
            sub = rng.integers(0, 3)
            if sub == 0:
                xs = np.full(L, x0); ys = np.full(L, rng.uniform(0.2, 0.8))
            elif sub == 1:
                speed = rng.uniform(0.006, 0.04)
                ys = rng.uniform(0.3, 0.9) - speed * np.arange(L); xs = np.full(L, x0)
            else:
                speed = rng.uniform(0.01, 0.03); stop = L // 2
                ys = np.concatenate([np.arange(stop) * speed, np.full(L - stop, (stop - 1) * speed)]) + rng.uniform(0.05, 0.3)
                xs = np.full(L, x0)
        ys = ys + rng.normal(0, noise, L); xs = xs + rng.normal(0, noise, L)
        base_a = rng.uniform(0.03, 0.15)
        areas = base_a * (1 + 0.8 * np.clip(ys - ys[0], -1, 1)) + rng.normal(0, 0.003, L)  # grows as it approaches
        hist = list(zip(np.arange(L) / REF_FPS, xs, ys, np.abs(areas)))
        f = features(hist, door, "down")
        X.append(f); Y.append(kind)
    return np.stack(X), np.array(Y)


def train(out="weights/intent.npz", n=40000, epochs=60, seed=0):
    import torch, torch.nn as nn

    rng = np.random.default_rng(seed)
    X, Y = synth_trajectories(n, rng)
    Xv, Yv = synth_trajectories(5000, np.random.default_rng(seed + 1))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd); Yt = torch.tensor(Y)
    Xvt = torch.tensor((Xv - mu) / sd); Yvt = torch.tensor(Yv)
    net = nn.Sequential(nn.Linear(N_FEAT, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            loss = nn.functional.cross_entropy(net(Xt[idx]), Yt[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        acc = (net(Xvt).argmax(1) == Yvt).float().mean().item()
        pred = net(Xvt).argmax(1)
        fp = ((pred == 0) & (Yvt != 0)).float().sum().item() / max(1, (Yvt != 0).sum().item())
        miss = ((pred != 0) & (Yvt == 0)).float().sum().item() / max(1, (Yvt == 0).sum().item())
    Path(out).parent.mkdir(exist_ok=True)
    p = [t.detach().numpy().T if t.ndim == 2 else t.detach().numpy() for t in net.parameters()]
    np.savez(out, W1=p[0], b1=p[1], W2=p[2], b2=p[3], W3=p[4], b3=p[5], mu=mu, sd=sd)
    print(f"[intent] val acc={acc:.3f}  false_open={fp:.3f}  miss={miss:.3f}  -> {out}")
    return acc


if __name__ == "__main__":
    train()
