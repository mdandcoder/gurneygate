# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Relative monocular depth (Depth Anything V2 Small). Median box depth -> per-track depth velocity.
Output is relative (inverse depth, larger = closer), not metres; use it for ratios and rates of change.
"""
from __future__ import annotations
import numpy as np, torch

class DepthEstimator:
    def __init__(self, model="depth-anything/Depth-Anything-V2-Small-hf", size=518, device=None):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModelForDepthEstimation.from_pretrained(model).to(self.device).eval()
        self.size = size

    @torch.no_grad()
    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """bgr (H,W,3) -> relative inverse-depth map (H,W), float32, larger = closer."""
        rgb = bgr[:, :, ::-1]
        inp = self.proc(images=rgb, return_tensors="pt", size={"height": self.size, "width": self.size}).to(self.device)
        out = self.model(**inp).predicted_depth  # (1,h,w)
        d = torch.nn.functional.interpolate(out[None], size=bgr.shape[:2], mode="bilinear", align_corners=False)[0, 0]
        return d.float().cpu().numpy()

def box_depth(dmap: np.ndarray, xyxy, shrink=0.25) -> float:
    """Median relative depth of the box interior (edges trimmed)."""
    x1, y1, x2, y2 = [int(v) for v in xyxy]; w, h = x2 - x1, y2 - y1
    x1 += int(w * shrink); x2 -= int(w * shrink); y1 += int(h * shrink); y2 -= int(h * shrink)
    if x2 <= x1 or y2 <= y1: return float("nan")
    return float(np.median(dmap[y1:y2, x1:x2]))
