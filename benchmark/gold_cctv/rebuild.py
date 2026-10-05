# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Rebuild the benchmark frames from the public YouTube videos (frames are not redistributed).

pip install yt-dlp
python benchmark/gold_cctv/rebuild.py          # writes images/

The labels were made on specific YouTube streams (yt_format in manifest.csv); the script downloads
exactly those, video only, and checks every decoded frame against the SHA-256 in manifest.csv.
Downloading is subject to YouTube's Terms of Service.
"""
import csv, glob, hashlib, os, shutil, subprocess, sys
import cv2

here = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(here, "manifest.csv"))))
os.makedirs(os.path.join(here, "images"), exist_ok=True)
tmp = os.path.join(here, "_videos"); os.makedirs(tmp, exist_ok=True)
videos = {r["youtube_id"]: r["yt_format"] for r in rows}


def local(vid):
    return next(iter(glob.glob(os.path.join(tmp, vid + ".*"))), None)


if any(local(v) is None for v in videos) and shutil.which("yt-dlp") is None:
    sys.exit("yt-dlp is needed to download the videos: pip install yt-dlp")

missing = changed = 0
for vid, fmt in sorted(videos.items()):
    if local(vid) is None:
        subprocess.run(["yt-dlp", "-q", "-f", fmt, "-o", os.path.join(tmp, "%(id)s.%(ext)s"), "--", vid], check=False)
    cap = cv2.VideoCapture(local(vid) or "")
    for r in (r for r in rows if r["youtube_id"] == vid):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(r["t_sec"]) * 1000)
        ok, im = cap.read()
        if not ok:
            missing += 1
            continue
        changed += hashlib.sha256(im.tobytes()).hexdigest() != r["frame_sha256"]
        cv2.imwrite(os.path.join(here, "images", r["image"]), im, [cv2.IMWRITE_JPEG_QUALITY, 92])  # as labelled
n = len(rows)
print(f"{n - missing}/{n} frames written, {n - missing - changed} identical to the labelled frames")
if missing:
    print(f"{missing} frames missing: a video could not be downloaded or decoded")
if changed:
    print(f"{changed} frames differ from the labelled ones (different stream or decoder); results may differ slightly")
sys.exit(1 if missing else 0)
