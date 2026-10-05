# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Draws Grounding DINO proposals as numbered boxes on 8-frame review sheets.
Boxes that land on the same object under different phrases are deduplicated (IoU > 0.7, highest score kept).
Output: <out>/sNN.jpg and <out>/index.json  (sheet -> tag -> {img, boxes:[[x1,y1,x2,y2],...]})

python tools/propose_sheets.py datasets/oi_unboxed datasets/oi_unboxed_gdino.json datasets/oi_unboxed_review [min_score] [--all]
"""
import json, os, sys
import cv2, numpy as np


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def dedupe(props, min_score):
    keep = []
    for p in sorted((p for p in props if p["score"] >= min_score), key=lambda p: -p["score"]):
        if all(iou(p["box"], k["box"]) <= 0.7 for k in keep):
            keep.append(p)
    return keep


def main():
    if len(sys.argv) < 4 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    src, pj, out = sys.argv[1:4]
    min_score = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    keep_empty = "--all" in sys.argv          # also show frames without proposals (test sets: check for missed objects)
    props = json.load(open(pj)); os.makedirs(out, exist_ok=True)
    items = [(k, dedupe(v or [], min_score)) for k, v in sorted(props.items()) if v is not None]
    items = [(k, v) for k, v in items if v or keep_empty]
    index = {}
    for s in range(0, len(items), 8):
        tiles, sheet = [], f"s{s // 8:02d}"
        index[sheet] = {}
        for j, (name, boxes) in enumerate(items[s:s + 8]):
            tag = "ABCDEFGH"[j]
            im = cv2.imread(os.path.join(src, name)); H, W = im.shape[:2]
            sc = 480 / max(W, H); im = cv2.resize(im, (int(W * sc), int(H * sc)))
            h, w = im.shape[:2]
            for i, p in enumerate(boxes):
                x1, y1, x2, y2 = (int(p["box"][0] * w), int(p["box"][1] * h), int(p["box"][2] * w), int(p["box"][3] * h))
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(im, (x1, y1), (x1 + 22, y1 + 20), (0, 0, 0), -1)
                cv2.putText(im, str(i), (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(im, (0, 0), (30, 26), (255, 255, 255), -1)
            cv2.putText(im, tag, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            tile = np.full((480, 480, 3), 30, np.uint8); tile[:h, :w] = im; tiles.append(tile)
            index[sheet][tag] = {"img": name, "boxes": [p["box"] for p in boxes],
                                 "phrases": [f'{p["phrase"]} {p["score"]:.2f}' for p in boxes]}
        while len(tiles) < 8:
            tiles.append(np.full((480, 480, 3), 30, np.uint8))
        cv2.imwrite(os.path.join(out, sheet + ".jpg"), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    print(len(items), "images,", sum(len(v) for _, v in items), "boxes,", len(index), "sheets")


if __name__ == "__main__":
    main()
