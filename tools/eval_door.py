# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Door-oriented model evaluation: measures models with different class schemas on the same ruler.

On a set with human-checked boxes (4 classes, docs/LABEL_SCHEMA.md) each GT box asks:
"did a door-opening prediction match this box with IoU >= 0.5?" This compares 4-class models
(stretcher/wheelchair/bed separate) and 2-class models (transport) fairly.

Reported (at a fixed operating threshold, default conf 0.25):
- recall per GT class (stretcher, wheelchair, bed) + 95% Wilson interval
- rate at which cart GTs are matched by a door-opening prediction (should be low)
- unmatched door-opening predictions per image (false positives)
- paired McNemar test on stretcher recall between model pairs

python tools/eval_door.py --models gurneygate=weights/gurneygate-yolo26s.pt   # hospital CCTV benchmark
python tools/eval_door.py --data path/to/set --split val --models a=a.pt b=b.pt   # compare two models
"""
import argparse, glob, math, os
from collections import defaultdict

import numpy as np

GT_NAMES = ["stretcher", "wheelchair", "bed", "cart"]
IGNORE = 9   # label class id for an ignore region: not a GT, and predictions inside it are not false positives
OPEN = {"transport", "stretcher", "gurney", "bed", "hospital_bed", "wheelchair"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar(a_hits, b_hits):
    """Paired binary outcomes (same GT boxes). Exact two-sided binomial p-value."""
    b = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    c = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * p)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def load_gt(lbl_dir):
    gt = {}
    for f in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
        rows = []
        for l in open(f):
            p = l.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            rows.append(("ignore" if c == IGNORE else GT_NAMES[c], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        gt[os.path.splitext(os.path.basename(f))[0]] = rows
    return gt


def _ioa(p, g):
    """Share of prediction p's area inside region g."""
    ix = max(0.0, min(p[2], g[2]) - max(p[0], g[0])); iy = max(0.0, min(p[3], g[3]) - max(p[1], g[1]))
    a = (p[2] - p[0]) * (p[3] - p[1])
    return ix * iy / a if a > 0 else 0.0


def match(rows, preds, min_iou=0.5):
    """Global greedy matching: all (GT, door-opening prediction) pairs sorted by IoU, largest first,
    so file order cannot let one GT steal another's only match. Returns (matched GTs, used predictions).
    min_iou 0.5 is the usual box criterion; a lower value asks only "was the object seen", which is
    what the door needs (it tracks the object, the exact box extent does not matter)."""
    pairs = sorted(((iou(gb, pb), gi, pj) for gi, (gname, gb) in enumerate(rows)
                    for pj, (pb, is_open) in enumerate(preds) if is_open), reverse=True)
    used, got = set(), set()
    for v, gi, pj in pairs:
        if v < min_iou:
            break
        if gi in got or pj in used:
            continue
        got.add(gi)
        if rows[gi][0] != "cart":
            used.add(pj)
    return got, used


def evaluate(model_path, img_dir, gt, conf, imgsz, min_iou=0.5):
    from ultralytics import YOLO
    m = YOLO(model_path)
    names = {i: n.lower() for i, n in m.names.items()}
    hits = {}                       # (stem, gt_idx) -> bool, matched by a door-opening prediction
    fp = 0
    for stem, rows in gt.items():
        ims = glob.glob(os.path.join(img_dir, stem + ".*"))
        if not ims:
            continue
        r = m.predict(ims[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        H, W = r.orig_shape
        preds = [((x1 / W, y1 / H, x2 / W, y2 / H), names[int(c)] in OPEN)
                 for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())]
        ign = [gb for gname, gb in rows if gname == "ignore"]
        keep = [gi for gi, r in enumerate(rows) if r[0] != "ignore"]   # original indices of real GTs
        rows = [rows[gi] for gi in keep]
        got, used = match(rows, preds, min_iou)
        for j, gi in enumerate(keep):
            hits[(stem, gi)] = j in got
        # false positive: a door-opening prediction matching no door-class GT
        for pj, (pb, is_open) in enumerate(preds):
            if is_open and pj not in used:
                if not any(gname != "cart" and iou(gb, pb) >= min_iou for gname, gb in rows) \
                        and not any(_ioa(pb, g) >= 0.5 for g in ign):
                    fp += 1
    return hits, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark/gold_cctv")
    ap.add_argument("--split", default="")
    ap.add_argument("--models", nargs="+", required=True, help="tag=weights.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-iou", type=float, default=0.5,
                    help="0.5 = box must fit; 0.3 = object was seen (what the tracker needs)")
    a = ap.parse_args()
    gt = load_gt(os.path.join(a.data, "labels", a.split))
    img_dir = os.path.join(a.data, "images", a.split)
    missing = [s for s in gt if not glob.glob(os.path.join(img_dir, s + ".*"))]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(gt)} labelled frames have no image in {img_dir}; "
                         "for the benchmark run benchmark/gold_cctv/rebuild.py first")
    n_img = len(gt)
    counts = defaultdict(int)
    for rows in gt.values():
        for gname, _ in rows:
            if gname != "ignore":
                counts[gname] += 1
    print(f"GT: {n_img} images, " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f"  (conf {a.conf}, IoU >= {a.min_iou})\n")
    res = {}
    for spec in a.models:
        tag, path = spec.split("=", 1)
        res[tag] = evaluate(path, img_dir, gt, a.conf, a.imgsz, a.min_iou)
    keys_by_cls = defaultdict(list)
    for stem, rows in gt.items():
        for gi, (gname, _) in enumerate(rows):
            keys_by_cls[gname].append((stem, gi))
    hdr = f"{'model':12s}" + "".join(f"{c + ' recall':>26s}" for c in ("stretcher", "wheelchair", "bed")) \
          + f"{'cart->opens':>16s}{'false boxes':>14s}"
    print(hdr)
    for tag, (hits, fp) in res.items():
        line = f"{tag:12s}"
        for c in ("stretcher", "wheelchair", "bed"):
            ks = keys_by_cls[c]
            k = sum(hits.get(x, False) for x in ks)
            p, lo, hi = wilson(k, len(ks))
            line += f"{k:>5d}/{len(ks):<4d} {p:.2f} [{lo:.2f}-{hi:.2f}]" if ks else f"{'-':>27s}"
        ks = keys_by_cls["cart"]
        k = sum(hits.get(x, False) for x in ks)
        p, lo, hi = wilson(k, len(ks))
        line += f"  {p:.2f} [{lo:.2f}-{hi:.2f}]" if ks else f"{'-':>18s}"
        line += f"{fp:>8d} ({fp / max(1, n_img):.2f}/img)"
        print(line)
    tags = list(res)
    if len(tags) > 1:
        print("\nPaired McNemar on stretcher recall (b = left caught, right missed; c = the reverse):")
        ks = keys_by_cls["stretcher"]
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                A = [res[tags[i]][0].get(x, False) for x in ks]
                B = [res[tags[j]][0].get(x, False) for x in ks]
                b, c, p = mcnemar(A, B)
                print(f"  {tags[i]} vs {tags[j]}: b={b} c={c} p={p:.3f}")


if __name__ == "__main__":
    main()
