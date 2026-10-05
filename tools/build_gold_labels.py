# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Turn review decisions into YOLO labels (0 stretcher, 1 wheelchair, 2 bed, 3 cart, 9 ignore region).

Input: a review folder made by tools/propose_sheets.py (index.json) plus a decisions.txt written by the
reviewer, one line per tile:
    <sheet> <tile> <letters> [M=<cls>:x1,y1,x2,y2 ...] [I=x1,y1,x2,y2 ...]
<letters> has one letter per proposed box, in order: s stretcher, w wheelchair, b bed, c cart, r reject.
"-" marks a verified empty frame (empty label file); "X" excludes the frame (an object that cannot be
boxed unambiguously), so it gets no label file and is left out of every measurement.
M= adds a hand-drawn box, I= adds an ignore region (normalised x1,y1,x2,y2). Lines starting with # are comments.

python tools/build_gold_labels.py <review_dir> <out_labels_dir>
"""
import json, os, sys
from collections import Counter

CLS = {"s": 0, "w": 1, "b": 2, "c": 3}


def yolo(c, b):
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in b)
    return f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def main():
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    R, out = sys.argv[1], sys.argv[2]
    ix = json.load(open(f"{R}/index.json"))
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f"{out} is not empty; choose a new folder so existing labels are not mixed or lost")
    os.makedirs(out, exist_ok=True)
    n = Counter()
    for line in open(f"{R}/decisions.txt"):
        if line.startswith("#") or not line.strip():
            continue
        s, t, lab, *extra = line.split()
        e = ix[s][t]
        if lab == "X":
            n["excluded"] += 1; continue
        rows = []
        if lab != "-":
            assert len(lab) == len(e["boxes"]), (s, t)
            rows += [yolo(CLS[ch], b) for ch, b in zip(lab, e["boxes"]) if ch != "r"]
        for x in extra:
            k, v = x.split("=", 1)
            if k == "M":
                c, box = v.split(":"); rows.append(yolo(CLS[c], map(float, box.split(","))))
            elif k == "I":
                rows.append(yolo(9, map(float, v.split(","))))
        open(f"{out}/{os.path.splitext(e['img'])[0]}.txt", "w").write("\n".join(rows) + ("\n" if rows else ""))
        n["frames"] += 1
        for r in rows:
            n[{0: "stretcher", 1: "wheelchair", 2: "bed", 3: "cart", 9: "ignore"}[int(r.split()[0])]] += 1
    print(dict(n))


if __name__ == "__main__":
    main()
