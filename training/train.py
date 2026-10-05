# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Train the GurneyGate detector.

Every source is a folder in YOLO layout (images/ and labels/, same file stems) labelled with the
4-class schema of docs/LABEL_SCHEMA.md: 0 stretcher, 1 wheelchair, 2 bed, 3 cart. Hard negatives are
images with an empty label file. Labels are merged to 2 classes for training: transport (0, 1, 2) and
cart (3).

A source can be weighted by appending ":N": every image of that source that contains a stretcher
(class 0) is used N times. Stretchers are the rarest and most important class.

The released weights were trained with:
    python training/train.py \\
        --train data/openimages:2 data/pexels data/pexels_more data/youtube data/gdino:3 data/hard_negatives \\
        --val data/youtube_val --out runs/gurneygate
Validation (used only to monitor training) is a video-level split of the YouTube frames; no benchmark
or test image is used.
"""
import argparse
import os
import shutil
from pathlib import Path

NAMES = ["transport", "cart"]
REMAP = {0: 0, 1: 0, 2: 0, 3: 1}            # stretcher, wheelchair, bed -> transport; cart -> cart
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Ultralytics training settings of the released weights
HYP = dict(epochs=100, imgsz=640, batch=32, patience=100,   # no early stopping; last weights are used
           degrees=10, scale=0.5, fliplr=0.5, mosaic=1.0, mixup=0.1, hsv_v=0.5, close_mosaic=10)


def remap(label_text):
    """4-class YOLO label text -> 2-class label text (rows with other classes are dropped)."""
    out = []
    for line in label_text.splitlines():
        f = line.split()
        if f and int(f[0]) in REMAP:
            out.append(" ".join([str(REMAP[int(f[0])])] + f[1:]))
    return "\n".join(out) + ("\n" if out else "")


def has_stretcher(label_text):
    return any(l.split()[:1] == ["0"] for l in label_text.splitlines())


def build(sources, dst, weighted=True):
    """Link images and write 2-class labels into dst/images, dst/labels. Returns the image count."""
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for spec in sources:
        path, _, w = spec.partition(":")
        src, weight = Path(path), (int(w or 1) if weighted else 1)
        for ip in sorted((src / "images").iterdir()):
            if ip.suffix.lower() not in IMG_EXT:
                continue
            lp = src / "labels" / (ip.stem + ".txt")
            if not lp.exists():                  # unlabelled image: skip, it is not a verified negative
                continue
            text = lp.read_text()
            copies = weight if has_stretcher(text) else 1
            for k in range(copies):
                stem = f"{src.name}__{ip.stem}" + (f"__{k}" if k else "")
                link = dst / "images" / (stem + ip.suffix)
                if not link.exists():
                    os.symlink(ip.resolve(), link)
                (dst / "labels" / (stem + ".txt")).write_text(remap(text))
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="source folders, optionally folder:N")
    ap.add_argument("--val", nargs="+", required=True, help="validation folders (monitoring only)")
    ap.add_argument("--out", default="runs/gurneygate")
    ap.add_argument("--model", default="yolo26s.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=HYP["epochs"])
    ap.add_argument("--batch", type=int, default=HYP["batch"])
    ap.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    ap.add_argument("--export", action="store_true", help="also export ONNX and NCNN")
    a = ap.parse_args()

    out = Path(a.out)
    work = out / "data"
    shutil.rmtree(work, ignore_errors=True)
    n_train = build(a.train, work / "train")
    n_val = build(a.val, work / "val", weighted=False)
    print(f"[data] {n_train} training images, {n_val} validation images")
    (work / "data.yaml").write_text(
        f"path: {work.resolve()}\ntrain: train/images\nval: val/images\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))

    from ultralytics import YOLO
    hyp = {**HYP, "epochs": a.epochs, "batch": a.batch}
    r = YOLO(a.model).train(data=str(work / "data.yaml"), project=str(out.resolve()), name="train",
                            exist_ok=True, device=a.device, **hyp)
    final = out / "gurneygate-yolo26s.pt"
    shutil.copy(Path(r.save_dir) / "weights" / "last.pt", final)
    print(f"[done] {final}")
    if a.export:
        m = YOLO(str(final))
        m.export(format="onnx", imgsz=hyp["imgsz"])
        m.export(format="ncnn", imgsz=hyp["imgsz"])


if __name__ == "__main__":
    main()
