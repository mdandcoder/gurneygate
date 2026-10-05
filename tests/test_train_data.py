# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
from train import build, remap


def test_remap_merges_transports_and_keeps_carts():
    assert remap("0 .5 .5 .1 .1\n1 .5 .5 .1 .1\n2 .5 .5 .1 .1\n3 .5 .5 .1 .1\n9 .5 .5 .1 .1\n") == \
        "0 .5 .5 .1 .1\n0 .5 .5 .1 .1\n0 .5 .5 .1 .1\n1 .5 .5 .1 .1\n"
    assert remap("") == ""                       # hard negative stays an empty label


def test_build_weights_only_stretcher_images(tmp_path):
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True); (src / "labels").mkdir()
    for stem, label in (("stretcher", "0 .5 .5 .2 .2\n"), ("cart", "3 .5 .5 .2 .2\n"), ("empty", ""), ("unlabelled", None)):
        (src / "images" / f"{stem}.jpg").write_bytes(b"x")
        if label is not None:
            (src / "labels" / f"{stem}.txt").write_text(label)
    n = build([f"{src}:3"], tmp_path / "out")
    assert n == 3 + 1 + 1                        # stretcher x3, cart x1, hard negative x1, unlabelled skipped
    labels = sorted(p.name for p in (tmp_path / "out" / "labels").iterdir())
    assert labels == ["src__cart.txt", "src__empty.txt", "src__stretcher.txt", "src__stretcher__1.txt", "src__stretcher__2.txt"]
    assert (tmp_path / "out" / "labels" / "src__cart.txt").read_text() == "1 .5 .5 .2 .2\n"
    assert build([f"{src}:3"], tmp_path / "val", weighted=False) == 3
