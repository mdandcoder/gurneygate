# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Box proposals INDEPENDENT of our model: Grounding DINO (open vocabulary). Proposals never enter
training without human/visual review; the goal is to box the stretcher types earlier detector versions missed.

python tools/propose_gdino.py datasets/oi_unboxed out.json
"""
import json, sys, glob, os
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL = "IDEA-Research/grounding-dino-base"
PROMPT = "stretcher. gurney. ambulance cot. hospital bed. wheelchair. cart."


def main():
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    src, out = sys.argv[1], sys.argv[2]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to(dev).eval()
    res = {}
    for i, p in enumerate(sorted(glob.glob(os.path.join(src, "*.jpg")))):
        bgr = cv2.imread(p)                    # read with OpenCV: some of these JPEGs fail to decode in certain Pillow builds
        if bgr is None:
            print("[corrupt]", os.path.basename(p)); res[os.path.basename(p)] = None; continue
        im = Image.fromarray(bgr[:, :, ::-1].copy())
        inp = proc(images=im, text=PROMPT, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**inp)
        r = proc.post_process_grounded_object_detection(o, inp.input_ids, box_threshold=0.25, text_threshold=0.2,
                                                        target_sizes=[im.size[::-1]])[0]
        W, H = im.size
        res[os.path.basename(p)] = [dict(box=[b[0] / W, b[1] / H, b[2] / W, b[3] / H], score=float(s), phrase=l)
                                    for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"])]
        if i % 25 == 0:
            print(i, os.path.basename(p), len(res[os.path.basename(p)]), flush=True)
    json.dump(res, open(out, "w"), indent=0)


if __name__ == "__main__":
    main()
