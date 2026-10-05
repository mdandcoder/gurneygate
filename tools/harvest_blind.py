# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Model-INDEPENDENT video frame harvest: search -> download -> one frame every N seconds (without
looking at any detection) -> per-clip dHash dedup. Boxes come later from Grounding DINO + human review.

python tools/harvest_blind.py --out datasets/blind --skip skip_ids.txt --per-query 15
"""
import argparse, os, subprocess, json, glob
import cv2, numpy as np

QUERIES = [
    "ambulance stretcher hospital", "gurney hospital corridor", "patient transport stretcher hospital",
    "emergency room stretcher arrival", "hospital transport trolley patient",
    "sedye hastane", "acil servis sedye", "ambulans sedye hastane",
    "camilla hospital paciente", "brancard hôpital urgences", "Krankentrage Krankenhaus",
    "barella ospedale pronto soccorso", "носилки больница скорая", "担架 医院 急诊",
    "ストレッチャー 病院 救急", "maca hospital ambulância", "ambulance cot loading",
    "stryker power pro cot", "paramedics wheel patient into hospital", "hospital porter trolley patient",
]


def dhash(img, n=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n))
    return (g[:, 1:] > g[:, :-1]).flatten()


CART_QUERIES = [
    "hospital linen cart", "hospital meal cart delivery", "medication cart hospital corridor",
    "hospital supply cart", "crash cart hospital", "housekeeping cart hospital corridor",
    "hospital food trolley", "laundry trolley hospital", "hospital waste cart",
    "hastane çamaşır arabası", "hastane yemek arabası", "ilaç arabası hastane",
    "carrello biancheria ospedale", "chariot linge hôpital", "Wäschewagen Krankenhaus",
    "carro de medicamentos hospital", "病院 配膳車", "医院 送餐车",
    "hospital porter pushing trolley", "hospital transport cart tug",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--skip", required=True)
    ap.add_argument("--per-query", type=int, default=15); ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=30); ap.add_argument("--max-dur", type=int, default=600)
    ap.add_argument("--carts", action="store_true", help="use the hospital-cart query list")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/images", exist_ok=True); os.makedirs(f"{a.out}/_tmp", exist_ok=True)
    skip = set(open(a.skip).read().split())
    cat_p = f"{a.out}/catalog.jsonl"
    done = {json.loads(l)["id"] for l in open(cat_p)} if os.path.exists(cat_p) else set()
    cat = open(cat_p, "a")
    for q in (CART_QUERIES if a.carts else QUERIES):
        lines = subprocess.run(["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(title)s",
                                f"ytsearch{a.per_query}:{q}"], capture_output=True, text=True).stdout.splitlines()
        for ln in lines:
            vid, dur, title = (ln.split("\t") + ["", ""])[:3]
            try:
                dur = float(dur)
            except ValueError:
                continue
            if vid in skip or vid in done or not (8 <= dur <= a.max_dur):
                continue
            done.add(vid)
            fp = f"{a.out}/_tmp/{vid}.mp4"
            subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", "b[height<=480][ext=mp4]/bv*[height<=480]",
                            "-o", fp, f"https://www.youtube.com/watch?v={vid}"], capture_output=True)
            if not os.path.exists(fp):
                continue
            cap = cv2.VideoCapture(fp); fps = cap.get(cv2.CAP_PROP_FPS) or 25; n = int(cap.get(7))
            step = max(1, int(round(fps * a.every))); hs, kept = [], 0
            for fi in range(0, n, step):
                cap.set(1, fi); ok, im = cap.read()
                if not ok:
                    break
                h = dhash(im)
                if any((h != x).sum() <= 6 for x in hs):   # near-duplicate frame
                    continue
                hs.append(h)
                cv2.imwrite(f"{a.out}/images/bl{vid}_{fi:06d}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90]); kept += 1
                if kept >= a.max_frames:
                    break
            cap.release(); os.remove(fp)
            cat.write(json.dumps(dict(id=vid, q=q, dur=dur, title=title, kept=kept), ensure_ascii=False) + "\n"); cat.flush()
            print(f"{q[:28]:28s} {vid} {kept:3d} {title[:50]}", flush=True)


if __name__ == "__main__":
    main()
