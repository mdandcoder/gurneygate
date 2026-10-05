# Hospital CCTV benchmark

247 frames from 4 public hospital CCTV videos on YouTube, sampled at 1 frame per second without running
any detector. Patient transports appear in 3 of the videos.

| Content | Count |
|---|---|
| Frames | 247: 14 with a transport, 32 with ignore regions only, 201 empty |
| Transport boxes | 14: 13 stretcher, 1 wheelchair, about 6 distinct transports |
| Ignore regions (class 9) | 32: partly visible transports, neither a required detection nor a false positive |

Labels follow `docs/LABEL_SCHEMA.md` (YOLO format, normalised coordinates).

## Rebuild the frames
Frames are not redistributed. `rebuild.py` downloads the videos and extracts the labelled frames; every
frame is checked against the SHA-256 in `manifest.csv` (image, YouTube ID, timestamp, frame hash).
```bash
pip install yt-dlp
python benchmark/gold_cctv/rebuild.py
python tools/eval_door.py --models gurneygate=weights/gurneygate-yolo26s.pt                # box must fit (IoU >= 0.5)
python tools/eval_door.py --models gurneygate=weights/gurneygate-yolo26s.pt --min-iou 0.3  # object seen
```
Downloading is subject to YouTube's Terms of Service; use the frames for research evaluation only. If a
video is taken down, its frames cannot be rebuilt.

## How it was made, and its limits
- 413 frames were sampled from 5 videos (near-duplicates in static scenes removed). 166 frames were
  excluded because they held an object that could not be boxed unambiguously: a patient in a partly
  visible bed or recliner, or tables that cannot be told from trolleys. One whole video was lost this way.
  Many excluded frames show a transport far away, early in its approach, so this benchmark is easier
  than an unfiltered stream.
- Boxes were proposed by Grounding DINO, independent of the detectors under test. Every box was
  accepted, corrected or rejected by a person (`tools/propose_sheets.py`, `tools/build_gold_labels.py`).
- Four frames that show only the mattress top from directly above a stretcher, a view a door camera
  does not have, are ignore regions.
- Consecutive frames show the same transport, so the 14 boxes are not independent observations.
- The benchmark was used to choose between candidate detectors. Treat it as a validation set.
- None of these videos was used for training.

## License
Labels and `manifest.csv`: CC BY 4.0 (`LICENSE` in this folder). Attribution: "GurneyGate hospital CCTV
benchmark, Baris Ozturk, 2026". The videos belong to their owners.
