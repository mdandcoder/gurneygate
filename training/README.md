# Training

`train.py` trains the detector from folders of labelled images.

```bash
python training/train.py --train data/source_a data/source_b:3 data/hard_negatives --val data/val
python training/train.py --train data/my_door --val data/my_door_val --model weights/gurneygate-yolo26s.pt   # fine-tune
```

## Data format
Each folder holds `images/` and `labels/` (YOLO format, same file stems) with the 4-class schema of
`docs/LABEL_SCHEMA.md`: `0 stretcher, 1 wheelchair, 2 bed, 3 cart`. An empty label file marks a
verified hard negative. Images without a label file are skipped. For training, the labels are merged into
two classes: `transport` (stretcher, wheelchair, bed) and `cart`.

`folder:N` uses every image of that folder that contains a stretcher N times; stretchers are the rarest
and most important class.

## Recipe of the released weights (`gurneygate-yolo26s.pt`)
- Start from `yolo26s.pt` (Ultralytics 8.4.157). 100 epochs, 640 px, batch 32, no early stopping, last
  weights kept. Augmentation: `degrees=10 scale=0.5 fliplr=0.5 mosaic=1.0 mixup=0.1 hsv_v=0.5 close_mosaic=10`.
- Sources:
  - Open Images: images with stretcher, wheelchair, bed or cart boxes; stretcher images weighted 2x.
  - Pexels: two sets of stock-video frames.
  - YouTube: frames from hospital videos.
  - Grounding DINO set: frames whose boxes were proposed by Grounding DINO and reviewed by hand;
    stretcher images weighted 3x.
  - 159 hard negatives: verified frames without any transport, on which an earlier detector fired.
- Every box was reviewed by a person. All splits are made by video, so frames of one video never
  appear on both sides.
- Validation during training is a held-out split of the YouTube frames. It is used only to monitor
  training.

The training images are not redistributed (copyright of their owners; some show people).

## Held-out sets and model selection
- None of the evaluation sets in the main README was used for training.
- The hospital CCTV benchmark (`benchmark/gold_cctv`) and the held-out sets were used to choose between
  candidate detectors (for example the cart experiment in the main README). Read the benchmark numbers
  as a validation result, not as an untouched test.
- The untouched test is a recording from the deployment door (`docs/FIELD_RECORDING.md`).
