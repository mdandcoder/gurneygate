# Field recording at a real door

How to record a real hospital door so that GurneyGate can be calibrated, fine-tuned and evaluated on it.
One or two days of video from the deployment camera give both training data for that camera and the
test that matters: did the door open on time, and how often did it open for nothing.

## Before recording
- Written permission from the hospital management; for a publication, ethics committee (IRB) approval.
- Personal data (KVKK in Turkey, GDPR in the EU, HIPAA in the US): inform staff with a notice at the
  door, record **no audio**, keep raw video on an encrypted disk and never upload it to a cloud service
  or any public place. Blur faces before any frame leaves the local machine.

## Camera
- A phone on a tripod or wall bracket is enough. Mains power, not battery.
- Mount it **above the door on the approach side**, 2.4-3 m high, tilted down the corridor
  (30-45 degrees below horizontal). The camera must see at least **6-8 m of the approach**: a stretcher moves
  about 1-1.5 m/s and the door needs its opening time plus a margin of lead.
- No fisheye or wide-angle "0.5x" lens; ordinary 1x.
- 1080p or 720p, **at least 15 fps**, exposure and focus locked (no auto-exposure pumping when the
  door opens to a bright room).
- Storage: about 2-4 GB per hour at 1080p/15 fps.

## What to record
- At least **24 hours** in one go, so day shift, night shift and lighting changes are covered. Two
  days, or two doors, is better.
- Everything that passes is useful: stretchers, beds, wheelchairs, carts (linen, meal, medication,
  cleaning), people walking. Carts from this corridor are the real negatives.

## Ground truth sheet
Write one row per passage through the door, in the format `tools/eval_events.py` reads (`clip,kind,t`):
- `transport`: a stretcher, bed or wheelchair **going through the door**; `t` = the second its leading
  edge crosses the threshold (scene time of the recording).
- `cart` or `pedestrian`: a passage that must not open the door; `t` = passage time.
- one row `clip,duration,<seconds>` per recording, for unnecessary openings per hour.

```
clip,kind,t
door1_day1,transport,734.2
door1_day1,cart,1210.0
door1_day1,pedestrian,1388.5
door1_day1,duration,86400
```
A transport that only passes by or moves away is not a `transport` row. Even a partial sheet (for
example two hours, with the duration of those two hours) is enough to score the system.

Also write down once:
- the door's real opening time (stopwatch, from activation to fully open), measured 3 times;
- the distance from the door to a few floor marks visible in the image (tape measure), for the
  door-line and distance calibration;
- a photo of the camera mount and the view.

## After recording
1. Sample frames from the video and blur faces before anything leaves the local machine.
2. Propose boxes with Grounding DINO (`tools/propose_gdino.py`) and review them on numbered sheets
   (`tools/propose_sheets.py`); every box is accepted, corrected or rejected by a person.
3. Split by **day/time block**, not by frame, so test frames never neighbour training frames.
4. Fine-tune the detector on the training block with `training/train.py --model weights/gurneygate-yolo26s.pt`.
5. Run the full system on the held-out block and score it per event with `tools/eval_events.py`:
   share of passages opened on time, lead time, unnecessary openings per hour.
