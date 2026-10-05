# Related work

## Products and prior art
| Source | What it does | How GurneyGate differs |
|---|---|---|
| Allegion, US 12188288 B2 (priority 2022), "Automatic door with radar sensing" | mm-wave radar + IMU; computes the arrival time T_a from target speed and position and opens if T_open < T_a + margin. Distinguishes living from non-living targets by breathing and height. | Same time-to-arrival idea; no object classes. GurneyGate adds class-selective opening (transport vs. cart) from a camera. |
| Optex VVS-1 image recognition sensor | 2D camera; person/object distinction, speed and direction; software can be tuned to "a specific shape and size". | No patient-transport classes; closed system. |
| Dorma DE19522760A1 (1995) | Presence detection in the door zone from pixel differences (CCD camera). | No classification. |
| US 10,257,470 | Cameras on both sides of the door; opening by motion or gesture pattern. | Not specific to patient transports. |
| NABCO / BEA hospital sensors | Radar + infrared; detect stretchers and carts but do not tell them apart. | Open for every object. |
| NABCO NATRUS+e W ("Image Sensing W"), https://nabco.nabtesco.com/en/solutions/topics/natrus_ew/ | For hospitals; computes the time of arrival from walking speed and opens on time for wheelchairs and carts. | Timing by arrival time is already a product; it does not open selectively by class. |
| MERL, US 2004/0022439 A1 (2002), https://patents.google.com/patent/US20040022439A1/en | Stereo camera above an elevator door; recognises wheelchair users by height and shape and holds the door open longer. | Class-specific door behaviour with a camera exists; GurneyGate targets corridor doors and lead time, not hold-open time. |
| Jurosch et al., IJCARS 2025 | "Patient Checker": patient-bed box overlap. | Same mechanism as the optional occupancy check (`require_occupant`), which needs a detector with a person class. |
| Capogrosso et al., ICPR-W 2022 | Intent prediction for a smart door. | Not specific to patient transports. |
| AWID LR-3000 UHF RFID | Tag on the stretcher, reader at the door. | Needs a tag on every stretcher; a camera-free alternative. |

## Method sources
- **Time to arrival / time to collision:** impact time and speed from target motion with a Kalman filter (US 8447472); time to contact from apparent size growth in robot navigation (IntechOpen 57651). In GurneyGate: a constant-velocity Kalman filter in seconds on the door-side box edge.
- **Pedestrian intent prediction:** crossing/not-crossing classification from trajectories and visual context (PMC8619260, PMC12694338, TrajFusionNet 2025). In GurneyGate: a small trajectory MLP (`gurneygate/intent.py`, entering / passing / other) trained on synthetic tracks, combined with the rule-based decision.
- **Hospital object detection:** YOLO models with bed, stretcher and wheelchair classes; Roboflow Universe sets of 500-800 images. Not used here: the GurneyGate detector is trained on its own 4-class labels (Open Images, Pexels and YouTube frames, `training/README.md`).
- **Privacy-preserving vision:** low-resolution depth (arXiv 1811.09950), low-resolution infrared person detection (arXiv 2209.11335), on-device processing without storing frames. In GurneyGate: processing at 640 px, frames stored only with `--save`, an event log instead of video.
- **Edge hardware:** on the Raspberry Pi 5 CPU with NCNN, YOLO26s inference takes about 173 ms (Ultralytics Raspberry Pi guide), about 6 frames/s; with a Hailo-8L accelerator a community measurement reports about 37 frames/s for YOLO26 end to end (https://github.com/DanielDubinsky/yolo26_hailo). GurneyGate ships an NCNN export for this target; it has not been benchmarked on the device yet.
- **Standards:** EN 16005 separates activation and safety sensors and monitors the safety sensor at every opening. GurneyGate is only an **activation** source (ANSI/BHMA A156.10); the door's safety sensors stay as they are.

## Contribution
Class-specific door behaviour with a camera (MERL 2002) and occupancy checking (Jurosch 2025) exist, and
timing by arrival time is a product (NABCO NATRUS+e W). GurneyGate contributes:
1. **Class-selective opening** at a corridor door: a patient transport opens it, a supply cart does not;
   the decision is made per track, from the voted class.
2. **Lead time** from a time-to-arrival estimate in seconds, designed for open edge hardware.
3. An **open benchmark** of hospital CCTV frames and an **event-level evaluation protocol** with
   confidence intervals. No site data yet.

## Datasets for evaluation
- HIOD, https://github.com/Wangmmstar/Hospital_Scene_Data: hospital scene objects.
- TUM MITI, https://link.springer.com/article/10.1007/s11548-025-03344-x: operating room / hospital.

Check their licences before training on them.

## Metrics for a field study
- Opening rate for transports (on time and at all), missed transports, false openings (carts, people, passers-by).
- Lead time: the moment the leading edge crosses the threshold minus the trigger moment (target: at least the door opening time).
- Waiting time of a transport at the threshold, compared with the push-button baseline.
- Number of times staff touch the door button, before and after.
