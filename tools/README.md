# Tools

| Script | What it does |
|---|---|
| `eval_door.py` | Per-box evaluation: recall per class with Wilson intervals, false boxes, paired McNemar test between models; ignore regions (class 9) are neither hits nor false positives |
| `eval_events.py` | Per-event evaluation of a door log against a ground-truth sheet: on-time opening rate, lead time, unnecessary openings per hour (Clopper-Pearson intervals) |
| `propose_gdino.py` | Model-independent box proposals with Grounding DINO (open vocabulary), so labels are not biased toward the detector being tested |
| `propose_sheets.py` | Numbered review sheets of the proposals for fast accept / reject / fix |
| `build_gold_labels.py` | Turns review decisions into YOLO labels, including hand-drawn boxes and ignore regions |
| `harvest_blind.py` | Samples frames from public videos found by search queries in several languages, without running any detector, so the sample does not depend on the model being tested |

Usage is in each script's docstring.
