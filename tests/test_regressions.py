# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Regression tests for failure modes found in code review and on test clips (no model, no camera)."""
import sys, pathlib, threading, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gurneygate.decision import ApproachDecider
from gurneygate.detector import Detection
from gurneygate.door import DoorBase

CFG = dict(door_line_y=0.85, approach_direction="down", min_box_area=0.03, use_intent=False)


def box(tid, cy, size=0.3, cls="transport"):
    return Detection(tid, cls, 0.9, 0.35, cy - size / 2, 0.65, cy + size / 2)


def test_missed_frames_do_not_reset_track():
    """A detection missed every 7th frame must not prevent opening (track must not be reset)."""
    d = ApproachDecider(CFG, fps=30)
    opened = False
    for i in range(40):
        dets = [] if i % 7 == 6 else [box(1, 0.1 + i * 0.012)]
        o, _ = d.update(dets)
        opened |= o
    assert opened, "short detection gaps must not delete the track and block opening"


def test_track_expires_after_long_gap():
    d = ApproachDecider(CFG, fps=30)
    d.update([box(1, 0.5)])
    lost = []
    for _ in range(int(30 * 0.5)):          # 0.5 s: still alive
        d.update([]); lost += d.last_lost
    assert 1 in d.tracks and not lost
    for _ in range(int(30 * 1.5)):          # 2 s total: must be dropped, exactly once
        d.update([]); lost += d.last_lost
    assert 1 not in d.tracks and lost == [1]


def test_cart_never_opens():
    """A cart track must not open the door; the decision is made per track."""
    d = ApproachDecider(CFG, fps=30)
    for i in range(30):
        o, _ = d.update([box(1, 0.1 + i * 0.03, cls="cart")])
        assert not o


def test_class_vote_gates_trigger():
    """A single-frame 'cart' flip must not stop a transport track; a mostly-cart track must not open."""
    d = ApproachDecider(CFG, fps=30)
    opened = False
    for i in range(30):
        cls = "cart" if i == 12 else "transport"
        o, _ = d.update([box(1, 0.1 + i * 0.03, cls=cls)])
        opened |= o
    assert opened
    d2 = ApproachDecider(CFG, fps=30)
    opened2 = False
    for i in range(30):
        cls = "transport" if i % 4 == 0 else "cart"
        o, _ = d2.update([box(1, 0.1 + i * 0.03, cls=cls)])
        opened2 |= o
    assert not opened2, "a track that is mostly cart must not open the door"


def test_tta_same_in_seconds_across_fps():
    """The same physical approach (0.3 units/s) must predict the same ARRIVAL TIME (+-5%) at 30 and 6 fps.
    Truth: leading edge from 0.25 to 0.85 at 0.3 units/s -> t = 2.0 s."""
    arrive = {}
    for fps in (30.0, 6.0):
        d = ApproachDecider(CFG, fps=fps)
        k_last = int(round(1.5 * fps)) - 1
        for k in range(k_last + 1):
            d.update([box(1, 0.1 + (k / fps) * 0.3)])
        arrive[fps] = k_last / fps + d.tracks[1].last_tta
    for fps, t in arrive.items():
        assert abs(t - 2.0) < 0.1, arrive


def test_tta_uses_leading_edge():
    """TTA must be computed from the door-side edge of the box, the same point as the crossing check."""
    d = ApproachDecider(CFG, fps=30)
    for i in range(12):
        d.update([box(1, 0.2 + i * 0.03)])
    st = d.tracks[1]
    # last box: cy=0.53, bottom edge 0.68 -> edge distance 0.17, speed 0.03/frame -> ~0.19 s
    assert 0.1 < st.last_tta < 0.35, st.last_tta


def test_far_object_not_opened_by_distance_shortcut():
    """A fast object (short TTA) farther than max_trigger_distance from the door must not open it."""
    d = ApproachDecider(CFG, fps=30)
    for i in range(30):
        cy = 0.05 + i * 0.9 / 30              # leading edge 0.20->0.47: >= 0.38 from the door, TTA < 1 s
        o, info = d.update([box(1, cy)])
        if 0.85 - (cy + 0.15) > 0.42:
            assert info[0]["tta"] < 2.2 or i < 5
            assert not o, (i, info[0]["status"])


class _FakeIntent:
    ready = True
    def p_enter(self, hist, door_line, direction):
        return 0.01 if len(hist) >= 10 else None   # always says "passing"; None before 10 frames


def test_intent_veto_not_bypassed_before_window_fills():
    """With the intent model on, the first trigger must not happen with p_enter=None (the MLP was bypassed)."""
    d = ApproachDecider(dict(CFG, use_intent=True), fps=30)
    d.intent = _FakeIntent(); d.use_intent = True
    for i in range(9):   # 9 frames: before the window fills; outside the at-door band (leading edge <= 0.51)
        o, _ = d.update([box(1, 0.2 + i * 0.02)])
        assert not o, f"frame {i}: triggered before the intent window filled"


def test_at_door_band_opens_despite_intent():
    """A transport moving toward the door right in front of it must open even if the intent model says no."""
    d = ApproachDecider(dict(CFG, use_intent=True), fps=30)
    d.intent = _FakeIntent(); d.use_intent = True
    opened = False
    for i in range(30):
        o, _ = d.update([box(1, 0.35 + i * 0.012)])
        opened |= o
    assert opened


class _RecDoor(DoorBase):
    def __init__(self, hold=60.0, cooldown=0.0):
        super().__init__(hold_seconds=hold, cooldown=cooldown)
        self.events = []
    def _open(self): self.events.append("open")
    def _close(self): self.events.append("close")


def test_shutdown_closes_open_door():
    door = _RecDoor()
    door.trigger()
    assert door.is_open
    door.shutdown()
    assert not door.is_open and door.events[-1] == "close"


def test_watchdog_force_closes_on_stall():
    """If the main loop stalls (blocked RTSP read) the watchdog must close the door."""
    from gurneygate.main import Watchdog
    door = _RecDoor()
    door.trigger()
    wd = Watchdog(door, timeout_s=0.2, on_fire=lambda gap: None)
    wd.beat()
    wd.start()
    time.sleep(0.6)            # the loop never calls beat()
    wd.stop()
    assert not door.is_open and "close" in door.events


def test_at_door_band_ignores_lateral_pass():
    """A jittering transport passing across right at the door and drifting slightly toward it (0.12/s,
    above the band threshold) must not trigger the band: lateral speed exceeds speed toward the door.
    Intent says 'passing'."""
    import random
    for seed in range(10):
        random.seed(seed)
        d = ApproachDecider(dict(CFG, use_intent=True), fps=18)
        d.intent = _FakeIntent(); d.use_intent = True
        for k in range(45):
            t = k / 18; cx = -0.05 + 0.45 * t
            y2 = 0.76 + 0.12 * t + random.uniform(-0.005, 0.005)
            if y2 >= 0.84:
                break
            o, _ = d.update([Detection(1, "transport", 0.9, cx - 0.12, y2 - 0.24, cx + 0.12, y2)])
            assert not o, f"seed {seed}"


def test_at_door_band_opens_for_straight_slow_approach():
    """Same speed toward the door without lateral motion: the band must open (counterpart of the test above)."""
    d = ApproachDecider(dict(CFG, use_intent=True), fps=18)
    d.intent = _FakeIntent(); d.use_intent = True
    opened = False
    for k in range(20):
        y2 = 0.76 + 0.12 * k / 18
        o, _ = d.update([Detection(1, "transport", 0.9, 0.38, y2 - 0.24, 0.62, y2)])
        opened |= o
    assert opened


def test_parked_at_door_with_jitter_stays_closed():
    """A stretcher parked at the door with a jittering box must not open it (intent off: worst case).
    The Kalman's instantaneous velocity jumped with jitter; heading now comes from the window slope."""
    import random
    for seed in range(20):
        random.seed(seed)
        d = ApproachDecider(dict(CFG, use_intent=False), fps=18)
        for _ in range(54):
            j = random.uniform(-0.02, 0.02)
            o, _ = d.update([Detection(1, "transport", 0.9, 0.38, 0.56 + j, 0.62, 0.80 + j)])
            assert not o, f"seed {seed}"


def test_cart_taking_over_transport_id_does_not_open():
    """The tracker matches by IoU regardless of class, so a cart can take over a transport's track id.
    The transport approaches slowly (TTA > 2.2 s, no open); a fast cart takes over its id.
    While the votes still say 'transport', only the raw-class veto keeps the first cart frames from opening."""
    d = ApproachDecider(CFG, fps=30)
    y2 = 0.40
    for i in range(30):                       # 0.08/s: 0.45 from the door -> TTA ~5 s
        y2 += 0.08 / 30
        o, _ = d.update([Detection(1, "transport", 0.9, 0.38, y2 - 0.24, 0.62, y2)])
        assert not o
    for i in range(12):                       # same id, cart, 0.6/s: TTA < 1 s
        y2 += 0.6 / 30
        o, info = d.update([Detection(1, "cart", 0.9, 0.38, y2 - 0.24, 0.62, y2)])
        assert not o, (i, info[0]["status"])


def test_half_labelled_cart_rarely_opens():
    """A cart the model calls 'transport' half the time (coin flip). A simple vote majority opened ~35%;
    with the Wilson lower bound at most 3 of 40 (measured: 8 of 200). Not zero: it is a detector error."""
    import random
    n = 0
    for seed in range(40):
        random.seed(seed)
        d = ApproachDecider(CFG, fps=30)
        for i in range(40):
            o, _ = d.update([box(1, 0.2 + i * 0.015, cls=random.choice(("transport", "cart")))])
            if o:
                n += 1
                break
    assert n <= 3, n


def test_mostly_transport_with_rare_cart_label_still_opens():
    d = ApproachDecider(CFG, fps=30)
    opened = False
    for i in range(40):
        o, _ = d.update([box(1, 0.2 + i * 0.015, cls="cart" if i % 6 == 5 else "transport")])
        opened |= o
    assert opened


def test_opens_at_low_fps():
    """At 3 fps a 0.5 s window held 2 samples: heading was never measured and the door never opened."""
    d = ApproachDecider(CFG, fps=3)
    opened = False
    for k in range(18):
        y2 = 0.35 + 0.24 * k / 3
        if y2 > 0.9:
            break
        o, _ = d.update([Detection(1, "transport", 0.9, 0.38, y2 - 0.24, 0.62, y2)])
        opened |= o
    assert opened


def test_real_timestamps_used_for_velocity():
    """Live, the decision clock must use real frame times: a 0.9 s stall at 10 fps and the default 30 fps
    must not make a very slow (0.03/s) object 0.3 from the door look fast."""
    d = ApproachDecider(CFG, fps=30)          # live start: the camera's declared rate
    t = 0.0
    for k in range(40):
        t += 0.9 if k == 20 else 0.1
        y2 = 0.55 + 0.03 * t
        o, info = d.update([Detection(1, "transport", 0.9, 0.38, y2 - 0.24, 0.62, y2)], t=t)
        assert not o, (k, info[0]["status"])


def test_door_tick_releases_after_hold():
    door = _RecDoor(0.05, 0.0)
    door.trigger()
    assert door.is_open
    time.sleep(0.08)
    door.tick()
    assert not door.is_open and door.events[-1] == "close"


def test_http_door_releases_relay_in_order():
    """HTTP relay: 'open' and 'close' go out in order; the last request on shutdown is 'close'."""
    import http.server
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path); self.send_response(200); self.end_headers()
        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    from gurneygate.door import HttpDoor
    d = HttpDoor(base + "/on", base + "/off", hold_seconds=60, cooldown=0)
    d.trigger(); d.force_close(); d.trigger(); d.shutdown()
    srv.shutdown()
    assert seen[:3] == ["/on", "/off", "/on"] and seen[-1] == "/off", seen


def test_http_door_refuses_latching_url_without_close():
    import pytest
    from gurneygate.door import HttpDoor
    with pytest.raises(SystemExit):
        HttpDoor("http://127.0.0.1:9/relay/0?turn=on", None, hold_seconds=1, cooldown=0)
    HttpDoor("http://127.0.0.1:9/relay/0?turn=on&timer=3", None, hold_seconds=1, cooldown=0).shutdown()


def test_is_file_treats_streams_as_live(tmp_path):
    from gurneygate.main import _is_file
    f = tmp_path / "a.mp4"; f.write_bytes(b"x")
    assert _is_file(str(f))
    for s in ("rtsps://x/y", "rtmp://x/y", "udp://@:1234", "/dev/video0", "rtsp://x"):
        assert not _is_file(s), s


def test_track_born_past_door_line_never_opens():
    """A track born past the door line at a scene cut, with a growing box, must not open (seen on a test clip)."""
    d = ApproachDecider(CFG, fps=25)
    for k in range(25):
        y2 = min(0.999, 0.90 + 0.005 * k)     # 0.12 units/s in the clip
        o, info = d.update([Detection(1, "transport", 0.9, 0.2, y2 - 0.6 - 0.004 * k, 0.8, y2)])
        assert not o, (k, info[0]["status"])


def test_unknown_door_backend_is_refused():
    """A typo in door.backend must stop the program, not silently fall back to a dry run."""
    import pytest
    from gurneygate.door import make_door
    with pytest.raises(SystemExit):
        make_door({"backend": "GPIO"}, 6.0, 1.0)
    make_door({"backend": "dryrun"}, 6.0, 1.0).shutdown()
