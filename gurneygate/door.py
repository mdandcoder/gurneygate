# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Baris Ozturk
"""Door drivers: dry run, GPIO relay, serial port (Arduino), HTTP.

The system is an ACTIVATION sensor (in the ANSI/BHMA A156.10 sense), not a safety device:
the door's own safety sensors always stay active. On every failure path the trigger
is withdrawn (relay released) and the door falls back to its own sensors.
"""
from __future__ import annotations

import threading
import time


class DoorBase:
    def __init__(self, hold_seconds: float, cooldown: float):
        self.hold = hold_seconds
        self.cooldown = cooldown
        self.open_until = 0.0
        self.last_trigger = -1e9
        self.is_open = False
        self._lock = threading.Lock()          # main loop + watchdog thread

    def trigger(self) -> bool:
        """Approach signal: open the door if needed and extend the hold time."""
        with self._lock:
            now = time.monotonic()
            self.open_until = now + self.hold
            if not self.is_open and now - self.last_trigger >= self.cooldown:
                self.last_trigger = now
                self.is_open = True
                self._open()
                return True
            return False

    def tick(self):
        """Called every frame; closes when the hold time has run out."""
        with self._lock:
            if self.is_open and time.monotonic() >= self.open_until:
                self.is_open = False
                self._close()

    def force_close(self) -> bool:
        """Withdraw the trigger immediately (watchdog, shutdown). Does nothing if already closed."""
        with self._lock:
            if not self.is_open:
                return False
            self.is_open = False
            self.open_until = 0.0
            self._close()
            return True

    def shutdown(self):
        """On exit or crash: release the relay and free the hardware."""
        try:
            self.force_close()
        finally:
            self._cleanup()

    def remaining(self) -> float:
        return max(0.0, self.open_until - time.monotonic()) if self.is_open else 0.0

    def _open(self):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError

    def _cleanup(self):
        pass


class DryRunDoor(DoorBase):
    def _open(self):
        print(f"[door] OPEN  ({time.strftime('%H:%M:%S')})")

    def _close(self):
        print(f"[door] CLOSE ({time.strftime('%H:%M:%S')})")


class GpioDoor(DoorBase):
    """Raspberry Pi: relay board on a GPIO pin. The relay's NO contact goes to the door
    operator's external activation input."""

    def __init__(self, pin: int, **kw):
        super().__init__(**kw)
        import RPi.GPIO as GPIO  # type: ignore

        self.GPIO = GPIO
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    def _open(self):
        self.GPIO.output(self.pin, self.GPIO.HIGH)

    def _close(self):
        self.GPIO.output(self.pin, self.GPIO.LOW)

    def _cleanup(self):
        # never leave the pin HIGH when the process exits: drive it low first, then free it
        try:
            self.GPIO.output(self.pin, self.GPIO.LOW)
        finally:
            self.GPIO.cleanup(self.pin)


class SerialDoor(DoorBase):
    """Relay via Arduino/ESP32: 'O\\n' opens, 'C\\n' closes.
    A hardware-side timeout is recommended (e.g. release if no new 'O' arrives within 10 s)."""

    def __init__(self, port: str, baud: int, **kw):
        super().__init__(**kw)
        import serial  # pyserial

        self.ser = serial.Serial(port, baud, timeout=1)

    def _open(self):
        self.ser.write(b"O\n")

    def _close(self):
        self.ser.write(b"C\n")

    def _cleanup(self):
        try:
            self.ser.write(b"C\n")
            self.ser.flush()
        finally:
            self.ser.close()


class HttpDoor(DoorBase):
    """Network relay (Shelly, Tasmota, ...). Requests go IN ORDER on a single background thread:
    the video loop never blocks and "close" can never overtake "open".
    close_url is required to release the relay; otherwise the open URL must time out by itself
    (e.g. Shelly ...?turn=on&timer=3). With neither the program refuses to start: a latched relay could not be released."""

    def __init__(self, url: str, close_url: str | None = None, **kw):
        super().__init__(**kw)
        if not close_url and "timer" not in url.lower():
            raise SystemExit("[door] http: the relay could not be released. Set door.http_close_url or use a "
                             "self-timing open URL (e.g. ...?turn=on&timer=3).")
        import queue
        self.url, self.close_url = url, close_url
        self.q = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        import urllib.request
        while True:
            u = self.q.get()
            if u is None:
                return
            try:
                urllib.request.urlopen(u, timeout=2)
            except Exception as e:  # do not stay silent if the door failed to open/close
                print(f"[door] HTTP error: {e}")

    def _open(self):
        self.q.put(self.url)

    def _close(self):
        if self.close_url:
            self.q.put(self.close_url)

    def _cleanup(self):
        # let everything queued (including the final "close") go out, then stop the thread
        if self.close_url:
            self.q.put(self.close_url)
        self.q.put(None)
        self.worker.join(timeout=5)


def make_door(cfg: dict, hold: float, cooldown: float) -> DoorBase:
    kw = dict(hold_seconds=hold, cooldown=cooldown)
    b = cfg.get("backend", "dryrun")
    if b == "gpio":
        return GpioDoor(int(cfg["gpio_pin"]), **kw)
    if b == "serial":
        return SerialDoor(cfg["serial_port"], int(cfg["serial_baud"]), **kw)
    if b == "http":
        return HttpDoor(cfg["http_url"], cfg.get("http_close_url"), **kw)
    if b == "dryrun":
        return DryRunDoor(**kw)
    raise SystemExit(f"[door] unknown backend {b!r}; use dryrun | gpio | serial | http")
