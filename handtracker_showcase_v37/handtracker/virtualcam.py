"""
virtualcam.py
--------------
Makes the app usable alongside Zoom / Meet / Teams.

THE PROBLEM: a webcam can normally be opened by only ONE program at a time.
This app needs the camera to see your hands, so while it's running the call
app would find the camera "in use" (or this app would fail to start if the
call app got there first).

THE FIX (the same approach OBS and Snap Camera use): this app keeps the real
camera, and re-publishes the video to a *virtual camera* device. In the call
app you pick the virtual camera ("OBS Virtual Camera", "Unity Video Capture",
etc.) instead of your real webcam - the call gets normal video, and gesture
control keeps working at the same time.

What's sent is the CLEAN, un-mirrored camera frame - not the debug window
with hand skeletons and HUD drawn on it.

Needs the optional `pyvirtualcam` package plus a virtual-camera driver
(on Windows, installing OBS Studio registers "OBS Virtual Camera"). Neither
is required to run the app: without them start() just returns a helpful
message and the rest of the app is unaffected.

Frames are handed to a worker thread through a one-slot mailbox (newest
wins, older ones are dropped), so a slow virtual-camera driver can never
stall hand tracking. The worker re-sends the latest frame at the target
fps, so the call app never sees a frozen source if tracking dips.
"""
import os
import sys
import threading
import time

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import pyvirtualcam as _pvc
except Exception:
    _pvc = None

INSTALL_HELP = ("Virtual camera needs the 'pyvirtualcam' package and a virtual-camera driver "
                "(Windows: install OBS Studio, which provides 'OBS Virtual Camera'). "
                "Then pick that camera in Zoom/Meet/Teams.")



def diagnose(backend=None):
    """What's missing for a virtual camera to work, as (check, ok, how_to_fix)
    rows - printed when starting one fails so "no virtual camera is created"
    comes with a concrete reason. A virtual camera is a DEVICE other apps can
    select, so besides the Python package it needs a driver that provides one;
    on Windows/macOS that driver comes with OBS Studio."""
    have_pkg = (backend if backend is not None else _pvc) is not None
    rows = [("Python package 'pyvirtualcam'", have_pkg, "pip install pyvirtualcam")]
    if sys.platform.startswith("win"):
        roots = [os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
        obs = any(os.path.isdir(os.path.join(r, "obs-studio")) for r in roots)
        rows.append(("OBS Studio (provides 'OBS Virtual Camera')", obs,
                     "install OBS Studio from obsproject.com, then pick 'OBS Virtual Camera' in your call app"))
    else:
        rows.append(("A virtual-camera driver (OBS Studio on macOS, v4l2loopback on Linux)", None,
                     "install OBS Studio (macOS) or the v4l2loopback module (Linux)"))
    rows.append(("OBS NOT already using its virtual camera", None,
                 "if OBS itself has 'Start Virtual Camera' running, stop it - only one program can feed it"))
    return rows


def format_diagnosis(rows):
    out = []
    for check, ok, fix in rows:
        mark = "OK " if ok else ("?  " if ok is None else "NO ")
        out.append(f"  [{mark}] {check}" + ("" if ok else f"  ->  {fix}"))
    return "\n".join(out)


class VirtualCamera:
    def __init__(self, backend=None):
        self._backend = backend if backend is not None else _pvc
        self._cam = None
        self._thread = None
        self._cond = threading.Condition()
        self._frame = None
        self._stop = False
        self.sent = 0
        self.dropped = 0
        self.error = None
        self.device = None
        self.size = None

    @property
    def available(self):
        return self._backend is not None and cv2 is not None

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive() and self.error is None

    def start(self, width, height, fps=30):
        """Returns (ok, message) - message is always safe to show the user."""
        if self.active:
            return True, f"Virtual camera already on ({self.device})"
        if not self.available:
            return False, INSTALL_HELP
        try:
            self._cam = self._backend.Camera(int(width), int(height), int(fps))
        except Exception as e:
            self._cam = None
            return False, f"Couldn't open a virtual camera ({e}). {INSTALL_HELP}"
        self.size = (int(width), int(height))
        self.device = getattr(self._cam, "device", "virtual camera")
        self.error = None
        self._stop = False
        self._frame = None
        self.sent = self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="virtualcam", daemon=True)
        self._thread.start()
        return True, f"Virtual camera ON - choose '{self.device}' in your call app"

    def send(self, bgr_frame):
        """Non-blocking: hand over the newest frame (BGR, un-mirrored)."""
        if not self.active:
            return
        with self._cond:
            if self._frame is not None:
                self.dropped += 1            # previous frame never got sent: superseded
            self._frame = bgr_frame
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None

    # -- worker ---------------------------------------------------------------
    def _run(self):
        w, h = self.size
        last_rgb = None
        while True:
            with self._cond:
                if self._stop:
                    return
                if self._frame is None and last_rgb is None:
                    self._cond.wait(timeout=0.25)
                frame, self._frame = self._frame, None
            try:
                if frame is not None:
                    if frame.shape[1] != w or frame.shape[0] != h:
                        frame = cv2.resize(frame, (w, h))
                    last_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if last_rgb is not None:
                    self._cam.send(last_rgb)
                    self.sent += 1
                    self._cam.sleep_until_next_frame()
            except Exception as e:           # driver went away, device closed, ...
                self.error = str(e)
                return
