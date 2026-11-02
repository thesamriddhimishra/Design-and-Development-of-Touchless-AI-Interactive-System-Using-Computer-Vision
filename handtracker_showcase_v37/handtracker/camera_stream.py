"""
camera_stream.py
----------------
Background-thread camera capture that always returns the NEWEST frame.

Why: `cv2.VideoCapture.read()` blocks the main loop while the webcam
delivers + decodes a frame (MJPEG decode alone is several ms at 720p), and
frames you were too slow to consume pile up in a driver buffer, so what you
see is increasingly STALE (input lag) when the loop runs under camera speed.
Here a thread keeps reading, the main loop takes the latest frame and never
sees stale ones, and decode overlaps with tracking + drawing.

Also requests MJPG / FPS / a 1-frame buffer when opening the camera: many
webcams default to uncompressed YUY2 which is bandwidth-limited to ~10 fps
at 1280x720, while MJPG delivers 30.

Drop-in for the three VideoCapture methods the loop uses: read(),
isOpened(), release(). Unit-tested with a fake capture (tests/test_perf.py).
"""
import threading
import time

import cv2

_BACKENDS = {"dshow": "CAP_DSHOW", "msmf": "CAP_MSMF", "v4l2": "CAP_V4L2",
             "avfoundation": "CAP_AVFOUNDATION", "gstreamer": "CAP_GSTREAMER"}


def open_capture(index, width, height, fps=30, fourcc="MJPG", backend=None, buffer_size=1,
                 autofocus=True):
    """cv2.VideoCapture with low-latency properties requested. Every property
    is best-effort: drivers may ignore them, so nothing here can fail."""
    flag = getattr(cv2, _BACKENDS.get(str(backend).lower(), ""), None) if backend else None
    cap = cv2.VideoCapture(index, flag) if flag is not None else cv2.VideoCapture(index)

    def setp(prop, val):
        try:
            cap.set(prop, val)
        except Exception:
            pass
    if fourcc:
        setp(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))   # before size: some drivers key modes on it
    setp(cv2.CAP_PROP_FRAME_WIDTH, width)
    setp(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        setp(cv2.CAP_PROP_FPS, fps)
    if buffer_size:
        setp(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    if autofocus:
        # A camera another app (OBS, Zoom, the vendor's own utility) left in
        # manual focus stays soft in this app too; ask for continuous
        # autofocus. Ignored by cameras without a focus motor.
        setp(cv2.CAP_PROP_AUTOFOCUS, 1)
    return cap


def describe_capture(cap):
    """What the camera ACTUALLY delivers, e.g. "1280x720 @ 30 fps MJPG" (or
    None if the driver won't say). Drivers silently substitute the nearest
    mode they support, so asking for a size isn't the same as getting it - a
    non-native size is scaled by the driver and looks soft, which is worth
    being able to see."""
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        fcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") if code else "?"
        return w, h, fps, fcc
    except Exception:
        return None


class FrameGrabber:
    def __init__(self, capture, stall_timeout=3.0):
        self._cap = capture
        self._stall = stall_timeout
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0            # frames captured
        self._taken = 0          # seq of the last frame handed out
        self._eof = False
        self._stop = False
        self.delivered = 0
        self.dropped = 0         # captured but replaced before anyone read them
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                ret, frame = self._cap.read()
            except Exception:
                ret, frame = False, None
            with self._cond:
                if not ret:
                    self._eof = True
                    self._cond.notify_all()
                    return
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self):
        """(True, newest_unseen_frame) or (False, None) on EOF/stall. Blocks
        until a frame newer than the last one returned exists."""
        deadline = time.monotonic() + self._stall
        with self._cond:
            while self._seq == self._taken and not self._eof and not self._stop:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False, None
                self._cond.wait(left)
            if self._seq == self._taken:
                return False, None
            self.dropped += self._seq - self._taken - 1
            self._taken = self._seq
            self.delivered += 1
            return True, self._frame

    def isOpened(self):
        return (not self._stop) and (self._cap.isOpened() or self._seq != self._taken)

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.5)
        try:
            self._cap.release()
        except Exception:
            pass
