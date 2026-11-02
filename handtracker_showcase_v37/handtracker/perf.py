"""
perf.py
-------
Adaptive quality + small fast-path helpers for the main loop.

PerfGovernor watches how long each frame's COMPUTE takes (not the camera
wait) and steps quality down when the machine can't keep up, and back up
when there's plenty of headroom. I can't know your CPU, so instead of
guessing settings it adapts to them:

    level  preview thumbnail   3D item budget   help panel
      0    every frame            450             on
      1    every 2nd              300             on
      2    every 3rd              170             on
      3    every 6th               90             off

It never changes tracking accuracy or gesture logic - only what is drawn
and how often. Changes are rate-limited and use hysteresis so it doesn't
flap, and a single slow frame (window drag, GC) is ignored by the EMA.
"""
import cv2
import numpy as np

LEVELS = (
    {"preview_every": 1, "item_budget": 450, "help": True},
    {"preview_every": 2, "item_budget": 300, "help": True},
    {"preview_every": 3, "item_budget": 170, "help": True},
    {"preview_every": 6, "item_budget": 90, "help": False},
)


class PerfGovernor:
    def __init__(self, target_fps=30.0, enabled=True, down_after=1.0, up_after=6.0,
                 min_interval=2.5, headroom=0.6):
        self.enabled = enabled
        self.budget_ms = 1000.0 / target_fps * 0.85     # compute must fit in ~85% of a frame
        self.down_after, self.up_after = down_after, up_after
        self.min_interval, self.headroom = min_interval, headroom
        self.level = 0
        self.ema = None
        self._over_since = None
        self._under_since = None
        self._changed_at = -1e9

    @property
    def settings(self):
        return LEVELS[self.level]

    def update(self, proc_ms, now):
        """Feed one frame's compute time (ms). True if the level changed."""
        if not self.enabled:
            return False
        self.ema = proc_ms if self.ema is None else self.ema + 0.1 * (proc_ms - self.ema)
        if self.ema > self.budget_ms:
            self._under_since = None
            self._over_since = self._over_since if self._over_since is not None else now
            if (now - self._over_since >= self.down_after and self.level < len(LEVELS) - 1
                    and now - self._changed_at >= self.min_interval):
                return self._set(self.level + 1, now)
        elif self.ema < self.budget_ms * self.headroom:
            self._over_since = None
            self._under_since = self._under_since if self._under_since is not None else now
            if (now - self._under_since >= self.up_after and self.level > 0
                    and now - self._changed_at >= self.min_interval):
                return self._set(self.level - 1, now)
        else:                       # comfortable middle band: hold steady
            self._over_since = self._under_since = None
        return False

    def _set(self, level, now):
        self.level, self._changed_at = level, now
        self._over_since = self._under_since = None
        return True


HAND_CONNECTIONS = np.array([
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)])


def draw_hand_fast(frame, hand_landmarks, line_color=(0, 255, 0), dot_color=(0, 0, 255)):
    """Hand skeleton via two OpenCV calls + 21 dots (~0.2 ms). The stock
    mediapipe draw_landmarks walks landmarks in Python and builds drawing
    specs per call, which is several times slower per hand."""
    h, w = frame.shape[:2]
    lm = hand_landmarks.landmark
    pts = (np.array([(p.x, p.y) for p in lm]) * (w, h)).astype(np.int32)
    cv2.polylines(frame, pts[HAND_CONNECTIONS], False, line_color, 2)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 3, dot_color, -1)
