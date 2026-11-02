"""
tracker_sched.py
----------------
Cheap two-hand tracking for AR mode.

THE PROBLEM: MediaPipe Hands only skips its palm detector when it is already
tracking as many hands as `max_num_hands`. A 2-hand tracker with ONE hand in
view therefore runs full palm detection on EVERY frame, forever, hunting for
the second hand - roughly a 40% per-frame cost, paid exactly when you bring
your hand in. A 1-hand tracker tracking 1 hand is much cheaper.

THE FIX: keep both and schedule them.

  single state  (0 or 1 hand)   use the cheap 1-hand tracker every frame, and
                                every `probe_every`-th frame (only while a hand
                                is visible) run the 2-hand tracker INSTEAD, to
                                look for a second hand. Costs one detection
                                pass per probe, not per frame.
  dual state    (2 hands seen)  use the 2-hand tracker every frame (with both
                                hands tracked it skips detection too, so this
                                is cheap). After `back_after` consecutive
                                frames with fewer than 2 hands, return to the
                                single state.

Trade-off: a second hand is noticed within `probe_every` frames (6 frames =
0.2 s at 30 fps) of entering view, instead of instantly. A miss on a probe
frame while a hand was tracked reuses the previous frame's result once, so a
probe can't make your hand flicker out.

Pure logic over two objects with .process(rgb) -> result.multi_hand_landmarks,
so it is unit-tested with fakes (tests/test_perf.py).
"""
from collections import Counter

SINGLE, DUAL = "single", "dual"


def _count(res):
    return len(getattr(res, "multi_hand_landmarks", None) or [])


class TwoHandScheduler:
    def __init__(self, single, dual, probe_every=6, back_after=12):
        self.single, self.dual = single, dual
        self.probe_every = max(2, int(probe_every))
        self.back_after = max(1, int(back_after))
        self.state = SINGLE
        self._n = 0
        self._streak = 0          # consecutive dual-state frames with < 2 hands
        self._last = None
        self.stats = Counter()

    def reset(self):
        self.state, self._n, self._streak, self._last = SINGLE, 0, 0, None

    def process(self, rgb):
        if self.state == SINGLE:
            self._n += 1
            tracking = _count(self._last) >= 1
            if tracking and self._n % self.probe_every == 0:
                res = self.dual.process(rgb)
                self.stats["probe"] += 1
                n = _count(res)
                if n >= 2:
                    self.state, self._streak = DUAL, 0
                    self.stats["to_dual"] += 1
                elif n == 0:
                    res = self._last          # transient miss on a probe frame: don't drop the hand
            else:
                res = self.single.process(rgb)
                self.stats["single"] += 1
            self._last = res
            return res
        res = self.dual.process(rgb)
        self.stats["dual"] += 1
        self._streak = 0 if _count(res) >= 2 else self._streak + 1
        if self._streak >= self.back_after:
            self.state = SINGLE
            self.stats["to_single"] += 1
        self._last = res
        return res
