"""
pinch_stable.py
----------------
PinchStableCursor, shared by the AR hand pointer (ar3d_input.HandPointer)
and the main overlay's always-on reticle / drawing / Whack-a-Mole input
(showcase.py), so every mode points the same way.

Moved out of ar3d_input.py unchanged so showcase.py can use it without
pulling in the AR code. ar3d_input still re-exports the name.
"""
import config


def _cfg(name, default):
    return getattr(config, name, default)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class PinchStableCursor:
    """Cursor point that doesn't sag when you pinch.

    The index fingertip physically drops toward the thumb as a pinch
    closes, so a fingertip cursor ends up BELOW what you were aiming at -
    and by the time the pinch is detected, most of that drop has already
    happened. So:

      * While fingers are open we remember the fingertip-to-PALM offset
        (palm centre doesn't move when fingers curl) for the last ~0.4 s.
      * When the pinch ratio falls under `on`, we lock to the offset from
        the MOST OPEN moment in that window - i.e. where you were aiming
        before the fingers started closing - and follow only the palm from
        then on. A short ease (`lock_blend`) takes the cursor there so it
        doesn't snap.
      * Above `off` it unlocks and eases back to the live fingertip over
        `blend` seconds.
    Inputs/outputs are normalized camera coordinates."""

    HISTORY_SECONDS = 0.4

    def __init__(self, on=None, off=None, blend=None, lock_blend=0.08):
        self.on = _cfg("AR_PINCH_LOCK_ON", 0.90) if on is None else on
        self.off = _cfg("AR_PINCH_LOCK_OFF", 1.05) if off is None else off
        self.blend = _cfg("AR_PINCH_RELEASE_BLEND", 0.15) if blend is None else blend
        self.lock_blend = lock_blend
        self.locked = False
        self.offset = (0.0, 0.0)
        self.last = None
        self._hist = []              # (t, ratio, offset_x, offset_y) while unlocked
        self._from = None            # ease start (either transition)
        self._t0 = None
        self._eq = None              # "lock" | "release"

    @staticmethod
    def _smooth(k):
        k = _clamp(k, 0.0, 1.0)
        return k * k * (3 - 2 * k)

    def update(self, tip, palm, ratio, now):
        if not self.locked:
            self._hist.append((now, ratio, tip[0] - palm[0], tip[1] - palm[1]))
            while self._hist and now - self._hist[0][0] > self.HISTORY_SECONDS:
                self._hist.pop(0)
        if not self.locked and ratio < self.on:
            best = max(self._hist, key=lambda h: h[1])          # most open recent sample
            self.locked = True
            self.offset = (best[2], best[3])
            self._from, self._t0, self._eq = self.last, now, "lock"
        elif self.locked and ratio > self.off:
            self.locked = False
            self._hist.clear()
            self._from, self._t0, self._eq = self.last, now, "release"
        target = ((palm[0] + self.offset[0], palm[1] + self.offset[1]) if self.locked else tip)
        out = target
        if self._t0 is not None and self._from is not None:
            dur = self.lock_blend if self._eq == "lock" else self.blend
            k = (now - self._t0) / max(dur, 1e-6)
            if k >= 1.0:
                self._t0 = None
            else:
                k = self._smooth(k)
                out = (self._from[0] + (target[0] - self._from[0]) * k,
                       self._from[1] + (target[1] - self._from[1]) * k)
        self.last = out
        return out

    def approach(self, ratio):
        """0..1 progress from 'open' (LOCK_OFF) toward a full pinch (~0.42)."""
        return _clamp((self.off - ratio) / (self.off - 0.42), 0.0, 1.0)
