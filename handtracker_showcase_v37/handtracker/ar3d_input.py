"""
ar3d_input.py
-------------
Turns hand tracking into 3D-object manipulation. Three pieces:

  HandSlots   - keeps "hand A" and "hand B" stable across frames (MediaPipe
                returns hands in arbitrary order), so two-hand gestures
                don't swap hands mid-grab.
  HandPointer - one hand's screen point, smoothed pinch state, apparent
                size (for push/pull depth). Same smoothing/pinch rules the
                rest of showcase.py uses, but one instance per hand.
  AR3DController - the state machine:

      1 hand pinch on object body ........ MOVE (x/y; hand nearer/further
                                            camera = object nearer/further)
      1 hand pinch on a bar / surface .... SELECT a portion (drag to extend)
      1 hand pinch INSIDE the selection .. STRETCH it: drag left/right = width,
                                            up/down = height
      tap the body (no drag) ............. clear the selection
      2 hands pinch, nothing selected .... SCALE (hand distance), ROLL (hand
                                            angle), YAW (one hand nearer)
      2 hands pinch, portion selected .... STRETCH it: hands apart = wider,
                                            both hands up/down = taller/shorter
      pinch a "3D" palette tile .......... pick up a new object, drop it
      fist over object / two-finger ...... remove / refresh (same as 2D charts) -
                                            ONLY when no pinch is active or was
                                            in the last 0.8 s (a hard two-hand
                                            pinch curls your fingers and can
                                            look like a fist!). Undo: scene.undo_remove()

The controller is pure logic over HandState values (no Tk, no camera), so
it is unit-tested with synthetic input in tests/test_ar3d.py.
"""
import math
from dataclasses import dataclass

import config
from ar3d import (SIZE_MAX, SIZE_MIN, Z_MAX, Z_MIN, make_object, orthonormalize,
                  rot_y, rot_z)

try:
    import function_graphs
    _FUNC_GRAPHS_OK = True
except Exception as _e:  # pragma: no cover
    print(f"[import] AR function graphs unavailable: {_e}")
    _FUNC_GRAPHS_OK = False

IDLE, GHOST, GRAB1, SELECT, STRETCH1, GRAB2 = "idle", "ghost", "grab1", "select", "stretch1", "grab2"


def _cfg(name, default):
    return getattr(config, name, default)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


@dataclass
class HandState:
    present: bool = False
    pt: tuple = (0.0, 0.0)     # screen px (fingertip, smoothed)
    pinching: bool = False
    size: float = 0.0          # smoothed apparent hand size, camera px
    gesture: str = None
    approach: float = 0.0      # 0 = fingers open .. 1 = pinched (cursor feedback)


class HandSlots:
    """Sticky hand identity. Slot 0 is the 'primary' hand (drives the 2D
    code path in showcase.py); slot 1 is the second hand. A lone hand goes
    to slot 0; when two appear together the right-most (in the mirrored
    frame, so usually your right hand) becomes slot 0. After that, hands
    keep their slot by staying nearest their last position."""

    def __init__(self, hold_seconds=1.0):
        self.hold = hold_seconds
        self.last = [None, None]          # (x, y, time)

    def _live(self, now):
        return [l is not None and now - l[2] <= self.hold for l in self.last]

    def _dist(self, slot, p):
        l = self.last[slot]
        return math.hypot(l[0] - p[0], l[1] - p[1])

    def assign(self, hands, now):
        out = [None, None]
        pts = [(h.landmark[0].x, h.landmark[0].y) for h in hands[:2]]
        live = self._live(now)
        if len(pts) == 1:
            p = pts[0]
            if live[0] and live[1]:
                k = 0 if self._dist(0, p) <= self._dist(1, p) else 1
            elif live[1] and not live[0]:
                k = 1 if self._dist(1, p) < 0.3 else 0
            else:
                k = 0
            out[k] = hands[0]
            self.last[k] = (*p, now)
        elif len(pts) == 2:
            if live[0] and live[1]:
                straight = self._dist(0, pts[0]) + self._dist(1, pts[1])
                swapped = self._dist(0, pts[1]) + self._dist(1, pts[0])
                order = (0, 1) if straight <= swapped else (1, 0)
            elif live[0]:
                order = (0, 1) if self._dist(0, pts[0]) <= self._dist(0, pts[1]) else (1, 0)
            elif live[1]:
                order = (1, 0) if self._dist(1, pts[0]) <= self._dist(1, pts[1]) else (0, 1)
            else:
                order = (0, 1) if pts[0][0] >= pts[1][0] else (1, 0)
            for hand, p, k in zip(hands[:2], pts, order):
                out[k] = hand
                self.last[k] = (*p, now)
        return out


from pinch_stable import PinchStableCursor  # noqa: E402  (moved; re-exported)


class HandPointer:
    """Per-hand smoothed pointer. `mapper` converts a normalized camera
    point to a screen pixel point (desktop_control.map_to_screen)."""

    def __init__(self, mapper, frame_w, frame_h):
        from gestures import PointSmoother, ScalarSmoother
        self._PointSmoother, self._ScalarSmoother = PointSmoother, ScalarSmoother
        self.mapper, self.fw, self.fh = mapper, frame_w, frame_h
        self._make()

    def _make(self):
        self.idx = self._PointSmoother(min_cutoff=config.CURSOR_SMOOTHING_MIN_CUTOFF,
                                       beta=config.CURSOR_SMOOTHING_BETA)
        self.palm = self._PointSmoother(min_cutoff=config.CURSOR_SMOOTHING_MIN_CUTOFF,
                                        beta=config.CURSOR_SMOOTHING_BETA)
        self.cursor = PinchStableCursor()
        self.scale = self._ScalarSmoother(alpha=0.12)      # pinch-ratio divisor
        self.size = self._ScalarSmoother(alpha=0.25)       # depth cue
        self.ratio = self._ScalarSmoother(alpha=0.75)
        self.pinching = False

    def reset(self):
        self._make()

    def update(self, lm, now):
        if lm is None:
            self.reset()
            return HandState(present=False)
        from gestures import hand_scale_px, is_pinching_from_ratio, pinch_ratio
        from gesture_engine import classify_static_gesture, is_strict_fist
        L = lm.landmark
        tip = self.idx.update((L[8].x, L[8].y), now)
        raw_scale = hand_scale_px(lm, self.fw, self.fh)
        s = self.scale.update(raw_scale)
        r = self.ratio.update(pinch_ratio((L[8].x * self.fw, L[8].y * self.fh),
                                          (L[4].x * self.fw, L[4].y * self.fh), s))
        gesture = classify_static_gesture(lm)
        pinching = is_pinching_from_ratio(r, self.pinching)
        # "a closing fist is not a pinch" - stops a fist from being
        # misread as the START of a pinch/drag (self.pinching here is
        # still LAST frame's value, read before being overwritten
        # below). Deliberately does NOT apply once a drag is already in
        # progress (self.pinching already True): the destructive-action
        # path (remove_at/refresh_at in ar3d_input.AR3DController) has
        # its own cooldown precisely because squeezing a two-hand pinch
        # tight - or a brief tracking blip mid-drag - naturally curls
        # the other fingers too and can get misclassified as a fist for
        # a stray frame, but THIS check had no equivalent tolerance: it
        # force-cleared `pinching` unconditionally, which every one of
        # _grab1/_select/_stretch1/_grab2's own "if not pin[i]: end the
        # drag" checks then treated as a genuine release - so a drag or
        # a two-hand resize could silently drop itself mid-gesture for
        # no reason visible to the person doing it. Only suppressing a
        # fresh pinch start (not an ongoing one) keeps the original
        # protection while fixing that.
        if (gesture == "fist" or is_strict_fist(lm)) and not self.pinching:
            pinching = False
        self.pinching = pinching
        # Palm centre (wrist + 4 knuckles) doesn't move when fingers curl.
        palm = self.palm.update((sum(L[i].x for i in (0, 5, 9, 13, 17)) / 5,
                                 sum(L[i].y for i in (0, 5, 9, 13, 17)) / 5), now)
        stable = self.cursor.update(tip, palm, r, now)
        approach = 1.0 if pinching else self.cursor.approach(r)
        return HandState(True, tuple(self.mapper(stable)), pinching, self.size.update(raw_scale),
                         gesture, approach)


def mid0_of(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


class AR3DController:
    def __init__(self, scene):
        self.scene = scene
        self.mode = IDLE
        self.gi = None               # the hand driving GHOST/GRAB1/SELECT
        self.obj = None
        self.prev = [False, False]
        self.consumed = [False, False]   # pinch already used (e.g. palette click)
        self.g = {}
        self.primary_idx = 0
        self.last_status = ""
        self._hint = None
        self._now = None
        self.dt = 1 / 30
        self._last_active = -1e9

    # -- public API used by showcase.py ---------------------------------------
    def claimed(self, idx):
        """True if hand `idx`'s current pinch belongs to the 3D layer, so
        the 2D chart logic must ignore it."""
        return self.consumed[idx] or self.mode == GRAB2 or \
            (self.mode in (GHOST, GRAB1, SELECT, STRETCH1) and self.gi == idx)

    def step(self, hands, now, ui=None):
        """hands: [HandState (slot 0), HandState (slot 1)]. ui: the AROverlay
        (palette/tile hit-testing). Returns True if the PRIMARY hand's pinch
        is claimed by the 3D layer."""
        pin = [h.present and h.pinching for h in hands]
        self.dt = _clamp(now - self._now, 1 / 120, 0.1) if self._now is not None else 1 / 30
        self._now = now
        if any(pin) or self.mode != IDLE:
            self._last_active = now
        self.primary_idx = 0 if hands[0].present else 1
        for i in (0, 1):
            if not pin[i]:
                self.consumed[i] = False
        getattr(self, "_" + self.mode)(hands, pin, now, ui)
        self.scene.grabbed = {self.obj.id} if self.mode in (GRAB1, STRETCH1, GRAB2) and self.obj else set()
        self.scene.calm = {self.obj.id} if self.mode == SELECT and self.obj else set()
        self._hover(hands, pin)
        self._hud(hands, pin)
        self.prev = pin
        return self.claimed(self.primary_idx) and pin[self.primary_idx]

    def reset(self):
        """Drop any in-progress gesture (mode switched away from AR, overlay
        hidden). Objects keep their state; nothing is committed or lost."""
        self._end()
        self.prev, self.consumed = [False, False], [False, False]
        self.scene.grabbed, self.scene.calm = set(), set()
        self.scene.hover, self.scene.hud = {}, None

    def _destructive_ok(self, now):
        """A fist/two-finger over an object may only delete/refresh it when
        NO pinch is active or was in the last DESTRUCTIVE_COOLDOWN seconds.
        Squeezing a two-hand pinch curls the other fingers and the gesture
        classifier can call that a fist - that used to delete the object
        mid-resize (and two-finger reset the stretch you were making)."""
        if now is None:
            return True
        return self.mode == IDLE and not any(self.prev) and \
            now - self._last_active > _cfg("AR3D_DESTRUCTIVE_COOLDOWN", 0.8)

    def delete_at(self, pt, now=None):
        """Fist-delete with an honest result: "removed" (object gone),
        "blocked" (an object IS under the point but a pinch was active/
        recent, see _destructive_ok - the caller should retry shortly
        rather than treat the gesture as spent), or None (nothing there).
        remove_at() below keeps the old bool contract on top of this."""
        hit = self.scene.object_at(*pt)
        if not hit:
            return None
        if self._destructive_ok(now):
            self.scene.remove(hit[0])
            return "removed"
        return "blocked"

    def remove_at(self, pt, now=None):
        """True if a 3D object is under the point (whether or not it was
        removed - so the 2D path never acts on the same gesture)."""
        return self.delete_at(pt, now) is not None

    def refresh_at(self, pt, now=None):
        hit = self.scene.object_at(*pt)
        if not hit:
            return False
        if self._destructive_ok(now):
            hit[0].refresh()
            self.scene.dirty = True
        return True

    # -- state handlers ----------------------------------------------------------
    def _end(self):
        if self.mode == GHOST:
            self.scene.ghost = None
        self.mode, self.gi, self.obj, self.g = IDLE, None, None, {}

    def _idle(self, hands, pin, now, ui):
        for i in (0, 1):
            if pin[i] and not self.prev[i] and not self.consumed[i]:
                if self._begin(i, hands[i], now, ui):
                    break

    def _begin(self, i, h, now, ui):
        pt = h.pt
        kind = ui.palette_kind_at(pt) if ui is not None else None
        if kind == "shapes3d":
            ui.toggle_shape_browser()
            self.consumed[i] = True
            return True
        if kind == "functions":
            ui.toggle_function_browser()
            self.consumed[i] = True
            return True
        if kind == "dense":
            # A straight toggle, not an opener - no browser, no ghost,
            # just flips Scene3D.dense_mode (which immediately restyles
            # every current AND future Surface3D - see toggle_dense's own
            # comment) the instant the pinch lands, same "point, pinch,
            # done" feel as every other palette button's pick-up gesture,
            # just without anything to drag afterwards.
            self.scene.toggle_dense()
            self.consumed[i] = True
            return True
        tile = ui.shape_tile_at(pt) if ui is not None else None
        if tile:
            obj = make_object(tile, *self.scene.clamp_xy(*pt))
            if obj is not None:
                self.scene.ghost = obj
                self.mode, self.gi = GHOST, i
                ui.shapes_open = False
                return True
        func_tile = ui.function_tile_at(pt) if (ui is not None and _FUNC_GRAPHS_OK) else None
        if func_tile:
            ui.functions_open = False
            if func_tile.get("error"):
                # Belt-and-braces: _scan_functions already tried
                # compiling every tile up front so a broken one can be
                # drawn differently (see ar_overlay's draw step) - this
                # is the "ignore it anyway if someone pinches it" guard,
                # same spirit as start_ghost_from_file's try/except.
                print(f"[WARN] AR: {func_tile['name']} has an invalid function "
                      f"({func_tile['error']}) - fix the .txt file and reopen "
                      f"the Functions browser")
                return True
            try:
                fn = function_graphs.read_function_file(func_tile["path"])
                obj = make_object("surface3d", *self.scene.clamp_xy(*pt),
                                   expr=fn.source, expr_fn=fn)
            except function_graphs.FunctionParseError as e:
                print(f"[WARN] AR: couldn't load {func_tile['path']} ({e})")
                return True
            self.scene.ghost = obj
            self.mode, self.gi = GHOST, i
            return True
        if kind is not None:
            return False                       # another palette button: 2D's job
        hit = self.scene.object_at(*pt)
        if not hit:
            return False
        obj, part = hit
        self.obj, self.gi = obj, i
        if obj.supports_selection and part[0] in ("bar", "cell"):
            if obj.has_selection and obj.in_selection(part):
                obj.stretch_begin()                    # the selection IS the handle
                self.mode = STRETCH1
                self.g = dict(start=h.pt, size0=obj.size)
            else:
                self.g = dict(prev=obj.selection, start=h.pt, moved=False)
                obj.select_start(part)
                self.mode = SELECT
            self.scene.dirty = True
        else:
            self._start_grab1(i, h, obj, now, tap=True)
        return True

    def _over(self, pt, obj):
        return obj.pick(pt[0], pt[1], pad=40.0) is not None

    def _ghost(self, hands, pin, now, ui):
        h = self.scene.ghost, hands[self.gi]
        ghost, hand = h
        if not hand.present:
            return self._end()
        if pin[self.gi]:
            ghost.pos[0], ghost.pos[1] = self.scene.clamp_xy(*hand.pt)
            return
        if ui is not None and ui.in_palette_zone(hand.pt):
            return self._end()                 # dropped back on the palette = cancel
        ghost.pos[0], ghost.pos[1] = self.scene.clamp_xy(*hand.pt)
        self.scene.add(ghost)
        self.scene.ghost = None
        self._end()

    # one-hand move -----------------------------------------------------------
    def _start_grab1(self, i, h, obj, now, tap):
        obj.vx = obj.vy = 0.0
        obj._last_move = None
        self.mode, self.gi, self.obj = GRAB1, i, obj
        self.g = dict(offset=(obj.pos[0] - h.pt[0], obj.pos[1] - h.pt[1]), z0=float(obj.pos[2]),
                      hs0=h.size, start=h.pt, t0=now, moved=False, tap=tap)

    def _k(self, tau):
        """Per-frame blend factor for an exponential ease with time
        constant `tau` (frame-rate independent)."""
        return 1.0 - math.exp(-self.dt / max(tau, 1e-6))

    def _depth(self, ratio_minus_1, z0):
        if not _cfg("AR3D_DEPTH_FROM_HAND_SIZE", True):
            return z0
        dead = _cfg("AR3D_DEPTH_DEADZONE", 0.05)
        r = 0.0 if abs(ratio_minus_1) < dead else ratio_minus_1 - math.copysign(dead, ratio_minus_1)
        return _clamp(z0 - _cfg("AR3D_DEPTH_GAIN", 700.0) * r, Z_MIN, Z_MAX)

    def _ease_depth(self, obj, target):
        obj.pos[2] += (target - obj.pos[2]) * self._k(0.10)

    def _grab1(self, hands, pin, now, ui):
        h, obj, g = hands[self.gi], self.obj, self.g
        if not pin[self.gi]:
            if g["tap"] and not g["moved"] and now - g["t0"] < _cfg("AR3D_TAP_MAX_SECONDS", 0.35) \
                    and obj.has_selection:
                obj.select_clear()
                self.scene.dirty = True
            return self._end()
        other = 1 - self.gi
        if pin[other] and self._over(hands[other].pt, obj):
            return self._start_grab2(obj, hands, now)
        nx, ny = self.scene.clamp_xy(h.pt[0] + g["offset"][0], h.pt[1] + g["offset"][1])
        if obj._last_move is not None:
            dt = _clamp(now - obj._last_move, 1e-3, 0.2)
            obj.vx = _clamp((nx - obj.pos[0]) / dt, -3000, 3000)
            obj.vy = _clamp((ny - obj.pos[1]) / dt, -3000, 3000)
        obj.pos[0], obj.pos[1] = nx, ny
        obj._last_move = now
        if math.hypot(h.pt[0] - g["start"][0], h.pt[1] - g["start"][1]) > _cfg("AR3D_TAP_MOVE_PX", 22):
            g["moved"] = True
        if g["hs0"] > 0 and h.size > 0:
            self._ease_depth(obj, self._depth(h.size / g["hs0"] - 1.0, g["z0"]))
        self.scene.dirty = True

    # select a portion ---------------------------------------------------------
    def _select(self, hands, pin, now, ui):
        h, obj, g = hands[self.gi], self.obj, self.g
        if not pin[self.gi]:
            return self._end()                 # selection persists
        if math.hypot(h.pt[0] - g["start"][0], h.pt[1] - g["start"][1]) > _cfg("AR3D_TAP_MOVE_PX", 22):
            g["moved"] = True
        other = 1 - self.gi
        if pin[other] and self._over(hands[other].pt, obj):
            if not g["moved"]:
                # Both hands landed together: the first one wasn't starting a
                # new selection, it was the first half of a two-hand gesture.
                # Put back what was selected before (or nothing -> whole-object
                # scale) instead of destroying it.
                obj.selection = g["prev"]
            return self._start_grab2(obj, hands, now)
        obj.select_update(obj.nearest_part(*h.pt))
        self.scene.dirty = True

    # one-hand stretch of the selection ---------------------------------------------
    def _stretch1(self, hands, pin, now, ui):
        h, obj, g = hands[self.gi], self.obj, self.g
        if not pin[self.gi]:
            return self._end()
        other = 1 - self.gi
        if pin[other] and self._over(hands[other].pt, obj):
            return self._start_grab2(obj, hands, now)
        dead = 8.0
        soft = lambda v: 0.0 if abs(v) < dead else v - math.copysign(dead, v)
        dx, dy = soft(h.pt[0] - g["start"][0]), soft(h.pt[1] - g["start"][1])
        sx = _clamp(2.0 ** (dx / 200.0), 0.25, 4.0)      # 200 px right = 2x wider
        sy = _clamp(2.0 ** (-dy / 150.0), 0.25, 3.0)     # 150 px up = 2x taller
        obj.stretch_apply(sx, sy)
        self._grow_floor_for_stretch(obj, g, sx, sy)
        self.last_status = f"WIDTH x{sx:.2f}   HEIGHT x{sy:.2f}"
        self.scene.dirty = True

    def _grow_floor_for_stretch(self, obj, g, sx, sy):
        """Stretching a selected part (BarChart3D.stretch_apply /
        Surface3D.stretch_apply, above) deliberately keeps the object's
        OVERALL footprint fixed and only reshuffles widths inside it -
        "pulling and squeezing a piece of putty", not resizing the whole
        object. That's the intended feel, but it falls apart on a small
        object: cw/rw's multiplier is clamped up to 6x, but 6x of a
        sliver of a 45px-wide floor is still only a few pixels, so
        pulling a selection on a small object visibly does nothing even
        though the underlying numbers did change - which reads as the
        gesture just not working, not as "the object happens to be
        small". Letting the floor itself grow a little when a stretch is
        pulling OUTWARD (never shrink it - compressing a part shouldn't
        shrink the whole object) gives a small object room to actually
        show the change. Growth is dampened (sqrt) relative to the
        stretch factor itself so a small selection tug doesn't balloon
        the whole object - it's headroom for visibility, not a second
        way to scale the object."""
        size0 = g.get("size0")
        if size0 is None:
            return
        growth = max(sx, sy, 1.0) ** 0.5
        if growth <= 1.0:
            return
        want = _clamp(size0 * growth, SIZE_MIN, SIZE_MAX)
        obj.size = _clamp(self.scene.fit_size(obj, want), SIZE_MIN * 0.5, SIZE_MAX)

    # two hands ------------------------------------------------------------------
    def _start_grab2(self, obj, hands, now):
        li, ri = (0, 1) if hands[0].pt[0] <= hands[1].pt[0] else (1, 0)
        pl, pr = hands[li].pt, hands[ri].pt
        dx, dy = pr[0] - pl[0], pr[1] - pl[1]
        obj.vx = obj.vy = 0.0
        stretch = obj.supports_selection and obj.has_selection
        if stretch:
            obj.stretch_begin()
        self.mode, self.obj, self.gi = GRAB2, obj, None
        self.g = dict(li=li, ri=ri, d0=max(math.hypot(dx, dy), 60.0), dx0=max(abs(dx), 60.0),
                      dy0=abs(dy), last_ang=math.atan2(dy, dx), roll=0.0,
                      mid0=((pl[0] + pr[0]) / 2, (pl[1] + pr[1]) / 2),
                      pos0=obj.pos.copy(), rot0=obj.rot.copy(), size0=obj.size,
                      hsl=hands[li].size, hsr=hands[ri].size, stretch=stretch,
                      # smoothed copies of the controls (start where the object is: no jump)
                      s_scale=1.0, s_roll=0.0, s_yaw=0.0, s_mid=mid0_of(pl, pr))

    def _grab2(self, hands, pin, now, ui):
        g, obj = self.g, self.obj
        if not (pin[0] and pin[1]):
            rem = [i for i in (0, 1) if pin[i]]
            if rem and self._over(hands[rem[0]].pt, obj):
                if g["stretch"]:
                    self._end()            # don't turn a half-finished stretch into a drag
                else:
                    self._start_grab1(rem[0], hands[rem[0]], obj, now, tap=False)
                return
            return self._end()
        hl, hr = hands[g["li"]], hands[g["ri"]]
        dx, dy = hr.pt[0] - hl.pt[0], hr.pt[1] - hl.pt[1]
        d = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        g["roll"] += (ang - g["last_ang"] + math.pi) % (2 * math.pi) - math.pi
        g["last_ang"] = ang
        self.scene.dirty = True
        if g["stretch"]:
            # hands apart = wider (any direction); both hands up/down = taller/shorter
            sx = _clamp(d / g["d0"], 0.25, 4.0)
            dm = (hl.pt[1] + hr.pt[1]) / 2 - g["mid0"][1]
            dm = 0.0 if abs(dm) < 8 else dm - math.copysign(8, dm)
            sy = _clamp(2.0 ** (-dm / 150.0), 0.25, 3.0)
            obj.stretch_apply(sx, sy)
            self._grow_floor_for_stretch(obj, g, sx, sy)
            self.last_status = f"WIDTH x{sx:.2f}   HEIGHT x{sy:.2f}"
            return
        k = self._k(_cfg("AR3D_SMOOTH_TAU", 0.06))
        g["s_scale"] += (d / g["d0"] - g["s_scale"]) * k
        g["s_roll"] += (g["roll"] - g["s_roll"]) * k
        want = _clamp(g["size0"] * g["s_scale"], SIZE_MIN, SIZE_MAX)
        obj.size = _clamp(self.scene.fit_size(obj, want), SIZE_MIN * 0.5, SIZE_MAX)
        rl = hl.size / g["hsl"] - 1.0 if g["hsl"] > 0 and hl.size > 0 else 0.0
        rr = hr.size / g["hsr"] - 1.0 if g["hsr"] > 0 and hr.size > 0 else 0.0
        diff = rr - rl
        diff = 0.0 if abs(diff) < 0.04 else diff - math.copysign(0.04, diff)
        yaw_t = _clamp(_cfg("AR3D_YAW_GAIN", 2.5) * diff, -1.4, 1.4)
        g["s_yaw"] += (yaw_t - g["s_yaw"]) * k
        yaw = g["s_yaw"]
        obj.rot = orthonormalize(rot_y(yaw) @ rot_z(g["s_roll"]) @ g["rot0"])
        mid = mid0_of(hl.pt, hr.pt)
        g["s_mid"] = (g["s_mid"][0] + (mid[0] - g["s_mid"][0]) * k * 2,     # position stays snappier
                      g["s_mid"][1] + (mid[1] - g["s_mid"][1]) * k * 2)
        obj.pos[0], obj.pos[1] = self.scene.clamp_xy(g["pos0"][0] + g["s_mid"][0] - g["mid0"][0],
                                                    g["pos0"][1] + g["s_mid"][1] - g["mid0"][1])
        self._ease_depth(obj, self._depth((rl + rr) / 2, float(g["pos0"][2])))
        self.last_status = (f"SCALE x{obj.size / g['size0']:.2f}  ROLL {math.degrees(g['s_roll']):.0f}"
                            f"  YAW {math.degrees(yaw):.0f}")

    # -- feedback -----------------------------------------------------------------
    def _hover(self, hands, pin):
        self.scene.hover = {}
        hint = None
        for i, h in enumerate(hands):
            if not h.present or pin[i] or self.mode != IDLE:
                continue
            hit = self.scene.object_at(*h.pt)
            if hit:
                self.scene.hover[i] = hit
                if hint is None:
                    obj, part = hit
                    data = obj.supports_selection and part[0] in ("bar", "cell")
                    if data and obj.in_selection(part):
                        text = "PINCH+DRAG: STRETCH  (L/R width, U/D height)"
                    elif data:
                        text = "PINCH+DRAG: SELECT"
                    else:
                        text = "PINCH: GRAB  |  2 HANDS: RESIZE"
                    hint = (h.pt, text)
        self._hint = hint

    def _hud(self, hands, pin):
        cursors = [(h.pt, pin[i], "#c58cff", h.approach) for i, h in enumerate(hands)
                   if h.present and i != self.primary_idx]
        hud = {"cursors": cursors, "hint": getattr(self, "_hint", None)}
        if self.mode == GRAB2:
            hud["tether"] = (hands[0].pt, hands[1].pt)
            hud["text"] = self.last_status
        elif self.mode == SELECT and self.obj is not None:
            hud["hint"] = (hands[self.gi].pt, "SELECTING - release, then pinch inside it to stretch")
        elif self.mode == STRETCH1:
            hud["hint"] = (hands[self.gi].pt, self.last_status)
        self.scene.hud = hud
