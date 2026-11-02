"""Run:  python tests/test_ar3d.py     (or: pytest tests/test_ar3d.py)
No display, camera or Tk needed."""
import json
import math
import os
import random
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import ar3d
from ar3d import BarChart3D, Object3D, Scene3D, Shape3D, Surface3D, rot_x, rot_y
from ar3d_input import (GRAB1, GRAB2, GHOST, IDLE, SELECT, AR3DController, HandPointer,
                        HandSlots, HandState)
from pil_canvas import PILCanvas


class CountCanvas:
    def __init__(self):
        self.polys = 0
        self.lines = 0
        self.texts = []

    def create_polygon(self, *a, **k): self.polys += 1
    def create_line(self, *a, **k): self.lines += 1
    def create_rectangle(self, *a, **k): pass
    def create_oval(self, *a, **k): pass
    def create_text(self, x, y, text="", **k): self.texts.append(text)


def scene_with(*objs, bounds=(1000, 700), settled=True):
    """settled=True skips the "pop in" entrance animation (Object3D.
    start_pop_in) so tests that probe geometry/picking right after
    building a scene see objects at their real size - the animation is
    a ~250 ms real-time effect that has nothing to do with what those
    tests check, and synthetic hands here only advance 1/30 s per tick.
    test_objects_pop_in_when_placed covers the animation itself."""
    sc = Scene3D(bounds)
    for o in objs:
        sc.add(o)
        if settled:
            o._pop = 1.0
    sc.step(0.03)
    return sc


# ---------------------------------------------------------------- engine ---
def test_rotations_are_proper():
    for R in (rot_x(.7), rot_y(-1.1), rot_x(.4) @ rot_y(.9)):
        assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-9
    d = ar3d.orthonormalize(rot_x(.5) * 1.02 + 0.01)
    assert np.allclose(d @ d.T, np.eye(3), atol=1e-9)


def test_backface_culling_cube_shows_three_faces():
    c = Shape3D("cube", 300, 300)
    c.spin = 0
    c.rot = rot_x(0.5) @ rot_y(0.6)
    sc = scene_with(c)
    cv = CountCanvas()
    ar3d.render_object(cv, c)
    assert cv.polys == 3, cv.polys        # generic orientation: exactly 3 of 6


def test_perspective_far_is_smaller_near_is_bigger():
    def width(z):
        o = Shape3D("cube", 400, 300, z=z)
        o.spin = 0
        scene_with(o)
        return o.geom.sx.max() - o.geom.sx.min()
    assert width(900) < 0.6 * width(0) < 0.6 * width(-300) and width(-300) > width(0)


def test_bar_pick_and_body_pick():
    b = BarChart3D(400, 350, data=[.5] * 5)
    b.sway = 0
    scene_with(b)
    g = b.geom
    for i in range(5):
        idx = next(ix for k, ix in g.mesh.parts if k == ("bar", i))
        cx, cy = g.sx[idx].mean(), g.sy[idx].mean()
        assert b.pick(cx, cy) == ("bar", i), (i, b.pick(cx, cy))
    body = g.mesh.body                      # front edge midpoint of the floor plate
    fx, fy = (g.sx[body[0]] + g.sx[body[1]]) / 2, (g.sy[body[0]] + g.sy[body[1]]) / 2
    assert b.pick(fx, fy) == ("body", None)
    assert b.pick(5, 5) is None


def test_surface_pick_gives_cell_inside_and_body_on_frame():
    s = Surface3D(400, 350)
    s.sway = 0
    scene_with(s)
    g = s.geom
    cx, cy = g.sx[s._vi[10, 10]], g.sy[s._vi[10, 10]]
    assert s.pick(cx, cy) == ("cell", (10, 10))
    body = g.mesh.body
    fx, fy = (g.sx[body[0]] + g.sx[body[1]]) / 2, (g.sy[body[0]] + g.sy[body[1]]) / 2
    assert s.pick(fx, fy)[0] == "body"


def test_bar_stretch_keeps_total_width_and_scales_selection():
    b = BarChart3D(400, 350, data=[.5] * 7)
    b.select_start(("bar", 2))
    b.select_update(("bar", 4))
    assert b.selection == (2, 4)
    b.stretch_begin()
    b.stretch_apply(2.0, 1.5)
    b.wx_d, b.hy_d = list(b.wx), list(b.hy)
    slots, edges = b._layout()
    assert abs(slots.sum() - b.W) < 1e-9                 # footprint constant
    assert abs(slots[2] / slots[0] - 2.0) < 1e-9         # selected twice as wide
    assert abs(slots[0] - slots[6]) < 1e-9               # others untouched relative
    assert b.hy[3] == 1.5 and b.hy[0] == 1.0
    b.stretch_apply(0.5, 1.0)                            # compress is relative to the BASE, not cumulative
    assert b.wx[3] == 0.5
    b.stretch_apply(100, 100)                            # clamped
    assert b.wx[3] == 6.0 and b.hy[3] == 3.0


def test_surface_stretch_and_single_vertex_selection():
    s = Surface3D(400, 350)
    s.select_start(("cell", (20, 20)))                   # corner vertex -> still >=1 segment
    s.stretch_begin()
    s.stretch_apply(2.0, 2.0)
    assert (s.cw[-1] == 2.0) and (s.rw[-1] == 2.0) and s.amp[20, 20] == 2.0
    assert s.amp[0, 0] == 1.0                            # far away untouched
    assert np.isfinite(s.build().finish().verts).all()


def test_persistence_roundtrip():
    random.seed(1)
    sc = Scene3D((1000, 700))
    b = BarChart3D(300, 300, z=120)
    b.select_start(("bar", 1)); b.select_update(("bar", 3))
    b.stretch_begin(); b.stretch_apply(1.7, 0.8)
    b.size = 200
    s = Surface3D(600, 300)
    s.select_start(("cell", (3, 4))); s.select_update(("cell", (9, 8)))
    s.stretch_begin(); s.stretch_apply(1.4, 1.9)
    for o in (b, s, Shape3D("torus", 500, 500)):
        sc.add(o)
    blob = json.loads(json.dumps(sc.to_list()))          # through real JSON
    sc2 = Scene3D((1000, 700))
    sc2.load_list(blob)
    assert [o.kind for o in sc2.objects] == ["bars3d", "surface3d", "torus"]
    b2, s2 = sc2.objects[0], sc2.objects[1]
    assert b2.data == b.data and b2.wx == b.wx and b2.selection == (1, 3) and b2.size == 200
    assert abs(b2.pos[2] - 120) < 1e-9 and np.allclose(b2.rot, b.rot)
    assert np.allclose(s2.amp, s.amp) and s2.selection == s.selection


def test_corrupt_saved_objects_are_survivable():
    sc = Scene3D((1000, 700))
    sc.load_list([{"kind": "nonsense"}, {"kind": "bars3d", "extra": {"data": "x", "sel": [9, 1]}},
                  {"kind": "surface3d", "extra": {"cw": [1], "amp": [[float("nan")]]}},
                  {"kind": "cube", "pos": ["a", "b", "c"]}, {"kind": "cube", "rot": [1, 2]}])
    for o in sc.objects:
        o.update(0.03, 0.0, False)                       # must all still build/render
        ar3d.render_object(CountCanvas(), o)


def test_scene_inertia_and_bounds():
    b = BarChart3D(500, 300)
    sc = scene_with(b)
    b.vx = 800
    for _ in range(60):
        sc.step(0.03)
    assert 500 < b.pos[0] <= 1000 - 40 and b.vx == 0.0


# ----------------------------------------------------------- hand slots ---
def fake_hand(x, y):
    lm = [types.SimpleNamespace(x=x, y=y, z=0) for _ in range(21)]
    return types.SimpleNamespace(landmark=lm)


def test_slots_lone_hand_is_primary_and_two_hands_keep_identity():
    s = HandSlots()
    a = fake_hand(0.7, 0.5)
    assert s.assign([a], 0.0) == [a, None]
    b = fake_hand(0.3, 0.5)
    out = s.assign([b, a], 0.03)                         # order swapped by "MediaPipe"
    assert out == [a, b]
    a2, b2 = fake_hand(0.55, 0.5), fake_hand(0.45, 0.5)  # hands nearly meeting, still no swap
    assert s.assign([b2, a2], 0.06) == [a2, b2]
    a3, b3 = fake_hand(0.40, 0.5), fake_hand(0.60, 0.5)  # they cross: identity follows motion
    out = s.assign([a3, b3], 0.09)
    assert out == [b3, a3] or out == [a3, b3]            # either is valid at the crossing; must be a permutation
    assert set(map(id, out)) == {id(a3), id(b3)}


def test_slots_first_pair_rightmost_is_primary_and_lost_hand_returns_to_its_slot():
    s = HandSlots()
    r, l = fake_hand(0.8, 0.5), fake_hand(0.2, 0.5)
    assert s.assign([l, r], 0.0) == [r, l]
    assert s.assign([l], 0.1) == [None, l]               # right hand left the frame briefly
    r2 = fake_hand(0.75, 0.5)
    assert s.assign([l, r2], 0.2) == [r2, l]


# ---------------------------------------------------- pointer (real funcs) ---
def synth_hand(pinch):
    """Open hand, index+thumb tips either apart or touching."""
    pts = {0: (.5, .8), 1: (.42, .74), 2: (.38, .68), 3: (.36, .62), 4: (.40, .50) if not pinch else (.5, .42)}
    for f, x in zip((5, 9, 13, 17), (.47, .5, .53, .56)):
        pts[f] = (x, .62)
        pts[f + 1] = (x, .52)
        pts[f + 2] = (x, .44)
        pts[f + 3] = (x, .36) if not (f == 5 and pinch) else (.5, .42)
    lm = [types.SimpleNamespace(x=pts[i][0], y=pts[i][1], z=0) for i in range(21)]
    return types.SimpleNamespace(landmark=lm)


def test_stray_fist_reading_does_not_interrupt_an_active_pinch():
    """Regression test for "pinch/drag releases itself mid-gesture": a
    tight pinch (especially mid two-hand resize) naturally curls the
    other fingers too, which can get misread as a fist for a single
    frame. HandPointer.update() used to force pinching=False the instant
    that happened, with no tolerance at all - every one of _grab1/
    _select/_stretch1/_grab2's "if not pin[i]: end the gesture" checks
    then read that as a genuine release. gesture_engine's real classifier
    is mocked here rather than hand-crafting landmark geometry that
    happens to satisfy both pinch-ratio AND is_strict_fist at once (easy
    to get subtly wrong without a real camera to check against) - this
    tests the exact behavioural contract that changed, deterministically.
    """
    import gesture_engine
    real_classify, real_is_fist = gesture_engine.classify_static_gesture, gesture_engine.is_strict_fist
    try:
        p = HandPointer(lambda n: (n[0] * 1000, n[1] * 700), 1280, 720)
        t = 0.0
        for _ in range(6):                                 # establish an ongoing pinch
            t += 1 / 30
            st = p.update(synth_hand(True), t)
        assert st.pinching, "pinch pose should register before the fist blip"

        gesture_engine.classify_static_gesture = lambda lm: "fist"
        gesture_engine.is_strict_fist = lambda lm: True
        t += 1 / 30
        st = p.update(synth_hand(True), t)
        assert st.pinching, \
            "a single stray 'fist' reading must not drop an ALREADY ACTIVE pinch"

        # The original protection must still hold: a fist reading must
        # still stop a pinch from being detected FRESH (not already
        # active) in the first place.
        p2 = HandPointer(lambda n: (n[0] * 1000, n[1] * 700), 1280, 720)
        t2 = 0.0
        for _ in range(6):
            t2 += 1 / 30
            st2 = p2.update(synth_hand(False), t2)
        assert not st2.pinching
        for _ in range(6):
            t2 += 1 / 30
            st2 = p2.update(synth_hand(True), t2)           # ratio says pinch, but...
        assert not st2.pinching, \
            "a fist reading must still block a NEW pinch from starting"
    finally:
        gesture_engine.classify_static_gesture = real_classify
        gesture_engine.is_strict_fist = real_is_fist


def test_hand_pointer_detects_pinch_with_hysteresis_and_absence():
    p = HandPointer(lambda n: (n[0] * 1000, n[1] * 700), 1280, 720)
    t = 0.0
    for _ in range(6):
        t += .03
        st = p.update(synth_hand(False), t)
    assert st.present and not st.pinching
    for _ in range(6):
        t += .03
        st = p.update(synth_hand(True), t)
    assert st.pinching, "pinch pose should register"
    for _ in range(8):
        t += .03
        st = p.update(synth_hand(False), t)
    assert not st.pinching
    assert p.update(None, t).present is False


# ------------------------------------------------------------ controller ---
class FakeUI:
    def __init__(self):
        self.shapes_open = False
        self.tile_pt = None
        self.tile_kind = None
        self.toggled = 0

    def palette_kind_at(self, pt):
        if pt[0] < 100 and pt[1] < 100:
            return "shapes3d"
        if 100 <= pt[0] < 200 and pt[1] < 100:
            return "bar"
        return None

    def toggle_shape_browser(self):
        self.toggled += 1
        self.shapes_open = not self.shapes_open

    def shape_tile_at(self, pt):
        if self.shapes_open and pt[0] < 200 and 150 <= pt[1] < 200:
            return "cube"
        return None

    def function_tile_at(self, pt):
        # No synthetic test hand in this file ever lands on the
        # "functions" palette button (only "shapes3d"/"bar" are mapped
        # in palette_kind_at above), so always-None is a faithful stub -
        # ar_overlay's own tests (test_ar_overlay_integration.py) cover
        # the real tile hit-testing/placement behaviour.
        return None

    def in_palette_zone(self, pt):
        return pt[0] < 260 and pt[1] < 260


class Rig:
    """Drives the controller with synthetic hands at 30 fps."""

    def __init__(self, *objs):
        self.sc = scene_with(*objs)
        self.c = AR3DController(self.sc)
        self.ui = FakeUI()
        self.t = 100.0
        self.h = [HandState(), HandState()]

    def hand(self, i, pt=None, pinch=None, size=None, present=True):
        h = self.h[i]
        h.present = present
        if pt is not None: h.pt = tuple(pt)
        if pinch is not None: h.pinching = pinch
        if size is not None: h.size = size
        elif h.size == 0: h.size = 100.0
        return self

    def tick(self, n=1):
        r = None
        for _ in range(n):
            self.t += 1 / 30
            r = self.c.step(self.h, self.t, self.ui)
            self.sc.step(1 / 30)
        return r


def body_point(o):
    g = o.geom
    b = g.mesh.body
    if b is None:                                   # shapes have no floor plate
        return (float(o.pos[0]), float(o.pos[1]))
    return ((g.sx[b[0]] + g.sx[b[1]]) / 2, (g.sy[b[0]] + g.sy[b[1]]) / 2)


def bar_point(o, i):
    g = o.geom
    idx = next(ix for k, ix in g.mesh.parts if k == ("bar", i))
    return (g.sx[idx].mean(), g.sy[idx].mean())


def make_bars(**kw):
    b = BarChart3D(500, 350, data=[.5, .8, .6, .9, .4, .7, .3], **kw)
    b.sway = 0
    return b


def test_pinch_on_empty_space_is_not_claimed():
    r = Rig(make_bars())
    r.hand(0, (900, 600), False).tick(2)
    claimed = r.hand(0, pinch=True).tick(2)
    assert claimed is False and r.c.mode == IDLE


def test_one_hand_grab_moves_without_jump_and_glides_on_release():
    b = make_bars()
    r = Rig(b)
    bp = body_point(b)
    r.hand(0, bp, False).tick(2)
    assert r.hand(0, pinch=True).tick(1) is True and r.c.mode == GRAB1
    x0, y0 = b.pos[0], b.pos[1]
    r.tick(1)
    assert abs(b.pos[0] - x0) < 1 and abs(b.pos[1] - y0) < 1, "must not jump to fingertip"
    for k in range(1, 11):
        r.hand(0, (bp[0] + 20 * k, bp[1] - 5 * k)).tick(1)
    assert abs(b.pos[0] - (x0 + 200)) < 2 and abs(b.pos[1] - (y0 - 50)) < 2
    r.hand(0, pinch=False).tick(1)
    assert r.c.mode == IDLE and b.vx > 100                    # keeps moving after release
    x1 = b.pos[0]
    r.tick(30)
    assert b.pos[0] > x1 and b.vx == 0.0                      # coasted, then settled


def test_push_pull_depth_from_hand_size_with_deadzone():
    b = make_bars()
    r = Rig(b)
    r.hand(0, body_point(b), False, size=100).tick(2)
    r.hand(0, pinch=True).tick(1)
    r.hand(0, size=103).tick(30)
    assert b.pos[2] == 0.0                                    # inside deadzone
    r.hand(0, size=130).tick(60)
    assert b.pos[2] < -100                                    # hand nearer -> object nearer
    r.hand(0, size=70).tick(60)
    assert b.pos[2] > 100
    r.hand(0, size=1000).tick(80)
    assert b.pos[2] < ar3d.Z_MIN + 1                             # clamped


def test_pinch_bar_drag_selects_range_and_tap_body_clears():
    b = make_bars()
    r = Rig(b)
    r.hand(0, bar_point(b, 2), False).tick(2)
    r.hand(0, pinch=True).tick(1)
    assert r.c.mode == SELECT and b.selection == (2, 2)
    for i in (3, 4, 5):
        r.hand(0, bar_point(b, i)).tick(1)
    assert b.selection == (2, 5)
    r.hand(0, pinch=False).tick(1)
    assert r.c.mode == IDLE and b.selection == (2, 5), "selection persists after release"
    bp = body_point(b)                                        # quick tap on the floor
    r.hand(0, bp, False).tick(1)
    r.hand(0, pinch=True).tick(2)
    r.hand(0, pinch=False).tick(1)
    assert b.selection is None


def test_slow_press_or_drag_on_body_does_not_clear_selection():
    b = make_bars()
    b.selection = (1, 2)
    r = Rig(b)
    bp = body_point(b)
    r.hand(0, bp, False).tick(1)
    r.hand(0, pinch=True).tick(15)                            # held ~0.5 s
    r.hand(0, pinch=False).tick(1)
    assert b.selection == (1, 2)


def test_two_hand_stretch_on_selection_and_compress():
    b = make_bars()
    b.selection = (2, 4)
    r = Rig(b)
    lp, rp = (400.0, 400.0), (600.0, 400.0)
    r.hand(0, rp, False).hand(1, lp, False).tick(2)
    r.hand(1, body_point(b), True).tick(1)                    # left hand grabs the floor...
    assert r.c.mode == GRAB1
    r.hand(1, lp).hand(0, rp, True).tick(1)                   # ...right hand joins over the chart
    assert r.c.mode == GRAB2 and r.c.g["stretch"]
    r.hand(0, (800.0, 400.0)).tick(1)                         # hands apart: distance 200 -> 400
    assert abs(b.wx[3] - 2.0) < 1e-6 and b.wx[0] == 1.0 and b.hy[3] == 1.0
    assert "WIDTH" in r.c.last_status
    r.hand(0, (500.0, 400.0)).tick(1)                         # together: 200 -> 100 = compress
    assert abs(b.wx[3] - 0.5) < 1e-6


def test_two_hand_stretch_height_is_both_hands_up_or_down():
    b = make_bars()
    b.selection = (0, 6)
    r = Rig(b)
    lp, rp = (400.0, 400.0), (600.0, 400.0)                   # level hands: works without any diagonal trick
    r.hand(1, body_point(b), True).hand(0, rp, False).tick(1)
    r.hand(1, lp).hand(0, pinch=True).tick(1)
    r.hand(0, pinch=True)
    assert r.c.mode == GRAB2 and r.c.g["stretch"]
    r.hand(1, (400.0, 242.0)).hand(0, (600.0, 242.0)).tick(1)  # both hands up 150 px (+8 px deadzone)
    assert abs(b.hy[3] - 2.0) < 1e-6 and abs(b.wx[3] - 1.0) < 1e-6
    r.hand(1, (400.0, 558.0)).hand(0, (600.0, 558.0)).tick(1)  # both hands down 150 px
    assert abs(b.hy[3] - 0.5) < 1e-6


def test_two_hand_scale_roll_move_without_selection():
    c = Shape3D("cube", 500, 350)
    c.spin = 0
    r = Rig(c)
    lp, rp = (400.0, 350.0), (600.0, 350.0)
    r.hand(1, body_point(c), False).hand(0, rp, False).tick(1)
    r.hand(1, pinch=True).tick(1)
    r.hand(1, lp).hand(0, rp, True).tick(1)
    assert r.c.mode == GRAB2 and not r.c.g["stretch"]
    s0, R0 = c.size, c.rot.copy()
    r.hand(0, (800.0, 350.0)).tick(60)                        # distance x2
    assert abs(c.size / s0 - 2.0) < 1e-3
    ang = math.radians(30)                                    # rotate the hand pair 30 deg (clockwise on screen)
    cx = 500.0
    r.hand(1, (cx - 100 * math.cos(ang), 350 - 100 * math.sin(ang)))
    r.hand(0, (cx + 100 * math.cos(ang), 350 + 100 * math.sin(ang))).tick(80)
    assert abs(c.size / s0 - 1.0) < 1e-3                      # distance back to 200
    expect = ar3d.rot_z(ang) @ R0
    assert np.allclose(c.rot, expect, atol=1e-3)
    assert np.allclose(c.rot @ c.rot.T, np.eye(3), atol=1e-9)
    r.hand(1, (cx - 100 * math.cos(ang) + 50, 350 - 100 * math.sin(ang) + 20))
    r.hand(0, (cx + 100 * math.cos(ang) + 50, 350 + 100 * math.sin(ang) + 20)).tick(80)
    assert abs(c.pos[0] - 550) < 2 and abs(c.pos[1] - 370) < 2   # midpoint drag moves it
    r.hand(0, pinch=False).tick(1)
    assert r.c.mode == GRAB1, "still holding one hand -> keeps dragging, no snap"
    r.hand(1, pinch=False).tick(1)
    assert r.c.mode == IDLE


def test_two_hand_yaw_from_one_hand_nearer():
    c = Shape3D("cube", 500, 350)
    c.spin = 0
    r = Rig(c)
    lp, rp = (400.0, 350.0), (600.0, 350.0)
    r.hand(1, body_point(c), True, size=100).hand(0, rp, False, size=100).tick(1)
    r.hand(1, lp).hand(0, rp, True).tick(1)
    R0 = c.rot.copy()
    r.hand(0, size=140).tick(1)                               # right hand nearer camera
    assert not np.allclose(c.rot, R0)
    # right side comes toward viewer (negative z for the local +x axis)
    assert (c.rot @ np.array([1.0, 0, 0]))[2] < (R0 @ np.array([1.0, 0, 0]))[2]


def test_palette_tile_ghost_drop_and_cancel():
    r = Rig()
    r.hand(0, (50, 50), False).tick(1)
    assert r.hand(0, pinch=True).tick(1) is True             # palette button claimed
    assert r.ui.shapes_open and r.ui.toggled == 1
    r.tick(3)
    assert r.ui.toggled == 1, "one pinch must toggle once"
    r.hand(0, pinch=False).tick(1)
    r.hand(0, (50, 170), False).tick(1)                       # over the tile
    r.hand(0, pinch=True).tick(1)
    assert r.c.mode == GHOST and r.sc.ghost is not None and not r.ui.shapes_open
    r.hand(0, (600, 500)).tick(1)
    assert tuple(r.sc.ghost.pos[:2]) == (600.0, 500.0)
    r.hand(0, pinch=False).tick(1)
    assert len(r.sc.objects) == 1 and r.sc.ghost is None and r.c.mode == IDLE
    # cancel path
    r.ui.shapes_open = True
    r.hand(0, (50, 170), False).tick(1)
    r.hand(0, pinch=True).tick(1)
    assert r.c.mode == GHOST
    r.hand(0, (120, 120)).tick(1)                              # back over palette zone
    r.hand(0, pinch=False).tick(1)
    assert len(r.sc.objects) == 1 and r.sc.ghost is None


def test_other_palette_buttons_fall_through_to_2d():
    r = Rig()
    r.hand(0, (150, 50), False).tick(1)
    assert r.hand(0, pinch=True).tick(1) is False


def test_second_hand_can_drive_everything_and_lost_hand_releases():
    b = make_bars()
    r = Rig(b)
    r.hand(0, present=False).hand(1, body_point(b), False).tick(1)
    r.hand(1, pinch=True).tick(1)
    assert r.c.mode == GRAB1 and r.c.gi == 1
    assert r.c.primary_idx == 1
    r.hand(1, present=False, pinch=False).tick(1)              # tracking dropout mid-grab
    assert r.c.mode == IDLE and r.sc.grabbed == set()


def test_remove_and_refresh():
    b = make_bars()
    b.selection = (1, 2)
    r = Rig(b)
    old = list(b.data)
    assert r.c.refresh_at(bar_point(b, 1)) and b.selection is None and b.data != old
    assert r.c.remove_at(bar_point(b, 1)) and r.sc.objects == []
    assert r.c.remove_at((10, 10)) is False


def test_hud_marks_two_hand_tether_and_second_cursor():
    b = make_bars()
    b.selection = (0, 1)
    r = Rig(b)
    r.hand(1, body_point(b), True).hand(0, (600, 400), False).tick(1)
    r.hand(1, (400, 400)).hand(0, pinch=True).tick(1)
    hud = r.sc.hud
    assert r.c.mode == GRAB2 and hud["tether"] and len(hud["cursors"]) == 1
    cv = CountCanvas()
    r.sc.draw(cv)


# =========================================================== v23 polish ===
from ar3d_input import PinchStableCursor


def test_stable_cursor_ignores_finger_curl_but_follows_the_hand():
    """Fingertip sags 50px toward the thumb while the pinch closes. The cursor
    must stay on the target - and still track real hand movement."""
    c = PinchStableCursor()
    palm = (0.5, 0.7)
    t = 0.0
    out = None
    for _ in range(10):                                   # open hand, tip 0.35 above palm
        t += .03
        out = c.update((0.5, 0.35), palm, 1.3, t)
    aim = out
    assert aim == (0.5, 0.35)
    for k in range(1, 13):                                # ratio 1.3 -> 0.4, tip drops 0.07
        t += .03
        ratio = 1.3 - 0.075 * k
        out = c.update((0.5, 0.35 + 0.07 * k / 12), palm, ratio, t)
    assert c.locked and abs(out[1] - aim[1]) < 0.012, (out, aim)      # <~8px of sag at 720p
    moved = c.update((0.5, 0.42), (0.55, 0.7), 0.4, t + .03)          # hand slides right while pinched
    assert abs(moved[0] - 0.55) < 1e-9 and abs(moved[1] - aim[1]) < 0.012


def test_stable_cursor_release_eases_back_without_a_snap():
    c = PinchStableCursor()
    palm, t = (0.5, 0.7), 0.0
    for r in (1.3, 0.8, 0.5, 0.4):
        t += .03
        c.update((0.5, 0.35 if r > 0.85 else 0.40), palm, r, t)
    locked_y = c.last[1]
    t += .03
    first = c.update((0.5, 0.40), palm, 1.2, t)           # unlocks: live tip is 0.05 below the locked point
    assert abs(first[1] - locked_y) < 0.01, "must not jump on release"
    ys = [first[1]]
    for _ in range(8):
        t += .03
        ys.append(c.update((0.5, 0.40), palm, 1.2, t)[1])
    assert abs(ys[-1] - 0.40) < 1e-9
    steps = [b - a for a, b in zip(ys, ys[1:])]
    assert max(steps) < 0.03 and all(s >= -1e-9 for s in steps)          # monotone glide


def synth_hand_frac(f):
    """Open pointing hand (f=0) -> pinch (f=1): index tip sags, thumb rises."""
    pts = {0: (.5, .8), 1: (.42, .74), 2: (.38, .68), 3: (.36, .62),
           4: (.40 + .10 * f, .50 - .08 * f)}
    for fng, x in zip((5, 9, 13, 17), (.47, .5, .53, .56)):
        pts[fng], pts[fng + 1], pts[fng + 2] = (x, .62), (x, .52), (x, .44)
        pts[fng + 3] = (x, .36)
    pts[8] = (.47 + .03 * f, .36 + .06 * f)
    lm = [types.SimpleNamespace(x=pts[i][0], y=pts[i][1], z=0) for i in range(21)]
    return types.SimpleNamespace(landmark=lm)


def test_pointer_cursor_does_not_sag_during_a_real_pinch_motion():
    H = 700.0
    p = HandPointer(lambda n: (n[0] * 1000, n[1] * H), 1280, 720)
    t = 0.0
    for _ in range(12):
        t += .033
        st = p.update(synth_hand_frac(0), t)
    aim = st.pt
    raw_drop = 0.06 * H
    ys, pinched_at = [], None
    for k in range(1, 15):                                # ~0.45 s closing the pinch
        t += .033
        st = p.update(synth_hand_frac(min(1.0, k / 10)), t)
        ys.append(st.pt[1])
        if st.pinching and pinched_at is None:
            pinched_at = k
    assert pinched_at is not None, "pinch must still register"
    assert abs(st.pt[1] - aim[1]) < 0.35 * raw_drop, (st.pt, aim, raw_drop)   # naive tip cursor sags all 42px
    assert st.approach == 1.0


def test_pinch_selects_the_bar_you_aimed_at_not_the_one_below():
    """End to end: aim at a bar's upper body, close the pinch with the same
    fingertip sag - the controller must see the pinch on that bar."""
    b = make_bars()
    r = Rig(b)
    tx, ty = bar_point(b, 3)
    aim = (tx, ty - 20)                                     # upper part of bar 3
    r.hand(0, aim, False).tick(3)
    r.hand(0, pinch=True).tick(1)                           # stable cursor: still at `aim`
    assert r.c.mode == SELECT and b.selection == (3, 3)


def test_object_can_never_be_left_with_its_floor_off_screen():
    s = Surface3D(500, 600, size=150)
    sc = scene_with(s, bounds=(1000, 700))
    for y in (650, 690, 900, 5000, -400):
        s.pos[1] = y
        sc.step(1 / 30)
        g = s.geom
        assert g.sy.max() <= 700 - Scene3D.MARGINS[3] + 1e-6, (y, g.sy.max())
        assert g.sy.min() >= Scene3D.MARGINS[1] - 1e-6
        body = g.mesh.body                                  # the grab-able floor is fully visible
        assert g.sy[body].max() <= 700 - Scene3D.MARGINS[3] + 1e-6
    for x in (-500, 1500):
        s.pos[0] = x
        sc.step(1 / 30)
        assert s.geom.sx.min() >= Scene3D.MARGINS[0] - 1e-6 and s.geom.sx.max() <= 1000 - Scene3D.MARGINS[2] + 1e-6


def test_dragging_to_the_bottom_edge_stops_with_floor_reachable_and_can_be_pulled_back():
    s = Surface3D(500, 350, size=150)
    s.sway = 0
    r = Rig(s)
    bp = body_point(s)
    r.hand(0, bp, False).tick(2)
    r.hand(0, pinch=True).tick(1)
    for k in range(40):                                     # drag way below the screen
        r.hand(0, (bp[0], bp[1] + 20 * k)).tick(1)
    assert s.geom.sy[s.geom.mesh.body].max() <= 700 - 64 + 1e-6
    fx, fy = body_point(s)                                  # floor front edge is on screen -> grabbable
    assert 0 < fy < 700 and s.pick(fx, fy) is not None
    r.hand(0, pinch=False).tick(2)
    r.hand(0, (fx, fy), False).tick(2)
    r.hand(0, pinch=True).tick(1)
    assert r.c.mode == GRAB1
    y_before = s.pos[1]
    for k in range(1, 25):
        r.hand(0, (fx, fy - 15 * k)).tick(1)                # pull it back up
    assert s.pos[1] < y_before - 150


def test_two_hand_growth_stops_at_the_screen_instead_of_pushing_off():
    c = Shape3D("cube", 500, 350)
    c.spin = 0
    r = Rig(c)
    r.hand(1, (400.0, 350.0), True).hand(0, (600.0, 350.0), False).tick(1)
    r.hand(0, pinch=True).tick(1)
    r.hand(0, (2000.0, 350.0)).tick(120)                    # absurdly far apart
    g = c.geom
    assert g.sx.min() >= 15.9 and g.sx.max() <= 1000 - 15.9
    assert g.sy.min() >= 15.9 and g.sy.max() <= 700 - 63.9


def test_saved_off_screen_object_is_rescued_on_load():
    sc = Scene3D((1000, 700))
    sc.load_list([{"kind": "surface3d", "pos": [500, 690, 0], "size": 300}])
    g = sc.objects[0].geom
    assert g.sy.max() <= 700 - 64 + 1e-6 and g.sx.min() >= 15.9


def test_sway_and_spin_stop_when_a_hand_hovers_and_resume_after():
    b = BarChart3D(500, 350)
    c = Shape3D("cube", 800, 350)
    sc = scene_with(b, c)
    for _ in range(90):
        sc.step(1 / 30)
    moving = not np.allclose(b._disp_rot, b.rot)
    assert moving and b._sway_gain > 0.9
    sc.hover = {0: (b, ("body", None)), 1: (c, ("body", None))}
    ang0 = c.rot.copy()
    for _ in range(45):
        sc.step(1 / 30)
    assert b._sway_gain < 0.02 and c._spin_gain < 0.02
    before_b, before_c = b._disp_rot.copy(), c.rot.copy()
    for _ in range(10):
        sc.step(1 / 30)
    assert np.allclose(b._disp_rot, before_b, atol=1e-3) and np.allclose(c.rot, before_c, atol=1e-3)
    sc.hover = {}
    for _ in range(60):
        sc.step(1 / 30)
    assert b._sway_gain > 0.9 and c._spin_gain > 0.9


def test_overlay_pointer_holds_the_aimed_spot_while_the_fingertip_sags_during_a_pinch():
    """The overlay reticle / drawing / Whack-a-Mole now read a
    PinchStableCursor fed (fingertip, index knuckle, pinch ratio) - the
    scenario that matters: aim, then close a pinch. The raw fingertip
    drops toward the thumb as the ratio falls; the stabilised point must
    stay where it was aimed, and must still follow real hand movement
    (the knuckle) so strokes/drags work."""
    from pinch_stable import PinchStableCursor
    import ar3d_input
    assert ar3d_input.PinchStableCursor is PinchStableCursor, "AR and the overlay share one implementation"

    c = PinchStableCursor()
    t = 0.0
    knuckle = (0.50, 0.50)
    aim = (0.50, 0.40)                            # fingertip 0.10 above the knuckle, hand open
    for _ in range(12):                           # open hand, ratio ~1.4
        t += 1 / 30
        out = c.update(aim, knuckle, 1.4, t)
    assert abs(out[1] - 0.40) < 1e-6

    # pinch closes over ~8 frames: ratio 1.4 -> 0.40, fingertip sags 0.10 -> 0.04 above the knuckle
    for i in range(1, 9):
        t += 1 / 30
        k = i / 8
        tip = (0.50, 0.40 + 0.06 * k)             # raw tip drifts DOWN (y grows)
        out = c.update(tip, knuckle, 1.4 - 1.0 * k, t)
    raw_sag = tip[1] - aim[1]
    assert raw_sag > 0.05
    assert abs(out[1] - aim[1]) < 0.01, f"stabilised point drifted {out[1] - aim[1]:+.3f} (raw tip sagged {raw_sag:+.3f})"

    # while still pinching, moving the HAND (knuckle) moves the point by the same amount
    for i in range(10):
        t += 1 / 30
        knuckle = (knuckle[0] + 0.01, knuckle[1])
        out = c.update((knuckle[0], 0.46), knuckle, 0.40, t)
    assert abs(out[0] - 0.60) < 0.015, "must follow real hand movement while pinching (drawing/dragging)"
    assert abs(out[1] - aim[1]) < 0.01

    # open the hand again: releases back to the live fingertip
    for i in range(12):
        t += 1 / 30
        out = c.update((knuckle[0], 0.40), knuckle, 1.4, t)
    assert abs(out[1] - 0.40) < 0.01 and not c.locked


def test_delete_at_reports_blocked_so_the_fist_can_retry_instead_of_being_spent():
    """The fist-delete "sometimes doesn't work" bug: the controller used
    to return the same True for "removed" and "an object is there but a
    recent pinch made me refuse", so the caller treated both as done and
    the gesture was silently spent. delete_at distinguishes them."""
    b = make_bars()
    r = Rig(b)
    px, py = body_point(b)
    assert r.c.delete_at((9000, 9000), r.t) is None          # nothing there
    r.c._last_active = r.t                                    # a pinch ended just now
    assert r.c.delete_at((px, py), r.t) == "blocked"
    assert b in r.sc.objects, "blocked must NOT remove"
    assert r.c.remove_at((px, py), r.t) is True               # legacy bool contract unchanged
    r.t += 2.0                                                # cooldown elapsed
    assert r.c.delete_at((px, py), r.t) == "removed"
    assert b not in r.sc.objects


def test_objects_pop_in_when_placed_without_touching_their_real_size():
    """Placing an object eases it up from a small scale instead of it
    appearing at full size between two frames. The effect must be
    purely visual: self.size (persisted, resized by gestures) must never
    change, the drawn size (eff_size) must rise monotonically to exactly
    the real size, and picking must use the same drawn size so hit-tests
    never disagree with what's on screen mid-animation."""
    b = BarChart3D(500, 350)
    b.sway = 0
    real_size = b.size
    sc = scene_with(b, settled=False)
    assert b.size == real_size
    sizes = []
    for _ in range(30):
        sc.step(1 / 30)
        sizes.append(b.eff_size)
        assert b.size == real_size, "pop-in must never modify the real, persisted size"
    assert sizes == sorted(sizes), "drawn size must grow monotonically"
    assert sizes[0] < real_size * 0.9, "first frames should still be visibly small"
    assert sizes[-1] == real_size and b._pop == 1.0, "must settle at exactly the real size"
    assert b.to_dict()["size"] == real_size

    # Mid-animation, the cached geometry must track the animation (the
    # geometry cache key includes _pop) rather than freezing at frame 0.
    b2 = BarChart3D(500, 350); b2.sway = 0
    sc2 = scene_with(b2, settled=False)
    w0 = float(np.ptp(b2.geom.sx))
    for _ in range(10):
        sc2.step(1 / 30)
    assert float(np.ptp(b2.geom.sx)) > w0 * 1.2


def test_idle_sway_has_per_object_phase_and_a_secondary_tilt_axis():
    """Regression test for "boring, swaying left-right, all in lockstep":
    every object used to sway on the exact same single Y-axis, same
    frequency, same phase - reading as one flat shared metronome rather
    than independent objects. Checks two different objects (so two
    different ids -> two different _sway_phase/_sway_freq) visibly
    diverge from each other over time, and that the rotation includes a
    real X-axis (tilt) component, not just Y (yaw)."""
    b1 = BarChart3D(400, 350)
    b2 = BarChart3D(900, 350)
    b1.sway = b2.sway = 0.3
    sc = scene_with(b1, b2)
    assert b1._sway_phase != b2._sway_phase or b1._sway_freq != b2._sway_freq, \
        "two different objects must not land on identical phase AND frequency"

    diverged = False
    for _ in range(120):
        sc.step(1 / 30)
        if not np.allclose(b1._disp_rot, b2._disp_rot, atol=1e-3):
            diverged = True
    assert diverged, "two objects with the same sway amplitude must still move differently from each other"

    # A pure single-axis (Y only) sway, as it used to be, keeps the
    # rotation matrix symmetric in a way that a combined Y+X rotation
    # does not - row 0 (affected only by yaw) and column 1 (affected
    # only by tilt) both move away from the identity's [1,0,0]/[0,1,0]
    # when a genuine secondary tilt axis is present.
    off_axis_motion = abs(b1._disp_rot[0, 2]) + abs(b1._disp_rot[2, 1])
    assert off_axis_motion > 1e-4, "expected a secondary (non-Y) rotation component, not pure yaw"


def test_selection_drag_holds_the_chart_still():
    b = make_bars()
    b.sway = 0.2
    r = Rig(b)
    r.tick(60)
    r.hand(0, bar_point(b, 1), False).tick(2)
    r.hand(0, pinch=True).tick(1)
    assert r.c.mode == SELECT and b.id in r.sc.calm
    r.tick(45)
    assert b._sway_gain < 0.02


def test_inertia_glide_is_frame_rate_independent():
    def glide(fps):
        o = BarChart3D(300, 350)
        sc = scene_with(o, bounds=(3000, 900))
        o.vx = 600
        x0 = o.pos[0]
        for _ in range(int(fps * 1.5)):
            sc.step(1 / fps)
        return o.pos[0] - x0
    a, b = glide(30), glide(60)
    assert abs(a - b) / a < 0.05, (a, b)


def test_two_hand_scale_jitter_is_attenuated():
    c = Shape3D("cube", 500, 350)
    c.spin = 0
    r = Rig(c)
    r.hand(1, (400.0, 350.0), True).hand(0, (600.0, 350.0), False).tick(1)
    r.hand(0, pinch=True).tick(60)
    sizes = []
    for k in range(60):                                     # +-4% distance noise, alternating
        r.hand(0, (400 + 200 * (1.04 if k % 2 else 0.96), 350.0)).tick(1)
        sizes.append(c.size)
    swing = (max(sizes) - min(sizes)) / c.size
    assert swing < 0.04, swing                              # raw input swings ~8%


# ======================================= v24: deletion bug + reshaping =====
def test_fist_or_two_finger_can_not_delete_or_reset_during_a_two_hand_pinch():
    """The reported bug: squeezing a two-hand pinch curls the other fingers,
    the classifier says 'fist', and the object was removed mid-resize."""
    surf = Surface3D(500, 350)
    surf.sway = 0
    surf.selection = (5, 9, 5, 9)
    r = Rig(surf)
    lp, rp = (400.0, 350.0), (600.0, 350.0)
    r.hand(1, body_point(surf), True).hand(0, rp, False).tick(1)
    r.hand(1, lp).hand(0, pinch=True).tick(1)
    assert r.c.mode == GRAB2
    r.hand(0, (700.0, 350.0)).tick(5)
    wrapped = list(surf.cw)
    now = r.t
    assert r.c.remove_at((500, 350), now) is True             # swallowed: 2D must not act either
    assert r.c.refresh_at((500, 350), now) is True
    assert r.sc.objects == [surf] and list(surf.cw) == wrapped and surf.selection == (5, 9, 5, 9)
    r.hand(0, pinch=False).hand(1, pinch=False).tick(3)       # released a moment ago: still protected
    assert r.c.remove_at((500, 350), r.t) is True and r.sc.objects == [surf]
    r.tick(40)                                                # ~1.3 s of calm: a deliberate fist works
    assert r.c.remove_at(body_point(surf), r.t) is True and r.sc.objects == []
    assert r.sc.undo_remove() and r.sc.objects == [surf], "undo brings it back"
    assert r.sc.undo_remove() is False


def test_small_object_floor_grows_when_stretching_a_selection_outward():
    """Regression test for "resize doesn't do anything - floor isn't big
    enough": stretch_apply deliberately keeps an object's overall
    footprint (size) fixed and only reshuffles width/height multipliers
    inside it, which is invisible on a small object (a selected part
    growing 6x inside a 45px-wide floor is still only a few pixels).
    AR3DController._grow_floor_for_stretch is meant to let the floor
    grow a little when a stretch pulls outward, so a small object
    actually shows the change."""
    from ar3d import SIZE_MIN
    b = make_bars(size=SIZE_MIN + 5.0)          # start deliberately tiny
    b.selection = (2, 4)
    size_before = b.size
    r = Rig(b)
    px, py = bar_point(b, 3)
    r.hand(0, (px, py), False).tick(2)
    r.hand(0, pinch=True).tick(1)
    assert r.c.mode == "stretch1"
    r.hand(0, (px + 208, py)).tick(1)           # 200 px right -> pulling the selection wider
    assert b.size > size_before, \
        "floor should grow to give a small object's stretch room to be visible"
    assert b.wx[3] > 1.9, "the underlying selection stretch itself must still apply as before"

    # Compressing must NOT shrink the floor below where it started - only
    # an outward pull is meant to grow it, same as the method's own
    # "never shrink it" rule.
    size_after_grow = b.size
    r.hand(0, (px - 100, py)).tick(1)           # back toward/past the start = compress
    assert b.size >= size_after_grow - 1e-6 or b.size >= size_before


def test_one_hand_pinch_inside_selection_stretches_width_and_height():
    b = make_bars()
    b.selection = (2, 4)
    r = Rig(b)
    px, py = bar_point(b, 3)
    r.hand(0, (px, py), False).tick(2)
    r.hand(0, pinch=True).tick(1)
    assert r.c.mode == "stretch1" and b.selection == (2, 4), "must NOT start a new selection"
    r.hand(0, (px + 208, py)).tick(1)                         # 200 px right (after 8 px deadzone) = 2x wide
    assert abs(b.wx[3] - 2.0) < 1e-6 and b.wx[0] == 1.0 and b.hy[3] == 1.0
    r.hand(0, (px + 208, py - 158)).tick(1)                   # 150 px up = 2x taller
    assert abs(b.hy[3] - 2.0) < 1e-6
    r.hand(0, (px - 208, py + 158)).tick(1)                   # left + down = compress both
    assert abs(b.wx[3] - 0.5) < 1e-6 and abs(b.hy[3] - 0.5) < 1e-6
    assert "WIDTH" in r.c.last_status
    r.hand(0, pinch=False).tick(1)
    assert r.c.mode == IDLE and b.selection == (2, 4) and abs(b.wx[3] - 0.5) < 1e-6


def test_stretching_twice_compounds_instead_of_snapping_back():
    b = make_bars()
    b.selection = (2, 4)
    r = Rig(b)
    px, py = bar_point(b, 3)
    for _ in range(2):                                        # two separate drags of 2x
        r.hand(0, (px, py), False).tick(2)
        r.hand(0, pinch=True).tick(1)
        r.hand(0, (px + 208, py)).tick(1)
        r.hand(0, pinch=False).tick(1)
        px, py = bar_point(b, 3)
    assert abs(b.wx[3] - 4.0) < 1e-6


def test_pinch_outside_selection_still_starts_a_new_selection():
    b = make_bars()
    b.selection = (2, 4)
    r = Rig(b)
    r.hand(0, bar_point(b, 0), False).tick(2)
    r.hand(0, pinch=True).tick(1)
    assert r.c.mode == SELECT and b.selection == (0, 0)


def test_second_hand_joining_does_not_destroy_the_existing_selection():
    """Both hands land on the chart to stretch: the first pinch used to start
    a NEW one-cell selection, wiping the one you had made."""
    b = make_bars()
    b.selection = (2, 4)
    r = Rig(b)
    r.hand(0, bar_point(b, 0), False).hand(1, bar_point(b, 6), False).tick(2)
    r.hand(0, pinch=True).tick(1)                             # hand A lands first, on bar 0
    assert r.c.mode == SELECT and b.selection == (0, 0)
    r.hand(1, pinch=True).tick(1)                             # hand B lands a frame later
    assert r.c.mode == GRAB2 and r.c.g["stretch"] and b.selection == (2, 4)
    dx = r.h[1].pt[0] - r.h[0].pt[0]
    r.hand(1, (r.h[1].pt[0] + dx, r.h[1].pt[1])).tick(1)      # spread the hands: distance x2
    assert abs(b.wx[3] - 2.0) < 1e-6 and b.wx[0] == 1.0


def test_two_hand_landing_on_data_with_no_selection_scales_the_whole_object():
    """Previously the first pinch made a tiny selection so two hands stretched
    ONE bar instead of resizing the chart."""
    b = make_bars()
    r = Rig(b)
    r.hand(0, bar_point(b, 1), False).hand(1, bar_point(b, 5), False).tick(2)
    r.hand(0, pinch=True).tick(1)
    r.hand(1, pinch=True).tick(1)
    assert r.c.mode == GRAB2 and not r.c.g["stretch"] and b.selection is None
    s0 = b.size
    r.hand(1, (r.h[1].pt[0] + (r.h[1].pt[0] - r.h[0].pt[0]) * 0.5, r.h[1].pt[1])).tick(60)
    assert b.size > s0 * 1.2


def floor_point(o, x, z):
    """Where the floor point (x, z) is drawn (independent re-derivation of the projection)."""
    R = o._disp_rot
    W = o.size * (R @ np.array([x, 0.5, z]))
    s = ar3d.FOCAL / (ar3d.FOCAL + o.pos[2] + W[2])
    return (o.pos[0] + W[0] * s, o.pos[1] + W[1] * s)


def test_floor_unprojection_inverts_the_projection_at_any_pose():
    rng = random.Random(4)
    for _ in range(40):
        s = Surface3D(rng.uniform(300, 700), rng.uniform(250, 450), z=rng.uniform(-300, 600), size=rng.uniform(80, 250))
        s.rot = rot_y(rng.uniform(-1.0, 1.0)) @ ar3d.rot_z(rng.uniform(-1.2, 1.2)) @ rot_x(rng.uniform(.4, .9))
        s._disp_rot = s.rot.copy()
        x, z = rng.uniform(-1, 1), rng.uniform(-1, 1)
        if abs((s.rot @ np.array([0.0, -1.0, 0.0]))[2]) < 0.15:
            continue                                          # floor (near) edge-on: fallback path, tested separately
        checked = locals().get("checked", 0) + 1
        back = s._unproject_floor(*floor_point(s, x, z))
        assert back is not None and abs(back[0] - x) < 1e-7 and abs(back[1] - z) < 1e-7, (x, z, back)


def test_edge_on_floor_falls_back_to_nearest_vertex_without_crashing():
    s = Surface3D(500, 350, size=150)
    s.rot = rot_y(math.pi / 2) @ rot_x(0.5)                   # yawed 90 deg: floor seen edge-on
    s.sway = 0
    scene_with(s)
    s._disp_rot = s.rot.copy()
    s.update(0.0, 0.0, True)
    assert s._unproject_floor(500, 350) is None
    part = s.nearest_part(*(s.geom.sx[s._vi[5, 5]], s.geom.sy[s._vi[5, 5]]))
    assert part[0] == "cell"
    s.select_start(part)
    s.select_update(s.nearest_part(510, 340))                 # dragging must not raise
    assert s.selection is not None


def test_surface_selection_follows_the_floor_even_under_tall_peaks():
    s = Surface3D(500, 350, size=150)
    s.sway = 0
    scene_with(s)
    s.amp[:] = 3.0                                            # very tall everywhere
    s.amp_d[:] = 3.0
    s.update(0.0, 0.0, True)
    xs, zs = s._xs, s._zs
    for (i, j) in ((3, 4), (10, 10), (16, 7), (0, 20), (20, 0)):
        assert s.nearest_part(*floor_point(s, xs[i], zs[j])) == ("cell", (i, j))
    s.select_start(("cell", (4, 4)))
    s.select_update(s.nearest_part(*floor_point(s, xs[12], zs[9])))
    assert s.selection == (4, 12, 4, 9)


def test_surface_select_then_stretch_end_to_end_and_selection_is_drawn():
    s = Surface3D(500, 350, size=150)
    s.sway = 0
    r = Rig(s)
    xs, zs = s._xs, s._zs
    g = s.geom
    vi = s._vi
    start = (g.sx[vi[6, 6]], g.sy[vi[6, 6]])                  # touch the mesh point itself
    r.hand(0, start, False).tick(2)
    r.hand(0, pinch=True).tick(1)
    assert r.c.mode == SELECT
    r.hand(0, floor_point(s, xs[13], zs[13])).tick(2)         # rubber-band to the far corner on the floor
    assert s.selection[0] <= 6 <= s.selection[1] and s.selection[1] >= 12 and s.selection[3] >= 12
    r.hand(0, pinch=False).tick(1)
    sel = s.selection
    cv = CountCanvas()
    ar3d.render_object(cv, s)
    assert cv.polys >= 1, "the amber floor patch of the selection is drawn"
    cx, cy = floor_point(s, (xs[sel[0]] + xs[sel[1]]) / 2, (zs[sel[2]] + zs[sel[3]]) / 2)
    r.hand(0, (cx, cy), False).tick(2)
    r.hand(0, pinch=True).tick(1)
    assert r.c.mode == "stretch1" and s.selection == sel
    r.hand(0, (cx + 208, cy - 158)).tick(60)
    assert s.cw[sel[0]] > 1.9 and s.amp[(sel[2] + sel[3]) // 2, (sel[0] + sel[1]) // 2] > 1.9
    assert s.cw[0] == 1.0 or sel[0] > 0                       # untouched columns stay put


# -------------------------------------------------------------- visual smoke ---
def test_full_scene_renders_offscreen():
    random.seed(5)
    sc = scene_with(BarChart3D(250, 350), Surface3D(650, 350), Shape3D("sphere", 450, 120))
    c = PILCanvas(900, 600)
    sc.draw(c)
    px = np.array(c.img.convert("L"))
    assert (px > 40).sum() > 3000


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL  {n}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
