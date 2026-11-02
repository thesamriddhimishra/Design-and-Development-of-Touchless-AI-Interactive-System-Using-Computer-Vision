"""Tests for main_overlay.py's pure, Tk-free helper functions.
Run: python tests/test_main_overlay.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import fake_tk
fake_tk.install()

from main_overlay import _fit_scale_offset


def test_matching_aspect_ratio_fills_exactly_with_no_offset():
    scale, ox, oy, dw, dh = _fit_scale_offset(1280, 720, 1920, 1080)
    assert scale == 1.5 and dw == 1920 and dh == 1080 and ox == 0 and oy == 0


def test_mismatched_aspect_ratio_is_centered_not_stretched():
    """Regression test for "the drawing gets crooked on the overlay": a
    16:9 frame (1280x720) into a 16:10 screen (1920x1200, deliberately a
    DIFFERENT aspect ratio) must scale by the SAME factor on both axes
    (so a diagonal line's angle is preserved) and be centered in
    whichever axis has room left over - not stretched non-uniformly to
    exactly fill 1920x1200, which would change every diagonal's slope."""
    src_w, src_h = 1280, 720
    dst_w, dst_h = 1920, 1200                 # 16:10, not 16:9 - deliberately mismatched
    scale, ox, oy, dw, dh = _fit_scale_offset(src_w, src_h, dst_w, dst_h)

    # A single uniform scale, not independent x/y stretch factors.
    assert abs(dw / src_w - dh / src_h) < 1e-9, \
        "x and y scale factors must be identical, or angles get distorted"
    assert dw <= dst_w and dh <= dst_h            # fits within, doesn't overflow either axis
    assert dw == dst_w or dh == dst_h             # touches at least one edge (no wasted scale)
    # Centered: equal leftover space split on whichever axis has slack.
    assert ox == (dst_w - dw) // 2 and oy == (dst_h - dh) // 2

    # The actual "crooked line" check: a point exactly on the diagonal
    # (e.g. the frame's center-to-corner line) must land on the SAME
    # diagonal (same slope) after scaling, not a different one.
    import math
    p1, p2 = (0, 0), (src_w, src_h)
    orig_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    q1 = (p1[0] * scale, p1[1] * scale)
    q2 = (p2[0] * scale, p2[1] * scale)
    new_angle = math.atan2(q2[1] - q1[1], q2[0] - q1[0])
    assert abs(orig_angle - new_angle) < 1e-9


def test_portrait_screen_letterboxes_the_other_way():
    """Same guarantee, flipped: a landscape frame onto a taller-than-
    wide (portrait) screen must still use one uniform scale. Tolerance
    allows for int() truncation on the final pixel dimensions - the
    underlying `scale` factor itself (used for both axes) is exact."""
    scale, ox, oy, dw, dh = _fit_scale_offset(1280, 720, 1080, 1920)
    assert abs(dw / 1280 - dh / 720) < 1e-3
    assert dw == 1080                         # width is the binding constraint here
    assert oy > 0 and ox == 0                 # letterboxed top/bottom, flush left/right


def test_status_drawer_eases_open_instead_of_snapping():
    """Regression/behaviour-lock test for the "more interactive, less
    static" pass: the status drawer used to flip instantly between its
    collapsed pill and expanded panel the moment the hover check passed.
    _status_open is now a continuous 0..1 value eased toward the target
    each frame (see MainOverlay._ease_toward) - checked here by directly
    controlling elapsed time (_last_update_t) rather than real sleeps,
    so the test is both deterministic and fast."""
    from main_overlay import MainOverlay
    ov = MainOverlay(screen_size=(1600, 900))
    MODES = {"ar": {"name": "AR", "color": (0, 180, 255), "help": ["pinch to grab"]}}
    MODE_ORDER = ["ar"]

    tab_x1, tab_y1, tab_x2, tab_y2 = ov._status_tab_rect()
    inside_tab = ((tab_x1 + tab_x2) / 2, (tab_y1 + tab_y2) / 2)

    assert ov._status_open == 0.0
    openness_trace = []
    for _ in range(6):
        ov._last_update_t = time.time() - (1 / 30)   # pretend exactly one frame has elapsed
        ov.update("ar", MODES, MODE_ORDER, True, "pinch", 0.9, 30.0, True, hand_point=inside_tab)
        openness_trace.append(ov._status_open)

    assert openness_trace == sorted(openness_trace), "openness must rise monotonically, not jump around"
    assert 0.0 < openness_trace[0] < 0.9, \
        "the very first frame after hovering must NOT already be fully open - that's a snap, not an ease"
    assert openness_trace[-1] > 0.9, "openness should approach 1.0 (fully expanded) after several frames"

    # Move away: it stays open through the grace period (so a wobble can't
    # slam it shut), then must ease back down - not snap to 0 either.
    ov.update("ar", MODES, MODE_ORDER, True, "pinch", 0.9, 30.0, True, hand_point=(10, 10))
    assert ov._status_open > 0.9, "must not collapse the instant the pointer leaves"
    ov._status_hold_until = 0.0
    closing_trace = []
    for _ in range(6):
        ov._last_update_t = time.time() - (1 / 30)
        ov.update("ar", MODES, MODE_ORDER, True, "pinch", 0.9, 30.0, True, hand_point=(10, 10))
        closing_trace.append(ov._status_open)
    assert closing_trace == sorted(closing_trace, reverse=True), "openness must ease back down, not jump"
    assert closing_trace[0] < openness_trace[-1], "closing should visibly start from where it left off"
    ov.close()


class _Rec:
    """Canvas stand-in that records every create_* call (name, coords, kwargs)."""
    def __init__(self): self.calls = []
    def delete(self, *a): self.calls = []
    def __getattr__(self, name):
        if name.startswith("create_"):
            def f(*c, **k): self.calls.append((name[7:], c, k))
            return f
        raise AttributeError(name)
    def named(self, n): return [c for c in self.calls if c[0] == n]


class _FakePhoto:
    made = pastes = 0
    def __init__(self, image, master=None): _FakePhoto.made += 1; self.size = image.size
    def paste(self, image): _FakePhoto.pastes += 1


MODES = {"drawing": {"name": "Drawing", "color": (0, 180, 255), "help": ["pinch to draw"]},
         "desktop": {"name": "Desktop", "color": (0, 220, 100), "help": ["point to move"]}}


def _overlay():
    import types
    import main_overlay
    main_overlay.ImageTk = types.SimpleNamespace(PhotoImage=_FakePhoto)
    ov = main_overlay.MainOverlay(screen_size=(1920, 1080))
    ov.canvas = _Rec()
    return ov


def _has_reticle_at(rec, x, y):
    """The reticle's white centre dot (a 5 px oval centred on the point)."""
    return any(n == "oval" and abs((c[0] + c[2]) / 2 - x) < 0.6 and abs((c[1] + c[3]) / 2 - y) < 0.6
               and k.get("fill") == "#ffffff" for n, c, k in rec.calls)


def test_reticle_follows_the_hand_in_any_mode_and_only_when_asked():
    """Regression for "I can't see where my finger is without the camera
    feed": the overlay now draws an always-on reticle at the tracked
    fingertip. It's drawn whether or not hand control is on (the `enabled`
    flag isn't consulted), and not at all when the caller says another
    window already owns the cursor (reticle=False)."""
    ov = _overlay()
    for enabled in (True, False):
        ov.update("desktop", MODES, ["desktop"], enabled, "-", 0.0, 30.0, True,
                  hand_point=(640, 360), reticle=True)
        assert _has_reticle_at(ov.canvas, 640, 360), f"reticle missing with control enabled={enabled}"
    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True,
              hand_point=(640, 360), reticle=False)
    assert not _has_reticle_at(ov.canvas, 640, 360)
    ov.update("desktop", MODES, ["desktop"], False, "-", 0.0, 30.0, True, hand_point=None, reticle=True)
    assert not ov.canvas.named("oval") or not any(
        k.get("fill") == "#ffffff" for _, _, k in ov.canvas.named("oval")), "no hand -> no reticle"
    # still visible while the mode menu is open (you point at tiles to pick one)
    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True, menu_open=True,
              hand_point=(300, 300), reticle=True)
    assert _has_reticle_at(ov.canvas, 300, 300)
    ov.close()


def test_pinching_reticle_is_visibly_different_from_the_open_one():
    ov = _overlay()
    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True,
              hand_point=(500, 500), reticle=True, pinching=False)
    open_r = max(c[2] - c[0] for n, c, k in ov.canvas.calls if n == "oval" and len(c) == 4 and k.get("width") == 2)
    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True,
              hand_point=(500, 500), reticle=True, pinching=True)
    pinch_r = max(c[2] - c[0] for n, c, k in ov.canvas.calls if n == "oval" and len(c) == 4 and k.get("width") == 2)
    assert pinch_r < open_r
    ov.close()


def test_ink_upload_touches_only_dirty_tiles_not_the_whole_screen():
    """Regression for the 13 fps drawing mode: only tiles a stroke touched
    are converted/pasted; a frame with no new ink uploads nothing."""
    from overlay_drawing import OverlayDrawing
    _FakePhoto.made = _FakePhoto.pastes = 0
    ov = _overlay()
    od = OverlayDrawing(1920, 1080)

    def frame():
        ov.update("drawing", MODES, ["drawing"], True, "-", 0.0, 30.0, True,
                  ink=od.ink, paint=od.view(), hand_point=(900, 500), reticle=True)

    frame()
    assert _FakePhoto.made == 0 and not ov.canvas.named("image"), "empty canvas -> nothing uploaded or drawn"

    t = 0.0
    for i in range(30):
        t += 1 / 30
        od.update((500 + 6 * i, 500), True, t)
    od.update((700, 500), False, t + 0.03)
    frame()
    n_tiles = len(ov.canvas.named("image"))
    assert 1 <= n_tiles <= 4 and _FakePhoto.made == n_tiles, (n_tiles, _FakePhoto.made)

    made_before = _FakePhoto.made
    frame()                                             # nothing new drawn
    assert _FakePhoto.made == made_before and _FakePhoto.pastes == 0, "idle frame must not re-upload"
    assert len(ov.canvas.named("image")) == n_tiles, "already-uploaded tiles are still placed each frame"
    ov.close()


def test_paint_dock_is_drawn_with_a_tooltip_for_the_hovered_button():
    from overlay_drawing import OverlayDrawing
    ov = _overlay()
    od = OverlayDrawing(1920, 1080)
    b = next(x for x in od.layout() if x["id"] == "eraser")["rect"]
    od.update(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), False, 1.0)      # hover only
    ov.update("drawing", MODES, ["drawing"], True, "-", 0.0, 30.0, True,
              ink=od.ink, paint=od.view(), hand_point=((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), reticle=True)
    texts = [k.get("text") for n, c, k in ov.canvas.calls if n == "text"]
    assert "Eraser" in texts, texts
    assert len(ov.canvas.calls) > 40, "the dock should actually be rendered"
    ov.close()


def _texts(ov):
    return [k.get("text") for n, c, k in ov.canvas.calls if n == "text"]


def test_tutorial_card_call_bar_toast_and_hold_ring_are_drawn_only_when_given():
    import tutorial, callapp
    ov = _overlay()
    base = dict(hand_point=(900, 500), reticle=True)
    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True, **base)
    assert not ov.canvas.named("arc"), "no ring unless a hold is in progress"
    plain = len(ov.canvas.calls)

    t = tutorial.Tutorial(1920, 1080); t.start()
    tv = t.view(); tv["hover"] = "next"
    bar = {"buttons": callapp.call_bar_layout(1920, 1080), "hover": "mute",
           "status": "Zoom detected", "ok": True}
    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True, **base,
              extras={"tutorial": tv, "call": bar, "toast": "Mute / unmute  (Zoom)", "ring": 0.5})
    txt = _texts(ov)
    assert "Point" in txt and "Next" in txt and "Close" in txt and "1 / 6" in txt
    assert any(x and "move your index finger" in x for x in txt)
    assert all(lbl in txt for lbl in ("Mute", "Camera", "Hand", "Share"))
    assert "Zoom detected" in txt and "Mute / unmute  (Zoom)" in txt
    assert ov.canvas.named("arc"), "hold-to-confirm ring around the reticle"
    assert len(ov.canvas.calls) > plain + 30

    # the tutorial card must also show while the mode menu is open (one of its steps IS opening the menu)
    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True, menu_open=True, **base,
              extras={"tutorial": tv})
    assert "Point" in _texts(ov)
    ov.close()


def test_help_panel_exposes_its_tutorial_button_only_while_fully_open():
    ov = _overlay()
    assert ov.help_button_rect() is None
    hx0, hy0, hx1, hy1 = ov._help_tab_rect()
    inside = ((hx0 + hx1) / 2, (hy0 + hy1) / 2)
    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True, hand_point=(900, 300))
    assert ov.help_button_rect() is None, "collapsed pill: no button to press"
    for _ in range(12):                                     # hover until the drawer is fully open
        ov._last_update_t = time.time() - 1 / 30
        ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True, hand_point=inside)
    r = ov.help_button_rect()
    assert r is not None and "Tutorial" in _texts(ov)
    assert "T tutorial" in " ".join(x for x in _texts(ov) if x)
    # it lives inside the panel, so a pointer parked on the pill can reach it
    assert r[2] - r[0] > 60 and r[3] - r[1] > 15
    ov.close()


def test_capture_exclusion_is_applied_once_per_change_not_every_frame():
    import window_capture
    calls = []
    real = window_capture.set_capture_excluded
    window_capture.set_capture_excluded = lambda wid, flag, user32=None: calls.append(flag) or "excluded"
    try:
        ov = _overlay()
        for _ in range(30):
            ov.set_capture_excluded(True)
        assert calls == [True], "30 identical requests must make exactly one OS call"
        ov.set_capture_excluded(False)
        ov.set_capture_excluded(False)
        assert calls == [True, False]
        ov.close()
    finally:
        window_capture.set_capture_excluded = real


def _open_drawer(ov, tab_pt, mode="desktop", frames=14):
    for _ in range(frames):
        ov._last_update_t = time.time() - 1 / 30
        ov.update(mode, MODES, [mode], True, "-", 0.0, 30.0, True, hand_point=tab_pt)


def test_help_drawer_stays_open_while_you_reach_for_the_tutorial_button_inside_it():
    """Regression: the hover test used only the small collapsed tab, so as
    soon as the finger moved up toward the Tutorial button (inside the
    grown panel, outside the tab) the panel collapsed - the tutorial could
    not be reached without a mouse or keyboard."""
    ov = _overlay()
    tx0, ty0, tx1, ty1 = ov._help_tab_rect()
    _open_drawer(ov, ((tx0 + tx1) / 2, (ty0 + ty1) / 2))
    rect = ov.help_button_rect()
    assert rect is not None
    assert not (tx0 <= (rect[0] + rect[2]) / 2 <= tx1 and ty0 <= (rect[1] + rect[3]) / 2 <= ty1), \
        "the button is outside the tab - that's the whole problem"
    on_button = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    for _ in range(90):                                       # 3 seconds of aiming at the button
        ov._last_update_t = time.time() - 1 / 30
        ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True, hand_point=on_button)
    assert ov._help_open > 0.99 and ov.help_button_rect() is not None, "must stay open under the pointer"

    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True, hand_point=(1500, 300))
    assert ov._help_open > 0.9, "grace period: a wobble off the panel must not slam it shut"
    ov._help_hold_until = 0.0
    for _ in range(40):
        ov._last_update_t = time.time() - 1 / 30
        ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True, hand_point=(1500, 300))
    assert ov._help_open < 0.05, "and it does close once you really leave"
    ov.close()


def test_status_drawer_buttons_are_reachable_and_report_camera_and_control_state():
    ov = _overlay()
    tx0, ty0, tx1, ty1 = ov._status_tab_rect()
    _open_drawer(ov, ((tx0 + tx1) / 2, (ty0 + ty1) / 2))
    rects = ov.status_button_rects()
    assert set(rects) == {"camera", "control"}
    cam = rects["camera"]
    on_cam = ((cam[0] + cam[2]) / 2, (cam[1] + cam[3]) / 2)
    assert not (tx0 <= on_cam[0] <= tx1 and ty0 <= on_cam[1] <= ty1)
    for _ in range(60):
        ov._last_update_t = time.time() - 1 / 30
        ov.update("desktop", MODES, ["desktop"], False, "-", 0.0, 30.0, True, hand_point=on_cam,
                  extras={"camera_on": True})
    assert ov.status_button_rects(), "stays open while pointing at its button"
    assert ov._status_hover == "camera"
    t = " ".join(x for x in _texts(ov) if x)
    assert "Camera: ON" in t and "Control: OFF" in t
    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True, hand_point=on_cam,
              extras={"camera_on": False})
    t = " ".join(x for x in _texts(ov) if x)
    assert "Camera: OFF" in t and "Control: ON" in t
    ov.close()


def test_released_camera_banner_is_drawn():
    ov = _overlay()
    ov.update("desktop", MODES, ["desktop"], True, "-", 0.0, 30.0, True,
              extras={"banner": {"title": "Camera released", "lines": ["Press Ctrl+Alt+K to take it back"]}})
    t = _texts(ov)
    assert "Camera released" in t and "Press Ctrl+Alt+K to take it back" in t
    ov.close()


def test_degenerate_source_size_does_not_crash():
    scale, ox, oy, dw, dh = _fit_scale_offset(0, 0, 1920, 1080)
    assert dw >= 1 and dh >= 1


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                f()
                print("PASS ", n)
            except Exception as e:
                fails += 1
                import traceback
                print("FAIL ", n, repr(e))
                traceback.print_exc(limit=4)
    sys.exit(1 if fails else 0)
