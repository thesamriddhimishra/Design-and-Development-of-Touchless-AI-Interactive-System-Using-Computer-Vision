"""Tests for overlay_drawing.py (Tk-free). Run: python tests/test_overlay_drawing.py"""
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from overlay_drawing import InkCanvas, OverlayDrawing, TILE

SW, SH = 1920, 1080


def pinch_stroke(od, pts, t0=0.0):
    """Feed a pinch held over `pts` (one point per frame), then released."""
    t = t0
    for p in pts:
        t += 1 / 30
        od.update(p, True, t)
    od.update(pts[-1], False, t + 1 / 30)
    return t + 1 / 30


def test_stroke_lands_under_the_point_it_was_fed():
    """The whole reason the panel moved to screen space: ink must appear
    where the reticle is, not somewhere a frame->screen stretch put it."""
    od = OverlayDrawing(SW, SH)
    y = 600
    pinch_stroke(od, [(900 + 10 * i, y) for i in range(40)])
    ys, xs = np.nonzero(od.ink.buf[..., 3])
    assert len(xs) > 0
    assert abs(float(ys.mean()) - y) < 6, "ink must sit on the line the point travelled"
    assert 880 < xs.min() and xs.max() < 1400, "and span roughly where the point travelled"


def test_only_touched_tiles_are_dirty_so_a_stroke_does_not_reupload_the_screen():
    od = OverlayDrawing(SW, SH)
    od.ink.pop_dirty()
    pinch_stroke(od, [(500 + 6 * i, 500) for i in range(30)])
    tiles = od.ink.pop_dirty()
    total = od.ink.nty * od.ink.ntx
    assert 0 < len(tiles) <= 4 < total, (len(tiles), total)
    assert od.ink.pop_dirty() == [], "nothing new drawn -> nothing to upload"
    ty, tx = tiles[0]
    assert od.ink.tile_pixels(ty, tx).shape[2] == 4


def test_erasing_empties_tiles_and_clear_wipes_everything():
    od = OverlayDrawing(SW, SH)
    pinch_stroke(od, [(700 + 8 * i, 400) for i in range(20)])
    assert not od.ink.is_empty()
    od.ink.pop_dirty()
    od.tool = "eraser"
    pinch_stroke(od, [(660 + 6 * i, 400) for i in range(60)])
    od.ink.pop_dirty()
    assert od.ink.is_empty(), "a wide eraser pass over the whole stroke removes it"
    assert not od.ink.has_ink.any(), "emptied tiles must stop being drawn"
    od.tool = "pen"
    pinch_stroke(od, [(300 + 8 * i, 300) for i in range(10)])
    od.press("clear")
    assert od.ink.is_empty()


def test_toolbar_needs_a_pinch_click_not_a_hover():
    od = OverlayDrawing(SW, SH)
    clear = next(b for b in od.layout() if b["id"] == "clear")["rect"]
    c = ((clear[0] + clear[2]) / 2, (clear[1] + clear[3]) / 2)
    pinch_stroke(od, [(900 + 5 * i, 500) for i in range(15)])
    assert not od.ink.is_empty()
    for i in range(10):                      # hovering over CLEAR without pinching
        od.update(c, False, 10 + i / 30)
    assert od.hover_id == "clear" and not od.ink.is_empty(), "hover alone must never fire a button"
    od.update(c, True, 11.0)                 # now an actual pinch
    assert od.ink.is_empty()


def test_a_pinch_that_starts_on_the_toolbar_never_draws():
    od = OverlayDrawing(SW, SH)
    b = next(b for b in od.layout() if b["id"] == "color1")["rect"]
    c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    t = 0.0
    od.update(c, True, t)                    # pinch begins on a colour swatch
    for i in range(12):                      # ...and drags out across the canvas
        t += 1 / 30
        od.update((c[0] + 40 * (i + 1), c[1]), True, t)
    assert od.color_idx == 1
    assert od.ink.is_empty(), "dragging out of a toolbar press must not scribble"


def test_colour_pick_thickness_and_eraser_selection():
    od = OverlayDrawing(SW, SH)
    od.press("color3")
    assert od.color_idx == 3 and od.tool == "pen"
    od.press("eraser")
    assert od.tool == "eraser"
    od.press("color2")
    assert od.tool == "pen", "choosing a colour switches back from the eraser"
    od.press("thick2")
    pinch_stroke(od, [(800 + 6 * i, 300) for i in range(20)])
    ys, _ = np.nonzero(od.ink.buf[..., 3])
    assert ys.max() - ys.min() >= od.THICKNESS[2] - 2
    r, g, b, a = od.ink.buf[300, 840]
    assert (r, g, b) == od.COLORS[2][1] and a == 255


def test_hide_collapses_to_one_button_and_show_restores_the_dock():
    od = OverlayDrawing(SW, SH)
    full = len(od.layout())
    assert full > 10
    od.press("hide")
    assert [b["id"] for b in od.layout()] == ["show"]
    od.press("show")
    assert len(od.layout()) == full


def test_layout_fits_the_screen_and_clears_the_top_and_taskbar_zones():
    for sw, sh in ((1920, 1080), (1366, 768), (1280, 720)):
        od = OverlayDrawing(sw, sh)
        x0, y0, x1, y1 = od.dock_rect()
        assert y0 >= 40 and y1 <= sh - 70, (sw, sh, y0, y1)
        assert x0 >= 0 and x1 < sw * 0.2


def test_save_writes_a_png_and_reports_it():
    od = OverlayDrawing(SW, SH)
    pinch_stroke(od, [(500 + 5 * i, 500) for i in range(12)])
    path = os.path.join(tempfile.mkdtemp(), "d.png")
    assert od.save(path).startswith("Saved") and os.path.getsize(path) > 0
    od.press("save")
    assert od.view(time.time())["toast"] is not None


def test_no_hand_or_release_breaks_the_stroke_instead_of_joining_it():
    od = OverlayDrawing(SW, SH)
    t = 0.0
    for i in range(8):
        t += 1 / 30
        od.update((300 + 20 * i, 300), True, t)
    od.update(None, False, t + 0.03)         # hand lost
    t += 2.0
    for i in range(8):
        t += 1 / 30
        od.update((1500 + 20 * i, 800), True, t)
    # nothing may have been drawn between the two separated strokes
    mid = od.ink.buf[:, 700:1300, 3]
    assert not mid.any(), "a lost hand must not draw a line across the screen"


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                f(); print("PASS ", n)
            except Exception as e:
                fails += 1
                import traceback
                print("FAIL ", n, repr(e)); traceback.print_exc(limit=4)
    sys.exit(1 if fails else 0)
