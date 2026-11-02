"""Overlay + 3D integration against a stub tkinter (no display needed).
Run: python tests/test_ar_overlay_integration.py"""
import os, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


class TclError(Exception):
    pass


class _Root:
    def __init__(self): self.calls = []
    def __getattr__(self, name):
        if name == "winfo_screenwidth": return lambda: 1600
        if name == "winfo_screenheight": return lambda: 900
        return lambda *a, **k: None


class _Canvas:
    counts = {}
    def __init__(self, *a, **k): pass
    def pack(self, *a, **k): pass
    def delete(self, *a): _Canvas.counts.clear()
    def __getattr__(self, name):
        if name.startswith("create_"):
            def f(*a, **k): _Canvas.counts[name] = _Canvas.counts.get(name, 0) + 1
            return f
        raise AttributeError(name)


fake = types.ModuleType("tkinter")
fake.Tk, fake.Canvas, fake.TclError = _Root, _Canvas, TclError
fake.Label = lambda *a, **k: _Root()
fake.PIESLICE, fake.ARC = "pieslice", "arc"
sys.modules["tkinter"] = fake

import ar_overlay
import function_graphs
from ar3d import BarChart3D
from ar3d_input import AR3DController, HandState

tmp = tempfile.mkdtemp()
ar_overlay.DATA_FOLDER = os.path.join(tmp, "ar_data")
ar_overlay.WORKSPACE_FILE = os.path.join(tmp, "ws.json")
function_graphs.FUNC_FOLDER = os.path.join(tmp, "ar_functions")


def test_palette_has_3d_button_and_tiles_work():
    ov = ar_overlay.AROverlay(screen_size=(1600, 900))
    kinds = [b["kind"] for b in ov.palette]
    # "functions" then "dense" were added AFTER "shapes3d" (see
    # AROverlay.PALETTE_KINDS) so shapes3d is now third-to-last.
    assert kinds[-4:] == ["shapes3d", "functions", "dense", "clear"] and len(kinds) == 9
    b = palette_btn(ov, "shapes3d")
    pt = ((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2)
    assert ov.palette_kind_at(pt) == "shapes3d"
    assert ov.start_ghost("shapes3d", pt) is None       # must NOT create a bogus 2D chart
    assert ov.shape_tile_at(pt) is None
    ov.toggle_shape_browser()
    assert ov.shapes_open and [t["kind"] for t in ov._shape_tiles] == \
        ["bars3d", "surface3d", "cube", "sphere", "torus"]
    t = ov._shape_tiles[2]
    assert ov.shape_tile_at(((t["x0"] + t["x1"]) / 2, (t["y0"] + t["y1"]) / 2)) == "cube"
    ov.toggle_file_browser()
    assert ov.files_open and not ov.shapes_open          # rows are mutually exclusive
    ov.close()


def test_function_folder_readme_is_not_treated_as_a_function_tile():
    """Regression test: ar_functions/README.txt shares the same .txt
    extension every real function file uses (unlike ar_data/'s CSVs,
    where the README can't collide since it filters by .csv). Without
    an explicit exclusion, the instructions file itself would show up
    as a broken red tile in the Functions browser."""
    assert os.path.isfile(os.path.join(function_graphs.FUNC_FOLDER, "README.txt"))
    names = [os.path.basename(p) for p in function_graphs.list_function_files()]
    assert "README.txt" not in names


def palette_btn(ov, kind):
    """Palette button by KIND, not position - positions shift every time a
    button is added (they did, three times), and no test is about order."""
    ov.set_palette_open(True)      # buttons are only live while the box is open
    return next(b for b in ov.palette if b["kind"] == kind)


def test_clear_all_needs_a_confirming_second_press_and_wipes_2d_and_3d():
    """The CLEAR button exists because removing graphs one by one with a
    fist is unreliable and tedious - but a button that wipes the whole
    workspace on ONE stray pinch would be a worse problem than the one it
    solves, so the first press only arms it (red, "SURE?") and a second
    press within CLEAR_CONFIRM_S clears. An unconfirmed arm times out."""
    from ar3d import BarChart3D
    ov = ar_overlay.AROverlay(screen_size=(1600, 900))
    assert "clear" in [b["kind"] for b in ov.palette]
    assert ov.start_ghost("clear", (100, 100)) is None        # never turns into a chart

    ov.charts.append(ar_overlay.ChartWidget(400, 300, kind="bar"))
    ov.scene3d.add(BarChart3D(900, 400))
    assert ov.charts and ov.scene3d.objects

    t0 = 1000.0
    assert ov.press_clear(t0) == "armed"
    assert ov.charts and ov.scene3d.objects, "first press must NOT clear anything"
    assert ov.press_clear(t0 + ov.CLEAR_CONFIRM_S + 1) == "armed", "an expired arm starts over, doesn't clear"
    assert ov.charts and ov.scene3d.objects
    assert ov.press_clear(t0 + ov.CLEAR_CONFIRM_S + 2) == "cleared"
    assert not ov.charts and not ov.scene3d.objects
    assert ov.scene3d.trash, "cleared 3D objects stay recoverable via undo"
    ov.close()


def test_toolbox_is_a_small_tab_that_opens_on_hover_and_closes_after_a_grace_period():
    """Regression for "the toolbar blocks the whole top-left of the screen
    whenever AR mode is on": by default only a small tab is drawn and no
    button is live; pointing at the tab opens a compact box; it closes a
    moment after the pointer leaves, and stays open while something is
    being picked up or CLEAR is armed."""
    ov = ar_overlay.AROverlay(screen_size=(1920, 1080))
    tx0, ty0, tx1, ty1 = ov.tab_rect
    px0, py0, px1, py1 = ov.panel_rect
    # the box is a lot smaller than the old full-width row (9 x 114 px wide)
    assert (px1 - px0) < 400 and (py1 - py0) < 340
    assert (tx1 - tx0) < 140 and (ty1 - ty0) < 60

    shape_btn = next(b for b in ov.palette if b["kind"] == "shapes3d")
    over_button = ((shape_btn["x0"] + shape_btn["x1"]) / 2, (shape_btn["y0"] + shape_btn["y1"]) / 2)
    assert ov.palette_kind_at(over_button) is None, "collapsed: invisible buttons must not be pressable"

    tab_pt = ((tx0 + tx1) / 2, (ty0 + ty1) / 2)
    t = [1000.0]
    def tick(pt, n=1, dt=1 / 30):
        for _ in range(n):
            t[0] += dt
            ov._update_palette_state(pt, t[0])

    tick(None, 5)
    assert ov._pal_open == 0.0
    tick(tab_pt, 1)
    assert 0.0 < ov._pal_open < 1.0, "opens by easing, not by snapping"
    tick(tab_pt, 20)
    assert ov._pal_open == 1.0 and ov.palette_visible()
    assert ov.palette_kind_at(over_button) == "shapes3d"

    tick(over_button, 30)                        # moving onto a button keeps it open
    assert ov._pal_open == 1.0
    tick((1500, 900), 5)                         # pointer leaves: still open within the grace period
    assert ov._pal_open > 0.9
    tick((1500, 900), 40)                        # ...then it closes
    assert ov._pal_open == 0.0
    assert ov.palette_kind_at(over_button) is None

    # holding a ghost keeps it open (it's where you drop to cancel)
    ov.ghost = ar_overlay.ChartWidget(900, 500, kind="bar")
    tick(tab_pt, 20); tick((1500, 900), 60)
    assert ov._pal_open == 1.0, "must stay open while a pick-up is in progress"
    ov.ghost = None
    tick((1500, 900), 60)
    assert ov._pal_open == 0.0
    # ...and while CLEAR is armed, so "SURE?" is visible
    ov.press_clear(t[0])
    tick((1500, 900), 30)
    assert ov._pal_open == 1.0
    ov.close()


def test_collapsed_toolbox_draws_far_less_than_the_open_box():
    ov = ar_overlay.AROverlay(screen_size=(1920, 1080))
    ov.update(cursor_point=(1500, 900))
    collapsed = sum(_Canvas.counts.values())
    ov.set_palette_open(True)
    ov.update(cursor_point=(60, 60))
    opened = sum(_Canvas.counts.values())
    assert collapsed < 25, collapsed
    assert opened > collapsed * 4, (collapsed, opened)
    ov.close()


def test_browser_rows_sit_right_of_the_box_not_under_it():
    ov = ar_overlay.AROverlay(screen_size=(1920, 1080))
    bx, by = ov.browser_origin()
    assert bx > ov.panel_rect[2] and by < ov.panel_rect[3]
    ov.toggle_file_browser()
    if ov._file_tiles:
        assert all(tile["x0"] >= bx for tile in ov._file_tiles)
    ov.close()


def test_hovered_palette_button_shows_a_charge_arc_only_while_a_pinch_is_closing():
    """Interactive feedback: the hovered palette button grows a charge-up
    arc as cursor_approach (how close the pinch is to closing) rises, so
    committing has a visible "about to fire" moment. No arc when nothing
    is hovered or the fingers aren't closing."""
    ov = ar_overlay.AROverlay(screen_size=(1600, 900))
    ov.set_palette_open(True)
    b = ov.palette[0]
    kind = b["kind"]

    ov.update(hover_kind=kind, cursor_approach=0.0)
    arcs_idle = _Canvas.counts.get("create_arc", 0)
    ov.update(hover_kind=kind, cursor_approach=0.7)
    arcs_closing = _Canvas.counts.get("create_arc", 0)
    ov.update(hover_kind=None, cursor_approach=0.7)
    arcs_unhovered = _Canvas.counts.get("create_arc", 0)

    assert arcs_closing == arcs_idle + 1, "exactly one charge arc while hovering + closing"
    assert arcs_unhovered == arcs_idle, "no charge arc when no button is hovered"
    ov.close()


def test_dense_toggle_button_fills_surfaces_with_real_faces():
    """The "DENSE" palette button: a straight toggle (no browser, no
    ghost) that fills Surface3D's grid with real shaded quad faces
    instead of leaving it hollow lines-over-nothing - see Scene3D.
    toggle_dense/Surface3D.build's own "if self.dense" branch."""
    ov = ar_overlay.AROverlay(screen_size=(1600, 900))
    kinds = [b["kind"] for b in ov.palette]
    assert "dense" in kinds
    assert ov.scene3d.dense_mode is False

    db = palette_btn(ov, "dense")
    pt = ((db["x0"] + db["x1"]) / 2, (db["y0"] + db["y1"]) / 2)
    assert ov.palette_kind_at(pt) == "dense"

    from ar3d import Surface3D
    s = Surface3D(500, 350)
    ov.scene3d.objects.append(s)
    assert len(s.build().faces) == 0, "no selection and dense off -> no faces at all yet"

    ctl = AR3DController(ov.scene3d)
    absent = HandState()
    t = 30.0
    def frame(point, pin):
        nonlocal t
        t += 1 / 30
        return ctl.step([HandState(True, point, pin, 100.0), absent], t, ov)
    frame(pt, False); frame(pt, True); frame(pt, False)   # pinch the DENSE button
    assert ov.scene3d.dense_mode is True
    assert s.dense is True, "an already-placed surface must pick up the toggle immediately"

    assert len(s.build().faces) > 10, "dense mode should add a real grid of filled faces"
    assert ov.scene3d.effective_item_budget() > ov.scene3d.item_budget, \
        "dense mode should also widen the LOD system's budget, not just add items to fight it"

    # Toggling back off must return to the light, hollow look.
    frame(pt, False); frame(pt, True); frame(pt, False)
    assert ov.scene3d.dense_mode is False and s.dense is False
    assert len(s.build().faces) == 0
    # This test's own surface was added directly (not via the normal
    # pick-up/drop flow) purely to check the toggle - removed again
    # before closing so it doesn't leak into the shared workspace file
    # other tests in this module reuse (ov.close() saves whatever's
    # still in scene3d.objects).
    ov.scene3d.objects.remove(s)
    ov.close()


def test_functions_palette_tiles_and_placement():
    """The "\u0192(x)" button: opens a row of tiles (one per starter .txt
    file, seeded automatically), picking a valid one places a Surface3D
    carrying that expression, and a deliberately broken .txt shows up
    as an error tile rather than crashing the browser."""
    ov = ar_overlay.AROverlay(screen_size=(1600, 900))
    kinds = [b["kind"] for b in ov.palette]
    assert "functions" in kinds
    fb = palette_btn(ov, "functions")
    pt = ((fb["x0"] + fb["x1"]) / 2, (fb["y0"] + fb["y1"]) / 2)
    assert ov.palette_kind_at(pt) == "functions"
    assert ov.function_tile_at(pt) is None              # closed: no tiles yet
    assert os.path.isdir(function_graphs.FUNC_FOLDER)   # seeded on first AROverlay()

    ov.toggle_function_browser()
    assert ov.functions_open and not ov.shapes_open and not ov.files_open
    assert len(ov._function_tiles) >= 1
    assert all(t["error"] is None for t in ov._function_tiles), \
        "a starter function file failed to compile"

    # A broken file should show up as an error tile, not crash the scan.
    # Named to sort FIRST (not last) so it stays within the tile row's
    # shared 6-tile budget (see _scan_functions/_scan_files) regardless
    # of how many starter functions happen to be shipped alongside it.
    with open(os.path.join(function_graphs.FUNC_FOLDER, "aa_broken.txt"), "w") as f:
        f.write("sin(x) +")   # invalid syntax
    ov.toggle_function_browser(); ov.toggle_function_browser()   # close, reopen = rescans
    broken = [t for t in ov._function_tiles if t["name"] == "aa_broken"]
    assert broken and broken[0]["error"] is not None

    good_tile = next(t for t in ov._function_tiles if t["error"] is None)
    ctl = AR3DController(ov.scene3d)
    absent = HandState()
    tp = ((good_tile["x0"] + good_tile["x1"]) / 2, (good_tile["y0"] + good_tile["y1"]) / 2)
    t = 20.0
    def frame(point, pin):
        nonlocal t
        t += 1 / 30
        return ctl.step([HandState(True, point, pin, 100.0), absent], t, ov)
    frame(tp, False); frame(tp, True)                   # pinch the tile
    assert ov.scene3d.ghost is not None and ov.scene3d.ghost.kind == "surface3d"
    assert ov.scene3d.ghost.expr is not None
    frame((800, 500), True); frame((800, 500), False)    # drag + drop
    placed = [o for o in ov.scene3d.objects if o.kind == "surface3d"]
    assert len(placed) == 1 and placed[0].expr is not None
    ov.close()                                            # saves

    ov2 = ar_overlay.AROverlay(screen_size=(1600, 900))   # "next launch"
    restored = [o for o in ov2.scene3d.objects if o.kind == "surface3d"]
    assert len(restored) == 1 and restored[0].expr == placed[0].expr
    ov2.close()


def test_place_via_controller_draw_save_and_restore():
    if os.path.exists(ar_overlay.WORKSPACE_FILE): os.remove(ar_overlay.WORKSPACE_FILE)
    ov = ar_overlay.AROverlay(screen_size=(1600, 900))
    ctl = AR3DController(ov.scene3d)
    absent = HandState()
    t = 10.0
    def frame(pt, pin):
        nonlocal t
        t += 1 / 30
        r = ctl.step([HandState(True, pt, pin, 100.0), absent], t, ov)
        ov.update(cursor_point=pt, pinching=pin, hover_shape=ov.shape_tile_at(pt))
        return r
    b = palette_btn(ov, "shapes3d")
    btn = ((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2)
    frame(btn, False); assert frame(btn, True) is True; frame(btn, False)   # open tiles
    assert ov.shapes_open
    tile = ov._shape_tiles[0]                            # BARS
    tp = ((tile["x0"] + tile["x1"]) / 2, (tile["y0"] + tile["y1"]) / 2)
    frame(tp, False); frame(tp, True)
    assert ov.scene3d.ghost is not None
    frame((800, 500), True); frame((800, 500), False)
    assert len(ov.scene3d.objects) == 1 and ov.scene3d.objects[0].kind == "bars3d"
    obj = ov.scene3d.objects[0]
    obj.select_start(("bar", 1)); obj.select_update(("bar", 3)); obj.stretch_begin(); obj.stretch_apply(1.6, 1.2)
    ov.scene3d.dirty = True
    for _ in range(5): frame((100, 800), False)
    assert _Canvas.counts.get("create_polygon", 0) > 0 and _Canvas.counts.get("create_line", 0) > 0
    ov.close()                                           # saves
    assert os.path.exists(ar_overlay.WORKSPACE_FILE)
    ov2 = ar_overlay.AROverlay(screen_size=(1600, 900))  # "next launch"
    assert len(ov2.scene3d.objects) == 1
    o2 = ov2.scene3d.objects[0]
    assert o2.selection == (1, 3) and o2.wx == obj.wx and abs(o2.pos[0] - 800) < 5
    ov2.update()
    ov2.close()


def test_2d_charts_and_3d_coexist_in_one_workspace():
    if os.path.exists(ar_overlay.WORKSPACE_FILE): os.remove(ar_overlay.WORKSPACE_FILE)
    ov = ar_overlay.AROverlay(screen_size=(1600, 900))
    ov.start_ghost("bar", (400, 400)); ov.commit_ghost()
    ov.scene3d.add(BarChart3D(900, 400))
    ov.close()
    ov2 = ar_overlay.AROverlay(screen_size=(1600, 900))
    assert len(ov2.charts) == 1 and len(ov2.scene3d.objects) == 1
    ov2.close()


def test_old_workspace_without_3d_key_still_loads():
    import json
    json.dump({"charts": [{"x": 300, "y": 300, "kind": "bar", "data": [.5, .6]}]},
              open(ar_overlay.WORKSPACE_FILE, "w"))
    ov = ar_overlay.AROverlay(screen_size=(1600, 900))
    assert len(ov.charts) == 1 and ov.scene3d.objects == []
    ov.close()


def test_preview_frame_conversion_is_correct_fast_and_has_a_working_fallback():
    import numpy as np, time
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, :640] = (255, 0, 0)            # BGR blue on the left half
    frame[:, 640:] = (0, 0, 255)            # BGR red on the right half
    out = ar_overlay.frame_to_preview_rgb(frame, 260, 146)
    assert out.shape == (146, 260, 3) and out.dtype == np.uint8
    assert tuple(out[70, 20]) == (0, 0, 255) and tuple(out[70, 240]) == (255, 0, 0)   # RGB: blue left, red right
    t = time.perf_counter()
    for _ in range(40): ar_overlay.frame_to_preview_rgb(frame, 260, 146)
    assert (time.perf_counter() - t) / 40 < 0.006                                      # was ~13 ms
    saved = ar_overlay._cv2
    try:
        ar_overlay._cv2 = None                                                          # no-OpenCV fallback path
        out2 = ar_overlay.frame_to_preview_rgb(frame, 260, 146)
        assert out2.shape == (146, 260, 3) and tuple(out2[70, 20]) == (0, 0, 255)
    finally:
        ar_overlay._cv2 = saved


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS ", n)
            except Exception as e:
                fails += 1; import traceback; print("FAIL ", n, repr(e)); traceback.print_exc(limit=4)
    sys.exit(1 if fails else 0)
