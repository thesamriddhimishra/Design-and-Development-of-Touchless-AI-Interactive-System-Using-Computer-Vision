"""The optimised renderer (vectorised sort/cull, cached colours, LOD) must
draw the SAME picture as the original at full detail. Compares every canvas
call against the saved v24 engine (tests/reference/ar3d_v24.py).
Run: python tests/test_render_equivalence.py"""
import importlib.util, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import ar3d as new

spec = importlib.util.spec_from_file_location("ar3d_ref", os.path.join(HERE, "reference", "ar3d_v24.py"))
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)


class Rec:
    def __init__(self): self.calls = []
    def _r(self, name, coords, **kw):
        self.calls.append((name, tuple(round(float(c), 3) for c in coords), kw))
    def create_line(self, *c, **kw): self._r("line", c, **kw)
    def create_polygon(self, *c, **kw): self._r("poly", c, **kw)
    def create_rectangle(self, *c, **kw): self._r("rect", c, **kw)
    def create_oval(self, *c, **kw): self._r("oval", c, **kw)
    def create_text(self, x, y, **kw): self._r("text", (x, y), **kw)


def build(mod):
    random.seed(11)
    sc = mod.Scene3D((1600, 900))
    b = mod.BarChart3D(300, 450, size=150)
    s = mod.Surface3D(800, 450, size=140, func=1)
    s2 = mod.Surface3D(1250, 450, size=120, func=3)
    c = mod.Shape3D("cube", 300, 160, size=70)
    sp = mod.Shape3D("sphere", 800, 160, size=70)
    t = mod.Shape3D("torus", 1250, 160, size=70)
    for o in (b, s, s2, c, sp, t):
        sc.add(o)
        # Idle sway/spin motion is expected to keep evolving as its own
        # thing (phase/frequency variety, secondary tilt axes, etc. -
        # see Object3D.update's own comments) independently of whatever
        # this test is actually meant to guard: that an OPTIMIZATION
        # pass (vectorised sort/cull, cached colours, LOD) draws the
        # same picture as the original. Zeroing sway here decouples the
        # two - this test now compares pure geometry/culling/LOD/colour
        # logic at a fixed pose, not whichever idle-animation flourish
        # happens to be implemented today. (The REST of build() below -
        # selection, stretch, 60 steps of dt accumulation for easing/
        # caching to settle - still happens exactly as before.)
        o.sway = 0.0
    b.select_start(("bar", 1)); b.select_update(("bar", 3)); b.stretch_begin(); b.stretch_apply(1.7, 1.3)
    s.select_start(("cell", (4, 4))); s.select_update(("cell", (12, 11))); s.stretch_begin(); s.stretch_apply(1.5, 1.8)
    for _ in range(60):
        sc.step(1 / 30)
    return sc


def hexrgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _is_shade_of(color_hex, base_hex, tol=0.04):
    """True if color_hex looks like base_hex passed through ar3d.shade()
    (same r:g:b ratio, just scaled brighter/dimmer - shade() multiplies
    all three channels by the same factor)."""
    cr, cg, cb = hexrgb(color_hex)
    br, bg, bb = hexrgb(base_hex)
    if max(cr, cg, cb) == 0 or max(br, bg, bb) == 0:
        return cr == br and cg == bg and cb == bb
    c_norm = [v / max(cr, cg, cb) for v in (cr, cg, cb)]
    b_norm = [v / max(br, bg, bb) for v in (br, bg, bb)]
    return all(abs(x - y) <= tol for x, y in zip(c_norm, b_norm))


def _is_expected_bar_heatmap_drift(ref_color, new_color):
    """BarChart3D bars used to all share ONE flat colour (ref.HOLO,
    shaded for lighting) - deliberately changed to a per-bar heat-map
    colour (new.BUCKETS[...], picked by that bar's data value) as part
    of making AR mode's 3D objects less visually flat/same-colour. This
    one, specific, documented difference is excluded from the colour-
    drift check below rather than loosening that check generally - every
    other object (surfaces, shapes) and every other aspect of a bar
    (geometry, selection highlight, line widths, draw order) still has
    to match the v24 reference exactly."""
    if not _is_shade_of(ref_color, ref.HOLO):
        return False
    return any(_is_shade_of(new_color, bucket) for bucket in new.BUCKETS)


def test_full_detail_matches_reference_call_for_call():
    a, b = build(ref), build(new)
    ra, rb = Rec(), Rec()
    a.draw(ra)
    b.lod = 0
    b.draw(rb)
    # One deliberate, documented extra item per BarChart3D/Surface3D
    # object (3 of the 6 objects build() creates): a dark "halo" line
    # behind the floor/body outline so it stays visible over a light
    # background - see the matching comment next to each m.line(...,
    # "#0a0a0a", ...) call in ar3d.py. Filtered out by its distinctive
    # colour before the call-for-call comparison below, rather than
    # loosening the count/order check generally - everything else about
    # line/face count and ordering still has to match exactly.
    def is_halo(c):
        # The halo's base colour (#0a0a0a) is itself depth-shaded
        # (shade_q) like every other line before it reaches the canvas,
        # so it won't match that exact string - "all channels quite
        # dark" is what actually identifies it (nothing else in this
        # scene draws a near-black line).
        fill = c[2].get("fill")
        return c[0] == "line" and isinstance(fill, str) and fill.startswith("#") \
            and max(hexrgb(fill)) < 40
    n_halo = sum(1 for c in rb.calls if is_halo(c))
    assert n_halo == 3, f"expected exactly 3 halo lines (bar + 2 surfaces), got {n_halo}"
    rb.calls = [c for c in rb.calls if not is_halo(c)]
    # Likewise one dark label "pill" rectangle per labelled object (6) -
    # see render_object's label drawing; identified by its fixed fill.
    is_pill = lambda c: c[0] == "rect" and c[2].get("fill") == new.LABEL_PILL
    n_pill = sum(1 for c in rb.calls if is_pill(c))
    assert n_pill == 6, f"expected one label pill per object (6), got {n_pill}"
    rb.calls = [c for c in rb.calls if not is_pill(c)]
    assert len(ra.calls) == len(rb.calls), (len(ra.calls), len(rb.calls))
    key = lambda c: (c[0], c[1])
    A, B = sorted(ra.calls, key=key), sorted(rb.calls, key=key)
    worst = 0
    for (na, ca, ka), (nb, cb, kb) in zip(A, B):
        assert na == nb and ca == cb, (na, ca, nb, cb)
        is_label = (na == "text")
        # Label text keeps its position and wording but deliberately has a
        # new font/colour (readable-on-any-background restyle).
        for k in ("width", "dash", "stipple", "text", "font", "anchor"):
            if is_label and k == "font":
                continue
            assert ka.get(k) == kb.get(k), (k, ka.get(k), kb.get(k))
        for k in ("fill", "outline"):
            if is_label:
                continue
            if isinstance(ka.get(k), str) and ka[k].startswith("#"):
                if _is_expected_bar_heatmap_drift(ka[k], kb[k]):
                    continue  # see helper docstring - one deliberate, documented exception
                worst = max(worst, max(abs(x - y) for x, y in zip(hexrgb(ka[k]), hexrgb(kb[k]))))
    assert worst <= 8, f"colour drift {worst}/255 (depth shading is quantised to 1/32 steps)"
    # draw ORDER: lines far->near must match modulo exact depth ties
    order_a = [c[1][:2] for c in ra.calls if c[0] == "line"]
    order_b = [c[1][:2] for c in rb.calls if c[0] == "line"]
    same = sum(x == y for x, y in zip(order_a, order_b)) / len(order_a)
    assert same > 0.98, same


def test_lod_reduces_items_and_budget_picks_the_right_level():
    sc = build(new)
    full, half, coarse = (sc.total_items(l) for l in (0, 1, 2))
    assert full > half > coarse and coarse < full * 0.35, (full, half, coarse)
    sc.lod = 0; sc.item_budget = full + 10
    sc._choose_lod(); assert sc.lod == 0                     # fits: stay at full detail
    sc.item_budget = full - 1
    sc._choose_lod(); assert sc.lod >= 1                     # doesn't fit: drop detail
    sc.item_budget = 1
    sc._choose_lod(); assert sc.lod == 2                     # nothing fits: coarsest
    for lod in (0, 1, 2):                                    # every level renders
        sc.lod = lod
        r = Rec(); sc.draw(r)
        assert len(r.calls) > 10
    # hysteresis: from LOD 1, a budget that only JUST fits full detail must not flip back
    sc.lod = 1; sc.item_budget = full + 5
    sc._choose_lod(); assert sc.lod == 1
    sc.item_budget = int(full / 0.8) + 2                     # comfortable headroom: now it refines
    sc._choose_lod(); assert sc.lod == 0


def test_selection_is_still_visible_at_every_detail_level():
    sc = build(new)
    surf = sc.objects[1]                                     # its only polygon IS the selection footprint
    for lod in (0, 1, 2):
        r = Rec(); new.render_object(r, surf, lod=lod)
        assert any(c[0] == "poly" for c in r.calls), f"selection footprint missing at lod {lod}"
        assert any(c[0] == "line" and c[2].get("fill", "").lower() != "" for c in r.calls)


def test_geometry_cache_skips_unchanged_objects_but_not_changed_ones():
    sc = build(new)
    o = sc.objects[0]
    calls = {"n": 0}
    orig = o.geometry
    def counting(*a, **k): calls["n"] += 1; return orig(*a, **k)
    o.geometry = counting
    o.sway = 0.0; o._sway_gain = 0.0; o._spin_gain = 0.0      # calm
    sc.step(1 / 30); base = calls["n"]
    for _ in range(10): sc.step(1 / 30)
    assert calls["n"] == base, "a calm, unmoved object must not be re-projected"
    o.pos[0] += 5
    sc.step(1 / 30)
    assert calls["n"] == base + 1
    o.size *= 1.1
    sc.step(1 / 30)
    assert calls["n"] == base + 2
    o.select_start(("bar", 2)); o.select_update(("bar", 4))
    sc.step(1 / 30)
    assert calls["n"] >= base + 3, "changing the selection changes the mesh"


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS ", n)
            except Exception as e:
                fails += 1; import traceback; print("FAIL ", n, repr(e)); traceback.print_exc(limit=3)
    sys.exit(1 if fails else 0)
