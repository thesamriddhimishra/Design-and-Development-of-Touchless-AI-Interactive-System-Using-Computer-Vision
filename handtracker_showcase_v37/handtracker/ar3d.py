"""
ar3d.py
-------
Software-3D "hologram" objects for AR mode: 3D bar charts, 3D surface
graphs and wireframe shapes you can move, resize, rotate, select a
portion of, and stretch/compress.

WHY SOFTWARE 3D (no OpenGL): AR mode's whole trick is a transparent,
click-through Tk canvas on top of the desktop (see ar_overlay.py). A GL
window can't be made transparent like that portably, and wireframe +
stippled faces is exactly the Iron-Man-hologram look anyway. Everything
here is numpy math + plain canvas calls (create_line / create_polygon /
create_text), so it needs no new dependency and can be unit-tested with
a fake canvas (see tests/test_ar3d.py).

COORDINATES: screen pixels, +x right, +y DOWN, +z away from the viewer
(right-handed). Each object has `pos` = (screen x, screen y, depth z):
x/y are where its centre appears on screen; z only changes apparent
size/parallax (perspective scale = FOCAL / (FOCAL + z)). `rot` is a 3x3
local->world matrix, `size` is world pixels per local unit.

NOT VERIFIED ON A REAL DISPLAY: geometry, picking and stretching are
covered by tests, and frames were rendered offscreen to check them, but
Tk's stipple/line rendering on your machine is untested.
"""
import math
import random
from collections import namedtuple

import numpy as np

FOCAL = 900.0
Z_MIN, Z_MAX = -450.0, 1200.0
SIZE_MIN, SIZE_MAX = 45.0, 460.0
MAX_OBJECTS = 6

HOLO = "#39e6ff"
HOT = "#ffb84d"
GREEN = "#39ff88"
PINK = "#ff5d8f"
VIOLET = "#c58cff"
BUCKETS = ("#2f8cff", "#39e6ff", "#39ff88", "#ffe14d", "#ff7a4d")

UI_FONT = "Segoe UI"            # falls back to Tk's default family where it isn't installed
LABEL_PILL = "#06080c"
# Label text: white-ish/cyan on the dark pill reads on any background; the
# object's own accent colour is only used when it's bright enough to read.
LABEL_TEXT = {HOLO: "#9ff4ff", GREEN: "#a8ffd0", VIOLET: "#e0c8ff"}

SHAPE_KINDS = ("bars3d", "surface3d", "cube", "sphere", "torus")


# -- small math helpers ------------------------------------------------
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def orthonormalize(R):
    """Snap a drifting matrix back to a true rotation (repeated small
    incremental rotations slowly skew it otherwise)."""
    u, _, vt = np.linalg.svd(R)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def _rgb(h):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def shade(color, k):
    """Multiply a #rrggbb colour's brightness by k (0..1.3), clamped."""
    r, g, b = _rgb(color)
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(c * k))) for c in (r, g, b))


def _short_label(expr, max_len=16):
    """A typed expression as a surface's floating label - trimmed so a
    long one doesn't dwarf the object it's labelling (same text spot
    presets use, just "SURFACE 3D" replaced with the actual formula)."""
    text = expr.strip()
    return text if len(text) <= max_len else text[:max_len - 1] + "\u2026"


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _point_in_poly(px, py, xs, ys):
    inside = False
    n = len(xs)
    j = n - 1
    for i in range(n):
        if (ys[i] > py) != (ys[j] > py):
            xin = (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i] + 1e-12) + xs[i]
            if px < xin:
                inside = not inside
        j = i
    return inside


def _seg_range(a, b, n_vertices):
    """Vertex-index range [a,b] -> segment range [lo,hi) with hi>lo, so a
    single-vertex selection still covers at least one segment."""
    lo, hi = min(a, b), max(a, b)
    if hi <= lo:
        if hi < n_vertices - 1:
            hi = lo + 1
        else:
            lo = hi - 1
    return max(0, lo), min(n_vertices - 1, hi)


Line = namedtuple("Line", "idx color width layer lod_lo lod_hi group")


class Mesh:
    """Local-space geometry for one frame. faces: (idx, normal, colour,
    stipple, outline). lines: Line records (polylines), layer 0 = behind
    faces, 1 = in front. `lod_lo..lod_hi` = detail levels the line is drawn
    at (0 = full); `group` = (axis, k, is_end) for grid lines that may be
    dropped at coarser detail. parts: (key, idx) used for picking. texts:
    (local xyz, text, colour).

    finish() also builds flat arrays so the renderer can depth-sort every
    line and cull every face with a couple of vectorised numpy calls
    instead of thousands of tiny ones."""

    def __init__(self):
        self.verts = []
        self.faces, self.lines, self.parts, self.texts = [], [], [], []
        self.body = None       # idx of the 4 floor-plate corners
        self.region = None     # idx of a closed outline (surface data area)
        self._masks = {}

    def add(self, pts):
        start = len(self.verts)
        self.verts.extend(pts)
        return np.arange(start, start + len(pts))

    def line(self, idx, color, width, layer=1, lod=(0, 2), group=None):
        self.lines.append(Line(np.asarray(idx), color, width, layer, lod[0], lod[1], group))

    def finish(self):
        self.verts = np.array(self.verts, dtype=float).reshape(-1, 3)
        n = len(self.lines)
        self.n_lines = n
        if n:
            self.line_cnt = np.array([len(l.idx) for l in self.lines], dtype=float)
            self.line_off = np.concatenate(([0], np.cumsum(self.line_cnt[:-1]))).astype(int)
            self.line_cat = np.concatenate([l.idx for l in self.lines]).astype(int)
            self.line_layer = [l.layer for l in self.lines]
        if self.faces:
            self.face_idx = np.array([f[0] for f in self.faces], dtype=int)
            self.face_normals = np.array([f[1] for f in self.faces], dtype=float)
        else:
            self.face_idx = np.zeros((0, 4), dtype=int)
            self.face_normals = np.zeros((0, 3))
        return self

    def lod_mask(self, lod):
        """Which lines are drawn at detail level `lod` (0 full .. 2 coarse)."""
        m = self._masks.get(lod)
        if m is None:
            m = []
            for l in self.lines:
                ok = l.lod_lo <= lod <= l.lod_hi and len(l.idx) >= 2
                if ok and lod >= 1 and l.group is not None:
                    ok = l.group[1] % 2 == 0 or l.group[2]     # keep every other grid line + both edges
                m.append(ok)
            self._masks[lod] = m
        return m

    def item_cost(self, lod):
        """Canvas items this mesh costs at `lod` (what actually limits Tk)."""
        return sum(self.lod_mask(lod)) + len(self.faces) // 2 + 2 * len(self.texts)


def _box(x0, x1, y0, y1, z0, z1):
    """8 corners; index = 4*xi + 2*yi + zi (0=min,1=max)."""
    return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


# face definitions for _box(): (corner indices, outward normal)
_BOX_FACES = (
    ((0, 1, 3, 2), (-1, 0, 0)), ((4, 5, 7, 6), (1, 0, 0)),
    ((0, 1, 5, 4), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)),
    ((0, 2, 6, 4), (0, 0, -1)), ((1, 3, 7, 5), (0, 0, 1)),
)


class Geom:
    def __init__(self, mesh, W, sx, sy, z, s):
        self.mesh, self.W, self.sx, self.sy, self.z, self.s = mesh, W, sx, sy, z, s


class Object3D:
    """Base: transform, physics, picking helpers, persistence."""
    kind = "object"
    supports_selection = False
    _next_id = 1

    def __init__(self, x, y, z=0.0, size=140.0, label=None):
        self.id = Object3D._next_id
        Object3D._next_id += 1
        self.pos = np.array([float(x), float(y), float(z)])
        self.rot = rot_x(0.5)
        self.size = float(size)
        self.vx = self.vy = 0.0
        self.spin = 0.0            # rad/s turntable about screen-vertical
        self.sway = 0.0            # idle wobble amplitude (rad), aids depth cue
        self._sway_gain = 1.0
        # Unique-per-object phase/frequency (derived from the object's
        # own id, so it's stable across frames without needing to store
        # anything extra) - every object used to swing on the exact same
        # single Y-axis, same frequency, same phase, as the shared scene
        # clock `t` directly, which reads as one flat metronome rather
        # than several independent objects idling. 2.399 is the "golden
        # angle" in radians - irrational relative to 2*pi, so successive
        # ids spread around the phase circle instead of clustering or
        # repeating a short cycle the way a small rational step would.
        self._sway_phase = (self.id * 2.399) % (2 * math.pi)
        self._sway_freq = 0.55 + (self.id % 7) * 0.045
        self._spin_gain = 1.0
        self.color = HOLO
        self.label = label or self.kind.upper()
        self.selection = None
        self._disp_rot = self.rot.copy()
        self._geom = None
        self._gkey = None
        self._last_move = None
        # Transient "pop in" scale multiplier (1.0 = settled). Separate
        # from self.size on purpose: size is the object's REAL footprint
        # - persisted to disk, driven by resize gestures, read by
        # picking - and a purely visual entrance animation shouldn't be
        # able to leak into any of that (e.g. saving mid-pop and
        # restoring a tiny object next launch).
        self._pop = 1.0

    @property
    def eff_size(self):
        """size as currently DRAWN (size x the transient pop-in
        multiplier) - geometry() and every picking routine that
        re-projects vertices by hand (Surface3D.nearest_part etc.) must
        use this, not self.size, or hit-testing would briefly disagree
        with what's on screen right after something is placed."""
        return self.size * self._pop

    def start_pop_in(self):
        """Called by Scene3D.add() the moment an object becomes real
        (dropped from a ghost, or restored on launch): starts it small
        and lets update() ease it up to full size, so placing something
        has a little physical "settles into place" moment instead of
        the object just appearing at full size between one frame and
        the next."""
        self._pop = 0.35

    # -- per-frame ------------------------------------------------------
    def update(self, dt, t, calm):
        """calm=True while a hand holds / hovers / is selecting on this
        object: idle sway and spin ease to a stop so the thing you are
        trying to pinch isn't moving under your fingertip."""
        target = 0.0 if calm else 1.0
        self._spin_gain += (target - self._spin_gain) * min(1.0, dt * 8)
        if self.spin and self._spin_gain > 0.01:
            self.rot = orthonormalize(rot_y(self.spin * self._spin_gain * dt) @ self.rot)
        self._sway_gain += (target - self._sway_gain) * min(1.0, dt * 8)
        # Primary sway (Y-axis, left-right) at this object's own
        # phase/frequency, PLUS a smaller secondary tilt (X-axis) at a
        # deliberately non-harmonic frequency ratio (0.63x) and a
        # different phase multiplier (1.7x) - the two never fall back
        # into lockstep with each other the way a simple multiple would,
        # which is what turns a single flat left-right swing into
        # something that reads as gently alive/orbiting instead.
        amp = self.sway * self._sway_gain
        yaw = amp * math.sin(t * self._sway_freq + self._sway_phase)
        tilt = amp * 0.35 * math.sin(t * self._sway_freq * 0.63 + self._sway_phase * 1.7)
        self._disp_rot = rot_y(yaw) @ rot_x(tilt) @ self.rot
        self._ease(dt)
        # Recompute geometry only if something that affects it changed. While
        # a hand is interacting most objects are calm and unchanged, so this
        # skips the projection for everything you aren't touching.
        if self._pop < 1.0:
            self._pop += (1.0 - self._pop) * min(1.0, dt * 14)
            if self._pop > 0.995:
                self._pop = 1.0
        mk = self.mesh_key()
        gk = (self.pos.tobytes(), self._disp_rot.tobytes(), self.size, self._pop, mk)
        if self._geom is None or mk is None or gk != self._gkey:
            self._geom = self.geometry(mk)
            self._gkey = gk

    def _ease(self, dt):
        pass

    def build(self):
        raise NotImplementedError

    def mesh_key(self):
        """Anything that changes the LOCAL geometry. Rotation/position/size
        don't - they're applied per frame - so the mesh is rebuilt only when
        this key changes (selection, stretch animation, data)."""
        return None

    def geometry(self, key=...):
        if key is ...:
            key = self.mesh_key()
        if key is None or key != getattr(self, "_mkey", None):
            self._mesh = self.build().finish()
            self._mkey = key
        mesh = self._mesh
        W = (mesh.verts @ self._disp_rot.T) * self.eff_size
        z = self.pos[2] + W[:, 2]
        s = FOCAL / np.maximum(FOCAL + z, 60.0)
        return Geom(mesh, W, self.pos[0] + W[:, 0] * s, self.pos[1] + W[:, 1] * s, z, s)

    @property
    def geom(self):
        if self._geom is None:
            self._geom = self.geometry()
        return self._geom

    # -- picking --------------------------------------------------------
    def _overall_box(self, pad):
        g = self.geom
        return (g.sx.min() - pad, g.sy.min() - pad, g.sx.max() + pad, g.sy.max() + pad)

    def hit_body(self, px, py, pad=10.0):
        g = self.geom
        m = g.mesh
        if m.body is not None:
            xs, ys = g.sx[m.body], g.sy[m.body]
            cx, cy = xs.mean(), ys.mean()
            k = 1.0 + pad / max(20.0, max(xs.max() - xs.min(), ys.max() - ys.min()))
            if _point_in_poly(px, py, cx + (xs - cx) * k, cy + (ys - cy) * k):
                return True
        x0, y0, x1, y1 = self._overall_box(pad)
        return m.body is None and x0 <= px <= x1 and y0 <= py <= y1

    def pick(self, px, py, pad=10.0):
        """None | ("body", None) | ("bar", i) | ("cell", (i, j))."""
        part = self.pick_part(px, py)
        if part is not None:
            return part
        return ("body", None) if self.hit_body(px, py, pad) else None

    def pick_part(self, px, py):
        return None

    def nearest_part(self, px, py):
        return None

    def depth_key(self):
        return self.pos[2]

    # -- selection / stretch (overridden by graphs) ------------------------
    @property
    def has_selection(self):
        return self.selection is not None

    def select_start(self, part): pass
    def select_update(self, part): pass

    def in_selection(self, part):
        """Is this picked part inside the current selection? (A pinch there
        grabs the selection to stretch it instead of starting a new one.)"""
        return False

    def select_clear(self):
        self.selection = None

    def stretch_begin(self): pass
    def stretch_apply(self, sx, sy): pass
    def reset_warp(self): pass
    def refresh(self):
        self.select_clear()
        self.reset_warp()

    # -- persistence ----------------------------------------------------
    def to_dict(self):
        return {"kind": self.kind, "pos": [float(v) for v in self.pos],
                "rot": [float(v) for v in self.rot.flatten()], "size": self.size,
                "extra": self.extra_dict()}

    def extra_dict(self):
        return {}

    def load_extra(self, d):
        pass

    @classmethod
    def from_dict(cls, d):
        obj = make_object(d.get("kind"), *d.get("pos", [400, 300, 0])[:2], z=d.get("pos", [0, 0, 0])[2])
        if obj is None:
            return None
        obj.size = _clamp(float(d.get("size", obj.size)), SIZE_MIN, SIZE_MAX)
        rot = np.array(d.get("rot", []), dtype=float)
        if rot.size == 9 and np.all(np.isfinite(rot)):
            obj.rot = orthonormalize(rot.reshape(3, 3))
        obj.load_extra(d.get("extra", {}))
        return obj


# ---------------------------------------------------------------------------
class BarChart3D(Object3D):
    kind = "bars3d"
    supports_selection = True
    W, H, D = 2.0, 1.0, 0.26

    def __init__(self, x, y, z=0.0, size=150.0, data=None):
        super().__init__(x, y, z, size, "BARS 3D")
        self.sway = 0.16
        self.data = list(data) if data else [random.uniform(0.2, 1.0) for _ in range(7)]
        self.wx = [1.0] * len(self.data)      # width factor per bar (target)
        self.hy = [1.0] * len(self.data)      # height factor per bar (target)
        self.wx_d, self.hy_d = list(self.wx), list(self.hy)   # eased, drawn
        self._base = None
        self._anchor = None

    def _ease(self, dt):
        a = min(1.0, dt * 12)
        snap = lambda d, t: t if abs(t - d) < 1e-4 else d + (t - d) * a
        self.wx_d = [snap(d, t) for d, t in zip(self.wx_d, self.wx)]
        self.hy_d = [snap(d, t) for d, t in zip(self.hy_d, self.hy)]

    def mesh_key(self):
        return (self.selection, tuple(self.wx_d), tuple(self.hy_d), tuple(self.data), self.label)

    def _layout(self):
        wx = np.array(self.wx_d)
        slots = wx / wx.sum() * self.W          # constant total width
        edges = -self.W / 2 + np.concatenate(([0.0], np.cumsum(slots)))
        return slots, edges

    def build(self):
        m = Mesh()
        slots, edges = self._layout()
        n = len(self.data)
        hs = np.array(self.data) * np.array(self.hy_d) * self.H
        hmax = max(self.H, hs.max()) + 0.05
        fy, D = 0.5, self.D
        pw = self.W / 2 + 0.16
        body = m.add([(-pw, fy, -D - 0.3), (pw, fy, -D - 0.3), (pw, fy, D + 0.3), (-pw, fy, D + 0.3)])
        m.body = body
        # A dark "halo" copy drawn first (wider, near-black), then the
        # real colour on top - same exact points, so they project to
        # the same spot on screen; the stable depth-sort (see
        # render_object) preserves this insertion order for exact ties,
        # keeping the halo underneath. Cheap (ONE extra line per object,
        # not one per grid line) way to keep the single most important
        # visibility cue - "is this object even here" - readable over a
        # light background, without the cost of haloing the whole
        # internal wireframe (which would roughly double every object's
        # item count and fight the LOD budget for comparatively little
        # benefit, since the floor outline is what actually reads as the
        # object's silhouette from a glance).
        m.line(np.append(body, body[0]), "#0a0a0a", 4, 0)
        m.line(np.append(body, body[0]), self.color, 2, 0)
        for k in range(0, n + 1):                           # floor grid, one per slot edge
            g = m.add([(edges[k], fy, -D - 0.3), (edges[k], fy, D + 0.3)])
            m.line(g, shade(self.color, 0.45), 1, 0)
        sel = self.selection
        for i in range(n):
            x0 = edges[i] + slots[i] * 0.14
            x1 = edges[i + 1] - slots[i] * 0.14
            top = fy - hs[i]
            col = m.add(_box(edges[i], edges[i + 1], fy - hmax, fy, -D, D))
            m.parts.append((("bar", i), col))
            idx = m.add(_box(x0, x1, top, fy, -D, D))
            selected = sel is not None and sel[0] <= i <= sel[1]
            # Heat-mapped by VALUE (not just lit/shaded the same flat
            # colour every bar always was) - short bars cool blue, tall
            # ones warm red, same BUCKETS gradient Surface3D's grid
            # lines already use. A flat single colour across every bar
            # regardless of what they actually represent was a real
            # part of "looks boring/basic" - this makes the data itself
            # visible at a glance instead of only in the bar heights.
            base = HOT if selected else BUCKETS[min(4, int(self.data[i] * 4.999))]
            for corners, normal in _BOX_FACES:
                if normal == (0, 1, 0):
                    continue
                k = 1.0 if normal[1] < 0 else 0.78 if normal[2] else 0.55
                stip = "gray25" if normal[1] < 0 else "gray50"
                m.faces.append((idx[list(corners)], normal, shade(base, k), stip,
                                "#ffffff" if selected else base))
        if sel is not None:                                  # selection bracket + end handles
            a, b = edges[sel[0]], edges[sel[1] + 1]
            yb, zb = fy + 0.03, -(D + 0.42)   # -z = toward the viewer
            br = m.add([(a, yb - 0.12, zb), (a, yb, zb), (b, yb, zb), (b, yb - 0.12, zb)])
            m.line(br, "#ffffff", 3, 1)
        m.texts.append(((0, fy - hmax - 0.14, 0), self.label, self.color))
        return m

    def _col_screen(self, i):
        g = self.geom
        idx = next(ix for key, ix in g.mesh.parts if key == ("bar", i))
        return g.sx[idx], g.sy[idx], g.z[idx]

    def pick_part(self, px, py):
        best = None
        for key, idx in self.geom.mesh.parts:
            xs, ys = self.geom.sx[idx], self.geom.sy[idx]
            if xs.min() - 4 <= px <= xs.max() + 4 and ys.min() - 4 <= py <= ys.max() + 4:
                d = abs(px - xs.mean())
                if best is None or d < best[0]:
                    best = (d, key)
        return best[1] if best else None

    def nearest_part(self, px, py):
        g = self.geom
        best = min(g.mesh.parts, key=lambda p: abs(px - g.sx[p[1]].mean()))
        return best[0]

    def in_selection(self, part):
        return bool(self.selection and part and part[0] == "bar"
                    and self.selection[0] <= part[1] <= self.selection[1])

    def select_start(self, part):
        if part and part[0] == "bar":
            self._anchor = part[1]
            self.selection = (part[1], part[1])

    def select_update(self, part):
        if part and part[0] == "bar" and self._anchor is not None:
            self.selection = (min(self._anchor, part[1]), max(self._anchor, part[1]))

    def stretch_begin(self):
        self._base = (list(self.wx), list(self.hy))

    def stretch_apply(self, sx, sy):
        if self.selection is None or self._base is None:
            return
        a, b = self.selection
        wx, hy = list(self._base[0]), list(self._base[1])
        for i in range(a, b + 1):
            wx[i] = _clamp(wx[i] * sx, 0.2, 6.0)
            hy[i] = _clamp(hy[i] * sy, 0.2, 3.0)
        self.wx, self.hy = wx, hy

    def reset_warp(self):
        n = len(self.data)
        self.wx, self.hy = [1.0] * n, [1.0] * n

    def refresh(self):
        super().refresh()
        self.data = [random.uniform(0.2, 1.0) for _ in self.data]

    def extra_dict(self):
        return {"data": self.data, "wx": self.wx, "hy": self.hy, "sel": self.selection}

    def load_extra(self, d):
        data = d.get("data")
        if isinstance(data, list) and 2 <= len(data) <= 20:
            self.data = [_clamp(float(v), 0.0, 1.0) for v in data]
        n = len(self.data)
        for name in ("wx", "hy"):
            v = d.get(name)
            ok = isinstance(v, list) and len(v) == n
            setattr(self, name, [_clamp(float(x), 0.2, 6.0) for x in v] if ok else [1.0] * n)
        self.wx_d, self.hy_d = list(self.wx), list(self.hy)
        sel = d.get("sel")
        if isinstance(sel, list) and len(sel) == 2 and 0 <= sel[0] <= sel[1] < n:
            self.selection = (int(sel[0]), int(sel[1]))


# ---------------------------------------------------------------------------
def _f_ripple(u, v):
    r = np.hypot(u, v) * 7 + 1e-3
    return np.sin(r) / r


def _f_peaks(u, v):
    g = lambda cx, cy, s: np.exp(-((u - cx) ** 2 + (v - cy) ** 2) / s)
    return g(-.5, -.4, .12) + 0.8 * g(.5, .3, .10) + 0.6 * g(-.1, .6, .08)


def _f_saddle(u, v):
    return u * v + 0.15 * (u * u - v * v)


def _f_waves(u, v):
    return np.sin(3 * u) * np.cos(3 * v)


SURFACE_FUNCS = (_f_ripple, _f_peaks, _f_saddle, _f_waves)


class Surface3D(Object3D):
    kind = "surface3d"
    supports_selection = True
    N = 21

    def __init__(self, x, y, z=0.0, size=150.0, func=0, expr=None, expr_fn=None):
        # expr: the original typed-out text (e.g. "sin(sqrt(x^2+y^2)*6)"),
        #   kept around for display and for extra_dict() so it survives a
        #   save/reload. expr_fn: the already-compiled callable for it
        #   (function_graphs.compile_function's result) - passed in by
        #   the caller (ar3d_input._begin / from_dict) so this class
        #   itself never needs to import function_graphs or know
        #   anything about how expressions get parsed; it just calls
        #   whatever callable it's handed, same as it indexes into
        #   SURFACE_FUNCS for a preset. If expr is given without
        #   expr_fn (e.g. restoring from a saved workspace file),
        #   load_extra() below compiles it the one time that's needed.
        label = "SURFACE 3D" if expr is None else _short_label(expr)
        super().__init__(x, y, z, size, label)
        self.sway = 0.14
        self.func = func % len(SURFACE_FUNCS)
        self.expr = expr
        self._expr_fn = expr_fn
        # Scene3D.toggle_dense()/step() keep this in sync with the
        # scene-wide DENSE palette toggle - see that method's own
        # comment for why dense mode is opt-in rather than always on.
        self.dense = False
        self._make_field()
        self.reset_warp()
        self.cw_d, self.rw_d, self.amp_d = self.cw.copy(), self.rw.copy(), self.amp.copy()
        self._base = None
        self._anchor = None

    def _make_field(self):
        t = np.linspace(-1, 1, self.N)
        u, v = np.meshgrid(t, t)                      # rows = v (j), cols = u (i)
        if self.expr is not None and self._expr_fn is not None:
            h = self._expr_fn(u, v)
        else:
            h = SURFACE_FUNCS[self.func](u, v)
        span = h.max() - h.min()
        h = h - h.min()
        # A typed expression can easily be exactly flat (e.g. "z = 1",
        # or one that's constant over the whole grid) - the built-in
        # presets never are, so only the custom-expression path needs
        # this guard against a zero-height, zero-span "pancake" surface.
        self.H0 = h / span if span > 1e-9 else np.zeros_like(h)

    def reset_warp(self):
        self.cw = np.ones(self.N - 1)
        self.rw = np.ones(self.N - 1)
        self.amp = np.ones((self.N, self.N))

    def _ease(self, dt):
        a = min(1.0, dt * 12)
        for d, t in ((self.cw_d, self.cw), (self.rw_d, self.rw), (self.amp_d, self.amp)):
            diff = t - d
            d += np.where(np.abs(diff) < 1e-4, diff, diff * a)

    def mesh_key(self):
        return (self.selection, self.func, self.expr, self.dense, self.cw_d.tobytes(),
                self.rw_d.tobytes(), self.amp_d.tobytes(), self.label)

    def build(self):
        m = Mesh()
        N = self.N
        xs = -1.0 + 2.0 * np.concatenate(([0.0], np.cumsum(self.cw_d / self.cw_d.sum())))
        zs = -1.0 + 2.0 * np.concatenate(([0.0], np.cumsum(self.rw_d / self.rw_d.sum())))
        ys = 0.42 - self.H0 * self.amp_d * 0.85
        self._xs, self._zs = xs, zs
        pts = [(xs[i], ys[j, i], zs[j]) for j in range(N) for i in range(N)]
        vi = m.add(pts).reshape(N, N)                  # vi[j, i]
        pw = 1.14
        m.body = m.add([(-pw, 0.5, -pw), (pw, 0.5, -pw), (pw, 0.5, pw), (-pw, 0.5, pw)])
        # See BarChart3D.build()'s identical comment on this halo line -
        # same technique, same reasoning (cheap, one extra line, keeps
        # the silhouette readable over any background).
        m.line(np.append(m.body, m.body[0]), "#0a0a0a", 4, 0)
        m.line(np.append(m.body, m.body[0]), self.color, 2, 0)
        m.region = np.concatenate((vi[0, :], vi[:, -1], vi[-1, ::-1], vi[::-1, 0]))
        sel = self.selection
        if sel:
            si0, si1 = _seg_range(sel[0], sel[1], N)
            sj0, sj1 = _seg_range(sel[2], sel[3], N)
            # Unmistakable selection: an amber patch on the floor + four posts
            # rising to the surface, so you see exactly which footprint you hold.
            fy = 0.5 - 0.004
            c4 = m.add([(xs[si0], fy, zs[sj0]), (xs[si1], fy, zs[sj0]),
                        (xs[si1], fy, zs[sj1]), (xs[si0], fy, zs[sj1])])
            m.faces.append((c4, (0, -1, 0), shade(HOT, 0.9), "gray25", "#ffffff"))
            for k, (ci, cj) in enumerate(((si0, sj0), (si1, sj0), (si1, sj1), (si0, sj1))):
                m.line(np.array([c4[k], vi[cj, ci]]), "#ffffff", 1, 1)
        hn = self.H0 * self.amp_d

        def bucket(a, b):
            return min(4, int(((hn[a] + hn[b]) / 2) * 4.999))

        def in_sel(j0, i0, j1, i1):    # segment between (j0,i0)-(j1,i1)
            if not sel:
                return False
            if j0 == j1:
                return sj0 <= j0 <= sj1 and si0 <= min(i0, i1) < si1
            return si0 <= i0 <= si1 and sj0 <= min(j0, j1) < sj1

        def runs(coords, group):        # coords: [(j,i), ...] along one polyline
            cur, colour = [coords[0]], None
            for a, b in zip(coords, coords[1:]):
                c = "#ffffff" if in_sel(a[0], a[1], b[0], b[1]) else BUCKETS[bucket(a, b)]
                if colour is None or c == colour:
                    cur.append(b)
                    colour = c
                else:
                    m.line(np.array([vi[p] for p in cur]), colour, 2 if colour == "#ffffff" else 1, 1,
                           lod=(0, 1), group=group)
                    cur, colour = [a, b], c
            m.line(np.array([vi[p] for p in cur]), colour or BUCKETS[0], 1, 1, lod=(0, 1), group=group)
            # coarse detail: the whole grid line as ONE item, coloured by its mean height
            mean_h = float(np.mean([hn[p] for p in coords]))
            m.line(np.array([vi[p] for p in coords]), BUCKETS[min(4, int(mean_h * 4.999))], 1, 1,
                   lod=(2, 2), group=group)

        for j in range(N):
            runs([(j, i) for i in range(N)], ("r", j, j in (0, N - 1)))
        for i in range(N):
            runs([(j, i) for j in range(N)], ("c", i, i in (0, N - 1)))

        if self.dense:
            # DENSE mode (the "\u25a3 DENSE" palette toggle - see
            # Scene3D.toggle_dense): fills the grid in with real shaded,
            # heat-coloured quad faces instead of leaving it as hollow
            # lines over empty space - this is the actual visual
            # difference "dense" is for. A coarser step than the full
            # N x N grid (fixed at roughly 10x10 quads regardless of N)
            # keeps even a dense surface's item cost modest - see
            # item_cost()/effective_item_budget() - so turning dense on
            # for several surfaces at once still degrades through the
            # normal LOD system rather than needing its own separate
            # safety valve. Every quad uses one flat "facing up" normal
            # (matching the selection floor patch above, the one other
            # place this class draws a face) rather than a true per-quad
            # slope normal - simpler, and avoids guessing at a winding-
            # order sign convention with no way to check it against a
            # real render here.
            step = max(1, (N - 1) // 6)          # ~7x7 quads: half the items of 10x10
            up_normal = (0, -1, 0)
            for j in range(0, N - 1, step):
                j2 = min(j + step, N - 1)
                for i in range(0, N - 1, step):
                    i2 = min(i + step, N - 1)
                    corners = np.array([vi[j, i], vi[j, i2], vi[j2, i2], vi[j2, i]])
                    avg_h = float((hn[j, i] + hn[j, i2] + hn[j2, i2] + hn[j2, i]) / 4)
                    face_color = BUCKETS[min(4, int(avg_h * 4.999))]
                    # No stipple: Tk renders a stippled polygon through a
                    # bitmap-mask path that is several times slower than a
                    # plain solid fill, and with ~50 faces on screen that
                    # was a big part of dense mode's frame-rate cost. A
                    # darkened solid fill keeps the lines on top readable
                    # and the outline matches the fill so there are no
                    # seams between neighbouring quads.
                    fill_c = shade(face_color, 0.42)
                    m.faces.append((corners, up_normal, fill_c, "", fill_c))

        self._vi = vi
        m.texts.append(((0, float(ys.min()) - 0.15, 0), self.label, self.color))
        return m

    def _unproject_floor(self, px, py):
        """Screen point -> (x, z) on the floor plane in local units. Selection
        follows the ground under your finger regardless of how tall the
        surface is there. Exact: with s = F/(F+z+W_z), the projection
        (px-x0)*(F+z0+W_z) = F*W_x is LINEAR in the two unknowns, so this is
        one 2x2 solve - no iteration. None if the floor is (nearly) edge-on."""
        R, sz, pos, fy, F = self._disp_rot, self.eff_size, self.pos, 0.5, FOCAL
        a, b, D = px - pos[0], py - pos[1], F + pos[2]
        A = np.array([[a * R[2, 0] - F * R[0, 0], a * R[2, 2] - F * R[0, 2]],
                      [b * R[2, 0] - F * R[1, 0], b * R[2, 2] - F * R[1, 2]]]) * sz
        rhs = np.array([-a * D - a * sz * R[2, 1] * fy + F * sz * R[0, 1] * fy,
                        -b * D - b * sz * R[2, 1] * fy + F * sz * R[1, 1] * fy])
        # normalise so the edge-on test is scale-free: ~|normal . view direction|
        if abs(np.linalg.det(np.array([[R[0, 0], R[0, 2]], [R[1, 0], R[1, 2]]]))) < 0.08:
            return None
        try:
            x, z = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return None
        return float(x), float(z)

    def _cell_at_floor(self, x, z):
        i = int(round(float(np.interp(x, self._xs, np.arange(self.N)))))
        j = int(round(float(np.interp(z, self._zs, np.arange(self.N)))))
        return ("cell", (int(_clamp(i, 0, self.N - 1)), int(_clamp(j, 0, self.N - 1))))

    def _vertex_near(self, px, py):
        g = self.geom
        d = (g.sx[self._vi] - px) ** 2 + (g.sy[self._vi] - py) ** 2
        j, i = np.unravel_index(np.argmin(d), d.shape)
        near = math.sqrt(float(d[j, i])) <= 0.09 * self.eff_size * float(g.s.mean())
        return ("cell", (int(i), int(j))), near

    def pick_part(self, px, py):
        """Start of a pinch: touching a mesh point (even a tall peak that
        projects beyond the floor outline) or the ground under the surface."""
        part, near = self._vertex_near(px, py)
        if near:
            return part
        fp = self._unproject_floor(px, py)
        if fp is not None and abs(fp[0]) <= 1.03 and abs(fp[1]) <= 1.03:
            return self._cell_at_floor(*fp)
        return None

    def nearest_part(self, px, py):
        """While dragging a selection: the floor cell under the finger, so the
        rubber-band box is predictable at any height."""
        fp = self._unproject_floor(px, py)
        if fp is None:
            return self._vertex_near(px, py)[0]
        return self._cell_at_floor(*fp)

    def in_selection(self, part):
        if not (self.selection and part and part[0] == "cell"):
            return False
        i, j = part[1]
        s = self.selection
        return s[0] - 1 <= i <= s[1] + 1 and s[2] - 1 <= j <= s[3] + 1

    def select_start(self, part):
        if part and part[0] == "cell":
            self._anchor = part[1]
            i, j = part[1]
            self.selection = (i, i, j, j)

    def select_update(self, part):
        if part and part[0] == "cell" and self._anchor is not None:
            (ai, aj), (i, j) = self._anchor, part[1]
            self.selection = (min(ai, i), max(ai, i), min(aj, j), max(aj, j))

    def stretch_begin(self):
        self._base = (self.cw.copy(), self.rw.copy(), self.amp.copy())

    def stretch_apply(self, sx, sy):
        if not self.selection or self._base is None:
            return
        N = self.N
        si0, si1 = _seg_range(self.selection[0], self.selection[1], N)
        sj0, sj1 = _seg_range(self.selection[2], self.selection[3], N)
        cw, rw, amp = (b.copy() for b in self._base)
        cw[si0:si1] = np.clip(cw[si0:si1] * sx, 0.2, 6.0)
        rw[sj0:sj1] = np.clip(rw[sj0:sj1] * sx, 0.2, 6.0)
        ramp = lambda k, lo, hi: np.clip(1 - np.maximum(np.maximum(lo - k, k - hi), 0) / 2.0, 0, 1)
        k = np.arange(N)
        win = ramp(k, self.selection[2], self.selection[3])[:, None] * ramp(k, self.selection[0], self.selection[1])[None, :]
        self.cw, self.rw = cw, rw
        self.amp = np.clip(amp * (1 + (sy - 1) * win), 0.1, 3.5)

    def refresh(self):
        super().refresh()
        self.func = (self.func + 1) % len(SURFACE_FUNCS)
        self._make_field()

    def extra_dict(self):
        return {"func": self.func, "expr": self.expr,
                "cw": self.cw.tolist(), "rw": self.rw.tolist(),
                "amp": self.amp.tolist(), "sel": self.selection}

    def load_extra(self, d):
        expr = d.get("expr")
        if expr:
            try:
                import function_graphs
                self._expr_fn = function_graphs.compile_function(expr)
                self.expr = expr
                self.label = _short_label(expr)
            except function_graphs.FunctionParseError as e:
                # A saved expression that no longer compiles (hand-edited
                # between sessions, say) falls back to a preset rather
                # than refusing to restore the whole workspace over it.
                print(f"[WARN] AR: saved surface function is no longer valid ({e}) "
                      f"- showing a default surface instead")
                self.expr, self._expr_fn = None, None
        self.func = int(d.get("func", 0)) % len(SURFACE_FUNCS)
        self._make_field()
        N = self.N
        try:
            cw, rw = np.array(d["cw"], float), np.array(d["rw"], float)
            amp = np.array(d["amp"], float)
            if cw.shape == (N - 1,) and rw.shape == (N - 1,) and amp.shape == (N, N) \
                    and np.all(np.isfinite(amp)) and cw.min() > 0 and rw.min() > 0:
                self.cw, self.rw, self.amp = cw, rw, amp
        except (KeyError, ValueError, TypeError):
            pass
        self.cw_d, self.rw_d, self.amp_d = self.cw.copy(), self.rw.copy(), self.amp.copy()
        sel = d.get("sel")
        if isinstance(sel, list) and len(sel) == 4 and all(0 <= v < N for v in sel):
            self.selection = tuple(int(v) for v in sel)


# ---------------------------------------------------------------------------
class Shape3D(Object3D):
    """Plain wireframe solids - no data, so two hands always act on the
    whole object."""

    def __init__(self, shape, x, y, z=0.0, size=110.0):
        self.kind = shape
        super().__init__(x, y, z, size, shape.upper())
        self.spin = 0.6
        self.color = {"cube": GREEN, "sphere": HOLO, "torus": VIOLET}.get(shape, HOLO)

    def build(self):
        m = Mesh()
        if self.kind == "cube":
            idx = m.add(_box(-.7, .7, -.7, .7, -.7, .7))
            for corners, normal in _BOX_FACES:
                m.faces.append((idx[list(corners)], normal, shade(self.color, 0.5), "gray12", self.color))
        elif self.kind == "sphere":
            for lat in range(-60, 61, 30):
                a = math.radians(lat)
                ring = [(0.85 * math.cos(a) * math.cos(t), -0.85 * math.sin(a), 0.85 * math.cos(a) * math.sin(t))
                        for t in np.linspace(0, 2 * math.pi, 33)]
                m.line(m.add(ring), self.color, 1, 1)
            for lon in range(0, 180, 30):
                b = math.radians(lon)
                mer = [(0.85 * math.cos(t) * math.cos(b), -0.85 * math.sin(t), 0.85 * math.cos(t) * math.sin(b))
                       for t in np.linspace(0, 2 * math.pi, 41)]
                m.line(m.add(mer), shade(self.color, 0.8), 1, 1)
        else:  # torus
            R, r = 0.65, 0.28
            for a in np.linspace(0, 2 * math.pi, 17)[:-1]:
                ring = [((R + r * math.cos(b)) * math.cos(a), r * math.sin(b), (R + r * math.cos(b)) * math.sin(a))
                        for b in np.linspace(0, 2 * math.pi, 17)]
                m.line(m.add(ring), self.color, 1, 1)
            for b in np.linspace(0, 2 * math.pi, 9)[:-1]:
                ring = [((R + r * math.cos(b)) * math.cos(a), r * math.sin(b), (R + r * math.cos(b)) * math.sin(a))
                        for a in np.linspace(0, 2 * math.pi, 41)]
                m.line(m.add(ring), shade(self.color, 0.8), 1, 1)
        m.texts.append(((0, -1.05, 0), self.label, self.color))
        return m

    def mesh_key(self):
        return (self.kind, self.color, self.label)


def make_object(kind, x, y, z=0.0, expr=None, expr_fn=None):
    if kind == "bars3d":
        return BarChart3D(x, y, z)
    if kind == "surface3d":
        return Surface3D(x, y, z, expr=expr, expr_fn=expr_fn)
    if kind in ("cube", "sphere", "torus"):
        return Shape3D(kind, x, y, z)
    return None


# ---------------------------------------------------------------------------
_SHADE_CACHE = {}


def shade_q(color, k):
    """shade() with results cached on (colour, k rounded to 1/32): depth
    dimming only ever uses a couple dozen levels, so after the first frame
    this is a dict lookup instead of hex parsing + string formatting."""
    kq = int(k * 32 + 0.5)
    key = (color, kq)
    v = _SHADE_CACHE.get(key)
    if v is None:
        if len(_SHADE_CACHE) > 4096:
            _SHADE_CACHE.clear()
        v = _SHADE_CACHE[key] = shade(color, kq / 32.0)
    return v


def render_object(canvas, obj, ghost=False, highlighted=False, hover_part=None, lod=0):
    """Draw one object via canvas.create_line / create_polygon /
    create_text. Far parts are dimmer and thinner (depth cue). `lod` 0 =
    full detail, 1 = every other grid line, 2 = coarse (one item per grid
    line): the Scene picks it from how many canvas items are on screen.

    Hot path: everything per-line/per-face that used to be a Python loop of
    tiny numpy calls (depth means, culling, coordinate flattening) is done
    as a few vectorised operations per object."""
    g = obj.geom
    m = g.mesh
    zmin, zmax = float(g.z.min()), float(g.z.max())
    inv_span = 0.5 / max(zmax - zmin, 1e-6)
    wscale = _clamp(float(g.s.mean()), 0.7, 1.6)
    xy = np.column_stack((g.sx, g.sy))
    dash = (6, 4) if ghost else None
    boost = 1.35 if highlighted else 1.0
    wcache = {}

    def wpx(w):
        v = wcache.get(w)
        if v is None:
            v = wcache[w] = max(1, int(round(w * wscale)))
        return v

    if m.n_lines:
        zm = np.add.reduceat(g.z[m.line_cat], m.line_off) / m.line_cnt
        order = np.argsort(-zm, kind="stable").tolist()        # far first
        mask = m.lod_mask(lod)
        zml = zm.tolist()
        lines, layers = m.lines, m.line_layer

        def draw_lines(layer):
            for i in order:
                if mask[i] and layers[i] == layer:
                    ln = lines[i]
                    canvas.create_line(*xy[ln.idx].ravel().tolist(),
                                       fill=shade_q(ln.color, (1.0 - (zml[i] - zmin) * inv_span) * boost),
                                       width=wpx(ln.width), dash=dash)
    else:
        def draw_lines(layer):
            pass

    draw_lines(0)

    F = m.face_idx
    if len(F):
        R = obj._disp_rot
        cam_offset = np.array([0.0, 0.0, FOCAL + obj.pos[2]])          # camera sits at -cam_offset
        normals_w = m.face_normals @ R.T
        view = g.W[F].mean(axis=1) + cam_offset                        # camera -> each face centre
        visible = np.einsum("ij,ij->i", normals_w, view) < 0            # front-facing only
        zf = g.z[F].mean(axis=1)
        for i in np.argsort(-zf, kind="stable").tolist():               # far first (painter's)
            if visible[i]:
                _idx, _n, col, stip, outline = m.faces[i]
                k = 1.0 - (zf[i] - zmin) * inv_span
                canvas.create_polygon(*xy[F[i]].ravel().tolist(), fill=shade_q(col, k),
                                      outline=shade_q(outline, k * boost), width=wpx(1.5),
                                      stipple=("gray12" if ghost else stip))

    draw_lines(1)

    if hover_part is not None and obj.kind in ("bars3d",) and hover_part[0] == "bar":
        for key, idx in m.parts:
            if key == hover_part:
                xs, ys = g.sx[idx], g.sy[idx]
                canvas.create_rectangle(xs.min(), ys.min(), xs.max(), ys.max(),
                                        outline="#ffffff", width=1, dash=(3, 3))
    R = obj._disp_rot
    for (x, y, z), text, col in m.texts:
        w = np.array([[x, y, z]]) @ R.T * obj.eff_size
        zz = obj.pos[2] + w[0, 2]
        s = FOCAL / max(FOCAL + zz, 60.0)
        size = int(_clamp(10 * s, 9, 15))
        tx, ty = obj.pos[0] + w[0, 0] * s, obj.pos[1] + w[0, 1] * s
        # A small dark pill behind the label (one extra canvas item, the
        # same trick the 2D overlay uses): a bare cyan/green word floating
        # over whatever's on the desktop is unreadable on a light
        # background, and the old fix for that (a ring of offset black
        # copies) is what made text look doubled/blurry.
        hw, hh = len(text) * size * 0.34 + 8, size * 0.85 + 2
        canvas.create_rectangle(tx - hw, ty - hh, tx + hw, ty + hh,
                                fill=LABEL_PILL, outline="#243447", width=1)
        canvas.create_text(tx, ty, text=text, fill=LABEL_TEXT.get(col, "#e8fbff"),
                           font=(UI_FONT, size, "bold"))


class Scene3D:
    """Owns the 3D objects: stepping (idle spin, easing, inertia), picking,
    drawing, persistence. The controller (ar3d_input.py) mutates objects;
    AROverlay just calls step()/draw()."""

    def __init__(self, bounds=(1920, 1080)):
        self.objects = []
        self.ghost = None
        self.bounds = bounds
        self.grabbed = set()         # ids of objects a hand currently holds
        self.calm = set()            # ids to hold still (e.g. a selection drag)
        self.hover = {}              # hand idx -> (obj, part)
        self.hud = None              # dict set by the controller each frame
        self.dirty = False
        self.trash = []              # recently removed objects (undo)
        self.item_budget = 450       # max canvas items the 3D layer may use per frame
        self.lod = 0                 # current detail level (0 full .. 2 coarse)
        self._t = 0.0
        # User-facing "DENSE" toggle (see the palette button in
        # ar_overlay.py / ar3d_input.py's PALETTE_KINDS handling): off by
        # default so every machine gets the light, hollow-wireframe look
        # that's always been performance-safe; turning it on lets
        # Surface3D fill in real shaded, heat-coloured faces between the
        # grid lines instead of just the lines themselves - see
        # Surface3D.build()'s own "if self.dense" branch. The existing
        # adaptive LOD system (_choose_lod/total_items below) still
        # protects frame rate either way - dense mode raises how much
        # budget is AVAILABLE (effective_item_budget), it doesn't bypass
        # the budget system itself, so a machine that genuinely can't
        # keep up still degrades gracefully instead of just grinding to
        # a halt.
        self.dense_mode = False

    def effective_item_budget(self):
        return self.item_budget * (2.4 if self.dense_mode else 1.0)

    def toggle_dense(self):
        self.dense_mode = not self.dense_mode
        for o in self.objects:
            o.dense = self.dense_mode
        self.dirty = True

    # Keep-on-screen margins (left, top, right, bottom). Bottom is larger so
    # the floor never hides under a taskbar / the screen edge.
    MARGINS = (16, 16, 16, 64)

    def _avail(self):
        W, H = self.bounds
        l, t, r, b = self.MARGINS
        return l, t, W - r, H - b

    def constrain(self, o):
        """Keep the WHOLE projected object - floor plate included - on
        screen. (Clamping only the centre let big tilted objects hang off
        the bottom with their grab-able floor out of reach.) Cheap no-op
        when already inside."""
        x0, y0, x1, y1 = self._avail()
        g = o.geom
        if g.sx.min() >= x0 and g.sx.max() <= x1 and g.sy.min() >= y0 and g.sy.max() <= y1:
            return
        for _ in range(4):                       # too big for the screen: push back, then shrink
            g = o.geometry()
            bw, bh = g.sx.max() - g.sx.min(), g.sy.max() - g.sy.min()
            f = min((x1 - x0) / max(bw, 1e-6), (y1 - y0) / max(bh, 1e-6))
            if f >= 1.0:
                break
            newz = (FOCAL + o.pos[2]) / f - FOCAL
            if newz <= Z_MAX:
                o.pos[2] = newz
            else:
                o.size = max(20.0, o.size * f)
        g = o.geometry()
        lo_x, hi_x = x0 - (g.sx.min() - o.pos[0]), x1 - (g.sx.max() - o.pos[0])
        lo_y, hi_y = y0 - (g.sy.min() - o.pos[1]), y1 - (g.sy.max() - o.pos[1])
        nx = (lo_x + hi_x) / 2 if lo_x > hi_x else _clamp(o.pos[0], lo_x, hi_x)
        ny = (lo_y + hi_y) / 2 if lo_y > hi_y else _clamp(o.pos[1], lo_y, hi_y)
        if nx != o.pos[0]:
            o.vx = 0.0
        if ny != o.pos[1]:
            o.vy = 0.0
        o.pos[0], o.pos[1] = nx, ny
        o._geom = o.geometry()
        self.dirty = True

    def fit_size(self, o, size):
        """`size` limited so the object's projected box still fits on screen
        (used while two-hand scaling, so growing never pushes it off)."""
        x0, y0, x1, y1 = self._avail()
        g = o.geom
        bw = (g.sx.max() - g.sx.min()) / max(o.size, 1e-6)
        bh = (g.sy.max() - g.sy.min()) / max(o.size, 1e-6)
        return min(size, (x1 - x0) / max(bw, 1e-6), (y1 - y0) / max(bh, 1e-6))

    def clamp_xy(self, x, y):
        w, h = self.bounds
        return _clamp(x, 40, w - 40), _clamp(y, 40, h - 40)

    def add(self, obj):
        obj.start_pop_in()
        self.objects.append(obj)
        while len(self.objects) > MAX_OBJECTS:
            self.objects.pop(0)
        self.dirty = True
        return obj

    def remove(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)
            self.trash = (self.trash + [obj])[-5:]
            self.dirty = True

    def undo_remove(self):
        """Bring back the most recently removed object. True if one existed."""
        if not self.trash:
            return False
        obj = self.trash.pop()
        obj.vx = obj.vy = 0.0
        self.add(obj)
        return True

    def clear(self):
        self.objects.clear()
        self.dirty = True

    def step(self, dt):
        self._t += dt
        dt = _clamp(dt, 1e-3, 0.1)
        hovered = {ob.id for ob, _ in self.hover.values()}
        for o in self.objects:
            if getattr(o, "dense", None) != self.dense_mode and hasattr(o, "dense"):
                o.dense = self.dense_mode   # newly-placed surfaces pick up the current toggle state
            held = o.id in self.grabbed
            if not held and (abs(o.vx) > 6 or abs(o.vy) > 6):
                o.pos[0] += o.vx * dt
                o.pos[1] += o.vy * dt
                k = 0.85 ** (dt * 30)            # same glide at any frame rate
                o.vx *= k
                o.vy *= k
                self.dirty = True
            elif not held:
                o.vx = o.vy = 0.0
            o.update(dt, self._t, held or o.id in hovered or o.id in self.calm)
            self.constrain(o)
        self._choose_lod()
        if self.ghost is not None:
            self.ghost.update(dt, self._t, True)

    def total_items(self, lod):
        return sum(o.geom.mesh.item_cost(lod) for o in self.objects)

    def _choose_lod(self):
        """Tk's cost is per canvas item, not per pixel: with many big
        surfaces the overlay (not the maths) is what drops your FPS. Pick the
        finest detail whose total item count fits the budget. Hysteresis
        (must drop under 80% of budget to refine again) stops flicker."""
        if not self.objects:
            self.lod = 0
            return
        budget = self.effective_item_budget()
        lod = self.lod
        while lod < 2 and self.total_items(lod) > budget:
            lod += 1
        while lod > 0 and self.total_items(lod - 1) <= budget * 0.8:
            lod -= 1
        self.lod = lod

    def object_at(self, px, py, pad=10.0):
        """Topmost (nearest) object under the point -> (obj, part) or None."""
        best = None
        for o in self.objects:
            part = o.pick(px, py, pad)
            if part is not None and (best is None or o.depth_key() <= best[0].depth_key()):
                best = (o, part)
        return best

    def draw(self, canvas):
        for o in sorted(self.objects, key=lambda o: -o.depth_key()):
            hp = next((p for ob, p in self.hover.values() if ob is o), None)
            render_object(canvas, o, highlighted=o.id in self.grabbed, hover_part=hp, lod=self.lod)
        if self.ghost is not None:
            # Used to always render the ghost at lod=0 (full detail),
            # regardless of what self.lod actually was - so a big
            # surface could look crisp the whole time it was being
            # dragged and then visibly lose detail the instant it was
            # dropped, as soon as _choose_lod() recomputed with the new
            # object's own (substantial) item cost now counted in. What
            # you see while holding it should be what you get, so the
            # ghost is drawn at the lod it would actually land on if
            # dropped on this exact frame - see _lod_for_ghost.
            render_object(canvas, self.ghost, ghost=True, lod=self._lod_for_ghost())
        self._draw_hud(canvas)

    def _lod_for_ghost(self):
        """Same hysteresis-free search _choose_lod does, but against
        the budget as it would be WITH the ghost counted too - i.e. the
        lod the ghost will actually render at the moment it's dropped
        and becomes a real object. Doesn't mutate self.lod (that still
        only accounts for already-placed objects until drop actually
        happens, same as before)."""
        if self.ghost is None:
            return self.lod
        base = self.total_items
        ghost_cost = self.ghost.geom.mesh.item_cost
        budget = self.effective_item_budget()
        lod = self.lod
        while lod < 2 and base(lod) + ghost_cost(lod) > budget:
            lod += 1
        return lod

    def _draw_hud(self, canvas):
        hud = self.hud or {}
        for pt, pinching, col, approach in hud.get("cursors", []):
            x, y = pt
            r = 14 if pinching else 16 - 6 * approach          # ring tightens as you close in
            c = "#ff4d4d" if pinching else ("#ffd24d" if approach > 0.05 else col)
            canvas.create_oval(x - r, y - r, x + r, y + r, outline=c, width=2)
            canvas.create_line(x - r - 8, y, x - r + 2, y, fill=c, width=2)
            canvas.create_line(x + r - 2, y, x + r + 8, y, fill=c, width=2)
        tether = hud.get("tether")
        if tether:
            (x0, y0), (x1, y1) = tether
            canvas.create_line(x0, y0, x1, y1, fill="#ffffff", width=2, dash=(8, 5))
            for x, y in tether:
                canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#ffffff", width=2)
        if hud.get("text") and tether:
            canvas.create_text((tether[0][0] + tether[1][0]) / 2,
                               (tether[0][1] + tether[1][1]) / 2 - 22, text=hud["text"],
                               fill="#ffffff", font=("Consolas", 14, "bold"))
        hint = hud.get("hint")
        if hint:
            (x, y), text = hint
            canvas.create_text(x + 18, y - 26, text=text, fill="#ffffff",
                               font=("Consolas", 9, "bold"), anchor="w")

    # -- persistence ------------------------------------------------------
    def to_list(self):
        return [o.to_dict() for o in self.objects]

    def load_list(self, items):
        for d in items or []:
            try:
                obj = Object3D.from_dict(d)
            except Exception as e:      # one corrupt entry must not lose the rest
                print(f"[WARN] AR 3D: skipped a saved object ({e})")
                continue
            if obj is not None:
                obj.pos[0], obj.pos[1] = self.clamp_xy(obj.pos[0], obj.pos[1])
                obj.pos[2] = _clamp(obj.pos[2], Z_MIN, Z_MAX)
                obj.update(0.0, 0.0, True)
                self.objects.append(obj)
                self.constrain(obj)             # rescue anything saved off-screen
        del self.objects[:-MAX_OBJECTS]
