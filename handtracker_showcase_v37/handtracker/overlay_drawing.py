"""
overlay_drawing.py
-------------------
The Drawing mode whiteboard, rebuilt to live ON the desktop overlay
instead of inside the small cv2 preview window. Two Tk-free pieces (so
both can be unit-tested without a display) plus the drawing done by
main_overlay.MainOverlay from the plain data they expose:

  InkCanvas       the strokes themselves: an RGBA pixel buffer at SCREEN
                  resolution, split into tiles so only the few tiles a
                  stroke actually touched are re-uploaded to Tk each frame.
  OverlayDrawing  the toolbar (pen, eraser, colours, thickness, clear,
                  save, hide) + the pinch-to-draw / pinch-to-click logic.

WHY THIS EXISTS (and why it's built this way)
  - Everything is in SCREEN space and fed the same map_to_screen() point
    the on-screen reticle uses, so ink lands exactly under the reticle.
    The old panel worked in camera-frame pixels and was simply stretched
    onto the overlay - the same two-coordinate-spaces mismatch that broke
    Whack-a-Mole (see features/game.py).
  - It was running at ~13 fps because every frame the WHOLE drawing was
    colour-converted, resized to screen size, alpha-masked, stacked into a
    fresh full-screen RGBA array and re-pasted into Tk (~35-40 ms of
    numpy/PIL work for 1080p). Now strokes are drawn incrementally into a
    persistent buffer, and only dirty tiles (typically one or two small
    256 px squares) are converted/uploaded. A frame with no new ink costs
    nothing but re-placing already-uploaded images.
  - Clicking a toolbar button needs a pinch (like Whack-a-Mole), not just
    hovering - brushing past the CLEAR button must not wipe the drawing.
"""
import os
import time

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

TILE = 256


class InkCanvas:
    def __init__(self, width, height, tile=TILE):
        self.w, self.h, self.tile = int(width), int(height), int(tile)
        self.buf = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        self.nty = -(-self.h // self.tile)
        self.ntx = -(-self.w // self.tile)
        self.dirty = set()
        self.has_ink = np.zeros((self.nty, self.ntx), dtype=bool)

    # -- drawing ---------------------------------------------------------
    def _mark(self, p0, p1, pad):
        x0, x1 = sorted((int(p0[0]), int(p1[0])))
        y0, y1 = sorted((int(p0[1]), int(p1[1])))
        tx0 = max(0, (x0 - pad) // self.tile)
        tx1 = min(self.ntx - 1, (x1 + pad) // self.tile)
        ty0 = max(0, (y0 - pad) // self.tile)
        ty1 = min(self.nty - 1, (y1 + pad) // self.tile)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                self.dirty.add((ty, tx))

    def stroke(self, p0, p1, rgb, thickness):
        """Round-capped line segment in colour rgb (opaque)."""
        a, b = (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))
        col = (int(rgb[0]), int(rgb[1]), int(rgb[2]), 255)
        cv2.line(self.buf, a, b, col, int(thickness))
        cv2.circle(self.buf, b, max(1, int(thickness) // 2), col, -1)
        self._mark(a, b, int(thickness) + 2)

    def erase(self, p0, p1, radius):
        a, b = (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))
        cv2.line(self.buf, a, b, (0, 0, 0, 0), int(radius) * 2)
        cv2.circle(self.buf, b, int(radius), (0, 0, 0, 0), -1)
        self._mark(a, b, int(radius) + 2)

    def clear(self):
        self.buf[:] = 0
        for ty in range(self.nty):
            for tx in range(self.ntx):
                self.dirty.add((ty, tx))

    # -- upload bookkeeping ------------------------------------------------
    def tile_rect(self, ty, tx):
        x0, y0 = tx * self.tile, ty * self.tile
        return x0, y0, min(self.w, x0 + self.tile), min(self.h, y0 + self.tile)

    def pop_dirty(self):
        """Dirty tiles since the last call; refreshes has_ink for them
        (an eraser stroke can empty a tile - drawing an empty tile would
        just waste a canvas item)."""
        tiles = sorted(self.dirty)
        self.dirty = set()
        for ty, tx in tiles:
            x0, y0, x1, y1 = self.tile_rect(ty, tx)
            self.has_ink[ty, tx] = bool(self.buf[y0:y1, x0:x1, 3].any())
        return tiles

    def tile_pixels(self, ty, tx):
        x0, y0, x1, y1 = self.tile_rect(ty, tx)
        return np.ascontiguousarray(self.buf[y0:y1, x0:x1])

    def is_empty(self):
        return not self.buf[..., 3].any()


class OverlayDrawing:
    COLORS = (("white", (245, 245, 245)), ("red", (255, 82, 82)), ("green", (80, 230, 120)),
              ("blue", (70, 140, 255)), ("yellow", (255, 224, 77)), ("pink", (255, 110, 200)))
    THICKNESS = (4, 9, 18)
    SAVE_NAME = "drawing.png"
    MIN_PINCH_FRAMES = 3          # fingertip jumps toward the thumb as a pinch closes - skip those frames
    SMOOTH = 0.55                 # fraction of the way toward the new point per frame (stroke smoothing)

    def __init__(self, screen_w, screen_h):
        self.sw, self.sh = int(screen_w), int(screen_h)
        self.ink = InkCanvas(self.sw, self.sh)
        self.tool = "pen"
        self.color_idx = 0
        self.thick_idx = 1
        self.panel_open = True
        self.hover_id = None
        self.toast = None                    # (text, expires_at)
        self._prev_pinch = False
        self._pinch_frames = 0
        self._blocked = False                # this pinch began on the toolbar: never draw from it
        self._last = None
        self.eraser_radius = max(18, int(28 * self.sw / 1280))

    # -- layout (screen space) ---------------------------------------------
    def layout(self):
        """Toolbar buttons as dicts {id, kind, rect, ...}. A vertical dock
        centred on the left edge (away from the taskbar and from the top
        edge fullscreen apps use); collapses to a single round button
        when hidden."""
        margin = 18
        if not self.panel_open:
            s = 52
            y = self.sh // 2 - s // 2
            return [{"id": "show", "kind": "show", "rect": (margin, y, margin + s, y + s)}]
        n_btn, n_sep = 14, 3
        size = int(max(30, min(44, (self.sh - 220 - n_sep * 14) / n_btn - 6)))
        gap = 6
        total = n_btn * (size + gap) + n_sep * 14
        y = (self.sh - total) // 2
        x = margin + 12
        out = []

        def add(i, kind, **extra):
            nonlocal y
            out.append({"id": i, "kind": kind, "rect": (x, y, x + size, y + size), **extra})
            y += size + gap

        def sep():
            nonlocal y
            y += 14

        add("pen", "tool"); add("eraser", "tool")
        sep()
        for i, (name, rgb) in enumerate(self.COLORS):
            add(f"color{i}", "color", rgb=rgb, name=name)
        sep()
        for i, t in enumerate(self.THICKNESS):
            add(f"thick{i}", "thick", px=t)
        sep()
        add("clear", "action"); add("save", "action"); add("hide", "action")
        return out

    def dock_rect(self):
        btns = self.layout()
        if len(btns) == 1:
            return btns[0]["rect"]
        x0 = min(b["rect"][0] for b in btns) - 12
        y0 = min(b["rect"][1] for b in btns) - 12
        x1 = max(b["rect"][2] for b in btns) + 12
        y1 = max(b["rect"][3] for b in btns) + 12
        return (x0, y0, x1, y1)

    def button_at(self, pt):
        if pt is None:
            return None
        for b in self.layout():
            x0, y0, x1, y1 = b["rect"]
            if x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1:
                return b["id"]
        return None

    def over_dock(self, pt):
        if pt is None:
            return False
        x0, y0, x1, y1 = self.dock_rect()
        return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1

    # -- actions -------------------------------------------------------------
    def press(self, button_id, now=None):
        now = time.time() if now is None else now
        if button_id in ("pen", "eraser"):
            self.tool = button_id
        elif button_id.startswith("color"):
            self.color_idx = int(button_id[5:])
            self.tool = "pen"                 # picking a colour implies drawing with it
        elif button_id.startswith("thick"):
            self.thick_idx = int(button_id[5:])
        elif button_id == "clear":
            self.ink.clear()
            self._toast("Cleared", now)
        elif button_id == "save":
            self._toast(self.save(), now)
        elif button_id == "hide":
            self.panel_open = False
        elif button_id == "show":
            self.panel_open = True
        return button_id

    def save(self, path=None):
        """Writes the drawing as a transparent PNG; returns a short status
        string for the toast."""
        path = path or os.path.join(os.getcwd(), self.SAVE_NAME)
        try:
            ok = cv2.imwrite(path, cv2.cvtColor(self.ink.buf, cv2.COLOR_RGBA2BGRA))
        except Exception:
            ok = False
        return f"Saved {os.path.basename(path)}" if ok else "Save failed"

    def _toast(self, text, now):
        self.toast = (text, now + 2.0)

    # -- per-frame input -------------------------------------------------------
    def update(self, pt, pinching, now=None):
        """pt: the screen-mapped fingertip (the reticle's position) or None;
        pinching: the (hysteresis-stabilised) pinch state. Returns a
        description of what happened, mainly for tests."""
        now = time.time() if now is None else now
        if self.toast and now > self.toast[1]:
            self.toast = None
        self.hover_id = self.button_at(pt)
        click = bool(pinching and not self._prev_pinch)
        self._prev_pinch = bool(pinching)
        event = None

        if click:
            self._pinch_frames = 0
            self._last = None
            bid = self.button_at(pt)
            self._blocked = bid is not None or self.over_dock(pt)
            if bid is not None:
                self.press(bid, now)
                event = ("press", bid)
        if not pinching or pt is None:
            self._pinch_frames = 0
            self._last = None
            if not pinching:
                self._blocked = False
            return event

        if self._blocked:
            return event
        self._pinch_frames += 1
        if self._pinch_frames < self.MIN_PINCH_FRAMES:
            return event
        p = (float(pt[0]), float(pt[1]))
        if self._last is None:
            self._last = p
            return event
        q = (self._last[0] + (p[0] - self._last[0]) * self.SMOOTH,
             self._last[1] + (p[1] - self._last[1]) * self.SMOOTH)
        if self.tool == "eraser":
            self.ink.erase(self._last, q, self.eraser_radius)
        else:
            self.ink.stroke(self._last, q, self.COLORS[self.color_idx][1],
                            self.THICKNESS[self.thick_idx])
        self._last = q
        return event or ("draw", self.tool)

    # -- plain data for the renderer -------------------------------------------
    def view(self, now=None):
        now = time.time() if now is None else now
        toast = self.toast[0] if (self.toast and now <= self.toast[1]) else None
        sel = {"pen": self.tool == "pen", "eraser": self.tool == "eraser"}
        return {
            "buttons": self.layout(), "dock": self.dock_rect(), "open": self.panel_open,
            "hover": self.hover_id, "tool": self.tool,
            "color": self.COLORS[self.color_idx][1], "color_idx": self.color_idx,
            "thick_idx": self.thick_idx, "selected": sel, "toast": toast,
            "cursor_radius": (self.eraser_radius if self.tool == "eraser"
                              else self.THICKNESS[self.thick_idx] / 2),
        }
