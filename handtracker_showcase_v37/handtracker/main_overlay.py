"""
main_overlay.py
----------------
The "whole program as a desktop overlay" piece: a full-screen,
always-on-top, click-through Tkinter window (same technique as
ar_overlay.AROverlay - see that module's docstring for the
'-transparentcolor' explanation and its Windows-only caveat) that shows
the mode name, control on/off state, current gesture + confidence, FPS,
the per-mode help panel, and the gesture-driven mode menu - everything
draw_hud()/draw_mode_menu() in showcase.py already draw onto the cv2
preview window, just floating over the real desktop instead of confined
inside that window.

showcase.py's own cv2 window (with draw_hud/draw_mode_menu) is left
completely unchanged and keeps running alongside this - it's the
troubleshooting view, this is the everyday one.

DRAWER DESIGN
--------------
The HUD spends almost all of its time collapsed into two small, fixed
corner tabs - a status pill top-right and a help pill bottom-left - and
only expands into the full panel when the real mouse cursor is actually
resting over the relevant tab. This replaces an earlier version that
permanently painted a full-width opaque bar across the top of the
screen and a help/hint block anchored right at the bottom edge, which
caused two real problems:
  - the top bar sat over the top few dozen pixels of EVERY window on
    the desktop, fullscreen apps included, all the time, whether you
    were looking at it or not.
  - the bottom-left help text was anchored close enough to the screen
    edge that on a normal Windows taskbar it rendered underneath/behind
    the taskbar and was never actually visible.
Neither panel now ever occupies more than a small corner of the screen,
collapsed or expanded, and both are kept a fixed, generous margin away
from the bottom edge so a standard taskbar can never cover them.

Hover detection uses root.winfo_pointerxy() (an OS-level cursor query)
rather than Tkinter <Enter>/<Leave> bindings, because the transparent
parts of this window are click-through - the OS routes mouse events to
whatever's beneath them, so ordinary widget-level hover events never
fire over the transparent canvas. Polling the real cursor position once
per update() call sidesteps that entirely.

Layout is deliberately kept out of two zones so it coexists with
AROverlay (AR mode) and CameraPreview without either drawing over the
other:
  - top-left: left for AROverlay's chart-type palette (see
    ar_overlay.AROverlay._build_palette) - this window's own status tab
    lives top-RIGHT instead, so the two can never overlap.
  - bottom-right corner: left for CameraPreview's thumbnail (off by
    default - see showcase.py). This window's help tab is anchored
    bottom-LEFT instead.

Call update() once per camera frame, same as every other overlay window
in this project - see overlay.py's module docstring for why that's done
instead of running Tkinter on its own thread.
"""
import time
import tkinter as tk

import numpy as np

import window_capture

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False

try:
    import cv2
    _CV2_OK = True
except Exception:  # pragma: no cover
    _CV2_OK = False

_TRANSPARENT_KEY = "magenta"

# Tk falls back to its default family where Segoe UI isn't installed.
UI_FONT = "Segoe UI"


def _fit_scale_offset(src_w, src_h, dst_w, dst_h):
    """The "letterbox/pillarbox" fit used by _draw_drawing_canvas,
    pulled out as a small pure function so it's directly testable:
    a UNIFORM scale (same factor both axes - preserves every angle and
    circle-not-oval shape) that fits src_w x src_h inside dst_w x dst_h,
    plus the (off_x, off_y) to center it. Returns (scale, off_x, off_y,
    disp_w, disp_h)."""
    if src_w <= 0 or src_h <= 0:
        return 1.0, 0, 0, max(1, dst_w), max(1, dst_h)
    scale = min(dst_w / src_w, dst_h / src_h)
    disp_w, disp_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    return scale, (dst_w - disp_w) // 2, (dst_h - disp_h) // 2, disp_w, disp_h

# Kept well clear of a standard Windows taskbar (~40-48px tall) with
# margin to spare, regardless of whether a panel is collapsed or
# expanded - nothing this module draws is ever allowed to render below
# this line.
_BOTTOM_SAFE_Y = 70


class MainOverlay:
    HOVER_GRACE_S = 0.8

    def __init__(self, screen_size=None):
        self._closed = False
        self.hidden = True
        self.fullscreen_transparent = False

        self.root = tk.Tk()
        self.root.title("Hand Tracker HUD")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)

        try:
            self.root.attributes("-transparentcolor", _TRANSPARENT_KEY)
            self.fullscreen_transparent = True
        except tk.TclError:
            self.fullscreen_transparent = False

        if self.fullscreen_transparent:
            screen_w, screen_h = screen_size or (
                self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            )
            self.root.geometry(f"{screen_w}x{screen_h}+0+0")
            bg = _TRANSPARENT_KEY
        else:
            # SAFETY FALLBACK, same reasoning as AROverlay: never size an
            # opaque window to the full screen - that would block the
            # entire desktop with no title bar to close it. A modest
            # corner window at least keeps the app usable (with a
            # visible warning) on a platform where transparentcolor
            # isn't supported.
            print("[WARN] Main overlay: '-transparentcolor' not supported on "
                  "this platform - falling back to a small opaque window "
                  "instead of a full-screen transparent HUD. The floating "
                  "\"whole program on your desktop\" effect needs Windows; "
                  "the cv2 preview window still shows everything either way.")
            self.root.geometry("480x260+40+360")
            bg = "#141414"

        self.w, self.h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        if self.fullscreen_transparent:
            self.w, self.h = screen_size or (self.w, self.h)

        self.canvas = tk.Canvas(self.root, bg=bg, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.withdraw()

        # Hover state for the two drawers - re-derived every update()
        # from the real cursor position, not cached across frames.
        self._status_expanded = False
        self._help_expanded = False
        self._status_hold_until = 0.0     # grace period: a drawer stays open this long after the pointer leaves
        self._help_hold_until = 0.0
        self._status_btn_rects = {}
        # Continuous 0 (collapsed pill) .. 1 (fully expanded panel)
        # "openness", eased toward _status_expanded/_help_expanded each
        # frame (see _ease_toward) instead of the panels just snapping
        # between their two states - a drawer that visibly GROWS open
        # reads as a deliberate, responsive piece of UI; one that pops
        # between two fixed shapes instantly reads as static/unfinished
        # even though the underlying information is the same either way.
        self._status_open = 0.0
        self._help_open = 0.0
        self._last_update_t = None
        # Tkinter garbage-collects a PhotoImage the instant nothing in
        # Python still references it, even though the canvas is still
        # displaying it - this reference is what keeps Drawing mode's
        # floating canvas (see _draw_drawing_canvas) from going blank a
        # frame after it's drawn.
        self._drawing_photo = None
        self._ink_photos = {}            # (ty, tx) -> PhotoImage for Drawing mode's ink tiles
        self._capture_excluded = None    # last state sent to window_capture (None = never set)
        self._help_btn_rect = None

    # -- lifecycle -----------------------------------------------------
    def show(self):
        if self.hidden and not self._closed:
            self.root.deiconify()
            self.hidden = False

    def hide(self):
        if not self.hidden and not self._closed:
            self.root.withdraw()
            self.hidden = True

    def toggle(self):
        if self.hidden:
            self.show()
        else:
            self.hide()

    def close(self):
        if not self._closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._closed = True

    # -- hover helpers -------------------------------------------------
    def _pointer(self):
        """Real OS cursor position in screen coordinates, or None if it
        can't be read (window closing, platform doesn't support it, a
        test double standing in for Tk that doesn't implement it, etc).
        Deliberately NOT a Tk event - see module docstring."""
        try:
            pt = self.root.winfo_pointerxy()
        except tk.TclError:
            return None
        if (isinstance(pt, (tuple, list)) and len(pt) == 2
                and all(isinstance(v, (int, float)) for v in pt)):
            return pt
        return None

    @staticmethod
    def _inside(pt, x1, y1, x2, y2):
        return pt is not None and x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

    def _ease_toward(self, current, target, dt, rate=11.0):
        """Exponential ease, same style as ar3d.Object3D's own
        _sway_gain easing - current moves a fraction of the remaining
        distance to target each frame, scaled by dt so the speed is
        consistent regardless of actual frame rate. rate=11 reaches
        ~90% of the way there in about 200ms, which reads as snappy/
        responsive rather than sluggish while still being a visible
        animation rather than an instant jump."""
        return current + (target - current) * min(1.0, dt * rate)

    @staticmethod
    def _lerp_rect(a, b, t):
        return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))

    # -- drawing helpers -------------------------------------------------
    # Anything drawn straight onto the transparent canvas - no opaque
    # panel behind it - is only as visible as its own color is against
    # WHATEVER happens to be on the real desktop at that spot. Status/
    # help panels are fine (they paint their own dark rectangle first),
    # but corner brackets, the cursor dot, edge-hold zone outlines and
    # the volume flash text all used to be a single bare color, which
    # could all but disappear over a light-themed app, a bright photo,
    # a white document, etc. A dark halo drawn first, under the real
    # color, fixes this the same way subtitles/game HUDs stay readable
    # over any footage - it doesn't need to know what's actually behind
    # the window (reading real desktop pixels isn't something Tkinter
    # can do cheaply, if at all), it just guarantees contrast either way.
    def _halo_line(self, x1, y1, x2, y2, color, width):
        c = self.canvas
        c.create_line(x1, y1, x2, y2, fill="#000000", width=width + 3)
        c.create_line(x1, y1, x2, y2, fill=color, width=width)

    def _halo_oval(self, x1, y1, x2, y2, color, width):
        c = self.canvas
        c.create_oval(x1, y1, x2, y2, outline="#000000", width=width + 3)
        c.create_oval(x1, y1, x2, y2, outline=color, width=width)

    def _halo_rect(self, x1, y1, x2, y2, color, width):
        c = self.canvas
        c.create_rectangle(x1, y1, x2, y2, outline="#000000", width=width + 3)
        c.create_rectangle(x1, y1, x2, y2, outline=color, width=width)

    @staticmethod
    def _rounded_points(x1, y1, x2, y2, r):
        """Point list for a rounded rectangle via create_polygon(...,
        smooth=True) - Tkinter has no native rounded-rect primitive, but
        a smoothed polygon with a short straight run on each side of
        every corner (rather than a point AT the corner) renders as a
        clean curve there. A flat-cornered rectangle reads as a slightly
        colder, more "default-widget" look than the rest of this HUD's
        softer aesthetic is going for - this is the one shape primitive
        that gets reused for every panel/pill below."""
        r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        return [
            x1 + r, y1,  x2 - r, y1,  x2, y1,  x2, y1 + r,
            x2, y2 - r,  x2, y2,  x2 - r, y2,  x1 + r, y2,
            x1, y2,  x1, y2 - r,  x1, y1 + r,  x1, y1,
        ]

    def _rounded_rect(self, x1, y1, x2, y2, r, fill=None, outline=None, width=1):
        c = self.canvas
        kwargs = {"smooth": True, "width": width}
        if fill is not None:
            kwargs["fill"] = fill
        if outline is not None:
            kwargs["outline"] = outline
        return c.create_polygon(*self._rounded_points(x1, y1, x2, y2, r), **kwargs)

    def _halo_rounded_rect(self, x1, y1, x2, y2, r, color, width, fill=None):
        c = self.canvas
        pts = self._rounded_points(x1, y1, x2, y2, r)
        c.create_polygon(*pts, smooth=True, fill=fill or "", outline="#000000", width=width + 3)
        c.create_polygon(*pts, smooth=True, fill="", outline=color, width=width)

    def _grad_rounded(self, x1, y1, x2, y2, r, top, mid, bottom, outline=None, width=1):
        """Rounded rectangle shaded dark-at-the-bottom -> lighter-at-the-
        top (three stacked fills; Tkinter has no gradient). Opaque, so
        text on it is legible over any desktop, and softer than a flat
        black slab."""
        h = y2 - y1
        for col, frac in ((bottom, 1.0), (mid, 0.66), (top, 0.33)):
            yy2 = y1 + h * frac
            self._rounded_rect(x1, y1, x2, yy2, min(r, (yy2 - y1) / 2), fill=col)
        if outline:
            self._rounded_rect(x1, y1, x2, y2, r, outline=outline, width=width)

    def _label(self, x, y, text, color, size=10, bold=True, anchor="center"):
        self.canvas.create_text(x, y, text=text, fill=color, anchor=anchor,
                                font=(UI_FONT, size, "bold" if bold else "normal"))

    def _halo_text(self, x, y, text, color, font, anchor="center"):
        """Floating text. Small/medium text sits on a dark gradient pill
        (readable on ANY background - the old ring of offset black copies
        looked doubled and out of focus on bright ones); big display text
        (countdown digits, flash messages) gets a single soft drop shadow
        instead, since a pill that size would look like a banner."""
        size = int(font[1]) if len(font) > 1 else 10
        bold = (len(font) > 2 and font[2] == "bold")
        c = self.canvas
        if size >= 28:
            c.create_text(x + 3, y + 3, text=text, fill="#000000", anchor=anchor,
                          font=(UI_FONT, size, "bold" if bold else "normal"))
            self._label(x, y, text, color, size, bold, anchor)
            return
        size = max(10, size)
        w = len(text) * size * 0.60 + 22
        h = size * 1.95
        if anchor == "w":
            x1, x2 = x, x + w
        elif anchor in ("e", "se", "ne"):
            x1, x2 = x - w, x
        else:
            x1, x2 = x - w / 2, x + w / 2
        if anchor in ("sw", "s", "se"):
            y1, y2 = y - h, y
        else:
            y1, y2 = y - h / 2, y + h / 2
        self._grad_rounded(x1, y1, x2, y2, h / 2, "#1a2432", "#10171f", "#080b10", outline="#26384a")
        self._label((x1 + x2) / 2, (y1 + y2) / 2, text, color, size, bold)

    # -- always-on reticle -------------------------------------------------
    def _reticle(self, pt, pinching, accent, radius=None, ring=None):
        """Where the tracked fingertip currently is, on the real desktop,
        in EVERY mode (not only the ones that happen to draw their own
        cursor) and whether or not hand control is switched on - so you
        can always see what you're pointing at, e.g. to reach the help
        tab with control off, without needing the camera feed open. Ring
        tightens and fills while pinching. `radius` lets Drawing mode
        size the ring to the actual brush/eraser."""
        c = self.canvas
        x, y = pt
        r = radius if radius else (9 if pinching else 15)
        if pinching:
            c.create_oval(x - r, y - r, x + r, y + r, fill=accent, outline="", stipple="gray50")
        c.create_oval(x - r, y - r, x + r, y + r, outline="#000000", width=5)
        c.create_oval(x - r, y - r, x + r, y + r, outline=accent, width=2)
        c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#000000", outline="")
        c.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, fill="#ffffff", outline="")
        if ring:
            # Hold-to-confirm progress (Call mode): an arc filling around
            # the reticle while a gesture is being held, so you can see an
            # action is about to fire - and drop your hand to cancel it.
            rr = r + 9
            c.create_oval(x - rr, y - rr, x + rr, y + rr, outline="#000000", width=6)
            c.create_arc(x - rr, y - rr, x + rr, y + rr, start=90, extent=-max(2, 359 * ring),
                         style="arc", outline="#7cf2a1", width=4)

    def _corner_brackets(self, color, length=22, thickness=2, margin=14):
        w, h = self.w, self.h
        for x, y, dx, dy in ((margin, margin, 1, 1), (w - margin, margin, -1, 1),
                              (margin, h - margin, 1, -1), (w - margin, h - margin, -1, -1)):
            self._halo_line(x, y, x + dx * length, y, color, thickness)
            self._halo_line(x, y, x, y + dy * length, color, thickness)

    # -- STATUS drawer (top-right): collapsed pill or expanded panel,
    # never wider than itself - the rest of the top edge is always left
    # completely clear for whatever's running underneath.
    def _status_tab_rect(self):
        w = self.w
        x2, y1 = w - 16, 14
        x1, y2 = x2 - 150, y1 + 34
        return x1, y1, x2, y2

    def _status_panel_rect(self):
        px2, py1 = self.w - 16, 14
        return px2 - 360, py1, px2, py1 + 196

    def _status_bar(self, mode_name, color, enabled, gesture, confidence, fps, ai_label, openness,
                    camera_on=True):
        """openness: 0 (collapsed pill) .. 1 (fully expanded panel) -
        see MainOverlay._ease_toward. The panel's own OUTLINE grows
        smoothly between the two rects every frame; the detailed content
        only appears past openness=0.6 (Tkinter has no real per-shape
        alpha to fade text in/out with, so a threshold is the practical
        stand-in - still reads as the panel "arriving" rather than
        content snapping into an already-settled shape) and the
        collapsed pill's own dot+label fade out symmetrically below
        openness=0.4, so there's always exactly one coherent thing on
        screen, never a flicker of both at once."""
        c = self.canvas
        self._status_btn_rects = {}
        tx1, ty1, tx2, ty2 = self._status_tab_rect()
        px1, py1, px2, py2 = self._status_panel_rect()
        x1, y1, x2, y2 = self._lerp_rect((tx1, ty1, tx2, ty2), (px1, py1, px2, py2), openness)
        r = 16 * (1 - openness) + 6 * openness

        self._grad_rounded(x1, y1, x2, y2, r, "#1b2230", "#121823", "#0b0f16", outline=color, width=2 if openness > 0.5 else 1)

        if openness < 0.4:
            k = 1 - openness / 0.4
            c.create_oval(tx1 + 12, (ty1 + ty2) // 2 - 6, tx1 + 24, (ty1 + ty2) // 2 + 6,
                           fill=color, outline="")
            status_dot = "#6effaa" if enabled else "#6e6eff"
            c.create_oval(tx2 - 22, (ty1 + ty2) // 2 - 5, tx2 - 12, (ty1 + ty2) // 2 + 5,
                           fill=status_dot, outline="")
            if k > 0.4:  # label itself only once the pill's basically settled
                c.create_text((tx1 + tx2) // 2 - 4, (ty1 + ty2) // 2, text=mode_name,
                               fill="#f5f5f5", font=(UI_FONT, 10, "bold"))
        if openness < 0.6:
            return

        c.create_oval(px1 + 16, py1 + 16, px1 + 30, py1 + 30, fill=color, outline="")
        c.create_text(px1 + 42, py1 + 23, text=mode_name, fill="#f5f5f5",
                       font=(UI_FONT, 15, "bold"), anchor="w")

        status_txt = "CONTROL ON" if enabled else "CONTROL OFF"
        status_color = "#6effaa" if enabled else "#6e6eff"
        (bx1, by1, bx2, by2) = (px2 - 150, py1 + 10, px2 - 14, py1 + 36)
        self._rounded_rect(bx1, by1, bx2, by2, 10, outline=status_color, width=2)
        c.create_oval(bx1 + 10, (by1 + by2) // 2 - 4, bx1 + 18, (by1 + by2) // 2 + 4,
                       fill=status_color, outline="")
        c.create_text(bx1 + 26, (by1 + by2) // 2, text=status_txt, fill=status_color,
                       font=(UI_FONT, 9, "bold"), anchor="w")

        y = py1 + 72
        c.create_text(px1 + 16, y - 14, text="GESTURE", fill="#8c8c96",
                       font=(UI_FONT, 8), anchor="w")
        c.create_text(px1 + 16, y + 8, text=gesture or "-", fill="#ffffff",
                       font=(UI_FONT, 13, "bold"), anchor="w")

        bar_x, bar_w, bar_h = px1 + 150, 130, 16
        by1, by2 = y - bar_h + 2, y + 6
        c.create_text(bar_x, y - 14, text="CONFIDENCE", fill="#8c8c96",
                       font=(UI_FONT, 8), anchor="w")
        self._rounded_rect(bar_x, by1, bar_x + bar_w, by2, bar_h / 2, fill="#2d2d32")
        fill_w = int(bar_w * max(0.0, min(1.0, confidence)))
        if fill_w > 3:
            bar_color = "#64ff8c" if confidence > 0.7 else "#3cd2ff" if confidence > 0.45 else "#5a6eff"
            self._rounded_rect(bar_x, by1, bar_x + fill_w, by2, bar_h / 2, fill=bar_color)

        # Two buttons you can PINCH with the pointer - so the things people
        # most need to flip (can the app touch my computer? is it holding my
        # camera?) never require the mouse or keyboard.
        if openness > 0.85:
            by1, by2 = py1 + 112, py1 + 146
            bw = (px2 - px1 - 32 - 10) / 2
            cam = (px1 + 16, by1, px1 + 16 + bw, by2)
            ctl = (cam[2] + 10, by1, cam[2] + 10 + bw, by2)
            self._status_btn_rects = {"camera": cam, "control": ctl}
            for key, rect, label, on in (("camera", cam, "Camera: ON  (release)" if camera_on
                                          else "Camera: OFF  (resume)", camera_on),
                                         ("control", ctl, "Control: ON" if enabled else "Control: OFF", enabled)):
                hot = self._status_hover == key
                top, mid, bot = ("#1d4a31", "#153824", "#0d2417") if on else ("#4a3a14", "#382c0f", "#241c09")
                self._grad_rounded(*rect, 12, top, mid, bot,
                                   outline="#ffffff" if hot else ("#7cf2a1" if on else "#ffd24d"),
                                   width=2 if hot else 1)
                self._label((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2, label,
                            "#d6ffe6" if on else "#ffe9a8", 10)

        fps_color = "#64ff8c" if fps >= 20 else "#3cd2ff" if fps >= 12 else "#5a6eff"
        c.create_text(px2 - 14, py2 - 16, text=f"{fps:.0f} FPS", fill=fps_color,
                       font=(UI_FONT, 10, "bold"), anchor="e")
        if ai_label:
            c.create_text(px1 + 16, py2 - 16, text=f"AI  {ai_label}", fill="#beafff",
                           font=(UI_FONT, 8, "bold"), anchor="w")

    # -- HELP drawer (bottom-left): collapsed pill or expanded panel,
    # both always kept above _BOTTOM_SAFE_Y from the bottom edge so a
    # taskbar can never cover either state.
    def _help_tab_rect(self):
        y2 = self.h - _BOTTOM_SAFE_Y
        y1 = y2 - 34
        x1, x2 = 24, 24 + 118
        return x1, y1, x2, y2

    def _help_panel_rect(self, mode_name, help_lines):
        """Expanded help panel rect. Grows UPWARD from the same bottom
        anchor the tab uses, so its bottom edge never drifts closer to the
        taskbar than the collapsed tab already safely is. Shared by the
        drawing code AND the hover test (see update) - the hover test used
        to look only at the small collapsed tab, so reaching for anything
        inside the grown panel (the Tutorial button!) meant leaving the
        tab, and the panel collapsed under your finger."""
        tx1, ty1, tx2, ty2 = self._help_tab_rect()
        line_h = 24
        panel_h = 56 + line_h * (len(help_lines) + 1)
        title_txt = f"{mode_name}  -  CONTROLS"
        content_w = max(300, max([len(title_txt)] + [len(l) for l in help_lines]) * 7 + 60)
        x1 = 24
        return x1, ty2 - panel_h, min(self.w - 24, x1 + content_w), ty2

    def _help_panel(self, mode_name, color, help_lines, openness):
        """openness: 0 (collapsed pill) .. 1 (fully expanded panel) -
        same smooth-grow/threshold-content approach as _status_bar
        above; see that method's docstring for the reasoning."""
        c = self.canvas
        tx1, ty1, tx2, ty2 = self._help_tab_rect()
        line_h = 24
        title_txt = f"{mode_name}  -  CONTROLS"
        panel_x1, panel_y1, panel_x2, panel_y2 = self._help_panel_rect(mode_name, help_lines)

        self._help_btn_rect = None
        x1, y1, x2, y2 = self._lerp_rect((tx1, ty1, tx2, ty2),
                                         (panel_x1, panel_y1, panel_x2, panel_y2), openness)
        r = 16 * (1 - openness) + 6 * openness
        self._grad_rounded(x1, y1, x2, y2, r, "#1b2230", "#121823", "#0b0f16", outline=color, width=2 if openness > 0.5 else 1)

        if openness < 0.4:
            if (1 - openness / 0.4) > 0.4:
                c.create_text((tx1 + tx2) // 2, (ty1 + ty2) // 2, text="?  Help",
                               fill="#e1e1e6", font=(UI_FONT, 10, "bold"))
        if openness < 0.6:
            return

        c.create_line(panel_x1 + 16, panel_y1 + 34, panel_x2 - 16, panel_y1 + 34,
                       fill="#4a4a50", width=1)
        c.create_text(panel_x1 + 16, panel_y1 + 20, text=title_txt, fill=color,
                       font=(UI_FONT, 11, "bold"), anchor="w")
        for i, line in enumerate(help_lines):
            ly = panel_y1 + 52 + i * line_h
            c.create_oval(panel_x1 + 18, ly - 4, panel_x1 + 24, ly + 2, fill=color, outline="")
            c.create_text(panel_x1 + 32, ly, text=line, fill="#e1e1e6",
                           font=(UI_FONT, 10), anchor="w")
        hint_y = panel_y1 + 52 + len(help_lines) * line_h
        c.create_text(panel_x1 + 16, hint_y, text="T tutorial   C call   V virtual cam   K release camera   O hide from share",
                       fill="#8c8c96", font=(UI_FONT, 8), anchor="w")
        # "Tutorial" button: pinch it to start the walkthrough (only ever
        # starts when asked - never automatically).
        bx2, by2 = panel_x2 - 14, panel_y1 + 30
        bx1, by1 = bx2 - 96, by2 - 26
        if openness > 0.85:
            self._help_btn_rect = (bx1, by1, bx2, by2)
            self._grad_rounded(bx1, by1, bx2, by2, 12, "#16414f", "#0f2f3b", "#0a1f28",
                               outline="#39e6ff", width=1)
            self._label((bx1 + bx2) / 2, (by1 + by2) / 2, "Tutorial", "#e8fbff", 10)

    # -- Drawing mode: the whiteboard itself ---------------------------
    def _draw_drawing_canvas(self, bgr_canvas, frame_size):
        """Drawing mode's actual content - the strokes themselves - used
        to only ever get composited onto the cv2 preview window's image
        (features/drawing.py bakes cv2.line() calls straight into a
        persistent pixel buffer, frame by frame; there's no separate
        list of strokes to replay here the way AR's charts/3D objects
        work). Floating the strokes over the real desktop instead of
        just the chrome means taking that same pixel buffer, treating
        its black (undrawn) background as transparent, and painting the
        rest onto this window - still a single operation per frame, just
        an image instead of canvas shapes.

        bgr_canvas: drawing.canvas, an (H, W, 3) uint8 OpenCV/numpy array
          in the camera frame's own resolution (frame_size = (W, H)),
          NOT the screen's. Scaled up UNIFORMLY (same factor on both
          axes) and centered, not stretched to exactly fill the screen -
          see the scale/off_x/off_y comments below for why a non-uniform
          stretch isn't just a minor visual quirk here.
        """
        if not _PIL_OK or bgr_canvas is None:
            return
        fh, fw = bgr_canvas.shape[:2]
        if fw == 0 or fh == 0:
            return
        # UNIFORM scale (same factor for both axes), not a direct
        # fw,fh -> self.w,self.h stretch. The camera frame's aspect
        # ratio (e.g. 16:9) essentially never exactly matches the real
        # screen's (16:10, ultrawide, a laptop's native ratio, ...) - a
        # non-uniform stretch quietly changes the ANGLE of every
        # diagonal stroke to fit the mismatched shape, which is exactly
        # what reads as the drawing coming out "crooked" on the overlay
        # even though the same canvas looks perfectly straight inside
        # the square-pixel cv2 window. Scaling both axes by the same
        # factor and centering (letterboxed/pillarboxed, like a video
        # that doesn't match its player's aspect ratio) keeps every
        # angle and the circle-not-oval shape of a dot exactly as drawn.
        scale, off_x, off_y, disp_w, disp_h = _fit_scale_offset(fw, fh, self.w, self.h)

        # Nearest-neighbour (not a smooth resize) deliberately: this is
        # a hard black/not-black mask turned into alpha below, and a
        # smoothing resize would leave half-transparent grey fringing
        # around every stroke instead of a clean edge.
        if _CV2_OK:
            rgb = cv2.cvtColor(bgr_canvas, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        else:
            rgb = bgr_canvas[:, :, ::-1]
            rgb = np.array(Image.fromarray(rgb).resize((disp_w, disp_h), Image.NEAREST))

        # Full screen-sized, fully transparent canvas with the
        # (correctly-proportioned, possibly letterboxed) drawing pasted
        # into its center - rather than the drawing itself being
        # distorted to fill the whole screen.
        full_rgba = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        alpha = np.where(rgb.any(axis=2), 255, 0).astype(np.uint8)
        full_rgba[off_y:off_y + disp_h, off_x:off_x + disp_w, :3] = rgb
        full_rgba[off_y:off_y + disp_h, off_x:off_x + disp_w, 3] = alpha
        image = Image.fromarray(full_rgba, mode="RGBA")

        # Same allocate-once/paste-repeatedly pattern CameraPreview uses
        # (ar_overlay.CameraPreview.update) - this image is full-screen
        # sized, bigger than that thumbnail ever was, so skipping a
        # fresh PhotoImage allocation every single frame matters even
        # more here.
        if self._drawing_photo is not None:
            try:
                self._drawing_photo.paste(image)
            except Exception:
                self._drawing_photo = None
        if self._drawing_photo is None:
            # master=self.canvas (NOT omitted) matters here: this app
            # opens several independent tk.Tk() windows (this overlay,
            # the AR overlay), each its own separate Tcl interpreter.
            # Without an explicit master, ImageTk.PhotoImage binds to
            # whichever one happens to be Tkinter's "default root" at
            # that moment - which is NOT guaranteed to be the interpreter
            # self.canvas actually lives in - so create_image() would
            # then be asked for an image name that only exists in a
            # DIFFERENT interpreter and fail with a bare
            # "image ... doesn't exist" TclError that crashed the whole
            # program. Binding explicitly to this window's own canvas
            # removes the ambiguity entirely.
            self._drawing_photo = ImageTk.PhotoImage(image, master=self.canvas)
        self.canvas.create_image(0, 0, image=self._drawing_photo, anchor="nw")

    # -- Whack-a-Mole: the board itself ---------------------------------
    def _draw_whackamole(self, game_state):
        """Same situation as Drawing mode (see _draw_drawing_canvas) but
        the other way around: WhackAMoleGame's state (mole position,
        score, countdown, hit flash, button rects - see features/
        game.py) is already plain numbers, not baked-in pixels, so it's
        drawn natively as Tkinter shapes instead - cheaper, and it gets
        the same dark-halo contrast treatment as the rest of this HUD
        (see _halo_* above) for free.

        game_state: a plain dict built by showcase.py from the live
        WhackAMoleGame instance each frame (see its call to
        main_overlay_win.update) - kept as a dict rather than passing
        the game object itself so this module never needs to import
        features/game.py or know about pygame/cv2 asset loading.
        """
        c = self.canvas
        mx, my = self.w // 2, 150

        if game_state.get("countdown_active"):
            remaining = game_state["countdown_remaining"]
            text = str(remaining) if remaining > 0 else "GO!"
            color = "#ff5050" if remaining > 0 else "#50ff78"
            self._halo_text(mx, self.h // 2, text, color, (UI_FONT, 64, "bold"))
            return

        if not game_state.get("active"):
            # Not started yet - WhackAMoleGame only ever starts via
            # handle_tap() hitting its Start button rect (there's no
            # dedicated start gesture), so that button needs to actually
            # be drawn here, in the right screen-mapped spot, not just
            # described - otherwise there'd be nothing on the real
            # desktop telling you where to point to begin.
            sx1, sy1, sx2, sy2 = game_state["start_button"]
            self._halo_text(mx, my - 50, "Pinch the button to start", "#ffe14d",
                            (UI_FONT, 14, "bold"))
            c.create_rectangle(sx1, sy1, sx2, sy2, fill="#1a1a1a", outline="#50ff78", width=2)
            self._halo_text((sx1 + sx2) // 2, (sy1 + sy2) // 2, "START", "#50ff78",
                            (UI_FONT, 14, "bold"))
            return

        self._halo_text(mx, my, "Pinch the mole to whack it!", "#ff6a4c",
                        (UI_FONT, 14, "bold"))

        bx1, by1, bx2, by2 = game_state["bounds"]
        self._halo_rect(bx1, by1, bx2, by2, "#ffff00", 2)

        hit = game_state.get("hit_flash")
        hx, hy = game_state["mole"]
        r = game_state["mole_radius"]
        if hit:
            r = int(r * 1.4)
            c.create_oval(hx - r, hy - r, hx + r, hy + r, outline="#ffffff", width=0,
                          fill="#ffe14d", stipple="gray50")
            r = game_state["mole_radius"]
        # A simple drawn mole (ears, face, eyes) rather than loading the
        # game's own PNG asset into Tkinter - keeps this module free of
        # another image-loading path and fits the rest of the HUD's flat
        # vector look.
        c.create_oval(hx - r, hy - r, hx + r, hy + r, fill="#8b5a2b", outline="#4a2f16", width=2)
        ear = r * 0.35
        for sign in (-1, 1):
            ex, ey = hx + sign * r * 0.7, hy - r * 0.7
            c.create_oval(ex - ear, ey - ear, ex + ear, ey + ear,
                         fill="#8b5a2b", outline="#4a2f16", width=2)
        for sign in (-1, 1):
            c.create_oval(hx + sign * r * 0.35 - 4, hy - r * 0.1 - 4,
                         hx + sign * r * 0.35 + 4, hy - r * 0.1 + 4, fill="#1a1a1a", outline="")
        c.create_oval(hx - 5, hy + r * 0.25 - 5, hx + 5, hy + r * 0.25 + 5,
                     fill="#2a1810", outline="")

        self._halo_text(70, 60, f"Score: {game_state['score']}", "#ffffff",
                        (UI_FONT, 18, "bold"))

        cx1, cy1, cx2, cy2 = game_state["close_button"]
        c.create_rectangle(cx1, cy1, cx2, cy2, fill="#1a1a1a", outline="#ff4040", width=2)
        self._halo_text((cx1 + cx2) // 2, (cy1 + cy2) // 2, "Pinch to close",
                        "#ff8080", (UI_FONT, 9, "bold"))

    _status_hover = None

    def status_button_rects(self):
        """{"camera": rect, "control": rect} while the top-right panel is
        fully open (else {}) - showcase.py hit-tests a pinch against these."""
        return dict(self._status_btn_rects)

    def help_button_rect(self):
        """Screen rect of the help panel's Tutorial button while it is
        actually showing (else None) - showcase.py hit-tests a pinch on it."""
        return self._help_btn_rect

    def set_capture_excluded(self, excluded):
        """Hide (True) / show (False) this window in screen sharing and
        recordings while staying visible on the real monitor - see
        window_capture.py. Cheap to call every frame: only acts on change."""
        excluded = bool(excluded)
        if excluded == self._capture_excluded:
            return self._capture_excluded
        try:
            self.root.update_idletasks()
            result = window_capture.set_capture_excluded(self.root.winfo_id(), excluded)
        except Exception:
            result = "unsupported"
        # Remember the request even when unsupported, so a platform that
        # can't do it isn't retried (and re-failed) 30 times a second.
        self._capture_excluded = excluded
        self.capture_result = result
        return excluded

    # -- tutorial card, call bar, toast --------------------------------------------------
    def _draw_tutorial(self, v):
        c = self.canvas
        x0, y0, x1, y1 = v["rect"]
        self._grad_rounded(x0, y0, x1, y1, 20, "#1c2635", "#121a26", "#0a0f17", outline="#39e6ff", width=2)
        self._label(x0 + 22, y0 + 28, v["title"], "#ffffff", 16, anchor="w")
        self._label(x1 - 22, y0 + 28, f"{v['idx'] + 1} / {v['total']}", "#7aa6bd", 10, anchor="e")
        if v["done"]:
            self._grad_rounded(x1 - 150, y0 + 14, x1 - 84, y0 + 42, 14, "#1d5a37", "#154328", "#0d2c1a",
                               outline="#7cf2a1", width=1)
            self._label(x1 - 117, y0 + 28, "Nice!", "#b6ffd2", 10)
        elif v["progress"] is not None:
            self._rounded_rect(x0 + 22, y0 + 46, x1 - 22, y0 + 52, 3, fill="#1d2735")
            if v["progress"] > 0.02:
                self._rounded_rect(x0 + 22, y0 + 46, x0 + 22 + (x1 - x0 - 44) * v["progress"], y0 + 52, 3,
                                   fill="#39e6ff")
        c.create_text(x0 + 22, y0 + 62, text=v["body"], fill="#c9d6e2", anchor="nw",
                      width=(x1 - x0 - 44), font=(UI_FONT, 12))
        for i in range(v["total"]):                       # progress dots
            cx = (x0 + x1) / 2 + (i - (v["total"] - 1) / 2) * 16
            cy = y1 - 31
            r = 4 if i == v["idx"] else 3
            c.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#39e6ff" if i <= v["idx"] else "#2a3646",
                          outline="")
        for b in v["buttons"]:
            bx0, by0, bx1, by1 = b["rect"]
            hot = v.get("hover") == b["id"]
            primary = b["id"] == "next"
            top, mid, bot = (("#16414f", "#0f2f3b", "#0a1f28") if primary else ("#1b2230", "#121823", "#0b0f16"))
            self._grad_rounded(bx0, by0, bx1, by1, 12, top, mid, bot,
                               outline="#ffffff" if hot else ("#39e6ff" if primary else "#33445a"),
                               width=2 if hot else 1)
            self._label((bx0 + bx1) / 2, (by0 + by1) / 2, b["label"], "#e8fbff", 11)

    def _draw_call_bar(self, v):
        for b in v["buttons"]:
            x0, y0, x1, y1 = b["rect"]
            hot = v.get("hover") == b["id"]
            self._grad_rounded(x0, y0, x1, y1, 16, "#1b2636", "#111a26", "#0a0f16",
                               outline="#ffffff" if hot else "#39e6ff", width=3 if hot else 2)
            self._label((x0 + x1) / 2, (y0 + y1) / 2, b["label"], "#e8fbff", 12)
        x0 = min(b["rect"][0] for b in v["buttons"])
        x1 = max(b["rect"][2] for b in v["buttons"])
        y0 = min(b["rect"][1] for b in v["buttons"])
        self._halo_text((x0 + x1) / 2, y0 - 22, v["status"], "#7cf2a1" if v["ok"] else "#ffd24d",
                        (UI_FONT, 11, "bold"))

    def _draw_banner(self, v):
        """Big centred notice (used while the camera is released: the app
        is deliberately blind, so say so and say how to wake it)."""
        c = self.canvas
        w, h = 640, 60 + 26 * len(v["lines"]) + 26
        x0, y0 = (self.w - w) // 2, (self.h - h) // 2 - 80
        self._grad_rounded(x0, y0, x0 + w, y0 + h, 22, "#2a2412", "#1c180c", "#110f07",
                           outline="#ffd24d", width=2)
        self._label(self.w // 2, y0 + 34, v["title"], "#ffe9a8", 18)
        for i, line in enumerate(v["lines"]):
            self._label(self.w // 2, y0 + 70 + i * 26, line, "#e8e2cf", 12, bold=False)

    def _draw_toast(self, text):
        self._halo_text(self.w // 2, self.h - 190, text, "#e8fbff", (UI_FONT, 12, "bold"))

    # -- Drawing mode: ink tiles + toolbar ----------------------------------
    def _draw_ink(self, ink):
        """Only tiles touched since the last frame are converted and
        pasted into Tk (see overlay_drawing.InkCanvas); everything else
        is just re-placed from the already-uploaded images. Tiles with no
        ink are skipped entirely."""
        if not _PIL_OK:
            return
        for ty, tx in ink.pop_dirty():
            img = Image.fromarray(ink.tile_pixels(ty, tx), mode="RGBA")
            photo = self._ink_photos.get((ty, tx))
            if photo is not None:
                try:
                    photo.paste(img)
                except Exception:
                    photo = None
            if photo is None:
                # master= for the same wrong-Tk-interpreter reason as
                # _draw_drawing_canvas
                photo = ImageTk.PhotoImage(img, master=self.canvas)
                self._ink_photos[(ty, tx)] = photo
        for (ty, tx), photo in self._ink_photos.items():
            if ink.has_ink[ty, tx]:
                x0, y0, _, _ = ink.tile_rect(ty, tx)
                self.canvas.create_image(x0, y0, image=photo, anchor="nw")

    _TOOLTIPS = {"pen": "Pen", "eraser": "Eraser", "clear": "Clear all", "save": "Save PNG",
                 "hide": "Hide panel", "show": "Paint"}

    def _draw_paint_dock(self, v):
        c = self.canvas
        hover = v["hover"]
        if not v["open"]:
            x0, y0, x1, y1 = v["buttons"][0]["rect"]
            hot = hover == "show"
            self._grad_rounded(x0, y0, x1, y1, (x1 - x0) / 2, "#1b2636", "#111a26", "#0a0f16",
                               outline="#ffffff" if hot else "#39e6ff", width=2)
            self._pen_icon(c, (x0 + x1) / 2, (y0 + y1) / 2, 11, "#e8fbff")
            if hot:
                self._halo_text(x1 + 12, (y0 + y1) / 2, "Paint", "#e8fbff", (UI_FONT, 11, "bold"), anchor="w")
            return
        dx0, dy0, dx1, dy1 = v["dock"]
        self._grad_rounded(dx0, dy0, dx1, dy1, 24, "#1a2230", "#111722", "#0a0e14",
                           outline="#2b3a4d", width=1)
        tip = None
        for b in v["buttons"]:
            x0, y0, x1, y1 = b["rect"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bid, kind = b["id"], b["kind"]
            hot = hover == bid
            sel = ((kind == "tool" and v["selected"].get(bid)) or
                   (kind == "color" and v["color_idx"] == int(bid[5:])) or
                   (kind == "thick" and v["thick_idx"] == int(bid[5:])))
            ring = "#ffffff" if hot else ("#39e6ff" if sel else None)
            if kind == "color":
                r = (x1 - x0) / 2 - 4
                if sel:
                    c.create_oval(cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5, outline="#ffffff", width=2)
                elif hot:
                    c.create_oval(cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4, outline="#ffffff", width=1)
                c.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#%02x%02x%02x" % b["rgb"],
                              outline="#05070a", width=2)
                if hot:
                    tip = (x1, cy, b["name"].title())
                continue
            if sel:
                self._grad_rounded(x0, y0, x1, y1, 10, "#16414f", "#0f2f3b", "#0a1f28",
                                   outline=ring, width=2)
            else:
                self._grad_rounded(x0, y0, x1, y1, 10, "#171d28", "#10151d", "#0a0d13",
                                   outline=ring or "#222c3b", width=2 if hot else 1)
            ink = "#e8fbff" if (sel or hot) else "#9fb4c4"
            if bid == "pen":
                self._pen_icon(c, cx, cy, 9, ink)
            elif bid == "eraser":
                c.create_polygon(cx - 9, cy + 2, cx - 1, cy - 8, cx + 9, cy - 1, cx + 1, cy + 9,
                                 fill=ink, outline="")
                c.create_line(cx - 5, cy + 6, cx + 5, cy - 4, fill="#0a0d13", width=2)
            elif kind == "thick":
                rr = 2 + int(bid[5:]) * 2.5 + 1
                c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, fill=ink, outline="")
            elif bid == "clear":
                col = "#ff8a8a" if (hot or sel) else "#d98080"
                c.create_line(cx - 7, cy - 7, cx + 7, cy + 7, fill=col, width=3)
                c.create_line(cx - 7, cy + 7, cx + 7, cy - 7, fill=col, width=3)
            elif bid == "save":
                c.create_line(cx, cy - 8, cx, cy + 3, fill=ink, width=3)
                c.create_polygon(cx - 6, cy + 1, cx + 6, cy + 1, cx, cy + 8, fill=ink, outline="")
                c.create_line(cx - 8, cy + 10, cx + 8, cy + 10, fill=ink, width=2)
            elif bid == "hide":
                c.create_line(cx + 4, cy - 8, cx - 4, cy, cx + 4, cy + 8, fill=ink, width=3)
            if hot:
                tip = (x1, cy, self._tooltip_for(bid))
        if tip is not None:
            self._halo_text(dx1 + 12, tip[1], tip[2], "#e8fbff", (UI_FONT, 11, "bold"), anchor="w")
        if v["toast"]:
            self._halo_text(self.w // 2, self.h - 110, v["toast"], "#7cf2a1", (UI_FONT, 13, "bold"))

    def _tooltip_for(self, bid):
        if bid.startswith("thick"):
            return ("Thin", "Medium", "Thick")[int(bid[5:])]
        return self._TOOLTIPS.get(bid, bid.title())

    @staticmethod
    def _pen_icon(c, cx, cy, r, color):
        c.create_line(cx - r, cy + r, cx + r * 0.6, cy - r * 0.6, fill=color, width=4, capstyle="round")
        c.create_polygon(cx - r - 1, cy + r + 1, cx - r + 1, cy + r - 5, cx - r + 5, cy + r - 1,
                         fill=color, outline="")

    def _mode_menu(self, mode_order, modes, highlight_index):
        c = self.canvas
        w, h = self.w, self.h
        c.create_rectangle(0, 0, w, h, fill="#0c0c10", outline="")

        c.create_text(w // 2, 80, text="MODE MENU", fill="#f5f5f5",
                       font=(UI_FONT, 22, "bold"))
        c.create_text(w // 2, 118, text="Point at a tile and pinch to select   |   Thumb Down to cancel",
                       fill="#bebec3", font=(UI_FONT, 11))

        n = len(mode_order)
        margin, gap = 60, 20
        tile_w = (w - 2 * margin - gap * (n - 1)) // n
        tile_h = 200
        tile_y = h // 2 - tile_h // 2

        for i, key in enumerate(mode_order):
            m = modes[key]
            x1 = margin + i * (tile_w + gap)
            x2 = x1 + tile_w
            selected = (i == highlight_index)
            if selected:
                self._grad_rounded(x1, tile_y, x2, tile_y + tile_h, 18, "#243044", "#172131",
                                   "#0e141d", outline=m["color"], width=3)
            else:
                self._grad_rounded(x1, tile_y, x2, tile_y + tile_h, 18, "#171d28", "#10151d",
                                   "#0a0d13", outline="#2b3646", width=1)
            c.create_text((x1 + x2) // 2, tile_y + 60, text=str(i + 1), fill=m["color"],
                           font=(UI_FONT, 18, "bold"))
            words = m["name"].split()
            line_y = tile_y + tile_h - 50
            for word in words:
                c.create_text((x1 + x2) // 2, line_y, text=word, fill="#ebebf0",
                               font=(UI_FONT, 10))
                line_y += 22

    def _cursor_dot(self, screen_point, pinching, color="#39ff88"):
        if screen_point is None:
            return
        x, y = screen_point
        ring_color = "#ff4d4d" if pinching else color
        r = 13 if pinching else 9
        self._halo_oval(x - r, y - r, x + r, y + r, ring_color, 2)

    def _edge_zones(self, edge_ratio, active_zone, progress, color):
        """Left/right vertical bars marking Media mode's hold-to-seek
        zones, matching the ones showcase.py already draws on the cv2
        frame (draw_edge_zones) - ported here so Media mode's seek
        gesture is discoverable without the debug window open."""
        c = self.canvas
        w, h = self.w, self.h
        zone_w = max(1, int(edge_ratio * w))
        for side, x1, x2 in (("left", 0, zone_w), ("right", w - zone_w, w)):
            active = (active_zone == side)
            outline = color if active else "#9a9aa2"
            self._halo_rect(x1, 0, x2, h, outline, 2 if active else 1)
            if active and progress > 0:
                # This one DOES get an opaque-ish fill (the progress
                # bar itself), so it reads fine without a halo - only
                # its own thin border above needed one.
                fill = "#3cdc5a" if progress >= 1.0 else color
                fill_h = int(h * progress)
                fy1, fy2 = (h - fill_h) // 2, (h + fill_h) // 2
                c.create_rectangle(x1 + 3, fy1, x2 - 3, fy2, fill=fill, outline="#000000")
            label = "<- HOLD" if side == "left" else "HOLD ->"
            self._halo_text((x1 + x2) // 2, h // 2, label, "#ffffff", (UI_FONT, 10, "bold"))

    def _flash_text(self, text, color):
        self._halo_text(self.w // 2, 150, text, color, (UI_FONT, 20, "bold"))

    # -- main entry point --------------------------------------------------
    def update(self, mode_key, modes, mode_order, enabled, gesture, confidence, fps,
               show_help, ai_label=None, menu_open=False, menu_highlight=0,
               cursor_point=None, pinching=False, edge_info=None, flash=None,
               help_hidden_note=None, drawing_canvas=None, frame_size=None,
               game_state=None, hand_point=None, reticle=False, ink=None, paint=None,
               extras=None):
        """Redraws the whole HUD and pumps Tkinter's event queue. Call
        once per camera frame while the overlay is visible; call hide()
        otherwise (see showcase.py's overlay_visible flag).

        Both the status panel (top-right) and the help panel
        (bottom-left) are collapsed into small tabs by default and only
        expand while the real mouse cursor is resting over them - see
        the module docstring. Neither ever spans the screen, so
        fullscreen windows and the taskbar are left alone.

          mode_key/modes/mode_order: same MODES dict / MODE_ORDER list
            showcase.py already builds for draw_hud/draw_mode_menu. Each
            mode's "color" is read as a cv2 BGR tuple and converted to
            an RGB hex string here, so callers don't need to convert.
          menu_open/menu_highlight: mirrors draw_mode_menu's state - when
            menu_open, the menu replaces the drawers for this frame
            (same as the cv2 version).
          cursor_point/pinching: an optional generic fingertip marker,
            for modes without their own dedicated indicator (Desktop has
            CursorReticle, Presentation's laser has LaserPointer, AR has
            its own cursor - pass None for those; Media mode is the
            typical caller).
          edge_info: optional (edge_ratio, active_zone, progress) tuple
            to draw Media mode's hold-to-seek zone bars.
          flash: optional (text, color) tuple, color as a cv2 BGR tuple,
            for a brief center-screen message (Media mode's
            "VOLUME UP/DOWN").
          help_hidden_note: if the performance governor is suppressing
            the help panel even though the user still wants it, pass the
            same note shown on the cv2 HUD - shown in the collapsed
            help tab's place instead of just vanishing silently.
          drawing_canvas/frame_size: Drawing mode's raw stroke buffer
            (drawing.canvas, BGR numpy array) and the camera frame's own
            (width, height) it was drawn at - see _draw_drawing_canvas.
            Pass None (the default) for every other mode.
          game_state: Whack-a-Mole's live state as a plain dict, already
            scaled to screen coordinates by the caller - see
            _draw_whackamole for the exact keys expected. Pass None (the
            default) for every other mode.
          extras: optional dict of video-call / help-system pieces:
            "tutorial" (tutorial.Tutorial.view() + "hover"), "call" (call bar:
            buttons/hover/status/ok), "toast" (short message), "ring" (0..1
            hold-to-confirm progress drawn around the reticle).
          reticle: draw the always-on pointer at hand_point (see _reticle).
            showcase.py turns this off only where another window already
            draws its own cursor (AR mode, Desktop mode with control on).
          ink/paint: Drawing mode's tiled ink canvas and toolbar view (see
            overlay_drawing.py) - drawn natively on this window.
          hand_point: where the tracked hand currently maps to on screen
            (any mode, not just the ones with their own cursor_point
            above) - lets the status/help drawers expand by POINTING at
            them, not only by hovering an actual mouse there. Without
            this, a hands-only session had no way to ever see the
            collapsed help panel's contents - defeating the point of a
            hands-driven overlay. See _pointer_or_hand below.
        """
        if self._closed:
            return
        self.show()
        c = self.canvas
        c.delete("all")

        mode = modes[mode_key]
        color = _bgr_to_hex(mode["color"])

        if menu_open:
            menu_colors = {k: {"name": v["name"], "color": _bgr_to_hex(v["color"])}
                           for k, v in modes.items()}
            self._mode_menu(mode_order, menu_colors, menu_highlight)
            if extras and extras.get("tutorial"):
                self._draw_tutorial(extras["tutorial"])
            if reticle and hand_point is not None:
                self._reticle(hand_point, pinching, color)
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                self._closed = True
            return

        now = time.time()
        dt = 0.0 if self._last_update_t is None else max(0.0, min(0.25, now - self._last_update_t))
        self._last_update_t = now

        pointer = self._pointer()

        def zone(tab, panel, openness):
            """What counts as "still pointing at this drawer": just the tab
            while collapsed, but the WHOLE grown panel (plus a margin) once
            it's opening - so you can reach the buttons inside it."""
            if openness > 0.1:
                return (min(tab[0], panel[0]) - 14, min(tab[1], panel[1]) - 14,
                        max(tab[2], panel[2]) + 14, max(tab[3], panel[3]) + 14)
            return (tab[0] - 6, tab[1] - 6, tab[2] + 6, tab[3] + 6)

        def hovering(z):
            return self._inside(pointer, *z) or self._inside(hand_point, *z)

        self._status_hover = None
        for _k, _r in self._status_btn_rects.items():
            if self._inside(hand_point, *_r) or self._inside(pointer, *_r):
                self._status_hover = _k
        s_zone = zone(self._status_tab_rect(), self._status_panel_rect(), self._status_open)
        if hovering(s_zone):
            self._status_hold_until = now + self.HOVER_GRACE_S
        self._status_expanded = now < self._status_hold_until

        h_zone = zone(self._help_tab_rect(), self._help_panel_rect(mode["name"], mode["help"]),
                      self._help_open)
        if show_help and hovering(h_zone):
            self._help_hold_until = now + self.HOVER_GRACE_S
        self._help_expanded = bool(show_help) and now < self._help_hold_until
        self._status_open = self._ease_toward(self._status_open, 1.0 if self._status_expanded else 0.0, dt)
        self._help_open = self._ease_toward(self._help_open, 1.0 if self._help_expanded else 0.0, dt)

        # Mode content (Drawing's strokes / Whack-a-Mole's board) is the
        # background layer here - drawn before the chrome below so the
        # status/help drawers and corner brackets stay on top of it
        # rather than getting painted over by a full-screen image.
        if drawing_canvas is not None:
            self._draw_drawing_canvas(drawing_canvas, frame_size)
        if game_state is not None:
            self._draw_whackamole(game_state)
        if ink is not None:
            self._draw_ink(ink)

        self._status_bar(mode["name"], color, enabled, gesture, confidence, fps, ai_label,
                          openness=self._status_open,
                          camera_on=(extras or {}).get("camera_on", True))
        if edge_info is not None:
            edge_ratio, active_zone, progress = edge_info
            self._edge_zones(edge_ratio, active_zone, progress, color)
        self._cursor_dot(cursor_point, pinching, color)
        if flash is not None:
            text, flash_color_bgr = flash
            self._flash_text(text, _bgr_to_hex(flash_color_bgr))
        if show_help:
            self._help_panel(mode["name"], color, mode["help"], openness=self._help_open)
        elif help_hidden_note:
            tx1, ty1, tx2, ty2 = self._help_tab_rect()
            c.create_text(tx1, ty1 - 10, text=help_hidden_note, fill="#8c9cff",
                           font=(UI_FONT, 9), anchor="sw")
        if paint is not None:
            self._draw_paint_dock(paint)
        ex = extras or {}
        if ex.get("call"):
            self._draw_call_bar(ex["call"])
        if ex.get("tutorial"):
            self._draw_tutorial(ex["tutorial"])
        if ex.get("banner"):
            self._draw_banner(ex["banner"])
        if ex.get("toast"):
            self._draw_toast(ex["toast"])
        if reticle and hand_point is not None:
            if paint is not None:
                acc = "#%02x%02x%02x" % paint["color"] if paint["tool"] == "pen" else "#ffffff"
                self._reticle(hand_point, pinching, acc, radius=max(8, paint["cursor_radius"]))
            else:
                self._reticle(hand_point, pinching, color, ring=ex.get("ring"))

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True


def _bgr_to_hex(bgr_color):
    b, g, r = bgr_color
    return "#%02x%02x%02x" % (r, g, b)


def _dim_hex_from_rgb(hex_color, factor):
    r = int(hex_color[1:3], 16) * factor
    g = int(hex_color[3:5], 16) * factor
    b = int(hex_color[5:7], 16) * factor
    return "#%02x%02x%02x" % (int(r), int(g), int(b))
