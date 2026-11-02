"""
ar_overlay.py
--------------
AR mode: floating charts you can conjure, drag, and dismiss with your
hand, drawn on a full-screen transparent window that sits on top of the
whole desktop - not just inside the webcam preview window - so it reads
as "stuff floating over your screen" rather than "stuff floating inside
an app".

Built with the exact same technique as overlay.py's CursorReticle /
LaserPointer (Tkinter's Windows-only '-transparentcolor' attribute for a
borderless, click-through, always-on-top window) - just scaled up from a
30px dot to the whole screen, with several movable shapes drawn on it
instead of one fixed one.

IMPORTANT - read before relying on this:
  - '-transparentcolor' is WINDOWS-ONLY, same limitation as the rest of
    this project's overlay windows. On any other platform this class
    falls back to a small, ORDINARY (opaque, movable, corner-anchored)
    window instead of a full-screen one - see the safety note in
    AROverlay.__init__. A full-screen window that failed to go
    transparent would otherwise block the entire desktop with no
    obvious way to close it, which is a much worse failure than a
    small window in the corner.
  - Only the actual drawn shapes (bars, outlines, text) are opaque/
    click-through-BLOCKING pixels - everywhere else on the transparent
    canvas is fully click-through, so normal desktop use elsewhere on
    screen is unaffected.
  - This has not been run against a real display - it follows the same
    pattern already proven elsewhere in this project, scaled up, but a
    live test pass on your actual machine is the next step, not a
    formality. Performance of pumping several independent Tkinter roots
    (Overlay, CursorReticle, LaserPointer, AROverlay, CameraPreview) once
    per camera frame is untested; if it feels sluggish, the first thing
    to try is consolidating them into one tk.Tk() root with Toplevel()
    children instead of each calling tk.Tk() independently.
"""
import csv
import json
import os
import random
import time
import tkinter as tk

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except Exception:
    _PIL_OK = False

import numpy as np

try:
    import cv2 as _cv2
except Exception:  # pragma: no cover
    _cv2 = None


def frame_to_preview_rgb(bgr, width, height):
    """BGR camera frame -> small RGB uint8 array for the thumbnail. Shrinks
    with OpenCV's INTER_AREA FIRST (area-averaging a 1280x720 frame down to
    ~260 px is ~10x cheaper than PIL's default bicubic resize of the full
    frame, which was ~13 ms per frame) and only then swaps channels on the
    tiny image."""
    if _cv2 is not None:
        small = _cv2.resize(bgr, (width, height), interpolation=_cv2.INTER_AREA)
        return _cv2.cvtColor(small, _cv2.COLOR_BGR2RGB)
    return np.asarray(Image.fromarray(bgr[:, :, ::-1]).resize((width, height)))


# 3D hologram objects (bars / surface / shapes). Optional: if numpy or the
# module is unavailable, AR mode simply runs without the "3D" palette button.
try:
    from ar3d import SHAPE_KINDS, Scene3D
    _AR3D_OK = True
except Exception as _e:  # pragma: no cover
    print(f"[import] AR 3D objects unavailable: {_e}")
    _AR3D_OK = False

# Typed math-expression -> 3D surface ("Functions" palette button).
# Needs ar3d (it places a Surface3D), so gated on that too - no separate
# feature flag, just "AR 3D is available or it isn't".
try:
    import function_graphs
    _FUNC_GRAPHS_OK = _AR3D_OK
except Exception as _e:  # pragma: no cover
    print(f"[import] AR function graphs unavailable: {_e}")
    _FUNC_GRAPHS_OK = False

_TRANSPARENT_KEY = "magenta"

_CHART_W, _CHART_H = 260, 170
_BAR_COLORS = ("#39e6ff", "#39ff88", "#ffb84d", "#ff5d8f", "#c58cff")

# Where CSV files for the "Files" palette button are read from, and
# where the workspace (chart layout, so it survives closing the app) is
# saved - both next to this module, not the current working directory,
# so they're found the same way regardless of where showcase.py is run
# from.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(_MODULE_DIR, "ar_data")
WORKSPACE_FILE = os.path.join(_MODULE_DIR, "ar_workspace.json")
WORKSPACE_SAVE_INTERVAL = 2.0  # seconds - throttles disk writes while dragging


_CSV_MAX_BYTES = 5 * 1024 * 1024   # refuse anything bigger - load runs on the gesture loop
_CSV_MAX_ROWS = 20                 # more bars than this can't be read in a 260px panel anyway
_CSV_DELIMITERS = (",", ";", "\t", "|")
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # latin-1 never fails, so this always ends


def _parse_number(text, decimal_comma_hint=False):
    """Forgiving number parser for values people paste out of
    spreadsheets: "1200", "1,200" (US thousands), "1.200,50" (EU),
    "1,5" (decimal comma), "$3.50", "45%", stray spaces / non-breaking
    spaces. Returns a finite float, or None if it isn't a number.

    decimal_comma_hint: True when the file is semicolon-delimited (the
    usual sign of a European-locale export) - there, a lone comma is a
    DECIMAL separator ("1,5" = 1.5), not a thousands separator. In a
    comma-delimited file a comma inside a number can only be a thousands
    separator (otherwise it'd have split the field), so "1,200" = 1200.
    """
    s = text.strip().replace("\u00a0", "").replace(" ", "")
    s = s.lstrip("$€£¥").rstrip("%")
    if not s:
        return None
    if "," in s and "." in s:
        # Both present: whichever comes LAST is the decimal separator.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        looks_thousands = (
            not decimal_comma_hint
            and all(len(part) == 3 for part in s.lstrip("-+").split(",")[1:])
        )
        s = s.replace(",", "") if looks_thousands else s.replace(",", ".")
    try:
        value = float(s)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None  # "nan"/"inf" parse as floats but would poison normalization
    return value


def _read_text(path):
    """Reads a whole file as text, trying encodings in order. Excel on
    Windows writes "CSV" in the local ANSI code page (cp1252) rather
    than UTF-8 whenever a file has accented characters, so UTF-8 alone
    fails on exactly the files a Windows user is most likely to have."""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")  # unreachable in practice


def _pick_delimiter(lines):
    """Chooses among , ; tab | by parsing a sample of lines with each and
    scoring: every line must split into the same number (>=2) of fields
    (a real delimiter is consistent; a wrong one isn't), then prefer the
    one whose second column actually looks numeric, then the one that
    leaves no OTHER candidate delimiter stuck inside a label (that's the
    tell for "1,5" in a semicolon file being wrongly split on the comma).
    Ties fall back to the order of _CSV_DELIMITERS (comma first)."""
    sample = lines[:12]
    best, best_score = ",", None
    for order, delim in enumerate(_CSV_DELIMITERS):
        try:
            rows = list(csv.reader(sample, delimiter=delim))
        except csv.Error:
            continue
        rows = [r for r in rows if r and any(c.strip() for c in r)]
        if not rows:
            continue
        counts = {len(r) for r in rows}
        consistent = len(counts) == 1 and next(iter(counts)) >= 2
        if not consistent:
            continue
        hint = delim == ";"
        numeric = sum(1 for r in rows if _parse_number(r[1], hint) is not None)
        others = [d for d in _CSV_DELIMITERS if d != delim]
        stuck = sum(1 for r in rows if any(o in r[0] for o in others))
        score = (numeric, -stuck, -order)
        if best_score is None or score > best_score:
            best, best_score = delim, score
    return best


def load_csv(path):
    """Reads a simple two-column (label, value) CSV and returns
    (labels, values) as parallel lists. Raises ValueError with a
    readable message for anything that can't be used - callers catch
    that (plus OSError for a missing/locked file) and print a warning
    instead of letting one bad file take the whole app down.

    Built to survive real-world spreadsheet exports, not just tidy test
    files: comma / semicolon / tab / pipe delimiters (auto-detected),
    UTF-8 (with or without BOM) or Windows cp1252 encoding, optional
    header row, blank lines, extra columns (ignored), rows with a
    non-numeric value (skipped, not fatal), thousands separators and
    decimal commas, and quoted labels containing the delimiter. Files
    over _CSV_MAX_BYTES are refused and only the first _CSV_MAX_ROWS
    rows are used (more bars can't be read in the chart panel anyway).
    Only the first two columns are used - richer formats (multiple
    series, XML, Excel) are deliberately later work.
    """
    if os.path.getsize(path) > _CSV_MAX_BYTES:
        raise ValueError(f"{os.path.basename(path)}: file too large "
                         f"(> {_CSV_MAX_BYTES // (1024 * 1024)} MB)")
    text = _read_text(path).replace("\x00", "")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{os.path.basename(path)}: file is empty")

    delim = _pick_delimiter(lines)
    hint = delim == ";"
    try:
        rows = [r for r in csv.reader(lines, delimiter=delim)
                if r and any(c.strip() for c in r)]
    except csv.Error as e:  # csv.Error is NOT a ValueError - convert it
        raise ValueError(f"{os.path.basename(path)}: unreadable CSV ({e})")

    # Header row: present if the first row's value column isn't a number.
    start = 0
    if len(rows[0]) >= 2 and _parse_number(rows[0][1], hint) is None:
        start = 1

    labels, values = [], []
    for row in rows[start:]:
        if len(row) < 2:
            continue
        value = _parse_number(row[1], hint)
        if value is None:
            continue  # skip a malformed row rather than fail the whole file
        labels.append(row[0].strip() or f"#{len(labels) + 1}")
        values.append(value)
    if not values:
        raise ValueError(f"{os.path.basename(path)}: no valid (label, value) "
                         "rows - need a label column and a numeric value column")
    if len(values) > _CSV_MAX_ROWS:
        print(f"[INFO] AR: {os.path.basename(path)} has {len(values)} rows - "
              f"showing the first {_CSV_MAX_ROWS}")
        labels, values = labels[:_CSV_MAX_ROWS], values[:_CSV_MAX_ROWS]
    return labels, values


def list_csv_files(folder):
    """CSV files in `folder`, case-insensitively (a plain "*.csv" glob
    misses "DATA.CSV" on case-sensitive filesystems), skipping hidden
    files and Office's "~$" temp/lock files. Sorted by name."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(
        os.path.join(folder, n) for n in names
        if n.lower().endswith(".csv") and not n.startswith((".", "~$"))
        and os.path.isfile(os.path.join(folder, n))
    )


def _ensure_sample_csv():
    """Writes one example CSV into DATA_FOLDER so the Files palette button
    has something to show immediately instead of an empty folder with no
    explanation. Only called on a true first run (see AROverlay.__init__),
    and never overwrites an existing file."""
    try:
        sample_path = os.path.join(DATA_FOLDER, "sample_sales.csv")
        if os.path.exists(sample_path):
            return
        with open(sample_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "value"])
            for label, value in (("Q1", 120), ("Q2", 95), ("Q3", 160), ("Q4", 140)):
                w.writerow([label, value])
    except OSError:
        pass  # non-fatal - Files browser will just show "no files found"


class ChartWidget:
    """One floating chart's data + position. Position is its CENTER, in
    real screen pixel coordinates (same space as desktop_control.
    map_to_screen's output) - not normalized camera coordinates.

    Two data fields, not one: `data` is the current TARGET values
    (snapped instantly when they change - a randomize(), a live-feed
    update, a file reload); `display_data` is what's actually drawn,
    eased toward `data` a little each frame (see step_toward_target())
    so a chart's numbers visibly morph into new values instead of
    jump-cutting - part of what was making this feel inert rather than
    alive.

    live_source, when set ("fps" or "gesture_mix"), marks this as a
    LIVE chart: showcase.py's update_live_data() drives its `data` every
    frame from real, currently-true numbers instead of random mock
    values - two_finger cycles which live metric it's showing.

    file_source, when set (a path under DATA_FOLDER), marks this as a
    FILE chart: `data` came from a real CSV via load_csv(), `labels` and
    raw_values hold the real category names and un-normalized numbers
    (data itself is normalized 0-1 for drawing, same as every other
    chart) - two_finger RE-READS the file instead of randomizing or
    cycling, so editing the CSV and refreshing shows the new numbers.
    """

    _next_id = 1
    LIVE_SOURCES = ("fps", "gesture_mix")

    def __init__(self, x, y, kind="bar", live_source=None, file_source=None,
                 label=None, load_data=True):
        self.id = ChartWidget._next_id
        ChartWidget._next_id += 1
        self.x, self.y = x, y
        self.w, self.h = _CHART_W, _CHART_H
        self.kind = kind
        self.live_source = live_source
        self.file_source = file_source
        self.labels = None       # real category names - FILE charts only
        self.raw_values = None   # real un-normalized numbers - FILE charts only
        self.color = _BAR_COLORS[(self.id - 1) % len(_BAR_COLORS)]
        self.data = []
        self.display_data = []
        self.vx, self.vy = 0.0, 0.0
        self._last_move_time = None
        if live_source:
            self.label = label or f"LIVE: {live_source.upper()}"
        elif file_source:
            self.label = label or os.path.splitext(os.path.basename(file_source))[0].upper()
            if load_data:
                self.reload_from_file()
        else:
            self.label = label or f"DATA {self.id}"
            if load_data:
                self.randomize()

    @property
    def is_live(self):
        return self.live_source is not None

    @property
    def is_file_backed(self):
        return self.file_source is not None

    def cycle_live_source(self):
        """Two-finger on a live chart moves to the next metric instead
        of re-randomizing (there's nothing to randomize - the data is
        real). Line <-> pie kind follows the metric, since gesture_mix
        reads as proportions (pie) and fps reads as a trend (line)."""
        if not self.is_live:
            return
        i = self.LIVE_SOURCES.index(self.live_source)
        self.live_source = self.LIVE_SOURCES[(i + 1) % len(self.LIVE_SOURCES)]
        self.kind = "pie" if self.live_source == "gesture_mix" else "line"
        self.label = f"LIVE: {self.live_source.upper()}"

    def set_live_data(self, values):
        """Called every frame (from AROverlay.update_live_data) for a
        live chart - sets the new TARGET values; step_toward_target()
        eases display_data toward them. A plain list copy, not a
        randomize() - the data isn't made up."""
        self.data = list(values)

    def reload_from_file(self):
        """(Re-)reads file_source from disk. Raises on a missing/broken
        file - callers (both initial creation and the two-finger reload)
        catch this and print a warning rather than let the whole app go
        down over one bad CSV. Values are normalized against their own
        max for display (a chart panel is a fixed size; the real numbers
        are kept in raw_values for the on-chart value labels)."""
        labels, raw_values = load_csv(self.file_source)
        # Normalize by the largest MAGNITUDE (not the largest value):
        # with negatives present, max() could be tiny or zero. A
        # negative value still draws as a zero-height bar (there's no
        # below-the-axis rendering yet) but its true number is still
        # shown as the bar's value label, so it isn't silently lost.
        peak = max(abs(v) for v in raw_values) or 1.0
        self.labels = labels
        self.raw_values = raw_values
        self.data = [max(0.0, v) / peak for v in raw_values]

    def randomize(self):
        if self.kind == "pie":
            n = random.randint(3, 5)
            raw = [random.uniform(0.2, 1.0) for _ in range(n)]
            total = sum(raw)
            self.data = [v / total for v in raw]
        else:
            n = 6 if self.kind == "bar" else 10
            self.data = [random.uniform(0.15, 1.0) for _ in range(n)]

    def step_toward_target(self, alpha=0.25):
        """Eases display_data toward data by `alpha` of the remaining
        distance each call - a chart's bars/line/pie visibly morph
        into new values instead of snapping. If the length changed
        (e.g. a pie chart's slice count, or the very first frame),
        there's nothing sensible to interpolate FROM, so it snaps just
        that once instead of drawing garbage."""
        if len(self.display_data) != len(self.data):
            self.display_data = list(self.data)
            return
        self.display_data = [
            d + (t - d) * alpha for d, t in zip(self.display_data, self.data)
        ]

    def contains(self, px, py):
        return (self.x - self.w / 2 <= px <= self.x + self.w / 2
                and self.y - self.h / 2 <= py <= self.y + self.h / 2)

    def bounds(self):
        return (self.x - self.w / 2, self.y - self.h / 2,
                self.x + self.w / 2, self.y + self.h / 2)

    # -- persistence (see AROverlay._save_workspace/_load_workspace) ---
    def to_dict(self):
        """Everything needed to recreate this chart next launch. Manual
        charts save their actual `data` (so specific numbers survive,
        not just "a bar chart existed"); live charts save nothing but
        the metric (they're re-fed real numbers on the next update
        anyway); file charts save the path (reloaded fresh from disk on
        restore, since the file may have changed since last save)."""
        d = {"x": self.x, "y": self.y, "kind": self.kind, "label": self.label}
        if self.is_live:
            d["live_source"] = self.live_source
        elif self.is_file_backed:
            d["file_source"] = self.file_source
        else:
            d["data"] = self.data
        return d

    @classmethod
    def from_dict(cls, d):
        """Returns a ChartWidget, or None if it couldn't be restored
        (e.g. a file chart whose source CSV has since been deleted) -
        callers should skip a None rather than crash the whole
        workspace load over one missing file."""
        kind = d.get("kind", "bar")
        x, y = d.get("x", 400), d.get("y", 300)
        label = d.get("label")
        if d.get("live_source"):
            return cls(x, y, kind=kind, live_source=d["live_source"], label=label)
        if d.get("file_source"):
            try:
                chart = cls(x, y, kind=kind, file_source=d["file_source"], label=label)
                chart.display_data = list(chart.data)
                return chart
            except (OSError, ValueError) as e:
                print(f"[WARN] AR workspace: couldn't restore chart from "
                      f"{d['file_source']} ({e}) - skipping it")
                return None
        chart = cls(x, y, kind=kind, label=label, load_data=False)
        chart.data = d.get("data", [])
        chart.display_data = list(chart.data)
        return chart


class AROverlay:
    """Full-screen transparent HUD that draws and manages ChartWidgets,
    plus a fixed chart-type PALETTE (a little toolbar of buttons) you
    pick a chart type from before placing it - rather than a pinch on
    empty canvas conjuring one directly, which was hard to land on a
    specific type and felt like it was firing by accident.

    Typical per-frame usage from the main loop:
        hover_kind = ar_overlay.palette_kind_at(screen_pt)   # for highlight
        # on pinch-down:
        kind = ar_overlay.palette_kind_at(screen_pt)
        if kind:
            ar_overlay.start_ghost(kind, screen_pt)          # pick up
        else:
            chart = ar_overlay.chart_at(screen_pt)           # grab existing
        # while still pinching:
        ar_overlay.move_ghost(screen_pt)                     # or move_chart
        # on pinch-release:
        if ar_overlay.in_palette_zone(screen_pt):
            ar_overlay.cancel_ghost()
        else:
            ar_overlay.commit_ghost()                        # drop = create
        ar_overlay.remove_chart(chart)                        # dismiss
        ar_overlay.refresh_chart_data(chart)                  # "live update"
        ar_overlay.update(hover_kind=hover_kind)               # redraw + show
        ar_overlay.hide() / .show() / .toggle()
    """

    PALETTE_KINDS = ("bar", "line", "pie", "live", "files", "shapes3d", "functions", "dense", "clear")
    PALETTE_LABELS = {"shapes3d": "3D", "functions": "\u0192(x)", "dense": "DENSE", "clear": "CLEAR"}

    def __init__(self, screen_size=None):
        self.charts = []
        self.ghost = None  # a ChartWidget being dragged from the palette, not yet placed
        self._closed = False
        self.hidden = True
        self.fullscreen_transparent = False
        self._last_update_time = None
        self.files_open = False
        self._file_tiles = []
        self.shapes_open = False
        self._shape_tiles = []
        self.functions_open = False
        self._function_tiles = []
        self._clear_armed_until = 0.0
        self._tab_hot = False
        self.scene3d = None
        self._dirty = False
        self._last_save_time = 0.0
        self._loading = True  # suppress dirty-marking while restoring on startup
        # Saved chart entries that couldn't be restored THIS launch because
        # their CSV exists but was unreadable (locked / mid-edit / briefly
        # malformed) - kept and re-saved untouched so the chart comes back
        # next launch once the file is fine, instead of an autosave quietly
        # deleting it from the layout forever.
        self._held_entries = []

        # Seed the sample CSV only on a TRUE first run (the data folder
        # didn't exist yet). Not "whenever the folder is empty" - if you
        # delete your files on purpose, they should stay deleted.
        first_run = not os.path.isdir(DATA_FOLDER)
        os.makedirs(DATA_FOLDER, exist_ok=True)
        if first_run:
            _ensure_sample_csv()
        if _FUNC_GRAPHS_OK and not os.path.isdir(function_graphs.FUNC_FOLDER):
            function_graphs.ensure_sample_functions()

        self.root = tk.Tk()
        self.root.title("AR Overlay")
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
            self._bounds_w, self._bounds_h = screen_w, screen_h
        else:
            # SAFETY FALLBACK: never size an opaque window to the full
            # screen - that would block the entire desktop with no
            # title bar to close it. A modest corner window at least
            # keeps the app usable (with a visible warning) on a
            # platform where transparentcolor isn't supported.
            print("[WARN] AR overlay: '-transparentcolor' not supported on "
                  "this platform - falling back to a small opaque window "
                  "instead of a full-screen transparent one. AR mode's "
                  "\"floating over the desktop\" effect needs Windows.")
            self.root.geometry("420x300+40+40")
            bg = "#141414"
            self._bounds_w, self._bounds_h = 420, 300

        self.canvas = tk.Canvas(self.root, bg=bg, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.withdraw()

        if _AR3D_OK:
            self.scene3d = Scene3D((self._bounds_w, self._bounds_h))
        self._build_palette()
        self._load_workspace()
        self._loading = False

    PALETTE_COLS = 3
    TAB_W, TAB_H = 124, 44

    def _build_palette(self):
        """Toolbox geometry, top-left of the overlay (screen pixels in
        fullscreen-transparent mode; the fallback window's own canvas
        pixels otherwise).

        Used to be one long row of nine buttons across the top of the
        screen, permanently - the whole top-left of the desktop was
        covered whenever AR mode was on. Now there are two shapes:
          tab_rect    the small "+ ADD" tab that is all you see normally
          panel_rect  a compact 3x3 box the tab grows into when you point
                      at it (see _update_palette_state)
        self.palette holds each button's rect IN the expanded box; they are
        only live while the box is actually open (palette_kind_at)."""
        btn_w, btn_h, gap, margin, pad, head = 92, 70, 12, 24, 16, 38
        cols = self.PALETTE_COLS
        kinds = [k for k in self.PALETTE_KINDS
                 if (k != "shapes3d" or self.scene3d is not None)
                 and (k != "functions" or (self.scene3d is not None and _FUNC_GRAPHS_OK))
                 and (k != "dense" or self.scene3d is not None)]
        rows = -(-len(kinds) // cols)
        self.palette = []
        for i, kind in enumerate(kinds):
            r, c = divmod(i, cols)
            x0 = margin + pad + c * (btn_w + gap)
            y0 = margin + head + r * (btn_h + gap)
            self.palette.append({"kind": kind, "x0": x0, "y0": y0, "x1": x0 + btn_w, "y1": y0 + btn_h})
        self.tab_rect = (margin, margin, margin + self.TAB_W, margin + self.TAB_H)
        self.panel_rect = (margin, margin,
                           margin + 2 * pad + cols * btn_w + (cols - 1) * gap,
                           margin + head + rows * btn_h + (rows - 1) * gap + pad)
        # Padded: "drop it back here to cancel" / "still hovering the
        # box" are a bit more forgiving than the exact pixels.
        px0, py0, px1, py1 = self.panel_rect
        self._palette_zone = (0, 0, px1 + margin, py1 + margin)
        self._tab_zone = (0, 0, self.tab_rect[2] + margin, self.tab_rect[3] + margin)
        self._pal_open = 0.0              # eased 0 (just the tab) .. 1 (box fully open)
        self._pal_hold_until = 0.0
        self._pal_last_t = None

    def browser_origin(self):
        """Top-left for the file/shape/function tile rows: to the RIGHT of
        the box, so they neither sit on top of it nor depend on whether
        it's currently open or collapsed."""
        return self.panel_rect[2] + 16, self.tab_rect[1] + 8

    def palette_visible(self):
        return self._pal_open > 0.55

    def set_palette_open(self, is_open):
        """Force the box open/closed immediately (no easing) - used by
        tests, and available for anything that needs a known state."""
        self._pal_open = 1.0 if is_open else 0.0
        self._pal_hold_until = time.time() + 5.0 if is_open else 0.0

    def _update_palette_state(self, cursor_point, now):
        """Opens the box while the pointer is on the tab or inside the
        open box, and for a short grace period (so a wobble doesn't snap
        it shut mid-reach); also stays open while a pick-up is in
        progress (the box is where you drop things to cancel, so it has
        to be visible) and while CLEAR is armed (so "SURE?" can be seen)."""
        dt = 0.0 if self._pal_last_t is None else max(0.0, min(0.25, now - self._pal_last_t))
        self._pal_last_t = now
        hovering = False
        if cursor_point is not None:
            zone = self._palette_zone if self._pal_open > 0.1 else self._tab_zone
            hovering = (zone[0] <= cursor_point[0] <= zone[2] and zone[1] <= cursor_point[1] <= zone[3])
        holding = (self.ghost is not None or (self.scene3d is not None and self.scene3d.ghost is not None)
                   or now < self._clear_armed_until)
        if hovering or holding:
            self._pal_hold_until = now + 0.7
        target = 1.0 if (hovering or holding or now < self._pal_hold_until) else 0.0
        self._pal_open += (target - self._pal_open) * min(1.0, dt * 11.0)
        if abs(target - self._pal_open) < 0.01:
            self._pal_open = target

    # -- lifecycle -----------------------------------------------------
    def show(self):
        if self.hidden and not self._closed:
            self.root.deiconify()
            self.hidden = False

    def hide(self):
        if not self.hidden and not self._closed:
            self.root.withdraw()
            self.hidden = True
            # Flush any unsaved changes NOW: autosave only runs while
            # update() is being called (i.e. while in AR mode), so a change
            # made just before switching modes would otherwise sit unsaved
            # until you came back - and be lost if the app died meanwhile.
            if self._dirty:
                self._save_workspace()
                self._dirty = False
                self._last_save_time = time.time()

    def toggle(self):
        if self.hidden:
            self.show()
        else:
            self.hide()

    # -- chart creation via the palette (pick up -> drag -> drop) ------
    def palette_kind_at(self, screen_point):
        """Which palette button (if any) the point is over - a chart
        kind string, or None. Check this every frame (not just on
        pinch) so the caller can highlight the hovered button even
        before picking it up."""
        if not self.palette_visible():
            return None            # collapsed: the buttons aren't on screen, so they can't be pressed
        px, py = screen_point
        for b in self.palette:
            if b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]:
                return b["kind"]
        return None

    def in_palette_zone(self, screen_point):
        """True if the point is over the toolbox (the whole box while it's
        open, just the tab while collapsed) - used to mean "dropped it
        back where it came from = cancel" when releasing a ghost chart."""
        x0, y0, x1, y1 = self._palette_zone if self._pal_open > 0.1 else self._tab_zone
        px, py = screen_point
        return x0 <= px <= x1 and y0 <= py <= y1

    def _clamp_position(self, x, y):
        """Keep a chart's center within the overlay's actual visible
        bounds - the whole screen in fullscreen-transparent mode, the
        small fallback window's own local canvas otherwise. Used for
        every explicit move AND every inertia step, so a flung chart
        can coast to a stop against an edge instead of sailing off it
        into coordinates nothing will ever draw again."""
        w, h = self._bounds_w, self._bounds_h
        x = min(max(x, _CHART_W // 2 + 4), w - _CHART_W // 2 - 4)
        y = min(max(y, _CHART_H // 2 + 4), h - _CHART_H // 2 - 4)
        return x, y

    def start_ghost(self, kind, screen_point):
        """Pick up a new not-yet-placed chart of `kind` at the given
        point - drawn dashed/translucent-looking (see _draw_chart) so
        it reads as "still deciding where to put this" rather than an
        already-placed chart. Replaces any ghost already in progress.

        kind "live" isn't a real rendering kind - it's the palette's
        4th button, meaning "a chart fed by real, currently-true data"
        rather than mock numbers. Translates to a real line chart
        showing live FPS (the default live_source; two_finger cycles
        it once placed). kind "files" isn't handled here at all - it
        opens the file browser instead of picking up a chart (see
        toggle_file_browser / start_ghost_from_file), so callers should
        check for that before calling this."""
        x, y = self._clamp_position(*screen_point)
        if kind == "live":
            self.ghost = ChartWidget(x, y, kind="line", live_source="fps")
        elif kind in ("files", "shapes3d", "functions", "dense", "clear"):
            return None   # browsers/toggles/actions - none of these pick up a chart
        else:
            self.ghost = ChartWidget(x, y, kind=kind)
        self._mark_dirty()
        return self.ghost

    def start_ghost_from_file(self, path, screen_point):
        """Pick up a new chart backed by a specific CSV (one of the
        tiles the file browser shows). Bar only for now - a file's
        categories/values map onto bars far more naturally than a line
        or pie, and keeping this to one shape keeps the first pass at
        file import simpler to get solid. Closes the file browser
        (picking a file IS the point of opening it)."""
        x, y = self._clamp_position(*screen_point)
        self.files_open = False
        try:
            self.ghost = ChartWidget(x, y, kind="bar", file_source=path)
            self._mark_dirty()
            return self.ghost
        except (OSError, ValueError) as e:
            print(f"[WARN] AR: couldn't load {path} ({e})")
            return None

    def move_ghost(self, screen_point):
        if self.ghost is not None:
            self.ghost.x, self.ghost.y = self._clamp_position(*screen_point)

    def commit_ghost(self, max_charts=12):
        """Drop the ghost onto the canvas as a real chart. No-op (returns
        None) if there's no ghost in progress."""
        if self.ghost is None:
            return None
        chart = self.ghost
        self.ghost = None
        self.charts.append(chart)
        # Safety cap - each chart is several Tkinter canvas items redrawn
        # every frame, so an enthusiastic demo shouldn't be able to pile
        # up enough of them to tank frame rate. Evicts the OLDEST chart
        # first, same as a real desktop only has so much room for
        # floating windows.
        while len(self.charts) > max_charts:
            self.charts.pop(0)
        self._mark_dirty()
        return chart

    def cancel_ghost(self):
        """Drop the ghost back at the palette - discarded, nothing is
        created. No-op if there's no ghost in progress."""
        self.ghost = None

    # -- the "Files" palette button: pick a CSV instead of a shape ------
    def toggle_file_browser(self):
        """Opens or closes the second row of file tiles. Re-scans
        DATA_FOLDER every time it opens, so a CSV dropped in mid-demo
        shows up without restarting anything."""
        self.files_open = not self.files_open
        if self.files_open:
            self.shapes_open = False
            self._scan_files()

    def _scan_files(self):
        paths = list_csv_files(DATA_FOLDER)[:6]
        btn_w, btn_h, gap = 100, 40, 10
        margin, row_y = self.browser_origin()
        self._file_tiles = []
        for i, path in enumerate(paths):
            x0 = margin + i * (btn_w + gap)
            name = os.path.splitext(os.path.basename(path))[0]
            self._file_tiles.append({
                "path": path, "name": name,
                "x0": x0, "y0": row_y, "x1": x0 + btn_w, "y1": row_y + btn_h,
            })

    # -- the "3D" palette button: pick a hologram shape --------------------------
    def toggle_shape_browser(self):
        """Opens/closes the row of 3D tiles (bars, surface, cube, sphere,
        torus). Same interaction as the FILES row: pinch a tile, drag, drop."""
        if self.scene3d is None:
            return
        self.shapes_open = not self.shapes_open
        if self.shapes_open:
            self.files_open = False
            btn_w, btn_h, gap = 100, 40, 10
            margin, row_y = self.browser_origin()
            self._shape_tiles = [
                {"kind": k, "name": k.replace("3d", "").upper(),
                 "x0": margin + i * (btn_w + gap), "y0": row_y,
                 "x1": margin + i * (btn_w + gap) + btn_w, "y1": row_y + btn_h}
                for i, k in enumerate(SHAPE_KINDS)]

    def shape_tile_at(self, screen_point):
        if not self.shapes_open:
            return None
        px, py = screen_point
        for t in self._shape_tiles:
            if t["x0"] <= px <= t["x1"] and t["y0"] <= py <= t["y1"]:
                return t["kind"]
        return None

    # -- the "\u0192(x)" palette button: pick a typed function graph ----------
    def toggle_function_browser(self):
        """Opens/closes the row of function tiles - one per .txt file in
        function_graphs.FUNC_FOLDER. Same interaction as FILES/3D: pinch
        a tile, drag, drop. Re-scans (and re-compiles each file) every
        time it opens, so adding/editing a .txt mid-demo shows up
        without restarting anything - same contract as _scan_files."""
        if not _FUNC_GRAPHS_OK:
            return
        self.functions_open = not self.functions_open
        if self.functions_open:
            self.files_open = False
            self.shapes_open = False
            self._scan_functions()

    def _scan_functions(self):
        paths = function_graphs.list_function_files()[:6]
        btn_w, btn_h, gap = 100, 40, 10
        margin, row_y = self.browser_origin()
        self._function_tiles = []
        for i, path in enumerate(paths):
            name = os.path.splitext(os.path.basename(path))[0]
            error = None
            try:
                function_graphs.read_function_file(path)
            except function_graphs.FunctionParseError as e:
                error = str(e)
            self._function_tiles.append({
                "path": path, "name": name, "error": error,
                "x0": margin + i * (btn_w + gap), "y0": row_y,
                "x1": margin + i * (btn_w + gap) + btn_w, "y1": row_y + btn_h,
            })

    def function_tile_at(self, screen_point):
        """Which function tile (if any) the point is over - only
        meaningful while functions_open is True; returns None
        otherwise. Returns the tile dict (not just a path) so the
        caller can check ["error"] before trying to place it."""
        if not self.functions_open:
            return None
        px, py = screen_point
        for tile in self._function_tiles:
            if tile["x0"] <= px <= tile["x1"] and tile["y0"] <= py <= tile["y1"]:
                return tile
        return None

    # NOTE: unlike start_ghost_from_file (2D charts, which this class
    # owns outright), actually BUILDING the Surface3D from a function
    # tile and picking it up as a ghost happens in ar3d_input.py's
    # AR3DController._begin() - right next to the identical shapes3d
    # tile-hit branch it mirrors, since 3D objects' pick-up/drag/drop
    # state machine (mode, gi, ...) lives there, not on this class.
    # function_tile_at() above is this class's half of that: hit-testing
    # + exposing each tile's path/parse-error for the controller to use.

    def file_tile_at(self, screen_point):
        """Which file tile (if any) the point is over - only meaningful
        while files_open is True; returns None otherwise."""
        if not self.files_open:
            return None
        px, py = screen_point
        for tile in self._file_tiles:
            if tile["x0"] <= px <= tile["x1"] and tile["y0"] <= py <= tile["y1"]:
                return tile["path"]
        return None

    # -- existing-chart management --------------------------------------
    CLEAR_CONFIRM_S = 3.0

    def press_clear(self, now=None):
        """Two-step "Clear all graphs": the first pinch ARMS the button
        (it turns red and says SURE?), a second pinch within
        CLEAR_CONFIRM_S actually clears. One stray pinch on a button
        that wipes the whole workspace would be far worse than the
        occasional fist that doesn't register - this is the "I'm
        sure" guard. Returns "armed" or "cleared"."""
        now = time.time() if now is None else now
        if now < self._clear_armed_until:
            self._clear_armed_until = 0.0
            self.clear_all()
            return "cleared"
        self._clear_armed_until = now + self.CLEAR_CONFIRM_S
        return "armed"

    def clear_all(self):
        """Remove every 2D chart and 3D object (3D ones go through
        Scene3D.remove, so the last few stay recoverable via undo) and
        cancel any pick-up in progress."""
        self.charts.clear()
        self.ghost = None
        self.files_open = self.shapes_open = self.functions_open = False
        if self.scene3d is not None:
            for o in list(self.scene3d.objects):
                self.scene3d.remove(o)
            self.scene3d.ghost = None
        self._mark_dirty()

    def remove_chart(self, chart):
        if chart in self.charts:
            self.charts.remove(chart)
            self._mark_dirty()

    def chart_at(self, screen_point):
        """Topmost (most-recently-added) chart under the point, or
        None. Iterates newest-first so overlapping charts grab the one
        that's visually on top."""
        px, py = screen_point
        for chart in reversed(self.charts):
            if chart.contains(px, py):
                return chart
        return None

    def move_chart(self, chart, screen_point):
        """Explicit drag (pinch held on a chart). Also records velocity
        (position delta / time delta) on the chart so that when it's
        released, _step_physics() can carry that motion forward for a
        few frames as decaying inertia - a dropped chart glides to a
        stop instead of dead-stopping the instant the pinch releases."""
        new_x, new_y = self._clamp_position(*screen_point)
        now = time.time()
        last_t = chart._last_move_time
        if last_t is not None:
            # Capped, not just floored: if this chart's last recorded
            # move was a long time ago (picked up again after sitting
            # still for a while), an uncapped dt could produce one wildly
            # oversized velocity sample - capping it means the worst
            # case is a single frame of somewhat-too-fast motion, which
            # decay kills within a few frames anyway, rather than a
            # chart rocketing off toward a screen edge.
            dt = min(max(1e-3, now - last_t), 0.2)
            chart.vx = (new_x - chart.x) / dt
            chart.vy = (new_y - chart.y) / dt
        chart.x, chart.y = new_x, new_y
        chart._last_move_time = now
        self._mark_dirty()

    def _step_physics(self, grabbed_chart, dt):
        """Coasts every chart that ISN'T currently being dragged along
        by whatever velocity move_chart() last recorded for it, decaying
        that velocity each frame until it settles - the "drop a chart
        and it glides a little" inertia feel. grabbed_chart is skipped
        entirely (its position is being driven directly by the pinch via
        move_chart(), not by physics)."""
        decay = 0.85
        stop_speed = 6.0  # px/sec - below this, just stop cleanly
        for chart in self.charts:
            if chart is grabbed_chart:
                continue
            if abs(chart.vx) < stop_speed and abs(chart.vy) < stop_speed:
                chart.vx = chart.vy = 0.0
                continue
            chart.x += chart.vx * dt
            chart.y += chart.vy * dt
            chart.x, chart.y = self._clamp_position(chart.x, chart.y)
            chart.vx *= decay
            chart.vy *= decay
            self._mark_dirty()

    def refresh_chart_data(self, chart):
        """Two-finger on a chart: re-randomize mock data, cycle to the
        next metric (Live charts - there's nothing to randomize about
        real data), or re-read from disk (File charts - lets "edit the
        CSV, then two-finger the chart" work as a live-reload)."""
        if chart.is_live:
            chart.cycle_live_source()
        elif chart.is_file_backed:
            try:
                chart.reload_from_file()
            except (OSError, ValueError) as e:
                print(f"[WARN] AR: couldn't reload {chart.file_source} ({e}) "
                      "- keeping the chart's last known data")
        else:
            chart.randomize()
        self._mark_dirty()

    def update_live_data(self, fps_history, gesture_counts):
        """Call once per camera frame (from showcase.py's rolling
        fps_history deque and gesture_counts Counter) - feeds every
        currently-placed Live chart (and the ghost, if it's a Live one
        mid-pick-up) real, current numbers instead of mock data.
        set_live_data() only sets the TARGET; step_toward_target() in
        update() handles the actual on-screen easing."""
        fps_values = None
        gesture_fracs = None
        targets = list(self.charts)
        if self.ghost is not None:
            targets.append(self.ghost)
        for chart in targets:
            if not chart.is_live:
                continue
            if chart.live_source == "fps":
                if fps_values is None:
                    # Normalize against a fixed 60fps scale (clamped) so
                    # the line's height means something consistent frame
                    # to frame, rather than auto-scaling to whatever the
                    # current min/max happen to be.
                    fps_values = [min(1.0, v / 60.0) for v in fps_history] or [0.0]
                chart.set_live_data(fps_values)
            elif chart.live_source == "gesture_mix":
                if gesture_fracs is None:
                    total = sum(gesture_counts.values())
                    if total > 0:
                        gesture_fracs = [v / total for v in gesture_counts.values()]
                    else:
                        gesture_fracs = [1.0]
                chart.set_live_data(gesture_fracs)

    def clear(self):
        self.charts.clear()
        if self.scene3d is not None:
            self.scene3d.clear()
        self._mark_dirty()

    # -- persistence: survive closing and reopening the app -------------
    def _mark_dirty(self):
        if not self._loading:
            self._dirty = True

    def _maybe_autosave(self, now):
        """Throttled - called every update() frame, but only actually
        touches disk if something changed AND it's been a couple of
        seconds since the last write (a chart mid-drag calls
        move_chart() every single frame; writing a file that often
        would be wasteful I/O for no benefit, since only the FINAL
        position when you stop matters)."""
        if self._dirty and (now - self._last_save_time) >= WORKSPACE_SAVE_INTERVAL:
            self._save_workspace()
            self._dirty = False
            self._last_save_time = now

    def _save_workspace(self):
        try:
            payload = {"charts": [c.to_dict() for c in self.charts] + self._held_entries}
            if self.scene3d is not None:
                payload["objects3d"] = self.scene3d.to_list()
            with open(WORKSPACE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
        except OSError as e:
            print(f"[WARN] AR: couldn't save workspace ({e})")

    def _load_workspace(self):
        if not os.path.exists(WORKSPACE_FILE):
            return
        try:
            with open(WORKSPACE_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[WARN] AR: couldn't read saved workspace ({e}) - starting empty")
            return
        if self.scene3d is not None:
            self.scene3d.load_list(payload.get("objects3d"))
        for entry in payload.get("charts", []):
            chart = ChartWidget.from_dict(entry)
            if chart is not None:
                chart.x, chart.y = self._clamp_position(chart.x, chart.y)
                self.charts.append(chart)
            elif os.path.exists(entry.get("file_source") or ""):
                # File is there but unreadable right now - hold the entry
                # for a later launch. (A file that's GONE is dropped: it's
                # not coming back, and holding it would leave an invisible
                # phantom entry in the save file forever.)
                self._held_entries.append(entry)

    # -- drawing ---------------------------------------------------------
    # The whole AR look is deliberately "glass HUD" - outline-only
    # panels and lines with no filled background (see _draw_chart's own
    # comment below), so the real desktop stays visible and the overlay
    # still reads as mostly click-through. The cost of that choice: a
    # bare colored line or bit of text has nothing behind it, so over a
    # light-themed app, a bright photo, a white document, etc. it can
    # all but vanish. A dark halo drawn first, under the real color,
    # fixes that without giving up the glass look - same technique
    # subtitles/game HUDs use to stay readable over any footage, and it
    # doesn't need to know what's actually behind the window (reading
    # real desktop pixels isn't something Tkinter can do cheaply, if at
    # all - this works regardless, which is the "dynamic" part).
    def _halo_line(self, x1, y1, x2, y2, color, width, dash=None):
        c = self.canvas
        c.create_line(x1, y1, x2, y2, fill="#000000", width=width + 3, dash=dash)
        c.create_line(x1, y1, x2, y2, fill=color, width=width, dash=dash)

    def _halo_oval(self, x1, y1, x2, y2, color, width):
        c = self.canvas
        c.create_oval(x1, y1, x2, y2, outline="#000000", width=width + 3)
        c.create_oval(x1, y1, x2, y2, outline=color, width=width)

    def _halo_rect(self, x1, y1, x2, y2, color, width, dash=None):
        c = self.canvas
        c.create_rectangle(x1, y1, x2, y2, outline="#000000", width=width + 3, dash=dash)
        c.create_rectangle(x1, y1, x2, y2, outline=color, width=width, dash=dash)

    @staticmethod
    def _rounded_points(x1, y1, x2, y2, r):
        """Same technique as main_overlay.MainOverlay._rounded_points (a
        smoothed polygon - Tkinter has no native rounded-rect), reused so
        AR's palette/tile browsers share the main HUD's softer-cornered
        look instead of feeling like a differently-styled app."""
        r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        return [
            x1 + r, y1,  x2 - r, y1,  x2, y1,  x2, y1 + r,
            x2, y2 - r,  x2, y2,  x2 - r, y2,  x1 + r, y2,
            x1, y2,  x1, y2 - r,  x1, y1 + r,  x1, y1,
        ]

    def _rounded_rect(self, x1, y1, x2, y2, r, fill=None, outline=None, width=1):
        kwargs = {"smooth": True, "width": width}
        if fill is not None:
            kwargs["fill"] = fill
        if outline is not None:
            kwargs["outline"] = outline
        return self.canvas.create_polygon(*self._rounded_points(x1, y1, x2, y2, r), **kwargs)

    def _halo_rounded_rect(self, x1, y1, x2, y2, r, color, width, fill=None):
        c = self.canvas
        pts = self._rounded_points(x1, y1, x2, y2, r)
        c.create_polygon(*pts, smooth=True, fill=fill or "", outline="#000000", width=width + 3)
        c.create_polygon(*pts, smooth=True, fill="", outline=color, width=width)

    UI_FONT = "Segoe UI"   # Tk falls back to its default family where this isn't installed

    def _grad_rounded(self, x1, y1, x2, y2, r, top, mid, bottom, outline=None, width=1):
        """A rounded rectangle shaded dark-at-the-bottom -> lighter-at-the-
        top, built from three stacked rounded fills (Tkinter canvases have
        no gradient fill). Reads as a soft glassy button/pill instead of a
        flat black slab, and - being OPAQUE - keeps whatever text sits on
        it legible over any desktop background."""
        h = y2 - y1
        for col, frac in ((bottom, 1.0), (mid, 0.66), (top, 0.33)):
            yy2 = y1 + h * frac
            self._rounded_rect(x1, y1, x2, yy2, min(r, (yy2 - y1) / 2), fill=col)
        if outline:
            self._rounded_rect(x1, y1, x2, y2, r, outline=outline, width=width)

    def _label(self, x, y, text, color, size=10, bold=True, anchor="center"):
        """Plain, crisp text for use ON an already-dark surface (button,
        tile, pill) - no halo copies, which is what made text look
        doubled/out-of-focus before."""
        self.canvas.create_text(x, y, text=text, fill=color, anchor=anchor,
                                font=(self.UI_FONT, size, "bold" if bold else "normal"))

    def _halo_text(self, x, y, text, color, font, anchor="center"):
        """Floating text (not on a button): drawn on its own dark gradient
        pill so it reads over ANY background. Name kept from when this
        drew a ring of offset black copies - that look (and its blur on
        bright backgrounds) is gone. `font` is (family, size, weight);
        only size/weight are used, the family is the shared UI font."""
        size = max(10, int(font[1])) if len(font) > 1 else 10
        bold = (len(font) > 2 and font[2] == "bold")
        w = len(text) * size * 0.60 + 20
        h = size * 1.9
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
        cx = (x1 + x2) / 2
        self._label(cx, (y1 + y2) / 2, text, color, size, bold)

    def _draw_chart(self, chart, highlighted=False, ghost=False):
        x0, y0, x1, y1 = chart.bounds()
        c = self.canvas
        outline_w = 3 if highlighted else 2
        glow = "#ffffff" if highlighted else chart.color
        dash = (6, 4) if ghost else None

        # Panel outline only (no filled background) - keeps the window
        # mostly click-through and reads as a glass HUD pane rather than
        # a solid card sitting over the desktop. Ghost (not-yet-placed)
        # charts are drawn dashed instead, so it's obvious at a glance
        # which one isn't real yet. A dark halo keeps this readable over
        # light backgrounds too - see the note above _draw_chart.
        self._halo_rect(x0, y0, x1, y1, glow, outline_w, dash=dash)
        if not ghost:
            # Corner ticks for a sci-fi "bracket" look - skipped for the
            # ghost so it stays visually simpler/lighter while dragging.
            tick = 14
            for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                self._halo_line(cx, cy, cx + dx * tick, cy, glow, outline_w + 1)
                self._halo_line(cx, cy, cx, cy + dy * tick, glow, outline_w + 1)

        self._halo_text((x0 + x1) / 2, y0 + 16, chart.label, glow, ("Consolas", 11, "bold"))

        plot_x0, plot_y0 = x0 + 16, y0 + 34
        # File-backed bar charts get real category labels beneath the
        # bars (rotated vertical - narrow bars, this is the only way
        # several fit without overlapping) - leave extra room for them.
        has_labels = chart.kind == "bar" and chart.labels
        plot_x1, plot_y1 = x1 - 16, y1 - (46 if has_labels else 16)

        if chart.kind == "pie":
            self._draw_pie(chart, plot_x0, plot_y0, plot_x1, plot_y1, ghost)
            return

        c.create_line(plot_x0, plot_y1, plot_x1, plot_y1, fill=chart.color, width=1)

        if chart.kind == "bar" and chart.display_data:
            n = len(chart.display_data)
            gap = 6
            bar_w = max(4, (plot_x1 - plot_x0 - gap * (n - 1)) / n)
            for i, v in enumerate(chart.display_data):
                bx0 = plot_x0 + i * (bar_w + gap)
                bh = max(0.0, v) * (plot_y1 - plot_y0)
                c.create_rectangle(bx0, plot_y1 - bh, bx0 + bar_w, plot_y1,
                                    fill=chart.color, outline="", stipple="gray50" if ghost else "")
                if not ghost and chart.raw_values and i < len(chart.raw_values):
                    # The real number, not just relative bar height -
                    # this is the whole point of a chart backed by an
                    # actual file instead of mock data.
                    val = chart.raw_values[i]
                    val_txt = f"{val:g}"
                    vx, vy = bx0 + bar_w / 2, plot_y1 - bh - 12
                    vw = len(val_txt) * 6.2 + 9
                    self._rounded_rect(vx - vw / 2, vy - 9, vx + vw / 2, vy + 9, 8, fill="#080b10")
                    self._label(vx, vy, val_txt, "#e8fbff", 9)
                if not ghost and chart.labels and i < len(chart.labels):
                    label_txt = chart.labels[i][:10]
                    lx = bx0 + bar_w / 2
                    c.create_rectangle(lx - 8, plot_y1 + 4, lx + 8, plot_y1 + 10 + len(label_txt) * 6.4,
                                       fill="#080b10", outline="")
                    c.create_text(lx, plot_y1 + 8, text=label_txt, fill="#e8fbff",
                                  font=(self.UI_FONT, 9), angle=90, anchor="w")
        elif chart.display_data:
            n = len(chart.display_data)
            step = (plot_x1 - plot_x0) / max(1, n - 1)
            pts = []
            for i, v in enumerate(chart.display_data):
                px = plot_x0 + i * step
                py = plot_y1 - max(0.0, v) * (plot_y1 - plot_y0)
                pts.extend((px, py))
            if len(pts) >= 4:
                c.create_line(*pts, fill=chart.color, width=2, smooth=True, dash=dash)
            if not ghost:
                for i in range(0, len(pts), 2):
                    c.create_oval(pts[i] - 3, pts[i + 1] - 3, pts[i] + 3, pts[i + 1] + 3,
                                   fill=chart.color, outline="")

    def _draw_pie(self, chart, x0, y0, x1, y1, ghost=False):
        c = self.canvas
        size = min(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 4
        r = size / 2
        start = 0.0
        total = sum(chart.display_data) or 1.0
        for i, frac in enumerate(chart.display_data):
            extent = (frac / total) * 360.0
            color = _BAR_COLORS[i % len(_BAR_COLORS)]
            c.create_arc(cx - r, cy - r, cx + r, cy + r,
                         start=start, extent=extent, fill=color,
                         outline=chart.color, width=1,
                         style=tk.PIESLICE, stipple="gray50" if ghost else "")
            start += extent

    def _draw_palette(self, hover_kind, active_kind=None, cursor_approach=0.0):
        """The chart-type toolbar - one button per PALETTE_KINDS entry,
        each drawn as a MINI version of that chart type so it's obvious
        what it makes, not just a text label. hover_kind (whichever
        button the fingertip currently sits over) gets a bright
        highlight; active_kind (mid-pick-up) stays highlighted even if
        the fingertip has since moved off it while dragging the ghost."""
        c = self.canvas
        dense_on = self.scene3d is not None and self.scene3d.dense_mode
        clear_armed = time.time() < self._clear_armed_until
        o = self._pal_open
        tx0, ty0, tx1, ty1 = self.tab_rect
        px0, py0, px1, py1 = self.panel_rect
        # The container grows from the tab's rect to the full box's rect.
        x0c, y0c, x1c, y1c = (tx0 + (px0 - tx0) * o, ty0 + (py0 - ty0) * o,
                              tx1 + (px1 - tx1) * o, ty1 + (py1 - ty1) * o)
        tab_hot = hover_kind is None and o < 0.1 and self._tab_hot
        self._grad_rounded(x0c, y0c, x1c, y1c, 14 + 4 * o, "#1a2330", "#111822", "#0a0e14",
                           outline="#ffffff" if tab_hot else "#39e6ff", width=2)
        if o < 0.45:
            # Collapsed tab: "+ ADD" (plus a small amber dot while DENSE is
            # on, so the mode is still visible with the box closed).
            cx, cy = (tx0 + tx1) / 2 - 8, (ty0 + ty1) / 2
            c.create_line(tx0 + 22, cy, tx0 + 36, cy, fill="#39e6ff", width=3)
            c.create_line(tx0 + 29, cy - 7, tx0 + 29, cy + 7, fill="#39e6ff", width=3)
            self._label(tx0 + 46, cy, "ADD", "#e8fbff", 11, anchor="w")
            if dense_on:
                c.create_oval(tx1 - 24, cy - 5, tx1 - 14, cy + 5, fill="#ffd24d", outline="")
        if not self.palette_visible():
            return
        self._label(px0 + 18, py0 + 19, "ADD TO WORKSPACE", "#7aa6bd", 9, anchor="w")
        for b in self.palette:
            kind = b["kind"]
            x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
            is_hot = (kind == hover_kind or kind == active_kind)
            glow = "#ffffff" if is_hot else "#39e6ff"
            top, mid, bot = "#1b2636", "#111a26", "#0a0f16"
            if kind == "dense" and dense_on:
                # A toggle, not an opener - lit amber is the one button
                # that needs to show its OWN state at a glance.
                glow = "#ffd24d"
                top, mid, bot = "#6b5520", "#4d3d16", "#32270e"
            if kind == "clear" and clear_armed:
                glow = "#ff6b6b"
                top, mid, bot = "#6b1f1f", "#4a1515", "#2e0d0d"
            self._grad_rounded(x0, y0, x1, y1, 12, top, mid, bot, outline=glow,
                               width=3 if is_hot else 2)
            if kind == hover_kind and cursor_approach > 0.02:
                # Charge-up arc around the hovered button, filling
                # clockwise from the top as the pinch closes.
                extent = max(2, int(340 * min(1.0, cursor_approach)))
                c.create_arc(x0 - 4, y0 - 4, x1 + 4, y1 + 4, start=90, extent=-extent,
                             style="arc", outline="#ffe14d", width=3)
            text = "SURE?" if (kind == "clear" and clear_armed) else self.PALETTE_LABELS.get(kind, kind.upper())
            self._label((x0 + x1) / 2, y1 - 13, text, glow if kind in ("dense", "clear") else "#e8fbff", 10)
            # Mini icon inside the button (above the label)
            ix0, iy0, ix1, iy1 = x0 + 16, y0 + 9, x1 - 16, y1 - 28
            if kind == "clear":
                cx, cy, r = (ix0 + ix1) / 2, (iy0 + iy1) / 2, min(ix1 - ix0, iy1 - iy0) / 2
                c.create_line(cx - r * .7, cy - r * .7, cx + r * .7, cy + r * .7, fill=glow, width=3)
                c.create_line(cx - r * .7, cy + r * .7, cx + r * .7, cy - r * .7, fill=glow, width=3)
                c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=glow, width=2)
            elif kind == "dense":
                # A little filled-vs-hollow grid swatch, echoing the
                # ON/OFF states this button actually switches between.
                gx0, gy0, gx1, gy1 = ix0 + 4, iy0 + 4, ix1 - 4, iy1 - 4
                if dense_on:
                    c.create_rectangle(gx0, gy0, gx1, gy1, fill=glow, outline="")
                for k in range(1, 3):
                    gx = gx0 + (gx1 - gx0) * k / 3
                    gy = gy0 + (gy1 - gy0) * k / 3
                    c.create_line(gx, gy0, gx, gy1, fill="#15151a" if dense_on else glow, width=1)
                    c.create_line(gx0, gy, gx1, gy, fill="#15151a" if dense_on else glow, width=1)
                c.create_rectangle(gx0, gy0, gx1, gy1, outline=glow, width=1)
            elif kind == "bar":
                heights = (0.4, 0.9, 0.6, 1.0)
                bw = (ix1 - ix0) / len(heights)
                for i, hfrac in enumerate(heights):
                    bx0 = ix0 + i * bw + 2
                    bh = hfrac * (iy1 - iy0)
                    c.create_rectangle(bx0, iy1 - bh, bx0 + bw - 4, iy1, fill=glow, outline="")
            elif kind == "line":
                pts = [ix0, iy1 - 6, (ix0 + ix1) / 2 - 6, iy0 + 8,
                       (ix0 + ix1) / 2 + 10, iy1 - 16, ix1, iy0 + 4]
                c.create_line(*pts, fill=glow, width=2, smooth=True)
            elif kind == "pie":
                r = min(ix1 - ix0, iy1 - iy0) / 2
                cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
                c.create_arc(cx - r, cy - r, cx + r, cy + r, start=0, extent=140,
                            fill=glow, outline="")
                c.create_arc(cx - r, cy - r, cx + r, cy + r, start=140, extent=220,
                            outline=glow, width=2, style=tk.ARC)
            elif kind == "live":
                # A little heartbeat/pulse trace - flat, spike, flat -
                # to read as "real-time signal" rather than a static shape.
                midy = (iy0 + iy1) / 2
                pts = [ix0, midy,
                       ix0 + (ix1 - ix0) * 0.35, midy,
                       ix0 + (ix1 - ix0) * 0.45, iy0 + 2,
                       ix0 + (ix1 - ix0) * 0.55, iy1 - 2,
                       ix0 + (ix1 - ix0) * 0.65, midy,
                       ix1, midy]
                c.create_line(*pts, fill=glow, width=2)
                c.create_oval(ix1 - 5, midy - 5, ix1 + 5, midy + 5, fill=glow, outline="")
            elif kind == "shapes3d":
                # Little wireframe cube: two offset squares + 4 joins.
                o = 9
                c.create_rectangle(ix0, iy0 + o, ix1 - o, iy1, outline=glow, width=2)
                c.create_rectangle(ix0 + o, iy0, ix1, iy1 - o, outline=glow, width=2)
                for ax, ay, bx, by in ((ix0, iy0 + o, ix0 + o, iy0), (ix1 - o, iy0 + o, ix1, iy0),
                                       (ix0, iy1, ix0 + o, iy1 - o), (ix1 - o, iy1, ix1, iy1 - o)):
                    c.create_line(ax, ay, bx, by, fill=glow, width=2)
            elif kind == "functions":
                # A little jagged "graph line over axes" icon - reads as
                # "math/data", distinct from the folder icon below it.
                c.create_line(ix0, iy1, ix0, iy0, fill=glow, width=2)
                c.create_line(ix0, iy1, ix1, iy1, fill=glow, width=2)
                pts = [ix0 + 3, iy1 - 4,
                       ix0 + (ix1 - ix0) * 0.3, iy0 + (iy1 - iy0) * 0.35,
                       ix0 + (ix1 - ix0) * 0.55, iy1 - (iy1 - iy0) * 0.3,
                       ix1 - 2, iy0 + 3]
                c.create_line(*pts, fill=glow, width=2, smooth=True)
            elif kind == "files":
                # A simple folder icon - tab + body - distinct enough
                # from the chart-shape icons to read as "not a chart,
                # a source of data" at a glance.
                fw, fh = ix1 - ix0, (iy1 - iy0) * 0.7
                fy0 = iy0 + (iy1 - iy0) * 0.3
                c.create_rectangle(ix0, iy0 + (iy1 - iy0) * 0.18,
                                   ix0 + fw * 0.4, fy0, outline=glow, width=2)
                c.create_rectangle(ix0, fy0, ix1, iy1, outline=glow, width=2)

    def _draw_file_browser(self, hover_file):
        """The second toolbar row that appears under the palette while
        files_open is True - one tile per CSV found in DATA_FOLDER (see
        _scan_files). If none were found, a single grayed-out placeholder
        explains why instead of just... nothing being there."""
        c = self.canvas
        if not self._file_tiles:
            margin, row_y = self.browser_origin()
            self._halo_text(margin, row_y + 20, "(no .csv files in ar_data/ yet)",
                            "#aaaaaa", ("Consolas", 9, "italic"), anchor="w")
            return
        for tile in self._file_tiles:
            is_hot = (tile["path"] == hover_file)
            glow = "#ffffff" if is_hot else "#ffb84d"
            self._grad_rounded(tile["x0"], tile["y0"], tile["x1"], tile["y1"], 10, "#1b2636", "#111a26", "#0a0f16", outline=glow, width=3 if is_hot else 2)
            name = tile["name"][:12]
            self._label((tile["x0"] + tile["x1"]) / 2, (tile["y0"] + tile["y1"]) / 2, name, "#ff9a9a" if (glow == "#ff5d5d") else "#e8fbff", 9)

    def _draw_shape_browser(self, hover_shape):
        for t in self._shape_tiles:
            hot = t["kind"] == hover_shape
            glow = "#ffffff" if hot else "#c58cff"
            self._grad_rounded(t["x0"], t["y0"], t["x1"], t["y1"], 10, "#1b2636", "#111a26", "#0a0f16", outline=glow, width=3 if hot else 2)
            self._label((t["x0"] + t["x1"]) / 2, (t["y0"] + t["y1"]) / 2, t["name"], "#ff9a9a" if (glow == "#ff5d5d") else "#e8fbff", 9)

    def _draw_function_browser(self, hover_function):
        """The row of function tiles under the palette while
        functions_open is True - one per .txt file in
        function_graphs.FUNC_FOLDER (see _scan_functions). A tile whose
        file failed to parse is drawn in red with a "!" rather than
        silently working or silently vanishing, so a typo in a .txt
        file is obvious before it's pinched, not just a failed pick-up
        with nothing on screen explaining why."""
        hover_path = hover_function["path"] if hover_function else None
        if not self._function_tiles:
            margin, row_y = self.browser_origin()
            self._halo_text(margin, row_y + 20, "(no .txt files in ar_functions/ yet)",
                            "#aaaaaa", ("Consolas", 9, "italic"), anchor="w")
            return
        for tile in self._function_tiles:
            is_hot = (tile["path"] == hover_path)
            if tile["error"]:
                glow = "#ff5d5d"
            else:
                glow = "#ffffff" if is_hot else "#7cf2a1"
            self._grad_rounded(tile["x0"], tile["y0"], tile["x1"], tile["y1"], 10, "#1b2636", "#111a26", "#0a0f16", outline=glow, width=3 if is_hot else 2)
            name = tile["name"][:12] if not tile["error"] else f"! {tile['name'][:10]}"
            self._label((tile["x0"] + tile["x1"]) / 2, (tile["y0"] + tile["y1"]) / 2, name, "#ff9a9a" if (glow == "#ff5d5d") else "#e8fbff", 9)

    def _draw_cursor(self, screen_point, pinching, approach=0.0):
        """The actual reason charts were hard to grab: with the overlay
        covering the real desktop instead of living inside the webcam
        window, there was nothing on screen showing where your fingertip
        was actually mapped to. Draws a small ring + crosshair at the
        live cursor position - green while idle, red while pinching
        (same red/green convention as the Desktop-mode reticle, so it
        reads the same way once you've seen either)."""
        if screen_point is None:
            return
        x, y = screen_point
        # green = open, amber = "locked on, about to pinch" (the ring tightens
        # as your fingers close), red = pinching.
        color = "#ff4d4d" if pinching else ("#ffd24d" if approach > 0.05 else "#39ff88")
        r = 14 if pinching else 16 - 6 * approach
        self._halo_oval(x - r, y - r, x + r, y + r, color, 2)
        self._halo_line(x - r - 8, y, x - r + 2, y, color, 2)
        self._halo_line(x + r - 2, y, x + r + 8, y, color, 2)
        self._halo_line(x, y - r - 8, x, y - r + 2, color, 2)
        self._halo_line(x, y + r - 2, x, y + r + 8, color, 2)

    def update(self, grabbed_chart=None, hover_kind=None, cursor_point=None, pinching=False,
               hover_file=None, hover_shape=None, hover_function=None, cursor_approach=0.0):
        """Advances physics/animation, redraws the palette (and the file
        browser row if open), every placed chart, the in-progress ghost
        (if any), and the live cursor indicator, then pumps Tkinter's
        event queue and autosaves the workspace if anything changed.
        Call once per camera frame while in AR mode and enabled; call
        hide() otherwise.
          grabbed_chart: an existing chart currently being dragged -
            drawn with a highlighted outline, and excluded from the
            inertia step (its position comes from move_chart() instead).
          hover_kind: whichever palette button the fingertip is
            currently over (from palette_kind_at()) - highlighted even
            without a pinch, as a "you're about to pick this up" cue.
          hover_file: whichever file tile the fingertip is over (from
            file_tile_at()), same idea as hover_kind but for row 2.
          cursor_point / pinching: where your fingertip currently maps
            to on screen, and whether you're pinching - draws a visible
            marker there. Pass these every frame; without them you're
            aiming blind, which is what prompted adding this at all.
        """
        if self._closed:
            return
        now = time.time()
        dt = min(0.1, max(1e-3, now - self._last_update_time)) if self._last_update_time else 1 / 30
        self._last_update_time = now

        self._step_physics(grabbed_chart, dt)
        if self.scene3d is not None:
            self.scene3d.step(dt)
            if self.scene3d.dirty:
                self.scene3d.dirty = False
                self._mark_dirty()
        for chart in self.charts:
            chart.step_toward_target()
        if self.ghost is not None:
            self.ghost.step_toward_target()

        self.show()
        self.canvas.delete("all")
        if self.scene3d is not None:
            self.scene3d.draw(self.canvas)      # 3D behind the flat charts
        for chart in self.charts:
            self._draw_chart(chart, highlighted=(chart is grabbed_chart))
        if self.ghost is not None:
            self._draw_chart(self.ghost, ghost=True)
        self._update_palette_state(cursor_point, time.time())
        self._tab_hot = (cursor_point is not None and
                         self.tab_rect[0] <= cursor_point[0] <= self.tab_rect[2] and
                         self.tab_rect[1] <= cursor_point[1] <= self.tab_rect[3])
        self._draw_palette(hover_kind, active_kind=(self.ghost.kind if self.ghost else None),
                           cursor_approach=cursor_approach)
        if self.files_open:
            self._draw_file_browser(hover_file)
        if self.shapes_open:
            self._draw_shape_browser(hover_shape)
        if self.functions_open:
            self._draw_function_browser(hover_function)
        self._draw_cursor(cursor_point, pinching, cursor_approach)
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True

        self._maybe_autosave(now)

    def close(self):
        if not self._closed:
            self._save_workspace()
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._closed = True


class CameraPreview:
    """Small always-on-top window showing the live camera feed - just
    enough to confirm hand tracking is actually working once the main
    interface has moved to the full-screen AR overlay instead of a
    normal cv2.imshow() window. Anchored to the bottom-right corner of
    the screen; hide()/show()/toggle() to get it out of the way.

    Needs Pillow (PIL) to convert an OpenCV BGR frame into something
    Tkinter can display - if Pillow isn't installed, this becomes a
    harmless no-op (see AR_PREVIEW_OK in showcase.py) rather than
    breaking the rest of the app.
    """

    def __init__(self, screen_size=None, width=260):
        self.width = width
        self.height = None  # set from the first frame's aspect ratio
        self._closed = False
        self.hidden = True
        self._photo = None  # keep a reference - Tkinter drops the image otherwise
        self.every = 1      # show every Nth camera frame (PerfGovernor raises this under load)
        self._tick = 0

        self.root = tk.Tk()
        self.root.title("Camera Preview")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)

        screen_w, screen_h = screen_size or (
            self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        )
        self._screen_w, self._screen_h = screen_w, screen_h
        # Bottom margin matches MainOverlay's _BOTTOM_SAFE_Y - this used
        # to sit only 16px off the bottom edge, which on a normal Windows
        # taskbar put it (and the plain light-grey Tk background showing
        # through around/before the image) partly behind the taskbar,
        # looking like an unexplained blank box. Placeholder geometry
        # until the first frame tells us the real aspect ratio -
        # repositioned in update() once height is known.
        self._bottom_margin = 70
        self.root.geometry(f"{width}x{int(width * 9 / 16)}+{screen_w - width - 16}+{screen_h - 200}")

        # Explicit dark background + thin border instead of the platform
        # default (a plain light-grey/white panel) so there's never a
        # bare pale box showing before the first frame arrives or around
        # the image's edges.
        self.root.configure(bg="#0c0c10")
        self.label = tk.Label(self.root, bd=2, bg="#0c0c10",
                               highlightbackground="#3a3a42", highlightthickness=1)
        self.label.pack()
        self.root.withdraw()

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

    def update(self, bgr_frame):
        """Push one camera frame (OpenCV BGR numpy array) into the
        preview window. Call once per camera frame; safe to call even
        while hidden (cheap to skip in that case)."""
        if self._closed or self.hidden or not _PIL_OK:
            return
        self._tick += 1
        if self.every > 1 and self._tick % self.every:
            return                                  # thumbnail only needs to be roughly live
        try:
            h, w = bgr_frame.shape[:2]
            if self.height is None:
                self.height = int(self.width * h / w)
                x = self._screen_w - self.width - 16
                y = self._screen_h - self.height - self._bottom_margin
                self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

            image = Image.fromarray(frame_to_preview_rgb(bgr_frame, self.width, self.height))
            if self._photo is not None:
                try:
                    self._photo.paste(image)      # reuse the Tk image: no per-frame allocation
                except Exception:
                    self._photo = None
            if self._photo is None:
                # master=self.label, not omitted - see the identical,
                # longer note in main_overlay._draw_drawing_canvas. This
                # window is one of several separate tk.Tk() roots this
                # app opens, so an unbound PhotoImage can silently
                # attach to the wrong interpreter and crash on use.
                self._photo = ImageTk.PhotoImage(image, master=self.label)
                self.label.configure(image=self._photo)
            self.root.update()
        except tk.TclError:
            self._closed = True

    def close(self):
        if not self._closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._closed = True
