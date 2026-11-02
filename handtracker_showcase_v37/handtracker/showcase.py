"""
showcase.py
-----------
ONE-COMMAND impressive demo for tomorrow.

This is the file you run when you need to show the project looking
polished, useful, and "real-world ready".

Features demonstrated:
  • AI-powered gesture recognition (MediaPipe pre-trained model)
  • Stable multi-frame confirmation + cooldown (no accidental clicks)
  • Fully gesture-driven mode switching (hold Open Palm 1s → on-screen
    menu; point + pinch to pick a mode) - no keyboard needed
  • Desktop mouse control (virtual cursor + pinch-to-click, hold pinch
    to drag)
  • Presentation control mode (point = next, two fingers = previous,
    thumb up = start, fist = toggle laser; while the laser is on, only
    Point to aim it works)
  • Media control (volume / play-pause via gestures)
  • Drawing mode
  • AR Workspace mode: conjure floating charts with a pinch and drag
    them freely around the actual desktop (a full-screen transparent
    overlay, not just inside this window - Windows only)
  • Clear on-screen HUD with mode indicator and live gesture feedback
  • Safe toggle (thumb-down) and fail-safe (mouse to corner)

Run:
    python showcase.py

Gesture control (primary):
    Hold Open Palm for 1 second → open the on-screen mode menu, then
    point at a tile and pinch to select it.

Keys while running (backup/dev shortcuts - not required for normal use):
    1 / d   → Desktop Mouse mode
    2 / p   → Presentation mode
    3 / m   → Media Control mode
    4 / w   → Drawing / Whiteboard mode
    5 / g   → Fun (Whack-a-Mole) mode
    6 / a   → AR Workspace mode (floating charts)
    c       → Toggle the camera-preview thumbnail
    h       → Toggle help overlay
    q       → Quit
"""

from __future__ import annotations

import sys
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np

from perf import PerfGovernor, draw_hand_fast
from camera_stream import FrameGrabber, describe_capture, open_capture

import config
from gestures import PointSmoother, ScalarSmoother, hand_scale_px, is_pinching_from_ratio, pinch_ratio
from ai_gesture_engine import AIGestureEngine
from gesture_actions import GestureActionDispatcher
from gesture_engine import (
    EdgeDwellDetector,
    GestureStabilizer,
    HoldTrigger,
    RotationDialDetector,
    classify_static_gesture,
    is_strict_fist,
)

# Optional desktop pieces – graceful when running headless
try:
    from desktop_control import DesktopController, map_to_screen, DesktopControlAborted
    from desktop_interaction import DesktopInteractionController
    from overlay import CursorReticle, LaserPointer, LaserBanner
    _DESKTOP_OK = True
except Exception as e:
    print(f"[import] Desktop modules unavailable: {e}")
    _DESKTOP_OK = False
    DesktopControlAborted = Exception
    DesktopInteractionController = None

from overlay_drawing import OverlayDrawing
from pinch_stable import PinchStableCursor
import callapp
from settings import Settings
from tutorial import Tutorial
from global_hotkey import GlobalHotkey
from virtualcam import VirtualCamera, diagnose as vcam_diagnose, format_diagnosis

# The main HUD/menu overlay - the "whole program floats over the real
# desktop" piece. Same degrade-gracefully pattern; needs _DESKTOP_OK for
# screen_size the same way AR mode does. NOTE: the old small `Overlay`
# status box (overlay.py) that used to show desktop-mode-only status is
# retired in favour of this - MainOverlay shows the same information
# (and more, for every mode) as part of one floating HUD, so keeping
# both running would just be two redundant boxes. The Overlay class
# itself is untouched in overlay.py in case it's useful elsewhere.
try:
    from main_overlay import MainOverlay
    _OVERLAY_OK = _DESKTOP_OK
except Exception as e:
    print(f"[import] Main overlay module unavailable: {e}")
    _OVERLAY_OK = False

# Optional AR overlay pieces (floating draggable charts + a hideable
# camera-preview thumbnail) - same "degrade gracefully" pattern as the
# desktop pieces above. Needs _DESKTOP_OK too (map_to_screen / screen
# size), so AR mode is only offered when both are available.
try:
    from ar_overlay import AROverlay, CameraPreview
    _AR_OK = _DESKTOP_OK
    try:
        from ar3d_input import AR3DController, HandPointer, HandSlots, HandState
        from tracker_sched import TwoHandScheduler
        _AR3D_INPUT_OK = True
    except Exception as e:
        print(f"[import] AR 3D controls unavailable: {e}")
        _AR3D_INPUT_OK = False
except Exception as e:
    print(f"[import] AR overlay module unavailable: {e}")
    _AR_OK = False
    _AR3D_INPUT_OK = False

# Feature modules
from features.drawing import DrawingPanel
from features.game import WhackAMoleGame


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

MODES = {
    "desktop": {
        "name": "DESKTOP MOUSE",
        "color": (0, 220, 100),
        "help": [
            "Point            -> Move cursor",
            "Pinch            -> Click (hold to drag)",
            "Three fingers    -> Right click",
            "Flick hand L/R   -> Back / Forward",
            "Flick hand Up/Down -> Scroll (faster = further)",
            "Thumb Down       -> Control on / off",
        ],
    },
    "presentation": {
        "name": "PRESENTATION",
        "color": (0, 180, 255),
        "help": [
            "Thumb Up         -> Start slideshow",
            "Point            -> Next slide",
            "Two fingers      -> Previous slide",
            "Fist             -> Laser on / off (Point to aim)",
            "Pinch            -> Click",
            "Laser on         -> slide gestures pause",
            "Thumb Down       -> Control on / off",
        ],
    },
    "media": {
        "name": "MEDIA CONTROL",
        "color": (255, 140, 0),
        "help": [
            "Thumb Up         -> Play",
            "Open Palm        -> Pause",
            "Fist / Point     -> Next / Previous track",
            "Pinch + twist    -> Volume (clockwise = up)",
            "Hold at edge L/R -> Seek forward / back",
            "Two fingers      -> Mute",
            "Thumb Down       -> Control on / off",
        ],
    },
    "drawing": {
        "name": "DRAWING / WHITEBOARD",
        "color": (200, 80, 255),
        "help": [
            "Pinch + move     -> Draw",
            "Open Palm        -> Lift pen",
            "Fist             -> Eraser",
            "Two fingers      -> Change color",
            "Thumb Up         -> Clear canvas",
        ],
    },
    "game": {
        "name": "WHACK-A-MOLE",
        "color": (50, 220, 50),
        "help": [
            "Pinch a mole     -> Whack it",
            "Open Palm        -> Start / restart",
        ],
    },
    "call": {
        "name": "VIDEO CALL",
        "color": (120, 230, 120),
        "help": [
            "Hold FIST ~1 s, hand still      -> Mute / unmute",
            "Hold PEACE SIGN ~1 s            -> Camera on / off",
            "Hold OPEN PALM ~1 s             -> Raise / lower hand",
            "Pinch a call-bar button         -> Same actions, aimed",
            "Waving hands while you talk does nothing",
            "Needs Zoom / Meet / Teams as the front window",
            "V virtual camera   O hide HUD from screen share",
            "Thumb Down                      -> Hand control on / off",
        ],
    },
    "ar": {
        "name": "AR WORKSPACE",
        "color": (255, 230, 0),
        "help": [
            "Pinch icon, drag out, release -> Place a chart",
            "Release on the palette        -> Cancel",
            "Pinch + drag a chart          -> Move it",
            "Fist on a chart               -> Dismiss it",
            "Two fingers on a chart        -> Refresh data",
            "Files icon                    -> Load a CSV from ar_data/",
            "3D icon -> tile               -> Place a 3D bars/surface/shape",
            "Pinch 3D body/floor + drag    -> Move (hand nearer = closer)",
            "Pinch a bar / surface + drag  -> SELECT a portion",
            "Both hands pinch              -> Scale / roll / twist",
            "Pinch INSIDE the selection    -> Stretch: L/R width, U/D height",
            "Both hands, portion selected  -> Hands apart = wider, up/down = taller",
            "Fist over 3D (when not pinching) -> Remove it   |  key U = undo",
            "Tap the floor                 -> Clear selection",
            "Thumb Down                    -> Hide / show overlay",
        ],
    },
}

MODE_ORDER = ["desktop", "presentation", "media", "drawing", "game", "ar", "call"]


def _rounded_rect(img, pt1, pt2, color, thickness=-1, radius=14):
    """Filled or outlined rectangle with rounded corners - cv2 has no
    built-in for this, so it's assembled from a plain rect for the
    straight edges plus four quarter-ellipses for the corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def _corner_brackets(frame, color, length=26, thickness=2, margin=14):
    """Thin HUD-style corner brackets framing the whole feed - a cheap
    way to make the window read as a deliberate interface rather than a
    bare webcam preview."""
    h, w = frame.shape[:2]
    pts = [(margin, margin, 1, 1), (w - margin, margin, -1, 1),
           (margin, h - margin, 1, -1), (w - margin, h - margin, -1, -1)]
    for x, y, dx, dy in pts:
        cv2.line(frame, (x, y), (x + dx * length, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy * length), color, thickness, cv2.LINE_AA)


def _text_with_shadow(frame, text, org, font, scale, color, thickness):
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _blend_solid(frame, x1, y1, x2, y2, color, alpha):
    """Tint the inclusive rectangle (x1,y1)-(x2,y2) with `color` at `alpha`,
    touching ONLY that region. (The old way copied and blended the whole
    720p frame to tint a strip a few dozen pixels tall - three times per
    frame.)"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    if x2 < x1 or y2 < y1:
        return
    roi = frame[y1:y2 + 1, x1:x2 + 1]
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1.0 - alpha, 0, roi)


def draw_hud(frame, mode_key, enabled, gesture, confidence, fps, show_help, ai_label=None,
             help_hidden_note=None):
    """Clean, modern on-screen HUD."""
    h, w = frame.shape[:2]
    mode = MODES[mode_key]
    color = mode["color"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    _corner_brackets(frame, color)

    # Top bar: dark glass panel with a mode-colored accent underline
    _blend_solid(frame, 0, 0, w, 78, (18, 18, 22), 0.78)
    cv2.rectangle(frame, (0, 76), (w, 79), color, -1)

    # Mode name with a colored bullet + faux-bold (double-drawn) text
    cv2.circle(frame, (26, 38), 7, color, -1, cv2.LINE_AA)
    _text_with_shadow(frame, mode["name"], (44, 46), font, 1.05, (245, 245, 245), 2)

    # Status pill (rounded), top-right
    status = "CONTROL ON" if enabled else "CONTROL OFF"
    status_color = (110, 255, 170) if enabled else (110, 110, 255)
    (tw, th), _ = cv2.getTextSize(status, font, 0.62, 2)
    pill_x2, pill_y1, pill_y2 = w - 24, 20, 56
    pill_x1 = pill_x2 - tw - 40
    _rounded_rect(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (30, 30, 34), -1, radius=17)
    _rounded_rect(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), status_color, 2, radius=17)
    cv2.circle(frame, (pill_x1 + 20, (pill_y1 + pill_y2) // 2), 5, status_color, -1, cv2.LINE_AA)
    cv2.putText(frame, status, (pill_x1 + 34, pill_y1 + 25), font, 0.62, status_color, 2, cv2.LINE_AA)

    # Gesture row
    y = 112
    cv2.putText(frame, "GESTURE", (26, y - 16), font, 0.42, (140, 140, 150), 1, cv2.LINE_AA)
    _text_with_shadow(frame, gesture or "-", (26, y + 8), font, 0.8, (255, 255, 255), 2)

    # Confidence bar (rounded), to the right of the gesture text
    bar_x, bar_w, bar_h = 300, 200, 20
    bar_y1, bar_y2 = y - bar_h + 2, y + 6
    cv2.putText(frame, "CONFIDENCE", (bar_x, y - 16), font, 0.42, (140, 140, 150), 1, cv2.LINE_AA)
    _rounded_rect(frame, (bar_x, bar_y1), (bar_x + bar_w, bar_y2), (45, 45, 50), -1, radius=10)
    fill = int(bar_w * max(0.0, min(1.0, confidence)))
    if fill > 4:
        bar_color = (100, 255, 140) if confidence > 0.7 else (60, 210, 255) if confidence > 0.45 else (90, 110, 255)
        _rounded_rect(frame, (bar_x, bar_y1), (bar_x + fill, bar_y2), bar_color, -1, radius=10)
    cv2.putText(frame, f"{confidence:.0%}", (bar_x + bar_w + 14, y + 6),
                font, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    if ai_label:
        badge_txt = f"AI  {ai_label}"
        (btw, bth), _ = cv2.getTextSize(badge_txt, font, 0.5, 1)
        _rounded_rect(frame, (26, y + 22), (26 + btw + 24, y + 22 + bth + 16), (40, 35, 60), -1, radius=9)
        cv2.putText(frame, badge_txt, (38, y + 22 + bth + 6), font, 0.5, (190, 175, 255), 1, cv2.LINE_AA)

    # Bottom strip: subtle panel behind FPS + key legend so it stays
    # legible over busy backgrounds
    _blend_solid(frame, 0, h - 42, w, h, (16, 16, 20), 0.6)
    fps_color = (100, 255, 140) if fps >= 20 else (60, 210, 255) if fps >= 12 else (90, 110, 255)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 130, h - 15), font, 0.6, fps_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Hold Open Palm 1s = Mode menu   |   Keys: 1-6 mode, H help, Q quit",
                (24, h - 15), font, 0.48, (170, 170, 178), 1, cv2.LINE_AA)

    # Help panel - width auto-fits the longest line instead of a fixed
    # 540px, so it doesn't look empty for short lists or clip longer
    # ones (Media mode's list grew a fair bit with the new gestures).
    if show_help:
        title_txt = f"{mode['name']}  -  CONTROLS"
        line_h = 27
        panel_x1, panel_y1 = 24, 158
        panel_h = 54 + line_h * len(mode["help"])

        (title_w, _), _ = cv2.getTextSize(title_txt, font, 0.58, 2)
        content_w = title_w
        for line in mode["help"]:
            (lw, _), _ = cv2.getTextSize(line, font, 0.54, 1)
            content_w = max(content_w, lw)
        panel_x2 = min(w - 24, panel_x1 + content_w + 70)
        panel_y2 = panel_y1 + panel_h

        # Shade only the panel's own bounding box (rounded corners drawn in a
        # small local copy), not the whole frame.
        roi = frame[panel_y1:panel_y2 + 1, panel_x1:panel_x2 + 1]
        shade = roi.copy()
        _rounded_rect(shade, (0, 0), (panel_x2 - panel_x1, panel_y2 - panel_y1), (22, 22, 26), -1, radius=14)
        cv2.addWeighted(shade, 0.85, roi, 0.15, 0, roi)
        _rounded_rect(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), color, 2, radius=14)
        cv2.line(frame, (panel_x1 + 16, panel_y1 + 40), (panel_x2 - 16, panel_y1 + 40), (70, 70, 78), 1, cv2.LINE_AA)
        cv2.putText(frame, title_txt, (panel_x1 + 16, panel_y1 + 28),
                    font, 0.58, color, 2, cv2.LINE_AA)
        for i, line in enumerate(mode["help"]):
            ly = panel_y1 + 64 + i * line_h
            cv2.circle(frame, (panel_x1 + 22, ly - 5), 3, color, -1, cv2.LINE_AA)
            cv2.putText(frame, line, (panel_x1 + 34, ly),
                        font, 0.52, (225, 225, 230), 1, cv2.LINE_AA)
    elif help_hidden_note:
        # The help panel is being auto-hidden by the performance governor
        # even though the user still wants it (show_help is True at the
        # call site, just overridden) - say so in its place, small and
        # dim, rather than just disappearing with no explanation. This
        # is the one line that tells you 'H' will actually bring it back.
        cv2.putText(frame, help_hidden_note, (26, 180),
                    font, 0.5, (130, 160, 255), 1, cv2.LINE_AA)


def draw_mode_menu(frame, highlight_index):
    """Full gesture-driven mode-select menu: one tile per mode, the one
    under the fingertip highlighted. Point left/right to move the
    highlight, pinch to confirm. This is the primary way to switch modes
    - no keyboard needed."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (w, h), (12, 12, 16), -1)
    cv2.addWeighted(shade, 0.72, frame, 0.28, 0, frame)

    _text_with_shadow(frame, "MODE MENU", (w // 2 - 120, 80), font, 1.1, (245, 245, 245), 2)
    _text_with_shadow(frame, "Point at a tile and pinch to select   |   Thumb Down to cancel",
                       (w // 2 - 300, 118), font, 0.55, (190, 190, 195), 1)

    n = len(MODE_ORDER)
    margin, gap = 60, 20
    tile_w = (w - 2 * margin - gap * (n - 1)) // n
    tile_h = 200
    tile_y = h // 2 - tile_h // 2

    for i, key in enumerate(MODE_ORDER):
        mode_def = MODES[key]
        x1 = margin + i * (tile_w + gap)
        x2 = x1 + tile_w
        selected = (i == highlight_index)
        fill = (40, 40, 46) if not selected else tuple(int(c * 0.35) for c in mode_def["color"])
        _rounded_rect(frame, (x1, tile_y), (x2, tile_y + tile_h), fill, -1, radius=18)
        border_color = mode_def["color"] if selected else (80, 80, 88)
        _rounded_rect(frame, (x1, tile_y), (x2, tile_y + tile_h), border_color, 3 if selected else 1, radius=18)

        (tw, _), _ = cv2.getTextSize(str(i + 1), font, 0.9, 2)
        cv2.putText(frame, str(i + 1), (x1 + (tile_w - tw) // 2, tile_y + 60),
                    font, 0.9, mode_def["color"], 2, cv2.LINE_AA)

        words = mode_def["name"].split()
        line_y = tile_y + tile_h - 50
        for word in words:
            (ww, _), _ = cv2.getTextSize(word, font, 0.52, 1)
            cv2.putText(frame, word, (x1 + (tile_w - ww) // 2, line_y),
                        font, 0.52, (235, 235, 240), 1, cv2.LINE_AA)
            line_y += 22


def draw_swipe_meter(frame, velocity, threshold, accent_color, y1=90, hint="L/R", arrows=("<-", "->")):
    """Live speed meter for the swipe gesture: shows current palm
    velocity against the threshold needed to fire, updated every frame
    (not just when a swipe actually lands), so you can see in real time
    whether you're moving fast/hard enough - and in which direction."""
    h, w = frame.shape[:2]
    bar_w, bar_h = 260, 24
    x1 = w - bar_w - 24
    x2, y2 = x1 + bar_w, y1 + bar_h
    center_x = (x1 + x2) // 2

    _rounded_rect(frame, (x1, y1), (x2, y2), (22, 22, 26), -1, radius=10)
    _rounded_rect(frame, (x1, y1), (x2, y2), (85, 85, 92), 1, radius=10)

    span = threshold * 2.2  # bar shows +/- 2.2x the firing threshold
    for sign in (-1, 1):
        tick_x = int(center_x + sign * (threshold / span) * (bar_w // 2))
        cv2.line(frame, (tick_x, y1), (tick_x, y2), (150, 150, 158), 1, cv2.LINE_AA)

    fired = abs(velocity) >= threshold
    fill_color = (60, 220, 90) if fired else accent_color
    vel_clamped = max(-span, min(span, velocity))
    fill_x = int(center_x + (vel_clamped / span) * (bar_w // 2))
    x_start, x_end = sorted((center_x, fill_x))
    cv2.rectangle(frame, (max(x1 + 2, x_start), y1 + 3), (min(x2 - 2, x_end), y2 - 3), fill_color, -1)
    cv2.line(frame, (center_x, y1), (center_x, y2), (200, 200, 205), 1, cv2.LINE_AA)

    if fired:
        label = f"SWIPE {arrows[1]}  fires now" if velocity > 0 else f"{arrows[0]} SWIPE  fires now"
    else:
        label = f"swipe speed (flick hand {hint})"
    label_color = fill_color if fired else (190, 190, 196)
    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                label_color, 1, cv2.LINE_AA)


def draw_edge_zones(frame, edge_dwell, now, accent_color):
    """Semi-transparent bars along the left/right edges marking exactly
    where the edge-hold zone is, so you don't have to guess. Fills in
    (bottom of the bar toward the middle) the longer a fingertip is held
    inside, so you can see the hold building toward firing."""
    h, w = frame.shape[:2]
    zone_w = max(1, int(edge_dwell.edge_ratio * w))
    current_zone, fraction = edge_dwell.progress(now)

    for side, x1, x2 in (("left", 0, zone_w), ("right", w - zone_w, w)):
        active = (current_zone == side)
        base_color = accent_color if active else (130, 130, 138)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, 0), (x2, h), base_color, -1)
        cv2.addWeighted(overlay, 0.24 if active else 0.10, frame, 1.0, 0, frame)
        cv2.rectangle(frame, (x1, 0), (x2, h), base_color, 2)

        if active and fraction > 0:
            fill_color = (60, 220, 90) if fraction >= 1.0 else accent_color
            fill_h = int(h * fraction)
            fy1, fy2 = (h - fill_h) // 2, (h + fill_h) // 2
            cv2.rectangle(frame, (x1 + 3, fy1), (x2 - 3, fy2), fill_color, -1)

        label = "<- HOLD" if side == "left" else "HOLD ->"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = x1 + max(4, (zone_w - tw) // 2)
        cv2.putText(frame, label, (tx, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_dial_progress(frame, dial):
    """Live progress ring for the media-mode volume knob, filling in as
    accumulated rotation approaches the next tick. This is the main
    "make the knob feel smoother/more intuitive" fix on the UI side:
    before, the only feedback was a flash of "VOLUME UP/DOWN" text
    AFTER a tick had already landed, so turning the wrist felt like
    blindly hoping something registered. Showing the turn build up in
    real time - like the resistance/travel on an actual physical knob -
    lets you see exactly how much further to turn before it will fire,
    and immediately shows if you've stopped moving in a direction the
    dial recognizes at all."""
    accumulated, degrees_per_tick, active = dial.progress()
    if not active:
        return
    h, w = frame.shape[:2]
    cx, cy, r = w - 24 - 42, 90 + 42, 42

    cv2.circle(frame, (cx, cy), r, (22, 22, 26), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, (85, 85, 92), 1, cv2.LINE_AA)

    fraction = min(1.0, abs(accumulated) / max(1e-6, degrees_per_tick))
    clockwise = accumulated >= 0
    color = (60, 220, 90) if clockwise else (60, 140, 230)
    if fraction > 0:
        sweep = fraction * 360
        start_angle = -90
        end_angle = start_angle + sweep if clockwise else start_angle - sweep
        cv2.ellipse(frame, (cx, cy), (r - 7, r - 7), 0,
                    min(start_angle, end_angle), max(start_angle, end_angle),
                    color, 5, cv2.LINE_AA)

    label = "VOL"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (200, 200, 205), 1, cv2.LINE_AA)
    hint = "keep turning" if fraction < 1.0 else "..."
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(frame, hint, (cx - hw // 2, cy + r + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (170, 170, 176), 1, cv2.LINE_AA)


def draw_laser_dot(frame, index_norm):
    """Visual-only laser pointer: a glowing red dot at the fingertip
    position, drawn directly on our own preview window. Deliberately
    does NOT touch the real OS mouse or any other window - see the long
    comment in send_presentation_key() for why that used to cause
    misclicks. This can't interfere with anything outside this app."""
    if index_norm is None:
        return
    h, w = frame.shape[:2]
    x = int(max(0.0, min(1.0, index_norm[0])) * w)
    y = int(max(0.0, min(1.0, index_norm[1])) * h)
    # Soft outer glow (a few translucent rings) plus a bright core -
    # cheap way to read as "laser" rather than a flat dot.
    glow = frame.copy()
    for radius, alpha in ((22, 0.15), (14, 0.25), (9, 0.35)):
        cv2.circle(glow, (x, y), radius, (40, 40, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(glow, alpha, frame, 1 - alpha, 0, frame)
        glow = frame.copy()
    cv2.circle(frame, (x, y), 5, (60, 60, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "LASER", (x + 16, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (80, 80, 255), 2, cv2.LINE_AA)


def send_presentation_key(action: str):
    """Send real keyboard shortcuts for presentation control."""
    if not _DESKTOP_OK:
        return
    import pyautogui
    if action == "next":
        pyautogui.press("right")
    elif action == "previous":
        pyautogui.press("left")
    elif action == "click":
        pyautogui.click()
    elif action == "confirm":
        pyautogui.press("enter")
    elif action == "start":
        pyautogui.press("f5")
    # No "laser_on"/"laser_off" here anymore - the laser pointer no longer
    # touches the real keyboard or mouse at all. It used to hold Ctrl while
    # moving the real cursor (the combination PowerPoint uses to draw its
    # own built-in laser dot), but that meant the real cursor - and every
    # real click - followed the fist gesture around, which is what caused
    # the misclicks. It's now a LaserPointer overlay (see overlay.py) that
    # draws its own dot on screen without moving anything real.


def send_media_key(action: str):
    if not _DESKTOP_OK:
        return
    import pyautogui
    if action in ("play_pause", "play", "pause"):
        # Most OSes/keyboards only expose a single PLAY/PAUSE TOGGLE key,
        # not separate Play and Pause keys - so both actions map to the
        # same key. This is also the actual bug fix for "can't resume
        # after Open Palm": the old code sent a distinct "stop" key for
        # Open Palm, which many media apps treat as ending the session
        # outright (not something a later play/pause press can resume
        # from). Using the same toggle key for both keeps them in sync.
        pyautogui.press("playpause")
    elif action == "volume_up":
        pyautogui.press("volumeup")
    elif action == "volume_down":
        pyautogui.press("volumedown")
    elif action == "mute":
        pyautogui.press("volumemute")
    elif action == "next_track":
        pyautogui.press("nexttrack")
    elif action == "prev_track":
        pyautogui.press("prevtrack")
    elif action == "seek_forward":
        # Right Arrow is YouTube's (and most video players') built-in
        # "skip ahead a few seconds" shortcut.
        pyautogui.press("right")
    elif action == "seek_backward":
        pyautogui.press("left")


def _press_hotkey(keys):
    """Presses a keyboard shortcut (module-level so tests can intercept it)."""
    import pyautogui
    pyautogui.hotkey(*keys)


def main():
    print("=" * 60)
    print("  HAND TRACKER – SHOWCASE DEMO")
    print("  AI-powered gesture control for real-world use")
    print("=" * 60)
    print("Starting camera + AI model…")

    import pygame
    pygame.init()
    pygame.mixer.init()

    # ---- video-call support: persisted settings + per-launch flags ----------
    #   --call   lighter CPU profile for running alongside Zoom/Meet/Teams
    #   --vcam   publish the camera to a virtual camera device (virtualcam.py)
    prefs = Settings()
    argv = sys.argv[1:]
    call_profile = ("--call" in argv) or prefs["call_profile"]
    if call_profile:
        # A call app is already encoding video and using the CPU: cap the
        # rate and use the lite hand model so both fit. The capture SIZE is
        # deliberately left alone - an earlier version shrank it to 960x540,
        # which most webcams don't support natively, so the driver scaled
        # the picture and it came out visibly soft. The tracking cost is in
        # the model and the frame rate, not the capture resolution.
        for name in ("CAMERA_FPS", "TARGET_FPS"):
            if hasattr(config, name):
                setattr(config, name, 24)
        print("[INFO] Call profile: 24 fps cap, lite hand model")
    if "--shared-camera" in argv:
        # Experimental: Media Foundation can let several apps open the same
        # camera on some Windows 10/11 setups (via the Camera Frame Server),
        # which would let a call app and this app both use it with no
        # virtual camera. Depends on the camera/driver; not guaranteed.
        config.CAMERA_BACKEND = "msmf"
        print("[INFO] Shared-camera mode: using the Media Foundation backend")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,  # single hand for cleaner demo
        model_complexity=0 if call_profile else 1,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    mp_draw = mp.solutions.drawing_utils

    ai_engine = AIGestureEngine(use_ai=True, num_hands=1)
    dispatcher = GestureActionDispatcher()
    # Desktop mode drives every action off classify_static_gesture()
    # directly (see the comment further down, by rule_raw_gesture) - not
    # the AI-hybrid engine. It needs its own GestureStabilizer for the
    # same reason: ai_engine.stabilizer confirms gestures off the mixed
    # AI/rule signal, which can flip between the two mid-hold even while
    # the hand shape isn't actually changing (real ML label one frame,
    # rule fallback the next), resetting its own confirm-frame streak
    # each time. Feeding that mismatched "confirmed" into
    # interaction.process_frame() - while the continuous drag/cursor/
    # pinch logic was already running off the clean rule-based
    # raw_gesture - was making right-click/toggle lag, drop, or double
    # fire relative to what the cursor/drag was actually doing, and made
    # the HUD gesture readout disagree with what was actually being
    # acted on. Keeping a dedicated stabilizer fed only by
    # classify_static_gesture() keeps desktop mode's "what's shown" and
    # "what fires" consistent with each other, same as main_desktop.py.
    desktop_stabilizer = GestureStabilizer()

    # ai_engine.process() runs a second, full ML pass over every frame -
    # only Presentation and Media mode actually look at its result
    # (Desktop mode uses classify_static_gesture() directly, per above;
    # Drawing/Game mode only look at raw pinch + fingertip position, not
    # gesture classification at all). Running that extra inference every
    # frame regardless of mode was pure wasted work three modes out of
    # five, and was a direct contributor to the sluggish, "off" feeling
    # in Desktop mode specifically - it was competing for frame time with
    # the exact cursor/click path that most needs to feel instant. Only
    # pay for it in the two modes that use it (see the mode dispatch
    # below).
    AI_ENGINE_MODES = ("presentation", "media")

    # Alternative to swipe for presentation-mode previous/next: hold the
    # pointing fingertip at the left/right edge of frame for a moment.
    # See gesture_engine.EdgeDwellDetector.
    edge_dwell = EdgeDwellDetector()

    # Desktop pieces
    desktop_ctrl = None
    interaction = None
    reticle = None
    laser_pointer = None
    laser_banner = None
    if _DESKTOP_OK:
        try:
            desktop_ctrl = DesktopController()
            interaction = DesktopInteractionController(desktop_ctrl)
            reticle = CursorReticle()
            laser_pointer = LaserPointer()
            laser_banner = LaserBanner()
            print("[OK] Desktop control ready")
        except Exception as e:
            print(f"[WARN] Desktop features limited: {e}")
            interaction = None
    else:
        print("[INFO] Running without desktop control (headless or missing deps)")
        interaction = None

    # AR pieces - the full-screen floating-chart overlay and the small
    # hideable camera-preview thumbnail. Independent of desktop_ctrl's
    # DesktopController (no mouse/keyboard actions involved at all - AR
    # mode never touches the real cursor or sends a single keypress),
    # but still needs it for screen_size / map_to_screen, so it's only
    # attempted once desktop_ctrl exists.
    ar_overlay_win = None
    # Camera preview thumbnail: disabled for now rather than just
    # defaulting to hidden - it wasn't pulling its weight even restyled
    # and off-by-default, so it's switched off at the source here
    # instead of leaving a dormant window + toggle around. CameraPreview
    # itself is untouched in ar_overlay.py (and camera_preview stays a
    # real, always-None variable, not deleted outright) so every
    # `if camera_preview is not None:` guard below still behaves
    # correctly and this is a one-line change to bring back if that
    # changes.
    camera_preview = None
    if _AR_OK and desktop_ctrl is not None:
        try:
            ar_overlay_win = AROverlay(screen_size=desktop_ctrl.screen_size)
            print("[OK] AR overlay ready" + ("" if ar_overlay_win.fullscreen_transparent
                                              else " (fallback window - see warning above)"))
        except Exception as e:
            print(f"[WARN] AR overlay unavailable: {e}")
            ar_overlay_win = None

    # 3D objects need TWO hands. A separate 2-hand MediaPipe instance is
    # used ONLY while in AR mode, so every other mode keeps the single-hand
    # pipeline (and its "hand [0] is THE hand" assumption) exactly as before.
    ctl3d = hands2 = hand_slots = tracker_sched = None
    pointers3d = []
    if (_AR3D_INPUT_OK and ar_overlay_win is not None and ar_overlay_win.scene3d is not None
            and desktop_ctrl is not None):
        try:
            ctl3d = AR3DController(ar_overlay_win.scene3d)
            # Lite model (not the full one) and the same tracking-
            # confidence fix as the single-hand tracker above - AR mode
            # already pays extra for tracking two hands at once, so the
            # heavy model here was doubling down on cost for no
            # documented reason (the full model is deliberately used in
            # main_desktop.py instead, specifically for single-hand fist-
            # drag reliability - a case that doesn't apply here), and a
            # fast two-hand gesture (the resize/rotate motions AR mode is
            # built around) is exactly where a too-strict tracking
            # confidence costs the most.
            hands2 = mp_hands.Hands(max_num_hands=2, model_complexity=config.MODEL_COMPLEXITY,
                                    min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                                    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE)
            hand_slots = HandSlots()
            tracker_sched = TwoHandScheduler(hands, hands2,
                                             probe_every=getattr(config, "AR_SECOND_HAND_PROBE_FRAMES", 6),
                                             back_after=getattr(config, "AR_DUAL_FALLBACK_FRAMES", 12))
            _map3d = lambda n: map_to_screen(n, desktop_ctrl.screen_size)
            pointers3d = [HandPointer(_map3d, config.FRAME_WIDTH, config.FRAME_HEIGHT) for _ in range(2)]
            print("[OK] AR 3D objects ready (two-hand)")
        except Exception as e:
            print(f"[WARN] AR 3D objects unavailable: {e}")
            ctl3d = hands2 = hand_slots = tracker_sched = None
    _no_hand_states = [HandState(), HandState()] if ctl3d is not None else None

    # Main HUD/menu overlay - the "whole program floats over your
    # desktop" piece: mode name, control state, gesture, confidence,
    # FPS, help panel, and the mode menu, all shown outside the cv2
    # preview window as small corner drawers (see main_overlay.py) that
    # only expand on real mouse hover. That cv2 window
    # (draw_hud/draw_mode_menu) is left fully intact alongside this for
    # troubleshooting.
    main_overlay_win = None
    if _OVERLAY_OK and desktop_ctrl is not None:
        try:
            main_overlay_win = MainOverlay(screen_size=desktop_ctrl.screen_size)
            print("[OK] Main overlay ready" + ("" if main_overlay_win.fullscreen_transparent
                                                else " (fallback window - see warning above)"))
        except Exception as e:
            print(f"[WARN] Main overlay unavailable: {e}")
            main_overlay_win = None

    # Drawing mode's overlay-native whiteboard (toolbar + ink) - only when
    # there's an overlay to put it on and a screen to map into.
    paint = None
    if main_overlay_win is not None and desktop_ctrl is not None:
        try:
            paint = OverlayDrawing(*desktop_ctrl.screen_size)
        except Exception as e:
            print(f"[WARN] Overlay drawing unavailable, using the in-window panel: {e}")
            paint = None

    # Features
    drawing = DrawingPanel(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    game = WhackAMoleGame(config.FRAME_WIDTH, config.FRAME_HEIGHT)

    # Smoothers
    index_smoother = PointSmoother(min_cutoff=config.CURSOR_SMOOTHING_MIN_CUTOFF,
                                    beta=config.CURSOR_SMOOTHING_BETA)
    thumb_smoother = PointSmoother(min_cutoff=config.CURSOR_SMOOTHING_MIN_CUTOFF,
                                    beta=config.CURSOR_SMOOTHING_BETA)
    hand_scale_smoother = ScalarSmoother(alpha=0.12)
    # Light smoothing on the pinch ratio itself, ahead of the
    # engage/release hysteresis thresholds in is_pinching_from_ratio()
    # (see gestures.py) - together these are the fix for pinch-click
    # flicker. alpha=0.5 is much lighter than hand_scale_smoother's
    # 0.12: it only needs to knock down single-frame noise spikes, not
    # meaningfully lag behind a real, deliberate pinch.
    pinch_ratio_smoother = ScalarSmoother(alpha=0.75)
    mcp_smoother = PointSmoother(min_cutoff=config.CURSOR_SMOOTHING_MIN_CUTOFF,
                                 beta=config.CURSOR_SMOOTHING_BETA)
    pinch_raw_state = False
    # Volume "knob" gesture (Media mode) - pinch and rotate the wrist.
    # The wrist is the pivot the dial measures rotation AROUND, so its
    # own jitter matters just as much as the fingertip's: an unsmoothed
    # pivot point wobbling by even a couple of pixels adds noticeable
    # angular noise right at the center of the turn, which is what made
    # the dial feel jumpy/unpredictable rather than like a real knob.
    # Same filter settings as index/thumb so all three stay consistent
    # with each other.
    wrist_smoother = PointSmoother(min_cutoff=config.CURSOR_SMOOTHING_MIN_CUTOFF,
                                    beta=config.CURSOR_SMOOTHING_BETA)
    volume_dial = RotationDialDetector()
    dial_flash = None  # (direction, expire_time) for a brief on-screen confirmation

    cap = open_capture(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT,
                       fps=getattr(config, "CAMERA_FPS", 30),
                       fourcc=getattr(config, "CAMERA_FOURCC", "MJPG"),
                       backend=getattr(config, "CAMERA_BACKEND", None))
    _got = describe_capture(cap)
    if _got and _got[0] > 0:
        print(f"[CAM] asked for {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}, camera delivers "
              f"{_got[0]}x{_got[1]} @ {_got[2]:.0f} fps ({_got[3]})")
        if (_got[0], _got[1]) != (config.FRAME_WIDTH, config.FRAME_HEIGHT):
            print("[CAM] WARNING: the camera is not using the requested size - the driver is scaling "
                  "the picture, which looks soft. Try a size your camera supports natively "
                  "(usually 1280x720 or 640x480) in config.FRAME_WIDTH/FRAME_HEIGHT.")
    if getattr(config, "CAMERA_THREADED", True):
        try:
            cap = FrameGrabber(cap)          # background capture: newest frame, decode overlaps tracking
        except Exception as e:
            print(f"[WARN] threaded camera unavailable, reading inline: {e}")
    perf = PerfGovernor(target_fps=getattr(config, "TARGET_FPS", 30),
                        enabled=getattr(config, "ADAPTIVE_QUALITY", True))
    fast_skeleton = getattr(config, "FAST_SKELETON", True)

    mode = "desktop"
    show_help = True
    # True once the user explicitly asks to see help while the perf
    # governor is suppressing it (see 'h' handling below) - makes that
    # request stick until they turn help off again, instead of the
    # governor silently re-hiding it next frame. Without this, pressing
    # H while AR mode's chart placement was keeping quality at level 3
    # looked like it did nothing (or worse, looked like it turned help
    # OFF, because the keypress still flipped show_help even though the
    # panel was already invisible - so a second press to "turn it back
    # on" actually turned the underlying flag off, and it stayed hidden
    # even once performance recovered, i.e. "can't be re-enabled").
    force_help_override = False
    enabled = False
    fps = 0.0
    prev_t = time.time()
    # Rolling telemetry for AR mode's "Live" chart source - updated every
    # frame regardless of mode, so a Live chart isn't starting from
    # nothing the moment you place one. fps_history: last N smoothed FPS
    # samples (a genuine live line). gesture_counts: how many times each
    # gesture has been read this session (a genuine live pie of "what
    # you've actually been doing with your hand").
    fps_history = deque(maxlen=50)
    gesture_counts = Counter()
    click_flash = 0.0
    last_action_time = 0.0
    presentation_was_pinching = False
    fist_latched = False              # one fist = one laser toggle
    game_prev_pinch = False
    ui_prev_pinch = False                   # pinch edge for tutorial / call-bar / help-button presses
    toast_text, toast_until = None, 0.0     # short on-screen confirmation messages

    def set_toast(text, now_t):
        nonlocal toast_text, toast_until
        toast_text, toast_until = text, now_t + 2.5

    # Video-call pieces (see virtualcam.py, callapp.py, tutorial.py)
    vcam = VirtualCamera()
    vcam_want = ("--vcam" in argv) or prefs["virtual_cam"]
    vcam_tried = False
    _scr = desktop_ctrl.screen_size if desktop_ctrl is not None else (1920, 1080)
    tut = Tutorial(*_scr)
    enabled_before_tutorial = None
    call_ctl = callapp.CallController(_press_hotkey, callapp.foreground_title, prefs["call_app"])
    call_hold = callapp.HoldToConfirm()
    call_bar_buttons = callapp.call_bar_layout(*_scr)
    call_status_at, call_status = 0.0, (False, "")
    prev_mode = "desktop"

    # ---- camera release: let OTHER apps use the webcam ------------------------
    # While released this app is deliberately blind (no camera = no hands), so
    # the way back is NOT a gesture: Ctrl+Alt+K from anywhere, K in the debug
    # window, or restarting. (The alternative that needs no switching at all
    # is the virtual camera - V.)
    camera_on = True
    cam_resume_request = False
    global_hk = GlobalHotkey()
    if global_hk.start():
        print(f"[CAM] Global hotkey {GlobalHotkey.LABEL} toggles the camera from any app")
    else:
        print(f"[CAM] No global hotkey ({global_hk.error}); use K in the debug window to resume")

    def set_control(value):
        nonlocal enabled
        enabled = bool(value)
        if interaction is not None:
            interaction.enabled = enabled

    def release_camera():
        nonlocal cap, camera_on, vcam_want, vcam_tried
        if not camera_on:
            return
        if vcam.active:
            vcam.stop()
            vcam_want, vcam_tried = True, True      # remembered: restarted on resume
        cap.release()
        camera_on = False
        print("[CAM] Camera released - other apps can use it now")

    def resume_camera():
        nonlocal cap, camera_on, vcam_tried
        try:
            new_cap = open_capture(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT,
                                   fps=getattr(config, "CAMERA_FPS", 30),
                                   fourcc=getattr(config, "CAMERA_FOURCC", "MJPG"),
                                   backend=getattr(config, "CAMERA_BACKEND", None))
            ok_open = new_cap.isOpened()
        except Exception:
            new_cap, ok_open = None, False
        if not ok_open:
            if new_cap is not None:
                new_cap.release()
            return False
        if getattr(config, "CAMERA_THREADED", True):
            try:
                new_cap = FrameGrabber(new_cap)
            except Exception:
                pass
        cap, camera_on = new_cap, True
        vcam_tried = False                           # restarts the virtual camera if it was on
        print("[CAM] Camera back")
        return True
    overlay_stable = PinchStableCursor()   # pinch-stable pointer for the overlay reticle/paint/game
    ar_trail = deque(maxlen=40)       # (time, screen_pt) while not a fist - see AR fist-delete
    fist_retry_until = 0.0
    laser_smoother = PointSmoother(min_cutoff=config.LASER_SMOOTHING_MIN_CUTOFF,
                                   beta=config.LASER_SMOOTHING_BETA)
    last_point_time = float("-inf")
    laser_pos = None                  # last screen point the laser was aimed at

    # Gesture-driven mode menu state (see the "Global: open-palm TAP vs.
    # HOLD" block below).
    menu_open = False
    menu_highlight = 0
    menu_was_pinching = False
    palm_hold_start = None
    palm_menu_fired = False
    laser_active = False

    # AR mode state: the chart currently grabbed by a held pinch (None
    # when nothing's being dragged), and whether the camera-preview
    # thumbnail is showing (starts visible - the whole point of it is
    # "check hand tracking is working", so it should be on by default).
    grabbed_chart = None
    was_pinching_ar = False
    # OFF by default - see the camera_preview.update() call above for
    # why. Press 'c' any time to bring up the raw feed.
    camera_preview_visible = False
    if camera_preview is not None:
        camera_preview.show()

    print("\nReady! Camera window will open.")
    print("  Thumb Down = toggle control")
    print("  1-5 or d/p/m/w/g = switch modes")
    print("  H = help, Q = quit\n")

    try:
        while True:
            if global_hk.poll():
                if camera_on:
                    release_camera()
                else:
                    cam_resume_request = True
            if not camera_on:
                if cam_resume_request:
                    cam_resume_request = False
                    if resume_camera():
                        set_toast("Camera back", time.time())
                    else:
                        set_toast("Camera is still in use by another app - close it there first", time.time())
                idle = np.zeros((300, 640, 3), np.uint8)
                cv2.putText(idle, "Camera released - other apps can use it", (24, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(idle, f"{GlobalHotkey.LABEL} (anywhere) or K here to take it back", (24, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1)
                cv2.imshow("Hand Tracker – Showcase (AI + Real-World Control)", idle)
                if main_overlay_win is not None:
                    try:
                        main_overlay_win.update(
                            mode, MODES, MODE_ORDER, enabled, "-", 0.0, 0.0, False,
                            extras={"camera_on": False,
                                    "banner": {"title": "Camera released",
                                               "lines": ["Other apps (Zoom, OBS, ...) can use your webcam now.",
                                                         "This app can't see your hands until you take it back:",
                                                         f"press {GlobalHotkey.LABEL} from anywhere, or K in the debug window."]},
                                    "toast": toast_text if time.time() < toast_until else None})
                    except Exception as e:
                        print(f"[WARN] Main overlay draw failed this frame: {e}")
                k_idle = cv2.waitKey(100) & 0xFF
                if k_idle == ord("q"):
                    break
                if k_idle == ord("k"):
                    cam_resume_request = True
                continue
            ret, frame = cap.read()
            if not ret:
                break
            t_compute0 = time.perf_counter()        # compute time excludes the camera wait

            if vcam_want and not vcam_tried:
                vcam_tried = True
                ok_v, msg_v = vcam.start(frame.shape[1], frame.shape[0], 30)
                print(f"[vcam] {msg_v}")
                if not ok_v:
                    print("[vcam] Setup check:\n" + format_diagnosis(vcam_diagnose()))
                set_toast(msg_v if ok_v else msg_v.split(" (")[0].split(". ")[0], time.time())
                vcam_want = ok_v
            if vcam.active:
                # What the call sees: the raw camera image (not the mirrored,
                # skeleton-and-HUD debug view). Copied because the camera
                # buffer may be reused while the worker thread converts it.
                vcam.send(frame.copy())

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            use_two_hands = ctl3d is not None and mode == "ar" and enabled
            # With two hands wanted, the scheduler uses the cheap 1-hand
            # tracker and only probes for a second hand every few frames
            # (see tracker_sched.py for why: a 2-hand tracker with one hand
            # in view re-runs palm detection on EVERY frame).
            if use_two_hands and tracker_sched is not None:
                results = tracker_sched.process(rgb)
            else:
                results = (hands2 if use_two_hands else hands).process(rgb)
            now = time.time()
            # lm_list[0] is always THE primary hand every mode below uses.
            # In AR mode it comes from stable slots (see HandSlots) so it
            # can't flip between your two hands from frame to frame.
            lm_list = list(results.multi_hand_landmarks or [])
            slot_hands = [None, None]
            if use_two_hands and lm_list:
                slot_hands = hand_slots.assign(lm_list, now)
                lm_list = [h for h in slot_hands if h is not None]

            # FPS
            dt = max(1e-6, now - prev_t)
            prev_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            fps_history.append(fps)

            raw_gesture = None
            confirmed = None
            confidence = 0.0
            ai_debug = None
            pinching = False
            index_norm = None
            overlay_norm = None       # pinch-stabilised fingertip (see PinchStableCursor)
            mcp_norm = None
            rule_raw_gesture = None
            rule_confirmed = None
            call_ring = 0.0

            if lm_list:
                hand_landmarks = lm_list[0]
                for _h in lm_list:
                    if fast_skeleton:
                        draw_hand_fast(frame, _h)
                    else:
                        mp_draw.draw_landmarks(frame, _h, mp_hands.HAND_CONNECTIONS)

                # Only run the AI-hybrid engine's (second, full-frame ML)
                # pass in the modes that actually consult its output -
                # see AI_ENGINE_MODES above for why. Desktop/Drawing/Game
                # skip it entirely, which is most of the perf win.
                if mode in AI_ENGINE_MODES and not (mode == "presentation" and laser_active):
                    raw_gesture, confirmed, confidence = ai_engine.process(
                        frame, hand_landmarks, int(now * 1000)
                    )
                    ai_debug = ai_engine.last_ai_debug

                # desktop_interaction.py's held-gesture mechanics (cursor
                # tracking while "pointing", drag while "fist", and the
                # "pointing" gate on pinch) were written and tuned against
                # classify_static_gesture()'s specific rule-based semantics.
                # Feeding it the AI-hybrid raw_gesture instead made that
                # "pointing" gate depend on the AI model's stricter
                # Pointing_Up category, which doesn't reliably match a
                # normal pointing-at-the-screen pose - causing pinch-clicks
                # to sit blocked for a beat (or several) until the rule
                # fallback happened to agree, and the fist-drag hold to
                # flicker in and out. Desktop mode uses the rule-based
                # classification directly for that reason; other modes
                # still use the AI-hybrid raw/confirmed values below.
                rule_raw_gesture = classify_static_gesture(hand_landmarks)
                if rule_raw_gesture is not None:
                    gesture_counts[rule_raw_gesture] += 1
                palm_lm = hand_landmarks.landmark[9]
                rule_confirmed = desktop_stabilizer.update(
                    rule_raw_gesture, (palm_lm.x, palm_lm.y), now
                )
                if mode == "presentation" and laser_active:
                    # Laser on: skip the second (AI) ML pass - roughly
                    # halves per-frame work, which is the biggest single
                    # cause of the dot lagging - and use the rule-based
                    # classification, which is all the laser needs (Point
                    # to aim, Fist to turn off).
                    raw_gesture, confirmed, confidence = rule_raw_gesture, rule_confirmed, 0.7

                # Pinch detection. The RATIO is computed from the RAW
                # thumb/index landmarks, not the One-Euro-smoothed cursor
                # points: those filters deliberately lag to steady the
                # cursor, and running the pinch through them delayed every
                # press by several frames (the "not snappy" feel). A ratio
                # of two nearby points is already stable; the hysteresis in
                # is_pinching_from_ratio handles the rest.
                idx_pt = index_smoother.update(
                    (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y), now
                )
                th_pt = thumb_smoother.update(
                    (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y), now
                )
                index_px = (idx_pt[0] * config.FRAME_WIDTH, idx_pt[1] * config.FRAME_HEIGHT)
                thumb_px = (th_pt[0] * config.FRAME_WIDTH, th_pt[1] * config.FRAME_HEIGHT)
                raw_index_px = (hand_landmarks.landmark[8].x * config.FRAME_WIDTH,
                                hand_landmarks.landmark[8].y * config.FRAME_HEIGHT)
                raw_thumb_px = (hand_landmarks.landmark[4].x * config.FRAME_WIDTH,
                                hand_landmarks.landmark[4].y * config.FRAME_HEIGHT)
                raw_scale = hand_scale_px(hand_landmarks, config.FRAME_WIDTH, config.FRAME_HEIGHT)
                smooth_scale = hand_scale_smoother.update(raw_scale)
                raw_pinch_ratio = pinch_ratio(raw_index_px, raw_thumb_px, smooth_scale)
                smoothed_pinch_ratio = pinch_ratio_smoother.update(raw_pinch_ratio)
                pinching = is_pinching_from_ratio(smoothed_pinch_ratio, pinch_raw_state)
                # A closed fist puts the thumb against the curled fingers,
                # which looks like a pinch by distance alone - so a real
                # fist must not START a pinch. Only a STRICT fist counts
                # (half-closed hands are not fists any more). It is
                # checked only when a pinch is not already held, so a
                # drag in progress isn't dropped when the fingers curl a
                # little while dragging.
                fist_now = rule_raw_gesture == "fist" or is_strict_fist(hand_landmarks)
                if fist_now:
                    # Desktop keeps a drag alive if the fingers curl a
                    # little mid-drag; every other mode drops the pinch
                    # the instant a fist is seen. (Latching a pinch that
                    # began while the hand was CLOSING into a fist is what
                    # made the presentation laser keep clicking.)
                    if mode != "desktop" or not pinch_raw_state:
                        pinching = False
                pinch_raw_state = pinching
                # Index knuckle (MCP): barely moves when the finger curls
                # into a pinch, so it's the stable reference used to
                # anchor the cursor during a pinch/drag.
                mcp_norm = mcp_smoother.update(
                    (hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y), now
                )
                index_norm = idx_pt
                # The overlay reticle (every mode), drawing and
                # Whack-a-Mole all read THIS, not the raw fingertip: the
                # fingertip physically drops toward the thumb as a pinch
                # closes, so a raw-tip cursor ends up below what you were
                # aiming at (misclicks, strokes starting off-target).
                # PinchStableCursor remembers where the tip sat relative
                # to the knuckle just BEFORE the fingers started closing,
                # locks to that, and follows only the (steady) knuckle
                # until the fingers open again - the same idea Desktop
                # mode and the AR pointer already use.
                overlay_norm = overlay_stable.update(idx_pt, mcp_norm, smoothed_pinch_ratio, now)

                # Wrist point, for the Media-mode volume "knob" gesture
                # (RotationDialDetector) - tracks the pinch point's angle
                # around the wrist as the hand twists.
                wrist_norm = wrist_smoother.update(
                    (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y), now
                )
                wrist_px = (wrist_norm[0] * config.FRAME_WIDTH, wrist_norm[1] * config.FRAME_HEIGHT)
                pinch_mid_px = ((index_px[0] + thumb_px[0]) / 2, (index_px[1] + thumb_px[1]) / 2)

                # Edge-dwell: alternative previous/next trigger, used in
                # Media mode for seeking (see EdgeDwellDetector). Updated
                # every frame a hand is visible so the dwell timer tracks
                # properly even before/after being in media mode; only
                # acted on below while mode == "media".
                edge_nav = edge_dwell.update(index_norm[0], now)
                # Live zone (not just the fired event above) - 'left'/
                # 'right' the whole time the fingertip is resting in
                # either edge zone, not only once the hold completes.
                # Used below to stop "pointing" (index finger extended -
                # which an edge-hold naturally IS) from also firing its
                # own separate prev_track action while someone's mid-way
                # through holding at the edge to seek; see the gate on
                # confirmed == "pointing" in Media mode.
                edge_zone_now, _edge_frac_now = edge_dwell.progress(now)

                # ---- Global: open-palm TAP vs. HOLD ----
                # A quick open-palm (released before MENU_OPEN_HOLD_SECONDS)
                # is a mode-specific tap action (start slideshow / pause
                # music). The SAME shape held past that threshold instead
                # opens or closes the gesture-driven mode menu, from any
                # mode - no keyboard needed. Splitting the two on hold
                # duration (like the pinch tap-vs-drag split in desktop
                # mode) means holding for the menu never also fires the
                # mode's tap action first.
                open_palm_tap = False
                if rule_raw_gesture == "open_palm":
                    if palm_hold_start is None:
                        palm_hold_start = now
                        palm_menu_fired = False
                    elif not palm_menu_fired and (now - palm_hold_start) >= config.MENU_OPEN_HOLD_SECONDS:
                        menu_open = not menu_open
                        palm_menu_fired = True
                        show_help = True
                else:
                    if palm_hold_start is not None and not palm_menu_fired:
                        open_palm_tap = True
                    palm_hold_start = None

                # Laser pointer only ever makes sense in Presentation mode
                # while control is enabled and the menu isn't up - hide the
                # overlay dot the instant any of that stops being true, so
                # it can never get left showing on screen by mistake.
                if laser_active and not (mode == "presentation" and enabled and not menu_open):
                    laser_active = False
                    laser_pos = None
                    if laser_pointer:
                        laser_pointer.hide()
                    if laser_banner:
                        laser_banner.hide()

                # The volume-knob gesture's rotation tracking only makes
                # sense while actually in Media mode - reset it whenever
                # we're not, so re-entering Media mode later (possibly
                # mid-pinch) starts from a fresh reference angle instead
                # of comparing against a stale one from ages ago.
                if mode != "media":
                    volume_dial.reset()

                # The AR overlay's charts only make sense while actually
                # in AR mode with control enabled - hide the whole
                # floating-chart window (this IS the gesture-driven
                # "hide the whole overlay" behavior: Thumb Down already
                # flips `enabled` for every mode via the toggle further
                # down, so it doubles as AR's show/hide gesture for
                # free) the instant either of those stops being true,
                # and drop whatever was mid-drag so it doesn't keep
                # following a fingertip that's no longer in AR mode.
                if ar_overlay_win is not None and not (mode == "ar" and enabled):
                    ar_overlay_win.hide()
                    grabbed_chart = None
                    was_pinching_ar = False
                    ar_overlay_win.files_open = False
                    ar_overlay_win.shapes_open = False
                    if ctl3d is not None:
                        ctl3d.reset()

                # ---- Mode-specific logic ----
                if menu_open:
                    # ---- Gesture-driven mode menu ----
                    # Point left/right to highlight a mode, pinch to
                    # select it (closes the menu and switches), or hold
                    # thumb-down to cancel without switching. This is the
                    # primary way to switch modes - the keyboard shortcuts
                    # still work too, as a backup for development/testing.
                    if index_norm is not None:
                        menu_highlight = min(
                            len(MODE_ORDER) - 1,
                            max(0, int((overlay_norm or index_norm)[0] * len(MODE_ORDER))),
                        )
                    if pinching and not menu_was_pinching:
                        mode = MODE_ORDER[menu_highlight]
                        menu_open = False
                        show_help = True
                    menu_was_pinching = pinching
                    if rule_confirmed == "thumb_down":
                        menu_open = False

                elif mode == "desktop" and interaction is not None:
                    try:
                        events = interaction.process_frame(
                            rule_raw_gesture, rule_confirmed, pinching, index_norm, now,
                            hand_ref_point=mcp_norm,
                            swipe_speed_y=desktop_stabilizer.last_swipe_velocity_y,
                        )
                        if any(e[0] == "click" for e in events):
                            click_flash = now + 0.25
                        enabled = interaction.enabled
                    except DesktopControlAborted as e:
                        print(f"[safety] {e}")
                        interaction.enabled = False
                        enabled = False

                elif mode == "presentation" and enabled:
                    if confirmed:
                        dispatcher.dispatch(confirmed)

                    # ---- Laser = a TOGGLE --------------------------------
                    # Make a fist once to turn laser mode ON, again to turn
                    # it OFF. (Holding a fist re-fires "confirmed" every
                    # cooldown, so the toggle latches until the fist is
                    # released - one fist = one flip.)
                    if raw_gesture != "fist":
                        fist_latched = False
                    elif confirmed == "fist" and not fist_latched:
                        fist_latched = True
                        laser_active = not laser_active
                        if laser_active:
                            laser_pos = None
                            laser_smoother.reset()
                            last_point_time = float("-inf")
                            if laser_banner:
                                laser_banner.show()
                        else:
                            ai_engine.reset()
                            if laser_pointer:
                                laser_pointer.hide()
                            if laser_banner:
                                laser_banner.hide()

                    if laser_active:
                        # While the laser is on, Point just AIMS it (an
                        # on-screen dot - the real mouse is never touched)
                        # and nothing else in this mode reacts: no next /
                        # previous / click / start. Only the global
                        # gestures still work (Open Palm hold = mode menu,
                        # Thumb Down = control off, Fist = laser off).
                        if desktop_ctrl is not None:
                            following = raw_gesture == "pointing"
                            if following:
                                if now - last_point_time > config.LASER_POINT_GRACE_SECONDS:
                                    laser_smoother.reset()  # re-acquire: jump, don't glide
                                last_point_time = now
                            if following or (now - last_point_time) <= config.LASER_POINT_GRACE_SECONDS:
                                # Raw fingertip through the laser's own
                                # light filter - not the cursor's heavy one.
                                aim = laser_smoother.update(
                                    (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y), now)
                                laser_pos = map_to_screen(aim, desktop_ctrl.screen_size)
                        if laser_pointer and laser_pos is not None:
                            laser_pointer.update(laser_pos)
                        if laser_banner:
                            laser_banner.show()
                    else:
                        # Normal slide control (laser off).
                        if now - last_action_time > 0.6:  # prevent spam
                            if confirmed == "pointing":
                                send_presentation_key("next")
                                last_action_time = now
                            elif confirmed == "two_finger":
                                send_presentation_key("previous")
                                last_action_time = now
                            elif confirmed == "thumb_up" and now - last_action_time > 1.0:
                                send_presentation_key("start")
                                last_action_time = now
                            elif pinching and not presentation_was_pinching:
                                # One click per pinch (fires as it starts,
                                # not repeatedly while held).
                                send_presentation_key("click")
                                last_action_time = now
                    presentation_was_pinching = pinching

                elif mode == "media" and enabled:
                    if open_palm_tap and now - last_action_time > 0.5:
                        send_media_key("pause")
                        last_action_time = now

                    # Volume "knob": pinch and twist your wrist like
                    # turning a dial - clockwise raises it, counter-
                    # clockwise lowers it. Not gated by the shared
                    # last_action_time cooldown (that's for discrete
                    # poses to avoid spamming one action many times a
                    # second) - RotationDialDetector has its own much
                    # shorter per-tick cooldown, since a real turning
                    # motion should feel continuous, like an actual
                    # volume knob, not rate-limited like a button press.
                    dial_result = volume_dial.update(pinching, wrist_px, pinch_mid_px, now)
                    if dial_result:
                        clockwise = (dial_result == "clockwise") != config.DIAL_INVERT
                        if clockwise:
                            send_media_key("volume_up")
                        else:
                            send_media_key("volume_down")
                        dial_flash = ("UP" if clockwise else "DOWN", now + 0.35)

                    # Edge-hold (point to the left/right edge and pause
                    # there) drives seek forward/rewind - freed up here
                    # once Presentation mode moved its own prev/next off
                    # edge-hold and onto discrete poses.
                    if edge_nav and now - last_action_time > 0.5:
                        if edge_nav == "swipe_right":
                            send_media_key("seek_forward")
                            last_action_time = now
                        elif edge_nav == "swipe_left":
                            send_media_key("seek_backward")
                            last_action_time = now
                    if confirmed and now - last_action_time > 0.5:
                        if confirmed == "thumb_up":
                            send_media_key("play")
                            last_action_time = now
                        elif confirmed == "two_finger":
                            send_media_key("mute")
                            last_action_time = now
                        elif confirmed == "fist":
                            send_media_key("next_track")
                            last_action_time = now
                        elif confirmed == "pointing" and edge_zone_now is None:
                            # Without this guard, holding a point at the
                            # screen edge to seek (EdgeDwellDetector,
                            # above) ALSO continuously re-confirms the
                            # plain "pointing" gesture - since an
                            # edge-hold is, by definition, pointing -
                            # which fired prev_track at the same time,
                            # so the track would change out from under
                            # you the moment you tried to seek. Only
                            # treat a point as "previous track" when the
                            # fingertip isn't currently sitting in either
                            # edge zone.
                            send_media_key("prev_track")
                            last_action_time = now
                        # open_palm is now the dedicated "pause" TAP action
                        # (see open_palm_tap above), not a confirmed-gesture
                        # action here - it used to send a distinct "stop"
                        # key, which some media apps/OSes treat as ending
                        # the session outright rather than something
                        # play/pause can resume from. Pause and Play now
                        # both go through the same play/pause toggle, so
                        # resuming always works.

                elif mode == "drawing" and paint is not None:
                    # The whole panel (toolbar, colours, strokes) lives on
                    # the overlay and works in SCREEN space, fed the same
                    # map_to_screen point the reticle shows - see
                    # overlay_drawing.py. The cv2 window just says so
                    # instead of re-compositing the drawing every frame
                    # (that composite + the full-screen conversion was a
                    # big part of the ~13 fps this mode used to run at).
                    paint_pt = map_to_screen(overlay_norm, desktop_ctrl.screen_size) if overlay_norm else None
                    paint.update(paint_pt, pinching, now)
                    cv2.putText(frame, "Drawing is on your overlay", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                elif mode == "drawing":
                    # Fallback (no overlay available): the original
                    # in-window panel. index_norm is
                    # always set whenever a hand is visible (computed
                    # above), so the fingertip pixel position is available
                    # here regardless of whether it's currently pinching.
                    ix = int(index_norm[0] * config.FRAME_WIDTH)
                    iy = int(index_norm[1] * config.FRAME_HEIGHT)

                    # Button taps are hover-based (fingertip near a
                    # button), same as main.py - no pinch required. This
                    # call was missing entirely before, which is why
                    # "Show Drawing Panel", the pencil/eraser/clear/save/
                    # color buttons, and therefore drawing_mode itself
                    # (only ever set True by tapping the pencil button)
                    # never did anything: pinching would move the
                    # fingertip, but there was no path that could ever
                    # turn drawing_mode on in the first place.
                    drawing.handle_tap(
                        ix, iy, frame,
                        lambda: cv2.addWeighted(frame, 1, drawing.canvas, 0.5, 0),
                    )

                    if pinching:
                        drawing.handle_pinch("right", ix, iy)
                    else:
                        drawing.handle_unpinch("right")
                    drawing.draw(frame)
                    # Overlay the drawing canvas semi-transparently
                    frame = cv2.addWeighted(frame, 1, drawing.canvas, 0.5, 0)

                elif mode == "call":
                    # Held, STILL gestures (fist / peace / open palm) or a
                    # pinch on a call-bar button - never a bare gesture, so
                    # talking with your hands can't mute you (callapp.py).
                    call_pt = (map_to_screen(overlay_norm, desktop_ctrl.screen_size)
                               if overlay_norm and desktop_ctrl else None)
                    if enabled and not tut.active:
                        call_ring, fired = call_hold.update(rule_raw_gesture, call_pt, now)
                        if fired is None and pinching and not ui_prev_pinch:
                            fired = callapp.call_bar_button_at(call_bar_buttons, call_pt)
                        if fired:
                            res = call_ctl.perform(fired)
                            set_toast(res["message"], now)
                            print(f"[call] {res['message']}")
                    else:
                        call_hold.reset()
                elif mode == "game":
                    # Use the existing WhackAMoleGame API
                    game.update()
                    # A whack (and Start/Close) is a fresh PINCH, not just
                    # the pointer being near something: click = the frame
                    # the pinch closes.
                    game_click = bool(pinching and not game_prev_pinch)
                    game_prev_pinch = bool(pinching)
                    if index_norm:
                        ix = int(overlay_norm[0] * config.FRAME_WIDTH)
                        iy = int(overlay_norm[1] * config.FRAME_HEIGHT)
                        game.handle_tap(ix, iy, clicked=game_click)   # frame-space: the cv2 window
                        # Screen-space: the floating overlay (see
                        # WhackAMoleGame's own class docstring for why
                        # this needs the SAME map_to_screen the cursor
                        # dot uses, not a plain frame->screen stretch -
                        # that mismatch was the actual reason the mole
                        # could only ever be hit from inside the small
                        # cv2 window, never from the overlay you're
                        # actually meant to be looking at). Both calls
                        # share the same score/mole state - this isn't a
                        # second game, it's the same one, playable from
                        # either view.
                        if main_overlay_win is not None and desktop_ctrl is not None:
                            sx_pt, sy_pt = map_to_screen(overlay_norm, desktop_ctrl.screen_size)
                            game.handle_tap(sx_pt, sy_pt, *desktop_ctrl.screen_size, clicked=game_click)
                    game.draw(frame)
                    if not game.active and not game.countdown_active:
                        game.draw_start_button(frame)

                elif mode == "ar" and enabled and ar_overlay_win is not None and desktop_ctrl is not None:
                    # AR mode never touches the real mouse or keyboard -
                    # it only ever reads screen_pt to position charts on
                    # its OWN transparent overlay window, the same
                    # map_to_screen() every other mode uses to place its
                    # own reticle/laser dot. Driven by rule_raw_gesture/
                    # rule_confirmed (not the AI-hybrid engine) for the
                    # same reason Desktop mode is - continuous drag needs
                    # to feel instant, and this mode is exactly that.
                    # With the 3D layer active the cursor comes from the
                    # per-hand PINCH-STABLE pointer (it stops following the
                    # curling fingertip as you close a pinch, so you hit what
                    # you aimed at). Drawn cursor == grab point == 2D point.
                    states3d = None
                    cursor_approach = 0.0
                    if ctl3d is not None:
                        states3d = [pointers3d[k].update(slot_hands[k], now) for k in (0, 1)]
                        _prim = 0 if slot_hands[0] is not None else 1
                        screen_pt = states3d[_prim].pt
                        cursor_approach = states3d[_prim].approach
                    else:
                        screen_pt = map_to_screen(index_norm, desktop_ctrl.screen_size)
                    hover_kind = ar_overlay_win.palette_kind_at(screen_pt)
                    hover_file = ar_overlay_win.file_tile_at(screen_pt)
                    hover_shape = ar_overlay_win.shape_tile_at(screen_pt)
                    hover_function = ar_overlay_win.function_tile_at(screen_pt)

                    # 3D layer goes first: if it claims this pinch (a 3D
                    # object, a 3D tile, the 3D palette button, or any part
                    # of a two-hand gesture) the 2D chart logic below must
                    # not ALSO act on it. pinching_2d is what 2D sees.
                    claimed_3d = False
                    if ctl3d is not None:
                        claimed_3d = ctl3d.step(states3d, now, ar_overlay_win)
                    pinching_2d = pinching and not claimed_3d

                    if pinching_2d and not was_pinching_ar:
                        # Rising edge of a pinch - checked in order:
                        #  1. a file tile (only live while the file
                        #     browser is open) -> pick THAT file up as a
                        #     new chart, closing the browser.
                        #  2. the "Files" palette button -> just opens/
                        #     closes the file browser row; doesn't pick
                        #     up a chart itself, there's nothing to drag
                        #     until a specific file is chosen.
                        #  3. any other palette button -> pick up a
                        #     ghost of that chart type, as before.
                        #  4. otherwise -> grab whatever existing chart
                        #     is already there (deliberately does NOT
                        #     create a chart from a pinch on empty
                        #     canvas - that made it too easy to conjure
                        #     the wrong type by accident with no way to
                        #     choose which kind you'd get).
                        if hover_file is not None:
                            ar_overlay_win.start_ghost_from_file(hover_file, screen_pt)
                            grabbed_chart = None
                        elif hover_kind == "files":
                            ar_overlay_win.toggle_file_browser()
                            grabbed_chart = None
                        elif hover_kind == "clear":
                            ar_overlay_win.press_clear(now)      # arms, then clears on the 2nd pinch
                            grabbed_chart = None
                        elif hover_kind is not None:
                            ar_overlay_win.start_ghost(hover_kind, screen_pt)
                            grabbed_chart = None
                        else:
                            grabbed_chart = ar_overlay_win.chart_at(screen_pt)
                    elif pinching_2d and ar_overlay_win.ghost is not None:
                        ar_overlay_win.move_ghost(screen_pt)
                    elif pinching_2d and grabbed_chart is not None:
                        ar_overlay_win.move_chart(grabbed_chart, screen_pt)
                    elif not pinching_2d:
                        # Release: a ghost dropped back over the palette
                        # is a cancel (nothing created); dropped anywhere
                        # else, it becomes a real chart.
                        if ar_overlay_win.ghost is not None:
                            if ar_overlay_win.in_palette_zone(screen_pt):
                                ar_overlay_win.cancel_ghost()
                            else:
                                ar_overlay_win.commit_ghost()
                        grabbed_chart = None
                    was_pinching_ar = pinching_2d

                    # --- fist = delete -------------------------------------
                    # Two reasons this used to "sometimes not work":
                    #  1. It hit-tested only the fingertip's CURRENT spot.
                    #     Closing a fist curls the index fingertip in toward
                    #     the palm, i.e. AWAY from the graph you were
                    #     pointing at, so by the time the fist was confirmed
                    #     the test often missed. Now it also tests where the
                    #     fingertip was over the last ~0.8 s (ar_trail,
                    #     recorded only while the hand wasn't a fist).
                    #  2. It was one-shot: if a recent pinch made the 3D
                    #     layer refuse (see AR3DController._destructive_ok)
                    #     the gesture was simply spent. Now a confirmed fist
                    #     keeps retrying for ~1 s while the fist is held.
                    if rule_raw_gesture != "fist":
                        ar_trail.append((now, screen_pt))
                    if rule_confirmed == "fist" and now - last_action_time > 0.4:
                        fist_retry_until = now + 1.0
                        last_action_time = now
                    if now < fist_retry_until and rule_raw_gesture == "fist":
                        fist_done = False
                        for cand in [screen_pt] + [p for t_, p in reversed(ar_trail) if now - t_ < 0.8]:
                            res = ctl3d.delete_at(cand, now) if ctl3d is not None else None
                            if res == "removed":
                                fist_done = True
                                break
                            if res == "blocked":
                                break                  # object found, retry next frame
                            target = ar_overlay_win.chart_at(cand)
                            if target is not None:
                                ar_overlay_win.remove_chart(target)
                                if target is grabbed_chart:
                                    grabbed_chart = None
                                fist_done = True
                                break
                        if fist_done:
                            fist_retry_until = 0.0
                    if rule_confirmed == "two_finger" and now - last_action_time > 0.4:
                        # Mock chart -> re-randomizes. Live chart ->
                        # nothing to randomize, cycles to the next real
                        # metric instead. File chart -> re-reads the CSV
                        # from disk, so editing the file and two-fingering
                        # the chart is a live-reload. Manually typing
                        # exact values by hand is still a good next step
                        # but stays out of scope for this pass.
                        target = ar_overlay_win.chart_at(screen_pt)
                        if ctl3d is not None and ctl3d.refresh_at(screen_pt, now):
                            pass                      # 3D: new data / next surface, warp reset
                        elif target is not None:
                            ar_overlay_win.refresh_chart_data(target)
                        last_action_time = now

                    # Feeds every currently-placed (and mid-pick-up) Live
                    # chart this frame's real fps/gesture-count numbers -
                    # has to happen before update() so the easing in
                    # step_toward_target() sees the fresh target.
                    ar_overlay_win.update_live_data(fps_history, gesture_counts)
                    ar_overlay_win.update(grabbed_chart=grabbed_chart, hover_kind=hover_kind,
                                          cursor_point=screen_pt, pinching=pinching,
                                          hover_file=hover_file, hover_shape=hover_shape,
                                          hover_function=hover_function,
                                          cursor_approach=cursor_approach)

                # Global toggle (works in all modes except while the mode
                # menu is open - there thumb-down cancels the menu
                # instead, handled above). Driven by rule_confirmed
                # (always computed above, cheaply, from
                # classify_static_gesture - no AI inference needed) rather
                # than the AI-hybrid `confirmed`, which is now only
                # populated in Presentation/Media mode - using it here
                # would silently break the toggle in Desktop/Drawing/Game.
                if not menu_open and rule_confirmed == "thumb_down" and interaction is not None:
                    # Already handled inside DesktopInteraction for desktop mode
                    if mode != "desktop":
                        enabled = not enabled
                        print(f"Control {'ON' if enabled else 'OFF'}")

            else:
                if mode in AI_ENGINE_MODES:
                    ai_engine.reset()
                desktop_stabilizer.reset()
                index_smoother.reset()
                thumb_smoother.reset()
                wrist_smoother.reset()
                hand_scale_smoother.reset()
                pinch_ratio_smoother.reset()
                mcp_smoother.reset()
                presentation_was_pinching = False
                pinch_raw_state = False
                edge_dwell.reset()
                palm_hold_start = None
                menu_was_pinching = False
                fist_latched = False
                if laser_active:
                    # Toggle stays on with the hand out of view; keep the
                    # dot where it was and the overlay windows serviced.
                    if laser_pointer and laser_pos is not None:
                        laser_pointer.update(laser_pos)
                    if laser_banner:
                        laser_banner.pump()
                volume_dial.reset()
                dial_flash = None
                grabbed_chart = None
                was_pinching_ar = False
                if ctl3d is not None:
                    # Hand(s) gone: release whatever 3D object was held.
                    for _p in pointers3d:
                        _p.reset()
                    ctl3d.step(_no_hand_states, now, ar_overlay_win)
                # Keep pumping the AR overlay's Tkinter event loop even
                # with no hand detected (a brief tracking dropout
                # shouldn't leave a full-screen always-on-top window
                # unresponsive to the OS while it waits for the hand to
                # come back) - just without moving anything, since
                # grabbed_chart is already cleared above.
                if mode == "ar" and ar_overlay_win is not None and enabled:
                    ar_overlay_win.update(grabbed_chart=None)
                if interaction:
                    interaction.reset()

            # Draw HUD. In desktop mode, show the same rule-based
            # gesture that's actually driving the mouse/keyboard (see
            # rule_raw_gesture/rule_confirmed above) rather than the
            # AI-hybrid engine's reading - otherwise the HUD can show one
            # gesture while a different one is actually clicking/
            # dragging, which is exactly the "out of sync" feeling.
            if mode in AI_ENGINE_MODES:
                hud_gesture = confirmed or raw_gesture
                hud_confidence = confidence
                hud_ai_label = ai_debug[0] if ai_debug else None
            else:
                # Desktop/Drawing/Game don't run the AI engine (see
                # AI_ENGINE_MODES above), so show the same rule-based
                # reading that's actually driving them instead - keeps
                # the HUD honest about what's really being acted on
                # rather than showing a stale/AI value from a different
                # mode, or nothing at all.
                hud_gesture = rule_confirmed if lm_list else None
                hud_gesture = hud_gesture or (rule_raw_gesture if lm_list else None)
                hud_confidence = 1.0 if hud_gesture else 0.0
                hud_ai_label = None

            perf_help_ok = perf.settings["help"]
            hud_show_help = show_help and (perf_help_ok or force_help_override)
            help_hidden_note = (
                "Help hidden to save performance - press H to show it anyway"
                if show_help and not hud_show_help else None
            )
            draw_hud(
                frame, mode, enabled,
                hud_gesture,
                hud_confidence, fps, hud_show_help,
                ai_label=hud_ai_label,
                help_hidden_note=help_hidden_note,
            )
            if perf.level:
                cv2.putText(frame, f"PERF L{perf.level}", (frame.shape[1] - 250, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 190, 255), 2, cv2.LINE_AA)
            if menu_open:
                draw_mode_menu(frame, menu_highlight)
            if laser_active and index_norm is not None:
                draw_laser_dot(frame, index_norm)

            # Live swipe-speed meter (Desktop only now - Presentation and
            # Media both moved off swipe entirely) and the edge-hold zone
            # overlay (Media's seek gesture) - only while the mode menu
            # isn't covering the screen anyway.
            if not menu_open and enabled:
                mode_color = MODES[mode]["color"]
                if mode == "desktop":
                    draw_swipe_meter(frame, desktop_stabilizer.last_swipe_velocity,
                                      config.SWIPE_VELOCITY_THRESHOLD, mode_color)
                    draw_swipe_meter(frame, desktop_stabilizer.last_swipe_velocity_y,
                                      config.SWIPE_VERTICAL_VELOCITY_THRESHOLD, mode_color,
                                      y1=150, hint="up/down", arrows=("UP", "DOWN"))
                elif mode == "media":
                    draw_edge_zones(frame, edge_dwell, now, mode_color)
                    draw_dial_progress(frame, volume_dial)
                    if dial_flash and now < dial_flash[1]:
                        direction, _ = dial_flash
                        label = f"VOLUME {direction}"
                        label_color = (60, 220, 90) if direction == "UP" else (60, 140, 230)
                        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                        _text_with_shadow(frame, label, ((frame.shape[1] - tw) // 2, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.85, label_color, 2)

            if overlay_norm is None:
                overlay_stable = PinchStableCursor()

            # Reticle for desktop mode - MainOverlay's status drawer
            # already shows enabled/gesture/FPS for every mode (the old
            # small per-mode `Overlay` box was redundant with it and is
            # retired - see the import comment near the top of this file).
            if reticle and mode == "desktop" and index_norm and enabled and desktop_ctrl:
                screen_pt = map_to_screen(index_norm, desktop_ctrl.screen_size)
                reticle.update(screen_pt, clicked=(now < click_flash))

            # Camera preview thumbnail (bottom-right) - OFF by default
            # (see camera_preview_visible below): it's a troubleshooting
            # aid ("is my hand actually being seen"), not part of the
            # everyday overlay, so it no longer shows up unasked-for.
            # Toggle with 'c' any time tracking looks off and you want to
            # check the raw feed.
            if camera_preview is not None and camera_preview_visible:
                camera_preview.update(frame)

            # Main HUD/menu overlay - floats the same information as
            # draw_hud/draw_mode_menu (just drawn) over the real desktop
            # instead of confined to this cv2 window, as small corner
            # drawers that expand on hover (see main_overlay.py). Media
            # mode also gets a generic cursor dot, the edge-hold zone
            # bars, and the volume-knob flash text, since it has no
            # dedicated indicator of its own the way Desktop (reticle)
            # and Presentation (laser) do.
            if mode != prev_mode:
                if mode == "call" and not enabled:
                    # Call mode does nothing while hand control is OFF (the
                    # app starts with it OFF, and nothing said so) - choosing
                    # Call mode IS asking for it, so switch it on. Thumb Down
                    # (or the Control button) still turns it off any time.
                    set_control(True)
                    set_toast("Hand control ON for Video Call (thumb down turns it off)", now)
                prev_mode = mode
            ui_click = bool(pinching and not ui_prev_pinch)
            ui_prev_pinch = bool(pinching)
            if mode != "call":
                call_hold.reset()

            _hp = None
            # ---- tutorial (on demand only: T key or the help panel's button) ----
            if main_overlay_win is not None and desktop_ctrl is not None:
                _hp = (map_to_screen(overlay_norm, desktop_ctrl.screen_size)
                       if overlay_norm else None)
                if ui_click and not tut.active and _hp is not None:
                    _r = main_overlay_win.help_button_rect()
                    if _r and _r[0] <= _hp[0] <= _r[2] and _r[1] <= _hp[1] <= _r[3]:
                        tut.start()
                        enabled_before_tutorial = enabled
                        enabled = False       # lessons must not click your real desktop
                if ui_click and not tut.active and _hp is not None:
                    for _name, _r in main_overlay_win.status_button_rects().items():
                        if _r[0] <= _hp[0] <= _r[2] and _r[1] <= _hp[1] <= _r[3]:
                            if _name == "camera":
                                release_camera()
                            else:
                                set_control(not enabled)
                                set_toast(f"Hand control {'ON' if enabled else 'OFF'}", now)
                if tut.active:
                    tut.update({"pointer": _hp, "click": ui_click,
                                "help_open": getattr(main_overlay_win, "_help_open", 0.0) > 0.6,
                                "menu_open": menu_open}, now)
                if not tut.active and enabled_before_tutorial is not None:
                    enabled = enabled_before_tutorial
                    enabled_before_tutorial = None
                    if tut.finished and not prefs["tutorial_done"]:
                        prefs.set("tutorial_done", True)
                # Hide the HUD from screen sharing - except in Drawing mode,
                # where the strokes ARE what the audience should see.
                _ss = prefs["share_safe"]
                main_overlay_win.set_capture_excluded(
                    _ss == "on" or (_ss == "auto" and mode != "drawing"))

            if mode == "call" and now - call_status_at > 0.5:
                call_status_at = now
                _app = callapp.detect_call_app(callapp.foreground_title())
                call_status = ((True, f"{callapp.APP_NAMES[_app]} is in front - ready") if _app else
                               (False, "Bring your call window to the front"))
                if not enabled:
                    call_status = (False, "Hand control is OFF - thumb down turns it on")

            if main_overlay_win is not None:
                overlay_extras = {"camera_on": camera_on}
                if toast_text and now < toast_until:
                    overlay_extras["toast"] = toast_text
                if tut.active:
                    _tv = tut.view()
                    _tv["hover"] = tut.button_at(_hp)
                    overlay_extras["tutorial"] = _tv
                if mode == "call":
                    overlay_extras["call"] = {
                        "buttons": call_bar_buttons, "status": call_status[1], "ok": call_status[0],
                        "hover": callapp.call_bar_button_at(call_bar_buttons, _hp if desktop_ctrl else None)}
                    if call_ring:
                        overlay_extras["ring"] = call_ring
                overlay_cursor_pt = None
                overlay_edge_info = None
                overlay_flash = None
                overlay_drawing_canvas = None
                overlay_ink = overlay_paint = None
                overlay_game_state = None
                # Where the tracked hand currently maps to on the real
                # screen, computed for EVERY mode (not just the ones with
                # their own cursor dot below) - this is what lets the
                # status/help drawers (see main_overlay.MainOverlay)
                # expand when you point at them, not just when a mouse
                # happens to hover there. A floating HUD that could only
                # be opened with a mouse defeated the point of a
                # hands-driven overlay: you'd need a free hand on the
                # mouse to ever see the help it's hiding.
                overlay_hand_pt = (map_to_screen(overlay_norm, desktop_ctrl.screen_size)
                                   if overlay_norm and desktop_ctrl else None)
                if mode == "media" and enabled and index_norm and desktop_ctrl:
                    overlay_cursor_pt = None   # the always-on reticle covers this
                    overlay_edge_info = (edge_dwell.edge_ratio,) + edge_dwell.progress(now)
                    if dial_flash and now < dial_flash[1]:
                        direction, _ = dial_flash
                        flash_color = (60, 220, 90) if direction == "UP" else (60, 140, 230)
                        overlay_flash = (f"VOLUME {direction}", flash_color)
                elif mode == "drawing" and paint is not None:
                    overlay_ink, overlay_paint = paint.ink, paint.view(now)
                elif mode == "drawing":
                    # The whiteboard content itself (see
                    # main_overlay._draw_drawing_canvas's own docstring
                    # for why this is a raw pixel buffer rather than a
                    # list of strokes) - this used to only ever show up
                    # inside the cv2 preview window, invisible once
                    # you're actually looking at the floating overlay
                    # instead.
                    overlay_drawing_canvas = drawing.canvas
                elif mode == "game":
                    # Whack-a-Mole has no dedicated cursor of its own the
                    # way Desktop (reticle) and Presentation (laser) do,
                    # so without this there was nothing on the real
                    # desktop showing where your point was actually
                    # landing - "I don't even see where I'm pointing at"
                    # - which also made the Start/Close buttons drawn by
                    # _draw_whackamole effectively unaimable.
                    overlay_cursor_pt = None   # the always-on reticle covers this
                    # Reads WhackAMoleGame's own screen-space accessors
                    # directly (bounds_px/mole_pos_px/button_px/...,
                    # given the real screen size) instead of stretching
                    # the frame-pixel positions by screen/frame ratio -
                    # that stretch was the actual bug behind hits only
                    # registering inside the cv2 window: it doesn't match
                    # map_to_screen's margin-cropped mapping, so the
                    # overlay's drawing and the real hit-test geometry
                    # (now both driven from the same normalized state -
                    # see the WhackAMoleGame class docstring) were subtly
                    # different rectangles. Not an issue any more since
                    # there's nothing left to keep in sync by hand.
                    screen_w, screen_h = desktop_ctrl.screen_size if desktop_ctrl else (1, 1)
                    bx0, by0, bx1, by1 = game.bounds_px(screen_w, screen_h)
                    close_pos, close_size = game.close_button_px(screen_w, screen_h)
                    start_pos, start_size = game.button_px(screen_w, screen_h)
                    overlay_game_state = {
                        "countdown_active": game.countdown_active,
                        "countdown_remaining": (
                            config.COUNTDOWN_DURATION_S - int(now - game.countdown_start_time)
                            if game.countdown_active else 0),
                        "active": game.active,
                        "mole": game.mole_pos_px(screen_w, screen_h),
                        "mole_radius": game.mole_radius_px(screen_w, screen_h),
                        "hit_flash": game._hit_pos is not None and now < game._hit_until,
                        "score": game.score,
                        "bounds": (bx0, by0, bx1, by1),
                        "close_button": (close_pos[0], close_pos[1],
                                         close_pos[0] + close_size[0], close_pos[1] + close_size[1]),
                        "start_button": (start_pos[0], start_pos[1],
                                         start_pos[0] + start_size[0], start_pos[1] + start_size[1]),
                    }
                try:
                    main_overlay_win.update(
                        mode, MODES, MODE_ORDER, enabled, hud_gesture, hud_confidence, fps,
                        hud_show_help, ai_label=hud_ai_label, menu_open=menu_open,
                        menu_highlight=menu_highlight, cursor_point=overlay_cursor_pt,
                        pinching=bool(pinching),
                        edge_info=overlay_edge_info, flash=overlay_flash,
                        help_hidden_note=help_hidden_note,
                        drawing_canvas=overlay_drawing_canvas,
                        frame_size=(config.FRAME_WIDTH, config.FRAME_HEIGHT),
                        game_state=overlay_game_state,
                        hand_point=overlay_hand_pt,
                        # Own-cursor cases: AR mode (AROverlay draws one) and
                        # Desktop mode while control is ON (CursorReticle).
                        # Everywhere else - including Desktop with control
                        # OFF, which is exactly when you most need to see
                        # what you're pointing at - this draws it.
                        reticle=(overlay_hand_pt is not None and mode != "ar"
                                 and not (mode == "desktop" and enabled)),
                        ink=overlay_ink, paint=overlay_paint, extras=overlay_extras,
                    )
                except Exception as e:
                    # A rendering problem in the floating overlay (a
                    # Tkinter/Pillow quirk, a platform-specific glitch,
                    # etc.) must never be able to take the whole hand-
                    # tracking session down with it the way one just did
                    # (see main_overlay._draw_drawing_canvas's own note
                    # on the ImageTk "wrong interpreter" crash this
                    # guards against). Logged once per occurrence rather
                    # than silently swallowed, and the loop carries on -
                    # the cv2 preview window keeps working either way.
                    print(f"[WARN] Main overlay draw failed this frame: {e}")

            cv2.imshow("Hand Tracker – Showcase (AI + Real-World Control)", frame)

            # Adaptive quality: feed this frame's compute time (camera wait
            # excluded) and apply whatever level it settles on.
            if perf.update((time.perf_counter() - t_compute0) * 1000.0, now):
                print(f"[PERF] quality level {perf.level} "
                      f"(compute EMA {perf.ema:.0f} ms vs budget {perf.budget_ms:.0f} ms)")
            _lv = perf.settings
            if camera_preview is not None:
                camera_preview.every = _lv["preview_every"]
            if ctl3d is not None:
                ar_overlay_win.scene3d.item_budget = _lv["item_budget"]

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("u") and ctl3d is not None:
                if ar_overlay_win.scene3d.undo_remove():
                    print("[AR] restored the last removed 3D object")
            elif key == ord("h"):
                # Toggle against what's actually ON SCREEN right now
                # (hud_show_help), not the raw show_help flag - that's
                # what makes this always do what the keypress visibly
                # shows, even while the perf governor is involved.
                if hud_show_help:
                    show_help = False
                    force_help_override = False
                else:
                    show_help = True
                    force_help_override = True  # win out over the perf governor
            elif key == ord("c"):
                mode = "call"
                show_help = True
            elif key == ord("k"):
                release_camera()
            elif key == ord("t"):
                tut.toggle()
                if tut.active:
                    enabled_before_tutorial = enabled
                    enabled = False           # lessons must not click your real desktop
            elif key == ord("v"):
                if vcam.active or vcam_want:
                    vcam.stop()
                    vcam_want, vcam_tried = False, True
                    prefs.set("virtual_cam", False)
                    set_toast("Virtual camera OFF", time.time())
                else:
                    vcam_want, vcam_tried = True, False     # starts on the next frame
                    prefs.set("virtual_cam", True)
            elif key == ord("o"):
                _new = prefs.cycle("share_safe")
                set_toast({"auto": "HUD hidden from screen share (except Drawing)",
                           "on": "HUD always hidden from screen share",
                           "off": "HUD visible in screen share"}[_new], time.time())
            elif key in (ord("1"), ord("d")):
                mode = "desktop"
                show_help = True
            elif key in (ord("2"), ord("p")):
                mode = "presentation"
                show_help = True
            elif key in (ord("3"), ord("m")):
                mode = "media"
                show_help = True
            elif key in (ord("4"), ord("w")):
                mode = "drawing"
                show_help = True
            elif key in (ord("5"), ord("g")):
                mode = "game"
                show_help = True
            elif key == ord("7"):
                mode = "call"
                show_help = True
            elif key in (ord("6"), ord("a")):
                mode = "ar"
                show_help = True

    finally:
        if interaction:
            interaction.reset()
        vcam.stop()
        global_hk.stop()
        if main_overlay_win:
            main_overlay_win.close()
        if reticle:
            reticle.close()
        if laser_pointer:
            laser_pointer.close()
        if laser_banner:
            laser_banner.close()
        if ar_overlay_win:
            ar_overlay_win.close()
        if camera_preview:
            camera_preview.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Showcase closed.")


if __name__ == "__main__":
    main()
