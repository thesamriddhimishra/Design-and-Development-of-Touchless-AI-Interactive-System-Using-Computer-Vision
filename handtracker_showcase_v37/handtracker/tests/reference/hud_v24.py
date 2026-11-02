"""v24 HUD drawing, frozen as the reference the optimised versions are compared against."""
import cv2, numpy as np
MODES = {}
MODE_ORDER = []
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

def draw_hud(frame, mode_key, enabled, gesture, confidence, fps, show_help, ai_label=None):
    """Clean, modern on-screen HUD."""
    h, w = frame.shape[:2]
    mode = MODES[mode_key]
    color = mode["color"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    _corner_brackets(frame, color)

    # Top bar: dark glass panel with a mode-colored accent underline
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 78), (18, 18, 22), -1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
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
    bottom = frame.copy()
    cv2.rectangle(bottom, (0, h - 42), (w, h), (16, 16, 20), -1)
    cv2.addWeighted(bottom, 0.6, frame, 0.4, 0, frame)
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

        shade = frame.copy()
        _rounded_rect(shade, (panel_x1, panel_y1), (panel_x2, panel_y2), (22, 22, 26), -1, radius=14)
        cv2.addWeighted(shade, 0.85, frame, 0.15, 0, frame)
        _rounded_rect(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), color, 2, radius=14)
        cv2.line(frame, (panel_x1 + 16, panel_y1 + 40), (panel_x2 - 16, panel_y1 + 40), (70, 70, 78), 1, cv2.LINE_AA)
        cv2.putText(frame, title_txt, (panel_x1 + 16, panel_y1 + 28),
                    font, 0.58, color, 2, cv2.LINE_AA)
        for i, line in enumerate(mode["help"]):
            ly = panel_y1 + 64 + i * line_h
            cv2.circle(frame, (panel_x1 + 22, ly - 5), 3, color, -1, cv2.LINE_AA)
            cv2.putText(frame, line, (panel_x1 + 34, ly),
                        font, 0.52, (225, 225, 230), 1, cv2.LINE_AA)

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
