"""
features/drawing.py
--------------------
Pinch-to-draw canvas + control panel (drawing mode, eraser, clear,
save, color picker). Same behavior as the original prototype, just
packaged as a class instead of a pile of globals so it can be toggled
on/off cleanly alongside the other features and, later, the gesture
engine's mode system (Milestone 7).
"""
import time

import cv2
import numpy as np

import config
from gestures import TapDebouncer, is_point_in_box

BUTTON_W, BUTTON_H = 200, 100


class DrawingPanel:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        self.drawing_mode = False
        self.eraser_mode = False
        self.panel_visible = False
        self._last_point_by_hand = {}
        self.selected_color_name = config.DEFAULT_DRAW_COLOR
        self._tap_debouncer = TapDebouncer()

        self.drawing_icon = self._load_icon(config.PENCIL_ICON)
        self.eraser_icon = self._load_icon(config.ERASER_ICON)

        self._color_button_positions = {
            "red": (1050, 400),
            "green": (1050, 500),
            "blue": (1050, 600),
            "yellow": (1150, 400),
            "white": (1150, 500),
        }
        self._color_button_size = (50, 50)
        self._show_panel_button = (frame_width - 220, 45, 210, BUTTON_H)

    @staticmethod
    def _load_icon(path):
        icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            raise FileNotFoundError(f"Could not load icon at {path}")
        if icon.shape[2] == 4:
            icon = cv2.cvtColor(icon, cv2.COLOR_BGRA2BGR)
        return cv2.resize(icon, (BUTTON_W, BUTTON_H))

    @property
    def selected_color_bgr(self):
        return config.COLORS_BGR[self.selected_color_name]

    def reset_canvas(self):
        self.canvas = np.zeros_like(self.canvas)

    def handle_pinch(self, hand_id, index_x, index_y):
        """Called every frame the given hand is pinching. Each hand keeps
        its own last-point, so two hands drawing at once produce two
        independent strokes instead of one line jumping between hands."""
        if self.drawing_mode:
            last_point = self._last_point_by_hand.get(hand_id)
            if last_point is not None:
                cv2.line(self.canvas, last_point, (index_x, index_y),
                          self.selected_color_bgr, config.DRAWING_LINE_THICKNESS)
            self._last_point_by_hand[hand_id] = (index_x, index_y)
        elif self.eraser_mode:
            cv2.circle(self.canvas, (index_x, index_y), config.ERASER_RADIUS, (0, 0, 0), -1)

    def handle_unpinch(self, hand_id):
        self._last_point_by_hand.pop(hand_id, None)

    def handle_hand_lost(self, hand_id):
        self._last_point_by_hand.pop(hand_id, None)

    def handle_tap(self, x, y, frame, frame_with_canvas_getter):
        """Check whether (x, y) hit a panel button; frame_with_canvas_getter
        is a zero-arg callable returning the current composited frame, used
        only when Save is pressed."""
        if not self._tap_debouncer.ready(time.time()):
            return

        if self.panel_visible:
            if is_point_in_box(x, y, 10, 50, BUTTON_W, BUTTON_H):
                self.drawing_mode, self.eraser_mode = True, False
            elif is_point_in_box(x, y, 220, 50, BUTTON_W, BUTTON_H):
                self.eraser_mode, self.drawing_mode = True, False
            elif is_point_in_box(x, y, 430, 50, BUTTON_W, BUTTON_H):
                self.reset_canvas()
            elif is_point_in_box(x, y, 640, 50, BUTTON_W, BUTTON_H):
                cv2.imwrite("drawing.png", frame_with_canvas_getter())
                print("Drawing saved to drawing.png")
            elif is_point_in_box(x, y, 850, 50, BUTTON_W, BUTTON_H):
                self.panel_visible = False

            for color, pos in self._color_button_positions.items():
                if is_point_in_box(x, y, pos[0], pos[1], *self._color_button_size):
                    self.selected_color_name = color
        else:
            bx, by, bw, bh = self._show_panel_button
            if is_point_in_box(x, y, bx, by, bw, bh):
                self.panel_visible = True

    def draw(self, frame):
        if self.panel_visible:
            self._draw_panel(frame)
        else:
            self._draw_show_panel_button(frame)

    def _draw_panel(self, frame):
        cv2.rectangle(frame, (0, 0), (1280, 150), (200, 200, 200), -1)
        cv2.putText(frame, "Control Panel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        frame[50:50 + BUTTON_H, 10:10 + BUTTON_W] = self.drawing_icon
        frame[50:50 + BUTTON_H, 220:220 + BUTTON_W] = self.eraser_icon

        cv2.rectangle(frame, (430, 50), (BUTTON_W + 430, 50 + BUTTON_H), (0, 0, 255), -1)
        cv2.putText(frame, "Clear Canvas", (440, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.rectangle(frame, (640, 50), (BUTTON_W + 640, 50 + BUTTON_H), (0, 255, 255), -1)
        cv2.putText(frame, "Save", (650, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.rectangle(frame, (850, 50), (BUTTON_W + 850, 50 + BUTTON_H), (255, 255, 0), -1)
        cv2.putText(frame, "Close Panel", (860, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        for color, pos in self._color_button_positions.items():
            color_bgr = config.COLORS_BGR[color]
            cv2.rectangle(frame, pos, (pos[0] + 50, pos[1] + 50), color_bgr, -1)
            if self.selected_color_name == color:
                cv2.rectangle(frame, pos, (pos[0] + 50, pos[1] + 50), (255, 255, 255), 2)

    def _draw_show_panel_button(self, frame):
        bx, by, bw, bh = self._show_panel_button
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (127, 5, 45), -1)
        cv2.putText(frame, "Show Drawing", (bx + 20, by + 45), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (255, 145, 200), 2)
        cv2.putText(frame, "Panel", (bx + 30, by + 80), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (255, 145, 200), 2)
