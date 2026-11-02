"""
features/game.py
-----------------
Whack-a-mole mini-game. Same rules as the original prototype: point at
the mole with your index finger inside the spawn box to score, mole
respawns randomly, countdown before start.
"""
import random
import time

import cv2
import pygame

import config
from gestures import is_near_point, is_point_in_box


class WhackAMoleGame:
    """One game, playable from two different spaces at once: the small
    cv2 preview window (frame-pixel space, the original behaviour,
    completely unchanged) and the floating desktop overlay (screen-pixel
    space - see showcase.py's Game-mode block and main_overlay.
    _draw_whackamole). Both read/write the SAME score, mole position and
    active/countdown state - there's one game, not two - so a hit
    registered from either view is a real hit everywhere.

    This only works because all the geometry (mole spawn bounds, Start/
    Close buttons) is stored NORMALIZED (as fractions of a width/height,
    not fixed pixel numbers) and only ever turned into actual pixels via
    the *_px()/handle_tap() methods below, each of which takes the
    (width, height) of whichever space is asking. Frame-space callers
    (handle_tap(x, y) with no width/height, draw(frame)) keep the exact
    original frame_width/frame_height behaviour - this is an additive
    change, not a breaking one, for main.py and the cv2 window.

    Previously, screen-space callers used a separate hack in showcase.py
    that took the frame-pixel positions below and uniformly stretched
    them by screen_size/frame_size. That LOOKS equivalent but isn't: the
    fingertip's actual on-screen position (main_overlay's cursor dot,
    computed via desktop_control.map_to_screen) goes through an
    ergonomic margin-crop, not a uniform stretch, so the stretched mole
    and the real cursor dot were using two subtly different coordinate
    mappings - close enough to look right, not close enough to actually
    land a hit. Normalized geometry denormalized directly to screen
    pixels removes that mismatch instead of trying to compensate for it.
    """

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.active = False
        self.countdown_active = False
        self.countdown_start_time = None
        self.score = 0

        self.mole_image = self._load_mole_image()
        self.start_button_image = self._load_start_button_image()
        self.hit_effect_image = self._load_hit_effect_image()
        self.hit_sound = pygame.mixer.Sound(config.HIT_SOUND_FILE)

        self._hit_pos = None
        self._hit_until = 0.0

        # Normalized (0-1) rects - the original frame-pixel constants
        # (1050,200)/(210,90)/etc divided by the original 1280x720 frame
        # they were tuned against, so *_px(frame_width, frame_height)
        # reproduces the exact original pixel positions bit for bit.
        self._button_frac = (1050 / 1280, 200 / 720, 210 / 1280, 90 / 720)
        self._close_button_frac = (1050 / 1280, 300 / 720, 210 / 1280, 90 / 720)
        bx0, by0, bx1, by1 = config.MOLE_SPAWN_BOUNDS
        self._bounds_frac = (bx0 / 1280, by0 / 720, bx1 / 1280, by1 / 720)
        self._mole_radius_frac = config.MOLE_RADIUS / 1280

        self._mole_frac = self._random_mole_frac()
        self._last_mole_time = time.time()

    # -- normalized geometry, denormalized to an arbitrary (width, height) --
    def bounds_px(self, width=None, height=None):
        width, height = width or self.frame_width, height or self.frame_height
        x0, y0, x1, y1 = self._bounds_frac
        return (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))

    def button_px(self, width=None, height=None):
        width, height = width or self.frame_width, height or self.frame_height
        x, y, w, h = self._button_frac
        return (int(x * width), int(y * height)), (int(w * width), int(h * height))

    def close_button_px(self, width=None, height=None):
        width, height = width or self.frame_width, height or self.frame_height
        x, y, w, h = self._close_button_frac
        return (int(x * width), int(y * height)), (int(w * width), int(h * height))

    def mole_radius_px(self, width=None, height=None):
        width = width or self.frame_width
        return max(12, int(self._mole_radius_frac * width))

    def mole_pos_px(self, width=None, height=None):
        width, height = width or self.frame_width, height or self.frame_height
        fx, fy = self._mole_frac
        return (int(fx * width), int(fy * height))

    # Frame-pixel-space accessors (unchanged external shape/behaviour -
    # main.py, showcase.py's cv2 drawing path, and draw()/draw_start_
    # button() below all keep reading these exactly as before).
    @property
    def mole_pos(self):
        return self.mole_pos_px()

    @property
    def button_pos(self):
        return self.button_px()[0]

    @property
    def button_size(self):
        return self.button_px()[1]

    @property
    def close_button_pos(self):
        return self.close_button_px()[0]

    @property
    def close_button_size(self):
        return self.close_button_px()[1]

    @property
    def _xmin(self):
        return self.bounds_px()[0]

    @property
    def _ymin(self):
        return self.bounds_px()[1]

    @property
    def _xmax(self):
        return self.bounds_px()[2]

    @property
    def _ymax(self):
        return self.bounds_px()[3]

    def _load_mole_image(self):
        img = cv2.imread(config.MOLE_IMAGE, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load {config.MOLE_IMAGE}")
        return cv2.resize(img, (config.MOLE_RADIUS * 2, config.MOLE_RADIUS * 2))

    def _load_start_button_image(self):
        img = cv2.imread(config.START_BUTTON_IMAGE, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load {config.START_BUTTON_IMAGE}")
        if img.shape[2] == 3:
            alpha = 255 * (img.sum(axis=2) > 0).astype("uint8")
            img = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha])
        return cv2.resize(img, (210, 90))

    def _load_hit_effect_image(self):
        img = cv2.imread(config.HIT_EFFECT_IMAGE, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load {config.HIT_EFFECT_IMAGE}")
        size = config.MOLE_RADIUS * 3
        return cv2.resize(img, (size, size))

    def _random_mole_frac(self):
        x0, y0, x1, y1 = self._bounds_frac
        r = self._mole_radius_frac
        rh = r * (self.frame_width / self.frame_height) if self.frame_height else r
        return (
            random.uniform(x0 + r, max(x0 + r, x1 - r)),
            random.uniform(y0 + rh, max(y0 + rh, y1 - rh)),
        )

    def start_countdown(self):
        self.countdown_active = True
        self.countdown_start_time = time.time()
        self.score = 0
        self._last_mole_time = time.time()

    def stop(self):
        self.active = False
        self.countdown_active = False

    def handle_tap(self, x, y, width=None, height=None, clicked=True):
        """(x, y) in the SAME space as (width, height) - frame pixels if
        both are omitted (the original behaviour: main.py, and
        showcase.py's own cv2-window game interaction), or whatever
        space (x, y) was computed in otherwise (showcase.py passes the
        real screen size here for overlay-driven play, so (x, y) should
        be the screen-mapped fingertip - see map_to_screen - not a raw
        frame-pixel point)."""
        if not clicked:
            # Showcase passes clicked=False on every frame the player is
            # NOT making a fresh pinch: bringing the pointer near the mole
            # (or brushing over Start/Close) must not count - a whack is a
            # deliberate pinch, like a mouse click. Defaults to True so
            # main.py's original hover-to-play behaviour is unchanged.
            return
        width, height = width or self.frame_width, height or self.frame_height
        btn_pos, btn_size = self.button_px(width, height)
        close_pos, close_size = self.close_button_px(width, height)
        if not self.active and is_point_in_box(x, y, *btn_pos, *btn_size):
            self.start_countdown()
        if self.active and is_point_in_box(x, y, *close_pos, *close_size):
            self.stop()
        # threshold scales with the canvas (not a flat 50px): a fixed
        # 50px hit radius made sense for a 1280px-wide frame but would
        # be a tiny, hard-to-hit target on a screen that's 1920px+ wide.
        if self.active and is_near_point(x, y, *self.mole_pos_px(width, height),
                                         threshold=self.mole_radius_px(width, height) * 1.5):
            self._register_hit()

    def _register_hit(self):
        self.score += 1
        self.hit_sound.play()
        self._hit_pos = self.mole_pos
        self._hit_until = time.time() + config.HIT_EFFECT_DURATION_S
        self._mole_frac = self._random_mole_frac()
        self._last_mole_time = time.time()

    def update(self):
        """Advance countdown -> active state. Call once per frame."""
        if not self.countdown_active:
            return
        elapsed = time.time() - self.countdown_start_time
        if elapsed >= config.COUNTDOWN_DURATION_S:
            self.countdown_active = False
            self.active = True

        if self.active and time.time() - self._last_mole_time > config.MOLE_SPAWN_INTERVAL_S:
            self._mole_frac = self._random_mole_frac()
            self._last_mole_time = time.time()

    def draw_start_button(self, frame):
        self._overlay_rgba(frame, self.start_button_image, self.button_pos)

    def draw(self, frame):
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start_time
            remaining = config.COUNTDOWN_DURATION_S - int(elapsed)
            text = str(remaining) if remaining > 0 else "GO!"
            color = (255, 0, 0) if remaining > 0 else (0, 255, 0)
            cv2.putText(frame, text, (self.frame_width // 2 - 50, self.frame_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10, cv2.LINE_AA)
            return

        if not self.active:
            return

        cv2.putText(frame, "Point at the mole to whack it!", (60, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 106, 76), 2)
        cv2.rectangle(frame, (self._xmin, self._ymin), (self._xmax, self._ymax), (255, 255, 0), 2)
        self._overlay_rgba(frame, self.mole_image, (
            self.mole_pos[0] - config.MOLE_RADIUS, self.mole_pos[1] - config.MOLE_RADIUS))

        if self._hit_pos is not None and time.time() < self._hit_until:
            size = self.hit_effect_image.shape[0]
            top_left = (self._hit_pos[0] - size // 2, self._hit_pos[1] - size // 2)
            self._overlay_rgba(frame, self.hit_effect_image, top_left)

        cv2.putText(frame, f"Score: {self.score}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, self.close_button_pos,
                      (self.close_button_pos[0] + self.close_button_size[0],
                       self.close_button_pos[1] + self.close_button_size[1]), (0, 0, 255), -1)
        cv2.putText(frame, "Close Game", (self.close_button_pos[0] + 10, self.close_button_pos[1] + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    @staticmethod
    def _overlay_rgba(frame, rgba_image, top_left):
        x, y = top_left
        h, w = rgba_image.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return
        roi = frame[y:y + h, x:x + w]
        if rgba_image.shape[2] == 4:
            alpha = rgba_image[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = alpha * rgba_image[:, :, c] + (1 - alpha) * roi[:, :, c]
        else:
            roi[:, :, :] = rgba_image[:, :, :3]
        frame[y:y + h, x:x + w] = roi
