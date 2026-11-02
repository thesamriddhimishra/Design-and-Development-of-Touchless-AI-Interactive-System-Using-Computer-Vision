"""
features/piano.py
------------------
On-screen piano: 7 keys spanning the frame width, played by touching a
key zone with the index finger. Same behavior as the original prototype.
"""
import cv2
import pygame

import config
from gestures import is_point_in_box


class Piano:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.active = False

        self.key_names = list(config.PIANO_NOTE_FILES.keys())
        self.sounds = {name: pygame.mixer.Sound(path) for name, path in config.PIANO_NOTE_FILES.items()}
        self.num_keys = len(self.key_names)
        self.key_width = frame_width // self.num_keys
        self.key_y = frame_height // 2
        self._touched_key_by_hand = {}  # hand_id -> key index currently held, or None

        self.start_button_pos = (frame_width - 220, 500)
        self.close_button_pos = (frame_width - 220, 600)
        self.button_size = (170, 60)

    def stop(self):
        self.active = False
        self._touched_key_by_hand.clear()

    def handle_tap(self, x, y):
        if not self.active and is_point_in_box(x, y, *self.start_button_pos, *self.button_size):
            self.active = True
        elif self.active and is_point_in_box(x, y, *self.close_button_pos, *self.button_size):
            self.stop()

    def handle_finger_position(self, hand_id, x, y):
        """Call once per frame per tracked hand while active. Each hand's
        currently-held key is tracked separately, so two hands can hold
        two different keys down at once - a note only (re)plays when a
        hand transitions onto a new key, not on every frame it's held."""
        if not self.active:
            return

        touched_index = None
        if self.key_y <= y < self.key_y + config.PIANO_KEY_HEIGHT:
            for i in range(self.num_keys):
                key_x_start = i * self.key_width
                key_x_end = (i + 1) * self.key_width
                if key_x_start <= x < key_x_end:
                    touched_index = i
                    break

        previous_index = self._touched_key_by_hand.get(hand_id)
        if touched_index != previous_index:
            if touched_index is not None:
                self.sounds[self.key_names[touched_index]].play()
            self._touched_key_by_hand[hand_id] = touched_index

    def release_hand(self, hand_id):
        """Call when a hand is no longer detected, so it doesn't keep a
        key visually 'held' forever."""
        self._touched_key_by_hand.pop(hand_id, None)

    def _currently_held_keys(self):
        return {i for i in self._touched_key_by_hand.values() if i is not None}

    def draw_start_button(self, frame):
        cv2.rectangle(frame, self.start_button_pos,
                       (self.start_button_pos[0] + self.button_size[0],
                        self.start_button_pos[1] + self.button_size[1]), (0, 255, 0), -1)
        cv2.putText(frame, "Start Piano", (self.start_button_pos[0] + 10, self.start_button_pos[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def draw(self, frame):
        if not self.active:
            return
        for i in range(self.num_keys):
            x_start = i * self.key_width
            x_end = (i + 1) * self.key_width
            color = (200, 200, 200) if i in self._currently_held_keys() else (255, 255, 255)
            cv2.rectangle(frame, (x_start, self.key_y), (x_end, self.key_y + config.PIANO_KEY_HEIGHT), color, -1)
            label_x = x_start + max(6, self.key_width // 2 - 18)
            cv2.putText(frame, self.key_names[i], (label_x, self.key_y + config.PIANO_KEY_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.rectangle(frame, self.close_button_pos,
                      (self.close_button_pos[0] + self.button_size[0],
                       self.close_button_pos[1] + self.button_size[1]), (0, 0, 255), -1)
        cv2.putText(frame, "Close Piano", (self.close_button_pos[0] + 10, self.close_button_pos[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
