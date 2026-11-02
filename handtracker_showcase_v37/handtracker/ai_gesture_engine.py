"""
ai_gesture_engine.py
--------------------
Milestone 5 quick-win: hybrid AI + rule-based gesture recognition.

Uses MediaPipe's official pre-trained GestureRecognizer model (a real
ML model trained on a large gesture dataset) for high-level static
gestures, while keeping the existing rule-based engine for pinch
(precision needed) and swipe (temporal).

This gives the project a genuine AI component that works out of the box
without requiring the user to collect and train their own dataset first.
The rule-based engine remains as a reliable fallback and for gestures
the official model does not cover well (especially pinch).

Supported AI gestures (from the official model):
  Closed_Fist, Open_Palm, Pointing_Up, Thumb_Up, Thumb_Down,
  Victory, ILoveYou, None

We map these into the project's existing gesture names so the rest of
the pipeline (stabilizer, action dispatcher, desktop interaction)
needs zero changes.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config
from gesture_engine import (
    classify_static_gesture,
    GestureStabilizer,
    is_strict_fist,
)

# Map MediaPipe's official labels → our project's gesture vocabulary
_AI_TO_PROJECT = {
    "Closed_Fist": "fist",
    "Open_Palm": "open_palm",
    "Pointing_Up": "pointing",
    "Thumb_Up": "thumb_up",
    "Thumb_Down": "thumb_down",
    "Victory": "two_finger",
    "ILoveYou": "two_finger",  # closest useful mapping
    "None": None,
}


class AIGestureEngine:
    """Hybrid recognizer: MediaPipe GestureRecognizer (ML) + rule-based
    pinch/swipe + the existing stability pipeline."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_ai: bool = True,
        num_hands: Optional[int] = None,
    ):
        self.use_ai = use_ai
        self.num_hands = num_hands if num_hands is not None else config.MAX_NUM_HANDS
        self._recognizer = None
        # check_swipe=False: neither Presentation nor Media mode acts on
        # swipe anymore (both moved to discrete poses / the volume-knob
        # gesture instead - see showcase.py), and leaving swipe-checking
        # on was actively hurting them: it runs before static-gesture
        # confirmation every frame and fires on hand SPEED alone,
        # regardless of hand SHAPE, so raising a fist or a pointing
        # finger with any real motion behind it could "eat" that frame's
        # confirmation slot as a false swipe instead of ever letting
        # fist/pointing accumulate enough consecutive frames to confirm.
        self.stabilizer = GestureStabilizer(check_swipe=False)
        self._last_ai_label = None
        self._last_ai_score = 0.0

        if use_ai:
            if model_path is None:
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "models",
                    "gesture_recognizer.task",
                )
            if not os.path.isfile(model_path):
                print(f"[AI] Model not found at {model_path} – falling back to rules only.")
                self.use_ai = False
            else:
                try:
                    base_options = python.BaseOptions(model_asset_path=model_path)
                    options = vision.GestureRecognizerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.VIDEO,
                        num_hands=self.num_hands,
                        min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
                    )
                    self._recognizer = vision.GestureRecognizer.create_from_options(options)
                    print("[AI] MediaPipe GestureRecognizer loaded successfully.")
                except Exception as e:
                    print(f"[AI] Failed to load model: {e} – falling back to rules only.")
                    self.use_ai = False

    def process(
        self,
        frame_bgr,
        hand_landmarks=None,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Returns (raw_gesture, confirmed_gesture, confidence).

        - raw_gesture: what the AI/rules currently see (can flicker)
        - confirmed_gesture: after multi-frame confirmation + cooldown
        - confidence: 0-1 from the AI model (or 1.0 for pure rules)
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        ai_gesture = None
        confidence = 0.0

        if self.use_ai and self._recognizer is not None:
            # Convert BGR → RGB and wrap for the Tasks API
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)

            if result.gestures and len(result.gestures) > 0:
                top = result.gestures[0][0]  # first hand, top category
                ai_label = top.category_name
                confidence = top.score
                ai_gesture = _AI_TO_PROJECT.get(ai_label)
                self._last_ai_label = ai_label
                self._last_ai_score = confidence

        # MediaPipe's Closed_Fist happily accepts a half-closed hand. Veto
        # it unless the fingers are genuinely folded (is_strict_fist) - a
        # vetoed frame falls through to the rule classifier, which reports
        # None for the in-between shape instead of a false "fist".
        if ai_gesture == "fist" and hand_landmarks is not None and not is_strict_fist(hand_landmarks):
            ai_gesture = None

        # Prefer AI when it is confident; otherwise fall back to rules
        if ai_gesture is not None and confidence >= 0.55:
            raw = ai_gesture
        elif hand_landmarks is not None:
            raw = classify_static_gesture(hand_landmarks)
            confidence = 0.7  # rule-based "confidence"
        else:
            raw = None
            confidence = 0.0

        # Stabilizer handles multi-frame confirmation + swipe detection
        palm_point = None
        if hand_landmarks is not None:
            palm = hand_landmarks.landmark[9]
            palm_point = (palm.x, palm.y)

        confirmed = self.stabilizer.update(raw, palm_point, time.time())

        return raw, confirmed, confidence

    def reset(self):
        self.stabilizer.reset()
        self._last_ai_label = None
        self._last_ai_score = 0.0

    @property
    def last_ai_debug(self):
        return self._last_ai_label, self._last_ai_score
