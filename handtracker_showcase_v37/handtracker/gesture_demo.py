"""
gesture_demo.py
-----------------
Milestone 2 test/demo harness. Shows the gesture engine's raw
per-frame classification, the stabilized/confirmed gesture, and the
resulting mapped action live on screen - use this to stress-test
recognition across the conditions milestone 2.1 calls for (lighting,
background, distance, hand size, partial visibility, fast movement,
one vs. two hands) before wiring real gestures into the drawing/game/
piano features.

Controls:
    q - quit
    d - toggle the per-hand debug readout (raw measurements behind each
        classification, with the threshold each is compared against -
        use this to tune the THUMB_* / FINGER_* values in config.py
        against your own hand)

Run with:
    python gesture_demo.py
"""
import time

import cv2
import mediapipe as mp

import config
from gesture_actions import GestureActionDispatcher
from gesture_engine import GestureStabilizer, classify_static_gesture, describe_hand
from gestures import PointSmoother, get_landmark_px


def get_hand_id(results, i):
    """A stable per-hand identifier across frames (see main.py for why
    this matters more than a raw list index)."""
    try:
        return results.multi_handedness[i].classification[0].label
    except (AttributeError, IndexError, TypeError):
        return f"hand_{i}"


def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=config.MAX_NUM_HANDS,
        model_complexity=config.MODEL_COMPLEXITY,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    stabilizers = {}
    palm_smoothers = {}
    dispatcher = GestureActionDispatcher()
    recent_events = []  # (timestamp, hand_id, gesture, action)

    prev_time = time.time()
    fps_value = 0.0
    show_debug = True

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            now = time.time()

            dt = max(1e-6, now - prev_time)
            prev_time = now
            fps_value = (1 / dt) if fps_value == 0 else (0.9 * fps_value + 0.1 * (1 / dt))

            seen_hand_ids = set()

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hand_id = get_hand_id(results, i)
                    seen_hand_ids.add(hand_id)

                    if hand_id not in stabilizers:
                        stabilizers[hand_id] = GestureStabilizer()
                        palm_smoothers[hand_id] = PointSmoother()

                    # Middle-finger MCP (landmark 9) is a stable palm-center
                    # proxy, smoothed the same way fingertip drawing is.
                    palm_lm = hand_landmarks.landmark[9]
                    palm_point = palm_smoothers[hand_id].update((palm_lm.x, palm_lm.y), now)

                    candidate = classify_static_gesture(hand_landmarks)
                    confirmed = stabilizers[hand_id].update(candidate, palm_point, now)

                    wrist_px = get_landmark_px(hand_landmarks, 0, config.FRAME_WIDTH, config.FRAME_HEIGHT)
                    cv2.putText(frame, candidate or "-", (wrist_px[0] - 30, wrist_px[1] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                    if show_debug:
                        d = describe_hand(hand_landmarks)
                        debug_lines = [
                            f"fingers ext: {d['fingers_extended']}",
                            f"thumb straight: {d['thumb_straightness']:.0f} (> {config.THUMB_STRAIGHT_ANGLE_DEG})",
                            f"thumb reach: {d['thumb_reach']:.2f} (> {config.THUMB_REACH_RATIO})",
                            f"thumb vertical: {d['thumb_verticality']:+.2f} (|.| > {config.THUMB_VERTICAL_RATIO})",
                        ]
                        for row, line in enumerate(debug_lines):
                            cv2.putText(frame, line, (wrist_px[0] - 30, wrist_px[1] + 78 + row * 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 120), 1)

                    if confirmed:
                        action = dispatcher.dispatch(confirmed, hand_id, now)
                        recent_events.append((now, hand_id, confirmed, action))
                        del recent_events[:-8]

            for stale_id in set(stabilizers) - seen_hand_ids:
                stabilizers.pop(stale_id).reset()
                palm_smoothers.pop(stale_id, None)

            panel_h = 34 + 22 * len(recent_events)
            cv2.rectangle(frame, (0, 0), (440, panel_h), (30, 30, 30), -1)
            cv2.putText(frame, "Confirmed gesture -> action", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            for row, (t, hid, gesture, action) in enumerate(reversed(recent_events)):
                age = now - t
                cv2.putText(frame, f"{hid}: {gesture} -> {action}  ({age:.1f}s ago)",
                            (10, 44 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

            cv2.putText(frame, f"FPS: {fps_value:.0f}", (config.FRAME_WIDTH - 140, config.FRAME_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Gesture Engine Demo - Milestone 2", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('d'):
                show_debug = not show_debug
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
