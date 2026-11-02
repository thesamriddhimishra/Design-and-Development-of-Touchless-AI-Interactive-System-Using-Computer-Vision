"""
main.py
-------
Entry point. Opens the webcam, runs MediaPipe hand tracking, and routes
the detected pinch/tap gestures to whichever feature (drawing, game,
piano) is currently active. This is a direct, cleaned-up refactor of
the original single-file prototype - same behavior, modular structure,
no hardcoded paths. Milestone 2 work (gesture stability, a real
gesture-to-action mapping layer, multi-gesture support) builds on top
of this.

Run with:
    python main.py
"""
import cv2
import mediapipe as mp
import pygame

import config
from features.drawing import DrawingPanel
from features.game import WhackAMoleGame
from features.piano import Piano
from gestures import (
    PointSmoother,
    ScalarSmoother,
    get_landmark_px,
    hand_scale_px,
    is_pinching_scaled,
)


def get_hand_id(results, i):
    """A stable per-hand identifier across frames, so each hand keeps its
    own smoothing/gesture state instead of two hands fighting over one
    shared state (which was causing positions to blend together)."""
    try:
        return results.multi_handedness[i].classification[0].label  # "Left" / "Right"
    except (AttributeError, IndexError, TypeError):
        return f"hand_{i}"


def main():
    pygame.init()
    pygame.mixer.init()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=config.MAX_NUM_HANDS,
        model_complexity=config.MODEL_COMPLEXITY,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    mp_draw = mp.solutions.drawing_utils

    # Per-hand smoothing state, keyed by hand_id.
    index_smoothers = {}
    thumb_smoothers = {}
    # Smoothed hand-size reference (wrist-to-middle-knuckle), one per
    # hand, so pinch detection scales with each hand's distance from the
    # camera instead of using a fixed pixel gap (see gestures.pinch_ratio -
    # a fixed threshold reads as a bigger gap when the hand is close to
    # the camera and a smaller one when it's far away, so it misses real
    # pinches at one distance and false-fires at another).
    hand_scale_smoothers = {}
    was_pinching_by_hand = {}

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    drawing_panel = DrawingPanel(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    game = WhackAMoleGame(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    piano = Piano(config.FRAME_WIDTH, config.FRAME_HEIGHT)

    fps_display_value = 0.0
    prev_frame_time = cv2.getTickCount() / cv2.getTickFrequency()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            now = cv2.getTickCount() / cv2.getTickFrequency()

            frame_dt = max(1e-6, now - prev_frame_time)
            prev_frame_time = now
            instant_fps = 1.0 / frame_dt
            fps_display_value = instant_fps if fps_display_value == 0 else (
                0.9 * fps_display_value + 0.1 * instant_fps)

            any_feature_active = game.active or game.countdown_active or piano.active

            if not any_feature_active and not drawing_panel.panel_visible:
                game.draw_start_button(frame)
                piano.draw_start_button(frame)

            seen_hand_ids = set()

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    hand_id = get_hand_id(results, i)
                    seen_hand_ids.add(hand_id)

                    raw_index = get_landmark_px(
                        hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        config.FRAME_WIDTH, config.FRAME_HEIGHT)
                    raw_thumb = get_landmark_px(
                        hand_landmarks, mp_hands.HandLandmark.THUMB_TIP,
                        config.FRAME_WIDTH, config.FRAME_HEIGHT)

                    if hand_id not in index_smoothers:
                        index_smoothers[hand_id] = PointSmoother()
                        thumb_smoothers[hand_id] = PointSmoother()
                        hand_scale_smoothers[hand_id] = ScalarSmoother(alpha=0.12)

                    # Smoothing removes the frame-to-frame jitter that was
                    # making drawn lines and pinch detection jagged.
                    # Cast to int here since this is pixel space and the
                    # drawing/button-hit-test code below needs integer
                    # coordinates for OpenCV - PointSmoother itself returns
                    # floats so it works correctly for normalized (0-1)
                    # coordinates too, as used elsewhere.
                    index_x, index_y = index_smoothers[hand_id].update(raw_index, now)
                    index_x, index_y = int(index_x), int(index_y)
                    thumb_x, thumb_y = thumb_smoothers[hand_id].update(raw_thumb, now)
                    thumb_point = (int(thumb_x), int(thumb_y))

                    raw_scale = hand_scale_px(hand_landmarks, config.FRAME_WIDTH, config.FRAME_HEIGHT)
                    smooth_scale = hand_scale_smoothers[hand_id].update(raw_scale)
                    pinching = is_pinching_scaled((index_x, index_y), thumb_point, smooth_scale)
                    was_pinching = was_pinching_by_hand.get(hand_id, False)

                    # Drawing/erasing needs an actual pinch (thumb+index together).
                    if not game.active and not piano.active:
                        if pinching:
                            drawing_panel.handle_pinch(hand_id, index_x, index_y)
                        elif was_pinching:
                            drawing_panel.handle_unpinch(hand_id)

                    # Button taps are hover-based (fingertip over the button), same as
                    # the original prototype - no pinch required. Only debounced where
                    # the original debounced them (the control panel buttons).
                    if not game.active and not piano.active:
                        drawing_panel.handle_tap(
                            index_x, index_y, frame,
                            lambda: cv2.addWeighted(frame, 1, drawing_panel.canvas, 0.5, 0))
                    if not drawing_panel.panel_visible and not piano.active:
                        game.handle_tap(index_x, index_y)
                    if not drawing_panel.panel_visible and not game.active:
                        piano.handle_tap(index_x, index_y)

                    if piano.active:
                        piano.handle_finger_position(hand_id, index_x, index_y)

                    was_pinching_by_hand[hand_id] = pinching

            # Clean up state for any hand that's no longer visible, so a
            # disappeared hand doesn't leave a stuck note or a stale
            # smoothing history behind.
            for stale_id in set(index_smoothers) - seen_hand_ids:
                del index_smoothers[stale_id]
                del thumb_smoothers[stale_id]
                hand_scale_smoothers.pop(stale_id, None)
                was_pinching_by_hand.pop(stale_id, None)
                drawing_panel.handle_hand_lost(stale_id)
                piano.release_hand(stale_id)

            game.update()

            # Mutually-exclusive modes: activating one clears the others,
            # matching the original prototype's behavior.
            if game.active or game.countdown_active:
                piano.stop()
                drawing_panel.panel_visible = False
                drawing_panel.reset_canvas()
            elif piano.active:
                game.stop()
                drawing_panel.panel_visible = False
                drawing_panel.reset_canvas()

            if not game.active and not piano.active:
                drawing_panel.draw(frame)
            game.draw(frame)
            piano.draw(frame)

            frame_with_canvas = cv2.addWeighted(frame, 1, drawing_panel.canvas, 0.5, 0)

            if config.SHOW_FPS:
                cv2.putText(frame_with_canvas, f"FPS: {fps_display_value:.0f}", (10, config.FRAME_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Hand Tracker - Drawing, Game & Piano", frame_with_canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()
