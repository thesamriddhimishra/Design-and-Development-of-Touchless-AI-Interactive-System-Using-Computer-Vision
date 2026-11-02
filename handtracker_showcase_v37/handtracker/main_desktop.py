"""
main_desktop.py
-----------------
Milestone 3: turns the Milestone 2 gesture engine into real desktop
control - a floating always-on-top overlay plus gesture-driven mouse
and keyboard control of the whole desktop, not just the camera window.

Gesture -> behavior (see desktop_interaction.py for the full mapping):
    Point            -> move cursor (while held, frozen during a pinch)
    Pinch            -> left-click (scaled to hand size; works while pointing)
    Two-finger + pinch -> left-click
    Fist (held)      -> click-and-drag
    Two-finger       -> right-click
    Three-finger     -> right-click
    Swipe left/right -> previous/next (Alt+Left / Alt+Right)
    Thumb down       -> toggle desktop control on/off
    (Thumb up no longer maps to Enter — that caused accidental app opens)

Desktop control starts OFF - make a thumb-down gesture to turn it on,
and again to turn it off. Thumb-down was chosen deliberately over the
more obvious open-palm: open palm is a hand's natural relaxed/resting
shape, so using it as the toggle meant control flipped on and off
constantly by accident. Thumb-down needs an intentional, distinctive
hand shape that isn't made in passing.

If pyautogui isn't installed or no display is available, this falls
back to a dry-run mode that logs what it would have done instead of
touching the mouse/keyboard - so the interaction logic can still be
exercised.

A green ring follows your mapped cursor position on screen while
control is on, turning red for a moment on each click, so you can see
exactly where a pinch is about to click before it happens. The cursor
deliberately stops moving the instant a pinch starts (see
desktop_interaction.py) so the small hand motion of pinching itself
can't drag the click off target.

Only one hand drives the cursor at a time. If a second hand enters the
frame, it's ignored - the program stays "locked" onto whichever hand
it was already tracking until that hand is gone for more than
config.HAND_LOCK_TIMEOUT_S, rather than flickering between hands
(which caused erratic cursor jumps and misfired clicks/drags).

If pyautogui's corner fail-safe trips (moving the mouse into a screen
corner - either by accident or deliberately, as an emergency stop),
desktop control turns itself off and the program keeps running rather
than crashing; thumb-down turns it back on.

Controls in the camera preview window:
    q - quit

Run with:
    python main_desktop.py
"""
import time

import cv2
import mediapipe as mp

import config
from desktop_control import DesktopController, DesktopControlAborted, DryRunController, map_to_screen
from desktop_interaction import DesktopInteractionController
from gesture_engine import GestureStabilizer, classify_static_gesture
from gestures import PointSmoother, ScalarSmoother, hand_scale_px, is_pinching_scaled

try:
    from overlay import CursorReticle, Overlay
    _OVERLAY_AVAILABLE = True
except ImportError:
    _OVERLAY_AVAILABLE = False


class _NullOverlay:
    """Used when tkinter isn't available (rare on Windows, since it
    ships with the standard installer, but some minimal Python
    installs omit it). The HUD/reticle windows just don't appear;
    everything else still works, and status is still shown as text in
    the camera preview window."""

    def update(self, *args, **kwargs):
        pass

    def show(self):
        pass

    def hide(self):
        pass

    def close(self):
        pass


def build_controller():
    try:
        return DesktopController()
    except RuntimeError as e:
        print(f"[warning] {e}")
        print("[warning] Falling back to dry-run mode: actions will be logged, not executed.")
        return DryRunController()


def build_overlay(cls, *args, **kwargs):
    if not _OVERLAY_AVAILABLE:
        print("[warning] tkinter is not available - the floating HUD/cursor windows will be "
              "skipped. Status is still shown in the camera preview window.")
        return _NullOverlay()
    try:
        return cls(*args, **kwargs)
    except Exception as e:
        print(f"[warning] Could not create a HUD window ({e}); continuing without it.")
        return _NullOverlay()


def hand_label(results, i):
    """MediaPipe's handedness label for the i-th detected hand this
    frame, or None if unavailable. Used to keep control "locked" onto
    one physical hand even when a second hand is present."""
    try:
        return results.multi_handedness[i].classification[0].label
    except (AttributeError, IndexError, TypeError):
        return None


def select_locked_hand(results, locked_label, locked_last_seen, now):
    """Picks which detected hand (if any) should drive the cursor this
    frame, ignoring any other hand also in view.

    Once a hand is locked on by its handedness label, later frames only
    use that same label - a second hand entering the frame is simply
    ignored rather than being allowed to steal or blend with control,
    which is what caused erratic jumps when two hands were visible at
    once. If the locked hand hasn't been seen for
    config.HAND_LOCK_TIMEOUT_S, the lock is released and whichever hand
    appears next may be picked up.

    Returns (hand_landmarks_or_None, new_locked_label, new_locked_last_seen).
    """
    if not results.multi_hand_landmarks:
        return None, locked_label, locked_last_seen

    labels = [hand_label(results, i) for i in range(len(results.multi_hand_landmarks))]

    # Lock expired (or never set) - adopt whichever hand appears first.
    if locked_label is None or (now - locked_last_seen) > config.HAND_LOCK_TIMEOUT_S:
        locked_label = labels[0]

    if locked_label in labels:
        idx = labels.index(locked_label)
        return results.multi_hand_landmarks[idx], locked_label, now

    # The locked hand isn't in this frame (only some other hand is) -
    # treat this frame as "no hand" for control purposes rather than
    # picking up the unlocked one.
    return None, locked_label, locked_last_seen


def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,  # detect both, but select_locked_hand() below always drives from just one
        model_complexity=config.DESKTOP_CONTROL_MODEL_COMPLEXITY,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    desktop_controller = build_controller()
    interaction = DesktopInteractionController(desktop_controller)
    overlay = build_overlay(Overlay)
    reticle = build_overlay(CursorReticle)

    # Cursor smoothing uses its own snappier filter settings than the
    # library default (tuned for drawing strokes, not cursor speed).
    index_smoother = PointSmoother(min_cutoff=config.CURSOR_SMOOTHING_MIN_CUTOFF,
                                    beta=config.CURSOR_SMOOTHING_BETA)
    thumb_smoother = PointSmoother(min_cutoff=config.CURSOR_SMOOTHING_MIN_CUTOFF,
                                    beta=config.CURSOR_SMOOTHING_BETA)
    # Heavily damped scale for pinch_ratio divisor — hand size only changes
    # as the user moves toward/away from the camera, so aggressive smoothing
    # is correct and avoids amplifying landmark noise into false clicks.
    hand_scale_smoother = ScalarSmoother(alpha=0.12)
    stabilizer = GestureStabilizer()

    locked_hand_label = None
    locked_hand_last_seen = float("-inf")

    prev_time = time.time()
    fps_value = 0.0
    abort_message_until = 0.0
    click_flash_until = 0.0

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

            current_gesture_label = "-"
            screen_point = None

            hand_landmarks, locked_hand_label, locked_hand_last_seen = select_locked_hand(
                results, locked_hand_label, locked_hand_last_seen, now)

            if hand_landmarks is not None:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Smoothed, normalized positions - used directly for
                # cursor mapping (map_to_screen expects 0-1 coordinates)
                # and converted to pixel space for the pinch check. Using
                # the One-Euro-smoothed points (rather than raw landmarks)
                # for the pinch distance matters: an earlier version of
                # this measured pinch relative to hand size using raw,
                # unsmoothed wrist/knuckle positions, and MediaPipe's
                # landmarks get noticeably noisier during a fist (fingers
                # occluding each other), which fed straight into false
                # clicks. Sticking to the already-filtered points and a
                # fixed threshold is less clever but much more reliable.
                index_norm = index_smoother.update(
                    (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y), now)
                thumb_norm = thumb_smoother.update(
                    (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y), now)
                index_px = (index_norm[0] * config.FRAME_WIDTH, index_norm[1] * config.FRAME_HEIGHT)
                thumb_px = (thumb_norm[0] * config.FRAME_WIDTH, thumb_norm[1] * config.FRAME_HEIGHT)
                # Scaled pinch: distance / smoothed hand size, so the same
                # physical finger gap triggers whether the hand is near or far.
                raw_scale = hand_scale_px(hand_landmarks, config.FRAME_WIDTH, config.FRAME_HEIGHT)
                smooth_scale = hand_scale_smoother.update(raw_scale)
                pinching = is_pinching_scaled(index_px, thumb_px, smooth_scale)

                raw_gesture = classify_static_gesture(hand_landmarks)
                palm_lm = hand_landmarks.landmark[9]
                confirmed = stabilizer.update(raw_gesture, (palm_lm.x, palm_lm.y), now)

                try:
                    events = interaction.process_frame(raw_gesture, confirmed, pinching, index_norm, now)
                except DesktopControlAborted as e:
                    # A tripped fail-safe (or other OS-level control error)
                    # used to crash the whole program here. Now it just
                    # turns control off and keeps everything else running.
                    print(f"[safety stop] {e}")
                    interaction.enabled = False
                    interaction.reset()
                    abort_message_until = now + 4.0
                    events = []

                if any(evt == "click" for evt, _ in events):
                    click_flash_until = now + 0.25

                current_gesture_label = raw_gesture or "-"
                if interaction.enabled:
                    screen_point = map_to_screen(index_norm, desktop_controller.screen_size)
            else:
                interaction.reset()
                stabilizer.reset()
                index_smoother.reset()
                thumb_smoother.reset()
                hand_scale_smoother.reset()

            status_color = (0, 255, 0) if interaction.enabled else (0, 0, 255)
            cv2.putText(frame, f"Desktop control: {'ON' if interaction.enabled else 'OFF'} (thumb down to toggle)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Gesture: {current_gesture_label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps_value:.0f}", (config.FRAME_WIDTH - 140, config.FRAME_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if now < abort_message_until:
                cv2.putText(frame, "SAFETY STOP: mouse hit a screen corner - thumb down to resume",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            overlay.update(interaction.enabled, current_gesture_label, interaction._dragging, fps_value)

            if screen_point is not None:
                reticle.update(screen_point, clicked=(now < click_flash_until))
            else:
                reticle.hide()

            cv2.imshow("Desktop Control - Milestone 3", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        interaction.reset()  # make sure a held drag doesn't stay stuck down on exit
        overlay.close()
        reticle.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
