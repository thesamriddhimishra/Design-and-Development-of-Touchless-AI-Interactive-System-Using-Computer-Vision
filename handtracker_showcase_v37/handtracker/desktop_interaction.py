"""
desktop_interaction.py
------------------------
Milestone 3: the decision layer between the gesture engine and a
desktop controller (desktop_control.DesktopController or
DryRunController). Pure logic, no pyautogui/OS calls of its own - so it
can be unit tested without a real mouse, keyboard, or display. Builds
directly on gesture_actions.py's gesture->action-name mapping from
Milestone 2; this class only adds the desktop-specific mechanics on top
of that: continuous cursor tracking, edge-triggered clicks, held-gesture
dragging, and momentary keyboard shortcuts.

Gesture -> behavior (see gesture_actions.DEFAULT_GESTURE_ACTION_MAP for
the authoritative mapping - this list just describes the mechanics):
    pointing (held)  -> cursor follows the index fingertip, EXCEPT while
                         pinching (see below)
    pinch            -> behaves like a physical mouse button: button DOWN
                         the instant the pinch engages, UP when it
                         releases. A quick pinch is a click, a held pinch
                         is a drag, two quick pinches a double-click.
                         The cursor is first rewound to where the
                         fingertip was ~config.PINCH_ANCHOR_LOOKBACK_SECONDS
                         before the pinch (closing the fingers drags the
                         fingertip down), stays locked there, and only
                         moves - following the index KNUCKLE, which
                         doesn't dip - once the hand travels past
                         config.PINCH_DRAG_DEADZONE_NORM.
    three_finger     -> right-click
    swipe_left/right -> previous/next (default: Alt+Left / Alt+Right)
    swipe_up/down    -> scroll (open-hand flick; faster = further)
    thumb_up         -> unmapped (was Enter; caused accidental app opens
                         while closing into a fist)
    two_finger       -> unmapped (was an alternate right-click; removed
                         because it sat too close to a genuine pinch in
                         practice, occasionally firing a right-click
                         alongside an intended left click)
    thumb_down       -> toggle desktop control on/off (checked even
                         while control is off, since it's the only
                         gesture that can turn control back on). Whatever
                         gesture the mapping assigns "activate_toggle" to
                         gets this behavior - it isn't hardcoded to one
                         gesture name, so remapping it doesn't need a
                         code change.

The cursor deliberately STOPS following the fingertip the instant a
pinch begins, resuming only once the pinch releases. Without this, the
small hand motion needed to actually perform a pinch (bringing thumb
and index together) drags the cursor along with it, so the click lands
somewhere other than where the user was aiming when they decided to
click.

Pinch is trusted directly from the raw, hand-size-scaled thumb/index
distance (gestures.is_pinching_from_ratio, with hysteresis - see that
function's docstring) and debounced on its own (config.PINCH_CONFIRM_FRAMES)
- it does NOT require classify_static_gesture() to agree the hand is
"pointing" first. Making a pinch physically curls the index tip (and
the thumb) past that classifier's "extended" angle threshold, so it
very often reads as "fist" on exactly the frames a pinch is happening -
gating pinch behind that reading used to make deliberate pinches
misfire as fist-drags instead of clicks. Trusting the pinch-ratio
signal directly fixes that.

Pinch itself is already debounced (config.PINCH_CONFIRM_FRAMES) before
any of this drag-vs-click logic sees it, so a single stray misread
can't start or drop a drag. Guards against a misfired action:
  - No click, right-click, or keyboard-shortcut action fires while a
    drag is active, since none of those actions make sense in the
    middle of a drag anyway.
  - Right-click (three_finger -> secondary_action) is suppressed while
    a pinch is active or was active within the last click_cooldown
    window, so a hand relaxing out of a pinch (which can transiently
    look three-finger-ish) can't immediately fire a right-click right
    alongside the left click it was actually meant to be.
"""
import time
from collections import deque

import config
from gesture_actions import GestureActionDispatcher

# Keyboard shortcuts for the momentary, non-mouse actions. Alt+Left/Right
# is "back/forward" in most browsers and file managers - a safe generic
# default until a specific target application calls for something else.
# Override via the action_keys constructor argument.
DEFAULT_ACTION_KEYS = {
    "previous": ["alt", "left"],
    "next": ["alt", "right"],
    "confirm": ["enter"],
    "cancel": ["esc"],  # not assigned to a gesture by default; available to remap onto one
}


class DesktopInteractionController:
    def __init__(self, controller, dispatcher=None, action_keys=None,
                 click_cooldown=None, start_enabled=None):
        self.controller = controller
        self.dispatcher = dispatcher or GestureActionDispatcher()
        self.action_keys = dict(action_keys or DEFAULT_ACTION_KEYS)
        self.click_cooldown = config.CLICK_COOLDOWN_SECONDS if click_cooldown is None else click_cooldown
        self.enabled = (
            config.DESKTOP_CONTROL_DEFAULT_ENABLED if start_enabled is None else start_enabled
        )

        self._pinch_held = False          # debounced pinch state
        self._release_count = 0
        self._was_pinching = False
        self._last_click_time = float("-inf")
        self._dragging = False            # True once the pinch has moved past the deadzone
        self._button_down = False
        self._pinch_started_at = None
        self._history = deque()           # (time, fingertip point) for anchor rewind
        self._anchor = None               # cursor point locked at pinch start
        self._ref_at_start = None         # hand reference point at pinch start

    # ------------------------------------------------------------------
    def _anchor_point(self, t, hand_ref_now):
        """Where the cursor should sit for a pinch that just engaged.

        Uses the fingertip from ~t seconds ago (before it dipped toward
        the thumb) and, when a stable hand reference (index knuckle) is
        available, shifts it by however far the HAND has travelled since
        - so it removes only the finger-curl dip, not real hand movement.
        Aiming and pinching while the hand is still moving therefore
        lands where the hand is now, not where it was 200ms ago."""
        if not self._history:
            return None
        entry = self._history[-1]
        for e in self._history:
            if e[0] >= t:
                entry = e
                break
        _, tip, ref = entry
        if hand_ref_now is not None and ref is not None:
            return (min(1.0, max(0.0, tip[0] + hand_ref_now[0] - ref[0])),
                    min(1.0, max(0.0, tip[1] + hand_ref_now[1] - ref[1])))
        return tip

    def process_frame(self, raw_gesture, confirmed_gesture, pinching, index_point, now,
                      hand_ref_point=None, swipe_speed_y=0.0):
        """Call once per frame for the tracked hand.

        raw_gesture: classify_static_gesture() result - used for the held
            cursor-tracking behavior.
        confirmed_gesture: GestureStabilizer result - momentary actions.
        pinching: this frame's pinch boolean (already hysteresis-filtered
            by gestures.is_pinching_from_ratio). Debounced here on RELEASE
            only; engaging is instant.
        index_point: (x, y) normalized index fingertip.
        hand_ref_point: optional (x, y) normalized point that does NOT
            move when the finger curls into a pinch (the index knuckle).
            Used to move the cursor during a drag; without it a drag
            follows the fingertip instead.
        swipe_speed_y: the stabilizer's last vertical palm speed (frame
            heights/sec); scales how far a scroll swipe scrolls.
        now: time.time().

        Returns [(event, detail), ...] describing what fired.
        """
        events = []

        if index_point is not None:
            self._history.append((now, index_point, hand_ref_point))
            while self._history and now - self._history[0][0] > 1.0:
                self._history.popleft()

        action = self.dispatcher.dispatch(confirmed_gesture, timestamp=now) if confirmed_gesture else None

        if action == "activate_toggle":
            self.enabled = not self.enabled
            self._release_pinch(events)
            events.append(("toggle", self.enabled))

        if not self.enabled:
            return events

        # --- pinch = real mouse button --------------------------------
        # Engage instantly; release after PINCH_RELEASE_CONFIRM_FRAMES.
        if pinching:
            self._release_count = 0
            if not self._pinch_held:
                self._pinch_held = True
                self._pinch_started_at = now
                # Rewind the cursor to where the fingertip was BEFORE it
                # started dipping toward the thumb, then press.
                anchor = self._anchor_point(now - config.PINCH_ANCHOR_LOOKBACK_SECONDS, hand_ref_point)
                self._anchor = anchor if anchor is not None else index_point
                self._ref_at_start = hand_ref_point
                self._dragging = False
                if self._anchor is not None:
                    self.controller.move_cursor(self._anchor)
                self.controller.mouse_down()
                self._button_down = True
                events.append(("mouse_down", None))
        elif self._pinch_held:
            self._release_count += 1
            if self._release_count >= config.PINCH_RELEASE_CONFIRM_FRAMES:
                self._release_pinch(events)

        # --- cursor ----------------------------------------------------
        if self._pinch_held:
            # Locked to the anchor until the HAND (not the curling
            # finger) travels past the deadzone; then it follows the
            # hand's movement relative to where the pinch began.
            if self._anchor is not None:
                target = None
                if hand_ref_point is not None and self._ref_at_start is not None:
                    dx = hand_ref_point[0] - self._ref_at_start[0]
                    dy = hand_ref_point[1] - self._ref_at_start[1]
                    if not self._dragging and (dx * dx + dy * dy) ** 0.5 >= config.PINCH_DRAG_DEADZONE_NORM:
                        self._dragging = True
                    if self._dragging:
                        target = (min(1.0, max(0.0, self._anchor[0] + dx)),
                                  min(1.0, max(0.0, self._anchor[1] + dy)))
                elif hand_ref_point is None and index_point is not None:
                    # No stable reference available: fall back to the
                    # fingertip, but only once it has clearly moved.
                    ddx = index_point[0] - self._anchor[0]
                    ddy = index_point[1] - self._anchor[1]
                    if not self._dragging and (ddx * ddx + ddy * ddy) ** 0.5 >= config.PINCH_DRAG_DEADZONE_NORM * 2:
                        self._dragging = True
                    if self._dragging:
                        target = index_point
                if target is not None:
                    self.controller.move_cursor(target)
                    events.append(("move_cursor", target))
        elif raw_gesture == "pointing" and index_point is not None:
            self.controller.move_cursor(index_point)
            events.append(("move_cursor", index_point))

        effective_pinching = self._pinch_held
        self._was_pinching = effective_pinching

        # Momentary keyboard/mouse-button actions, from the stabilizer's
        # confirmed gestures. Suppressed while a pinch is held. Right-click
        # is also suppressed right after a pinch releases, since a hand
        # relaxing out of a pinch can briefly look three-finger-ish.
        if not self._pinch_held:
            recently_pinched = (now - self._last_click_time) <= self.click_cooldown
            if action == "secondary_action" and not recently_pinched:
                self.controller.right_click()
                events.append(("right_click", None))
            elif action in ("scroll_up", "scroll_down"):
                notches = max(config.SCROLL_NOTCHES_MIN, min(
                    config.SCROLL_NOTCHES_MAX,
                    round(abs(swipe_speed_y) * config.SCROLL_NOTCHES_PER_VELOCITY)))
                amount = notches if action == "scroll_up" else -notches
                if config.SCROLL_INVERT:
                    amount = -amount
                self.controller.scroll(amount)
                events.append(("scroll", amount))
            elif action in self.action_keys:
                self.controller.press_key(self.action_keys[action])
                events.append(("press_key", self.action_keys[action]))

        return events

    def _release_pinch(self, events):
        """Button up, and report a 'click' if it was a tap (no drag)."""
        if self._button_down:
            self.controller.mouse_up()
            self._button_down = False
            events.append(("mouse_up", None))
            if not self._dragging:
                events.append(("click", None))
            self._last_click_time = time.time()
        self._pinch_held = False
        self._dragging = False
        self._release_count = 0
        self._pinch_started_at = None
        self._anchor = None
        self._ref_at_start = None

    def reset(self):
        """Call when the hand is lost, so the button doesn't stay stuck
        down and stale state doesn't leak into the next pinch."""
        self._release_pinch([])
        self._was_pinching = False
        self._history.clear()
