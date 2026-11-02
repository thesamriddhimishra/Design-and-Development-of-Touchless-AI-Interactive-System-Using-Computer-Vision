"""
gesture_actions.py
-------------------
Milestone 2.4: "the important goal is to separate what gesture was
detected from what action should happen." gesture_engine.py only ever
outputs a gesture name - this module is the one place that decides what
that name means. Real desktop/meeting/AR actions get wired in from
Milestone 3 onward; for now actions are just labeled and logged so the
mapping shape (and its user-customization hook) is proven out before
anything touches the OS.
"""

DEFAULT_GESTURE_ACTION_MAP = {
    "pinch": "select",
    # fist is intentionally unmapped here: desktop-mode drag now runs
    # off a HELD PINCH instead of a held fist (see
    # desktop_interaction.py / config.DRAG_HOLD_SECONDS), which fixed
    # the cursor jumping right as a drag started. fist is used directly
    # (not through this map) as the presentation-mode laser-pointer
    # hold gesture - see showcase.py.
    "pointing": "move_cursor",
    # two_finger is intentionally unmapped in Desktop mode: it used to
    # be an alternate right-click trigger, but it sat too close to
    # normal pointing/pinching in practice (a hand mid-pinch often
    # transiently reads as two_finger), causing right-clicks to fire
    # when a left click/drag was intended. Right-click is now driven
    # solely by three_finger, a clearly distinct pose.
    "three_finger": "secondary_action",  # right-click
    "swipe_left": "previous",
    "swipe_right": "next",
    "swipe_up": "scroll_down",    # content follows the hand, like a touchscreen
    "swipe_down": "scroll_up",
    # thumb_up intentionally unmapped: while closing into a fist the
    # thumb often lags and briefly reads as "thumbs up", which used to
    # fire Enter and open whatever desktop icon was focused. Leaving it
    # unmapped stops those accidental app opens.
    "thumb_down": "activate_toggle",
    # open_palm is intentionally left unmapped: it's a natural, frequent
    # resting hand shape, so anything assigned to it fires constantly by
    # accident (this is exactly what happened when it was the toggle in
    # an earlier version). thumb_down is deliberate and rarely made
    # incidentally, which is why it took over the toggle role.
}


class GestureActionDispatcher:
    def __init__(self, mapping=None, history_length=20):
        self.mapping = dict(mapping or DEFAULT_GESTURE_ACTION_MAP)
        self.history_length = history_length
        self.history = []  # list of (timestamp, hand_id, gesture, action)

    def dispatch(self, gesture_name, hand_id=None, timestamp=None):
        """Looks up the action for a confirmed gesture and records it.
        Returns the action name, or None if this gesture isn't mapped."""
        action = self.mapping.get(gesture_name)
        if action is not None:
            self.history.append((timestamp, hand_id, gesture_name, action))
            del self.history[:-self.history_length]
        return action

    def set_mapping(self, gesture_name, action_name):
        """User-customization hook: Milestone 3.4 ('Add User Customization')
        exposes this through the overlay UI so users can remap gestures
        to actions themselves instead of being stuck with the defaults."""
        self.mapping[gesture_name] = action_name
