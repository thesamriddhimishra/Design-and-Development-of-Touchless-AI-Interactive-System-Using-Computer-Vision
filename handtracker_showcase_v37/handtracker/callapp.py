"""
callapp.py
-----------
Video-call controls (mute, camera, raise hand, share) for Zoom, Google Meet
and Microsoft Teams, built to be SAFE to use while you're actually talking.

Two different kinds of accidents are guarded against:

1. SENDING A SHORTCUT TO THE WRONG WINDOW. The call apps' shortcuts only
   work when their window is focused, and the same keys do other things
   elsewhere (Ctrl+D bookmarks a page in a browser; Alt+A / Alt+Y mean
   nothing good in most apps). CallController therefore only sends a
   shortcut when the FOREGROUND window title identifies a call app, and
   tells you why it didn't otherwise. (If you're sharing your screen and
   focus is on the shared app, nothing is sent - bring the call window
   forward, or use the app's own toolbar.)

2. FIRING BY ACCIDENT FROM NORMAL TALKING GESTURES. People move their hands
   when they speak. HoldToConfirm only fires an action when ONE specific
   gesture is held, with the hand nearly still, for ~1 s (a progress ring is
   shown meanwhile), then ignores that gesture for a few seconds so a long
   hold can't toggle twice. Leaving a call is deliberately NOT mappable.

Shortcut tables are the apps' documented Windows defaults. They can differ
by version or if you've remapped them - edit HOTKEYS below if so.
"""
import sys
import time

HOTKEYS = {
    "zoom":  {"mute": ("alt", "a"), "camera": ("alt", "v"), "hand": ("alt", "y"),
              "share": ("alt", "shift", "s")},
    "meet":  {"mute": ("ctrl", "d"), "camera": ("ctrl", "e"), "hand": ("ctrl", "alt", "h"),
              "share": None},
    "teams": {"mute": ("ctrl", "shift", "m"), "camera": ("ctrl", "shift", "o"),
              "hand": ("ctrl", "shift", "k"), "share": ("ctrl", "shift", "e")},
}

ACTION_LABELS = {"mute": "Mute / unmute", "camera": "Camera on / off",
                 "hand": "Raise / lower hand", "share": "Share screen"}

APP_NAMES = {"zoom": "Zoom", "meet": "Google Meet", "teams": "Microsoft Teams"}


def detect_call_app(title):
    """Which call app a window title belongs to, or None."""
    if not title:
        return None
    t = title.lower()
    if "microsoft teams" in t:
        return "teams"
    # Zoom's own windows only - NOT "anything starting with zoom": a Word
    # document called "Zoom into the details" must not be treated as a call.
    if t == "zoom" or t.startswith(("zoom meeting", "zoom workplace", "zoom webinar",
                                    "zoom - ", "zoom | ")):
        return "zoom"
    # Meet runs in a browser tab titled "Meet - abc-defg-hij" / "Meet – ..." or "Google Meet".
    if "google meet" in t or t.startswith("meet - ") or t.startswith("meet \u2013 ") \
            or t.startswith("meet \u2014 "):
        return "meet"
    return None


def foreground_title(user32=None):
    """Title of the focused window (Windows), else None."""
    if user32 is None:
        if not sys.platform.startswith("win"):
            return None
        try:
            import ctypes
            user32 = ctypes.windll.user32
        except Exception:
            return None
    try:
        import ctypes
        hwnd = user32.GetForegroundWindow()
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value
    except Exception:
        return None


class CallController:
    def __init__(self, sender, title_fn=foreground_title, preferred="auto"):
        """sender(keys_tuple) presses a hotkey (pyautogui.hotkey in the real
        app); title_fn() returns the foreground window title; preferred is
        "auto" (use whichever call app is in front) or a fixed app key."""
        self._send = sender
        self._title = title_fn
        self.preferred = preferred

    def perform(self, action):
        """Returns {"sent": bool, "message": str, ...} - message is always
        suitable for an on-screen toast."""
        label = ACTION_LABELS.get(action, action)
        title = self._title()
        app = detect_call_app(title)
        if app is None:
            return {"sent": False, "app": None, "action": action,
                    "message": f"{label}: bring your call window to the front first"}
        if self.preferred != "auto" and app != self.preferred:
            return {"sent": False, "app": app, "action": action,
                    "message": f"{label}: front window is {APP_NAMES[app]}, not {APP_NAMES[self.preferred]}"}
        keys = HOTKEYS[app].get(action)
        if not keys:
            return {"sent": False, "app": app, "action": action,
                    "message": f"{label}: {APP_NAMES[app]} has no shortcut for this"}
        try:
            self._send(keys)
        except Exception as e:
            return {"sent": False, "app": app, "action": action, "message": f"{label}: failed ({e})"}
        return {"sent": True, "app": app, "action": action, "keys": keys,
                "message": f"{label}  ({APP_NAMES[app]})"}


class HoldToConfirm:
    """Turns a per-frame gesture stream into rare, deliberate actions.

    update(gesture, pointer, now) -> (progress, fired_action)
      progress: 0..1 of the current hold (drives the ring around the reticle)
      fired:    the action name on the frame the hold completes, else None

    Tolerates brief classifier dropouts: a real held fist is often read as
    something else for a frame or two, and resetting on every such blip made
    the hold feel dead. Misses are budgeted (max_miss_s in total per hold);
    a hand that merely FLICKERS between gestures burns the budget in a few
    frames and resets, so it still can never fire.
    """
    def __init__(self, mapping=None, hold_s=1.0, cooldown_s=3.0, still_px=70, max_miss_s=0.25):
        self.mapping = mapping or {"fist": "mute", "two_finger": "camera", "open_palm": "hand"}
        self.hold_s, self.cooldown_s, self.still_px = hold_s, cooldown_s, still_px
        self.max_miss_s = max_miss_s
        self._gesture = None
        self._since = 0.0
        self._anchor = None
        self._cool_until = 0.0
        self._miss = 0.0
        self._last_t = None

    def reset(self):
        self._gesture, self._anchor, self._miss = None, None, 0.0

    def update(self, gesture, pointer, now):
        dt = 0.0 if self._last_t is None else max(0.0, min(0.5, now - self._last_t))
        self._last_t = now
        if now < self._cool_until:
            self.reset()
            return 0.0, None
        if self._gesture is not None and gesture != self._gesture:
            self._miss += dt
            if self._miss <= self.max_miss_s:
                return min(1.0, (now - self._since) / self.hold_s), None   # brief dropout: hold on
            self.reset()
        if gesture not in self.mapping:
            self.reset()
            return 0.0, None
        if gesture != self._gesture:
            self._gesture, self._since, self._anchor, self._miss = gesture, now, pointer, 0.0
            return 0.0, None
        if pointer is not None and self._anchor is not None:
            dx, dy = pointer[0] - self._anchor[0], pointer[1] - self._anchor[1]
            if (dx * dx + dy * dy) ** 0.5 > self.still_px:
                # waving the hand about = talking, not a command: restart the hold
                self._since, self._anchor, self._miss = now, pointer, 0.0
                return 0.0, None
        progress = min(1.0, (now - self._since) / self.hold_s)
        if progress >= 1.0:
            action = self.mapping[gesture]
            self._cool_until = now + self.cooldown_s
            self.reset()
            return 1.0, action
        return progress, None

CALL_BAR_ACTIONS = (("mute", "Mute"), ("camera", "Camera"), ("hand", "Hand"), ("share", "Share"))


def call_bar_layout(screen_w, screen_h):
    """On-screen call buttons (pinch to press) for Call mode: a compact bar
    bottom-centre, kept clear of the taskbar. A pinch on a button is a
    deliberate, aimed action - the safe alternative to the held gestures."""
    bw, bh, gap = 96, 46, 10
    total = len(CALL_BAR_ACTIONS) * bw + (len(CALL_BAR_ACTIONS) - 1) * gap
    x = (screen_w - total) // 2
    y = screen_h - 70 - 34 - bh          # same 70 px taskbar-safe margin the HUD uses
    out = []
    for i, (aid, label) in enumerate(CALL_BAR_ACTIONS):
        x0 = x + i * (bw + gap)
        out.append({"id": aid, "label": label, "rect": (x0, y, x0 + bw, y + bh)})
    return out


def call_bar_button_at(buttons, pt):
    if pt is None:
        return None
    for b in buttons:
        x0, y0, x1, y1 = b["rect"]
        if x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1:
            return b["id"]
    return None
