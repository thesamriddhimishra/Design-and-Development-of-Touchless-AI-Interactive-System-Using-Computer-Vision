"""
settings.py
------------
Small persisted settings store (JSON), so choices like "publish a virtual
camera" or "hide the HUD from screen sharing" survive a restart instead of
resetting every launch.

Stored in the user's home (~/.handtracker/settings.json) so it works no
matter where the project folder lives; falls back to the project folder if
the home directory isn't writable. Unknown keys in the file are ignored and
missing ones fall back to DEFAULTS, so old/hand-edited files never crash
the app. Saves are atomic (write temp + replace) so a crash mid-save can't
leave a half-written file.
"""
import json
import os

DEFAULTS = {
    # "auto": hide the main HUD overlay from screen-share/recording except in
    # Drawing mode (where the strokes ARE the point - annotating for an
    # audience). "on": always hide it. "off": never hide it.
    "share_safe": "auto",
    "virtual_cam": False,        # publish clean camera video to a virtual camera device
    "call_profile": False,       # lighter CPU profile for running alongside a call app
    "call_app": "auto",          # "auto" | "zoom" | "meet" | "teams"
    "tutorial_done": False,      # only ever used to show a tick - the tutorial never auto-starts
}

_VALID = {
    "share_safe": ("auto", "on", "off"),
    "call_app": ("auto", "zoom", "meet", "teams"),
}


def default_path():
    home = os.path.join(os.path.expanduser("~"), ".handtracker")
    return os.path.join(home, "settings.json")


class Settings:
    def __init__(self, path=None):
        self.path = path or default_path()
        self.data = dict(DEFAULTS)
        self.load()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError):
            return False
        if not isinstance(raw, dict):
            return False
        for k, default in DEFAULTS.items():
            if k in raw and type(raw[k]) is type(default):
                if k in _VALID and raw[k] not in _VALID[k]:
                    continue
                self.data[k] = raw[k]
        return True

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)
            return True
        except OSError:
            return False

    def __getitem__(self, key):
        return self.data[key]

    def set(self, key, value, save=True):
        if key not in DEFAULTS:
            raise KeyError(key)
        if key in _VALID and value not in _VALID[key]:
            raise ValueError(f"{key} must be one of {_VALID[key]}")
        self.data[key] = value
        if save:
            self.save()

    def cycle(self, key):
        """Advance an enumerated setting to its next value (returns it)."""
        opts = _VALID[key]
        nxt = opts[(opts.index(self.data[key]) + 1) % len(opts)]
        self.set(key, nxt)
        return nxt

    def toggle(self, key):
        self.set(key, not self.data[key])
        return self.data[key]
