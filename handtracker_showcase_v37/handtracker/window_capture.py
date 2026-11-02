"""
window_capture.py
------------------
Lets the overlay be visible to YOU but not to people watching your screen
share / recording.

Windows has SetWindowDisplayAffinity: a window flagged
WDA_EXCLUDEFROMCAPTURE (Windows 10 version 2004+) is drawn normally on your
monitor but is left out of screen capture, so Zoom/Meet/Teams screen share,
OBS and the Snipping Tool simply don't see it. On older Windows the best
available flag is WDA_MONITOR, which shows a black rectangle in capture
instead of the window - still better than leaking the HUD, so it's used as
a fallback. Everything else (macOS/Linux, or a failed call) reports
"unsupported" and the overlay just stays visible, never an error.

Tk quirk worth knowing: root.winfo_id() is the id of Tk's inner client
window, not the top-level frame Windows treats as "the window"; the
top-level handle is its parent (user32.GetParent) - passing the inner id to
SetWindowDisplayAffinity fails.
"""
import sys

WDA_NONE = 0x0
WDA_MONITOR = 0x1
WDA_EXCLUDEFROMCAPTURE = 0x11


def _user32():
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        return ctypes.windll.user32
    except Exception:
        return None


def toplevel_hwnd(winfo_id, user32=None):
    user32 = user32 or _user32()
    if user32 is None:
        return None
    try:
        parent = user32.GetParent(winfo_id)
    except Exception:
        parent = 0
    return parent or winfo_id


def set_capture_excluded(winfo_id, excluded, user32=None):
    """Returns one of: "excluded", "monitor" (older-Windows fallback),
    "visible" (flag cleared), or "unsupported"."""
    user32 = user32 or _user32()
    if user32 is None:
        return "unsupported"
    hwnd = toplevel_hwnd(winfo_id, user32)
    try:
        if not excluded:
            return "visible" if user32.SetWindowDisplayAffinity(hwnd, WDA_NONE) else "unsupported"
        if user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
            return "excluded"
        if user32.SetWindowDisplayAffinity(hwnd, WDA_MONITOR):
            return "monitor"
    except Exception:
        pass
    return "unsupported"
