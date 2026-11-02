"""
desktop_control.py
--------------------
Milestone 3: turns decisions into real OS mouse/keyboard actions via
pyautogui. Kept as a thin adapter - see desktop_interaction.py for the
actual decision logic (which gesture means which action), so that logic
can be unit tested without a real mouse, keyboard, or display.
"""
import config

try:
    import pyautogui
    pyautogui.FAILSAFE = True  # slam the mouse into a screen corner to abort - keep this on
    pyautogui.PAUSE = config.PYAUTOGUI_PAUSE
    _PYAUTOGUI_AVAILABLE = True
except Exception:
    _PYAUTOGUI_AVAILABLE = False


class DesktopControlAborted(Exception):
    """Raised when pyautogui's corner fail-safe trips (or another OS-level
    control error occurs), instead of letting that exception crash the
    whole program. main_desktop.py catches this, turns desktop control
    off, and keeps running - camera and overlay stay up so the user can
    re-enable with an open-palm gesture rather than restarting."""


def get_screen_size():
    if _PYAUTOGUI_AVAILABLE:
        return tuple(pyautogui.size())
    return (1920, 1080)  # fallback so the rest of the pipeline still runs without a display


def map_to_screen(norm_point, screen_size, margin=None):
    """Maps a normalized (0-1, 0-1) camera-space point to screen pixel
    coordinates, cropping a margin off each edge first so the user
    doesn't have to stretch their hand to the very edge of the camera
    frame to reach the edge of the screen. Pure function - no OS calls -
    so it's directly unit-testable.

    The result is also inset a few pixels from the absolute screen edge
    (config.CURSOR_EDGE_INSET_PX). Without this, a hand near the crop
    margin maps to the literal (0, 0)-type corner pixel - which is
    exactly the pixel pyautogui's fail-safe watches for, so ordinary
    tracking noise near the edge of the frame could self-trigger an
    emergency abort. The physical mouse can still reach the true corner
    to trigger a deliberate fail-safe at any time; this only keeps our
    own virtual cursor from landing there by accident."""
    margin = config.CURSOR_MARGIN_RATIO if margin is None else margin
    nx, ny = norm_point
    lo, hi = margin, 1 - margin
    span = max(1e-6, hi - lo)
    nx = (min(max(nx, lo), hi) - lo) / span
    ny = (min(max(ny, lo), hi) - lo) / span
    screen_w, screen_h = screen_size
    inset = config.CURSOR_EDGE_INSET_PX
    x = inset + nx * max(1, screen_w - 2 * inset)
    y = inset + ny * max(1, screen_h - 2 * inset)
    return int(x), int(y)


class DesktopController:
    """Thin wrapper over pyautogui. All real OS side effects live here,
    behind the same small interface as DryRunController below, so
    desktop_interaction.py's decision logic never needs to know which
    one it's talking to. Every pyautogui call is guarded: a tripped
    fail-safe (or any other pyautogui error) is converted into
    DesktopControlAborted rather than propagating as a raw exception
    that would take the whole program down with it."""

    def __init__(self):
        if not _PYAUTOGUI_AVAILABLE:
            raise RuntimeError(
                "pyautogui is not installed (or no display is available). "
                "Install it with 'pip install pyautogui' and run this on "
                "your desktop, not in a headless environment."
            )
        self.screen_size = get_screen_size()

    def _guarded(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except pyautogui.FailSafeException as e:
            raise DesktopControlAborted(
                "Mouse hit a screen corner - pyautogui's safety fail-safe stopped desktop "
                "control. Make an open-palm gesture to turn it back on."
            ) from e
        except Exception as e:
            raise DesktopControlAborted(f"Desktop control action failed: {e}") from e

    def move_cursor(self, norm_point):
        x, y = map_to_screen(norm_point, self.screen_size)
        self._guarded(pyautogui.moveTo, x, y, _pause=False)

    def click(self):
        self._guarded(pyautogui.click)

    def right_click(self):
        self._guarded(pyautogui.rightClick)

    def mouse_down(self):
        self._guarded(pyautogui.mouseDown)

    def mouse_up(self):
        self._guarded(pyautogui.mouseUp)

    def scroll(self, notches):
        """Positive = scroll up, negative = down (mouse-wheel notches)."""
        self._guarded(pyautogui.scroll, int(notches))

    def press_key(self, key_or_combo):
        keys = list(key_or_combo) if isinstance(key_or_combo, (list, tuple)) else [key_or_combo]
        self._guarded(pyautogui.hotkey, *keys)


class DryRunController:
    """Drop-in replacement for DesktopController that only logs what it
    would have done. Used automatically when pyautogui/a display isn't
    available (this environment), and useful on a real machine too for
    a first trial run where you want to see decisions before your mouse
    actually starts moving."""

    def __init__(self):
        self.screen_size = get_screen_size()
        self.log = []

    def _record(self, *entry):
        self.log.append(entry)
        del self.log[:-200]

    def move_cursor(self, norm_point):
        self._record("move_cursor", map_to_screen(norm_point, self.screen_size))

    def click(self):
        self._record("click")

    def right_click(self):
        self._record("right_click")

    def mouse_down(self):
        self._record("mouse_down")

    def mouse_up(self):
        self._record("mouse_up")

    def scroll(self, notches):
        self._record("scroll", notches)

    def press_key(self, key_or_combo):
        self._record("press_key", key_or_combo)
