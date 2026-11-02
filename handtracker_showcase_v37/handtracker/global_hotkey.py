"""
global_hotkey.py
-----------------
A system-wide hotkey (default Ctrl+Alt+K) that works while ANOTHER app has
focus. Its job: take the camera back after you've released it.

Why a hotkey and not a hand gesture: once the camera is released so Zoom/
Teams/OBS can use it, this app can't see your hands at all - there is
nothing for a gesture to be detected by. So the way back has to come from
somewhere else. (If you'd rather never switch at all, use the virtual
camera instead - see virtualcam.py - which lets the call and hand control
share the camera at the same time.)

Windows only (RegisterHotKey). Elsewhere start() returns False and the
rest of the app is unaffected. The key must be registered by the same
thread that reads its messages, so the whole lifecycle lives in one worker
thread; the main loop just poll()s a flag.
"""
import sys
import threading
import time

MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x1, 0x2, 0x4000
WM_HOTKEY = 0x0312
PM_REMOVE = 0x1
VK_K = 0x4B


def _user32():
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        return ctypes.windll.user32
    except Exception:
        return None


class GlobalHotkey:
    LABEL = "Ctrl+Alt+K"

    def __init__(self, vk=VK_K, modifiers=MOD_CONTROL | MOD_ALT, user32=None):
        self._vk, self._mods = vk, modifiers
        self._user32 = user32 if user32 is not None else _user32()
        self._event = threading.Event()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self.error = None

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive() and self.error is None

    def start(self):
        """True if registered. Safe to call on any platform."""
        if self._user32 is None:
            self.error = "global hotkeys are only supported on Windows"
            return False
        self._thread = threading.Thread(target=self._run, name="global-hotkey", daemon=True)
        self._thread.start()
        self._ready.wait(2.0)
        return self.error is None

    def poll(self):
        """True once per key press (consumes it)."""
        if self._event.is_set():
            self._event.clear()
            return True
        return False

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self):
        import ctypes
        from ctypes import wintypes
        u = self._user32
        try:
            if not u.RegisterHotKey(None, 1, self._mods | MOD_NOREPEAT, self._vk):
                self.error = f"{self.LABEL} is already taken by another program"
                return
        except Exception as e:
            self.error = str(e)
            return
        finally:
            self._ready.set()
        msg = wintypes.MSG()
        try:
            while not self._stop.is_set():
                while u.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                    if msg.message == WM_HOTKEY:
                        self._event.set()
                time.sleep(0.03)
        finally:
            try:
                u.UnregisterHotKey(None, 1)
            except Exception:
                pass
