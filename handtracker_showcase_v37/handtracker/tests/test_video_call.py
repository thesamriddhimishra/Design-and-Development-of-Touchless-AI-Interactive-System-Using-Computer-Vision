"""Tests for the video-call support modules (no camera, display or call app
needed). Run: python tests/test_video_call.py"""
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import callapp
import global_hotkey
import settings
import virtualcam
import window_capture


# ------------------------------------------------------------------ settings
def test_settings_roundtrip_defaults_and_bad_files():
    path = os.path.join(tempfile.mkdtemp(), "sub", "settings.json")
    s = settings.Settings(path)
    assert s["share_safe"] == "auto" and s["virtual_cam"] is False
    s.set("virtual_cam", True)
    s.set("share_safe", "on")
    s2 = settings.Settings(path)
    assert s2["virtual_cam"] is True and s2["share_safe"] == "on"

    # wrong types / unknown values / unknown keys are ignored, never crash
    with open(path, "w") as f:
        json.dump({"virtual_cam": "yes", "share_safe": "banana", "mystery": 1, "call_app": "zoom"}, f)
    s3 = settings.Settings(path)
    assert s3["virtual_cam"] is False and s3["share_safe"] == "auto" and s3["call_app"] == "zoom"
    with open(path, "w") as f:
        f.write("{not json")
    assert settings.Settings(path)["share_safe"] == "auto"

    try:
        s3.set("share_safe", "maybe")
        assert False
    except ValueError:
        pass
    assert s3.cycle("share_safe") in ("on", "off")
    assert s3.toggle("virtual_cam") is True


# ------------------------------------------------------------ window capture
class _FakeUser32:
    def __init__(self, supports_exclude=True, supports_monitor=True):
        self.calls, self.ex, self.mon = [], supports_exclude, supports_monitor
    def GetParent(self, h): return 777 if h == 42 else 0
    def SetWindowDisplayAffinity(self, hwnd, flag):
        self.calls.append((hwnd, flag))
        if flag == window_capture.WDA_EXCLUDEFROMCAPTURE: return self.ex
        if flag == window_capture.WDA_MONITOR: return self.mon
        return True


def test_overlay_is_excluded_from_capture_using_the_toplevel_handle():
    u = _FakeUser32()
    assert window_capture.set_capture_excluded(42, True, u) == "excluded"
    assert u.calls == [(777, window_capture.WDA_EXCLUDEFROMCAPTURE)], \
        "must target Tk's top-level (parent of winfo_id), not the inner client window"
    assert window_capture.set_capture_excluded(42, False, u) == "visible"
    assert u.calls[-1] == (777, window_capture.WDA_NONE)


def test_older_windows_falls_back_to_monitor_flag_and_other_platforms_are_unsupported():
    u = _FakeUser32(supports_exclude=False)
    assert window_capture.set_capture_excluded(42, True, u) == "monitor"
    u2 = _FakeUser32(supports_exclude=False, supports_monitor=False)
    assert window_capture.set_capture_excluded(42, True, u2) == "unsupported"
    assert window_capture.set_capture_excluded(42, True, None) in ("unsupported", "excluded", "monitor")

    class Boom:
        def GetParent(self, h): raise OSError
        def SetWindowDisplayAffinity(self, *a): raise OSError
    assert window_capture.set_capture_excluded(42, True, Boom()) == "unsupported"


# ------------------------------------------------------------- virtual camera
class _FakeCam:
    def __init__(self, w, h, fps, fail_after=None):
        self.w, self.h, self.fps, self.frames, self.closed = w, h, fps, [], False
        self.device, self.fail_after = "Fake Virtual Cam", fail_after
    def send(self, rgb):
        if self.fail_after is not None and len(self.frames) >= self.fail_after:
            raise RuntimeError("device removed")
        self.frames.append(rgb.copy())
    def sleep_until_next_frame(self): time.sleep(0.005)
    def close(self): self.closed = True


class _FakeBackend:
    def __init__(self, **kw): self.kw, self.cams = kw, []
    def Camera(self, w, h, fps):
        cam = _FakeCam(w, h, fps, **self.kw); self.cams.append(cam); return cam


def _wait(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond(): return True
        time.sleep(0.01)
    return False


def test_virtual_camera_publishes_clean_rgb_frames_and_keeps_republishing():
    be = _FakeBackend()
    vc = virtualcam.VirtualCamera(be)
    ok, msg = vc.start(64, 48, 30)
    assert ok and "Fake Virtual Cam" in msg and vc.active
    bgr = np.zeros((48, 64, 3), np.uint8); bgr[:, :, 2] = 200      # red in BGR
    vc.send(bgr)
    cam = be.cams[0]
    assert _wait(lambda: len(cam.frames) >= 3), "should keep re-sending the latest frame at fps"
    f = cam.frames[0]
    assert f.shape == (48, 64, 3) and tuple(f[0, 0]) == (200, 0, 0), "BGR must be converted to RGB"
    vc.send(np.zeros((24, 32, 3), np.uint8))                         # wrong size gets resized, not rejected
    assert _wait(lambda: cam.frames[-1].shape == (48, 64, 3))
    vc.stop()
    assert cam.closed and not vc.active


def test_virtual_camera_never_blocks_the_caller_and_drops_superseded_frames():
    class Slow(_FakeCam):
        def sleep_until_next_frame(self): time.sleep(0.05)
    class SlowBackend(_FakeBackend):
        def Camera(self, w, h, fps):
            cam = Slow(w, h, fps); self.cams.append(cam); return cam
    vc = virtualcam.VirtualCamera(SlowBackend())
    vc.start(32, 32, 30)
    t0 = time.perf_counter()
    for _ in range(200):
        vc.send(np.zeros((32, 32, 3), np.uint8))
    assert time.perf_counter() - t0 < 0.5, "send() must be non-blocking even with a slow driver"
    assert vc.dropped > 0
    vc.stop()


def test_virtual_camera_reports_missing_driver_and_survives_device_loss():
    vc = virtualcam.VirtualCamera(backend=None)
    vc._backend = None
    ok, msg = vc.start(64, 48)
    assert not ok and "pyvirtualcam" in msg and "OBS" in msg and not vc.active

    class Bad:
        def Camera(self, *a): raise RuntimeError("no virtual camera installed")
    ok, msg = virtualcam.VirtualCamera(Bad()).start(64, 48)
    assert not ok and "no virtual camera installed" in msg

    be = _FakeBackend(fail_after=2)
    vc = virtualcam.VirtualCamera(be)
    vc.start(16, 16)
    vc.send(np.zeros((16, 16, 3), np.uint8))
    assert _wait(lambda: not vc.active), "a vanishing device must stop the worker, not crash the app"
    assert "device removed" in vc.error
    vc.send(np.zeros((16, 16, 3), np.uint8))               # harmless after failure
    vc.stop()


def test_virtual_camera_setup_diagnosis_names_what_is_missing():
    rows = virtualcam.diagnose(backend=None)
    pkg = rows[0]
    saved, virtualcam._pvc = virtualcam._pvc, None
    try:
        rows = virtualcam.diagnose()
        assert rows[0][1] is False and "pip install pyvirtualcam" in rows[0][2]
        text = virtualcam.format_diagnosis(rows)
        assert "NO" in text and "pyvirtualcam" in text and "OBS" in text
    finally:
        virtualcam._pvc = saved
    assert virtualcam.diagnose(backend=object())[0][1] is True


# ------------------------------------------------------------------ call apps
def test_call_app_detection_from_window_titles():
    d = callapp.detect_call_app
    assert d("Zoom Meeting") == "zoom" and d("Zoom Workplace") == "zoom"
    assert d("Meet - abc-defg-hij - Google Chrome") == "meet"
    assert d("Meet \u2013 abc-defg-hij \u2013 Microsoft Edge") == "meet"
    assert d("Weekly sync | Microsoft Teams") == "teams"
    for not_a_call in ("Untitled - Notepad", "Zoom into the details - Word", "", None,
                       "Meeting notes.docx - Word", "Chrome"):
        assert d(not_a_call) is None, not_a_call


def test_shortcut_is_only_sent_when_a_call_app_is_in_front():
    sent = []
    def make(title, preferred="auto"):
        return callapp.CallController(lambda keys: sent.append(keys), lambda: title, preferred)

    r = make("Untitled - Notepad").perform("mute")
    assert not r["sent"] and not sent and "call window" in r["message"], \
        "must never fire Ctrl+D etc. into an unrelated window"
    assert make("Zoom Meeting").perform("mute")["sent"] and sent[-1] == ("alt", "a")
    assert make("Meet - abc-defg-hij - Google Chrome").perform("camera")["sent"] and sent[-1] == ("ctrl", "e")
    assert make("Chat | Microsoft Teams").perform("hand")["sent"] and sent[-1] == ("ctrl", "shift", "k")

    n = len(sent)
    r = make("Meet - abc-defg-hij - Google Chrome").perform("share")
    assert not r["sent"] and len(sent) == n and "no shortcut" in r["message"]
    r = make("Zoom Meeting", preferred="teams").perform("mute")
    assert not r["sent"] and len(sent) == n, "a pinned app must not act on a different app's window"

    def boom(keys): raise RuntimeError("keyboard blocked")
    r = callapp.CallController(boom, lambda: "Zoom Meeting").perform("mute")
    assert not r["sent"] and "failed" in r["message"]


def test_leaving_a_call_is_not_mappable():
    for app, table in callapp.HOTKEYS.items():
        assert "leave" not in table and "end" not in table, app


def test_hold_to_confirm_needs_a_still_held_gesture_and_fires_once():
    h = callapp.HoldToConfirm(hold_s=1.0, cooldown_s=3.0, still_px=70)
    t, fired = 0.0, []
    prog = []
    for i in range(40):                                   # ~1.3 s of a held, still fist
        t += 1 / 30
        p, a = h.update("fist", (500, 500), t)
        prog.append(p)
        if a: fired.append((t, a))
    assert [a for _, a in fired] == ["mute"], "fires exactly once per hold"
    upto = prog[:prog.index(1.0) + 1]
    assert upto == sorted(upto) and upto[0] < 0.1, "the ring fills steadily from empty up to full"
    assert 0.9 < fired[0][0] < 1.2

    # during the cooldown a continued hold does nothing
    for _ in range(60):
        t += 1 / 30
        assert h.update("fist", (500, 500), t)[1] is None


def test_talking_gestures_never_fire_an_action():
    h = callapp.HoldToConfirm(hold_s=1.0, still_px=70)
    t = 0.0
    for i in range(300):                                   # 10 s of a fist that keeps moving (gesturing)
        t += 1 / 30
        a = h.update("fist", (500 + 120 * (i % 2), 500), t)[1]
        assert a is None, "a moving hand is talking, not commanding"
    for i in range(300):                                   # flickering between gestures
        t += 1 / 30
        a = h.update(["fist", "open_palm", "pointing"][i % 3], (500, 500), t)[1]
        assert a is None
    # unmapped gestures reset any hold
    h.update("fist", (500, 500), t + 0.1)
    assert h.update("pointing", (500, 500), t + 0.6) == (0.0, None), "a long unmapped gesture ends the hold"


def test_a_real_fist_with_brief_classifier_dropouts_still_fires():
    """Regression: resetting on every misread frame made the hold feel
    dead - a real fist is routinely read as something else for a frame or
    two. A few single-frame dropouts must not stop it..."""
    h = callapp.HoldToConfirm(hold_s=1.0, still_px=70)
    t, fired = 0.0, []
    for i in range(60):
        t += 1 / 30
        g = None if i in (7, 8, 19, 33) else "fist"        # 4 dropout frames (~0.13 s) in ~1.3 s
        a = h.update(g, (500, 500), t)[1]
        if a: fired.append(a)
    assert fired == ["mute"], fired


def test_a_long_dropout_or_constant_flicker_still_resets_the_hold():
    h = callapp.HoldToConfirm(hold_s=1.0, still_px=70)
    t = 0.0
    for i in range(20):
        t += 1 / 30
        h.update("fist", (500, 500), t)
    for i in range(24):                                    # 0.8 s of something else: way over budget
        t += 1 / 30
        h.update("pointing", (500, 500), t)
    fired = []
    for i in range(20):                                    # fist again, but the hold had to restart
        t += 1 / 30
        a = h.update("fist", (500, 500), t)[1]
        if a: fired.append(a)
    assert not fired, "an interrupted hold must start over, not resume"
    # flicker: 1 fist frame in 3 for 10 s never accumulates into a hold
    h2 = callapp.HoldToConfirm(hold_s=1.0, still_px=70)
    t = 0.0
    for i in range(300):
        t += 1 / 30
        assert h2.update("fist" if i % 3 == 0 else "pointing", (500, 500), t)[1] is None


def test_foreground_title_reads_the_focused_window_via_user32():
    class U:
        def GetForegroundWindow(self): return 1
        def GetWindowTextLengthW(self, h): return 12
        def GetWindowTextW(self, h, buf, n): buf.value = "Zoom Meeting"
    assert callapp.foreground_title(U()) == "Zoom Meeting"
    class Bad:
        def GetForegroundWindow(self): raise OSError
    assert callapp.foreground_title(Bad()) is None


# ------------------------------------------------------------- global hotkey
class _FakeHotkeyUser32:
    def __init__(self, register_ok=True):
        self.register_ok, self.registered, self.unregistered = register_ok, [], 0
        self.pending = 0
        self.threads = set()
    def RegisterHotKey(self, hwnd, id_, mods, vk):
        import threading
        self.threads.add(threading.get_ident())
        self.registered.append((id_, mods, vk))
        return 1 if self.register_ok else 0
    def PeekMessageW(self, ref, *a):
        import threading
        self.threads.add(threading.get_ident())
        if self.pending:
            self.pending -= 1
            ref._obj.message = global_hotkey.WM_HOTKEY
            return 1
        return 0
    def UnregisterHotKey(self, hwnd, id_):
        self.unregistered += 1


def test_global_hotkey_registers_reports_presses_once_and_cleans_up():
    u = _FakeHotkeyUser32()
    hk = global_hotkey.GlobalHotkey(user32=u)
    assert hk.start() and hk.active
    (id_, mods, vk), = u.registered
    assert vk == global_hotkey.VK_K and mods & global_hotkey.MOD_CONTROL and mods & global_hotkey.MOD_ALT
    assert not hk.poll()
    u.pending = 1
    assert _wait(hk.poll), "a WM_HOTKEY message must surface as a press"
    assert not hk.poll(), "each press is reported exactly once"
    hk.stop()
    assert u.unregistered == 1
    assert len(u.threads) == 1, "register + message loop must share one thread (a Windows requirement)"


def test_global_hotkey_reports_a_taken_key_and_unsupported_platforms():
    hk = global_hotkey.GlobalHotkey(user32=_FakeHotkeyUser32(register_ok=False))
    assert not hk.start() and "already taken" in hk.error and not hk.active
    hk2 = global_hotkey.GlobalHotkey(user32=None)
    hk2._user32 = None
    assert not hk2.start() and "Windows" in hk2.error and not hk2.poll()
    hk2.stop()


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                f(); print("PASS ", n)
            except Exception as e:
                fails += 1
                import traceback
                print("FAIL ", n, repr(e)); traceback.print_exc(limit=4)
    sys.exit(1 if fails else 0)
