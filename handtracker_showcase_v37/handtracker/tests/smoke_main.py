"""
Headless smoke test of the REAL showcase.main() loop: fake camera (through the
real threaded FrameGrabber), fake MediaPipe Hands (scripted hands), stub Tk /
pygame / pyautogui, scripted keys. Proves the v25 glue (camera thread, tracker
scheduler, fast skeleton, adaptive quality, preview, HUD) runs end to end in
AR mode with 0 -> 1 -> 2 hands, pinches, and a clean shutdown.
Run: python tests/smoke_main.py
"""
import os, sys, time, types
from unittest import mock
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SMOKE_ROOT") or os.path.dirname(HERE)      # SMOKE_ROOT=<other version> for A/B runs
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.chdir(ROOT)
import numpy as np
import fake_tk
fake_tk.install()
for m in ("pygame", "pygame.mixer"):
    sys.modules[m] = mock.MagicMock()
pg = mock.MagicMock(); pg.size.return_value = (1600, 900); pg.position.return_value = (800, 450)
sys.modules["pyautogui"] = pg

import tempfile
os.environ["HOME"] = os.environ["USERPROFILE"] = tempfile.mkdtemp()   # Settings() must not touch the real ~/.handtracker
sys.argv += os.environ.get("SMOKE_ARGS", "").split()                   # e.g. "--call", "--vcam"

import cv2
import mediapipe as mp


def synth_hand_frac(f):
    """Open pointing hand (f=0) -> pinch (f=1)."""
    pts = {0: (.5, .8), 1: (.42, .74), 2: (.38, .68), 3: (.36, .62), 4: (.40 + .10 * f, .50 - .08 * f)}
    for fng, x in zip((5, 9, 13, 17), (.47, .5, .53, .56)):
        pts[fng], pts[fng + 1], pts[fng + 2] = (x, .62), (x, .52), (x, .44)
        pts[fng + 3] = (x, .36)
    pts[8] = (.47 + .03 * f, .36 + .06 * f)
    return types.SimpleNamespace(landmark=[types.SimpleNamespace(x=pts[i][0], y=pts[i][1], z=0) for i in range(21)])

STATE = {"cpu": [], "frame": 0, "tracker_calls": {"single": 0, "dual": 0}, "keys": [], "imshow": 0}
N_FRAMES = int(os.environ.get("SMOKE_FRAMES", 160))


def hands_at(f):
    """Scripted scenario by main-loop frame number."""
    def shifted(dx, pinch):
        h = synth_hand_frac(1.0 if pinch else 0.0)
        for p in h.landmark: p.x = min(0.99, max(0.01, p.x + dx))
        return h
    if f < 5:    return []                                              # no hand yet
    if f < 60:   return [shifted(0.15, False)]                          # one hand, open
    if f < 75:   return [shifted(0.15, True)]                           # one hand, pinching
    if f < 110:  return [shifted(0.15, False), shifted(-0.25, False)]   # second hand arrives
    if f < 135:  return [shifted(0.15, True), shifted(-0.25, True)]     # two hands pinching
    return [shifted(0.15, False)]                                       # second hand leaves


class FakeHands:
    def __init__(self, max_num_hands=1, **kw):
        self.max, self.kind = max_num_hands, "dual" if max_num_hands == 2 else "single"
    def process(self, rgb):
        STATE["tracker_calls"][self.kind] += 1
        hs = hands_at(STATE["frame"])[: self.max]
        return types.SimpleNamespace(multi_hand_landmarks=hs or None, multi_handedness=None)
    def close(self): pass


_RAW = []
class FakeCap:
    n = 0
    def __init__(self): FakeCap.n += 1; self.nth = FakeCap.n; self.i = 0; self.frame = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8); self.rel = False; _RAW[:] = [self.frame]
    def read(self):
        if STATE['frame'] > N_FRAMES + 20: return False, None   # stream ends when the script does
        time.sleep(0.012); self.i += 1
        return True, self.frame.copy()
    def isOpened(self):
        if os.environ.get("SMOKE_CAM_BUSY") and self.nth > 1: return False   # another app has the camera
        return not self.rel
    def release(self): self.rel = True
    def set(self, *a): return True
    def get(self, *a): return 0


mp.solutions = types.SimpleNamespace(
    hands=types.SimpleNamespace(Hands=FakeHands, HAND_CONNECTIONS=[(0, 1)]),
    drawing_utils=types.SimpleNamespace(draw_landmarks=lambda *a, **k: None))

import ar_overlay


class FakePhoto:
    """Stands in for ImageTk.PhotoImage (needs a real Tk). Same surface
    the preview/drawing-canvas code uses, master= included: real
    ImageTk.PhotoImage takes an optional master=<widget> to bind the
    image to the right Tk interpreter (this app opens several separate
    tk.Tk() roots - an unbound PhotoImage can silently attach to the
    wrong one and crash on use, which is exactly the real bug this
    master= argument exists to avoid - see main_overlay.
    _draw_drawing_canvas's own note). Accepted and ignored here, same as
    real Tkinter accepts it without changing this fake's behaviour."""
    made = pastes = 0
    def __init__(self, image, master=None): FakePhoto.made += 1; self.size = image.size
    def paste(self, image):
        assert image.size == self.size, "preview size must stay constant for paste()"
        FakePhoto.pastes += 1


ar_overlay.ImageTk = types.SimpleNamespace(PhotoImage=FakePhoto)
import main_overlay
main_overlay.ImageTk = types.SimpleNamespace(PhotoImage=FakePhoto)

if os.environ.get("SMOKE_BREAK_OVERLAY"):
    # Deliberately breaks MainOverlay.update() to prove showcase.py's
    # try/except around that call (see the "Main overlay draw failed"
    # log line) actually stops a rendering failure in there from taking
    # the whole tracking loop down with it - the exact crash a real user
    # hit (an unhandled TclError from the wrong-Tk-interpreter ImageTk
    # bug). Lives behind an env var (not a normal SMOKE_KEY script
    # branch) since it needs to patch the class before showcase.main()
    # ever constructs an instance of it.
    def _boom(self, *a, **k):
        raise RuntimeError("synthetic overlay rendering failure")
    main_overlay.MainOverlay.update = _boom


# ---- spies / fakes for the video-call features -------------------------------
STATE["extras"] = set()
STATE["toasts"] = []
STATE["capture_flags"] = []
_real_update = main_overlay.MainOverlay.update
def _spy_update(self, *a, **k):
    ex = k.get("extras") or {}
    STATE["extras"].update(ex.keys())
    if ex.get("toast") and (not STATE["toasts"] or STATE["toasts"][-1] != ex["toast"]):
        STATE["toasts"].append(ex["toast"])         # each distinct message once, in order
    return _real_update(self, *a, **k)
main_overlay.MainOverlay.update = _spy_update

import window_capture
def _fake_capture(winfo_id, excluded, user32=None):
    STATE["capture_flags"].append(bool(excluded))
    return "excluded"
window_capture.set_capture_excluded = _fake_capture

import virtualcam
class _FakeVCam:
    def __init__(self, w, h, fps):
        self.frames, self.closed, self.device = [], False, "Smoke Virtual Camera"
    def send(self, rgb):
        if len(self.frames) < 40: self.frames.append(rgb.copy())
    def sleep_until_next_frame(self): time.sleep(0.004)
    def close(self): self.closed = True
class _FakeVCamBackend:
    cams = []
    @classmethod
    def Camera(cls, w, h, fps):
        c = _FakeVCam(w, h, fps); cls.cams.append(c); return c
if os.environ.get("SMOKE_VCAM"):
    virtualcam._pvc = _FakeVCamBackend
    sys.argv.append("--vcam")

import showcase
import callapp

# ---- drive the video-call paths deterministically -------------------------
STATE["hotkeys"] = []
showcase._press_hotkey = lambda keys: STATE["hotkeys"].append(tuple(keys))
if os.environ.get("SMOKE_FOREGROUND") is not None:
    callapp.foreground_title = lambda user32=None: os.environ["SMOKE_FOREGROUND"]
if os.environ.get("SMOKE_HOLD_S"):
    _Hold = callapp.HoldToConfirm
    callapp.HoldToConfirm = lambda *a, **k: _Hold(*a, **{**k, "hold_s": float(os.environ["SMOKE_HOLD_S"])})

class _FakeGlobalHotkey:
    LABEL, error, polls = "Ctrl+Alt+K", None, 0
    def start(self): return True
    def stop(self): pass
    def poll(self):
        _FakeGlobalHotkey.polls += 1
        at = int(os.environ.get("SMOKE_HOTKEY_AT", "0"))
        return bool(at) and _FakeGlobalHotkey.polls == at
showcase.GlobalHotkey = _FakeGlobalHotkey
real_classify = showcase.classify_static_gesture
# Scripted gesture reading: thumb-down (frames 8-20) enables control, otherwise "pointing".
# (A held "open_palm" would, after ~1 s of wall-clock, open the mode menu and a pinch would pick a tile.)
_forced_gesture = os.environ.get("SMOKE_FORCE_GESTURE")    # e.g. "fist": hold ONE gesture the whole run
if _forced_gesture:
    showcase.classify_static_gesture = lambda lm: _forced_gesture
else:
    showcase.classify_static_gesture = lambda lm: "thumb_down" if 8 <= STATE["frame"] <= 20 else "pointing"
cv2.VideoCapture = lambda *a, **k: FakeCap()


class SpySched(getattr(showcase, 'TwoHandScheduler', object)):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); STATE["sched"] = self


class SpyOverlay(showcase.AROverlay):
    """Real overlay + a few 3D objects already placed, so the loop renders, hovers and LODs them."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        from ar3d import Surface3D, BarChart3D, Shape3D
        sc = self.scene3d
        sc.objects.clear()
        for o in (Surface3D(500, 450), Surface3D(900, 450), BarChart3D(1250, 450), Shape3D("torus", 1450, 250)):
            sc.add(o)
        STATE["overlay"] = self


if hasattr(showcase, "TwoHandScheduler"):
    showcase.TwoHandScheduler = SpySched
showcase.AROverlay = SpyOverlay

# Record every draw_hud call's (show_help, help_hidden_note) regardless of
# SMOKE_SLOW, so a test can assert on what the help panel actually did
# frame to frame - not just the final perf level - including under the
# SMOKE_SLOW path below, which still needs to go through this recorder
# first and then add its artificial per-frame cost on top.
STATE["help_trace"] = []
_real_hud = showcase.draw_hud
def recording_hud(frame, mode_key, enabled, gesture, confidence, fps, show_help, ai_label=None,
                   help_hidden_note=None):
    STATE["help_trace"].append({
        "frame": STATE["frame"], "show_help": bool(show_help),
        "note": help_hidden_note, "perf_level": STATE.get("perf_obj").level if STATE.get("perf_obj") else None,
    })
    return _real_hud(frame, mode_key, enabled, gesture, confidence, fps, show_help,
                      ai_label=ai_label, help_hidden_note=help_hidden_note)
showcase.draw_hud = recording_hud
if os.environ.get("SMOKE_SLOW"):                       # simulate an overloaded machine: +45 ms of "work" per frame
    _recording_hud = showcase.draw_hud
    def slow_hud(*a, **k):
        time.sleep(0.045); return _recording_hud(*a, **k)
    showcase.draw_hud = slow_hud
showcase.AIGestureEngine = mock.MagicMock()

# Grab the live PerfGovernor instance the real main() constructs, so the
# recorder above (and the test afterwards) can read its .level - main()
# builds it internally, so hook the class instead of reaching into a
# local variable we have no access to.
_RealPerfGovernor = showcase.PerfGovernor
class SpyPerfGovernor(_RealPerfGovernor):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        STATE["perf_obj"] = self
showcase.PerfGovernor = SpyPerfGovernor


def fake_waitKey(_ms):
    STATE["cpu"].append(time.process_time())
    STATE["frame"] += 1
    f = STATE["frame"]
    if f == 3: return ord("6")                                          # switch to AR mode
    for suffix in ("", "2", "3"):
        key_at = os.environ.get(f"SMOKE_KEY_AT{suffix}")
        if key_at and f == int(key_at):
            return ord(os.environ.get(f"SMOKE_KEY{suffix}", "h"))
    return 0xFF if f < N_FRAMES else ord("q")


cv2.waitKey = fake_waitKey
cv2.imshow = lambda *a, **k: STATE.__setitem__("imshow", STATE["imshow"] + 1)
cv2.destroyAllWindows = lambda: None


def run():
    t0 = time.perf_counter()
    showcase.main()
    return time.perf_counter() - t0


if __name__ == "__main__":
    import io, json, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dt = run()
    out = buf.getvalue()
    sc = STATE["overlay"].scene3d
    print(out)
    print("SMOKE_JSON " + json.dumps({
        "frames": STATE["frame"], "seconds": round(dt, 2), "imshow": STATE["imshow"],
        "trackers": STATE["tracker_calls"],
        "sched": dict(STATE["sched"].stats) if "sched" in STATE else {}, "sched_state": STATE["sched"].state if "sched" in STATE else None,
        "cpu_ms_per_frame": round((STATE["cpu"][-1] - STATE["cpu"][30]) / (len(STATE["cpu"]) - 31) * 1000, 2),
        "photo_made": FakePhoto.made, "photo_pastes": FakePhoto.pastes,
        "control_on": "Control ON" in out, "perf_lines": [l for l in out.splitlines() if l.startswith("[PERF]")],
        "objects": len(sc.objects), "lod": getattr(sc, "lod", None), "budget": getattr(sc, "item_budget", None),
        "mode_ar": True, "help_trace": STATE["help_trace"],
        "extras_seen": sorted(STATE["extras"]), "toasts": STATE["toasts"][:8],
        "hotkeys": STATE["hotkeys"], "cap_opens": FakeCap.n,
        "capture_flags": STATE["capture_flags"],
        "vcam_frames": sum(len(c.frames) for c in _FakeVCamBackend.cams),
        "vcam_closed": all(c.closed for c in _FakeVCamBackend.cams) if _FakeVCamBackend.cams else None,
        "vcam_clean": bool(_FakeVCamBackend.cams and _FakeVCamBackend.cams[0].frames and np.array_equal(
            _FakeVCamBackend.cams[0].frames[0], cv2.cvtColor(_RAW[0], cv2.COLOR_BGR2RGB))) if "_RAW" in globals() else None}))
