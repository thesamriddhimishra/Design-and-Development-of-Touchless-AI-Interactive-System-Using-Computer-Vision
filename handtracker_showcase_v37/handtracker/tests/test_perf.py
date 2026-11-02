"""Run: python tests/test_perf.py  - scheduler, governor, camera grabber, fast drawing."""
import os, sys, threading, time, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

from tracker_sched import DUAL, SINGLE, TwoHandScheduler
from perf import LEVELS, PerfGovernor, draw_hand_fast
from camera_stream import FrameGrabber


# ------------------------------------------------------------- scheduler ---
class FakeTracker:
    """Returns however many hands `script()` says; counts calls."""
    def __init__(self, name, script):
        self.name, self.script, self.calls = name, script, 0
    def process(self, rgb):
        self.calls += 1
        n = self.script()
        return types.SimpleNamespace(multi_hand_landmarks=[object() for _ in range(n)] or None)


def sched(hands_now, **kw):
    """hands_now: dict(n=..) mutated by the test. Both fake trackers report it
    (the 1-hand tracker can never report more than 1)."""
    single = FakeTracker("single", lambda: min(1, hands_now["n"]))
    dual = FakeTracker("dual", lambda: hands_now["n"])
    return TwoHandScheduler(single, dual, **kw), single, dual


def test_one_hand_mostly_uses_the_cheap_tracker():
    st = {"n": 1}
    s, single, dual = sched(st, probe_every=6)
    for _ in range(61):
        s.process(None)
    assert s.state == SINGLE
    assert dual.calls <= 11 and single.calls >= 50, (single.calls, dual.calls)   # vs 61 dual calls before


def test_no_hands_never_probes():
    st = {"n": 0}
    s, single, dual = sched(st)
    for _ in range(100):
        s.process(None)
    assert dual.calls == 0 and single.calls == 100


def test_second_hand_found_within_probe_interval_then_tracked_every_frame():
    st = {"n": 1}
    s, single, dual = sched(st, probe_every=6)
    for _ in range(10):
        s.process(None)
    st["n"] = 2
    frames = 0
    while s.state != DUAL and frames < 20:
        r = s.process(None); frames += 1
    assert s.state == DUAL and frames <= 6, frames
    before = dual.calls
    for _ in range(20):
        assert len(s.process(None).multi_hand_landmarks) == 2
    assert dual.calls == before + 20


def test_returns_to_single_after_second_hand_leaves_and_stays_there():
    st = {"n": 2}
    s, single, dual = sched(st, probe_every=6, back_after=12)
    s.state = DUAL
    for _ in range(5):
        s.process(None)
    st["n"] = 1
    for _ in range(11):
        s.process(None)
    assert s.state == DUAL                                   # not yet: tolerant of a brief occlusion
    s.process(None)
    assert s.state == SINGLE
    st["n"] = 1
    c = dual.calls
    for _ in range(30):
        s.process(None)
    assert dual.calls - c <= 6                               # only probes


def test_brief_second_hand_dropout_does_not_bounce_states():
    st = {"n": 2}
    s, _, _ = sched(st, back_after=12)
    s.state = DUAL
    for k in range(60):
        st["n"] = 1 if k % 10 == 0 else 2                    # 1-frame dropouts
        s.process(None)
    assert s.state == DUAL


def test_probe_miss_reuses_the_last_result_so_the_hand_never_flickers_out():
    seq = {"probe_miss": False}
    single = FakeTracker("s", lambda: 1)
    dual = FakeTracker("d", lambda: 0)                       # probe finds nothing at all
    s = TwoHandScheduler(single, dual, probe_every=3)
    outs = [len(s.process(None).multi_hand_landmarks or []) for _ in range(12)]
    assert all(o == 1 for o in outs), outs


# -------------------------------------------------------------- governor ---
def feed(g, ms, seconds, t0, fps=30):
    t = t0
    changed = []
    for _ in range(int(seconds * fps)):
        t += 1 / fps
        if g.update(ms, t):
            changed.append(round(t - t0, 1))
    return t, changed


def test_governor_degrades_stepwise_when_overloaded_and_recovers_when_not():
    g = PerfGovernor(target_fps=30)
    t, ch = feed(g, 45.0, 12, 0.0)                          # 45 ms compute vs ~28 ms budget
    assert g.level == 3 and len(ch) == 3
    assert all(b - a >= 2.4 for a, b in zip(ch, ch[1:])), ch   # rate limited
    t, ch = feed(g, 10.0, 40, t)                             # plenty of headroom
    assert g.level == 0 and len(ch) == 3


def test_governor_ignores_single_spikes_and_holds_in_the_comfort_band():
    g = PerfGovernor(target_fps=30)
    t = 0.0
    for k in range(300):
        t += 1 / 30
        g.update(200.0 if k % 100 == 50 else 12.0, t)        # one 200 ms hitch every ~3 s
    assert g.level == 0
    g2 = PerfGovernor(target_fps=30)
    feed(g2, 22.0, 30, 0.0)                                   # under budget (25.5) but not under 60% of it
    assert g2.level == 0
    g3 = PerfGovernor(target_fps=30); g3.level = 2
    feed(g3, 22.0, 30, 0.0)
    assert g3.level == 2                                      # no flapping in the middle band


def test_governor_levels_are_monotonically_cheaper_and_disable_works():
    for a, b in zip(LEVELS, LEVELS[1:]):
        assert b["item_budget"] < a["item_budget"] and b["preview_every"] >= a["preview_every"]
    g = PerfGovernor(enabled=False)
    feed(g, 500.0, 10, 0.0)
    assert g.level == 0


# --------------------------------------------------------- camera grabber ---
class FakeCap:
    def __init__(self, fps=400, n=10 ** 9, stall_after=None):
        self.period, self.n, self.i, self.stall_after, self.released = 1 / fps, n, 0, stall_after, False
    def read(self):
        if self.stall_after is not None and self.i >= self.stall_after:
            time.sleep(10); return False, None
        if self.i >= self.n:
            return False, None
        time.sleep(self.period)
        self.i += 1
        return True, np.full((4, 4, 3), self.i % 255, np.uint8) + 0 * self.i, 
    def isOpened(self): return not self.released
    def release(self): self.released = True
    def get(self, p): return 0


class FakeCap2(FakeCap):
    def read(self):
        r = super().read()
        if isinstance(r, tuple) and len(r) == 2 and r[0] is True:
            ok, frame = r[0], r[1]
            frame = frame.copy(); frame[0, 0, 0] = self.i & 0xFF; frame[0, 0, 1] = (self.i >> 8) & 0xFF
            return ok, frame
        return r


def idx(frame): return int(frame[0, 0, 0]) | (int(frame[0, 0, 1]) << 8)


def test_grabber_returns_newest_frames_never_stale_or_duplicate_under_a_slow_consumer():
    cap = FakeCap2(fps=400)
    g = FrameGrabber(cap)
    seen, lag = [], []
    for _ in range(12):
        ok, f = g.read()
        assert ok
        seen.append(idx(f))
        lag.append(cap.i - idx(f))                            # how many frames newer the camera already is
        time.sleep(0.02)                                      # consumer much slower than producer (50 Hz vs 400 Hz)
    g.release()
    assert all(b > a for a, b in zip(seen, seen[1:])), seen   # strictly increasing: no duplicates, no going back
    assert max(lag) <= 3, lag                                 # always the NEWEST frame (a FIFO queue would be hundreds behind)
    assert g.dropped > 20, g.dropped                          # the skipped frames were dropped, not queued


def test_grabber_waits_for_a_new_frame_when_consumer_is_faster_than_the_camera():
    cap = FakeCap2(fps=50)
    g = FrameGrabber(cap)
    t0 = time.perf_counter()
    got = [idx(g.read()[1]) for _ in range(10)]
    dt = time.perf_counter() - t0
    g.release()
    assert got == sorted(set(got)) and dt > 0.12, (got, dt)


def test_grabber_eof_stall_and_release():
    cap = FakeCap2(fps=500, n=5)
    g = FrameGrabber(cap)
    frames = []
    while True:
        ok, f = g.read()
        if not ok: break
        frames.append(idx(f))
    assert frames and frames[-1] == 5 and frames == sorted(set(frames)), frames   # delivered up to the last frame, then clean EOF
    g.release()
    assert cap.released
    stalled = FrameGrabber(FakeCap2(fps=500, stall_after=2), stall_timeout=0.3)
    assert all(stalled.read()[0] for _ in range(2))
    t0 = time.perf_counter()
    assert stalled.read() == (False, None) and time.perf_counter() - t0 < 1.0   # stall -> clean False, not a hang
    cap2 = FakeCap2(fps=200); g2 = FrameGrabber(cap2); g2.read(); g2.release()
    assert not g2._thread.is_alive()


def test_the_staleness_check_would_catch_a_fifo_buffer():
    """Sanity-check the test itself: a naive queue-everything reader lags badly."""
    cap = FakeCap2(fps=400)
    q, stop = [], threading.Event()
    def pump():
        while not stop.is_set():
            ok, f = cap.read()
            q.append(f)
    th = threading.Thread(target=pump, daemon=True); th.start()
    lag = []
    for k in range(12):
        while len(q) <= k: time.sleep(0.001)
        lag.append(cap.i - idx(q[k])); time.sleep(0.02)       # consumer takes frames IN ORDER
    stop.set(); th.join(1)
    assert max(lag) > 20, lag


# ----------------------------------------------------------- fast drawing ---
def test_fast_hand_drawing_paints_and_survives_off_screen_landmarks():
    frame = np.zeros((360, 640, 3), np.uint8)
    lm = [types.SimpleNamespace(x=0.3 + 0.02 * i, y=0.3 + 0.01 * i, z=0) for i in range(21)]
    draw_hand_fast(frame, types.SimpleNamespace(landmark=lm))
    assert frame.sum() > 0
    far = [types.SimpleNamespace(x=-0.4 + 2.0 * (i % 2), y=1.7, z=0) for i in range(21)]
    draw_hand_fast(frame, types.SimpleNamespace(landmark=far))   # must not raise


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS ", n)
            except Exception as e:
                fails += 1; import traceback; print("FAIL ", n, repr(e)); traceback.print_exc(limit=3)
    sys.exit(1 if fails else 0)
