"""Tests for tutorial.py (Tk-free). Run: python tests/test_tutorial.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import tutorial
from tutorial import Tutorial, STEPS, DONE_PAUSE_S


def center(rect):
    return ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


def btn(t, bid):
    return center(next(b["rect"] for b in t.buttons() if b["id"] == bid))


def test_never_starts_by_itself_and_costs_nothing_when_idle():
    t = Tutorial()
    assert not t.active and t.view() is None
    for i in range(100):
        assert t.update({"pointer": (500, 500), "click": i % 7 == 0, "help_open": True, "menu_open": True}, i / 30) is None
    assert not t.active and t.view() is None, "gestures must never start the tutorial on their own"


def test_move_step_needs_real_travel_and_ignores_tracking_jumps():
    t = Tutorial(); t.start()
    now = 0.0
    for i in range(30):                                   # jittering in place: no progress
        now += 1 / 30
        t.update({"pointer": (500 + (i % 2) * 3, 500)}, now)
    assert t.view()["progress"] < 0.1 and not t.view()["done"]
    t.update({"pointer": (1800, 900)}, now)               # a tracking glitch: huge single jump
    assert t.view()["progress"] < 0.2, "a one-frame jump must not count as moving"
    x = 500
    for i in range(60):                                   # genuinely sweeping across the screen
        now += 1 / 30
        x += 20
        t.update({"pointer": (x, 500)}, now)
    assert t.view()["done"], "moving the ring far enough completes the step"
    now += DONE_PAUSE_S + 0.05
    t.update({"pointer": (x, 500)}, now)
    assert t.idx == 1, "auto-advances after the short 'Nice!' pause"


def test_each_step_completes_only_on_its_own_gesture():
    t = Tutorial(); t.start(); t.idx = 1; t._reset_step()      # pinch step
    now = 1.0
    t.update({"pointer": (300, 700), "help_open": True, "menu_open": True}, now)
    assert not t.view()["done"], "help/menu state must not complete the pinch step"
    t.update({"pointer": (300, 700), "click": True}, now + 0.1)
    assert t.view()["done"]
    t.update({}, now + 0.1 + DONE_PAUSE_S + 0.01)
    assert t.idx == 2                                           # help step
    t.update({"pointer": (300, 700), "click": True}, now + 3)
    assert not t.view()["done"], "a pinch alone must not complete the help step"
    t.update({"pointer": (60, 1000), "help_open": True}, now + 3.1)
    assert t.view()["done"]
    t.update({}, now + 3.1 + DONE_PAUSE_S + 0.01)
    assert t.idx == 3                                           # menu step
    t.update({"menu_open": True}, now + 6)
    assert t.view()["done"]


def test_pressing_a_button_is_not_mistaken_for_the_lesson_and_buttons_navigate():
    t = Tutorial(); t.start(); t.idx = 1; t._reset_step()        # pinch lesson
    assert t.update({"pointer": btn(t, "back"), "click": True}, 1.0) == "back"
    assert t.idx == 0
    t.idx = 1; t._reset_step()
    assert t.update({"pointer": btn(t, "next"), "click": True}, 2.0) == "next"
    assert t.idx == 2, "Next moves on without needing the gesture"
    assert not t.view()["done"]
    assert t.update({"pointer": btn(t, "skip"), "click": True}, 3.0) == "skip"
    assert not t.active


def test_last_step_done_finishes_but_skip_does_not_count_as_finished():
    t = Tutorial(); t.start()
    for _ in STEPS:
        t.next()
    assert t.finished and not t.active

    t2 = Tutorial(); t2.start(); t2.stop()
    assert not t2.finished and not t2.active
    t3 = Tutorial(); t3.start(); t3.idx = len(STEPS) - 1
    assert [b["label"] for b in t3.buttons() if b["id"] == "next"] == ["Done"]


def test_buttons_fit_inside_the_card_on_small_screens_and_card_is_centred():
    for sw, sh in ((1920, 1080), (1366, 768), (1024, 600)):
        t = Tutorial(sw, sh); t.start(); t.idx = 3
        x0, y0, x1, y1 = t.card_rect()
        assert x0 >= 0 and x1 <= sw and abs((x0 + x1) / 2 - sw / 2) < 2
        for b in t.buttons():
            bx0, by0, bx1, by1 = b["rect"]
            assert x0 <= bx0 and bx1 <= x1 and y0 <= by0 and by1 <= y1, (sw, b)
        ids = [b["id"] for b in t.buttons()]
        assert len(ids) == len(set(ids))


def test_toggle_and_view_data():
    t = Tutorial(); t.toggle(); assert t.active and t.view()["total"] == len(STEPS)
    assert t.view()["title"] == STEPS[0]["title"] and t.view()["waiting"]
    t.toggle(); assert not t.active


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
