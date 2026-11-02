"""Tests for features/game.py's normalized-geometry refactor (no camera/
display needed, real cv2 + real asset files - only pygame is faked,
since it isn't installable in this environment and WhackAMoleGame only
ever touches it for one write-only call, pygame.mixer.Sound(...).play()).

Run: python tests/test_game.py

What this is actually checking: a real user reported that Whack-a-Mole's
score/hits only registered when their hand coincided with the mole
INSIDE the small cv2 preview window, never on the floating overlay they
were actually looking at. The cause was a mismatch between two different
coordinate mappings (see features.game.WhackAMoleGame's class
docstring) - these tests pin down that frame-space play is byte-for-byte
unchanged (so main.py and the cv2 window keep working exactly as
before), AND that screen-space play now uses genuinely consistent
geometry, not a second, subtly-different copy of it.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class _FakeSound:
    def __init__(self, *a, **k): pass
    def play(self): pass


_fake_pygame = types.ModuleType("pygame")
_fake_pygame.mixer = types.SimpleNamespace(Sound=_FakeSound, init=lambda: None)
_fake_pygame.init = lambda: None
sys.modules.setdefault("pygame", _fake_pygame)

import config
from features.game import WhackAMoleGame


def make_game():
    return WhackAMoleGame(config.FRAME_WIDTH, config.FRAME_HEIGHT)


def test_frame_space_geometry_is_unchanged_from_the_original_hardcoded_values():
    """button_pos/button_size/close_button_pos/close_button_size/_xmin.../
    mole_pos (frame-space, no width/height given) must reproduce the
    EXACT original hardcoded pixel positions - main.py and the cv2
    window's draw() depend on this being byte-for-byte the same as
    before the normalized-geometry refactor."""
    g = make_game()
    assert g.button_pos == (1050, 200) and g.button_size == (210, 90)
    assert g.close_button_pos == (1050, 300) and g.close_button_size == (210, 90)
    assert (g._xmin, g._ymin, g._xmax, g._ymax) == config.MOLE_SPAWN_BOUNDS
    x0, y0, x1, y1 = config.MOLE_SPAWN_BOUNDS
    mx, my = g.mole_pos
    assert x0 <= mx <= x1 and y0 <= my <= y1


def test_frame_space_handle_tap_still_starts_and_scores_exactly_as_before():
    g = make_game()
    assert not g.active
    bx, by = g.button_pos
    bw, bh = g.button_size
    g.handle_tap(bx + bw // 2, by + bh // 2)          # point at Start, frame-pixel space
    assert g.countdown_active
    g.countdown_start_time -= config.COUNTDOWN_DURATION_S + 1   # fast-forward
    g.update()
    assert g.active and not g.countdown_active

    mx, my = g.mole_pos
    before = g.score
    g.handle_tap(mx, my)                              # point at the mole, frame-pixel space
    assert g.score == before + 1


def test_screen_space_geometry_scales_to_an_arbitrary_screen_size():
    g = make_game()
    sw, sh = 2560, 1440                                # deliberately NOT the frame's 1280x720
    bx0, by0, bx1, by1 = g.bounds_px(sw, sh)
    assert 0 <= bx0 < bx1 <= sw and 0 <= by0 < by1 <= sh
    sx, sy = g.mole_pos_px(sw, sh)
    assert bx0 <= sx <= bx1 and by0 <= sy <= by1       # mole stays inside ITS OWN screen-space bounds
    assert g.mole_radius_px(sw, sh) > g.mole_radius_px(640, 360), \
        "hit radius should scale up on a bigger canvas, not stay a flat pixel count"


def test_screen_space_handle_tap_hits_the_screen_space_mole_not_the_frame_one():
    """The actual bug: hitting where the mole is drawn ON SCREEN must
    register, using ONLY the screen-space coordinates and size - no
    frame-pixel numbers involved at all, proving the two spaces are
    independently consistent rather than one being derived by stretching
    the other (which was the original bug)."""
    g = make_game()
    sw, sh = 1920, 1080
    start_pos, start_size = g.button_px(sw, sh)
    g.handle_tap(start_pos[0] + start_size[0] // 2, start_pos[1] + start_size[1] // 2, sw, sh)
    assert g.countdown_active
    g.countdown_start_time -= config.COUNTDOWN_DURATION_S + 1
    g.update()
    assert g.active

    mx, my = g.mole_pos_px(sw, sh)
    before = g.score
    g.handle_tap(mx, my, sw, sh)
    assert g.score == before + 1, "pointing at the screen-space mole position must register a hit"

    # And the SAME point must NOT register in frame-pixel space (unless
    # coincidentally inside frame bounds too) - this is what proves the
    # two spaces are genuinely independent, not silently sharing state
    # in a way that would mask the original bug.
    fx, fy = g.mole_pos        # the (different) frame-space mole position
    assert (mx, my) != (fx, fy) or (sw, sh) == (g.frame_width, g.frame_height)


def test_frame_and_screen_space_share_the_same_underlying_mole_not_two_separate_games():
    g = make_game()
    g.active = True
    frame_mole_before = g.mole_pos
    screen_mole_before = g.mole_pos_px(1920, 1080)
    g._register_hit()                                  # a hit from EITHER space moves the SAME mole
    assert g.mole_pos != frame_mole_before or g.mole_pos_px(1920, 1080) != screen_mole_before
    # both views immediately reflect the new position - there is exactly
    # one `_mole_frac`, not a frame copy and a screen copy.
    assert g.mole_pos == g.mole_pos_px(g.frame_width, g.frame_height)


def test_whack_start_and_close_need_a_pinch_click_not_just_proximity():
    """Requested behaviour: bringing the pointer onto the mole must not
    score - only a pinch (clicked=True) does. Same for Start and Close
    (brushing past Close would otherwise end the game)."""
    g = make_game()
    sw, sh = 1920, 1080
    sp, ss = g.button_px(sw, sh)
    start = (sp[0] + ss[0] // 2, sp[1] + ss[1] // 2)
    for _ in range(30):
        g.handle_tap(*start, sw, sh, clicked=False)        # hovering Start
    assert not g.countdown_active and not g.active, "hover must not start the game"
    g.handle_tap(*start, sw, sh, clicked=True)
    assert g.countdown_active
    g.countdown_start_time -= config.COUNTDOWN_DURATION_S + 1
    g.update()
    assert g.active

    mx, my = g.mole_pos_px(sw, sh)
    for _ in range(30):
        g.handle_tap(mx, my, sw, sh, clicked=False)        # pointer right on the mole
    assert g.score == 0, "pointing at the mole without pinching must not score"
    g.handle_tap(mx, my, sw, sh, clicked=True)
    assert g.score == 1

    cp, cs = g.close_button_px(sw, sh)
    g.handle_tap(cp[0] + 5, cp[1] + 5, sw, sh, clicked=False)
    assert g.active, "hovering Close must not end the game"
    g.handle_tap(cp[0] + 5, cp[1] + 5, sw, sh, clicked=True)
    assert not g.active


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                f()
                print("PASS ", n)
            except Exception as e:
                fails += 1
                import traceback
                print("FAIL ", n, repr(e))
                traceback.print_exc(limit=4)
    sys.exit(1 if fails else 0)
