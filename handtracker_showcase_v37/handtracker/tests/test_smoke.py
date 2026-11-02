"""Runs the headless real-main() smoke harness (tests/smoke_main.py) in a
subprocess and asserts on what the loop actually did. Run: python tests/test_smoke.py"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def run(**env):
    e = dict(os.environ, **{k: str(v) for k, v in env.items()})
    p = subprocess.run([sys.executable, os.path.join(HERE, "smoke_main.py")], capture_output=True, text=True,
                       timeout=300, env=e)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-2500:]
    line = next(l for l in p.stdout.splitlines() if l.startswith("SMOKE_JSON "))
    return json.loads(line[len("SMOKE_JSON "):]), p.stdout


def test_real_main_loop_runs_ar_mode_with_scripted_hands_end_to_end():
    r, out = run(SMOKE_FRAMES=160)
    assert r["frames"] >= 160 and r["control_on"], r
    s = r["sched"]
    assert s.get("probe", 0) >= 5, s                      # probes for a 2nd hand while one hand is up
    assert s.get("to_dual", 0) >= 1 and s.get("to_single", 0) >= 1, s    # saw hand 2 arrive AND leave
    assert r["trackers"]["single"] > 60                   # most 1-hand frames used the cheap tracker
    # Camera preview thumbnail is disabled at the source right now (see
    # showcase.py's camera_preview = None setup) rather than merely
    # defaulting to hidden - so even with 'c' pressed (frame 4 in
    # smoke_main.py's script) no preview image should ever get made.
    # If this starts failing because camera_preview was wired back up,
    # that's a deliberate product decision to revisit, not a bug -
    # update this assertion (back to photo_made==1 and
    # photo_pastes>20) alongside re-enabling it.
    assert r["photo_made"] == 0 and r["photo_pastes"] == 0, r
    assert r["objects"] == 4 and r["budget"] == 450       # 3D scene live; governor left the budget alone
    assert r["lod"] == 1                                  # 4 objects ~505 items > 450 budget -> scene itself trimmed to LOD 1
    assert not r["perf_lines"], r["perf_lines"]           # healthy machine: governor stays at level 0


def test_adaptive_quality_engages_inside_the_live_loop_when_the_machine_is_slow():
    r, out = run(SMOKE_FRAMES=170, SMOKE_SLOW=1)
    assert len(r["perf_lines"]) >= 2, r["perf_lines"]     # stepped down at least twice
    assert r["budget"] < 450, r                           # 3D item budget actually lowered in the overlay
    assert r["lod"] >= 1, r                               # ...and the scene responded by dropping detail


def test_help_panel_survives_perf_suppression_and_user_can_force_it_back_on():
    # Sustained overload -> governor settles at level 3 (help auto-hidden)
    # well before frame 150, so pressing H there exercises exactly the
    # "chart placement bogs AR mode down, help disappears" complaint, and
    # the "can't be re-enabled" complaint: an un-forced show_help=True
    # (the pre-fix behaviour) would stay invisible under the still-high
    # perf level with no way back short of the machine recovering.
    r, out = run(SMOKE_SLOW=1, SMOKE_FRAMES=250, SMOKE_KEY_AT=150, SMOKE_KEY="h")
    trace = {t["frame"]: t for t in r["help_trace"]}
    assert trace[149]["perf_level"] == 3 and trace[149]["show_help"] is False, trace[149]      # suppressed...
    assert trace[149]["note"], trace[149]                                                       # ...and said so
    assert trace[151]["show_help"] is True and trace[151]["perf_level"] == 3, trace[151]        # H forces it back
    assert trace[249]["show_help"] is True, trace[249]                                          # ...and it sticks

    # Full off -> forced-on -> off cycle, still under the same sustained
    # overload (perf level stays 3 throughout): the pre-fix bug was that
    # a second H press here (the user trying to turn help back OFF)
    # could silently do nothing or leave it wedged, since the toggle was
    # flipping a flag the display gate was independently overriding.
    r4, _ = run(SMOKE_SLOW=1, SMOKE_FRAMES=300, SMOKE_KEY_AT=150, SMOKE_KEY="h",
                SMOKE_KEY_AT2=200, SMOKE_KEY2="h")
    t4 = {t["frame"]: t for t in r4["help_trace"]}
    assert t4[149]["show_help"] is False and t4[151]["show_help"] is True   # off -> forced on
    assert t4[201]["show_help"] is False and t4[201]["perf_level"] == 3     # on -> off again, cleanly
    assert t4[299]["show_help"] is False


def test_game_mode_board_renders_on_the_main_overlay_without_crashing():
    # Whack-a-Mole's state is plain numbers (not a raster like Drawing
    # mode above), so this is really just confirming the frame-pixel ->
    # screen-pixel scaling in showcase.py and MainOverlay._draw_
    # whackamole's shape drawing don't blow up on real (scripted) game
    # state across the game's own countdown -> active transition.
    r, out = run(SMOKE_FRAMES=90, SMOKE_KEY_AT=10, SMOKE_KEY="g")
    assert r["frames"] >= 90
    assert "Traceback" not in out, out


def test_main_overlay_rendering_failure_cannot_crash_the_whole_session():
    """A real user hit a bare, unhandled TclError from inside
    main_overlay_win.update() (the ImageTk "wrong Tk interpreter" bug -
    see main_overlay._draw_drawing_canvas's own note, and the master=
    fix elsewhere) that took the entire tracking session down.
    showcase.py now wraps that call in a try/except that logs and
    continues instead - SMOKE_BREAK_OVERLAY (smoke_main.py) deliberately
    breaks MainOverlay.update() itself to confirm that guarantee holds
    on its own, independent of any one specific bug inside it ever being
    fixed or regressing again."""
    r, out = run(SMOKE_FRAMES=40, SMOKE_BREAK_OVERLAY=1)
    assert r["frames"] >= 40, "main loop must keep running frames despite the failure"
    assert "Traceback" not in out, out
    assert "synthetic overlay rendering failure" in out, \
        "the failure should be logged, not silently swallowed"


def test_drawing_mode_floats_strokes_on_the_main_overlay():
    """Drawing mode now runs entirely on the overlay (overlay_drawing.py:
    toolbar + tiled ink in screen space). The scripted hand never pinches,
    so no ink exists and no tile images are made - what this checks is the
    real wiring: the mode runs for many frames through the real main loop
    without any overlay draw failure, and without falling back to the old
    in-window panel."""
    r, out = run(SMOKE_FRAMES=40, SMOKE_KEY_AT=10, SMOKE_KEY="w")
    assert r["frames"] >= 40
    assert "Traceback" not in out, out
    assert "Main overlay draw failed" not in out, out
    assert "Overlay drawing unavailable" not in out, out
    assert r["photo_made"] == 0, "no ink drawn -> no tile images, and no full-screen image either"


def test_tutorial_never_starts_by_itself_but_starts_on_demand():
    """Requested: the tutorial must be available when needed and must NOT
    appear on launch. A plain run (even with many frames and hand activity)
    never draws it; pressing T does."""
    r, out = run(SMOKE_FRAMES=60)
    assert "tutorial" not in r["extras_seen"], "tutorial must never auto-start"
    r, out = run(SMOKE_FRAMES=60, SMOKE_KEY_AT=10, SMOKE_KEY="t")
    assert "tutorial" in r["extras_seen"], r["extras_seen"]
    assert "Traceback" not in out and "Main overlay draw failed" not in out, out


def test_video_call_mode_runs_in_the_real_loop_with_its_call_bar():
    r, out = run(SMOKE_FRAMES=60, SMOKE_KEY_AT=10, SMOKE_KEY="7")
    assert "call" in r["extras_seen"], "call bar must be drawn in Video Call mode"
    assert "Traceback" not in out and "Main overlay draw failed" not in out, out
    r2, _ = run(SMOKE_FRAMES=60, SMOKE_KEY_AT=10, SMOKE_KEY="c")       # C is the shortcut
    assert "call" in r2["extras_seen"]


def test_virtual_camera_receives_the_clean_unmirrored_camera_image():
    """The call must see what the camera sees - not the mirrored debug view
    with skeletons/HUD - and the device must be released on exit."""
    r, out = run(SMOKE_FRAMES=60, SMOKE_VCAM=1)
    assert "Virtual camera ON" in out, out
    assert r["vcam_frames"] >= 10, r["vcam_frames"]
    assert r["vcam_clean"] is True, "published frame must equal the raw (un-mirrored) camera frame"
    assert r["vcam_closed"] is True, "virtual camera must be closed when the app exits"
    assert "Traceback" not in out, out


def test_virtual_camera_is_off_by_default_and_missing_driver_does_not_break_the_app():
    r, out = run(SMOKE_FRAMES=40)
    assert r["vcam_frames"] == 0 and "Virtual camera" not in out
    r, out = run(SMOKE_FRAMES=40, SMOKE_ARGS="--vcam")              # no backend installed here
    assert r["frames"] >= 40 and "Traceback" not in out, out
    assert "pyvirtualcam" in out, "the user must be told how to get a virtual camera"


def test_hud_is_hidden_from_screen_share_except_in_drawing_mode():
    r, out = run(SMOKE_FRAMES=40)
    assert r["capture_flags"] and r["capture_flags"][0] is True, r["capture_flags"]
    r, out = run(SMOKE_FRAMES=40, SMOKE_KEY_AT=10, SMOKE_KEY="w")      # Drawing = annotating for an audience
    assert r["capture_flags"][-1] is False and True in r["capture_flags"], r["capture_flags"]


def test_call_profile_flag_shrinks_the_pipeline():
    r, out = run(SMOKE_FRAMES=40, SMOKE_ARGS="--call")
    assert "Call profile" in out and "Traceback" not in out, out
    assert r["frames"] >= 40


CALL = dict(SMOKE_FRAMES=80, SMOKE_KEY_AT=8, SMOKE_KEY="7", SMOKE_HOLD_S=0.3)


def test_call_mode_really_sends_the_shortcut_when_a_call_app_is_in_front():
    """The user's report: "mic mute, raise hand - nothing works". End to end
    through the real loop: Video Call mode, a still held fist, Zoom in front
    -> Alt+A is sent. (Hand control starts OFF in this app; entering Call
    mode now switches it on instead of silently doing nothing.)"""
    r, out = run(SMOKE_FORCE_GESTURE="fist", SMOKE_FOREGROUND="Zoom Meeting", **CALL)
    assert ["alt", "a"] in r["hotkeys"], (r["hotkeys"], r["toasts"])
    assert any("Hand control ON" in t for t in r["toasts"]), r["toasts"]
    assert any("Mute" in t for t in r["toasts"]), r["toasts"]
    assert "Traceback" not in out, out


def test_other_gestures_map_to_camera_and_raise_hand():
    r, _ = run(SMOKE_FORCE_GESTURE="two_finger", SMOKE_FOREGROUND="Chat | Microsoft Teams", **CALL)
    assert ["ctrl", "shift", "o"] in r["hotkeys"], r["hotkeys"]
    r, _ = run(SMOKE_FORCE_GESTURE="open_palm", SMOKE_FOREGROUND="Meet - abc-defg-hij - Google Chrome", **CALL)
    assert ["ctrl", "alt", "h"] in r["hotkeys"], r["hotkeys"]


def test_nothing_is_sent_and_you_are_told_why_when_no_call_window_is_in_front():
    r, out = run(SMOKE_FORCE_GESTURE="fist", SMOKE_FOREGROUND="Untitled - Notepad", **CALL)
    assert r["hotkeys"] == [], "must never fire shortcuts into an unrelated window"
    assert any("call window" in t for t in r["toasts"]), r["toasts"]


def test_camera_can_be_released_for_other_apps_and_taken_back_with_the_hotkey():
    """Requested switch: release the webcam so other apps can use it, and
    take it back without the mouse. (Hands can't do the taking-back - with
    no camera there is nothing to see them - so it's the global hotkey.)"""
    r, out = run(SMOKE_FRAMES=90, SMOKE_KEY_AT=10, SMOKE_KEY="k", SMOKE_HOTKEY_AT=30)
    assert "banner" in r["extras_seen"], "must tell the user the app is deliberately blind"
    assert r["cap_opens"] == 2, "camera released, then re-opened on the hotkey"
    assert any("Camera back" in t for t in r["toasts"]), r["toasts"]
    assert "Traceback" not in out and "Main overlay draw failed" not in out, out
    assert r["frames"] >= 90


def test_released_camera_stays_released_without_a_resume_and_exits_cleanly():
    r, out = run(SMOKE_FRAMES=60, SMOKE_KEY_AT=10, SMOKE_KEY="k")
    assert r["cap_opens"] == 1 and "banner" in r["extras_seen"]
    assert "Traceback" not in out, out


def test_resume_while_another_app_still_holds_the_camera_reports_it_instead_of_crashing():
    r, out = run(SMOKE_FRAMES=90, SMOKE_KEY_AT=10, SMOKE_KEY="k", SMOKE_HOTKEY_AT=30, SMOKE_CAM_BUSY=1)
    assert any("still in use" in t for t in r["toasts"]), r["toasts"]
    assert "Traceback" not in out, out


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS ", n)
            except Exception as e:
                fails += 1; print("FAIL ", n, str(e)[:1500])
    sys.exit(1 if fails else 0)
