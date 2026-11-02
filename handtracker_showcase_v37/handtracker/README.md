# Hand Tracker Project

## v26 changes (this round) - fixed the help panel getting stuck off

**The bug, as reported:** the help panel would disappear on its own while
placing charts in AR mode, and once it was gone, pressing H to bring it
back didn't work.

**Root cause - one bug wearing two symptoms.** The adaptive-quality
governor (`perf.py`) drops the help panel at its lowest quality level
(`LEVELS[3]["help"] = False`) to save a bit of per-frame compute under
heavy load - AR mode's chart placement is exactly the kind of load that
triggers it, which is why it looked tied to that specific action. The
display line computed `show_help and perf.settings["help"]` each frame:
whichever one was False, the panel stayed hidden - so far so good. The
break was in the `H` key handler, which blindly flipped the *other* flag
(`show_help = not show_help`) with no idea whether the panel was actually
visible. Under sustained load the panel was already hidden by the perf
flag; pressing H to "turn it back on" instead flipped `show_help` to
`False` (it had never left `True`) - genuinely turning it off. By the
time performance recovered and the perf flag went back to `True`, the
user-intent flag was now the one saying no, so it stayed hidden - "can't
be re-enabled."

**The fix**, in `showcase.py`:
- A clear split between what the user asked for (`show_help`) and
  whether the perf governor is currently allowed to honor it
  (`perf.settings["help"]`), plus a `force_help_override` the user can
  set explicitly.
- `H` now toggles against what's actually on screen (`hud_show_help`),
  not the raw intent flag - so the key always does what it visibly shows,
  regardless of perf state. Pressing it while the panel is hidden for any
  reason (perf or a previous explicit hide) turns it on **and** forces it
  past the perf governor; pressing it again turns it off and drops the
  override, letting the governor manage it normally again.
- When the governor is actively hiding a panel the user still wants
  (`show_help` True but not forced), a small dim note - "Help hidden to
  save performance - press H to show it anyway" - says so in the panel's
  place, instead of it just vanishing with no explanation.

**What's unchanged:** the governor itself still drops the help panel
under sustained load by default - that's a real, if small, perf saving,
and nothing about the bug was in `perf.py`. Mode switches still show
help normally when performance allows it; they don't force the override,
so the governor's suppression still works as designed for anyone who
never touches H.

**Tested:** `tests/test_smoke.py` now includes
`test_help_panel_survives_perf_suppression_and_user_can_force_it_back_on`,
which runs the real `main()` loop (see `tests/smoke_main.py`) under a
scripted, sustained overload until the governor settles at level 3 (help
suppressed), scripts an H press and confirms the panel comes back
immediately while the governor is still at level 3 (i.e. it's the
override working, not the machine recovering), then scripts a second H
press and confirms it turns cleanly back off. All 70 pre-existing tests
across `tests/test_perf.py`, `tests/test_ar3d.py`,
`tests/test_render_equivalence.py`, and `tests/test_ar_overlay_integration.py`
still pass unchanged.

## v25 changes - performance

Profiled first, then fixed what the numbers pointed at. Everything below is covered by tests (`python tests/run_all.py`, 75 tests, no camera/display needed), including a headless run of the **real `main()` loop** with a fake camera and scripted hands.

**Why FPS dropped the moment a hand came in (AR mode).** The AR tracker was a 2-hand MediaPipe instance. MediaPipe only skips its palm detector when it is already tracking as many hands as `max_num_hands`, so with ONE hand in view it ran full palm detection on **every frame**, hunting for a second hand. Now `tracker_sched.py` uses the cheap 1-hand tracker normally and probes for a second hand every 6th frame (only while a hand is visible); once two hands are seen it tracks both every frame, and falls back after the second leaves for 12 frames. Cost: a second hand is noticed within ~0.2 s of entering view instead of instantly. (`AR_SECOND_HAND_PROBE_FRAMES`, `AR_DUAL_FALLBACK_FRAMES`; set probe frames to 1 for the old behaviour.)

**Other fixes** (measured on this machine; absolute numbers will differ on yours):
- **Camera preview thumbnail: 13.2 ms -> ~1.3 ms per frame.** It bicubic-resized the full 1280x720 frame to 260 px, rebuilt the Tk image and ran two Tk updates per frame. Now: OpenCV area-resize first, reuse the Tk image buffer, one update.
- **3D rendering: 2.6-3.2x less Python time**, and in the worst case (6 surfaces) **12.6 -> 1.2 ms and 1,320 -> 144 canvas items**. Vectorised depth sorting/culling/coordinate flattening, cached colour shading, and geometry that is only recomputed when something actually changed (objects you aren't touching are skipped). Verified call-for-call identical to v24 at full detail.
- **Level of detail for many shapes.** Tk's cost is per canvas item, so the 3D layer has an item budget (450). With a few shapes nothing changes; pile on many surfaces and it thins grid lines (then draws one item per grid line) instead of dropping your frame rate. Selection patches stay visible at every level.
- **HUD: ~1.5x faster, pixel-identical** (it copied and blended the whole 720p frame three times to tint three small panels).
- **Hand skeleton drawn with direct OpenCV calls** instead of mediapipe's `draw_landmarks`.
- **Threaded camera capture + MJPG** (`camera_stream.py`): decode overlaps tracking, the loop always gets the NEWEST frame (no growing input lag when it can't keep up), and the driver is asked for MJPG/30 fps/1-frame buffer - many webcams default to uncompressed YUY2, which is only ~10 fps at 720p. If your camera misbehaves: `CAMERA_THREADED = False` and/or `CAMERA_FOURCC = ""`; on Windows `CAMERA_BACKEND = "dshow"` is worth trying.
- **Adaptive quality** (`PerfGovernor`): measures each frame's compute time (camera wait excluded). If it exceeds ~85% of a frame for a second it steps down what is *drawn* - thumbnail rate, 3D detail, then the help panel - and steps back up when there is headroom. Never touches tracking or gesture logic. A "PERF L<n>" tag shows on the HUD while degraded. (`ADAPTIVE_QUALITY`, `TARGET_FPS`.)

**A/B of the real main loop** (identical scripted AR session, v24 vs v25, fake trackers and no real Tk, so it measures only what changed): **26.3 -> 9.8 ms CPU per frame**; the 2-hand tracker was used for 207/220 frames in v24 vs 91/220 in v25 (and in a one-hand-only session, ~1 in 6 instead of every frame).

**Not measured (be aware):** MediaPipe inference time and real Tk drawing cost - neither can run in my environment. The tracker saving is architectural (verified by call counts with fake trackers, not stopwatch time), and the item-count reduction is verified but its Tk time saving is inferred. If FPS is still short of target, the `PERF L<n>` tag and the `[PERF]` console lines will show whether the governor is stepping in; telling me your numbers would let me target the next bottleneck.

## v24 changes - "the diagram got deleted" fix + reshaping that makes sense

**Why the surface (or any 3D object) was deleting itself:** a hard two-hand pinch curls your other fingers, and the gesture classifier can read that as a *fist* - which removes whatever is under the cursor. (The same path also has "two fingers = refresh", which reset your stretch and cycled the surface function mid-gesture - the shape "changing on its own".) Now a fist / two-finger over a 3D object only acts when **no pinch is active or was active in the last 0.8 s** (`AR3D_DESTRUCTIVE_COOLDOWN`), and even when blocked it's swallowed so nothing else reacts to it. **Press `U` to undo** the last removal (keeps the last 5).

**Reshaping, redesigned around one idea: the selection is the handle.**
- Select a portion as before (pinch a bar / surface point + drag). It's now drawn unmistakably: on a surface, an amber footprint on the floor with white posts up to the surface.
- **Pinch anywhere inside the selection and drag** = stretch it with ONE hand: **left/right = wider, up/down = taller** (200 px = 2x wide, 150 px = 2x tall; the opposite direction compresses). A hint by your fingertip tells you what a pinch there will do.
- **Two hands** on a selection: **hands apart = wider, both hands up/down together = taller/shorter.** (Height no longer needs the old diagonal-corner trick, which nobody would find.)
- Pinch *outside* the selection = start a new one. Tap the floor = clear it.
- **Fixed: the first hand of a two-hand gesture no longer wipes your selection.** Before, landing a hand on the chart started a fresh one-cell selection before the second hand arrived, so you ended up stretching a single bar/cell - or, with no selection, stretching one bar instead of resizing the chart. If the second hand joins before the first has dragged, your previous selection (or none = whole-object resize) is restored.
- **Surface selection now follows the floor under your finger** (exact inverse of the perspective projection) instead of snapping to whichever mesh point is nearest on screen, which jumped around on tall peaks. Touching a peak itself still selects there. When the floor is seen edge-on (yawed ~90 deg) it falls back to nearest point.

Tests: 50 in `tests/test_ar3d.py` (was 40), including the two-hand-pinch-then-fist deletion case, selection preservation, one/two-hand stretch, and floor-unprojection round trips at random poses.

*(The gesture table in v22 below is superseded for stretch/select by this section.)*

## v23 changes - pinch accuracy, floor-never-lost, smoother 3D

**Cursor no longer sags when you pinch (AR mode).** The fingertip physically drops toward the thumb as a pinch closes, so the old fingertip cursor landed ~25-30px below your target by the time the pinch registered. Now the cursor remembers where your fingertip sat relative to your palm while your fingers were most open (last ~0.4 s), and once you start closing toward a pinch it locks to *that* and follows only your hand's movement. It eases there in ~80 ms (no snap) and eases back to the live fingertip on release. The ring turns amber and tightens as you close in ("locked on"), then red when pinched. The drawn cursor, the 3D grab point and the 2D chart point are all the same stabilized point. On synthetic pinch motion the pinch lands within ~0.1 px of the aim point (naive fingertip: 26-29 px). Tunables: `AR_PINCH_LOCK_ON` / `AR_PINCH_LOCK_OFF` / `AR_PINCH_RELEASE_BLEND` in `config.py` - if the cursor locks too eagerly or too late for your hand, adjust `LOCK_ON` (higher = locks earlier).

**3D objects can't be lost off-screen any more.** Before, only the object's *centre* was kept on screen, so a tilted surface graph could sit with its grab-able floor hanging off the bottom. Now the whole projected object, floor plate included, is kept inside the screen (extra margin at the bottom for taskbars). Anything too big is pushed back in depth, then shrunk; two-hand scaling stops growing at the screen edge; and objects saved off-screen by the old version are rescued on load.

**Smoothness polish**
- Idle sway/spin now ease to a stop while a hand hovers over, selects on or holds an object - bars no longer drift under your fingertip mid-pinch - and resume afterwards.
- Two-hand scale, roll, yaw and push/pull depth are eased (time constant `AR3D_SMOOTH_TAU`, 0.06 s), so hand-distance noise doesn't flicker the size (4% input noise -> under 1% visible).
- Glide-after-release is now frame-rate independent (it used a per-frame decay).
- Larger two-hand readout.

Tests: 40 in `tests/test_ar3d.py` (was 28) - now including cursor sag on a real synthetic pinch motion, "pinch selects the bar you aimed at", floor-always-reachable, pull-back-from-bottom, calm-on-hover, frame-rate-independent glide, and jitter attenuation.

Still **not run on a real webcam**: the sag fix and thresholds are verified against synthetic hand motion; real hands vary, so the LOCK thresholds may want a tweak.

## v22 changes (this round) - 3D holograms + two-hand manipulation in AR mode

**A new "3D" button on the AR palette.** Pinch it to open a row of tiles - BARS, SURFACE, CUBE, SPHERE, TORUS - then pinch a tile, drag it out and drop it (same pick-up/drop as every other chart). They are wireframe/stippled "hologram" objects with real perspective, depth-dimmed lines, back-face culling and a gentle idle sway (or spin, for the solids) so they read as 3D even when nothing is touching them.

**How you use them**

| Gesture | Does |
|---|---|
| Pinch the floor plate / body + drag | **Move** it. Hand toward the camera = object comes toward you (bigger); away = pushes back |
| Pinch a **bar** (or a point on the surface) + drag | **Select a portion** - bars turn amber / surface patch turns white, with a bracket under the range |
| Tap the floor (quick pinch, no drag) | Clear the selection |
| **Both hands** pinch, nothing selected | **Scale** (hand distance), **roll** (angle between hands), **yaw** (push one hand toward the camera), move (midpoint) |
| **Both hands** pinch, portion selected | **Stretch / compress the portion.** Think of your hands as opposite corners of a box: pull them apart horizontally = spread (wider bars / magnified patch); apart vertically = amplitude (taller bars / higher peaks). Together = compress |
| Fist on an object | Remove it |
| Two fingers on an object | Refresh (bars: new data; surface: next function; either way resets stretch + selection) |

Stretch is a *focus + context* warp: the chart's overall footprint stays constant, so stretching the selected part squeezes the rest and compressing it gives the rest more room. It changes how the chart is drawn, **never the underlying data**. A green hint next to your fingertip tells you whether a pinch there will GRAB or SELECT. While two hands are pinching, a dashed tether shows the pair and a readout shows the live scale/roll/yaw or stretch factors.

**Under the hood**
- `ar3d.py` - scene + objects + software renderer (numpy only). Deliberately not OpenGL: AR mode's transparent click-through Tk overlay can't host a GL surface portably, and wireframe is the Iron-Man look anyway. Meshes are cached and rebuilt only when data/selection/stretch change.
- `ar3d_input.py` - `HandSlots` (keeps hand A/B identity stable frame to frame), `HandPointer` (per-hand smoothing + pinch, reusing the existing rules), `AR3DController` (the state machine).
- Two hands are only tracked **while in AR mode**: a second 2-hand MediaPipe instance is used just then, so every other mode keeps the single-hand pipeline unchanged. Expect somewhat lower FPS in AR mode with two hands in view.
- The 3D layer gets first refusal on each pinch; if it claims one, the 2D chart logic ignores it, so 2D charts and CSV charts behave exactly as before. 3D objects draw *behind* 2D charts.
- 3D objects (position, depth, rotation, size, data, selection, stretch) are saved in `ar_workspace.json` under `objects3d` and restored on launch. Old workspace files load fine.
- Tunables in `config.py` (bottom): `AR3D_DEPTH_FROM_HAND_SIZE`, `AR3D_DEPTH_GAIN`, `AR3D_DEPTH_DEADZONE`, `AR3D_YAW_GAIN`, `AR3D_TAP_*`.

**Tests (no display/camera needed):** `python tests/test_ar3d.py` (28 tests: projection, culling, picking, stretch invariants, persistence incl. corrupt saves, hand-slot identity, and the whole controller state machine driven by synthetic hands) and `python tests/test_ar_overlay_integration.py` (overlay + palette + save/restore against a stub Tk). Offscreen renders of every object type, the selection/stretch states and the two-hand HUD were also inspected as images.

**Known limits / honest caveats**
- **Not yet run on a real display + webcam.** Everything above is verified against synthetic hands and stub/offscreen canvases; Tk's real stipple/line rendering and MediaPipe's real two-hand behaviour are the unknowns. Treat first launch as a tuning session.
- **Yaw direction**: derived from the geometry, not checked live. If the object twists the wrong way, negate `AR3D_YAW_GAIN`.
- **Depth from hand size is noisy** (a rotating hand changes apparent size). There is a 5% deadzone; switch it off with `AR3D_DEPTH_FROM_HAND_SIZE = False` if it drifts.
- Two-hand selection-stretch needs the second hand's pinch to land over/near the object (40px margin).
- No pitch control yet (two hands can't define it); objects have a fixed downward tilt. No undo for a stretch (two-finger resets it). 3D bar charts use mock data - CSV-backed 3D bars are the obvious next step.
- Rendering cost: 6 large surfaces is the worst case (~12 ms/frame Python-side, ~1,300 canvas items); typical scenes are far lighter. `MAX_OBJECTS` (ar3d.py) caps the total at 6.
- Hands are identified by proximity to where they were last frame; if your hands cross closely, the two can swap identity for a frame.

## v15 changes (this round) - CSV import (stability first) + saved workspace

**CSV import, via a new FILES button on the AR palette.** Pinch FILES to
open a second row of tiles (one per `.csv` in `ar_data/`), pinch a tile,
drag it out and drop it - same pick-up/drag/drop as every other chart, no
keyboard. The folder is re-scanned every time the row opens, so a file you
drop in mid-demo just shows up. File charts show the **real category
names** (rotated under each bar) and the **real number** above each bar,
not just relative heights.

**Edit the CSV, then hold Two Fingers over its chart -> it reloads from
disk.** ("Edit the spreadsheet, watch the AR chart change.")

**Your layout is saved automatically and restored next launch** - chart
positions, types, live metrics, and file sources (`ar_workspace.json`).
Charts already survived switching modes within a session; this is what
makes them survive closing the app too. Saved throttled while you drag,
flushed whenever you leave AR mode, and once more on quit.

**CSV stability - tested against 28 messy-file cases, not just the tidy
sample.** The first version of the loader passed the clean sample and then
failed badly on realistic files, including one *silently wrong* result: a
semicolon-delimited file with decimal commas (`A;1,5`) was read as label
`"A;1"`, value `5` - and semicolon + decimal comma is Excel's default
export in many countries. Fixed. It now handles: comma / semicolon / tab /
pipe delimiters (auto-detected), UTF-8 (BOM or not) and Windows cp1252
encodings, optional header, blank lines, extra columns, quoted labels
containing commas, `1,200` / `1.200,50` / `1,5` / `$3.50` / `45%` numbers,
`nan`/`inf` (rejected), NUL bytes and binary garbage (clean error), and
oversized files (>5 MB refused, first 20 rows shown).

**Failure behavior, by design:** a bad/missing/locked CSV prints a
`[WARN]` line and the chart keeps its last good data - it never crashes the
app. A saved chart whose file is *temporarily* unreadable at launch is
skipped for that session but **kept in the save file** so it returns once
the file is fine (a file that's truly deleted is dropped, no phantom).
Sample data seeds only on a true first run, so files you delete stay deleted.

Also: case-insensitive `.csv` matching (`DATA.CSV` works), Office `~$`
lock files and dotfiles ignored. See `ar_data/README.txt` for the format.

**Known limits / honest caveats**
- **Still untested on a real display or webcam** - I ran the logic
  (loading, browsing, saving/restoring, reloading, rendering calls) against
  a stand-in for Tkinter, which catches logic bugs but not visual ones.
- Rotated category labels use Tk's `angle=` text option (Tk 8.6+, which any
  recent Python on Windows ships). On an ancient Tk they'd raise an error.
- Negative values draw as an empty bar (their true number is still printed
  above it) - no below-the-axis bars yet.
- Only the first two columns are used (one series per chart). XML, Excel,
  multi-series, editing values by hand, and a pie-chart legend are the
  obvious next steps - not attempted half-done.

## v14 changes (previous round) - Live data + physical polish (Phase 1 + 4 of the "make AR actually useful" plan)

Two of the four brainstormed directions for giving AR mode real "oomph":
real, currently-true data instead of random mock numbers, and motion
that feels alive instead of snapping between states. (The other two -
importing real files, and a genuine "why AR" workflow/arrangement - are
next, once a couple of open questions about file format and layout
persistence are settled.)

- **A 4th palette button: LIVE.** Pick it up and drop it like any other
  chart, but it's fed real numbers every frame instead of mock data:
  - **FPS** - your hand tracker's actual frame rate, as a live scrolling
    line.
  - **Gesture mix** - a pie of what fraction of your session so far has
    been spent in each gesture (fist/pointing/open palm/etc), built from
    real counts kept the whole time the app's been running - not just
    while in AR mode.
  - **Two fingers on a Live chart cycles between these** (there's
    nothing to "randomize" about real data, so two-finger means "show me
    something else" instead).
- **Charts now visibly morph into new values** instead of snapping -
  every randomize, live update, or metric switch eases the bars/line/pie
  toward its new numbers over a few frames (`ChartWidget.
  step_toward_target`).
- **Dropped charts now glide to a stop** instead of dead-stopping the
  instant you release the pinch - drag speed carries forward as decaying
  velocity for a few frames (real inertia, clamped so a chart can't be
  flung off-screen).
- Chart positions are now properly bounds-clamped in BOTH the normal
  full-screen mode and the non-Windows fallback window (previously only
  the fallback window was clamped - a fast drag in the normal mode could
  put a chart partially or fully off-screen with no clamp at all).

## v13 changes (previous round) - crash fix + visible AR cursor

- **Fixed a real crash**: `AROverlay.remove_chart` had been accidentally
  deleted during the v12 palette rewrite - a botched find-and-replace
  edit removed the `def remove_chart(self, chart):` line but left its
  body behind, which silently reattached itself to the end of the
  previous method (`cancel_ghost`) instead of raising an obvious error.
  `py_compile` only checks syntax, not whether a called method actually
  exists, so this got through - it would only surface once you actually
  triggered Fist-to-dismiss (`remove_chart`) or dropped a ghost back on
  the palette (`cancel_ghost`, which had the same problem). Both are
  fixed and this time actually exercised end-to-end (pick up, drag,
  drop, dismiss, cancel, pie/bar/line drawing) against a stub in place
  of the real Tkinter/display, not just syntax-checked - the previous
  bug specifically would NOT have been caught by a syntax check alone.
- **Added a visible cursor on the AR overlay itself.** The actual
  complaint behind "can't tell which chart I'm selecting" - since the
  overlay covers the real desktop instead of living inside the webcam
  window, there was nothing showing where a fingertip actually mapped
  to on screen. A small ring + crosshair now follows your fingertip
  directly on the overlay - green while idle, red while pinching (same
  color convention as Desktop mode's cursor reticle).

## v12 changes (previous round) - chart-type palette, replacing conjure-anywhere

The old "pinch on empty space conjures a random chart type" was clunky -
no way to pick which chart type you actually wanted. Replaced with a
proper toolbar:

- **A palette of chart-type buttons** (Bar / Line / Pie) sits top-left of
  the AR overlay, each drawn as a little icon of that chart type. Just
  pointing at one highlights it - no pinch needed to preview.
- **Pinch a palette button, drag it out, release -> places a chart of
  that type** where you released. Release it back over the palette
  instead -> cancels, nothing gets created. This is a real pick-up/
  drag/drop-to-place flow, not a conjure-on-any-pinch.
- **Pinch on empty canvas (not a palette button, not an existing chart)
  now does nothing** - removed entirely, per feedback that it made
  creation feel accidental and impossible to aim at a specific type.
- **Added a Pie chart type** alongside Bar and Line.
- Existing-chart interactions are unchanged: pinch+drag to move, Fist to
  dismiss, Two fingers to refresh data.
- **Deliberately not tackled yet** (flagged for later, not attempted
  half-done): editing a chart's exact values by hand, and importing real
  data from a file (CSV/Excel/XML) instead of mock random numbers. Two
  fingers still just re-randomizes for now.

## v11 changes (previous round) - AR Workspace mode

A sixth mode: **floating charts you conjure and drag around your actual
desktop** with a pinch - not confined to the app's own window, closer to
the "windows floating in the air" AR look. New module: `ar_overlay.py`.

- **A full-screen transparent, always-on-top, click-through window**
  (built the same way as the existing laser dot / cursor reticle - see
  `overlay.py` - just scaled up to the whole screen) sits over your
  entire desktop. Only the charts themselves are visible; everywhere
  else on screen is completely click-through, so it never gets in the
  way of anything else you're doing.
- **Pinch on an existing chart -> grabs and drags it**, following your
  fingertip smoothly for as long as the pinch is held.
- **Fist on a chart -> dismisses it.** **Two fingers on a chart ->
  refreshes its data** (a "live updating dashboard" effect).
- **Thumb Down hides/shows the WHOLE overlay** - reusing the exact same
  gesture every other mode already uses for "toggle control", so no new
  gesture vocabulary was needed for "hide everything."
- **A small hideable camera-preview thumbnail** (bottom-right corner, its
  own window) replaces needing the full webcam window on screen - just
  enough to confirm hand tracking is working. Toggle with `c`.
- Switch to this mode the same way as any other: hold Open Palm 1s for
  the mode menu, or press `6`/`a`.

**Read this before you demo it:**
- `-transparentcolor` (the trick that makes the overlay click-through and
  see-through) is **Windows-only** - same limitation the laser dot and
  cursor reticle already had. On any other OS, `AROverlay` safely falls
  back to a small ordinary window instead of trying to cover the whole
  screen opaquely (which would otherwise block your entire desktop with
  no way to close it) - watch the console for a `[WARN]` line telling you
  which mode it's running in.
- The camera preview needs **Pillow** (`pip install Pillow`) to convert
  camera frames for Tkinter. If it's missing, the preview silently does
  nothing rather than crashing the app - check the console for
  `[WARN] AR overlay unavailable`.
- **I could not test any of this against a real display or webcam** - it
  follows the same technique already proven elsewhere in this codebase,
  scaled up and given several moving pieces instead of one fixed dot,
  but a live pass on your actual machine is the real next step, not a
  formality. The most likely rough edges: performance of pumping five
  separate Tkinter windows (`Overlay`, `CursorReticle`, `LaserPointer`,
  `AROverlay`, `CameraPreview`) once per camera frame, and whether
  dragging feels smooth enough at your webcam's actual frame rate. If
  it's sluggish, the first thing to try is consolidating those into one
  shared `tk.Tk()` root with `Toplevel()` windows instead of five
  independent `tk.Tk()` calls.

## v9-v10 changes (previous rounds) - fist/mouse fix finished properly

- Replaced the real-cursor-based laser with a dedicated `LaserPointer`
  overlay window (own class in `overlay.py`) - the fist gesture now
  **only ever draws a dot in its own window**, never touches the real
  mouse or any key, so it can't cause a misclick or advance a slide no
  matter what.
- Added `is_hand_mostly_closed()` (`gesture_engine.py`) to catch a
  "half-open fist" that was still slipping through as a pinch even
  after the earlier fist-guard - a hand that's only partly curled can
  fail `classify_static_gesture()`'s stricter all-four-curled check and
  come back as `None` instead of `"fist"`, while the fingertip is still
  sitting right next to the thumb (i.e. still looks like a pinch by
  distance alone).

## v8 changes (previous round) - fixed the root cause + volume knob gesture

- **Found and fixed the real bug behind Presentation's misclicks.** A
  closed fist naturally brings the thumb right up against the curled
  fingers - which, by distance alone, looks exactly like a pinch. So
  making a fist was ALSO registering as a pinch-click underneath
  whatever the fist gesture was supposed to do. That's what was
  actually happening: "holding fist to laser" was firing real clicks,
  and since a left-click during a PowerPoint slideshow advances the
  slide, that's also why it kept "advancing/closing" and eventually
  ended the show. Fixed at the source - a fist can no longer register
  as a pinch, anywhere, in any mode.
- **Laser is back to the real thing.** Now that the fist/pinch bug above
  is actually fixed, it's safe to bring back the real version: holding
  a fist moves the actual system cursor while holding Ctrl, which is
  the exact combination PowerPoint uses to show its built-in laser dot
  on the slide itself - not just in our preview window.
- **Fixed fist/pointing being unreliable in Presentation & Media.**
  Traced this to a real architecture issue: swipe detection runs before
  static-gesture confirmation every frame and fires on hand SPEED
  alone, regardless of hand SHAPE - so raising a fist or a pointing
  finger with any real motion behind it could get read as a false swipe
  instead, eating that frame's confirmation. Since neither mode uses
  swipe for anything anymore, swipe-checking is now switched off
  entirely for their shared stabilizer, so it can't interfere.
- **Media - volume is now a knob gesture**, replacing swipe entirely as
  requested: pinch and twist your wrist like turning a dial - clockwise
  raises the volume, counter-clockwise lowers it, firing in discrete
  ticks like a real volume knob (or a mouse wheel). Flashes "VOLUME
  UP/DOWN" on screen each tick. If clockwise ends up lowering it for
  you instead (mirrored camera feeds can flip this), flip
  `DIAL_INVERT` in `config.py`.

## v7 changes (previous round)

- **Live swipe-speed meter** (top-right, Desktop and Media modes): a
  small bar showing your hand's current sideways speed against the
  speed actually needed to fire a swipe, with tick marks at the
  threshold and a "fires now" label once you've crossed it. Lets you
  see exactly how fast you need to move instead of guessing.
- **Edge-hold zone overlay** (Media mode): two semi-transparent bars
  now mark exactly where the left/right edge-hold zones are, so you're
  not guessing where "the edge" is. The bar fills in the longer you
  hold a fingertip inside it, showing the dwell timer building toward
  the seek-forward/rewind trigger.
- Both are purely visual/debug aids - no gesture behavior changed.

## v6 changes (previous round)

- **Presentation - laser fixed for real.** The laser used to hijack the
  real OS mouse (holding Ctrl + moving the actual cursor, which is how
  PowerPoint shows its laser), and that's exactly what caused "holding
  fist controls the mouse and misclicks when pinching." The laser is
  now a **glowing dot drawn only on our own preview window** - it never
  touches the real mouse, so it can't cause a misclick again.
- **Presentation - new Previous/Next.** Swipe and edge-hold are gone
  from Presentation; replaced with two static poses the AI model
  recognizes with real confidence: **Point (index up) = Next**,
  **Two fingers (V) = Previous**. No motion or timing required.
- **Presentation - Open Palm is menu-only now.** Starting the slideshow
  moved to **Thumb Up (tap)**. Open Palm doubling as both "start
  slideshow" and "open the mode menu" meant a presenter naturally
  resting an open hand for a moment (common while talking) could break
  the flow either way - Thumb Up has no such conflict.
- **Media - filled in the missing gestures:**
  - **Point = Previous track** (was missing entirely before)
  - **Edge-hold, point to the left/right edge and pause = Seek
    rewind/forward** (YouTube-style scrubbing, freed up here once
    Presentation stopped using edge-hold)
  - Volume (swipe) and Next track (fist) kept, swipe retuned to be
    easier to trigger at a normal hand speed
- **Desktop - two-finger right-click removed entirely** (including the
  "two-finger + pinch" left-click path) - it sat too close to a normal
  pinch and was firing right-clicks by accident. **Three fingers** is
  now the only right-click gesture.
- **Desktop - pinch-click flicker fixed.** Two changes stacked
  together: the raw pinch distance is now lightly smoothed before it's
  ever checked, and the yes/no check itself uses two thresholds instead
  of one (a tighter one to start a pinch, a looser one to end it), so a
  ratio sitting near the boundary can't flip the state every frame.
- **UI cleanup, including the real cause of the "???" glyphs:**
  OpenCV's on-screen font can't render the `→` arrow or `·` dot
  characters used in the help text - they were rendering as literal
  question marks. Replaced with plain `->`/`-` everywhere. Also made
  the help panel auto-size to its longest line instead of a fixed
  width, and tightened the spacing so it reads less cluttered.

## v5 changes (previous round)

- **Mode switching is gesture-driven, not keyboard-driven.** Hold
  **Open Palm for 1 second**, from any mode, to open an on-screen mode
  menu. Point left/right to highlight a mode, **pinch** to select it
  (or hold **Thumb Down** to cancel). Keyboard shortcuts (1-5/h/q)
  still work as a backup for development/testing.
- **Desktop mode:** dragging is triggered by **holding a pinch**
  instead of holding a fist - tap = click, hold = drag. This was the
  fix for the cursor jumping right before a drag started: closing a
  hand from "pointing" into a fist moves the fingertip during the
  transition, and the cursor was still tracking it. A pinch doesn't
  have that transition - the cursor already freezes the instant any
  pinch starts, so a held pinch turns into a drag with no jump.
- (Presentation/Media specifics from this round were superseded by v6
  above - see there for the current behavior.)


A webcam-based hand-gesture interaction platform, built up milestone by
milestone from a MediaPipe + OpenCV prototype into an AI-assisted,
AR-capable interaction system.

## 🚀 DEMO FOR TOMORROW (start here)

The most impressive single entry point is now:

```bash
python showcase.py
```

This launches a polished multi-mode demo that already includes:

| Mode | What it does | Real-world value |
|------|--------------|------------------|
| **1 Desktop Mouse** | Virtual cursor + tap-pinch click + hold-pinch drag + three-finger right-click | Control any app without touching mouse |
| **2 Presentation** | Point = next, Two fingers = previous, Thumb Up = start, Fist = laser | Control PowerPoint / Google Slides hands-free during a meeting |
| **3 Media** | Play (Thumb Up) / Pause (Open Palm), Point = previous, Fist = next, Pinch+twist = volume knob, Edge-hold = seek | Control music/video while hands are away from keyboard |
| **4 Drawing** | Pinch-to-draw whiteboard | Annotate while presenting |
| **5 Whack-a-Mole** | Fun mini-game | Shows the tracking is fun & responsive |
| **6 AR Workspace** | Pinch a palette icon (Bar/Line/Pie/Live/Files) and drag it out to place a chart on your desktop - Files loads a CSV from `ar_data/`; Fist = dismiss, Two fingers = refresh / cycle Live / reload CSV; layout is saved between launches | Iron-Man-style floating data, not confined to one app window |

Switching between these six modes is done entirely by gesture (hold Open Palm 1s → menu → point + pinch) - see below.

**AI component**: The demo uses MediaPipe’s official pre-trained **Gesture Recognizer** model (a real neural network) for high-level gestures, hybridized with the project’s own rule-based engine for precise pinch and swipe. You will see “AI: Closed_Fist / Open_Palm / …” on screen with a live confidence bar.

### Quick controls inside the demo
- **Thumb Down** → toggle control ON/OFF (safety first); also cancels the mode menu if it's open, and hides/shows the whole AR overlay in AR mode
- **Hold Open Palm for 1s** → open the gesture-driven mode menu (point + pinch to pick a mode) - the main way to switch modes
- **1 / 2 / 3 / 4 / 5 / 6** or **d / p / m / w / g / a** → switch modes directly (backup/dev shortcut)
- **C** → toggle the camera-preview thumbnail (bottom-right)
- **H** → toggle help overlay
- **Q** → quit

### Recommended 2-minute demo script
1. Start in **Desktop mode**, show cursor following your index finger, tap-pinch to click, hold-pinch to drag an icon.
2. Hold Open Palm for a second to open the **mode menu**, point at the Presentation tile and pinch to switch - no keyboard touched.
3. In **Presentation mode**, open a slide deck: point to advance, two-finger V to go back, thumb up to start the slideshow, hold a fist to show the laser dot.
4. Open the mode menu again and switch to **Media mode**: pinch and twist your wrist for volume, point for previous track, fist for next, hold at an edge to seek.
5. Show **Drawing mode** for annotation.
6. Mention that the gestures are recognized by a real ML model + a stability layer that prevents accidental triggers.

Also available:
```bash
python main_desktop.py      # pure Milestone-3 desktop control
python collect_gestures.py  # collect your own labeled data for a custom ML model
python main.py              # original drawing / game / piano demo
```

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
python generate_assets.py   # only needed once, or after deleting assets/
```

The showcase also needs the AI model (already included in `models/gesture_recognizer.task`).

`main_desktop.py` / `showcase.py` need `tkinter` for the HUD overlay.  
On Windows it ships with the standard python.org installer.  
On Linux: `sudo apt install python3-tk`.

Press `q` in the camera window to quit.

Swap any file in `assets/images/` or `assets/sounds/` with your own
(same filenames) any time — nothing else needs to change.

## Project layout

```
handtracker/
  config.py           # all paths & tunable constants live here - no hardcoded paths anywhere else
  gestures.py          # pinch/tap detection - the seed of the Milestone 2 gesture engine
  generate_assets.py   # regenerates placeholder icons/sounds
  main.py              # camera loop, wires hand tracking to features
  features/
    drawing.py         # pinch-to-draw canvas + control panel
    game.py             # whack-a-mole
    piano.py            # on-screen piano
  assets/
    images/, sounds/    # placeholder art/audio, freely replaceable
```

## Status

**Milestone 1 — Problem Definition & Planning: done.**
The original prototype refactored into this modular structure.

**Milestone 2 — Robust Hand Tracking & Gesture Engine: in progress.**
Built so far:
- `gesture_engine.py` — classifies each hand's landmarks every frame
  into `open_palm`, `fist`, `pointing`, `two_finger`, `thumb_up`, or
  `thumb_down` (pinch stays in `gestures.py` since it needs pixel
  distance, not angle/ratio classification). Also detects `swipe_left`/
  `swipe_right` from palm-center velocity.
- Stability layer (`GestureStabilizer`): a gesture must classify
  consistently for `GESTURE_CONFIRM_FRAMES` consecutive frames before
  it's confirmed, and won't re-fire for `GESTURE_COOLDOWN_SECONDS` after
  it does — this is the fix for "fingers briefly came close together
  shouldn't count as a click."
- `gesture_actions.py` — the gesture→action mapping layer, kept
  separate from recognition so remapping later doesn't touch detection
  code. Includes a `set_mapping()` hook for the user-customization work
  in Milestone 3.4.
- `gesture_demo.py` — a standalone harness showing live classification,
  confirmed gestures, and the resulting action, with an FPS readout.
  **Run this to stress-test recognition** across lighting, distance,
  hand size, partial visibility, and one-vs-two-hand conditions (per
  2.1) before we wire it into the real features.

Not done yet: the drawing/game/piano features still use the original
simple pinch/tap logic — they haven't been switched over to the new
engine yet. That's the next step, once the engine's been tested enough
on your end to trust it.

**Milestone 3 — Desktop Overlay & Computer Control: in progress.**
Built so far:
- `desktop_control.py` — thin adapter over `pyautogui` for real mouse/
  keyboard control (`DesktopController`), plus a `DryRunController` that
  logs actions instead of executing them. `main_desktop.py` falls back
  to dry-run automatically if `pyautogui`/a display isn't available, so
  the pipeline is still exercisable without touching your real mouse.
- `desktop_interaction.py` — the decision layer connecting gesture
  output to those calls: point to move the cursor (frozen during a
  pinch, so the pinch motion itself can't drag the click off target),
  pinch to click, a held fist to click-and-drag, two-finger to
  right-click, swipe to Alt+Left/Right, thumb up to Enter, and thumb
  down to toggle control on/off. Pure logic, unit tested with no real
  mouse/keyboard involved.
- `overlay.py` — a small always-on-top, movable HUD window (Milestone
  3.1) plus a `CursorReticle` — a ring that follows the mapped screen
  position live and flashes red on click, so it's clear where a pinch
  is about to click before it happens. Both run by being pumped once
  per camera frame rather than on their own thread, which avoids the
  flakiness of mixing Tkinter's and OpenCV's event loops across
  threads.
- `main_desktop.py` — the entry point wiring camera → gesture engine →
  desktop control → overlay together. **Desktop control starts OFF** —
  thumb-down toggles it on/off (not open palm: that's a natural resting
  hand shape, so it kept toggling by accident), so the program doesn't
  grab your mouse the instant it starts. Only one hand drives the
  cursor at a time; a second hand entering the frame is ignored rather
  than fighting for control, via a per-handedness-label "lock" that
  only releases once the tracked hand is gone for a beat.

Also fixed: `requirements.txt` had an unpinned `mediapipe>=0.10,<0.11`,
but MediaPipe releases from roughly 0.10.2x onward dropped the legacy
`solutions` API this whole project is built on (in favor of a newer
Tasks API) — a fresh install could silently pull a broken version.
Pinned to `mediapipe==0.10.14`, confirmed to still have `mp.solutions`.

Not done yet: application shortcuts (3.3 — open browser/notes, switch
apps, media controls), user-customizable gesture mappings via the
overlay UI (3.4), and click-through/transparent overlay styling.

## Roadmap (merged)

This merges the two milestone drafts into one ordered list — later
drafts split some of these into fewer, larger milestones, but the
scope is the same either way.

1. **Problem Definition & Planning** ✅ — this refactor.
2. **Robust Hand Tracking & Gesture Engine** — stress-test tracking
   across lighting/distance/hand-size conditions; build a real
   gesture-recognition layer (open palm, fist, point, pinch, two-finger,
   swipe, thumb up/down, directional movement); add stability measures
   (thresholds, time-based confirmation, smoothing, cooldowns,
   multi-frame confirmation); separate "gesture detected" from "action
   triggered" via a gesture→action mapping layer.
3. **Desktop Overlay & Computer Control** — always-on-top, movable,
   low-resource overlay; virtual cursor via index finger; pinch-to-click;
   scroll/drag; app shortcuts; user-defined gesture mappings.
4. **Video Meeting & Collaboration Integration** — slide control,
   screen-share/mute toggles, window switching, media control,
   presentation annotation (draw/circle/highlight/clear) — the
   project's original real-world use case.
5. **Machine Learning-Based Gesture Recognition** — collect a labeled
   landmark dataset, normalize for hand size/distance/user, train and
   compare classifiers (kNN/SVM/Random Forest/NN, sequence models for
   dynamic gestures), evaluate (accuracy/precision/recall/confusion
   matrix/latency/false-positive rate), integrate into the pipeline in
   place of hand-written rules.
6. **Augmented Reality Interaction** — render virtual objects over the
   camera feed; point/pinch/move/resize/rotate/release; object
   placement; a full point→grab→move→rotate→release demo.
7. **Complete AI-Assisted Interaction Platform** — combine everything
   into one system with context-aware gestures (pinch means different
   things in Desktop/Presentation/Drawing/AR modes), clear mode
   indication, optional voice input, user profiles/custom gestures.
8. **Real-World Testing, Optimization & Deployment** — functional +
   integration + real-world condition testing, performance
   optimization, privacy/security review (local processing, no
   unnecessary storage), user testing, packaging as a standalone
   installable app, full documentation, and a maintenance plan.

We'll work through these roughly in order, polishing and testing at
the end of each one before moving to the next.

---

## Changelog since v28 (v29 - v33)

**Overlay / UI**
- Drawing mode and Whack-a-Mole now render on the floating overlay (drawing canvas is letterboxed with a *uniform* scale so diagonals stay straight on any screen aspect ratio).
- Whack-a-Mole runs in screen space: geometry is stored normalized and denormalized per target space, and the overlay feeds it the same `map_to_screen` point the cursor dot uses (one shared game state, playable from the cv2 window or the overlay).
- Status/help drawers expand on mouse hover **or by pointing with the tracked hand**; they ease open (rounded panels, smooth grow) instead of snapping.
- Dark halos behind glass-style HUD lines/text so they stay visible over light backgrounds.
- Camera-preview thumbnail disabled; `ImageTk.PhotoImage` now bound to its own canvas (`master=`) - fixes the multi-`Tk()` "pyimage1 doesn't exist" crash. Overlay draw errors are caught and logged instead of ending the session.

**Tracking / gestures**
- `MIN_TRACKING_CONFIDENCE` 0.7 -> 0.5 (fast flicks no longer drop tracking); scroll notches scale up to 40.
- A stray single-frame "fist" reading no longer cancels an in-progress pinch/drag.
- Stretching a selection on a small object lets the floor grow a little so the change is visible.
- Media mode: point-and-hold at the screen edge to seek no longer also fires "previous track".

**AR 3D**
- Function graphs: type a formula in `ar_functions/*.txt` (safe AST-whitelisted parser; `if/else`, `and/or` vectorized).
- **DENSE** palette toggle: fills surfaces with shaded, heat-coloured faces (opt-in; raises the LOD item budget).
- Bars are heat-mapped by value; floor outline has a dark halo; each object sways with its own phase/frequency plus a secondary tilt; new objects ease in ("pop").
- Palette/tile buttons are rounded, and the hovered palette button shows a charge-up arc as a pinch closes.

Tests: `python tests/run_all.py` (no camera or display needed).

## Changelog v34

- **Always-on reticle** on the main overlay (every mode, with hand control on *or* off; hidden only where another window already draws a cursor: AR mode, Desktop with control on). Tightens/fills while pinching; in the mode menu you can point at tiles.
- **Whack-a-Mole needs a pinch**: whacking, Start and Close all fire on a fresh pinch, not on proximity.
- **Drawing mode lives on the overlay** (`overlay_drawing.py`): full toolbar (pen, eraser, 6 colours, 3 thicknesses, clear, save PNG, hide/show) in screen space, ink lands under the reticle. Ink is tiled so only touched tiles are uploaded (~0.1 ms/frame vs ~56 ms before).
- **Overlay look**: Segoe UI type, gradient rounded panels, text on dark pills (no more doubled "halo" text), corner brackets removed, polished mode menu.
- **AR**: CLEAR button (press once = arms "SURE?", press again = clears 2D + 3D; 3D stays recoverable via undo); gradient buttons with labels inside; readable chart numbers/labels/tiles; dense mode uses ~49 solid faces instead of ~100 stippled ones.
- **Fist-delete**: also tests where the fingertip was over the last ~0.8 s (a closing fist curls the fingertip away from the graph) and retries for ~1 s if a recent pinch made the 3D layer refuse.

## Changelog v35

- **Pinch-stable pointer on the overlay, in every mode.** The reticle, Drawing, Whack-a-Mole, the mode menu and help-hover now read a `PinchStableCursor` (shared `pinch_stable.py`, same logic AR mode and Desktop mode use): it locks to where the fingertip sat relative to the index knuckle just before the pinch closed and follows only the knuckle until the fingers open, so the pointer no longer slides down with the curling finger.
- **AR toolbox is a collapsible box.** Normally just a small "+ ADD" tab; pointing at it eases open a compact 3x3 box (closes shortly after you leave; stays open while you hold a pick-up or CLEAR is armed). Buttons are only live while it's open. File/shape/function tile rows now sit to the right of the box.

## Using it in video calls (Zoom / Google Meet / Microsoft Teams)

**One-time setup**
1. Install OBS Studio (it installs the "OBS Virtual Camera" driver) and `pip install pyvirtualcam`.
2. Start the app with `python showcase.py --vcam --call` (or press **V** while running to turn the virtual camera on/off; the choice is remembered).
3. In Zoom/Meet/Teams, pick **OBS Virtual Camera** as your camera. The app keeps the real webcam (so gesture control keeps working) and publishes the *clean, un-mirrored* video to the call - never the debug view with skeletons/HUD.

**Why a virtual camera:** a webcam can normally be opened by only one program at a time.

**Keys:** `C` or `7` Video Call mode - `V` virtual camera - `O` cycle "HUD hidden from screen share" (auto / on / off) - `T` tutorial.

**Call mode** (hand control must be ON; Thumb Down toggles it)
- Hold **fist** ~1 s with the hand still = mute/unmute; **peace sign** = camera on/off; **open palm** = raise/lower hand. A ring fills around the pointer while you hold; drop your hand to cancel.
- Or pinch a button on the on-screen call bar (Mute / Camera / Hand / Share).
- Talking with your hands never triggers anything: the gesture must be held, nearly still, then there is a 3 s cool-down. *Leaving a call is deliberately not mappable.*
- Shortcuts are only sent when the **foreground window** is a recognised call app (Zoom, Meet in a browser tab, Teams) - otherwise you get a hint instead, so Ctrl+D can't bookmark a page by accident. Defaults (Windows): Zoom Alt+A / Alt+V / Alt+Y; Meet Ctrl+D / Ctrl+E / Ctrl+Alt+H; Teams Ctrl+Shift+M / O / K. Edit `callapp.HOTKEYS` if you changed yours.

**Screen sharing:** on Windows 10 2004+ the main HUD overlay is excluded from capture (you see it, the audience doesn't), except in Drawing mode, where your strokes are meant to be seen. The laser pointer and AR charts stay visible to the audience. Share a window or your screen, but not the app's own debug window.

**CPU:** `--call` caps the rate at 24 fps and uses the lite hand model (it no longer changes the capture size - a non-native size such as 960x540 made most webcams scale the picture and look soft).

## Tutorial
Press **T** or pinch the **Tutorial** button in the help panel (point at the "? Help" tab bottom-left). It never starts by itself. While it runs, hand control is switched off so lessons can't click your real desktop; it is restored afterwards.

## Changelog v36
Video-call support (virtual camera, call mode with hold-to-confirm + foreground-app guard, HUD hidden from screen share, `--call` profile), on-demand tutorial, persisted settings (`~/.handtracker/settings.json`), help panel shortcuts + Tutorial button.


## Changelog v37 - camera sharing, reachable menus, call mode fixes

**Sharing the webcam with other apps - three options, best first**
1. **Virtual camera (`V`, or `--vcam`)** - the call and hand control use the camera at the same time, no switching. Needs OBS Studio (it provides the driver) + `pip install pyvirtualcam`. If it fails, the console prints a setup checklist saying exactly what is missing.
2. **Release the camera (`K`, or the "Camera: ON (release)" button in the top-right panel)** - frees the webcam for Zoom/OBS/etc. Resume with **Ctrl+Alt+K from anywhere** (Windows), or `K` in the debug window. *Hands cannot take it back:* with no camera there is nothing to see them, so the way back has to come from the keyboard.
3. **`--shared-camera` (experimental)** - uses the Media Foundation backend, which on some Windows 10/11 setups lets several apps open the same camera at once.

**Fixes**
- Top-right and bottom-left panels no longer collapse while you reach for a button inside them (they now count the whole grown panel, plus a 0.8 s grace period). The top-right panel has pinch buttons for **Camera** and **Control on/off**; the help panel's **Tutorial** button is reachable.
- Video Call mode switches hand control ON when you enter it (the app starts with control OFF, so call gestures silently did nothing) and says when control is off or no call window is in front.
- Held gestures tolerate brief misclassified frames; a flickering hand still can't fire.
- Soft camera image: caused by `--call` shrinking capture to a non-native size - fixed; the startup log now prints what the camera actually delivers and warns if it's not the requested size; autofocus is requested on open.
