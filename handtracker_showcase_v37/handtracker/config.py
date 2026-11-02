"""
config.py
---------
Single source of truth for paths, frame size, and shared constants.

The original script hardcoded absolute Windows paths like
D:\\Hand TRacker\\images\\pencil.png, which only ever worked on one
machine. Everything here is resolved relative to this file, and
resource_path() still supports PyInstaller (_MEIPASS) for when we
package the app in Milestone 8.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resource_path(*relative_parts):
    """Resolve a path that works both when run directly and when frozen
    into a PyInstaller executable."""
    base_path = getattr(sys, "_MEIPASS", BASE_DIR)
    return os.path.join(base_path, *relative_parts)


# ---- Frame / camera ----
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_INDEX = 0

# ---- Asset paths ----
IMAGES_DIR = resource_path("assets", "images")
SOUNDS_DIR = resource_path("assets", "sounds")

PENCIL_ICON = os.path.join(IMAGES_DIR, "pencil.png")
ERASER_ICON = os.path.join(IMAGES_DIR, "eraser.png")
MOLE_IMAGE = os.path.join(IMAGES_DIR, "mole.png")
START_BUTTON_IMAGE = os.path.join(IMAGES_DIR, "startgame.png")

# Two octaves (C4-B5), natural notes only, matching the original letter-key layout.
PIANO_NOTE_FILES = {
    "C4": os.path.join(SOUNDS_DIR, "C4.wav"),
    "D4": os.path.join(SOUNDS_DIR, "D4.wav"),
    "E4": os.path.join(SOUNDS_DIR, "E4.wav"),
    "F4": os.path.join(SOUNDS_DIR, "F4.wav"),
    "G4": os.path.join(SOUNDS_DIR, "G4.wav"),
    "A4": os.path.join(SOUNDS_DIR, "A4.wav"),
    "B4": os.path.join(SOUNDS_DIR, "B4.wav"),
    "C5": os.path.join(SOUNDS_DIR, "C5.wav"),
    "D5": os.path.join(SOUNDS_DIR, "D5.wav"),
    "E5": os.path.join(SOUNDS_DIR, "E5.wav"),
    "F5": os.path.join(SOUNDS_DIR, "F5.wav"),
    "G5": os.path.join(SOUNDS_DIR, "G5.wav"),
    "A5": os.path.join(SOUNDS_DIR, "A5.wav"),
    "B5": os.path.join(SOUNDS_DIR, "B5.wav"),
}

# ---- Gesture tuning ----
PINCH_DISTANCE_THRESHOLD_PX = 40
TAP_DELAY_SECONDS = 0.5

# One Euro Filter tuning (see gestures.PointSmoother). min_cutoff is the
# smoothing floor when the hand is still (lower = smoother but laggier);
# beta controls how much fast movement relaxes that smoothing (higher =
# less lag on fast moves, but more jitter tolerated while moving).
SMOOTHING_MIN_CUTOFF = 1.0
SMOOTHING_BETA = 0.5

# MediaPipe confidence thresholds.
# Detection only runs when a hand is being freshly ACQUIRED (first seen,
# or re-acquired after being lost) - raised from the library default
# (0.5) to reject noise/false positives at that moment, with no ongoing
# per-frame cost once a hand is actually being tracked.
MIN_DETECTION_CONFIDENCE = 0.7
# Tracking confidence, by contrast, is checked on EVERY frame a hand is
# already being tracked - this used to also be raised to 0.7 to reject
# jittery landmarks on a STILL hand, but that's the wrong tool for that
# job (the One Euro Filter above already adapts its own smoothing to
# velocity - heavy when still, light when moving fast - specifically so
# jitter-on-stillness doesn't need a blunt confidence cutoff to fix). A
# fast hand naturally produces blurrier, lower-confidence landmarks even
# when MediaPipe is tracking it perfectly well; at 0.7 that read as
# "tracking lost" on anything more than a gentle motion, forcing a full
# re-detect (slower, and can drop a frame or two) right in the middle of
# the one gesture - a fast swipe/flick - that most needs tracking to NOT
# drop out. Back to the library default here fixes that without giving
# up the detection-side noise rejection above.
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# 0 = lite model (fastest, slightly less precise), 1 = full model. A more
# consistent, higher framerate matters more than the small accuracy gain
# here, since the One Euro Filter above already depends on real elapsed
# time - a choppier framerate gives it worse data to smooth in the first
# place, however good the filter math is.
MODEL_COMPLEXITY = 0

# main_desktop.py uses the full model instead of the lite one above.
# MediaPipe's landmark estimates get noticeably noisier during a fist
# (fingers occluding each other, and the thumb overlapping the folded
# fingers), which was feeding directly into misfired clicks/drags - the
# full model handles that kind of occlusion better, at some FPS cost
# that's worth paying here specifically for the added reliability.
DESKTOP_CONTROL_MODEL_COMPLEXITY = 1

SHOW_FPS = True

# ---- Gesture engine (Milestone 2) ----
# A candidate gesture must classify the same way for this many consecutive
# frames before it's considered "confirmed" - this is what stops a single
# flickered misread (or fingers briefly brushing together) from firing an
# action, per the milestone's gesture-sensitivity goal.
GESTURE_CONFIRM_FRAMES = 5

# Once a gesture fires, it won't fire again until this many seconds pass,
# even if it's still being detected every frame (prevents a held gesture
# from spamming its action).
GESTURE_COOLDOWN_SECONDS = 0.8

# Swipe detection looks at palm-center displacement over this many frames...
SWIPE_HISTORY_FRAMES = 6
# ...and fires if the palm crosses this fraction of the frame width per
# second (normalized 0-1 coordinates, so this is resolution-independent).
# SWIPE_VELOCITY_THRESHOLD itself is defined further down, next to the
# edge-dwell settings it's tuned alongside.

# Hold a pointing fingertip inside this fraction of the frame's width,
# measured from each edge, for EDGE_DWELL_SECONDS to fire an edge
# action. Used by Media mode for seek forward/back (rewind) - a
# deliberate "move it there and hold" gesture is far more repeatable
# than a fast flick, and it was freed up here once Presentation mode's
# prev/next moved onto discrete Pointing/Two-finger poses instead.
EDGE_DWELL_ZONE_RATIO = 0.16
EDGE_DWELL_SECONDS = 0.35

# Retuned looser than the original 0.75 - swipe was still being missed
# by people who moved their hand at a normal, non-rushed pace. Now only
# used for Desktop mode's browser back/forward and Media mode's volume
# up/down (Presentation mode moved off swipe entirely, onto discrete
# Pointing/Two-finger poses - see showcase.py - since a motion gesture
# is inherently harder to land consistently than a static one).
SWIPE_VELOCITY_THRESHOLD = 0.45

# Vertical swipes (Desktop mode scroll). Palm speed in frame-HEIGHTS per
# second. Only counted while the hand is open (or mid-motion) - never
# while pointing or pinching, so moving the cursor up/down or dragging
# can't scroll the page.
SWIPE_VERTICAL_VELOCITY_THRESHOLD = 0.40
# A vertical swipe may repeat sooner than other gestures (a scroll is
# something you flick several times in a row).
SCROLL_SWIPE_COOLDOWN_SECONDS = 0.30
# Wheel notches per swipe: faster flick = bigger scroll, clamped.
# Raised across the board - a bare-threshold swipe used to round down to
# the 3-notch floor and even a very fast flick (2x frame-height/sec,
# hard to sustain in practice) only reached 10, short of the 12 ceiling,
# which read as "scrolling is slow" regardless of how hard the flick
# was. MIN is a touch higher too so even a gentle swipe moves the page a
# useful amount; MAX gives a real flick the big jump a mouse-wheel flick
# would give, PER_VELOCITY is raised to actually reach that ceiling at a
# realistically fast (not superhuman) flick speed.
SCROLL_NOTCHES_MIN = 5
SCROLL_NOTCHES_MAX = 40
SCROLL_NOTCHES_PER_VELOCITY = 12.0
# Swipe UP scrolls the page DOWN (touchscreen convention - the content
# follows your hand). Set True to reverse.
SCROLL_INVERT = False

# A finger/thumb joint is considered "straight" (i.e. the finger is
# extended) above this angle in degrees; below it, the joint reads as
# bent/curled. Thumb gets its own, slightly lower threshold since its
# natural range of motion rarely reaches as straight as the other
# fingers even when fully extended.
#
# 160 (the original value) is close to anatomically perfectly straight.
# In practice almost nobody's relaxed "open hand" PIP angle measures
# that high on every finger at once, especially the pinky and at
# anything but a face-on camera angle - MediaPipe's landmark jitter
# alone is often enough to keep one finger under 160 on any given
# frame. Since classify_static_gesture() requires ALL FOUR fingers to
# read "extended" for open_palm but ALL FOUR to read "curled" for fist,
# a too-strict threshold doesn't just miss open_palm - a hand that's
# genuinely open but has one finger dip under the line just misses
# open_palm's ">=4" test and, if the other three also read borderline,
# can fall through to fist's "==0 extended" test instead. That's the
# "open hand reads as a fist" symptom. 150 gives real hands enough
# margin while still clearly separating from a true fist's much
# sharper PIP bend (commonly <100deg). Tune against your own hand/
# camera with `python gesture_demo.py` (press d for the debug readout)
# if it still misreads.
FINGER_STRAIGHT_ANGLE_DEG = 150
# A finger only counts as CURLED (needed for "fist", thumbs up/down) when
# it is clearly folded, not merely "not straight". Fingers between the two
# (half-open hand) are neither, and never form a fist. Lower the angle /
# ratio to make fist stricter, raise them to make it more forgiving.
FINGER_CURLED_ANGLE_DEG = 105
FINGER_CURLED_REACH_RATIO = 1.25
THUMB_STRAIGHT_ANGLE_DEG = 140

# How far the thumb tip must reach from the wrist, as a multiple of how
# far the four fingertips reach from the wrist, before the thumb counts
# as extended. Both distances are measured in the same projection, so
# camera foreshortening largely cancels out - which a fixed-reference
# normalization does not (it broke when knuckles faced the camera).
# In a fist the thumb sits among the folded fingers (ratio near 1); in a
# thumbs up/down it reaches well past them.
THUMB_REACH_RATIO = 1.35

# How vertical the thumb must point (0 = sideways, 1 = straight up/down)
# to count as a thumbs up/down rather than just a thumb sticking out.
THUMB_VERTICAL_RATIO = 0.55

# ---- Media mode volume "knob" gesture ----
# Pinch and rotate your wrist like turning a dial: each
# DIAL_DEGREES_PER_TICK of accumulated rotation while pinching fires one
# volume step. Small enough to feel responsive, large enough that
# natural hand jitter while just holding a pinch still doesn't fire
# ticks by itself.
DIAL_DEGREES_PER_TICK = 18.0
DIAL_TICK_COOLDOWN_SECONDS = 0.12
# Flip this (True) if turning clockwise ends up LOWERING the volume for
# you instead of raising it - a mirrored webcam feed (most webcam UIs
# mirror the image so it behaves like a mirror) can flip which way a
# real-world clockwise turn reads on screen.
DIAL_INVERT = False

# ---- Gesture-driven mode menu ----
# Holding OPEN PALM steady for this many seconds opens (or closes) the
# on-screen mode menu, from any mode, with no keyboard involved. It's a
# deliberate HOLD (see gesture_engine.HoldTrigger) rather than the usual
# quick multi-frame confirm, so it can't be triggered by a hand just
# passing through an open-palm shape on its way to/from another gesture.
MENU_OPEN_HOLD_SECONDS = 1.0

# ---- Desktop control (Milestone 3) ----
# Cursor smoothing uses its own, snappier One Euro Filter settings than
# the general-purpose defaults above, which were tuned for smooth
# drawing strokes rather than fast, responsive cursor tracking. A higher
# beta lets the filter cut smoothing sharply during fast hand motion
# (reducing lag), while still damping jitter when the hand is nearly
# still.
CURSOR_SMOOTHING_MIN_CUTOFF = 1.5
CURSOR_SMOOTHING_BETA = 1.2

# If the hand being used for desktop control disappears, keep "locked"
# onto it (ignoring any other hand that enters frame) for this long
# before allowing a different hand to take over control. This is what
# stops a second hand entering the frame from stealing/destabilizing
# the cursor mid-use.
HAND_LOCK_TIMEOUT_S = 1.0

# Pinch, measured relative to the hand's own size rather than a fixed
# pixel distance (see gestures.pinch_ratio) - this keeps click detection
# consistent whether the hand is close to or far from the camera.
PINCH_RATIO_THRESHOLD = 0.4

# showcase.py uses hysteresis instead of the single threshold above: a
# TIGHTER ratio to start a pinch, a LOOSER one to end it. Without this, a
# ratio sitting right at the boundary (very common - people don't hold a
# pinch perfectly still) flickers the pinch state on/off every frame,
# which is what made clicking feel finicky even with debouncing after
# the fact. Combined with the light EMA smoothing showcase.py also
# applies to the raw ratio before thresholding, this is the actual fix
# for that flicker.
PINCH_RATIO_ENGAGE = 0.42
PINCH_RATIO_RELEASE = 0.49

# A held gesture (fist-drag, pinch) must change for this many consecutive
# frames before the change is believed - filters out single-frame
# misclassifications during hand motion without adding noticeable lag to
# a deliberate release. Drag gets slightly more tolerance than pinch
# since an interrupted drag is more disruptive than a slightly-delayed
# click.
# DRAG_HOLD_CONFIRM_FRAMES now gates the START of a fist-drag in
# desktop_interaction.py (it previously only existed here and was never
# actually applied - a single frame that read "fist" grabbed the mouse
# immediately, with no confirmation at all, while releasing already
# required DRAG_RELEASE_GRACE_FRAMES of the opposite. That asymmetry is
# what made an open, relaxed hand occasionally start a drag out of
# nowhere: one noisy frame was all it took.
# No longer used by desktop_interaction.py (drag is now pinch-hold
# based - see DRAG_HOLD_SECONDS below), kept in case a future gesture
# still wants fist-style hold-confirm behavior.
DRAG_HOLD_CONFIRM_FRAMES = 4
PINCH_CONFIRM_FRAMES = 3

# Desktop drag is now triggered by HOLDING a pinch (not a fist) - see
# desktop_interaction.py. A pinch shorter than this many seconds is a
# tap -> click; held past it, the same pinch becomes a click-and-drag.
# Using the pinch itself (instead of curling into a fist afterwards)
# means the cursor never has to jump mid-transition: the cursor already
# freezes the instant a pinch starts (see desktop_interaction.py), so
# there's no separate "closing into a fist" motion left to drag the
# reticle off target before the drag actually begins.
DRAG_HOLD_SECONDS = 0.35  # legacy: desktop pinch is now a real mouse button (see below)

# --- Desktop pinch behaves like a physical mouse button ---------------
# Pinch = button DOWN the instant it engages, release = button UP. A quick
# pinch is therefore a click, a held pinch is a drag, two quick pinches
# are a native double-click - no waiting for a hold timer or for the
# release to decide what the gesture "was", which is what made it feel
# laggy.
#
# Engaging is instant (the ratio hysteresis already prevents chatter);
# only RELEASE is debounced, so one jittery frame can't drop a drag.
PINCH_RELEASE_CONFIRM_FRAMES = 2
# Closing thumb and index drags the fingertip down before the pinch is
# detected, so the cursor is rewound to where the fingertip was this many
# seconds BEFORE the pinch engaged - i.e. where you were actually aiming.
PINCH_ANCHOR_LOOKBACK_SECONDS = 0.20
# While pinched the cursor stays locked to that anchor, and only starts
# moving (a drag) once the hand travels this far (normalized frame
# fraction) - so a tap can't nudge the click off target.
PINCH_DRAG_DEADZONE_NORM = 0.02

# No longer used by desktop_interaction.py now that drag ends when the
# (already-debounced) pinch releases, rather than waiting out a
# fist-misread grace period. Kept for reference.
DRAG_RELEASE_GRACE_FRAMES = 4

# Crop this fraction off each edge of the camera frame before mapping to
# the screen, so the user doesn't have to stretch their hand to the very
# edge of the frame to reach the edge of the screen.
CURSOR_MARGIN_RATIO = 0.15

# Keep the mapped cursor this many pixels inset from the true screen
# edge. Without this, a hand near the crop margin above maps to the
# literal corner pixel - which is exactly what pyautogui's fail-safe
# watches for, so ordinary tracking noise near the frame edge could
# self-trigger an emergency abort and crash the program.
CURSOR_EDGE_INSET_PX = 8

# Minimum time between two clicks, so a held pinch doesn't spam clicks.
CLICK_COOLDOWN_SECONDS = 0.4

# Desktop control starts OFF - a thumb-down gesture toggles it. This is
# a deliberate safety default so the program doesn't grab the mouse the
# instant it starts.
DESKTOP_CONTROL_DEFAULT_ENABLED = False

# pyautogui adds a small pause after every call by default (0.1s); that's
# too slow for continuous cursor tracking, so it's reduced here. Its
# fail-safe (drag the mouse into a screen corner to abort) is left on in
# desktop_control.py regardless of this setting.
PYAUTOGUI_PAUSE = 0.0

# ---- Drawing ----
DRAWING_LINE_THICKNESS = 10
ERASER_RADIUS = 20

COLORS_BGR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
}
DEFAULT_DRAW_COLOR = "green"

# ---- Whack-a-mole ----
MOLE_RADIUS = 40
MOLE_SPAWN_INTERVAL_S = 1.5
MOLE_SPAWN_BOUNDS = (100, 200, 800, 600)  # xmin, ymin, xmax, ymax
COUNTDOWN_DURATION_S = 3

# ---- Piano ----
PIANO_KEY_HEIGHT = 100

# ---- Mole hit feedback ----
HIT_SOUND_FILE = os.path.join(SOUNDS_DIR, "hit.wav")
HIT_EFFECT_IMAGE = os.path.join(IMAGES_DIR, "hit_effect.png")
HIT_EFFECT_DURATION_S = 0.35


# --- Presentation laser dot ---------------------------------------------
# Tuned for responsiveness rather than steadiness: a much higher beta than
# the cursor's, so the filter stops smoothing as soon as the fingertip
# moves (no trailing/rubber-band), while a still hand is still steadied.
LASER_SMOOTHING_MIN_CUTOFF = 1.0
LASER_SMOOTHING_BETA = 8.0
# The dot keeps following the fingertip for this long after the hand stops
# reading as "pointing" - bridges one-frame shape flicker mid-motion, which
# otherwise froze the dot and made it snap ahead when tracking resumed.
LASER_POINT_GRACE_SECONDS = 0.30


# ---------------------------------------------------------------------------
# AR mode: 3D objects (ar3d.py / ar3d_input.py)
# ---------------------------------------------------------------------------
# Push/pull depth: while holding an object, moving your hand toward the
# camera (hand looks bigger) brings it toward you; away pushes it back.
# Set False if hand-size noise makes objects drift in your lighting.
AR3D_DEPTH_FROM_HAND_SIZE = True
AR3D_DEPTH_GAIN = 700.0          # px of depth per +100% change in apparent hand size
AR3D_DEPTH_DEADZONE = 0.05       # ignore hand-size changes smaller than 5%

# Two-hand twist: pushing one hand toward the camera while the other stays
# put yaws the object. Radians of yaw per +100% relative hand-size change.
# NEGATE this if the object turns the wrong way on your setup - the sign is
# derived from geometry but has not been checked against a live camera.
AR3D_YAW_GAIN = 2.5

# Tapping the floor of a chart clears its selection: a pinch shorter than
# this, that moved less than this many pixels.
AR3D_TAP_MAX_SECONDS = 0.35
AR3D_TAP_MOVE_PX = 22

# Pinch-stable cursor (AR mode). Pinch ratio = thumb-index gap / hand size.
# Below LOCK_ON the cursor stops following the curling fingertip and tracks
# the palm instead (so it doesn't sag toward the thumb as you pinch); above
# LOCK_OFF it goes back to following the fingertip, easing over
# RELEASE_BLEND seconds. Pinch itself engages at ~0.42 (PINCH_* above).
AR_PINCH_LOCK_ON = 0.90
AR_PINCH_LOCK_OFF = 1.05
AR_PINCH_RELEASE_BLEND = 0.15

# Two-hand gestures ease toward their target with this time constant
# (seconds). Larger = smoother but laggier; hand-distance noise otherwise
# makes the object's size/rotation visibly flicker.
AR3D_SMOOTH_TAU = 0.06

# A fist / two-finger over a 3D object only removes / refreshes it when no
# pinch is active or was active within this many seconds. (A hard two-hand
# pinch curls your other fingers and can be classified as a fist.)
AR3D_DESTRUCTIVE_COOLDOWN = 0.8


# ---------------------------------------------------------------------------
# Performance (v25)
# ---------------------------------------------------------------------------
# Camera: capture on a background thread (always hands the loop the NEWEST
# frame, overlaps decode with tracking) and ask the driver for MJPG - many
# webcams default to uncompressed YUY2, which is only ~10 fps at 1280x720.
# If the picture looks wrong or the camera won't open, set CAMERA_THREADED
# False and/or CAMERA_FOURCC = "" to get the old behaviour.
CAMERA_THREADED = True
CAMERA_FOURCC = "MJPG"
CAMERA_FPS = 30
CAMERA_BACKEND = None            # None = OpenCV default. On Windows, "dshow" is worth trying.

# AR two-hand tracking: a 2-hand MediaPipe tracker with only ONE hand in view
# runs full palm detection on every frame (hunting for a second hand). Instead
# use the cheap 1-hand tracker and probe for a second hand every N frames;
# once two hands are seen, track both every frame until one has been gone for
# DUAL_FALLBACK_FRAMES. A second hand is noticed within ~N frames (0.2 s at 30
# fps, N=6). Set N = 1 to always use the 2-hand tracker.
AR_SECOND_HAND_PROBE_FRAMES = 6
AR_DUAL_FALLBACK_FRAMES = 12

# Adaptive quality: when a frame's compute exceeds ~85% of the frame budget
# for a second, step down what is DRAWN (thumbnail rate, 3D detail, help
# panel) - never tracking or gesture logic; step back up when there is
# headroom. A small "PERF L<n>" tag shows on the HUD while degraded.
ADAPTIVE_QUALITY = True
TARGET_FPS = 30

# Hand skeleton drawn with direct OpenCV calls (several times cheaper than
# mediapipe's draw_landmarks). False = use mediapipe's.
FAST_SKELETON = True
