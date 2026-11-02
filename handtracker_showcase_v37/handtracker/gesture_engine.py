"""
gesture_engine.py
------------------
Milestone 2: a dedicated, reusable gesture-recognition layer.

Instead of every feature (drawing/game/piano) reimplementing its own
ad-hoc "is this pinch" check, this module classifies a hand's landmarks
into one of a fixed set of named gestures every frame, then runs that
classification through a stability pipeline (multi-frame confirmation +
cooldown) before it's considered "detected". Confirmed gestures are
handed to a configurable gesture-to-action mapping (see
gesture_actions.py) so recognition stays decoupled from what happens
as a result - the same separation the milestone calls for.

Supported static gestures: open_palm, fist, pointing, two_finger,
three_finger, thumb_up, thumb_down. Pinch is intentionally handled by
gestures.is_pinching_from_points() instead (it needs pixel-space
distance between two points, not this angle/ratio-based classification).
Supported dynamic gestures: swipe_left, swipe_right, swipe_up, swipe_down.
"""
import math
from collections import deque

import config

_FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
_FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
_FINGER_MCPS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def _xyz(hand_landmarks, idx):
    lm = hand_landmarks.landmark[idx]
    return (lm.x, lm.y, getattr(lm, "z", 0.0))


def _angle_deg(a, b, c):
    """Angle at vertex b formed by rays b->a and b->c, in degrees. Using
    3D points (MediaPipe gives a rough relative-depth z) makes this
    reasonably stable across hand rotation toward/away from the camera,
    not just rotation in the image plane."""
    v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    if mag1 < 1e-9 or mag2 < 1e-9:
        return 180.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _is_finger_extended(hand_landmarks, finger):
    """A finger is 'extended' if its middle (PIP) joint is nearly
    straight - i.e. the angle formed by MCP-PIP-TIP is close to 180deg.
    A curled finger bends sharply at that joint instead. This is
    rotation-independent in a way a raw distance-from-wrist check isn't:
    it only cares about the finger's own joint angle, not how far the
    fingertip happens to project from the wrist at the camera's current
    viewing angle."""
    mcp = _xyz(hand_landmarks, _FINGER_MCPS[finger])
    pip = _xyz(hand_landmarks, _FINGER_PIPS[finger])
    tip = _xyz(hand_landmarks, _FINGER_TIPS[finger])
    return _angle_deg(mcp, pip, tip) > config.FINGER_STRAIGHT_ANGLE_DEG


def _thumb_straightness(hand_landmarks):
    """Minimum of the two thumb joint angles (at the MCP and IP joints).
    A fully extended thumb keeps both joints nearly straight; a thumb
    curled tightly into the palm bends sharply at one or both."""
    cmc = _xyz(hand_landmarks, 1)
    mcp = _xyz(hand_landmarks, 2)
    ip = _xyz(hand_landmarks, 3)
    tip = _xyz(hand_landmarks, 4)
    angle_at_mcp = _angle_deg(cmc, mcp, ip)
    angle_at_ip = _angle_deg(mcp, ip, tip)
    return min(angle_at_mcp, angle_at_ip)


def _dist2(a, b):
    """2D distance, ignoring z. MediaPipe's z is a rough relative-depth
    estimate and is by far its noisiest axis; including it in distance
    measurements adds jitter without adding much real information."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _thumb_reach_ratio(hand_landmarks):
    """How far the thumb tip reaches from the wrist, relative to how far
    the four fingertips reach from the wrist.

    This is deliberately a ratio of two distances measured in the *same*
    projection: when the hand is angled so the palm is foreshortened
    (e.g. knuckles pointed at the camera), both the numerator and the
    denominator shrink by roughly the same factor, so the ratio stays
    stable. Normalizing against a fixed hand dimension instead - like
    wrist-to-middle-knuckle - breaks in exactly that pose, because that
    reference length foreshortens while the thumb's apparent position
    doesn't shrink proportionally, inflating the result.

    In a fist the thumb sits among the folded fingers, so it reaches
    about as far as they do (ratio near 1). In a thumbs up/down it
    extends well past them (ratio comfortably above 1)."""
    wrist = _xyz(hand_landmarks, 0)
    thumb_reach = _dist2(_xyz(hand_landmarks, 4), wrist)
    finger_reaches = [_dist2(_xyz(hand_landmarks, i), wrist) for i in _FINGER_TIPS.values()]
    mean_finger_reach = sum(finger_reaches) / len(finger_reaches)
    return thumb_reach / max(1e-6, mean_finger_reach)


def _is_thumb_extended(hand_landmarks):
    """A thumb reads as 'extended' only if it's BOTH straight AND reaching
    clearly past the folded fingers. Straightness alone isn't enough: in a
    common fist the thumb lies flat across the curled fingers rather than
    curling back on itself, so its own joints stay nearly straight even
    though the hand is unmistakably a fist."""
    straight = _thumb_straightness(hand_landmarks) > config.THUMB_STRAIGHT_ANGLE_DEG
    reaching = _thumb_reach_ratio(hand_landmarks) > config.THUMB_REACH_RATIO
    return straight and reaching


def _thumb_direction_verticality(hand_landmarks):
    """How vertical the thumb's own axis (MCP -> tip) is, as a signed
    ratio: +1 = pointing straight up the screen, -1 = straight down, ~0 =
    sideways. Signed so the caller can use one number for both the
    'is it vertical enough to be a thumbs up/down at all' test and the
    up-vs-down decision."""
    mcp = _xyz(hand_landmarks, 2)
    tip = _xyz(hand_landmarks, 4)
    dx, dy = tip[0] - mcp[0], tip[1] - mcp[1]
    magnitude = math.hypot(dx, dy)
    if magnitude < 1e-9:
        return 0.0
    # Screen y grows downward, so negate to make "up" positive.
    return -dy / magnitude


def _is_finger_curled(hand_landmarks, finger):
    """A finger counts as CURLED only when it's clearly folded down into
    the palm - a much stricter test than "not extended". Between the two
    (a half-open hand, fingers loosely bent) a finger is neither, and
    that in-between state must NOT be treated as part of a fist.

    Two independent checks, both required:
      - the PIP joint angle is sharply bent (a folded finger, not a
        relaxed curve), and
      - the fingertip has actually come back in close to the wrist
        relative to its own knuckle (a folded finger tucks in; a loosely
        bent one still reaches well past its knuckle).
    """
    mcp = _xyz(hand_landmarks, _FINGER_MCPS[finger])
    pip = _xyz(hand_landmarks, _FINGER_PIPS[finger])
    tip = _xyz(hand_landmarks, _FINGER_TIPS[finger])
    wrist = _xyz(hand_landmarks, 0)
    bent = _angle_deg(mcp, pip, tip) < config.FINGER_CURLED_ANGLE_DEG
    tucked = (
        _dist2(tip, wrist) / max(1e-6, _dist2(mcp, wrist))
    ) < config.FINGER_CURLED_REACH_RATIO
    return bent and tucked


def has_open_fingers(hand_landmarks):
    """True if at least one of middle/ring/pinky is clearly extended.

    A deliberate pinch (thumb + index, like an OK sign) usually leaves
    the other fingers up or relaxed-open. A fist, or a half-closed hand
    that merely brings thumb and index together as it curls, has all
    three folded. Presentation mode uses this to tell a real pinch-click
    from a closing hand."""
    return any(_is_finger_extended(hand_landmarks, f) for f in ("middle", "ring", "pinky"))


def is_strict_fist(hand_landmarks):
    """True ONLY for a genuinely closed fist: all four fingers clearly
    folded AND the thumb not sticking out. A half-open / half-closed
    hand returns False (it is deliberately not a fist - reading it as
    one was firing fist actions - laser, next-track, pinch suppression -
    on hands that were only partly closed)."""
    return (
        all(_is_finger_curled(hand_landmarks, f) for f in _FINGER_TIPS)
        and not _is_thumb_extended(hand_landmarks)
    )


def classify_static_gesture(hand_landmarks):
    """Classifies one frame's landmarks into a static gesture name, or
    None if it doesn't clearly match any of them (e.g. a hand mid-transition
    between two shapes - better to report nothing than guess wrong)."""
    extended = {f: _is_finger_extended(hand_landmarks, f) for f in _FINGER_TIPS}
    thumb_extended = _is_thumb_extended(hand_landmarks)
    num_extended = sum(extended.values())

    if num_extended >= 4:
        return "open_palm"

    if num_extended == 0:
        # No finger is straight - but that alone is NOT a fist. A half-
        # open hand has no straight fingers either. Require every finger
        # to be genuinely folded; otherwise the hand is in-between and we
        # report nothing rather than guessing "fist".
        if not all(_is_finger_curled(hand_landmarks, f) for f in _FINGER_TIPS):
            return None
        if not thumb_extended:
            return "fist"
        # Thumb is out, but a thumbs up/down must also actually *point*
        # up or down. A thumb sticking out sideways (a relaxed or
        # half-open fist) is neither - report None rather than guessing,
        # so the stabilizer never confirms it.
        verticality = _thumb_direction_verticality(hand_landmarks)
        if verticality > config.THUMB_VERTICAL_RATIO:
            return "thumb_up"
        if verticality < -config.THUMB_VERTICAL_RATIO:
            return "thumb_down"
        return None

    if extended["index"] and not extended["middle"] and not extended["ring"] and not extended["pinky"]:
        return "pointing"
    if extended["index"] and extended["middle"] and not extended["ring"] and not extended["pinky"]:
        return "two_finger"
    # Three fingers (index + middle + ring) — used for right-click.
    if (extended["index"] and extended["middle"] and extended["ring"]
            and not extended["pinky"]):
        return "three_finger"
    return None


def describe_hand(hand_landmarks):
    """Diagnostic snapshot of the raw measurements behind a classification.
    Used by gesture_demo.py's debug overlay so thresholds can be tuned
    against real hands instead of guessed at."""
    return {
        "thumb_straightness": _thumb_straightness(hand_landmarks),
        "thumb_reach": _thumb_reach_ratio(hand_landmarks),
        "thumb_verticality": _thumb_direction_verticality(hand_landmarks),
        "fingers_extended": sum(_is_finger_extended(hand_landmarks, f) for f in _FINGER_TIPS),
    }


class GestureStabilizer:
    """Per-hand stability filter sitting on top of classify_static_gesture().

    - Multi-frame confirmation: a candidate gesture must win N consecutive
      frames before it's reported.
    - Cooldown: once a gesture fires, it won't fire again until the
      cooldown elapses, even while still being detected every frame.
    - Swipe detection uses palm-center velocity over a short window,
      computed from already One-Euro-smoothed points (see gestures.py),
      so raw per-frame jitter can't masquerade as a swipe.

    Give each tracked hand its own instance - sharing one across hands
    would blend their gesture histories the same way a shared position
    smoother blended their coordinates.
    """

    def __init__(self, confirm_frames=None, cooldown_seconds=None, check_swipe=True):
        self.confirm_frames = confirm_frames or config.GESTURE_CONFIRM_FRAMES
        self.cooldown_seconds = cooldown_seconds or config.GESTURE_COOLDOWN_SECONDS
        # Swipe detection runs BEFORE static-gesture confirmation every
        # frame (see update() below) and, being purely motion-based,
        # doesn't care what shape the hand is making while it moves -
        # so any real hand-speed motion while forming a static pose
        # (e.g. raising a fist, or just naturally drifting while
        # pointing) can "eat" that frame's confirmation slot with a
        # false swipe read, delaying or blocking the static gesture from
        # ever reaching its required consecutive-frame count. Pass
        # check_swipe=False for a stabilizer whose modes don't use swipe
        # at all, so it can't interfere with gestures that DO matter
        # there (this is exactly why Presentation/Media's fist and
        # pointing gestures were unreliable - see ai_gesture_engine.py).
        self.check_swipe = check_swipe
        self._history = deque(maxlen=self.confirm_frames)
        self._palm_history = deque(maxlen=config.SWIPE_HISTORY_FRAMES)
        self._last_fired_gesture = None
        self._last_fired_time = float("-inf")
        # Live palm velocity (normalized frame-widths/second, signed:
        # positive = moving right), updated every frame regardless of
        # whether a swipe actually fires - lets a HUD show the user how
        # fast they're moving relative to config.SWIPE_VELOCITY_THRESHOLD
        # in real time, not just after the fact.
        self.last_swipe_velocity = 0.0
        self.last_swipe_velocity_y = 0.0  # signed, +down (screen coords)

    def update(self, candidate_gesture, palm_point, now):
        """Feed one frame's raw static classification plus the hand's
        smoothed palm-center point. Returns a confirmed gesture name
        (fired at most once per cooldown window) or None."""
        self._history.append(candidate_gesture)
        self._palm_history.append((now, palm_point))

        if self.check_swipe:
            swipe = self._check_swipe(candidate_gesture)
            if swipe:
                return self._maybe_fire(swipe, now)

        if (len(self._history) == self._history.maxlen
                and candidate_gesture is not None
                and len(set(self._history)) == 1):
            return self._maybe_fire(candidate_gesture, now)

        return None

    def _maybe_fire(self, gesture, now):
        cooldown = self.cooldown_seconds
        if gesture in ("swipe_up", "swipe_down"):
            cooldown = min(cooldown, config.SCROLL_SWIPE_COOLDOWN_SECONDS)
        same_gesture_in_cooldown = (
            gesture == self._last_fired_gesture
            and (now - self._last_fired_time) < cooldown
        )
        if same_gesture_in_cooldown:
            return None
        self._last_fired_gesture = gesture
        self._last_fired_time = now
        return gesture

    def _check_swipe(self, candidate_gesture=None):
        if len(self._palm_history) < self._palm_history.maxlen:
            return None
        t0, p0 = self._palm_history[0]
        t1, p1 = self._palm_history[-1]
        dt = max(1e-3, t1 - t0)
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        velocity = dx / dt
        velocity_y = dy / dt
        self.last_swipe_velocity = velocity
        self.last_swipe_velocity_y = velocity_y
        # Horizontal: reject only when the motion is CLEARLY more
        # vertical than horizontal (1.2x margin) - real horizontal
        # swipes always have some vertical drift.
        if abs(dy) <= abs(dx) * 1.2:
            if velocity > config.SWIPE_VELOCITY_THRESHOLD:
                return "swipe_right"
            if velocity < -config.SWIPE_VELOCITY_THRESHOLD:
                return "swipe_left"
            return None
        # Vertical: mostly-vertical motion (same 1.2x margin the other
        # way). Only while the hand is open / in motion - never while
        # pointing, pinching or making any other named shape, so
        # steering the cursor up/down or dragging can't scroll.
        if abs(dx) * 1.2 < abs(dy) and candidate_gesture in (None, "open_palm"):
            if velocity_y < -config.SWIPE_VERTICAL_VELOCITY_THRESHOLD:
                return "swipe_up"
            if velocity_y > config.SWIPE_VERTICAL_VELOCITY_THRESHOLD:
                return "swipe_down"
        return None

    def reset(self):
        self._history.clear()
        self._palm_history.clear()
        self.last_swipe_velocity = 0.0
        self.last_swipe_velocity_y = 0.0


class EdgeDwellDetector:
    """Alternative to velocity-based swipe for "previous/next": fires
    when a pointing fingertip is held inside the left or right edge zone
    of the frame for config.EDGE_DWELL_SECONDS, instead of needing a
    fast flick timed just right.

    Swipe detection asks the user to move their hand across (most of)
    the frame within a short, fixed window of frames - easy to miss if
    you swipe a bit slow, or if the camera's framerate dips. This is the
    opposite tradeoff: move the fingertip to the edge and hold it there
    for a moment. Slower, but far more repeatable, and it's a
    deliberate "I am now asking for next/previous" motion rather than a
    fast gesture that can be misread mid-flight as something else.

    Use alongside swipe detection (not instead of it) - offering both
    means whichever motion feels more natural to the user works.
    """

    def __init__(self, edge_ratio=None, dwell_seconds=None, cooldown_seconds=None):
        self.edge_ratio = config.EDGE_DWELL_ZONE_RATIO if edge_ratio is None else edge_ratio
        self.dwell_seconds = config.EDGE_DWELL_SECONDS if dwell_seconds is None else dwell_seconds
        self.cooldown_seconds = (
            config.GESTURE_COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds
        )
        self._zone = None
        self._zone_since = None
        self._last_fired_zone = None
        self._last_fired_time = float("-inf")

    def update(self, index_x_norm, now):
        """index_x_norm: the pointing fingertip's normalized (0-1) x
        position, or None if no hand/fingertip is currently tracked.
        Returns 'swipe_left', 'swipe_right', or None - named to match
        GestureStabilizer's swipe output so callers can treat them
        interchangeably."""
        if index_x_norm is None:
            self._zone = None
            self._zone_since = None
            return None

        if index_x_norm < self.edge_ratio:
            zone = "left"
        elif index_x_norm > 1 - self.edge_ratio:
            zone = "right"
        else:
            zone = None

        if zone != self._zone:
            # Just entered a new zone (or left one) - start the dwell
            # clock fresh rather than firing on however long the OLD
            # zone had already been held.
            self._zone = zone
            self._zone_since = now if zone else None
            return None

        if zone is None or self._zone_since is None:
            return None

        if now - self._zone_since < self.dwell_seconds:
            return None

        same_zone_in_cooldown = (
            zone == self._last_fired_zone and (now - self._last_fired_time) < self.cooldown_seconds
        )
        if same_zone_in_cooldown:
            return None

        self._last_fired_zone = zone
        self._last_fired_time = now
        return "swipe_left" if zone == "left" else "swipe_right"

    def reset(self):
        self._zone = None
        self._zone_since = None

    def progress(self, now):
        """Read-only HUD helper - doesn't affect firing. Returns
        (zone, fraction): zone is 'left'/'right'/None for whichever edge
        zone the fingertip is currently inside, fraction is 0-1 for how
        far through the dwell period it is (so a HUD can draw a fill-up
        bar showing exactly when the hold will fire)."""
        if self._zone is None or self._zone_since is None:
            return None, 0.0
        fraction = min(1.0, (now - self._zone_since) / self.dwell_seconds)
        return self._zone, fraction


class HoldTrigger:
    """Fires once after a specific static gesture has been read for
    `hold_seconds` straight - a deliberate "hold this shape to confirm"
    trigger, distinct from GestureStabilizer's brief multi-frame confirm.

    Used for the gesture-driven mode menu (Milestone 3 UI rework): a
    quick, momentary open-palm shouldn't open a menu (that's what
    GestureStabilizer's fast confirm is for elsewhere), but a clearly
    sustained one is an unambiguous "I mean this" signal. Re-arms only
    once the gesture clears, so holding past the threshold can't fire
    twice in a row.
    """

    def __init__(self, gesture_name, hold_seconds):
        self.gesture_name = gesture_name
        self.hold_seconds = hold_seconds
        self._since = None
        self._armed = True

    def update(self, current_gesture, now):
        """Feed the current frame's raw (not confirmed) static gesture.
        Returns True on the single frame the hold threshold is crossed."""
        if current_gesture != self.gesture_name:
            self._since = None
            self._armed = True
            return False
        if self._since is None:
            self._since = now
        if self._armed and (now - self._since) >= self.hold_seconds:
            self._armed = False
            return True
        return False

    def reset(self):
        self._since = None
        self._armed = True


class RotationDialDetector:
    """Turns a held pinch into a volume-knob-style dial: while pinching,
    tracks the angle of the pinch point (thumb+index midpoint) around
    the wrist, frame to frame, and fires "clockwise"/"counter_clockwise"
    each time the accumulated rotation crosses degrees_per_tick - the
    same way a real knob (or a mouse scroll wheel) produces discrete
    ticks rather than a continuous value.

    Deliberately NOT routed through GestureStabilizer/classify_static_
    gesture - it needs a continuous angle, not a discrete pose, and
    checking it independently means it can't be disrupted by (or
    disrupt) swipe/static-pose confirmation the way stacking yet another
    behavior onto the shared stabilizer would.
    """

    # Below this radius (in the same units as center_point/ref_point - we
    # feed it pixels), the pinch point sits too close to the wrist for its
    # angle around the wrist to mean much: a tiny, meaningless wobble in
    # either point turns into a huge swing in atan2's angle at short
    # range, the classic "camera shake looks huge in a macro shot" effect.
    # That was firing stray/backwards ticks right as someone started the
    # gesture, before their hand had rotated out to a stable turning
    # radius yet - reads as "erratic" and undermines trust in the whole
    # gesture. Below this radius we simply wait rather than guess.
    MIN_RADIUS_PX = 25.0

    def __init__(self, degrees_per_tick=None, cooldown_seconds=None):
        self.degrees_per_tick = (
            config.DIAL_DEGREES_PER_TICK if degrees_per_tick is None else degrees_per_tick
        )
        self.cooldown_seconds = (
            config.DIAL_TICK_COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds
        )
        self._active = False
        self._last_angle = None
        self._accumulated = 0.0
        self._last_tick_time = float("-inf")

    def update(self, active, center_point, ref_point, now):
        """active: True while the "turn the knob" pose (a held pinch) is
        current. center_point: the point rotation is measured AROUND
        (the wrist works well - it barely moves while the fingers do
        the turning). ref_point: the point being tracked (the pinch
        location - thumb+index midpoint). Both points in the same
        (consistent) coordinate space; units don't matter, only angle
        does - except MIN_RADIUS_PX, which assumes pixels. Returns
        "clockwise", "counter_clockwise", or None."""
        if not active or center_point is None or ref_point is None:
            self._active = False
            self._last_angle = None
            self._accumulated = 0.0
            return None

        radius = math.hypot(ref_point[0] - center_point[0], ref_point[1] - center_point[1])
        if radius < self.MIN_RADIUS_PX:
            # Too close to the pivot to trust - hold everything exactly
            # where it is (don't reset _accumulated/_active) rather than
            # recording a shaky angle, so a brief close pass through the
            # dead zone mid-turn doesn't cost any accumulated progress.
            return None

        angle = math.degrees(math.atan2(
            ref_point[1] - center_point[1], ref_point[0] - center_point[0]
        ))

        if not self._active or self._last_angle is None:
            # Just started this pinch - record the starting angle rather
            # than comparing against a stale one from before the pinch
            # began (or from mode-switches / frames ago).
            self._active = True
            self._last_angle = angle
            self._accumulated = 0.0
            return None

        delta = angle - self._last_angle
        # Normalize into [-180, 180] so crossing the atan2 wrap-around
        # point (e.g. 179deg -> -179deg in one frame) doesn't register
        # as a huge spurious jump in the wrong direction.
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360
        self._last_angle = angle
        self._accumulated += delta

        result = None
        if (now - self._last_tick_time) >= self.cooldown_seconds:
            if self._accumulated >= self.degrees_per_tick:
                result = "clockwise"
            elif self._accumulated <= -self.degrees_per_tick:
                result = "counter_clockwise"

        if result:
            # Subtract exactly one tick's worth instead of zeroing out -
            # a fast turn that covers, say, 40deg in one frame (two full
            # ticks' worth) keeps its 4deg of overshoot toward the NEXT
            # tick instead of throwing it away. Discarding the overshoot
            # was the main reason fast turns used to feel like they were
            # lagging behind the hand: real rotation kept being dropped
            # on the floor every single tick.
            self._accumulated -= self.degrees_per_tick if result == "clockwise" else -self.degrees_per_tick
            self._last_tick_time = now

        return result

    def progress(self):
        """(accumulated_degrees, degrees_per_tick, active) - lets the
        caller draw a live progress ring for the current turn instead of
        only flashing feedback after a tick has already fired. Knowing
        how far you are into a turn *before* it fires is what actually
        makes a physical knob feel predictable to use."""
        return self._accumulated, self.degrees_per_tick, self._active

    def reset(self):
        self._active = False
        self._last_angle = None
        self._accumulated = 0.0
