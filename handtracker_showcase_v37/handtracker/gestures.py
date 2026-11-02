"""
gestures.py
-----------
Gesture-detection helpers, pulled out of the main loop so features don't
each reimplement their own pinch/tap math. This is the seed of the
Milestone 2 "dedicated gesture engine" - for now it covers exactly what
the original prototype used (pinch + point-in-box tap); we'll grow this
module with the palm/fist/swipe/thumb gestures as we work through
Milestone 2.
"""
import math
import time

import config


def get_landmark_px(hand_landmarks, landmark_id, frame_width, frame_height):
    """Convert a MediaPipe landmark to pixel coordinates."""
    lm = hand_landmarks.landmark[landmark_id]
    return int(lm.x * frame_width), int(lm.y * frame_height)


def is_pinching_from_points(index_point, thumb_point):
    """True if the (already-smoothed) index and thumb tips are closer
    than the fixed pixel pinch threshold. Prefer is_pinching_scaled()
    for desktop control so detection stays consistent at different
    camera distances."""
    distance = math.hypot(index_point[0] - thumb_point[0], index_point[1] - thumb_point[1])
    return distance < config.PINCH_DISTANCE_THRESHOLD_PX


def hand_scale_px(hand_landmarks, frame_width, frame_height):
    """Wrist-to-middle-MCP distance in pixels. A stable proxy for the
    hand's apparent size in frame; used as the divisor for scaled pinch
    so the same physical finger gap triggers regardless of how close
    the hand is to the camera."""
    wrist = hand_landmarks.landmark[0]
    mid_mcp = hand_landmarks.landmark[9]
    return math.hypot(
        (wrist.x - mid_mcp.x) * frame_width,
        (wrist.y - mid_mcp.y) * frame_height,
    )


def pinch_ratio(index_point, thumb_point, hand_scale):
    """Pinch distance as a fraction of the hand's own size (hand_scale -
    e.g. the wrist-to-middle-knuckle distance, in the same coordinate
    space as the two points), rather than a fixed pixel count. A fixed
    pixel threshold drifts with how far the hand is from the camera: the
    same physical finger gap covers more pixels when the hand is close
    and fewer when it's far away, so a threshold tuned for one distance
    can misfire at another. Dividing by the hand's own current size
    cancels that out, the same way the gesture engine's thumb-reach
    ratio does for camera angle.

    Always pair with a smoothed hand_scale (ScalarSmoother): dividing by
    a noisy live scale amplifies landmark jitter into false clicks,
    especially during fist/self-occlusion."""
    distance = math.hypot(index_point[0] - thumb_point[0], index_point[1] - thumb_point[1])
    return distance / max(1e-6, hand_scale)


def is_pinching_scaled(index_point, thumb_point, hand_scale, ratio_threshold=None):
    ratio_threshold = config.PINCH_RATIO_THRESHOLD if ratio_threshold is None else ratio_threshold
    return pinch_ratio(index_point, thumb_point, hand_scale) < ratio_threshold


def is_pinching_from_ratio(ratio, was_pinching, engage_threshold=None, release_threshold=None):
    """Hysteresis pinch test on an ALREADY-COMPUTED ratio (pinch_ratio()).

    A single fixed threshold flickers on/off whenever the ratio sits
    right near it - which, for a human hand, is most of the time a
    "pinch" is actually held, since nobody holds thumb and index
    perfectly still. This uses a TIGHTER threshold to start a pinch and
    a LOOSER one to end it: once you're in, the ratio has to open up
    noticeably further before it's considered released, so it can't
    chatter across one boundary every frame. Pass in the previous
    frame's pinch state - which threshold applies depends on it.

    Pair with a lightly-smoothed ratio (ScalarSmoother, small alpha) for
    the best result - hysteresis alone still leaves single-frame noise
    spikes able to flip the state right at the moment a real transition
    is also happening; smoothing the ratio first removes most of that
    noise before hysteresis even has to deal with it.
    """
    engage_threshold = config.PINCH_RATIO_ENGAGE if engage_threshold is None else engage_threshold
    release_threshold = config.PINCH_RATIO_RELEASE if release_threshold is None else release_threshold
    threshold = release_threshold if was_pinching else engage_threshold
    return ratio < threshold


class ScalarSmoother:
    """A plain, heavily-damped exponential moving average for a single
    scalar value - deliberately simpler and slower to react than
    PointSmoother/OneEuroFilter above.

    This exists specifically for values used as a DIVISOR, like the
    hand-size reference in pinch_ratio(): dividing by a noisy value
    amplifies that noise rather than averaging it out, and a hand's own
    size in frame only changes as fast as the user moves toward or away
    from the camera - much slower than fingertip motion - so there's no
    responsiveness reason not to smooth it hard. MediaPipe's landmarks
    are also especially prone to jitter during self-occlusion (a closing
    fist, or fingers passing close together), which is exactly when
    pinch detection matters most - an unsmoothed scale reference made
    pinch detection actively worse than a fixed threshold in exactly
    those moments.
    """

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._value = None

    def update(self, raw_value):
        self._value = raw_value if self._value is None else (
            self.alpha * raw_value + (1 - self.alpha) * self._value
        )
        return self._value

    def reset(self):
        self._value = None


class DebouncedState:
    """Only reports a change in a boolean (or any comparable) value once
    the new value has held steady for `confirm_frames` consecutive
    updates, in EITHER direction.

    This is different from gesture_engine.GestureStabilizer, which
    confirms a NEW momentary gesture before reporting it at all.
    DebouncedState is for not *losing* a state the caller is already in
    - e.g. a held fist used for dragging - when a single noisy frame
    misreads it as something else. Without this, one bad frame (very
    plausible while the hand is actively moving during a drag) was
    enough to end the drag instantly.
    """

    def __init__(self, confirm_frames=3, initial=False):
        self.confirm_frames = max(1, confirm_frames)
        self._current = initial
        self._pending = initial
        self._pending_count = 0

    def update(self, raw_value):
        if raw_value == self._pending:
            self._pending_count += 1
        else:
            self._pending = raw_value
            self._pending_count = 1
        if self._pending_count >= self.confirm_frames and self._current != self._pending:
            self._current = self._pending
        return self._current

    def reset(self, value=False):
        self._current = value
        self._pending = value
        self._pending_count = 0


class _LowPassFilter:
    """Single-pole exponential low-pass filter on one scalar value."""

    def __init__(self):
        self._filtered = None

    def apply(self, value, alpha):
        self._filtered = value if self._filtered is None else (
            alpha * value + (1 - alpha) * self._filtered
        )
        return self._filtered

    @property
    def last_filtered(self):
        return self._filtered

    def reset(self):
        self._filtered = None


class OneEuroFilter1D:
    """One Euro Filter (Casiez, Roussel & Vogel, 2012) for a single
    scalar value. Two properties make this a better fit than a fixed
    exponential average for hand-tracking jitter:

    - It's driven by real elapsed time (`timestamp`), not frame count,
      so smoothing behaves the same whether the camera loop runs at
      15fps or 60fps - a fixed-alpha filter effectively smooths *more*
      per second of real time as framerate drops, which is why the
      jitter felt inconsistent.
    - It adapts to how fast the value is changing: near-stationary
      values get filtered heavily (kills jitter), fast movements get
      filtered lightly (kills lag). `beta` controls how strongly speed
      relaxes the filtering; `min_cutoff` sets the baseline smoothing
      floor for a still hand.
    """

    def __init__(self, min_cutoff=1.2, beta=0.4, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._value_filter = _LowPassFilter()
        self._deriv_filter = _LowPassFilter()
        self._last_time = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, value, timestamp):
        dt = 1.0 / 30.0 if self._last_time is None else max(1e-3, timestamp - self._last_time)
        self._last_time = timestamp

        prev_value = self._value_filter.last_filtered
        derivative = 0.0 if prev_value is None else (value - prev_value) / dt
        smoothed_derivative = self._deriv_filter.apply(derivative, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(smoothed_derivative)
        return self._value_filter.apply(value, self._alpha(cutoff, dt))

    def reset(self):
        self._value_filter.reset()
        self._deriv_filter.reset()
        self._last_time = None


class PointSmoother:
    """Applies a One Euro Filter independently to the x and y axes of a
    2D point stream (e.g. one hand's fingertip position). Each tracked
    hand needs its own PointSmoother instance - sharing one between two
    hands is what was causing their positions to blend together."""

    def __init__(self, min_cutoff=config.SMOOTHING_MIN_CUTOFF, beta=config.SMOOTHING_BETA):
        self._x_filter = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)
        self._y_filter = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)

    def update(self, point, timestamp=None):
        """Returns the smoothed point as floats - deliberately NOT cast
        to int here, since this class is used for both pixel coordinates
        (main.py) and normalized 0-1 coordinates (main_desktop.py's
        cursor mapping, gesture_demo.py's swipe velocity). Casting to int
        here previously truncated every normalized coordinate to exactly
        (0, 0), which is why the desktop-control cursor was stuck in the
        corner. Callers that need integer pixels (cv2 drawing calls)
        should cast at the point of use."""
        timestamp = time.time() if timestamp is None else timestamp
        x = self._x_filter.apply(point[0], timestamp)
        y = self._y_filter.apply(point[1], timestamp)
        return (x, y)

    def reset(self):
        """Clears the filter's internal state, so the next update()
        after a hand reappears starts fresh instead of smoothing across
        a stale position from before it was lost (which would otherwise
        show up as a brief snap/drag toward the old point)."""
        self._x_filter.reset()
        self._y_filter.reset()


def is_point_in_box(point_x, point_y, box_x, box_y, box_w, box_h):
    """True if (point_x, point_y) falls inside the given box."""
    return box_x <= point_x <= box_x + box_w and box_y <= point_y <= box_y + box_h


def is_near_point(x, y, target_x, target_y, threshold=50):
    """True if (x, y) is within `threshold` px of (target_x, target_y)."""
    return math.hypot(x - target_x, y - target_y) < threshold


class TapDebouncer:
    """Prevents a single tap from being registered as many repeated clicks.

    This replaces the old global `last_tap_time` variable with something
    each feature can own an instance of, so features don't fight over
    shared debounce state.
    """

    def __init__(self, delay_seconds=config.TAP_DELAY_SECONDS):
        self.delay_seconds = delay_seconds
        self._last_tap_time = 0.0

    def ready(self, now):
        if now - self._last_tap_time > self.delay_seconds:
            self._last_tap_time = now
            return True
        return False
