"""
tutorial.py
------------
A short, interactive walkthrough that only runs when you ask for it - press
T, or pinch "Tutorial" in the help panel. It NEVER starts on its own, not
even on the first launch.

Each step waits for you to actually do the thing it's teaching (move the
ring, pinch, find the help tab, open the mode menu), tells you "Nice!", and
moves on - or you can pinch Next / Back / Skip at any point. Steps that only
explain something (control switch, video calls) just wait for Next.

Tk-free on purpose: it takes a per-frame context dict and returns plain
data (view()) that main_overlay draws, so the logic is unit-tested without a
display. Buttons are laid out in screen pixels and hit-tested against the
same pinch-stable pointer used everywhere else.
"""
import math

STEPS = [
    {"id": "point", "need": "move", "title": "Point",
     "body": "Hold your hand up and move your index finger. The ring on screen follows it."},
    {"id": "pinch", "need": "click", "title": "Pinch to click",
     "body": "Touch your thumb and index finger together, then let go. That is a click."},
    {"id": "help", "need": "help", "title": "Find the help",
     "body": "Point the ring at the  ?  Help  tab in the bottom-left corner to see what your hands can do in the current mode."},
    {"id": "menu", "need": "menu", "title": "Switch modes",
     "body": "Hold an open palm up for about a second to open the mode menu. Point at a mode and pinch to pick it."},
    {"id": "control", "need": None, "title": "Control on / off",
     "body": "Thumb down switches hand control on or off. It is your safety switch: nothing moves your real mouse while it is off."},
    {"id": "calls", "need": None, "title": "Using it in video calls",
     "body": "Press V for a virtual camera (needs OBS), or K to release the camera for other apps. C = Call mode: hold a still fist 1 s = mute, peace sign = camera, open palm = raise hand."},
]

MOVE_DEADZONE_PX = 14        # per-frame jitter is a few px and must never add up to 'movement'
MOVE_GOAL_PX = 700          # how far the ring has to travel to count as "I moved it"
DONE_PAUSE_S = 0.9          # how long "Nice!" shows before auto-advancing


class Tutorial:
    def __init__(self, screen_w=1920, screen_h=1080):
        self.sw, self.sh = int(screen_w), int(screen_h)
        self.active = False
        self.finished = False        # True once the last step was completed (not skipped)
        self.idx = 0
        self._done_at = None
        self._moved = 0.0
        self._last_pt = None

    # -- control -------------------------------------------------------------
    def start(self):
        self.active, self.finished, self.idx = True, False, 0
        self._reset_step()

    def stop(self):
        self.active = False

    def toggle(self):
        self.stop() if self.active else self.start()

    def next(self):
        if self.idx >= len(STEPS) - 1:
            self.finished = True
            self.active = False
        else:
            self.idx += 1
            self._reset_step()

    def back(self):
        if self.idx > 0:
            self.idx -= 1
            self._reset_step()

    def _reset_step(self):
        self._done_at, self._moved, self._last_pt = None, 0.0, None

    # -- layout -----------------------------------------------------------------
    def card_rect(self):
        w, h = min(640, self.sw - 80), 196
        x = (self.sw - w) // 2
        return (x, 44, x + w, 44 + h)

    def buttons(self):
        x0, y0, x1, y1 = self.card_rect()
        bw, bh, gap = 92, 34, 10
        y = y1 - bh - 14
        last = self.idx >= len(STEPS) - 1
        out = [{"id": "next", "label": "Done" if last else "Next", "rect": (x1 - 16 - bw, y, x1 - 16, y + bh)},
               {"id": "skip", "label": "Close", "rect": (x1 - 16 - 2 * bw - gap, y, x1 - 16 - bw - gap, y + bh)}]
        if self.idx > 0:
            out.append({"id": "back", "label": "Back", "rect": (x0 + 16, y, x0 + 16 + bw, y + bh)})
        return out

    def button_at(self, pt):
        if pt is None:
            return None
        for b in self.buttons():
            x0, y0, x1, y1 = b["rect"]
            if x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1:
                return b["id"]
        return None

    # -- per frame ------------------------------------------------------------------
    def update(self, ctx, now):
        """ctx: pointer (screen pt or None), click (a pinch just closed),
        help_open (help drawer expanded), menu_open (mode menu showing).
        Returns the button id pressed this frame (or None)."""
        if not self.active:
            return None
        pt, click = ctx.get("pointer"), bool(ctx.get("click"))
        pressed = None
        if click:
            pressed = self.button_at(pt)
            if pressed == "next":
                self.next()
            elif pressed == "skip":
                self.stop()
            elif pressed == "back":
                self.back()
            if pressed:
                return pressed

        step = STEPS[self.idx]
        if self._done_at is not None:
            if now - self._done_at >= DONE_PAUSE_S:
                self.next()
            return None

        need = step["need"]
        if need == "move":
            if pt is not None:
                if self._last_pt is None:
                    self._last_pt = pt
                else:
                    d = math.hypot(pt[0] - self._last_pt[0], pt[1] - self._last_pt[1])
                    if d > 400:                  # a tracking glitch, not a movement: re-anchor, don't count
                        self._last_pt = pt
                    elif d >= MOVE_DEADZONE_PX:  # real movement counts; sensor jitter never reaches this
                        self._moved += d
                        self._last_pt = pt
            ok = self._moved >= MOVE_GOAL_PX
        elif need == "click":
            ok = click and pressed is None
        elif need == "help":
            ok = bool(ctx.get("help_open"))
        elif need == "menu":
            ok = bool(ctx.get("menu_open"))
        else:
            ok = False
        if ok:
            self._done_at = now
        return None

    # -- plain data for the renderer ---------------------------------------------------
    def view(self):
        if not self.active:
            return None
        step = STEPS[self.idx]
        progress = min(1.0, self._moved / MOVE_GOAL_PX) if step["need"] == "move" else None
        return {"title": step["title"], "body": step["body"], "idx": self.idx, "total": len(STEPS),
                "done": self._done_at is not None, "rect": self.card_rect(),
                "buttons": self.buttons(), "progress": progress,
                "waiting": step["need"] is not None}
