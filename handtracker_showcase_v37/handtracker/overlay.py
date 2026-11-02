"""
overlay.py
-----------
Milestone 3.1: a small always-on-top window that stays visible over
other applications, showing whether desktop control is on, the current
gesture, and whether a drag is in progress.

This is deliberately a plain, opaque, movable window rather than a
transparent/click-through HUD - a solid first version that's simple
enough to trust without a real display to test it on. Click-through
transparency is a natural next step (Windows supports it via the
Tkinter '-transparentcolor' window attribute) once this basic version
has been tried out on your machine.

Call update() once per camera frame rather than running Tkinter on its
own thread - mixing Tkinter's mainloop with an OpenCV imshow loop in
separate threads is a common source of flaky, hard-to-debug window
behavior; pumping Tkinter's event loop manually alongside the camera
loop avoids that entirely.
"""
import tkinter as tk


class Overlay:
    def __init__(self, x=40, y=40):
        self.root = tk.Tk()
        self.root.title("Gesture Control")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)  # no OS title bar - drag via the label itself
        self.root.geometry(f"260x100+{x}+{y}")
        self.root.configure(bg="#202020")

        self.label = tk.Label(
            self.root, justify="left", anchor="w", fg="#39ff88", bg="#202020",
            font=("Consolas", 11), padx=10, pady=8,
        )
        self.label.pack(fill="both", expand=True)

        # overrideredirect() removes the OS's own title bar, and with it
        # the OS's own click-and-drag-to-move handling, so it's redone
        # by hand here.
        self._drag_origin = {"x": 0, "y": 0}
        self.label.bind("<ButtonPress-1>", self._start_drag)
        self.label.bind("<B1-Motion>", self._do_drag)

        self._closed = False

    def _start_drag(self, event):
        self._drag_origin["x"], self._drag_origin["y"] = event.x, event.y

    def _do_drag(self, event):
        x = self.root.winfo_pointerx() - self._drag_origin["x"]
        y = self.root.winfo_pointery() - self._drag_origin["y"]
        self.root.geometry(f"+{x}+{y}")

    def update(self, enabled, gesture, dragging, fps):
        """Refresh the HUD text and pump Tkinter's event queue. Call
        this once per camera frame."""
        if self._closed:
            return
        status = "ON" if enabled else "OFF"
        drag_note = "  (dragging)" if dragging else ""
        self.label.config(text=(
            f"Desktop control: {status}\n"
            f"Gesture: {gesture}{drag_note}\n"
            f"FPS: {fps:.0f}"
        ))
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            # The window was closed some other way - stop touching it.
            self._closed = True

    def close(self):
        if not self._closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._closed = True


class CursorReticle:
    """A small always-on-top ring that follows the mapped screen
    position of the tracked fingertip, so it's obvious where a pinch
    will click before it happens - the standard OS arrow cursor is easy
    to lose track of while you're watching your hand, not the screen.

    Uses Tkinter's Windows-only '-transparentcolor' attribute so the
    ring's background is both invisible and click-through, meaning it
    never blocks clicks on whatever is underneath it. On a platform
    where that attribute isn't supported, the window falls back to a
    small opaque square instead of failing outright.
    """

    SIZE = 34

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gesture Cursor")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.SIZE}x{self.SIZE}+0+0")

        self._transparent = True
        try:
            self.root.attributes("-transparentcolor", "magenta")
            bg = "magenta"
        except tk.TclError:
            self._transparent = False
            bg = "#202020"

        self.canvas = tk.Canvas(self.root, width=self.SIZE, height=self.SIZE,
                                 bg=bg, highlightthickness=0)
        self.canvas.pack()
        self._ring = self.canvas.create_oval(3, 3, self.SIZE - 3, self.SIZE - 3,
                                              outline="#39ff88", width=3)
        self._closed = False
        self.hidden = True
        self.root.withdraw()

    def show(self):
        if self.hidden and not self._closed:
            self.root.deiconify()
            self.hidden = False

    def hide(self):
        if not self.hidden and not self._closed:
            self.root.withdraw()
            self.hidden = True

    def update(self, screen_point, clicked=False):
        """Move the ring to center on screen_point (screen pixel
        coordinates, e.g. from desktop_control.map_to_screen), and pump
        Tkinter's event queue. Call once per frame while desktop control
        is enabled and a hand is detected; call hide() otherwise."""
        if self._closed:
            return
        self.show()
        x, y = screen_point
        self.root.geometry(f"+{x - self.SIZE // 2}+{y - self.SIZE // 2}")
        self.canvas.itemconfig(self._ring, outline="#ff3b3b" if clicked else "#39ff88")
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True

    def close(self):
        if not self._closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._closed = True


class LaserPointer:
    """An always-on-top, click-through red dot that marks the mapped
    screen position of the tracked fingertip - the Presentation-mode
    equivalent of CursorReticle, and built the exact same way (see its
    docstring for the transparent/click-through mechanics).

    This is now the ONLY thing the laser-pointer gesture does: it does
    not move the real OS cursor and does not hold any key down. Earlier
    versions made the fist gesture move the real cursor (with Ctrl held,
    the combination PowerPoint uses to draw its own built-in laser dot)
    so the dot would show up on the actual slide/projector - but that
    meant the real cursor, and therefore every real click, followed the
    fist around. Drawing our own dot in its own overlay window instead
    gets the same "see it on screen" result without ever touching the
    real cursor: the fist gesture can't move anything or cause a
    misclick anymore, in this app or any other.
    """

    SIZE = 30

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gesture Laser")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.SIZE}x{self.SIZE}+0+0")

        self._transparent = True
        try:
            self.root.attributes("-transparentcolor", "magenta")
            bg = "magenta"
        except tk.TclError:
            self._transparent = False
            bg = "#202020"

        self.canvas = tk.Canvas(self.root, width=self.SIZE, height=self.SIZE,
                                 bg=bg, highlightthickness=0)
        self.canvas.pack()
        # Soft outer glow ring plus a solid core, so it reads as a laser
        # dot rather than a flat circle.
        self.canvas.create_oval(1, 1, self.SIZE - 1, self.SIZE - 1,
                                 outline="#ff3b3b", width=1)
        self.canvas.create_oval(7, 7, self.SIZE - 7, self.SIZE - 7,
                                 fill="#ff2b2b", outline="")
        self._closed = False
        self.hidden = True
        self.root.withdraw()

    def show(self):
        if self.hidden and not self._closed:
            self.root.deiconify()
            self.hidden = False

    def hide(self):
        if not self.hidden and not self._closed:
            self.root.withdraw()
            self.hidden = True

    def update(self, screen_point):
        """Move the dot to center on screen_point (screen pixel
        coordinates, e.g. from desktop_control.map_to_screen), and pump
        Tkinter's event queue. Call once per frame while the laser
        gesture is held; call hide() the instant it isn't."""
        if self._closed:
            return
        self.show()
        x, y = int(screen_point[0]), int(screen_point[1])
        self._calls = getattr(self, "_calls", 0) + 1
        if (x, y) != getattr(self, "_last_xy", None):
            self._last_xy = (x, y)
            self.root.geometry(f"+{x - self.SIZE // 2}+{y - self.SIZE // 2}")
        try:
            # Applying the move only needs idle tasks; a full update()
            # (which also drains every queued event) is comparatively
            # slow, so do it only now and then - this runs every frame
            # and was a real chunk of the dot's lag.
            self.root.update_idletasks()
            if self._calls % 8 == 0:
                self.root.update()
        except tk.TclError:
            self._closed = True

    def close(self):
        if not self._closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._closed = True


class LaserBanner:
    """A small always-on-top "LASER ACTIVE" badge drawn on the real
    desktop (outside the tracker's own window), so you can tell laser
    mode is on while a slideshow fills the screen.

    Built to stay out of the way of a presentation: borderless, never
    takes keyboard focus, and - on Windows - click-through, so it can't
    swallow a click or pull focus away from PowerPoint."""

    def __init__(self, text="LASER ACTIVE", hint="Point to aim  |  Fist to turn off"):
        self.root = tk.Tk()
        self.root.title("Gesture Laser Status")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.root.configure(bg="#b3121a")
        frame = tk.Frame(self.root, bg="#b3121a", padx=18, pady=8)
        frame.pack()
        tk.Label(frame, text="\u25cf  " + text, fg="#ffffff", bg="#b3121a",
                 font=("Segoe UI", 16, "bold")).pack()
        tk.Label(frame, text=hint, fg="#ffd9db", bg="#b3121a",
                 font=("Segoe UI", 10)).pack()
        self.root.update_idletasks()
        w, h = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+16")
        try:
            self.root.attributes("-alpha", 0.92)
        except tk.TclError:
            pass
        self._closed = False
        self.hidden = True
        self._click_through_applied = False
        self.root.withdraw()

    def _make_click_through(self):
        """Windows only: WS_EX_TRANSPARENT (clicks pass through),
        WS_EX_NOACTIVATE (never takes focus), WS_EX_TOOLWINDOW (no
        taskbar button). Silently skipped elsewhere."""
        if self._click_through_applied:
            return
        self._click_through_applied = True
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            GWL_EXSTYLE = -20
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= 0x00000020 | 0x08000000 | 0x00000080 | 0x00080000
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass

    def show(self):
        if self._closed:
            return
        if self.hidden:
            self.root.deiconify()
            self.hidden = False
            self.root.update_idletasks()
            self._make_click_through()
        self.pump()

    def hide(self):
        if not self.hidden and not self._closed:
            self.root.withdraw()
            self.hidden = True

    def pump(self):
        if self._closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True

    def close(self):
        if not self._closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._closed = True
