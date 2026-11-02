"""Stub `tkinter` good enough for the overlay/preview/laser windows to run headless."""
import sys, types


class TclError(Exception):
    pass


class _W:
    """Any widget: every method is a harmless no-op returning another widget-ish."""
    def __init__(self, *a, **k): pass
    def __getattr__(self, name):
        if name.startswith("winfo_"):                     # geometry queries must return numbers
            return (lambda *a, **k: 900) if "height" in name else (lambda *a, **k: 1600 if "width" in name else 0)
        return lambda *a, **k: None


class Canvas(_W):
    counts = {}
    def delete(self, *a): Canvas.counts.clear()
    def __getattr__(self, name):
        if name.startswith("create_"):
            def f(*a, **k): Canvas.counts[name] = Canvas.counts.get(name, 0) + 1
            return f
        return super().__getattr__(name)


def install():
    mod = types.ModuleType("tkinter")
    mod.Tk = mod.Toplevel = mod.Label = mod.Frame = _W
    mod.Canvas = Canvas
    mod.TclError = TclError
    mod.PIESLICE, mod.ARC, mod.CHORD = "pieslice", "arc", "chord"
    mod.__getattr__ = lambda name: _W
    sys.modules["tkinter"] = mod
    return mod
