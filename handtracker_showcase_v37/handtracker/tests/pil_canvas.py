"""Offscreen stand-in for a Tk canvas: implements the few create_* calls
ar3d uses, drawing with PIL so frames can be inspected as PNGs. Stipple
is approximated by alpha; Tk's real rendering will differ in detail."""
from PIL import Image, ImageDraw

_ALPHA = {"gray12": 45, "gray25": 80, "gray50": 130, "": 255, None: 255}


class PILCanvas:
    def __init__(self, w=900, h=600, bg=(12, 16, 24)):
        self.img = Image.new("RGBA", (w, h), bg + (255,))
        self.d = ImageDraw.Draw(self.img, "RGBA")

    @staticmethod
    def _c(col, a=255):
        col = col.lstrip("#")
        return (int(col[0:2], 16), int(col[2:4], 16), int(col[4:6], 16), a)

    def create_line(self, *pts, fill="#fff", width=1, dash=None, **kw):
        self.d.line(list(pts), fill=self._c(fill), width=width)

    def create_polygon(self, *pts, fill="#fff", outline="", stipple="", width=1, **kw):
        self.d.polygon(list(pts), fill=self._c(fill, _ALPHA.get(stipple, 130)))
        if outline:
            self.d.line(list(pts) + list(pts[:2]), fill=self._c(outline), width=width)

    def create_rectangle(self, x0, y0, x1, y1, outline="#fff", width=1, **kw):
        self.d.rectangle([x0, y0, x1, y1], outline=self._c(outline), width=width)

    def create_oval(self, x0, y0, x1, y1, outline="#fff", width=1, fill="", **kw):
        self.d.ellipse([x0, y0, x1, y1], outline=self._c(outline), width=width)

    def create_text(self, x, y, text="", fill="#fff", anchor="center", **kw):
        w = self.d.textlength(text)
        self.d.text((x - (0 if anchor == "w" else w / 2), y - 5), text, fill=self._c(fill))

    def save(self, path):
        self.img.convert("RGB").save(path)
