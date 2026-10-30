"""Petite boîte à outils graphique au-dessus de skia : texte façonné (HarfBuzz),
peinture, chemins, formes polaires, projection 3D."""
import math
import numpy as np
import skia
import uharfbuzz as hb

import common as C

INTER = '/usr/share/fonts/opentype/inter/'
JB = '/usr/share/fonts/truetype/jetbrains-mono/'
FONT_FILES = {
    'black': INTER + 'InterDisplay-Black.otf',
    'bold': INTER + 'InterDisplay-Bold.otf',
    'medium': INTER + 'InterDisplay-Medium.otf',
    'mono': JB + 'JetBrainsMono-Medium.ttf',
    'monob': JB + 'JetBrainsMono-Bold.ttf',
}


def col(c, a=1.0):
    return skia.Color4f(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, a)


def paint(c, a=1.0, stroke=None, cap='round', join='round', aa=True):
    p = skia.Paint(AntiAlias=aa)
    p.setColor4f(col(c, a))
    if stroke is not None:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeCap({'round': skia.Paint.kRound_Cap, 'butt': skia.Paint.kButt_Cap,
                        'square': skia.Paint.kSquare_Cap}[cap])
        p.setStrokeJoin({'round': skia.Paint.kRound_Join, 'miter': skia.Paint.kMiter_Join}[join])
    return p


class Face:
    def __init__(self, path):
        self.tf = skia.Typeface.MakeFromFile(path)
        blob = hb.Blob.from_file_path(path)
        self.hbface = hb.Face(blob)
        self.hbfont = hb.Font(self.hbface)
        self.upem = self.hbface.upem
        self.font = skia.Font(self.tf, self.upem)
        self.font.setLinearMetrics(True)
        self.font.setHinting(skia.FontHinting.kNone)
        self._paths = {}
        g = self.font.textToGlyphs('H')[0]
        self.cap = -self.font.getPath(g).computeTightBounds().top()      # hauteur de capitale (unités)
        gx = self.font.textToGlyphs('x')[0]
        self.xh = -self.font.getPath(gx).computeTightBounds().top()

    def gpath(self, gid):
        if gid not in self._paths:
            self._paths[gid] = self.font.getPath(gid)
        return self._paths[gid]

    def shape(self, text, features=None):
        if not text:
            return [], 0
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hbfont, buf, features or {"kern": True, "liga": True})
        out = []
        x = 0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            out.append((info.codepoint, x + pos.x_offset, pos.x_advance))
            x += pos.x_advance
        return out, x


_FACES = {}


def face(name):
    if name not in _FACES:
        _FACES[name] = Face(FONT_FILES[name])
    return _FACES[name]


class Word:
    """Mot mis en forme : glyphes positionnés à une taille donnée (px), approche en em."""

    def __init__(self, fname, text, size, tracking=0.0):
        self.f = face(fname)
        self.text = text
        self.size = size
        s = size / self.f.upem
        self.s = s
        shaped, adv = self.f.shape(text)
        self.glyphs = []
        for i, (gid, xo, a) in enumerate(shaped):
            self.glyphs.append({'gid': gid, 'x': xo * s + i * tracking * size,
                                'adv': a * s, 'ch': text[i] if i < len(text) else ''})
        self.width = adv * s + max(0, len(shaped) - 1) * tracking * size
        self.cap = self.f.cap * s
        self.xh = self.f.xh * s
        for g in self.glyphs:
            b = self.f.gpath(g['gid']).computeTightBounds()
            g['bounds'] = skia.Rect(b.left() * s, b.top() * s, b.right() * s, b.bottom() * s)
            g['cx'] = g['x'] + (b.left() + b.right()) * 0.5 * s if b.width() > 0 else g['x'] + g['adv'] / 2

    def glyph_path(self, i, x0=0.0, y0=0.0):
        g = self.glyphs[i]
        p = skia.Path(self.f.gpath(g['gid']))
        p.transform(skia.Matrix.MakeAll(self.s, 0, x0 + g['x'], 0, self.s, y0, 0, 0, 1))
        return p

    def path(self, x0=0.0, y0=0.0):
        p = skia.Path()
        for i in range(len(self.glyphs)):
            p.addPath(self.glyph_path(i, x0, y0))
        return p


_WORDS = {}


def word(fname, text, size, tracking=0.0):
    k = (fname, text, round(size, 2), tracking)
    if k not in _WORDS:
        _WORDS[k] = Word(fname, text, size, tracking)
    return _WORDS[k]


def text(c, s, x, y, fname='mono', size=16, color=C.PAPER, a=1.0, align='left', tracking=0.0):
    """texte simple (HUD) : approche manuelle, alignement gauche/droite/centre"""
    if not s:
        return 0.0
    w = word(fname, s, size, tracking)
    if align == 'right':
        x -= w.width
    elif align == 'center':
        x -= w.width / 2
    c.drawPath(w.path(x, y), paint(color, a))
    return w.width


def trimmed(path, t0, t1):
    """portion [t0, t1] (fractions de longueur) d'un chemin, sur tous ses contours"""
    t0, t1 = max(0.0, t0), min(1.0, t1)
    out = skia.Path()
    if t1 <= t0:
        return out
    pm = skia.PathMeasure(path, False)
    lens = []
    while True:
        lens.append(pm.getLength())
        if not pm.nextContour():
            break
    total = sum(lens)
    pm = skia.PathMeasure(path, False)
    acc = 0.0
    for L in lens:
        a, b = max(t0 * total - acc, 0.0), min(t1 * total - acc, L)
        if b > a:
            pm.getSegment(a, b, out, True)
        acc += L
        pm.nextContour()
    return out


def poly(pts, closed=True):
    return skia.Path.Polygon([skia.Point(float(x), float(y)) for x, y in pts], closed)


# ------------------------------------------------------------------ formes polaires (morphing)
NPOLAR = 96
PHI = np.linspace(0, 2 * np.pi, NPOLAR, endpoint=False) - np.pi / 2


def _rect_exit(phi, a, b):
    c, s = np.abs(np.cos(phi)) + 1e-9, np.abs(np.sin(phi)) + 1e-9
    return np.minimum(a / c, b / s)


def shape_r(kind):
    p = PHI
    if kind == 'circle':
        r = np.ones_like(p)
    elif kind == 'square':
        r = _rect_exit(p, 0.86, 0.86)
    elif kind == 'diamond':
        r = _rect_exit(p + np.pi / 4, 0.8, 0.8)
    elif kind == 'plus':
        r = np.maximum(_rect_exit(p, 1.08, 0.36), _rect_exit(p, 0.36, 1.08))
    elif kind == 'triangle':
        rin = 0.55
        k = np.mod(p + np.pi / 2, 2 * np.pi / 3) - np.pi / 3
        r = rin / np.cos(k)
        r = r * 1.0
    elif kind == 'star':
        k = np.mod(p + np.pi / 2, 2 * np.pi / 5) / (2 * np.pi / 5)
        r = 0.48 + 0.62 * np.abs(1 - 2 * k) ** 1.6
    else:
        raise ValueError(kind)
    return r


SHAPES = {k: shape_r(k) for k in ('circle', 'square', 'diamond', 'plus', 'triangle', 'star')}


def polar_path(cx, cy, r, scale, rot=0.0, sx=1.0, sy=1.0):
    ang = PHI + rot
    xs = cx + np.cos(ang) * r * scale * sx
    ys = cy + np.sin(ang) * r * scale * sy
    return skia.Path.Polygon([skia.Point(float(x), float(y)) for x, y in zip(xs, ys)], True)


# ------------------------------------------------------------------ 3D
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def axis_rot(axis, a):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = math.cos(a), math.sin(a)
    C1 = 1 - c
    return np.array([[c + x * x * C1, x * y * C1 - z * s, x * z * C1 + y * s],
                     [y * x * C1 + z * s, c + y * y * C1, y * z * C1 - x * s],
                     [z * x * C1 - y * s, z * y * C1 + x * s, c + z * z * C1]])


def project(P, cam_d=1600.0, cx=C.W / 2, cy=C.H / 2, f=None):
    """P (..., 3) -> (x, y, profondeur, échelle)"""
    f = cam_d if f is None else f
    z = P[..., 2] + cam_d
    zc = np.maximum(z, 1.0)
    k = f / zc
    return cx + P[..., 0] * k, cy + P[..., 1] * k, z, k


def flip_matrix(xc, yc, theta, f=1800.0):
    """rotation autour d'un axe horizontal passant par (xc, yc), avec perspective"""
    c, s = math.cos(theta), math.sin(theta)
    m = skia.Matrix.MakeAll(1, 0, 0, 0, c, 0, 0, s / f, 1)
    t1 = skia.Matrix.Translate(xc, yc)
    t0 = skia.Matrix.Translate(-xc, -yc)
    return skia.Matrix.Concat(t1, skia.Matrix.Concat(m, t0))
