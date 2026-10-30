"""Constantes partagées entre la musique et l'image : une seule ligne de temps."""
import math

W, H = 1920, 1080
FPS = 60
DUR = 15.0
NFRAMES = int(round(DUR * FPS))          # 900
BPM = 128
BEAT = 60.0 / BPM                        # 0.46875 s
BAR = 4 * BEAT                           # 1.875 s -> 15 s = 8 mesures exactes
SR = 48000


def bt(b):
    """beat -> secondes"""
    return b * BEAT


def tb(t):
    """secondes -> beat"""
    return t / BEAT


# ---------------------------------------------------------------- palette
INK = (13, 13, 16)
PAPER = (244, 240, 232)
SIGNAL = (255, 86, 32)
COBALT = (40, 56, 255)
GREY = (120, 118, 112)


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def seg(t, t0, t1):
    if t1 == t0:
        return 1.0 if t >= t1 else 0.0
    return clamp((t - t0) / (t1 - t0))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def out_cubic(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def in_cubic(x):
    x = clamp(x)
    return x ** 3


def io_cubic(x):
    x = clamp(x)
    return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2


def out_quint(x):
    x = clamp(x)
    return 1 - (1 - x) ** 5


def io_quint(x):
    x = clamp(x)
    return 16 * x ** 5 if x < .5 else 1 - (-2 * x + 2) ** 5 / 2


def out_expo(x):
    x = clamp(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def in_expo(x):
    x = clamp(x)
    return 0.0 if x <= 0 else 2 ** (10 * x - 10)


def io_expo(x):
    x = clamp(x)
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return 2 ** (20 * x - 10) / 2 if x < .5 else (2 - 2 ** (-20 * x + 10)) / 2


def out_back(x, s=1.70158):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def in_back(x, s=1.70158):
    x = clamp(x)
    return (s + 1) * x ** 3 - s * x ** 2


def spring(dt, w=22.0, z=0.38):
    """Ressort amorti (réponse indicielle). dt en secondes depuis le déclenchement."""
    if dt <= 0:
        return 0.0
    if z >= 1:
        return 1 - math.exp(-w * dt) * (1 + w * dt)
    wd = w * math.sqrt(1 - z * z)
    e = math.exp(-z * w * dt)
    return 1 - e * (math.cos(wd * dt) + (z * w / wd) * math.sin(wd * dt))


class Bezier:
    """Courbe d'accélération façon CSS cubic-bezier(x1, y1, x2, y2)."""

    def __init__(self, x1, y1, x2, y2):
        self.p = (x1, y1, x2, y2)
        self.cx = 3 * x1
        self.bx = 3 * (x2 - x1) - self.cx
        self.ax = 1 - self.cx - self.bx
        self.cy = 3 * y1
        self.by = 3 * (y2 - y1) - self.cy
        self.ay = 1 - self.cy - self.by

    def _x(self, u):
        return ((self.ax * u + self.bx) * u + self.cx) * u

    def _y(self, u):
        return ((self.ay * u + self.by) * u + self.cy) * u

    def _dx(self, u):
        return (3 * self.ax * u + 2 * self.bx) * u + self.cx

    def __call__(self, x):
        x = clamp(x)
        u = x
        for _ in range(8):
            d = self._dx(u)
            if abs(d) < 1e-6:
                break
            u2 = u - (self._x(u) - x) / d
            if not 0 <= u2 <= 1:
                break
            u = u2
        if abs(self._x(u) - x) > 1e-4:          # repli : dichotomie
            lo, hi = 0.0, 1.0
            for _ in range(40):
                u = (lo + hi) / 2
                if self._x(u) < x:
                    lo = u
                else:
                    hi = u
        return self._y(u)


# la courbe "signature" du film : très douce au départ, très franche à l'arrivée
EASE_SIG = Bezier(0.83, 0.0, 0.17, 1.0)
EASE_SNAP = Bezier(0.16, 1.0, 0.3, 1.0)


# ---------------------------------------------------------------- couleurs (OKLab)
def _lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _gam(c):
    c = max(0.0, min(1.0, c))
    return 255.0 * (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055)


def to_oklab(rgb):
    r, g, b = (_lin(float(v)) for v in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def from_oklab(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return (_gam(r), _gam(g), _gam(bb))


def mix(c1, c2, t):
    if t <= 0:
        return c1
    if t >= 1:
        return c2
    a, b = to_oklab(c1), to_oklab(c2)
    return from_oklab(tuple(lerp(x, y, t) for x, y in zip(a, b)))


# ---------------------------------------------------------------- bruit déterministe
def hash11(n):
    x = math.sin(n * 127.1 + 311.7) * 43758.5453
    return x - math.floor(x)


def noise1(x, seed=0.0):
    i = math.floor(x)
    f = x - i
    a = hash11(i + seed * 17.0)
    b = hash11(i + 1 + seed * 17.0)
    u = f * f * (3 - 2 * f)
    return (a + (b - a) * u) * 2 - 1


# ---------------------------------------------------------------- partition visuelle
# Scènes (en beats). 32 beats = 15 s.
SCENES = [
    (0, 8, "01", "TIMING"),
    (8, 12, "02", "TYPOGRAPHY"),
    (12, 16, "03", "SYSTEMS"),
    (16, 20, "04", "DIMENSION"),
    (20, 24, "05", "PARTICLES"),
    (24, 28, "06", "RHYTHM"),
    (28, 32, "07", "SIGNATURE"),
]

# impacts (beat, force) -> secousse caméra + aberration chromatique
HITS = [(8, 1.0), (9, 0.35), (10, 0.5), (11, 0.5), (12, 0.45), (13, 0.25), (14, 0.25),
        (15, 0.3), (16, 0.8), (17, 0.3), (18, 0.4), (19, 0.3), (20, 0.9), (22, 0.3),
        (24, 0.55), (24.5, 0.35), (25, 0.35), (25.5, 0.35), (26, 0.35), (26.5, 0.4),
        (28, 0.6)]

# Intro : 8 images-clés posées en doubles-croches (b2 -> b3.75)
KEYFRAME_BEATS = [2 + 0.25 * i for i in range(8)]
# Phrases de l'intro
PHRASES = [(5.5, "NO FOOTAGE"), (6.0, "NO SAMPLES"), (6.5, "NO PRESETS"), (7.0, "JUST CODE")]
# Montage final : mots-fenêtres en croches
MONTAGE = [(24.0, "TYPE"), (24.5, "SHAPE"), (25.0, "SPACE"), (25.5, "FLOW"),
           (26.0, "SOUND"), (26.5, "CODE")]
# rebonds du point final (beats) : chute puis 2 rebonds
DOT_BOUNCES = [29.5, 29.5 + 0.5, 29.5 + 0.75]
