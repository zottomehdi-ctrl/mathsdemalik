"""Musique + design sonore, synthétisés de zéro (aucun sample).
Tout est calé sur la même ligne de temps que l'image (common.py)."""
import math
import sys
import numpy as np
from numba import njit
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.ndimage import minimum_filter1d
from scipy.io import wavfile

import common as C

SR = C.SR
N = int(C.DUR * SR)                 # 720 000 échantillons
L = N + int(3.0 * SR)               # marge pour les queues de réverb
RNG = np.random.default_rng(20261030)
TARGET_LUFS = -11.5


def bt(b):
    return b * C.BEAT


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def db(x):
    return 10 ** (x / 20.0)


def tarr(dur):
    return np.arange(int(dur * SR)) / SR


# ====================================================================== DSP (numba)
@njit(cache=True)
def _saw(freq, sr, ph):
    n = freq.shape[0]
    out = np.empty(n)
    for i in range(n):
        dt = freq[i] / sr
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
        v = 2.0 * ph - 1.0
        if ph < dt:
            x = ph / dt
            v -= x + x - x * x - 1.0
        elif ph > 1.0 - dt:
            x = (ph - 1.0) / dt
            v -= x * x + x + x + 1.0
        out[i] = v
    return out


@njit(cache=True)
def _svf(x, fc, q, mode, sr):
    n = x.shape[0]
    out = np.empty(n)
    ic1 = 0.0
    ic2 = 0.0
    k = 1.0 / q
    for i in range(n):
        f = fc[i]
        if f > sr * 0.45:
            f = sr * 0.45
        if f < 8.0:
            f = 8.0
        g = math.tan(math.pi * f / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            out[i] = v2
        elif mode == 1:
            out[i] = v1 * k
        else:
            out[i] = x[i] - k * v1 - v2
    return out


def saw(freq, ph=None):
    freq = np.asarray(freq, dtype=np.float64)
    return _saw(freq, float(SR), RNG.random() if ph is None else ph)


def square(freq, ph=None):
    ph = RNG.random() if ph is None else ph
    return 0.5 * (saw(freq, ph) - saw(freq, (ph + 0.5) % 1.0))


def sine(freq, ph=0.0):
    freq = np.broadcast_to(np.asarray(freq, dtype=np.float64), freq.shape if hasattr(freq, 'shape') else ())
    return np.sin(2 * np.pi * np.cumsum(freq) / SR + ph)


def svf(x, fc, q=0.707, mode='lp'):
    fc = np.broadcast_to(np.asarray(fc, dtype=np.float64), x.shape).copy()
    return _svf(np.ascontiguousarray(x, dtype=np.float64), fc, float(q),
                {'lp': 0, 'bp': 1, 'hp': 2}[mode], float(SR))


def bfilt(x, f, kind='low', order=2):
    sos = butter(order, f, btype=kind, fs=SR, output='sos')
    return sosfilt(sos, x, axis=-1)


def shelf(x, f0, gain_db, kind='high'):
    """filtre en plateau (RBJ cookbook), S = 1"""
    from scipy.signal import lfilter
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / SR
    cw, al = math.cos(w0), math.sin(w0) / 2 * math.sqrt(2)
    sA = math.sqrt(A)
    if kind == 'high':
        b = [A * ((A + 1) + (A - 1) * cw + 2 * sA * al), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - 2 * sA * al)]
        a = [(A + 1) - (A - 1) * cw + 2 * sA * al, 2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - 2 * sA * al]
    else:
        b = [A * ((A + 1) - (A - 1) * cw + 2 * sA * al), 2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - 2 * sA * al)]
        a = [(A + 1) + (A - 1) * cw + 2 * sA * al, -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - 2 * sA * al]
    return lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=-1)


def noise(n):
    return RNG.standard_normal(n)


def fade(x, a=0.002, r=0.01):
    n = x.shape[-1]
    na, nr = max(1, int(a * SR)), max(1, int(r * SR))
    e = np.ones(n)
    e[:na] = np.linspace(0, 1, na)
    e[-nr:] *= np.linspace(1, 0, nr)
    return x * e


def norm(x, peak=1.0):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


# ====================================================================== mixeur
class Mix:
    def __init__(self):
        self.bus = {}

    def get(self, name):
        if name not in self.bus:
            self.bus[name] = np.zeros((2, L))
        return self.bus[name]

    def add(self, name, sig, t, gain_db=0.0, pan=0.0):
        b = self.get(name)
        start = int(round(t * SR))
        g = db(gain_db)
        if sig.ndim == 1:
            a = (pan + 1) * math.pi / 4
            sig = np.vstack([sig * math.cos(a), sig * math.sin(a)]) * math.sqrt(2)
        else:
            sig = sig.copy()
            if pan:
                sig[0] *= min(1.0, 1 - pan)
                sig[1] *= min(1.0, 1 + pan)
        if start < 0:
            sig = sig[:, -start:]
            start = 0
        end = min(L, start + sig.shape[1])
        if end > start:
            b[:, start:end] += sig[:, :end - start] * g


M = Mix()


# ====================================================================== instruments
def kick(dur=0.5, f_hi=165.0, f_lo=47.0, pdec=0.030, adec=0.30, drive=2.0, click=0.30):
    t = tarr(dur)
    f = f_lo + (f_hi - f_lo) * np.exp(-t / pdec)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / adec)
    body = np.tanh(drive * body) / math.tanh(drive)
    cl = bfilt(noise(len(t)) * np.exp(-t / 0.0022), 2500, 'high')
    return fade(body + click * norm(cl), 0.0005, 0.03)


def clap(dur=0.45):
    t = tarr(dur)
    env = np.zeros_like(t)
    for off in (0.0, 0.010, 0.019):
        env += (t >= off) * np.exp(-np.maximum(t - off, 0) / 0.0055) * 0.9
    env += (t >= 0.027) * np.exp(-np.maximum(t - 0.027, 0) / 0.11)
    n = bfilt(noise(len(t)), [850, 3200], 'band')
    return fade(norm(n * env), 0.0003, 0.05)


def snare(dur=0.32, tone=190.0):
    t = tarr(dur)
    f = tone * (1 + 0.35 * np.exp(-t / 0.012))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.055)
    sn = bfilt(noise(len(t)), 1700, 'high') * np.exp(-t / 0.12)
    return fade(norm(0.8 * body + 0.7 * norm(sn)), 0.0003, 0.04)


_HAT_F = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * 1.35


def hat(open_=False):
    dur = 0.45 if open_ else 0.09
    t = tarr(dur)
    m = sum(np.sign(np.sin(2 * np.pi * f * t + RNG.random() * 6.28)) for f in _HAT_F)
    m = bfilt(m, [7000, 13500], 'band') + 0.5 * bfilt(noise(len(t)), 9000, 'high')
    env = np.exp(-t / (0.16 if open_ else 0.026))
    return fade(norm(m * env), 0.0002, 0.01)


def crash(dur=2.6):
    t = tarr(dur)
    out = []
    for _ in range(2):
        m = sum(np.sign(np.sin(2 * np.pi * f * 1.9 * t + RNG.random() * 6.28)) for f in _HAT_F)
        x = bfilt(0.6 * m + noise(len(t)), 3800, 'high', 3)
        env = 0.65 * np.exp(-t / 0.75) + 0.35 * np.exp(-t / 0.05)
        out.append(x * env)
    return fade(norm(np.vstack(out)), 0.0005, 0.2)


def sub_boom(dur=1.6, f0=72.0, f1=34.0, dec=0.55):
    t = tarr(dur)
    f = f1 + (f0 - f1) * np.exp(-t / 0.18)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / dec)
    return fade(np.tanh(1.4 * x), 0.002, 0.2)


def impact(dur=0.9):
    t = tarr(dur)
    fc = 200 + 7000 * np.exp(-t / 0.05)
    x = svf(noise(len(t)), fc, 0.8, 'lp') * np.exp(-t / 0.18)
    return fade(norm(x), 0.0005, 0.1)


def bloop(m=84, dur=0.35, drop=1.0):
    t = tarr(dur)
    f = mtof(m) * (1 + drop * np.exp(-t / 0.014))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.085)
    x += 0.35 * np.sin(2 * np.pi * np.cumsum(f * 0.5) / SR) * np.exp(-t / 0.05)
    return fade(norm(x), 0.0008, 0.05)


def tick(m=96, dur=0.06):
    t = tarr(dur)
    x = np.sin(2 * np.pi * mtof(m) * t) * np.exp(-t / 0.008)
    x += 0.5 * bfilt(noise(len(t)), 4000, 'high') * np.exp(-t / 0.002)
    return fade(norm(x), 0.0002, 0.01)


def flap(dur=0.05):
    t = tarr(dur)
    x = bfilt(noise(len(t)), [1800, 6000], 'band') * np.exp(-t / 0.004)
    x += 0.6 * np.sin(2 * np.pi * 820 * t) * np.exp(-t / 0.006)
    return fade(norm(x), 0.0001, 0.01)


def whoosh(dur, f0, f1, peak=0.6, q=1.4, pan0=0.0, pan1=0.0, stereo=True):
    t = tarr(dur)
    u = t / dur
    fc = f0 * (f1 / f0) ** u
    x = svf(noise(len(t)), fc, q, 'bp')
    env = np.where(u < peak, (u / peak) ** 2, ((1 - u) / (1 - peak)) ** 1.5)
    x = norm(x * env)
    if not stereo:
        return x
    p = pan0 + (pan1 - pan0) * u
    a = (p + 1) * np.pi / 4
    y = svf(noise(len(t)), fc * 1.07, q, 'bp')
    y = norm(y * env)
    return np.vstack([0.7 * x * np.cos(a) * 1.41 + 0.3 * y, 0.7 * x * np.sin(a) * 1.41 + 0.3 * y])


def riser(dur, semis=12, f_base=55):
    t = tarr(dur)
    u = t / dur
    fc = 300 * (9000 / 300) ** (u ** 1.3)
    nz = svf(noise(len(t)), fc, 1.8, 'bp')
    f = mtof(f_base) * 2 ** (semis * u ** 2 / 12)
    tone = sum(saw(f * 2 ** (d / 1200)) for d in (-12, 0, 11))
    tone = svf(tone, 400 + 6000 * u ** 2, 0.9, 'lp')
    env = u ** 2.2
    return norm(np.vstack([(0.6 * norm(nz) + 0.45 * norm(tone)) * env,
                           (0.6 * norm(svf(noise(len(t)), fc, 1.8, 'bp')) + 0.45 * norm(tone)) * env]))


def supersaw(f, t, voices=7, spread=18.0):
    """f: fréquence (scalaire), renvoie stéréo"""
    l = np.zeros_like(t)
    r = np.zeros_like(t)
    for v in range(voices):
        d = (v - (voices - 1) / 2) / ((voices - 1) / 2) * spread
        s = saw(np.full(len(t), f * 2 ** (d / 1200)))
        w = 1.0 if v == voices // 2 else 0.75
        pl = 0.5 + 0.45 * ((v % 2) * 2 - 1) * (abs(d) / spread)
        l += s * w * (1 - pl)
        r += s * w * pl
    return np.vstack([l, r])


def stab(notes, dur=0.32, bright=1.0, dec=0.16):
    t = tarr(dur)
    out = sum(supersaw(mtof(n), t, 7, 16) for n in notes)
    fc = 700 + 5200 * bright * np.exp(-t / 0.09)
    out = np.vstack([svf(out[0], fc, 0.9), svf(out[1], fc, 0.9)])
    env = np.exp(-t / dec) * (1 - np.exp(-t / 0.002))
    return fade(norm(out * env), 0.001, 0.03)


def pad(notes, dur, cutoff=1200, att=0.35, rel=0.6):
    t = tarr(dur + rel)
    out = sum(supersaw(mtof(n), t, 5, 12) for n in notes)
    lfo = 1 + 0.15 * np.sin(2 * np.pi * 0.35 * t)
    out = np.vstack([svf(out[0], cutoff * lfo, 0.7), svf(out[1], cutoff * lfo * 1.03, 0.7)])
    env = np.minimum(1, t / att) * np.where(t < dur, 1.0, np.exp(-(t - dur) / (rel / 3)))
    return fade(norm(out * env), 0.005, 0.05)


def pluck(m, dur=0.22, cutoff=1500.0, env_amt=4000.0, dec=0.16):
    t = tarr(dur)
    f = np.full(len(t), mtof(m))
    x = 0.7 * saw(f) + 0.4 * square(f * 1.003)
    fc = cutoff + env_amt * np.exp(-t / 0.05)
    x = svf(x, fc, 1.1)
    x *= np.exp(-t / dec)
    return fade(norm(x), 0.001, 0.02)


def bass(m, dur, cut=1600.0):
    t = tarr(dur)
    f = np.full(len(t), mtof(m))
    s = saw(f)
    fc = 220 + cut * np.exp(-t / 0.11)
    x = svf(s, fc, 1.0) * 0.9
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR)
    y = np.tanh(1.6 * (x + 0.9 * sub))
    env = np.minimum(1, t / 0.003) * np.where(t > dur - 0.03, np.clip((dur - t) / 0.03, 0, 1), 1)
    return norm(y * env)


def bell(m, dur=2.0, idx=3.2, ratio=3.5, dec=0.9):
    t = tarr(dur)
    f = mtof(m)
    I = idx * np.exp(-t / 0.22) + 0.4
    x = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t))
    x += 0.3 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t / 0.3)
    x *= np.exp(-t / dec)
    return fade(norm(x), 0.0005, 0.1)


def epiano(m, dur=2.0):
    t = tarr(dur)
    f = mtof(m)
    I = 1.8 * np.exp(-t / 0.35) + 0.15
    x = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * t))
    x += 0.15 * np.sin(2 * np.pi * f * 14 * t) * np.exp(-t / 0.02)
    x *= np.exp(-t / 1.1)
    return fade(norm(x), 0.001, 0.1)


def grain(f, dur):
    t = tarr(dur)
    w = np.sin(np.pi * t / dur) ** 2
    return np.sin(2 * np.pi * f * t + RNG.random() * 6.28) * w


def glide(freq_curve, dur, kind='tri'):
    """oscillateur qui suit une trajectoire de hauteur (Hz) donnée"""
    t = tarr(dur)
    f = np.interp(t / dur, np.linspace(0, 1, len(freq_curve)), freq_curve)
    s = saw(f)
    x = svf(s, f * 3.0, 0.8)
    return x


# ====================================================================== arrangement
def arrange():
    B = bt
    FM9 = [53, 56, 60, 63, 67]
    DBM7 = [53, 56, 60, 65]
    ABM7 = [56, 60, 63, 67]
    EB9 = [55, 58, 63, 65]
    C7 = [52, 58, 60, 64]
    FADD9 = [57, 60, 65, 67, 69]

    # ------------------------------------------------ INTRO (b0 -> b8)
    M.add('fx', bloop(84), B(0), -11)
    M.add('fx', tick(108, 0.03), B(0), -16)
    M.add('pad', pad(FM9, B(4), cutoff=700, att=1.2), B(0), -8)
    M.add('pad', pad(DBM7, B(3.75), cutoff=1100, att=0.3), B(4), -7)
    # anticipation : petit "inspir" avant l'étirement
    M.add('fx', whoosh(B(0.5), 1800, 500, peak=0.85, q=2.5), B(1.0), -20)
    # étirement du point en ligne : souffle stéréo qui s'ouvre
    w = whoosh(B(0.9), 400, 5000, peak=0.35, q=1.2)
    w[0] *= 1.0
    M.add('fx', w, B(1.42), -9)
    M.add('fx', sub_boom(0.6, 90, 50, 0.15), B(1.5), -14)
    # arpège : chaque note = une image-clé qui se pose (b2 -> b3.75), puis continue
    arp_fm = [65, 68, 72, 75, 79, 75, 72, 68]
    arp_db = [65, 68, 72, 73, 77, 73, 72, 68]
    step = 0
    b = 2.0
    while b < 7.75 - 1e-6:
        notes = arp_fm if b < 4 else arp_db
        m = notes[step % 8]
        prog = (b - 2) / 5.75
        p = pluck(m, 0.25, cutoff=600 + 4200 * prog ** 1.5, env_amt=2500 + 2500 * prog)
        acc = 0 if step % 4 == 0 else -3
        M.add('arp', p, B(b), -6 + acc, pan=0.35 * math.sin(step * 1.7))
        if b < 3.9:
            M.add('fx', tick(96 + (step % 8), 0.04), B(b), -18, pan=-0.6 + 0.17 * step)
        step += 1
        b += 0.25
    # éditeur de courbes (b4) : les clés montent
    M.add('fx', whoosh(B(0.5), 300, 2500, peak=0.5, q=2.0), B(3.9), -16)
    # le point parcourt la courbe : la hauteur suit littéralement l'easing
    run0, run1 = 4.55, 5.25
    curve = [C.EASE_SIG(u) for u in np.linspace(0, 1, 64)]
    fcurve = [mtof(72 + 24 * y) for y in curve]
    g = glide(fcurve, B(run1 - run0))
    env = np.minimum(1, np.arange(len(g)) / (0.02 * SR)) * np.linspace(1, 0.6, len(g))
    M.add('fx', fade(norm(g * env), 0.005, 0.03), B(run0), -15, pan=0.0)
    M.add('fx', bloop(91, 0.3, 0.6), B(run1), -10, pan=0.3)
    # repli du graphe
    M.add('fx', whoosh(B(0.25), 3000, 600, peak=0.7, q=2.0), B(5.25), -16)
    # kicks "battement de cœur" filtrés en mesure 2
    for bb in (4, 5, 6, 7):
        k = bfilt(kick(), 380, 'low')
        M.add('drums', k, B(bb), -5)
    # montée : caisse claire en croches puis doubles, riser
    rb = 5.5
    i = 0
    while rb < 7.75 - 1e-6:
        vel = -18 + 12 * (rb - 5.5) / 2.25
        M.add('drums', snare(0.25, 200 + 30 * (rb - 5.5)), B(rb), vel)
        rb += 0.5 if rb < 7.0 else 0.25
        i += 1
    for ph in C.PHRASES:  # chaque phrase claque
        M.add('fx', impact(0.5), B(ph[0]), -14)
        M.add('drums', clap(), B(ph[0]), -12)
    M.add('fx', riser(B(3.75)), B(4.0), -13)
    for sb, pn in ((5.56, (-0.6, 0.6)), (6.06, (0.6, -0.6)), (6.56, (-0.6, 0.6))):
        M.add('fx', whoosh(0.16, 1500, 6000, peak=0.25, q=2.0, pan0=pn[0], pan1=pn[1]), B(sb), -12)
    M.add('fx', bloop(86, 0.3, 0.7), B(7.02), -9, pan=0.3)
    M.add('fx', whoosh(B(0.25), 400, 7000, peak=0.97, q=1.2), B(7.75), -13)
    # ------------------------------------------------ DROP (b8)
    M.add('fx', crash(), B(8), -9)
    M.add('fx', sub_boom(1.8, 80, 33, 0.6), B(8), -4)
    M.add('fx', impact(1.0), B(8), -8)
    # tranche (b9) : "shhk" aller / retour
    M.add('fx', whoosh(0.22, 2500, 7000, peak=0.15, q=2.2, pan0=-0.6, pan1=0.6), B(9), -11)
    M.add('fx', whoosh(0.2, 6000, 2000, peak=0.8, q=2.2, pan0=0.6, pan1=-0.6), B(9.6), -14)
    # flips lettre par lettre (b10, b11) : clac de panneau à volets
    for fb in (10, 11):
        for i in range(6):
            M.add('fx', flap(), B(fb) + 0.03 * i + 0.12, -9, pan=-0.7 + 0.28 * i)
        M.add('fx', whoosh(0.3, 800, 4000, peak=0.3, q=1.5), B(fb) - 0.02, -18)
    # ------------------------------------------------ SYSTEMS (b12 -> b16)
    for i in range(10):
        M.add('fx', tick(100 + (i * 5) % 12, 0.03), B(12) + 0.022 * i, -20, pan=math.sin(i * 2.1) * 0.8)
    for wb, root in ((13, 72), (14, 75), (15, 79)):
        for j, iv in enumerate((0, 7, 12)):
            M.add('fx', bell(root + iv, 0.5, idx=1.2, ratio=2.0, dec=0.12), B(wb) + 0.03 * j, -17,
                  pan=(-0.5, 0, 0.5)[j])
    M.add('fx', whoosh(B(0.5), 6000, 900, peak=0.85, q=1.5), B(15.5), -14)
    # ------------------------------------------------ DIMENSION (b16 -> b20)
    M.add('fx', crash(), B(16), -14)
    M.add('fx', sub_boom(1.2, 60, 36, 0.5), B(16), -9)
    for i, m in enumerate((65, 72, 77)):
        M.add('bells', bell(m, 2.0), B(16) + 0.02 * i, -10, pan=(-0.4, 0.4, 0)[i])
    for i in range(8):  # les anneaux pivotent en cascade
        M.add('bells', bell((84, 82, 80, 77, 75, 72, 70, 68)[i], 1.0, idx=2.0, dec=0.5),
              B(17) + 0.04 * i, -16, pan=0.8 * math.sin(i))
    M.add('fx', whoosh(B(1), 200, 2000, peak=0.5, q=1.0), B(18), -15)
    for i, m in enumerate((68, 72, 75, 80)):
        M.add('bells', bell(m, 1.4, idx=2.5, dec=0.6), B(18) + 0.05 * i, -14, pan=-0.6 + 0.4 * i)
    M.add('fx', whoosh(B(1.0), 150, 9000, peak=0.92, q=1.1, pan0=0, pan1=0), B(19), -9)
    # ------------------------------------------------ PARTICLES (b20 -> b24)
    M.add('fx', crash(), B(20), -12)
    M.add('fx', impact(0.8), B(20), -9)
    M.add('fx', sub_boom(1.0, 70, 40, 0.3), B(20), -10)
    scale = [65, 68, 70, 72, 75]
    rng = np.random.default_rng(7)
    for i in range(110):
        tt = B(20) + rng.random() ** 1.4 * B(3.5)
        oc = 12 * rng.integers(1, 3)
        m = scale[rng.integers(0, 5)] + oc
        gdur = 0.03 + 0.07 * rng.random()
        M.add('sparkle', grain(mtof(m), gdur), tt, -20 + 6 * rng.random(), pan=rng.uniform(-1, 1))
    M.add('fx', whoosh(B(0.5), 9000, 300, peak=0.92, q=1.4), B(23.5), -12)
    # ------------------------------------------------ RHYTHM (b24 -> b28)
    words = {
        'TYPE': lambda tt: [M.add('fx', flap(), tt + 0.03 * i, -6, pan=-0.3 + 0.2 * i) for i in range(4)],
        'SHAPE': lambda tt: M.add('fx', bell(84, 0.4, idx=1.0, ratio=2.0, dec=0.1), tt, -10),
        'SPACE': lambda tt: M.add('bells', bell(77, 1.2, dec=0.5), tt, -11),
        'FLOW': lambda tt: M.add('fx', whoosh(0.3, 600, 6000, peak=0.3, q=1.4, pan0=-0.7, pan1=0.7), tt, -11),
        'SOUND': lambda tt: M.add('fx', sub_boom(0.4, 110, 45, 0.12), tt, -6),
        'CODE': lambda tt: [M.add('fx', tick(88 + (i * 7) % 15, 0.03), tt + 0.029 * i, -13,
                                  pan=math.sin(i * 1.3) * 0.7) for i in range(8)],
    }
    for wb, word in C.MONTAGE:
        words[word](B(wb))
        M.add('fx', impact(0.35), B(wb), -14)
    # stutter (b27 -> b27.75) : 16e qui montent
    for i in range(3):
        M.add('music', stab([m + 2 * i for m in C7], 0.12, bright=1.0, dec=0.05), B(27 + 0.25 * i), -6)
    M.add('fx', riser(B(1.75), semis=7, f_base=60), B(26.0), -11)
    M.add('fx', tick(103, 0.04), B(27.75), -14)
    # ------------------------------------------------ SIGNATURE (b28 -> b32)
    M.add('fx', crash(3.0), B(28), -9)
    M.add('fx', sub_boom(2.2, 75, 29, 0.8), B(28), -3)
    M.add('fx', impact(1.2), B(28), -10)
    for n in FADD9 + [72, 77]:
        M.add('final', epiano(n, 2.4), B(28), -13, pan=((n * 7) % 11 - 5) / 9)
    M.add('final', stab(FADD9, 2.2, bright=1.0, dec=0.65), B(28), -4)
    M.add('final', pad(FADD9 + [77], B(3.4), cutoff=1600, att=0.02, rel=1.0), B(28), -10)
    M.add('bass', bass(41, B(3.2), cut=900), B(28), -4)
    for i, n in enumerate((89, 93, 96, 101)):
        M.add('bells', bell(n, 1.6, idx=1.5, ratio=3.0, dec=0.5), B(28) + 0.045 * i, -18, pan=-0.6 + 0.4 * i)
    M.add('fx', whoosh(0.32, 900, 5000, peak=0.3, q=1.6, pan0=0.0, pan1=0.4), B(28) + 0.01, -15)
    for i, (bb, m) in enumerate(zip(C.DOT_BOUNCES, (77, 81, 84))):
        M.add('fx', bloop(m + 7, 0.35, 0.8), B(bb), (-3, -9, -14)[i], pan=0.25)

    # ------------------------------------------------ batterie / basse / accords (b8 -> b27.75)
    for bar in range(4):       # mesures 3 à 6
        b0 = 8 + 4 * bar
        for s in range(16):
            bb = b0 + s * 0.25
            if s % 4 == 0:
                M.add('drums', kick(), B(bb), 0)
            if s in (4, 12):
                M.add('drums', clap(), B(bb), -7)
                M.add('drums', snare(0.2), B(bb), -16)
            if s % 4 == 2:
                M.add('hats', hat(True), B(bb), -16, pan=0.15)
            elif s % 2 == 1:
                M.add('hats', hat(False), B(bb), -19 + (2 if s % 4 == 3 else 0), pan=-0.2)
    for s in range(16):        # mesure 7 : kick jusqu'au silence
        bb = 24 + s * 0.25
        if bb >= 27.75:
            break
        if s % 4 == 0:
            M.add('drums', kick(), B(bb), 0)
        if s in (4, 12):
            M.add('drums', clap(), B(bb), -7)
        if s % 2 == 1:
            M.add('hats', hat(False), B(bb), -18)
    # roulement final
    rb = 27.0
    while rb < 27.75 - 1e-6:
        M.add('drums', snare(0.15, 230), B(rb), -12 + 6 * (rb - 27))
        rb += 0.125
    M.add('drums', kick(0.9, adec=0.5), B(28), 1)

    roots = [(8, 41, FM9), (12, 37, DBM7), (16, 32, ABM7), (20, 39, EB9), (24, 37, DBM7), (26, 36, C7)]
    for (b0, root, chord) in roots:
        length = 2 if b0 >= 24 else 4
        for k in range(length):
            for off, oct_ in ((0.5, 0), (0.75, 12 if k == length - 1 else 0)):
                bb = b0 + k + off
                if bb >= 27.75:
                    continue
                dur = B(0.22) if off == 0.75 else B(0.3)
                M.add('bass', bass(root + oct_, dur), B(bb), -3)
        # stabs syncopées
        for s in (0, 3, 6, 8, 11, 14):
            bb = b0 + s * 0.25
            if bb >= b0 + length or bb >= 27.75:
                continue
            M.add('music', stab(chord, 0.3), B(bb), -9 if s else -6)
        M.add('pad', pad(chord, B(length) - 0.05, cutoff=1500, att=0.05, rel=0.2), B(b0), -14)
    # sub tenu sur le drop
    M.add('bass', bass(29, B(0.45)), B(8), -6)


def sidechain(kicks_beats, depth=0.75, rel=0.11):
    t = np.arange(L) / SR
    g = np.ones(L)
    for kb in kicks_beats:
        t0 = bt(kb)
        i0 = int(t0 * SR)
        i1 = min(L, i0 + int(0.5 * SR))
        u = t[i0:i1] - t0
        att = np.minimum(1, u / 0.004)
        g[i0:i1] = np.minimum(g[i0:i1], 1 - depth * att * np.exp(-u / rel))
    return g


def make_ir(dur=2.3, seed=3):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    ir = np.zeros((2, n))
    for ch in range(2):
        nz = rng.standard_normal(n)
        lo = bfilt(nz, 450, 'low') * np.exp(-t / 0.42)
        mi = bfilt(nz, [450, 3500], 'band') * np.exp(-t / 0.33)
        hi = bfilt(nz, 3500, 'high') * np.exp(-t / 0.16)
        x = lo + mi + 0.5 * hi
        x *= np.minimum(1, t / 0.012)
        pre = int(0.018 * SR)
        ir[ch, pre:] = x[:n - pre]
    ir /= np.sqrt(np.sum(ir ** 2) / 2)
    return ir


def limiter(x, ceiling_db=-1.2, look=0.004, rel=0.07):
    """limiteur à anticipation : gain <= besoin aux crêtes, attaque lissée, relâche exponentielle"""
    c = db(ceiling_db)
    peak = np.max(np.abs(x), axis=0)
    need = np.minimum(1.0, c / np.maximum(peak, 1e-9))
    d = max(2, int(look * SR))
    g = minimum_filter1d(need, size=2 * d + 1)
    g = _release(g, math.exp(-1.0 / (rel * SR)))
    k = np.ones(d + 1) / (d + 1)
    g = np.convolve(g, k, mode='same')
    return x * g[None, :], g


@njit(cache=True)
def _release(g, a):
    out = np.empty_like(g)
    v = 1.0
    for i in range(g.shape[0]):
        if g[i] < v:
            v = g[i]
        else:
            v = a * v + (1 - a) * g[i]
        out[i] = v
    return out


def lufs(x):
    """loudness intégrée BS.1770-4 (pondération K + portes absolue/relative)"""
    from scipy.signal import lfilter
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    y = lfilter(b2, a2, lfilter(b1, a1, x, axis=-1), axis=-1)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    p2 = y ** 2
    cs = np.concatenate([np.zeros((2, 1)), np.cumsum(p2, axis=1)], axis=1)
    starts = np.arange(0, y.shape[1] - blk, hop)
    z = ((cs[:, starts + blk] - cs[:, starts]) / blk).sum(axis=0)
    lk = -0.691 + 10 * np.log10(z + 1e-15)
    z = z[lk > -70]
    rel = -0.691 + 10 * np.log10(z.mean()) - 10
    z = z[(-0.691 + 10 * np.log10(z)) > rel]
    return -0.691 + 10 * np.log10(z.mean())


def render(path):
    arrange()
    # sidechain
    kicks = [8 + i for i in range(20) if 8 + i < 27.75]
    sc = sidechain(kicks)
    for name in ('bass', 'music', 'pad'):
        M.get(name)[:] *= sc[None, :]
    # bus EQ
    M.get('hats')[:] = bfilt(bfilt(M.get('hats'), 600, 'high'), 14000, 'low')
    M.get('music')[:] = bfilt(M.get('music'), 180, 'high')
    M.get('pad')[:] = bfilt(M.get('pad'), 150, 'high')
    M.get('arp')[:] = bfilt(M.get('arp'), 200, 'high')
    # délai ping-pong pointé sur l'arpège
    arp = M.get('arp')
    d = int(bt(0.75) * SR)
    dl = np.zeros_like(arp)
    for k in range(1, 5):
        g = 0.38 ** k
        src = bfilt(arp, 3500, 'low')
        ch = k % 2
        dl[ch, d * k:] += g * (src[0, :-d * k] + src[1, :-d * k]) * 0.5
    arp += dl
    # réverb commune (envois)
    ir = make_ir()
    sends = {'arp': 0.30, 'music': 0.22, 'pad': 0.35, 'bells': 0.55, 'fx': 0.18, 'sparkle': 0.6,
             'final': 0.45, 'drums': 0.07}
    send = sum(M.get(k) * v for k, v in sends.items())
    mono = bfilt(send.mean(axis=0), [250, 9000], 'band')
    wet = np.vstack([fftconvolve(mono, ir[0])[:L], fftconvolve(mono, ir[1])[:L]])
    # niveaux : chaque bus est calé sur une cible RMS mesurée sur sa zone active
    targets = {'drums': ((8, 24), -15.0), 'bass': ((8, 24), -19.0), 'music': ((8, 24), -23.0),
               'pad': ((0, 8), -27.0), 'hats': ((8, 24), -32.0), 'arp': ((2, 7.75), -22.0),
               'bells': ((16, 20), -23.0), 'sparkle': ((20, 24), -29.0), 'final': ((28, 30), -17.0),
               'fx': ((0, 32), -21.0)}
    dry = np.zeros((2, L))
    for k, ((b0, b1), tgt) in targets.items():
        x = M.get(k)
        seg_ = x[:, int(bt(b0) * SR):int(bt(b1) * SR)]
        r = 20 * np.log10(np.sqrt(np.mean(seg_ ** 2)) + 1e-12)
        dry += x * db(tgt - r)
        print(f'  {k:8s} rms {r:6.1f} -> {tgt:6.1f}  (gain {tgt - r:+.1f} dB)')
    mixd = dry + wet * db(-9)
    # bus : passe-haut, glue douce, saturation légère
    mixd = bfilt(mixd, 28, 'high')
    mixd = shelf(shelf(mixd, 7000, -3.5, 'high'), 75, -2.5, 'low')
    mixd = np.tanh(mixd * 1.6) / 1.6
    # loudness cible ~ -11 LUFS, puis limiteur
    cur = lufs(mixd[:, :N])
    mixd *= db(TARGET_LUFS - cur)
    mixd, gr = limiter(mixd, -1.3)
    print('limiter GR max dB', 20*np.log10(gr.min()), 'mean over drop', 20*np.log10(gr[int(bt(8)*SR):int(bt(24)*SR)].mean()))
    out = mixd[:, :N].copy()
    nf = int(0.35 * SR)
    out[:, -nf:] *= np.linspace(1, 0, nf) ** 2
    out[:, :int(0.002 * SR)] *= np.linspace(0, 1, int(0.002 * SR))
    wavfile.write(path, SR, (np.clip(out.T, -1, 1) * 32767).astype(np.int16))
    np.save(path.replace('.wav', '_mono.npy'), out.mean(axis=0).astype(np.float32))
    print('peak dBFS', 20 * np.log10(np.max(np.abs(out))), 'LUFS', lufs(out))


if __name__ == '__main__':
    render(sys.argv[1] if len(sys.argv) > 1 else 'music.wav')
