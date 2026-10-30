"""Les sept scènes du showreel. Chaque fonction dessine une image complète à l'instant t (s)."""
import math
import os
import numpy as np
import skia

import common as C
import gfx as G
from common import (W, H, INK, PAPER, SIGNAL, COBALT, GREY, bt, tb, seg, clamp, lerp, mix,
                    smooth, out_cubic, in_cubic, io_cubic, out_quint, out_expo, in_expo, io_expo,
                    out_back, in_back, spring, EASE_SIG)
from gfx import paint, col

CX, CY = W / 2, H / 2
HERE = os.path.dirname(os.path.abspath(__file__))


def bg(c, color):
    c.drawColor(col(color))


# ====================================================================== S1 — TIMING (b0 -> b8)
GX0, GX1, GY0, GY1 = 360.0, 1560.0, 820.0, 260.0          # zone du graphe
KX = [GX0 + (GX1 - GX0) * i / 7 for i in range(8)]         # abscisses des clés
KY = [GY0 + (GY1 - GY0) * EASE_SIG(i / 7) for i in range(8)]
TL_Y = CY                                                  # la ligne de temps
DOTGRID = np.array([(x, y) for x in np.arange(48, W, 48.0) for y in np.arange(36, H, 48.0)])
PHRASE_SIZE = 250
DOT_R = 19.0


def phrase_layout(k):
    """mot, x0, ligne de base, extrémités de la rature, position du point final"""
    txt = ['FOOTAGE', 'SAMPLES', 'PRESETS', 'JUST CODE'][k]
    w = G.word('black', txt, PHRASE_SIZE, -0.012)
    extra = 2.6 * DOT_R + 8 if k == 3 else 0
    x0 = CX - (w.width + extra) / 2
    base = CY + w.cap / 2
    sy = base - w.cap * 0.47
    strike = (x0 - 34, x0 + w.width + 34, sy)
    dot = (x0 + w.width + 0.45 * DOT_R + 18, base - DOT_R)
    return w, x0, base, strike, dot


def dot_state_s1(t):
    """position, rayon, étirement du point (le fil rouge) pendant l'intro"""
    b = tb(t)
    if b < 4.45:
        return None
    if b < 5.25:          # course sur la courbe
        u = seg(b, 4.55, 5.25)
        x = GX0 + (GX1 - GX0) * u
        y = GY0 + (GY1 - GY0) * EASE_SIG(u)
        r = 16 * spring(t - bt(4.45), 30, 0.45)
        return x, y, r, 1.0, 1.0, 0.0
    strikes = [phrase_layout(k)[3] for k in range(3)]
    dirs = [1, -1, 1]

    def sp(k, end):
        x0, x1, y = strikes[k]
        if dirs[k] < 0:
            x0, x1 = x1, x0
        return (x1 if end else x0), y
    # trajectoires : (beat début, beat fin, départ, arrivée, mode)
    keys = [(5.25, 5.52, (GX1, GY1), sp(0, False), 'hop'),
            (5.56, 5.80, sp(0, False), sp(0, True), 'strike'),
            (5.80, 6.02, sp(0, True), sp(1, False), 'hop'),
            (6.06, 6.30, sp(1, False), sp(1, True), 'strike'),
            (6.30, 6.52, sp(1, True), sp(2, False), 'hop'),
            (6.56, 6.80, sp(2, False), sp(2, True), 'strike'),
            (6.80, 7.02, sp(2, True), phrase_layout(3)[4], 'hop'),
            (7.02, 99.0, phrase_layout(3)[4], phrase_layout(3)[4], 'rest')]
    for (b0, b1, p0, p1, mode) in keys:
        if b < b1 or mode == 'rest':
            break
    u = seg(b, b0, b1)
    if mode == 'hop':
        e = io_cubic(u)
        x = lerp(p0[0], p1[0], e)
        y = lerp(p0[1], p1[1], e) - math.sin(math.pi * e) * 120
        vx = (p1[0] - p0[0])
        stretch = 1 + 0.5 * math.sin(math.pi * u)
        ang = math.atan2(-math.cos(math.pi * e) * 120 * math.pi / 2 + (p1[1] - p0[1]), vx)
        return x, y, DOT_R, stretch, 1 / stretch ** 0.5, ang
    if mode == 'strike':
        e = out_quint(u)
        x = lerp(p0[0], p1[0], e)
        st = 1 + 1.2 * (1 - e) * (u > 0)
        return x, p0[1], DOT_R, st, 1 / st ** 0.5, 0.0
    # repos : petite écrasée à l'atterrissage
    land = t - bt(7.02)
    sq = 0.35 * math.exp(-land / 0.06) * math.cos(land * 40) if land > 0 else 0
    return p1[0], p1[1], DOT_R, 1 + sq, 1 - sq, 0.0


def draw_dot(c, x, y, r, sx=1.0, sy=1.0, ang=0.0, color=SIGNAL, a=1.0):
    if r <= 0.05:
        return
    c.save()
    c.translate(x, y)
    c.rotate(math.degrees(ang))
    c.scale(sx, sy)
    c.drawCircle(0, 0, r, paint(color, a))
    c.restore()


def s1(c, t):
    b = tb(t)
    bg(c, INK)
    c.save()
    # ---- caméra : plongée dans le point (b7.75 -> b8)
    zu = seg(b, 7.75, 8.0)
    if zu > 0:
        dx, dy = phrase_layout(3)[4]
        s = 1 + 85 * in_expo(zu) ** 1.15
        e = io_cubic(zu)
        qx, qy = lerp(dx, CX, e), lerp(dy, CY, e)
        c.translate(qx, qy)
        c.scale(s, s)
        c.translate(-dx, -dy)

    # ---- 0. trame de fond révélée par l'onde du premier point
    gr = 1500 * out_cubic(seg(t, 0.0, 1.1))
    ga0 = 1 - seg(b, 5.2, 5.5)
    if gr > 0 and ga0 > 0:
        d = np.hypot(DOTGRID[:, 0] - CX, DOTGRID[:, 1] - TL_Y)
        for lo, hi, a in ((0, gr - 90, 0.11), (gr - 90, gr, 0.40)):
            sel = DOTGRID[(d >= lo) & (d < hi)]
            if len(sel):
                pp = paint(PAPER, a * ga0)
                pp.setStrokeWidth(2.2)
                pp.setStrokeCap(skia.Paint.kRound_Cap)
                c.drawPoints(skia.Canvas.kPoints_PointMode, [skia.Point(float(x), float(y)) for x, y in sel], pp)
    # ---- 1. le point apparaît, prend son élan, devient une ligne
    if b < 1.5:
        r = 20 * spring(t, 26, 0.32)
        u = smooth(seg(b, 0.95, 1.5))
        sx, sy = 1 - 0.32 * u, 1 + 0.28 * u
        draw_dot(c, CX, TL_Y, r, sx, sy)
        rr = seg(t, 0.0, 0.55)
        if 0 < rr < 1:
            c.drawCircle(CX, TL_Y, 20 + 150 * out_cubic(rr), paint(SIGNAL, (1 - rr) ** 2 * 0.8, stroke=2))
    # ---- 2. ligne de temps
    if 1.5 <= b < 5.5:
        lt = t - bt(1.5)
        hl = 800 * spring(lt, 15, 0.62)
        th = lerp(40 * 0.68, 3.0, out_cubic(seg(b, 1.5, 1.85)))
        # la ligne descend devenir l'axe du temps du graphe
        gy = lerp(TL_Y, GY0, io_cubic(seg(b, 3.95, 4.35)))
        cl = mix(SIGNAL, PAPER, seg(b, 1.55, 1.9))
        fade_out = 1 - seg(b, 5.25, 5.45)
        x0 = CX - hl
        x1 = CX + hl
        if b > 3.95:   # recadrage sur la zone du graphe
            v = io_cubic(seg(b, 3.95, 4.35))
            x0, x1 = lerp(x0, GX0 - 40, v), lerp(x1, GX1 + 160, v)
        tp = G.trimmed(G.poly([(x0, gy), (x1, gy)], False), seg(b, 5.25, 5.45), 1.0)
        c.drawPath(tp, paint(cl, fade_out, stroke=th))
        # graduations (images 0 -> 900)
        if b > 1.85:
            for k in range(37):
                fx = k / 36
                x = lerp(x0, x1, fx) if b > 3.95 else CX - 800 + 1600 * fx
                d = abs(x - CX) / 800
                g = spring(t - bt(1.85) - d * 0.22, 30, 0.5)
                major = k % 4 == 0
                hgt = (16 if major else 7) * g * fade_out
                if hgt > 0.2:
                    c.drawLine(x, gy - hgt, x, gy, paint(PAPER, 0.55 * fade_out, stroke=1.5, cap='butt'))
                if major and g > 0.05:
                    G.text(c, f'{k * 25:03d}', x, gy + 26, 'mono', 12, PAPER, 0.4 * min(1, g) * fade_out, 'center')
            # tête de lecture = l'instant présent du film
            ph = lerp(x0, x1, t / C.DUR)
            pa = seg(b, 2.0, 2.3) * fade_out
            c.drawLine(ph, gy - 34, ph, gy + 8, paint(SIGNAL, pa, stroke=2, cap='butt'))
            c.drawPath(G.poly([(ph - 7, gy - 44), (ph + 7, gy - 44), (ph, gy - 34)]), paint(SIGNAL, pa))
        # ---- 3. éditeur de courbes
        ga = seg(b, 3.95, 4.3) * (1 - seg(b, 5.2, 5.42))
        if ga > 0:
            for x in KX:
                c.drawLine(x, GY1 - 40, x, GY0, paint(PAPER, 0.07 * ga, stroke=1, cap='butt'))
            for k in range(5):
                y = lerp(GY0, GY1, k / 4)
                c.drawLine(GX0 - 40, y, GX1 + 40, y, paint(PAPER, 0.07 * ga, stroke=1, cap='butt'))
            G.text(c, 'VALUE', GX0 - 40, GY1 - 58, 'mono', 13, PAPER, 0.5 * ga)
            G.text(c, 'TIME  ->', GX1 + 160, GY0 + 52, 'mono', 13, PAPER, 0.5 * ga, 'right')
            G.text(c, 'cubic-bezier(.83, 0, .17, 1)', GX1 + 160, GY1 - 58, 'mono', 15, SIGNAL, 0.9 * ga, 'right')
            P0, P1 = (GX0, GY0), (GX0 + 0.83 * (GX1 - GX0), GY0)
            P2, P3 = (GX0 + 0.17 * (GX1 - GX0), GY1), (GX1, GY1)
            curve = skia.Path()
            curve.moveTo(*P0)
            curve.cubicTo(*P1, *P2, *P3)
            tr0 = seg(b, 5.22, 5.45)
            tr1 = out_cubic(seg(b, 4.05, 4.55))
            c.drawPath(G.trimmed(curve, tr0, tr1), paint(PAPER, 1.0, stroke=5))
            hu = out_cubic(seg(b, 4.2, 4.55)) * (1 - seg(b, 5.2, 5.4))
            for A, Bp in ((P0, P1), (P3, P2)):
                if hu > 0:
                    ex, ey = lerp(A[0], Bp[0], hu), lerp(A[1], Bp[1], hu)
                    c.drawLine(A[0], A[1], ex, ey, paint(PAPER, 0.55, stroke=1.5))
                    c.drawCircle(ex, ey, 7, paint(INK))
                    c.drawCircle(ex, ey, 7, paint(PAPER, 0.9, stroke=2))
            # lecture de la valeur : l'espacement réel du mouvement
            ru = seg(b, 4.55, 5.25)
            if b > 4.45:
                rx = GX1 + 120
                c.drawLine(rx, GY1, rx, GY0, paint(PAPER, 0.25 * ga, stroke=1.5, cap='butt'))
                for k in range(11):
                    uk = k / 10
                    if ru >= uk:
                        yk = GY0 + (GY1 - GY0) * EASE_SIG(uk)
                        xk = GX0 + (GX1 - GX0) * uk
                        c.drawCircle(xk, yk, 9, paint(PAPER, 0.35 * ga, stroke=1.5))
                        c.drawLine(rx - 12, yk, rx + 12, yk, paint(SIGNAL, 0.8 * ga, stroke=2, cap='butt'))
                yv = GY0 + (GY1 - GY0) * EASE_SIG(ru)
                c.drawCircle(rx, yv, 8, paint(SIGNAL, ga))
                if 0 < ru < 1:
                    dash = skia.Paint(paint(PAPER, 0.35 * ga, stroke=1.2))
                    dash.setPathEffect(skia.DashPathEffect.Make([6, 6], 0))
                    c.drawLine(GX0 + (GX1 - GX0) * ru, yv, rx, yv, dash)
        # ---- clés
        for i, kb in enumerate(C.KEYFRAME_BEATS):
            kt = t - bt(kb)
            if kt < 0:
                continue
            g = spring(kt, 34, 0.32)
            rise = spring(t - bt(3.95) - 0.02 * i, 24, 0.78)
            x = lerp(CX - 800 + 1600 * (i + 2) / 11.0, KX[i], io_cubic(seg(b, 3.95, 4.3)))
            gy = lerp(TL_Y, GY0, io_cubic(seg(b, 3.95, 4.35)))
            y = lerp(gy, KY[i], rise)
            out = 1 - out_cubic(seg(b, 5.18 + 0.012 * i, 5.36 + 0.012 * i))
            s = 16 * g * out
            if s > 0.1:
                d = G.poly([(x, y - s), (x + s, y), (x, y + s), (x - s, y)])
                c.drawPath(d, paint(SIGNAL))
                c.drawPath(d, paint(PAPER, 0.9, stroke=1.5))
            fl = seg(kt, 0, 0.3)
            if 0 < fl < 1:
                c.drawCircle(x, y, 12 + 26 * out_cubic(fl), paint(SIGNAL, (1 - fl) ** 2, stroke=1.5))
            if 2.0 <= tb(t) < 4.0:
                G.text(c, f'K{i + 1:02d}', x, y - 26, 'mono', 11, PAPER, 0.45 * min(1, g))

    # ---- 4. phrases : le point rature les mots, puis devient le point final
    for k, (pb, _) in enumerate(C.PHRASES):
        nb = C.PHRASES[k + 1][0] if k < 3 else 99
        if not (pb <= b < nb):
            continue
        w, x0, base, strike, dpos = phrase_layout(k)
        lt = t - bt(pb)
        s = 1 + 0.08 * (1 - out_expo(seg(lt, 0, 0.16)))
        c.save()
        c.translate(CX, CY)
        c.scale(s, s)
        c.translate(-CX, -CY)
        c.drawPath(w.path(x0, base), paint(PAPER, min(1, lt / 0.03)))
        if k < 3:
            ds = dot_state_s1(t)
            if ds:
                sx0, sx1, sy = strike
                start = sx0 if k != 1 else sx1
                xe = clamp(ds[0], min(sx0, sx1), max(sx0, sx1)) if tb(t) > pb + 0.06 else start
                if abs(xe - start) > 1:
                    c.drawLine(start, sy, xe, sy, paint(SIGNAL, 1, stroke=22, cap='butt'))
        c.restore()
    ds = dot_state_s1(t)
    if ds and b >= 4.45:
        x, y, r, sx, sy, ang = ds
        draw_dot(c, x, y, r, sx, sy, ang)
    c.restore()


# ====================================================================== S2 — TYPOGRAPHY (b8 -> b12)
TYPE_SIZE = 400
TYPE_TRACK = -0.02
WORDS = ['CLAUDE', 'MOTION', 'DESIGN']


def type_layout():
    out = []
    for wd in WORDS:
        w = G.word('black', wd, TYPE_SIZE, TYPE_TRACK)
        x0 = CX - w.width / 2
        out.append((w, x0))
    base = CY + out[0][0].cap / 2
    return out, base


def letter_state(t, i):
    """(index mot ancien, index mot nouveau, progression du flip)"""
    flips = [(bt(10) + 0.03 * i + 0.03, 0, 1), (bt(11) + 0.03 * i + 0.03, 1, 2)]
    dur = 0.30
    cur = 0
    for (s, a, bb) in flips:
        if t < s:
            return cur, cur, 0.0
        if t < s + dur:
            return a, bb, (t - s) / dur
        cur = bb
    return cur, cur, 0.0


def draw_letters(c, t, scheme, bgc, clip_reveal=True, band=None):
    """scheme : couleurs par mot. band : (y0, y1, dx, couleur forcée)"""
    lay, base = type_layout()
    cap = lay[0][0].cap
    yc = base - cap / 2
    b = tb(t)
    for i in range(6):
        a_, n_, p = letter_state(t, i)
        wa, xa = lay[a_]
        wn, xn = lay[n_]
        rise = spring(t - bt(8) - 0.014 * i, 34, 0.5)
        dy = (1 - rise) * (cap + 60) if b < 9 else 0.0
        if p <= 0:
            path = wa.glyph_path(i, xa, base + dy)
            colr = scheme[a_]
            c.drawPath(path, paint(band[3] if band and band[3] else colr))
            continue
        th = math.pi * out_back(p, 1.25)
        xc = lerp(xa + wa.glyphs[i]['cx'], xn + wn.glyphs[i]['cx'], smooth(p))
        if th < math.pi / 2:
            w_, x_, k_, ang = wa, xa, a_, th
        else:
            w_, x_, k_, ang = wn, xn, n_, th - math.pi
        gx = x_ + w_.glyphs[i]['cx']
        path = w_.glyph_path(i, x_ - gx + xc, base)
        path.transform(G.flip_matrix(xc, yc, ang))
        shade = 0.55 * abs(math.sin(ang))
        colr = mix(band[3] if band and band[3] else scheme[k_], bgc, shade)
        c.drawPath(path, paint(colr))


def s2(c, t):
    b = tb(t)
    lay, base = type_layout()
    cap = lay[0][0].cap
    top = base - cap - 14
    bot = base + 14
    wipe = 1150 * out_quint(seg(b, 10.0, 10.7)) if b >= 10.0 else 0.0

    def layer(scheme, bgc, guide_c):
        bg(c, bgc)
        # repères typographiques
        gu = out_cubic(seg(b, 8.25, 8.85))
        if gu > 0:
            for y, lab in ((base, 'BASELINE'), (base - cap, 'CAP HEIGHT')):
                c.drawLine(0, y, W * gu, y, paint(guide_c, 0.35, stroke=1, cap='butt'))
                G.text(c, lab, 52, y - 10, 'mono', 12, guide_c, 0.6 * gu)
            G.text(c, 'Inter Display Black  /  400 px  /  tracking -20', 52, bot + 120, 'mono', 14,
                   guide_c, 0.55 * gu)
        # tranche (b9 -> b9.9)
        su = seg(b, 9.0, 9.32)
        sr = io_expo(seg(b, 9.55, 9.9))
        amt = out_expo(su) - sr
        if 0.001 < amt or (9.0 <= b < 9.9):
            hh = (bot - top) / 3
            offs = [-170, 250, -120]
            for k in range(3):
                y0, y1 = top + k * hh, top + (k + 1) * hh
                c.save()
                c.clipRect(skia.Rect(0, y0 if k else -10, W, y1 if k < 2 else H + 10))
                c.translate(offs[k] * amt, 0)
                draw_letters(c, t, scheme, bgc, band=(y0, y1, 0, PAPER if (k == 1 and amt > 0.15) else None))
                c.restore()
            la = 1 - seg(b, 9.0, 9.6)
            lu = out_expo(seg(b, 9.0, 9.18))
            for k in (1, 2):
                y = top + k * hh
                c.drawLine(CX - 1000 * lu, y, CX + 1000 * lu, y, paint(PAPER, la, stroke=2, cap='butt'))
        else:
            if b < 8.9:
                c.save()
                c.clipRect(skia.Rect(0, -10, W, base + 2))
                draw_letters(c, t, scheme, bgc)
                c.restore()
            else:
                draw_letters(c, t, scheme, bgc)

    layer([INK, INK, INK], SIGNAL, INK)
    if wipe > 0:
        c.save()
        p = skia.Path.Circle(CX, CY, wipe)
        c.clipPath(p, doAntiAlias=True)
        layer([PAPER, PAPER, SIGNAL], INK, PAPER)
        c.restore()
        ra = 1 - seg(b, 10.0, 10.7)
        if ra > 0:
            c.drawCircle(CX, CY, wipe, paint(SIGNAL, ra, stroke=6))


# ====================================================================== S3 — SYSTEMS (b12 -> b16)
NX, NY, CELL = 16, 9, 120.0
GRID = [(i, j, 60 + CELL * i, 60 + CELL * j) for j in range(NY) for i in range(NX)]
SPH_R = 400.0


def cell_color(i, j, phase):
    k = (i - j + phase * 3) % 5
    if k == 0:
        return SIGNAL
    if (i + 2 * j + phase) % 9 == 0:
        return COBALT
    return PAPER


def halftone_r(x, y, t, rmax=27.0):
    dx, dy = (x - CX) / SPH_R, (y - CY) / SPH_R
    d2 = dx * dx + dy * dy
    edge = clamp((1.0 - math.sqrt(d2)) * SPH_R / 18.0)
    if edge <= 0:
        return 0.0, 0.0
    nz = math.sqrt(max(0.0, 1 - d2))
    a = lerp(-1.1, 0.5, smooth(seg(tb(t), 15.2, 16.0)))
    lx, ly, lz = math.cos(a) * 0.75, -0.55, 0.6
    n = math.sqrt(lx * lx + ly * ly + lz * lz)
    sh = max(0.0, (dx * lx + dy * ly + nz * lz) / n)
    return rmax * (0.16 + 0.84 * sh ** 0.9) * edge, sh


def s3(c, t, draw_bg=True):
    b = tb(t)
    if draw_bg:
        bg(c, INK)
    # les lettres de DESIGN s'effondrent
    if b < 12.5:
        lay, base = type_layout()
        w, x0 = lay[2]
        for i in range(6):
            u = in_back(seg(t, bt(12) + 0.02 * i, bt(12) + 0.02 * i + 0.16), 2.2)
            s = 1 - u
            if s <= 0.01:
                continue
            gx = x0 + w.glyphs[i]['cx']
            p = w.glyph_path(i, x0, base)
            p.transform(skia.Matrix.MakeAll(s, 0, gx * (1 - s), 0, s, (base - w.cap / 2) * (1 - s), 0, 0, 1))
            c.drawPath(p, paint(SIGNAL))
    sub = seg(t, bt(14.95), bt(14.95) + 0.40)
    for (i, j, x, y) in GRID:
        d = math.hypot(x - CX, y - CY)
        pop = spring(t - bt(12) - 0.05 - d / 1100 * 0.32, 26, 0.42)
        if pop <= 0.001:
            continue
        # vagues de morphing
        d1 = d / 1100 * 0.34
        p1 = out_back(seg(t, bt(13) + d1, bt(13) + d1 + 0.26), 1.4)
        d2 = x / W * 0.40
        p2 = out_back(seg(t, bt(14) + d2, bt(14) + d2 + 0.26), 1.4)
        r = G.SHAPES['circle'] + (G.SHAPES['square'] - G.SHAPES['circle']) * p1
        r = r + (G.SHAPES['plus'] - r) * p2
        rot = math.pi / 2 * (p1 + p2)
        breathe = 1 + 0.12 * math.sin(2 * math.pi * 0.95 * t - d * 0.011) * seg(b, 12.4, 13)
        phase = (p1 > 0.5) + (p2 > 0.5)
        cc = cell_color(i, j, int(phase))
        if b < 14.95 or sub <= 0:
            c.drawPath(G.polar_path(x, y, r, 46 * pop * breathe, rot), paint(cc))
            continue
        # subdivision -> trame demi-teinte d'une sphère
        dsub = d / 1100 * 0.12
        su = seg(t, bt(14.95) + dsub, bt(14.95) + dsub + 0.24)
        sp_ = spring(t - bt(14.95) - dsub, 24, 0.65)
        for (ox, oy) in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            px, py = x + ox * 30 * sp_, y + oy * 30 * sp_
            hr, sh = halftone_r(px, py, t)
            rr = r + (G.SHAPES['circle'] - r) * smooth(su)
            scale = lerp(46 * 0.55, hr, smooth(su))
            if scale < 0.3:
                continue
            cc2 = mix(cc, SIGNAL if sh < 0.3 else PAPER, smooth(su))
            c.drawPath(G.polar_path(px, py, rr, scale, rot * (1 - su)), paint(cc2))


# ====================================================================== S4 — DIMENSION (b16 -> b20)
NR = 18
RING_S = np.linspace(0, 2 * np.pi, 121)
RING_LOCAL = np.stack([np.cos(RING_S), np.sin(RING_S), np.zeros_like(RING_S)], 1)
R_MAJ, R_MIN = 310.0, 150.0


def ring_frames(t):
    """liste de (centre, rayon, matrice d'orientation) pour chaque anneau"""
    out = []
    for k in range(NR):
        fan = spring(t - bt(16) - 0.011 * k, 13, 0.55)
        gam_s = math.pi * k / NR * fan
        tw = io_cubic(seg(t, bt(17) + 0.018 * k, bt(17) + 0.018 * k + 0.42))
        m = io_cubic(seg(t, bt(18) + 0.008 * k, bt(18) + 0.008 * k + 0.36))
        gam = lerp(gam_s, 2 * math.pi * k / NR, m)
        tilt = 0.75 * tw * (1 - m)
        Rm = G.rot_y(-gam) @ G.rot_x(tilt)
        cen_r = R_MAJ * m
        center = np.array([math.cos(gam) * cen_r, 0.0, math.sin(gam) * cen_r])
        rad = lerp(SPH_R, R_MIN, m)
        out.append((center, rad, Rm))
    return out


def world_rot(t):
    b = tb(t)
    spin = (0.55 * (t - bt(16)) + 0.9 * out_cubic(seg(b, 16.0, 17.0))
            + 0.6 * io_cubic(seg(b, 18.4, 19.3)) + 2.2 * in_cubic(seg(b, 19.3, 20.0)))
    alpha = lerp(0.0, 0.42, out_cubic(seg(b, 16.0, 16.8)))
    alpha = lerp(alpha, 0.95, io_cubic(seg(b, 18.5, 19.4)))
    return G.rot_x(-alpha) @ G.rot_y(spin)


def cam_dist(t):
    return lerp(1600.0, 1150.0, io_cubic(seg(tb(t), 18.8, 20.0)))


def ring_points(t, cx=CX, cy=CY, scale=1.0):
    Wr = world_rot(t)
    cam = cam_dist(t)
    out = []
    for k, (cen, rad, Rm) in enumerate(ring_frames(t)):
        P = ((RING_LOCAL * rad) @ Rm.T + cen) @ Wr.T
        x, y, z, kk = G.project(P, cam, cx, cy, f=1600.0 * scale)
        out.append((k, x, y, z, kk))
    return out, cam


def draw_rings(c, t, cx=CX, cy=CY, scale=1.0, colors=(PAPER, SIGNAL), alpha=1.0, wmul=1.0, dissolve=True):
    b = tb(t)
    segs, cam = ring_points(t, cx, cy, scale)
    du = smooth(seg(b, 19.35, 19.95)) if dissolve else 0.0
    for k, x, y, z, kk in segs:
        colr = colors[1] if k % 3 == 0 else colors[0]
        zm = (z[:-1] + z[1:]) / 2
        front = np.clip((cam + 420 - zm) / 840, 0, 1)
        bucket = np.minimum((front * 4).astype(int), 3)
        valid = (z[:-1] > 40) & (z[1:] > 40)
        for q in range(4):
            sel = np.nonzero((bucket == q) & valid)[0]
            if len(sel) == 0:
                continue
            path = skia.Path()
            prev = -2
            for s_ in sel:
                if s_ != prev + 1:
                    path.moveTo(float(x[s_]), float(y[s_]))
                path.lineTo(float(x[s_ + 1]), float(y[s_ + 1]))
                prev = s_
            fr = (q + 0.5) / 4
            wdt = (1.4 + 3.6 * fr) * float(np.mean(kk[sel])) * scale * wmul
            p = paint(colr, alpha * (0.25 + 0.75 * fr), stroke=max(0.7, wdt * (1 + 0.6 * du)))
            if du > 0:
                on = lerp(400.0, 0.01, du ** 0.5)
                off = lerp(0.0, 13.0, du)
                p.setPathEffect(skia.DashPathEffect.Make([on, off + 0.01], 0))
            c.drawPath(path, p)
    return segs


def s4(c, t):
    b = tb(t)
    bg(c, COBALT)
    # la trame demi-teinte s'efface sur la coupe
    hf = 1 - seg(t, bt(16), bt(16) + 0.14)
    if hf > 0:
        c.saveLayerAlpha(None, int(255 * hf))
        s3(c, bt(16) - 0.001, draw_bg=False)
        c.restore()
    segs = draw_rings(c, t)
    # électrons sur trois anneaux
    for k, ph in ((2, 0.0), (8, 2.1), (14, 4.2)):
        _, x, y, z, kk = segs[k]
        u = (t * 0.9 + ph / 6.28) % 1.0
        idx = int(u * 120)
        if z[idx] > 40:
            c.drawCircle(float(x[idx]), float(y[idx]), 7.5 * float(kk[idx]), paint(SIGNAL if k != 8 else PAPER))
    # repère de suivi : l'étiquette reste fixe, seule la ligne de rappel suit le point
    ca = seg(b, 16.6, 16.9) * (1 - seg(b, 19.1, 19.35))
    if ca > 0:
        k = 4
        _, x, y, z, kk = segs[k]
        idx = 12
        px, py = float(x[idx]), float(y[idx])
        lx, ly = 1420.0, 300.0
        c.drawCircle(px, py, 7, paint(PAPER, ca, stroke=2))
        c.drawLine(px, py, lx, ly, paint(PAPER, 0.6 * ca, stroke=1.2))
        c.drawLine(lx, ly, lx + 330, ly, paint(PAPER, 0.6 * ca, stroke=1.2))
        G.text(c, f'RING {k:02d}   z {z[idx] - 1600:+07.1f}', lx + 6, ly - 12, 'mono', 14, PAPER, ca)
        G.text(c, f'x {px - CX:+07.1f}   y {CY - py:+07.1f}', lx + 6, ly + 26, 'mono', 14, PAPER, 0.75 * ca)
    lab = seg(b, 16.4, 16.7) * (1 - seg(b, 19.3, 19.5))
    if lab > 0:
        m = io_cubic(seg(b, 18.0, 18.5))
        G.text(c, 'SPHERE  ->  TORUS' if m < 0.5 else f'TORUS  /  R {R_MAJ:.0f}  r {R_MIN:.0f}', 56, 150, 'mono', 14, PAPER, 0.7 * lab)
        G.text(c, f'{NR} rings  x  120 vertices  /  perspective f=1600', 56, 174, 'mono', 14, PAPER, 0.45 * lab)


# ====================================================================== S5 — PARTICLES (b20 -> b24)
NP = 2400
P_DT = 1 / 480
P_HIST = (0.0, 0.025, 0.05, 0.08, 0.115)
_SIM = {}


_CURL_MODES = []
_r = np.random.default_rng(11)
for _k in range(5):
    _ang = _r.uniform(0, 2 * np.pi)
    _lam = _r.uniform(650, 1500)
    _CURL_MODES.append((math.cos(_ang) * 2 * math.pi / _lam, math.sin(_ang) * 2 * math.pi / _lam,
                        _r.uniform(-1.2, 1.2), _r.uniform(0, 6.28), _lam))


def curl(p, t):
    x, y = p[:, 0], p[:, 1]
    vx = np.zeros_like(x)
    vy = np.zeros_like(y)
    for kx, ky, om, ph, lam in _CURL_MODES:
        cs = np.cos(kx * x + ky * y + om * t + ph)
        vx += cs * ky * lam / (2 * math.pi)
        vy -= cs * kx * lam / (2 * math.pi)
    return np.stack([vx, vy], 1) / 2.2


def particles():
    if 'pos' in _SIM:
        return _SIM
    rng = np.random.default_rng(5)
    t0, t1 = bt(20) - 0.2, bt(24) + 0.05
    steps = int((t1 - t0) / P_DT) + 2
    # naissance : les points des anneaux à la fin de la scène 3D
    segs, _ = ring_points(bt(20))
    src, cols = [], []
    for k, x, y, z, kk in segs:
        for j in range(120):
            src.append((x[j], y[j]))
            cols.append(1 if k % 3 == 0 else 0)
    src = np.array(src)
    cols = np.array(cols)
    pick = rng.integers(0, len(src), NP)
    p = src[pick] + rng.normal(0, 4, (NP, 2))
    col_ = cols[pick].copy()
    col_[rng.random(NP) < 0.12] = 2
    cen = np.array([CX, CY])
    r = p - cen
    dist = np.linalg.norm(r, axis=1, keepdims=True) + 1e-3
    tang = np.concatenate([-r[:, 1:2], r[:, 0:1]], 1) / dist
    sp = rng.uniform(250, 1500, NP)[:, None] * rng.random((NP, 1)) ** 0.5
    v = r / dist * sp + tang * rng.uniform(200, 700, (NP, 1))
    pos = np.zeros((steps, NP, 2), np.float32)
    for s_ in range(steps):
        t = t0 + s_ * P_DT
        b = tb(t)
        pos[s_] = p
        if t < bt(20):
            continue
        wf = smooth(seg(b, 20.25, 20.9)) * (1 - smooth(seg(b, 22.0, 22.8)))
        wv = smooth(seg(b, 22.0, 22.8))
        tgt = np.zeros_like(p)
        if wf > 0:
            tgt += curl(p, t) * 720 * wf
        if wv > 0:
            r = p - cen
            dist = np.linalg.norm(r, axis=1, keepdims=True) + 1e-3
            tang = np.concatenate([-r[:, 1:2], r[:, 0:1]], 1) / dist
            tgt += (tang * 1150 * (dist / 380) ** -0.35 - r / dist * 300) * wv
        kdrag = 2.6 if b < 22 else 5.0
        v += (tgt - v) * (1 - math.exp(-P_DT * kdrag))
        p = p + v * P_DT
    _SIM['pos'] = pos
    _SIM['t0'] = t0
    _SIM['col'] = col_
    _SIM['size'] = rng.uniform(1.6, 3.8, NP)
    return _SIM


def particles_at(t):
    S = particles()
    f = (t - S['t0']) / P_DT
    f = clamp(f, 0, S['pos'].shape[0] - 1.001)
    i = int(f)
    a = f - i
    P = S['pos'][i] * (1 - a) + S['pos'][i + 1] * a
    imp = in_expo(seg(tb(t), 23.45, 24.0))
    if imp > 0:
        P = P + (np.array([CX, CY]) - P) * imp
    return P


def draw_particles(c, t, cols=(PAPER, SIGNAL, COBALT), ox=0.0, oy=0.0, sc=1.0, trail=True, amul=1.0):
    S = particles()
    hist = [particles_at(t - h) for h in (P_HIST if trail else (0.0,))]
    P = hist[0]
    b = tb(t)
    grow = 0.7 + 0.3 * out_cubic(seg(b, 20.0, 20.3))
    tr_a = 1 - seg(b, 23.6, 23.95)
    for ci, colr in enumerate(cols):
        sel = S['col'] == ci
        if trail and tr_a > 0:
            for h in range(len(hist) - 1):
                A, Bq = hist[h + 1][sel], hist[h][sel]
                lp = []
                for (x0, y0), (x1, y1) in zip(A, Bq):
                    lp.append(skia.Point(float(x0 * sc + ox), float(y0 * sc + oy)))
                    lp.append(skia.Point(float(x1 * sc + ox), float(y1 * sc + oy)))
                a_ = (0.55, 0.36, 0.22, 0.12)[h] * amul * tr_a
                c.drawPoints(skia.Canvas.kLines_PointMode, lp,
                             paint(colr, a_, stroke=max(0.8, (2.4 - 0.4 * h) * sc), cap='butt'))
        for (lo, hi) in ((0, 2.4), (2.4, 3.1), (3.1, 9)):
            ss = sel & (S['size'] >= lo) & (S['size'] < hi)
            if not np.any(ss):
                continue
            pts = [skia.Point(float(x * sc + ox), float(y * sc + oy)) for x, y in P[ss]]
            sz = (max(lo, 1.6) + min(hi, 3.8)) / 2 * 1.6 * grow * sc
            pnt = paint(colr, amul)
            pnt.setStrokeWidth(sz)
            pnt.setStrokeCap(skia.Paint.kRound_Cap)
            c.drawPoints(skia.Canvas.kPoints_PointMode, pts, pnt)


def s5(c, t):
    b = tb(t)
    bg(c, INK)
    # éclair d'entrée (les anneaux se dissolvent en particules)
    fl = 1 - seg(t, bt(20), bt(20) + 0.2)
    if fl > 0:
        c.drawCircle(CX, CY, 300 + 900 * (1 - fl), paint(PAPER, 0.35 * fl ** 2, stroke=36 * fl + 1))
    draw_particles(c, t)
    # point d'implosion
    pu = seg(b, 23.75, 24.0)
    if pu > 0:
        draw_dot(c, CX, CY, DOT_R * out_back(pu, 2.0))
    la = seg(b, 20.5, 20.8) * (1 - seg(b, 23.3, 23.6))
    if la > 0:
        G.text(c, f'n = {NP}   /   curl noise   /   dt = 1/480 s', 56, 150, 'mono', 14, PAPER, 0.6 * la)
        mode = 'FLOW FIELD' if b < 22.4 else 'VORTEX'
        G.text(c, f'mode: {mode}', 56, 174, 'mono', 14, SIGNAL, 0.8 * la)


# ====================================================================== S6 — RHYTHM (b24 -> b28)
MONT_SIZE = 440
_AUDIO = {}


def audio():
    if 'x' not in _AUDIO:
        p = os.environ.get('REEL_AUDIO_NPY')
        _AUDIO['x'] = np.load(p) if p and os.path.exists(p) else np.zeros(C.SR * 16, np.float32)
    return _AUDIO['x']


def _code_lines():
    src = open(os.path.join(HERE, 'scenes.py')).read().splitlines()
    src += open(os.path.join(HERE, 'music.py')).read().splitlines()
    return [ln[:70] for ln in src if ln.strip()]


_CODE = _code_lines()
_CODE_ROWS = []
for _i in range(0, len(_CODE) - 1, 2):
    _CODE_ROWS.append((_CODE[_i] + '    ' + _CODE[_i + 1].strip())[:118])


def tex_type(c, t, rect):
    c.drawColor(col(SIGNAL))
    base = 'KINETIC TYPE / TRACKING / KERNING / BASELINE / '
    for r in range(30):
        y = rect.top() + 30 + r * 34
        if y > rect.bottom() + 40:
            break
        sp = 900 * (1 if r % 2 else -1)
        x = rect.left() - 600 + (t * sp + r * 137) % 600
        G.text(c, (base * 3), x, y, 'monob', 26, INK, 1.0)


def tex_shape(c, t, rect):
    c.drawColor(col(PAPER))
    cell = 46
    shapes = ['circle', 'square', 'plus', 'triangle', 'diamond']
    for j in range(int(rect.top() // cell), int(rect.bottom() // cell) + 2):
        for i in range(int(rect.left() // cell), int(rect.right() // cell) + 2):
            x, y = i * cell + cell / 2, j * cell + cell / 2
            ph = (t * 4 + (i + j) * 0.15) % len(shapes)
            k = int(ph)
            u = smooth((ph - k) * 2.5)
            r = G.SHAPES[shapes[k]] + (G.SHAPES[shapes[(k + 1) % 5]] - G.SHAPES[shapes[k]]) * u
            c.drawPath(G.polar_path(x, y, r, 16, u * math.pi / 2), paint(INK if (i + j) % 4 else SIGNAL))


def tex_space(c, t, rect):
    c.drawColor(col(COBALT))
    draw_rings(c, bt(17.35) + (t - bt(25)) * 1.2, CX, CY, 0.72, wmul=2.2, dissolve=False)


def tex_flow(c, t, rect):
    c.drawColor(col(PAPER))
    draw_particles(c, bt(21.2) + (t - bt(25.5)) * 1.5, cols=(INK, SIGNAL, COBALT), trail=True)


def tex_sound(c, t, rect):
    c.drawColor(col(mix(INK, COBALT, 0.22)))
    x = audio()
    i0 = int(t * C.SR)
    n = 4096
    seg_ = x[max(0, i0 - n // 2): i0 + n // 2]
    if len(seg_) < n:
        seg_ = np.pad(seg_, (0, n - len(seg_)))
    F = np.abs(np.fft.rfft(seg_ * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / C.SR)
    edges = np.geomspace(35, 14000, 49)
    nb = 48
    bw = (rect.width() + 80) / nb
    for k in range(nb):
        m = (f >= edges[k]) & (f < edges[k + 1])
        v = 20 * np.log10(F[m].mean() + 1e-6) if np.any(m) else -80
        hgt = clamp((v + 10) / 55) * rect.height() * 1.1
        xx = rect.left() - 40 + k * bw
        c.drawRect(skia.Rect(xx + 3, rect.bottom() - hgt, xx + bw - 3, rect.bottom() + 20), paint(SIGNAL))
    # forme d'onde réelle
    wv = x[max(0, i0 - 2400): i0 + 2400]
    if len(wv) > 10:
        pts = []
        for k in range(0, len(wv), 12):
            pts.append((rect.left() - 40 + (rect.width() + 80) * k / len(wv), CY + wv[k] * 330))
        c.drawPath(G.poly(pts, False), paint(PAPER, 1.0, stroke=4))


def tex_code(c, t, rect):
    c.drawColor(col((22, 22, 27)))
    lh = 30
    off = (t - bt(26.5)) * 520
    first = int(off // lh)
    for r in range(48):
        ln = _CODE_ROWS[(first + r) % len(_CODE_ROWS)]
        y = rect.top() - (off % lh) + r * lh
        if y > rect.bottom() + lh:
            break
        s_ = ln.strip()
        colr = (GREY if s_.startswith('#') or s_.startswith('"""') else
                SIGNAL if s_.startswith(('def ', 'class ', 'for ', 'if ', 'return')) else PAPER)
        G.text(c, f'{(first + r) % 999 + 1:3d}  ' + ln, rect.left() - 60, y, 'mono', 22, colr, 0.95)


TEX = {'TYPE': tex_type, 'SHAPE': tex_shape, 'SPACE': tex_space, 'FLOW': tex_flow,
       'SOUND': tex_sound, 'CODE': tex_code}


def mont_word(txt):
    w = G.word('black', txt, MONT_SIZE, -0.03)
    x0 = CX - w.width / 2
    base = CY + w.cap / 2
    return w, x0, base


def s6(c, t):
    b = tb(t)
    bg(c, INK)
    cur = None
    for k, (mb, txt) in enumerate(C.MONTAGE):
        if mb <= b < mb + 0.5:
            cur = (mb, txt)
    if cur and b < 27.0:
        mb, txt = cur
        w, x0, base = mont_word(txt)
        lt = t - bt(mb)
        s = 1 + 0.07 * (1 - out_expo(seg(lt, 0, 0.14)))
        c.save()
        c.translate(CX, CY)
        c.scale(s, s)
        c.translate(-CX, -CY)
        path = w.path(x0, base)
        c.save()
        c.clipPath(path, doAntiAlias=True)
        rect = path.computeTightBounds()
        TEX[txt](c, t, rect)
        c.restore()
        c.drawPath(path, paint(PAPER, 0.9, stroke=2.5))
        G.text(c, f'{C.MONTAGE.index(cur) + 1:02d}', x0, base + 70, 'mono', 18, SIGNAL, 1)
        G.text(c, txt.lower(), x0 + w.width, base + 70, 'mono', 18, PAPER, 0.6, 'right')
        c.restore()
    # bégaiement : CODE se replie en trois temps vers le point
    if 27.0 <= b < 27.75:
        w, x0, base = mont_word('CODE')
        n = int((b - 27.0) / 0.25) + 1
        for k in range(n):
            sk = 0.62 ** k
            lt = t - bt(27.0 + 0.25 * k)
            pop = 1 + 0.12 * (1 - out_expo(seg(lt, 0, 0.1)))
            a = 1.0 if k == n - 1 else 0.35
            c.save()
            c.translate(CX, CY)
            c.scale(sk * pop, sk * pop)
            c.translate(-CX, -CY)
            c.drawPath(w.path(x0, base), paint(SIGNAL if k % 2 else PAPER, a, stroke=3.0 / sk))
            c.restore()
    if b >= 27.75:
        lt = t - bt(27.75)
        tremble = 1.5 * math.sin(lt * 90) * seg(b, 27.8, 28.0)
        draw_dot(c, CX + tremble, CY, DOT_R * spring(lt, 30, 0.4), 1 - 0.1 * seg(b, 27.85, 28.0), 1 + 0.1 * seg(b, 27.85, 28.0))


# ====================================================================== S7 — SIGNATURE (b28 -> b32)
SIG_SIZE = 320


def sig_layout():
    w = G.word('black', 'Claude', SIG_SIZE, -0.035)
    wp = G.word('black', 'Claude.', SIG_SIZE, -0.035)
    pb = wp.glyphs[-1]['bounds']
    px = wp.glyphs[-1]['x']
    dot_r = 0.5 * pb.width()
    total = px + pb.right()
    x0 = CX - total / 2
    base = CY + 40
    dot = (x0 + px + (pb.left() + pb.right()) / 2, base - dot_r)
    return w, x0, base, dot, dot_r


G_FALL = 6400.0


def sig_dot(t):
    """le point final : décolle, retombe, rebondit deux fois (vraie balistique)"""
    w, x0, base, (dx, dy), r = sig_layout()
    b = tb(t)
    t0 = bt(28)
    i1, i2, i3 = (bt(x) for x in C.DOT_BOUNCES)
    if t < t0 + 0.32:
        u = seg(t, t0, t0 + 0.32)
        e = out_cubic(u)
        return lerp(CX, dx, e), lerp(CY, -120, in_cubic(u) ** 0.8), lerp(DOT_R, r, e), 0.8, 1.5
    if t < i1:
        T = i1 - t
        y = dy - 0.5 * G_FALL * T * T
        v = G_FALL * T
        st = 1 + min(0.45, v / 9000)
        return dx, y, r, 1 / st ** 0.5, st
    sq = 0.0
    for (ta, tbn, amp) in ((i1, i2, 1.0), (i2, i3, 0.5), (i3, i3 + 0.06, 0.25)):
        if ta <= t < tbn:
            T = tbn - ta
            u = t - ta
            v0 = G_FALL * T / 2
            y = dy - (v0 * u - 0.5 * G_FALL * u * u)
            sq = amp * math.exp(-u / 0.035)
            return dx, y, r, 1 + 0.38 * sq, 1 - 0.32 * sq
    u = t - i3 - 0.06
    sq = 0.08 * math.exp(-u / 0.08) * math.cos(u * 30)
    return dx, dy, r, 1 + sq, 1 - sq


def s7(c, t):
    b = tb(t)
    bg(c, PAPER)
    c.save()
    pz = 1 + 0.028 * out_cubic(seg(b, 28.0, 32.0))
    c.translate(CX, CY)
    c.scale(pz, pz)
    c.translate(-CX, -CY)
    # onde de choc de l'accord final
    su = seg(t, bt(28), bt(28) + 0.7)
    if 0 < su < 1:
        c.drawCircle(CX, CY, 30 + 1300 * out_cubic(su), paint(SIGNAL, (1 - su) ** 1.5, stroke=2 + 10 * (1 - su)))
    w, x0, base, (dx, dy), r = sig_layout()
    c.save()
    c.clipRect(skia.Rect(0, 0, W, base + 3))
    for i in range(len(w.glyphs)):
        rise = spring(t - bt(28) - 0.035 * i - 0.02, 19, 0.62)
        oy = (1 - rise) * (w.cap + 80)
        c.drawPath(w.glyph_path(i, x0, base + oy), paint(INK))
    c.restore()
    # filet
    ru = out_expo(seg(b, 28.9, 29.7))
    rule_y = base + 46
    if ru > 0:
        c.drawLine(x0, rule_y, x0 + (dx + r - x0) * ru, rule_y, paint(INK, 1, stroke=2, cap='butt'))
    # sous-titres
    for (bb, txt, fn, sz, xx, al, a) in ((28.75, 'Motion Designer', 'medium', 54, x0, 'left', 0.92),
                                         (29.25, 'SHOWREEL  /  2026', 'mono', 18, dx + r, 'right', 0.6)):
        u = spring(t - bt(bb), 18, 0.7)
        if u <= 0:
            continue
        ww = G.word(fn, txt, sz)
        yb = rule_y + 30 + ww.cap
        c.save()
        c.clipRect(skia.Rect(0, rule_y + 4, W, yb + 16))
        G.text(c, txt, xx, yb + (1 - u) * 80, fn, sz, INK, a, al)
        c.restore()
    # ligne de pied : machine à écrire
    foot = 'every frame and every sound of this film was generated by code  /  no footage  /  no samples  /  no presets'
    fu = seg(b, 30.0, 31.2)
    if fu > 0:
        n = int(len(foot) * fu)
        fw = G.word('mono', foot, 15)
        fx = CX - fw.width / 2
        G.text(c, foot[:n], fx, 940, 'mono', 15, GREY, 1.0)
        if fu < 1 or (int(t * 3) % 2 == 0):
            cw = fw.width / len(foot)
            c.drawRect(skia.Rect(fx + n * cw + 2, 926, fx + n * cw + 11, 944), paint(SIGNAL))
    # le point
    x, y, rr, sx, sy = sig_dot(t)
    draw_dot(c, x, y, rr, sx, sy)
    # ondulation à l'impact
    for (bb, amp) in zip(C.DOT_BOUNCES, (1.0, 0.5, 0.25)):
        u = seg(t, bt(bb), bt(bb) + 0.5)
        if 0 < u < 1:
            c.drawCircle(dx, dy + r * 0.6, r + 90 * amp * out_cubic(u), paint(SIGNAL, (1 - u) ** 2 * amp, stroke=2))
    c.restore()


SCENE_FN = [s1, s2, s3, s4, s5, s6, s7]


def scene_index(t):
    b = tb(t)
    for k, (b0, b1, _, _) in enumerate(C.SCENES):
        if b < b1:
            return k
    return len(C.SCENES) - 1
