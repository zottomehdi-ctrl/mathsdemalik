"""Rendu : flou de bouger par sous-images, secousses caméra, aberration chromatique,
HUD, grain, vignettage, puis encodage H.264 + AAC via ffmpeg.

usage :
  python3 render.py stills OUT_DIR t1 t2 ...      (secondes)
  python3 render.py sheet OUT.png [n]             (planche contact)
  python3 render.py video OUT.mp4 AUDIO.wav [first last]
"""
import math
import os
import subprocess
import sys
import time

import numpy as np
import skia

import common as C
import gfx as G
import scenes as S
from common import W, H, FPS, INK, PAPER, SIGNAL, bt, tb, seg
from gfx import paint

INFO = skia.ImageInfo.Make(W, H, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
SHUTTER = 0.5            # obturateur 180°
SCRAMBLE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/<>#*+'


def samples_for(t):
    b = tb(t)
    if 8.0 <= b < 8.6 or 28.0 <= b < 28.7:
        return 16
    if 7.55 <= b < 8.0 or 19.4 <= b < 20.4:
        return 12
    if 8.6 <= b < 12.0 or 20.4 <= b < 21.0:
        return 9
    return 7


def bloom_for(t):
    """(intensité, seuil de luminance) du halo par scène"""
    b = tb(t)
    if 16.0 <= b < 20.0:
        return 0.55, 0.55
    if 20.0 <= b < 24.0:
        return 0.85, 0.22
    return 0.0, 1.0


def camera(t):
    dx = dy = 0.0
    zoom = 1.0
    for hb, s in C.HITS:
        u = t - bt(hb)
        if 0 <= u < 0.7:
            e = s * math.exp(-u / 0.10)
            dx += e * 16 * C.noise1(t * 34, hb)
            dy += e * 16 * C.noise1(t * 34, hb + 50)
            zoom += s * 0.05 * math.exp(-u / 0.075)
    return dx, dy, zoom


def ca_amount(t):
    a = 0.0
    for hb, s in C.HITS:
        u = t - bt(hb)
        if 0 <= u < 0.5:
            a += s * 7.0 * math.exp(-u / 0.06)
    return a


def hud_color(t):
    b = tb(t)
    if 8.0 <= b < 10.32:
        return INK
    if b >= 28.0:
        return INK
    return PAPER


def scramble(s, lt, i):
    if lt >= 0.34:
        return s
    out = []
    n = len(s)
    for j, ch in enumerate(s):
        if ch == ' ' or lt > 0.34 * j / n:
            out.append(ch)
        else:
            h = C.hash11(i * 13.1 + j * 7.7)
            out.append(SCRAMBLE[int(h * len(SCRAMBLE)) % len(SCRAMBLE)])
    return ''.join(out)


def hud(c, t, i):
    b = tb(t)
    a = 0.72 * seg(b, 0.2, 1.0)
    if a <= 0:
        return
    colr = hud_color(t)
    m = 30
    for (x, y, sx, sy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
        c.drawLine(x, y, x + 22 * sx, y, paint(colr, a, stroke=2, cap='butt'))
        c.drawLine(x, y, x, y + 22 * sy, paint(colr, a, stroke=2, cap='butt'))
    G.text(c, 'CLAUDE  /  MOTION REEL  /  2026', 56, 70, 'mono', 15, colr, a)
    f, s = i % FPS, i // FPS
    G.text(c, f'TC 00:00:{s:02d}:{f:02d}', W - 56, 70, 'mono', 15, colr, a, 'right')
    k = S.scene_index(t)
    b0, _, num, name = C.SCENES[k]
    lt = t - bt(b0)
    G.text(c, scramble(f'{num} / 07   {name}', lt, i), 56, H - 58, 'mono', 15, colr, a)
    beat = int(math.floor(b)) % 4
    x_end = W - 56
    for q in range(4):
        x = x_end - (3 - q) * 18 - 10
        on = (q == beat)
        c.drawRect(skia.Rect(x, H - 69, x + 10, H - 59), paint(colr, a if on else a * 0.3))
    G.text(c, f'{C.BPM} BPM', x_end - 4 * 18 - 8, H - 58, 'mono', 15, colr, a, 'right')
    y = H - 38
    c.drawLine(56, y, W - 56, y, paint(colr, a * 0.25, stroke=1, cap='butt'))
    c.drawLine(56, y, 56 + (W - 112) * min(1, t / C.DUR), y, paint(colr, a, stroke=2, cap='butt'))
    for (sb, _, _, _) in C.SCENES:
        x = 56 + (W - 112) * bt(sb) / C.DUR
        c.drawLine(x, y - 4, x, y + 4, paint(colr, a * 0.6, stroke=1, cap='butt'))


class Renderer:
    def __init__(self):
        self.surf = skia.Surface.MakeRaster(INFO)
        self.c = self.surf.getCanvas()
        self.buf = np.zeros((H, W, 4), np.uint8)
        rng = np.random.default_rng(99)
        from scipy.ndimage import uniform_filter
        gr = uniform_filter(rng.standard_normal((H + 128, W + 128)), 2)
        self.grain = (gr / gr.std() * 1.5).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        self.vig = (1 - 0.08 * r2 ** 1.3).astype(np.float32)[..., None]
        self._cm = {}
        for k in range(3):
            m = [0.0] * 20
            m[k * 5 + k] = 1.0
            m[18] = 1.0
            self._cm[k] = skia.ColorFilters.Matrix(m)

    def scene(self, t, k):
        c = self.c
        c.save()
        dx, dy, z = camera(t)
        c.translate(W / 2 + dx, H / 2 + dy)
        c.scale(z, z)
        c.translate(-W / 2, -H / 2)
        S.SCENE_FN[k](c, t)
        c.restore()

    def chroma(self, amt):
        img = self.surf.makeImageSnapshot()
        c = self.c
        c.clear(skia.Color4f(0, 0, 0, 1))
        for k, s in ((0, 1 + 2 * amt / 960), (1, 1 + amt / 960), (2, 1.0)):
            p = skia.Paint(ColorFilter=self._cm[k], BlendMode=skia.BlendMode.kPlus)
            c.save()
            c.translate(W / 2, H / 2)
            c.scale(s, s)
            c.translate(-W / 2, -H / 2)
            c.drawImage(img, 0, 0, skia.SamplingOptions(skia.FilterMode.kLinear), p)
            c.restore()

    def bloom(self, rgba, amount, thresh):
        """halo : sources lumineuses (seuil sur la luminance), flou à 1/4 de résolution, ajout"""
        from scipy.ndimage import gaussian_filter, zoom
        f = rgba[..., :3].reshape(H // 4, 4, W // 4, 4, 3).mean(axis=(1, 3)) / 255.0
        y = f @ np.array([0.2126, 0.7152, 0.0722])
        m = np.clip((y - thresh) / (1 - thresh), 0, 1)[..., None]
        src = f * m
        g = gaussian_filter(src, (3.2, 3.2, 0)) + 0.5 * gaussian_filter(src, (9, 9, 0))
        up = zoom(g, (4, 4, 1), order=1)
        out = rgba[..., :3].astype(np.float32) + up * (255.0 * amount)
        rgba[..., :3] = np.clip(out, 0, 255).astype(np.uint8)

    def frame(self, i):
        t = i / FPS
        k = S.scene_index(t)
        K = samples_for(t)
        acc = np.zeros((H, W, 3), np.uint16)
        for j in range(K):
            tj = t + ((j + 0.5) / K - 0.5) * SHUTTER / FPS
            self.scene(tj, k)
            self.c.readPixels(INFO, self.buf, W * 4, 0, 0)
            acc += self.buf[..., :3]
        rgba = np.empty((H, W, 4), np.uint8)
        rgba[..., :3] = ((acc + K // 2) // K).astype(np.uint8)
        rgba[..., 3] = 255
        bl, th = bloom_for(t)
        if bl > 0:
            self.bloom(rgba, bl, th)
        self.c.writePixels(INFO, rgba, W * 4, 0, 0)
        amt = ca_amount(t)
        if amt > 0.35:
            self.chroma(amt)
        hud(self.c, t, i)
        self.c.readPixels(INFO, self.buf, W * 4, 0, 0)
        img = self.buf[..., :3].astype(np.float32)
        ox = int(C.hash11(i * 3.3) * 127)
        oy = int(C.hash11(i * 5.9 + 1) * 127)
        img = img * self.vig + self.grain[oy:oy + H, ox:ox + W, None]
        return np.clip(img + 0.5, 0, 255).astype(np.uint8)


_R = None


def _init():
    global _R
    _R = Renderer()


def _work(i):
    return _R.frame(i).tobytes()


def stills(out_dir, times):
    os.makedirs(out_dir, exist_ok=True)
    r = Renderer()
    from PIL import Image
    for tt in times:
        i = int(round(float(tt) * FPS))
        t0 = time.time()
        img = r.frame(i)
        Image.fromarray(img).save(os.path.join(out_dir, f'f{i:04d}.png'))
        print(f'frame {i} ({float(tt):.3f}s, b{tb(i / FPS):.2f}) {time.time() - t0:.2f}s', flush=True)


def sheet(out, frames, cols=6, tw=480):
    import multiprocessing as mp
    from PIL import Image, ImageDraw
    S.particles()
    ctx = mp.get_context('fork')
    with ctx.Pool(4, initializer=_init) as pool:
        imgs = pool.map(_work, frames, chunksize=1)
    th = int(tw * H / W)
    rows = (len(frames) + cols - 1) // cols
    sh = Image.new('RGB', (cols * tw, rows * (th + 18)), (40, 40, 40))
    d = ImageDraw.Draw(sh)
    for n, (f, raw) in enumerate(zip(frames, imgs)):
        im = Image.frombytes('RGB', (W, H), raw).resize((tw, th), Image.LANCZOS)
        x, y = (n % cols) * tw, (n // cols) * (th + 18)
        sh.paste(im, (x, y))
        d.text((x + 4, y + th + 2), f'f{f} {f / FPS:.2f}s b{tb(f / FPS):.2f}', fill=(230, 230, 230))
    sh.save(out)


def video(out, audio, first=0, last=C.NFRAMES):
    import multiprocessing as mp
    S.particles()
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-']
    if audio:
        cmd += ['-ss', f'{first / FPS:.6f}', '-i', audio]
    cmd += ['-vf', 'scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p',
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-profile:v', 'high', '-level', '4.2',
            '-x264-params', 'aq-mode=3:colorprim=bt709:transfer=bt709:colormatrix=bt709',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv']
    if audio:
        cmd += ['-c:a', 'aac', '-b:a', '320k', '-ar', '48000', '-shortest']
    cmd += ['-movflags', '+faststart', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    ctx = mp.get_context('fork')
    t0 = time.time()
    with ctx.Pool(4, initializer=_init) as pool:
        for n, raw in enumerate(pool.imap(_work, range(first, last), chunksize=2)):
            p.stdin.write(raw)
            if n % 60 == 0:
                el = time.time() - t0
                print(f'  {n + first}/{last}  {el:6.1f}s', flush=True)
    p.stdin.close()
    p.wait()
    print(f'done in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'stills':
        stills(sys.argv[2], sys.argv[3:])
    elif mode == 'sheet':
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 48
        fr = [int(x) for x in np.linspace(0, C.NFRAMES - 1, n)] if len(sys.argv) <= 4 else [int(x) for x in sys.argv[4:]]
        sheet(sys.argv[2], fr)
    elif mode == 'video':
        a = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '-' else None
        f0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        f1 = int(sys.argv[5]) if len(sys.argv) > 5 else C.NFRAMES
        video(sys.argv[2], a, f0, f1)
