#!/usr/bin/env python3
"""
osmo_grade.py  v1.0  -  color pipeline for DJI Osmo Pocket 4P footage (Matteo)

Profiles
  dlog2   main 1x camera, 10-bit D-Log 2 / D-Gamut2 (best: pass DJI's official LUT via --input-lut)
  dlog    3x tele or zoomed main camera, 10-bit D-Log / D-Gamut (exact DJI white-paper math)
  hlg     the camera's 10-bit HDR mode (HLG / Rec.2020) mapped to SDR Rec.709
  normal  Normal / film-tone footage (already Rec.709)

Projects (exposure targets + look)
  fights    @matteofights: dark, punchy, desaturated, blood-red kept, teal shadows
  gypsy     Gypsy Amazon: warm, soft, human, fabric colors kept true
  samui     Premium Investment Samui / Crystal Bay View: bright, airy, clean whites, aqua water
  portrait  modeling / comp card: neutral, flattering, unified skin
  clean     neutral technical grade for anything else

Commands
  analyze CLIP...                      probe + detect profile + solve correction, print JSON
  grade   CLIP... --project P --out DIR  full grade: LUT per clip, render, before/after + scopes report
  luts    --out DIR [--input-lut FILE]    export look LUTs (.cube) for any editor
  scopes  IMAGE                         waveform / parade / vectorscope of a still

Needs: ffmpeg + ffprobe, numpy, Pillow. No network, no secrets, no API keys.
"""
import argparse, json, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------------------- color science
D65 = (0.3127, 0.3290)
PRIMARIES = {
    "rec709": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
    "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
    "dgamut": ((0.71, 0.31), (0.21, 0.88), (0.09, -0.08)),  # DJI D-Log/D-Gamut white paper
}
LUMA = np.array([0.2126, 0.7152, 0.0722])


def npm(prim, white=D65):
    xy = np.array(prim, float)
    P = np.stack([xy[:, 0] / xy[:, 1], np.ones(3), (1 - xy[:, 0] - xy[:, 1]) / xy[:, 1]])
    W = np.array([white[0] / white[1], 1.0, (1 - white[0] - white[1]) / white[1]])
    return P * np.linalg.solve(P, W)


def gamut_matrix(src, dst="rec709"):
    return np.linalg.inv(npm(PRIMARIES[dst])) @ npm(PRIMARIES[src])


# DJI D-Log (white paper): exact
def dlog_to_lin(y):
    y = np.asarray(y, float)
    return np.where(y <= 0.14, (y - 0.0929) / 6.025, (10 ** (3.89616 * y - 2.27752) - 0.0108) / 0.9892)


def lin_to_dlog(x):
    x = np.asarray(x, float)
    return np.where(x <= 0.0078, 6.025 * x + 0.0929,
                    np.log10(np.maximum(x * 0.9892 + 0.0108, 1e-10)) * 0.256663 + 0.584555)


# DJI D-Log 2: FALLBACK fit to Gamut's profiling (black 64/1023, mid grey 312/1023, ~10 stops over grey
# to ~0.94). DJI's official LUT / DLog2_DGamut2 DCTL is the exact transform: always prefer --input-lut.
_DL2_C = 0.0635 / math.log10(2)
_DL2_B = 0.18 / (10 ** ((312 / 1023 - 64 / 1023) / _DL2_C) - 1)
_DL2_D = 64 / 1023 - _DL2_C * math.log10(_DL2_B)


def dlog2_to_lin(y):
    return 10 ** ((np.asarray(y, float) - _DL2_D) / _DL2_C) - _DL2_B


def lin_to_dlog2(x):
    return _DL2_C * np.log10(np.maximum(np.asarray(x, float) + _DL2_B, 1e-10)) + _DL2_D


def hlg_to_lin(e):
    e = np.asarray(e, float)
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    lin = np.where(e <= 0.5, e * e / 3.0, (np.exp((e - c) / a) + b) / 12.0)
    return lin * (0.18 / (0.38 ** 2 / 3))  # HLG reference grey (38%) -> 0.18


PROFILES = {
    "dlog2": dict(decode=dlog2_to_lin, encode=lin_to_dlog2, gamut="dgamut"),
    "dlog": dict(decode=dlog_to_lin, encode=lin_to_dlog, gamut="dgamut"),
    "hlg": dict(decode=hlg_to_lin, encode=None, gamut="rec2020"),
    "normal": dict(decode=None, encode=None, gamut="rec709"),
}


def _filmic(v):  # ACES-fit style display curve: 0.18 scene -> ~0.44 display, ~8 stops of shoulder
    v = np.maximum(v, 0) * 0.6
    return np.clip((v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14), 0, 1)


def tonemap(lin):
    per = _filmic(lin)
    L = lin @ LUMA
    ratio = lin * (_filmic(L) / np.maximum(L, 1e-9))[..., None]
    ratio = ratio / np.clip(ratio.max(-1, keepdims=True), 1, None)
    return np.clip(0.6 * per + 0.4 * ratio, 0, 1) ** (1 / 2.4)


def gamut_compress(rgb, thr=0.8):
    ach = rgb.max(-1, keepdims=True)
    pos = ach > 1e-9
    safe = np.where(pos, ach, 1.0)
    d = np.where(pos, (ach - rgb) / safe, 0.0)
    over = np.maximum(d - thr, 0)
    dc = np.where(d > thr, thr + over / (1 + over / (1 - thr)), d)
    return np.where(pos, ach - dc * safe, rgb)


def soft_shoulder(x, k=0.85):
    over = np.maximum(x - k, 0)
    return np.where(x > k, k + over / (1 + over / (1 - k)), x)


def ycbcr(rgb):
    Y = rgb @ LUMA
    return Y, (rgb[..., 2] - Y) / 1.8556, (rgb[..., 0] - Y) / 1.5748


def from_ycbcr(Y, cb, cr):
    r = Y + 1.5748 * cr
    b = Y + 1.8556 * cb
    return np.stack([r, (Y - 0.2126 * r - 0.0722 * b) / 0.7152, b], -1)


SKIN_LINE = 123.0  # degrees on a BT.709 vectorscope


def skin_weight(rgb):
    Y, cb, cr = ycbcr(rgb)
    ang = np.degrees(np.arctan2(cr, cb))
    ch = np.hypot(cb, cr)
    return (np.clip(1 - np.abs(ang - SKIN_LINE) / 22, 0, 1) * np.clip((ch - 0.025) / 0.02, 0, 1)
            * np.clip((Y - 0.08) / 0.1, 0, 1) * np.clip((0.97 - Y) / 0.1, 0, 1))


def rgb_hue(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn + 1e-12
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h


# ----------------------------------------------------------------- .cube io
class Cube:
    def __init__(self, path):
        size, dmin, dmax, vals = None, np.zeros(3), np.ones(3), []
        for line in open(path, errors="ignore"):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            u = s.upper()
            if u.startswith("LUT_3D_SIZE"):
                size = int(s.split()[1])
            elif u.startswith("LUT_1D_SIZE"):
                raise ValueError("1D LUTs are not supported, need a 3D .cube")
            elif u.startswith("DOMAIN_MIN"):
                dmin = np.array(s.split()[1:4], float)
            elif u.startswith("DOMAIN_MAX"):
                dmax = np.array(s.split()[1:4], float)
            elif u[0].isalpha():
                continue
            else:
                vals.append([float(t) for t in s.split()[:3]])
        t = np.array(vals, float)
        if not size or len(t) != size ** 3:
            raise ValueError(f"bad .cube: {path}")
        self.n, self.t, self.dmin, self.dmax, self.path = size, t.reshape(size, size, size, 3), dmin, dmax, path

    def __call__(self, rgb):
        n, T = self.n, self.t
        x = np.clip((rgb - self.dmin) / (self.dmax - self.dmin), 0, 1) * (n - 1)
        i = np.minimum(np.floor(x).astype(int), n - 2)
        f = x - i
        r, g, b = i[..., 0], i[..., 1], i[..., 2]
        fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
        c00 = T[b, g, r] * (1 - fr) + T[b, g, r + 1] * fr
        c01 = T[b, g + 1, r] * (1 - fr) + T[b, g + 1, r + 1] * fr
        c10 = T[b + 1, g, r] * (1 - fr) + T[b + 1, g, r + 1] * fr
        c11 = T[b + 1, g + 1, r] * (1 - fr) + T[b + 1, g + 1, r + 1] * fr
        return (c00 * (1 - fg) + c01 * fg) * (1 - fb) + (c10 * (1 - fg) + c11 * fg) * fb


def write_cube(path, fn, n=65, title="osmo_grade"):
    g = np.linspace(0, 1, n)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([r, gg, b], -1).reshape(-1, 3)
    out = np.clip(fn(rgb), 0, 1)
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n')
        np.savetxt(f, out, fmt="%.6f")


# ----------------------------------------------------------------- looks & projects
HUES = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "cyan": 180, "blue": 220, "purple": 270, "magenta": 320}
_L = dict(contrast=1.0, pivot=0.43, black=0.0, white=1.0, shadows=0.0, highlights=0.0, sat=1.0, hue_sat={},
          skin_protect=0.0, skin_sat=1.0, hl_desat=0.0, skin_unify=0.0, shadow_tint=(0, 0, 0), hi_tint=(0, 0, 0))
LOOKS = {
    "clean": {**_L, "contrast": 1.04, "hl_desat": 0.15},
    "fights": {**_L, "contrast": 1.28, "pivot": 0.40, "black": -0.012, "white": 0.985, "shadows": -0.05,
               "highlights": -0.04, "sat": 0.80,
               "hue_sat": {"red": 1.22, "orange": 1.0, "yellow": 0.75, "green": 0.55, "cyan": 0.75,
                           "blue": 0.8, "purple": 0.7, "magenta": 0.85},
               "skin_protect": 0.6, "skin_sat": 0.92, "hl_desat": 0.35, "skin_unify": 0.25,
               "shadow_tint": (-0.010, 0.004, 0.014), "hi_tint": (0.012, 0.005, -0.010)},
    "gypsy": {**_L, "contrast": 1.03, "pivot": 0.45, "black": 0.022, "white": 0.975, "shadows": 0.03,
              "highlights": -0.05, "sat": 0.96, "hue_sat": {"green": 0.92, "yellow": 0.95},
              "skin_protect": 0.4, "hl_desat": 0.25, "skin_unify": 0.15,
              "shadow_tint": (0.006, 0.003, -0.004), "hi_tint": (0.010, 0.005, -0.008)},
    "samui": {**_L, "contrast": 1.06, "pivot": 0.45, "black": 0.004, "shadows": 0.10, "highlights": -0.06,
              "sat": 1.05, "hue_sat": {"cyan": 1.18, "blue": 1.08, "green": 1.04, "yellow": 0.96, "orange": 0.98},
              "skin_protect": 0.3, "hl_desat": 0.2, "shadow_tint": (0, 0, 0.004)},
    "portrait": {**_L, "contrast": 1.07, "black": 0.008, "white": 0.99, "shadows": 0.02, "highlights": -0.04,
                 "sat": 0.95, "hue_sat": {"orange": 0.97, "green": 0.9}, "skin_protect": 0.5, "skin_sat": 0.97,
                 "hl_desat": 0.25, "skin_unify": 0.35, "shadow_tint": (0, 0.001, 0.003),
                 "hi_tint": (0.004, 0.002, -0.003)},
}
PROJECTS = {  # exposure targets are measured on the FINAL graded image (Rec.709 luma 0-1)
    "fights": dict(look="fights", expose_by="skin", skin_target=0.46, mid_target=0.34, wb_strength=0.75),
    "gypsy": dict(look="gypsy", expose_by="skin", skin_target=0.55, mid_target=0.45, wb_strength=0.85),
    "samui": dict(look="samui", expose_by="mid", skin_target=0.58, mid_target=0.47, wb_strength=0.95),
    "portrait": dict(look="portrait", expose_by="skin", skin_target=0.54, mid_target=0.43, wb_strength=0.90),
    "clean": dict(look="clean", expose_by="skin", skin_target=0.52, mid_target=0.43, wb_strength=0.90),
}


def s_curve(x, c, p):
    x = np.clip(x, 0, 1)
    lo = p * (np.maximum(x, 1e-9) / p) ** c
    hi = 1 - (1 - p) * (np.maximum(1 - x, 1e-9) / (1 - p)) ** c
    return np.where(x < p, lo, hi)


def apply_look(rgb, lk, tw=None):
    tw = tw or {}
    c = lk["contrast"] * tw.get("contrast", 1.0)
    x = s_curve(rgb, c, lk["pivot"]) if abs(c - 1) > 1e-6 else np.clip(rgb, 0, 1)
    if lk["shadows"] or lk["highlights"]:
        Y = x @ LUMA
        ws = np.clip(1 - Y / 0.5, 0, 1) ** 2
        wh = np.clip((Y - 0.5) / 0.5, 0, 1) ** 2
        x = x + ((lk["shadows"] * ws + lk["highlights"] * wh) * 0.25)[..., None]
    b = lk["black"]
    x = b + (1 - b) * x if b >= 0 else (x + b) / (1 + b)
    x = np.clip(x, 0, 1) * lk["white"]
    Y = x @ LUMA
    mult = np.full(Y.shape, lk["sat"] * tw.get("sat", 1.0))
    if lk["hue_sat"]:
        h = rgb_hue(x)
        num, den = np.zeros_like(Y), np.zeros_like(Y)
        for name, s in lk["hue_sat"].items():
            dh = np.abs(((h - HUES[name]) + 180) % 360 - 180)
            w = np.where(dh < 45, 0.5 * (1 + np.cos(np.pi * dh / 45)), 0.0)
            num, den = num + w * s, den + w
        mult = mult * (num + np.maximum(0, 1 - den)) / np.maximum(den, 1)
    if lk["skin_protect"] > 0:
        sw = skin_weight(x) * lk["skin_protect"]
        mult = mult * (1 - sw) + lk["skin_sat"] * tw.get("sat", 1.0) * sw
    if lk["hl_desat"] > 0:
        mult = mult * (1 - lk["hl_desat"] * np.clip((Y - 0.75) / 0.25, 0, 1) ** 2)
    x = Y[..., None] + (x - Y[..., None]) * mult[..., None]
    if lk["skin_unify"] > 0:
        Y2, cb, cr = ycbcr(x)
        ang, ch = np.arctan2(cr, cb), np.hypot(cb, cr)
        da = ((ang - np.radians(SKIN_LINE) + np.pi) % (2 * np.pi)) - np.pi
        w = np.clip(1 - np.abs(np.degrees(da)) / 25, 0, 1) * np.clip((ch - 0.015) / 0.02, 0, 1)
        ang2 = ang - da * w * lk["skin_unify"]
        x = from_ycbcr(Y2, ch * np.cos(ang2), ch * np.sin(ang2))
    Y = x @ LUMA
    ws = np.clip(Y / 0.08, 0, 1) * np.clip(1 - Y / 0.45, 0, 1) ** 1.5
    wh = np.clip((Y - 0.55) / 0.45, 0, 1) ** 1.5
    x = x + np.array(lk["shadow_tint"]) * ws[..., None] + np.array(lk["hi_tint"]) * wh[..., None]
    return np.clip(x, 0, 1)


# ----------------------------------------------------------------- the chain
class Chain:
    """code values (full-range 0-1 RGB as decoded by ffmpeg) -> graded Rec.709 display RGB"""

    def __init__(self, profile, look=None, input_lut=None, gains=(1, 1, 1), stops=0.0, tw=None):
        self.p, self.look, self.lut = profile, look, input_lut
        self.g = np.asarray(gains, float) * 2 ** stops
        self.tw = tw or {}
        t, ti = self.tw.get("temp", 0.0), self.tw.get("tint", 0.0)
        self.g = self.g * np.array([1 + 0.01 * t, 1 - 0.01 * ti, 1 - 0.01 * t])

    def base(self, code):
        if self.lut is not None:  # official DJI LUT: decode -> linear gain -> re-encode -> LUT
            spec = PROFILES[self.p]  # (exact round trip at zero correction, so DJI's color is untouched)
            if spec["encode"] is not None and not np.allclose(self.g, 1):
                code = np.clip(spec["encode"](spec["decode"](code) * self.g), 0, 1)
            return np.clip(self.lut(np.clip(code, 0, 1)), 0, 1)
        if self.p == "normal":
            return np.clip(soft_shoulder(np.clip(code, 0, 1) ** 2.4 * self.g), 0, 1) ** (1 / 2.4)
        spec = PROFILES[self.p]
        lin = spec["decode"](code) @ gamut_matrix(spec["gamut"]).T
        return tonemap(np.maximum(gamut_compress(lin * self.g), 0))

    def __call__(self, code):
        b = self.base(code)
        if self.look is None:
            return b
        k = self.tw.get("intensity", 1.0)
        out = apply_look(b, self.look, self.tw)
        return np.clip(b + k * (out - b), 0, 1) if k != 1.0 else out


# ----------------------------------------------------------------- ffmpeg helpers
def run(cmd):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode:
        raise RuntimeError(f"command failed: {' '.join(cmd[:6])}...\n{r.stderr.decode()[-1500:]}")
    return r.stdout


def probe(path):
    j = json.loads(run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = int(float(v.get("tags", {}).get("rotate", rot)))
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    pix = v.get("pix_fmt", "")
    cs = (v.get("color_space") or "bt709").lower()
    return dict(path=path, width=w, height=h, rotation=rot, pix_fmt=pix, codec=v.get("codec_name"),
                bits=10 if ("10" in pix or "12" in pix) else 8,
                range="pc" if v.get("color_range") in ("pc", "full") else "tv",
                matrix="bt2020" if cs.startswith("bt2020") else "bt709",
                transfer=(v.get("color_transfer") or "").lower(), primaries=v.get("color_primaries"),
                fps=v.get("r_frame_rate"), duration=float(j["format"].get("duration", v.get("duration", 0)) or 0),
                has_audio=any(s["codec_type"] == "audio" for s in j["streams"]),
                tags={**j["format"].get("tags", {}), **{f"v.{k}": x for k, x in v.get("tags", {}).items()}})


def _inopt(info):
    return f"in_range={info['range']}:in_color_matrix={info['matrix']}:out_range=pc"


def grab(info, t, width=480):
    h = max(2, int(round(width * info["height"] / info["width"] / 2)) * 2)
    raw = run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", info["path"], "-frames:v", "1", "-vf",
               f"scale={width}:{h}:{_inopt(info)},format=rgb48le", "-f", "rawvideo", "-"])
    return np.frombuffer(raw, "<u2")[: width * h * 3].reshape(h, width, 3).astype(np.float64) / 65535.0


def grab_frames(info, n=6, width=480):
    d = info["duration"]
    times = [d * (i + 0.5) / n for i in range(n)] if d > 1 else [0.0]
    return np.stack([grab(info, t, width) for t in times])


def detect_profile(info, frames):
    if info["transfer"] == "arib-std-b67":
        return "hlg", "HLG transfer tag (HDR mode)"
    if info["transfer"] == "smpte2084":
        return "pq", "PQ HDR: not supported, reshoot in D-Log 2 / D-Log / Normal"
    if info["bits"] < 10:
        return "normal", "8-bit file"
    Y = frames @ LUMA
    p_lo, p_hi = np.quantile(Y, [0.002, 0.998])
    sat = float((frames.max(-1) - frames.min(-1)).mean())
    if not (p_lo > 0.045 and p_hi < 0.95 and sat < 0.13):
        return "normal", f"10-bit but contrasty (p0.2={p_lo:.3f}, p99.8={p_hi:.3f}, sat={sat:.3f})"
    if p_lo < 0.085:
        return "dlog2", f"log, blacks near 64/1023 (p0.2={p_lo:.3f})"
    return "dlog", f"log, blacks near 95/1023 (p0.2={p_lo:.3f}); bright scenes can fool this, confirm lens"


# ----------------------------------------------------------------- analysis
def wb_vote(disp):
    """Find the R/B gains (log2) that turn the MOST pixels neutral at once. Robust to sky, foliage,
    single colored walls: the true illuminant makes every grey/white surface neutral together."""
    small = disp[..., ::3, ::3, :].reshape(-1, 3)
    lin = np.clip(small, 1e-4, 1) ** 2.4
    Y = lin @ LUMA
    ok = (small.max(-1) < 0.96) & (Y > 0.012) & (Y < 0.8)
    if ok.sum() < 300:
        return np.zeros(2), 0.0
    L = np.log2(lin[ok])
    lr, lb = L[:, 0] - L[:, 1], L[:, 2] - L[:, 1]

    def score(gr_vals, gb_vals):
        best = (-1, 0, 0)
        for gr in gr_vals:
            rr = lr[None, :] + gr
            bb = lb[None, :] + gb_vals[:, None]
            ch = np.maximum(np.maximum(rr, bb), 0) - np.minimum(np.minimum(rr, bb), 0)
            s = np.exp(-(ch / 0.10) ** 2).sum(1)
            perp = (gr + gb_vals) / math.sqrt(2)  # off the blue-amber (Planckian-ish) axis = green/magenta
            s = s * np.exp(-(perp / 0.30) ** 2) * np.exp(-(gr ** 2 + gb_vals ** 2) / (2 * 0.9 ** 2))
            j = int(np.argmax(s))
            if s[j] > best[0]:
                best = (s[j], gr, gb_vals[j])
        return best

    g = np.linspace(-1.2, 1.2, 49)
    _, gr, gb = score(g, g)
    f = np.linspace(-0.06, 0.06, 13)
    _, gr, gb = score(gr + f, gb + f)
    rr, bb = lr + gr, lb + gb
    ch = np.maximum(np.maximum(rr, bb), 0) - np.minimum(np.minimum(rr, bb), 0)
    return np.array([gr, gb]), float((ch < 0.1).mean())


def _norm_gains(g):
    g = np.clip(np.asarray(g, float), 0.45, 2.2)
    return g / (g @ LUMA)


def solve(frames, profile, proj, lut=None, tw=None, wb_ref=None):
    tw = tw or {}
    look = LOOKS[proj["look"]]
    gains, wb_conf = np.ones(3), 1.0
    for it in range(3):  # iterate: estimate in display space, apply in scene space, re-measure
        disp = Chain(profile, None, lut, gains).base(wb_ref if wb_ref is not None else frames)
        if wb_ref is not None:
            h, w = disp.shape[-3:-1]
            m = (np.clip(disp[..., h // 3: 2 * h // 3, w // 3: 2 * w // 3, :], 1e-4, 1) ** 2.4).reshape(-1, 3)
            mean = np.median(m, 0)
            step = mean[1] / np.maximum(mean, 1e-6)
        else:
            (gr, gb), frac = wb_vote(disp)
            if it == 0:
                wb_conf = float(np.clip(frac / 0.08, 0.25, 1.0))  # few neutral surfaces = trust it less
            step = 2.0 ** np.array([gr, 0.0, gb])
        gains = _norm_gains(gains * step)
    full_gains = gains.copy()  # fully neutral version, used for skin detection only
    strength = 1.0 if wb_ref is not None else proj["wb_strength"] * wb_conf
    gains = _norm_gains(gains ** strength)

    def bisect(fn, target):
        lo, hi = -3.0, 3.0
        for _ in range(14):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fn(mid) < target else (lo, mid)
        return (lo + hi) / 2

    # skin mask found ONCE on a neutral render at a rough exposure, then the same pixels are measured
    # through the look (stylized looks desaturate skin, so re-detecting per step is unstable)
    rough = bisect(lambda st: float(np.median(Chain(profile, None, lut, full_gains, st)(frames) @ LUMA)), 0.43)
    skin_mask = skin_weight(Chain(profile, None, lut, full_gains, rough)(frames)) > 0.5
    has_skin = proj["expose_by"] == "skin" and skin_mask.mean() >= 0.005

    def measure(st, mode):
        out = Chain(profile, look, lut, gains, st, tw)(frames)
        Y = out @ LUMA
        return float(np.median(Y[skin_mask] if mode == "skin" else Y)), out

    s_mid = bisect(lambda st: measure(st, "mid")[0], proj["mid_target"])
    if has_skin:  # skin leads, scene key keeps it honest (stops one skin tone dictating everything)
        s_skin = bisect(lambda st: measure(st, "skin")[0], proj["skin_target"])
        stops, mode = 0.65 * s_skin + 0.35 * s_mid, "skin+mid"
    else:
        stops, mode = s_mid, "mid"
    if stops > 0:  # highlight guard: never blow out more than the source already does
        clip_at = lambda st: float((measure(st, "mid")[1].max(-1) > 0.985).mean())
        allow = max(clip_at(0.0), 0.01) + 0.01
        if clip_at(stops) > allow:
            lo2, hi2 = 0.0, stops
            for _ in range(10):
                m2 = (lo2 + hi2) / 2
                lo2, hi2 = (m2, hi2) if clip_at(m2) <= allow else (lo2, m2)
            stops, mode = lo2, mode + " (highlight-capped)"
    stops = float(np.clip(stops, -2.5, 2.5) + tw.get("exposure", 0.0))
    out = measure(stops, "mid")[1]
    return dict(gains=[round(float(x), 4) for x in gains], stops=round(stops, 3), expose_by=mode,
                wb_confidence=round(wb_conf, 2), targets=[proj["skin_target"], proj["mid_target"]],
                stats=image_stats(out))


def image_stats(img):
    Y, cb, cr = ycbcr(img)
    sk = skin_weight(img) > 0.5
    neutral = (img.max(-1) - img.min(-1)) < np.quantile(img.max(-1) - img.min(-1), 0.15)
    st = dict(p1=round(float(np.quantile(Y, 0.01)), 3), p50=round(float(np.median(Y)), 3),
              p99=round(float(np.quantile(Y, 0.99)), 3), clipped=round(float((img.max(-1) > 0.985).mean()), 4),
              skin_frac=round(float(sk.mean()), 4))
    if sk.any():
        st["skin_luma"] = round(float(np.median(Y[sk])), 3)
        st["skin_angle"] = round(float(np.degrees(np.arctan2(np.median(cr[sk]), np.median(cb[sk])))), 1)
    nm = img[neutral].mean(0)
    st["neutral_cast_rgb"] = [round(float(x - nm.mean()), 4) for x in nm]
    return st


# ----------------------------------------------------------------- scopes
def _wf(V, h, color):
    H, w = V.shape
    bins = np.clip((V * (h - 1)).astype(int), 0, h - 1)
    acc = np.zeros((h, w))
    np.add.at(acc, (bins.ravel(), np.broadcast_to(np.arange(w), V.shape).ravel()), 1)
    a = (np.log1p(acc) / np.log1p(acc.max() + 1e-9))[::-1]
    out = a[..., None] * np.array(color, float)[None, None]
    for lv in np.arange(0, 1.001, 0.1):
        r = int(round((1 - lv) * (h - 1)))
        out[r] = np.maximum(out[r], 55)
    return np.clip(out, 0, 255).astype(np.uint8)


def scopes(img, w=512, h=256):
    H, W, _ = img.shape
    sub = img[:, (np.arange(w) * W / w).astype(int)]
    wave = _wf(sub @ LUMA, h, (140, 255, 160))
    pw = w // 3
    sub3 = img[:, (np.arange(pw) * W / pw).astype(int)]
    parade = np.concatenate([_wf(sub3[..., i], h, c) for i, c in
                             enumerate([(255, 90, 90), (90, 255, 90), (110, 140, 255)])], 1)
    size = h
    Y, cb, cr = ycbcr(img)
    sc = (size / 2 - 6) / 0.30  # zoomed: +-0.30 Cb/Cr reaches the edge
    xs = np.clip((size / 2 + cb * sc).astype(int), 0, size - 1)
    ys = np.clip((size / 2 - cr * sc).astype(int), 0, size - 1)
    acc = np.zeros((size, size))
    np.add.at(acc, (ys.ravel(), xs.ravel()), 1)
    a = np.log1p(acc) / np.log1p(acc.max() + 1e-9)
    vs = Image.fromarray((a[..., None] * np.array([230, 230, 230])).astype(np.uint8))
    d = ImageDraw.Draw(vs)
    c0 = size / 2
    d.ellipse([6, 6, size - 6, size - 6], outline=(70, 70, 70))
    d.line([c0, 6, c0, size - 6], fill=(50, 50, 50))
    d.line([6, c0, size - 6, c0], fill=(50, 50, 50))
    ang = math.radians(SKIN_LINE)
    d.line([c0, c0, c0 + math.cos(ang) * (c0 - 6), c0 - math.sin(ang) * (c0 - 6)], fill=(255, 170, 90))
    for col, lab in zip(np.eye(3).tolist() + [[1, 1, 0], [0, 1, 1], [1, 0, 1]], "RGBYCM"):
        _, b_, r_ = ycbcr(np.array(col) * 0.75)
        px, py = c0 + b_ * sc * 0.5, c0 - r_ * sc * 0.5  # targets drawn at half scale (75% bars off-zoom)
        d.text((px - 3, py - 5), lab, fill=(120, 120, 120))
    return wave, parade, np.array(vs)


def _u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def report_image(before, after, title, lines, path):
    tw_ = 640
    def fit(a):
        im = Image.fromarray(_u8(a))
        return im.resize((tw_, max(2, int(tw_ * im.height / im.width))), Image.LANCZOS) if im.width != tw_ else im
    b, a = fit(before), fit(after)
    wave, parade, vs = scopes(after)
    top_h = max(b.height, a.height)
    W, H = tw_ * 2 + 12, top_h + 256 + 70
    canvas = Image.new("RGB", (W, H), (14, 14, 14))
    canvas.paste(b, (0, 24)); canvas.paste(a, (tw_ + 12, 24))
    y2 = top_h + 52
    canvas.paste(Image.fromarray(wave), (0, y2))
    canvas.paste(Image.fromarray(parade), (512 + 8, y2))
    canvas.paste(Image.fromarray(vs), (512 + 8 + parade.shape[1] + 8, y2))
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), "SOURCE (as shot)", fill=(200, 200, 200))
    d.text((tw_ + 18, 6), title, fill=(200, 200, 200))
    d.text((6, y2 - 22), "  |  ".join(lines)[:220], fill=(160, 160, 160))
    canvas.save(path)


# ----------------------------------------------------------------- render
PRESETS = {
    "reel": dict(ext=".mp4", pix="yuv420p", scale="scale=w='if(gt(iw,ih),1920,-2)':h='if(gt(iw,ih),-2,1920)':flags=lanczos:",
                 codec=["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high", "-maxrate", "30M",
                        "-bufsize", "60M", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k"]),
    "master": dict(ext=".mp4", pix="yuv420p10le", scale="scale=",
                   codec=["-c:v", "libx265", "-preset", "medium", "-crf", "14", "-tag:v", "hvc1", "-x265-params",
                          "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited:log-level=error",
                          "-movflags", "+faststart", "-c:a", "aac", "-b:a", "256k"]),
    "prores": dict(ext=".mov", pix="yuv422p10le", scale="scale=",
                   codec=["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-c:a", "pcm_s16le"]),
}


def render(info, cube, out, preset="reel", crop=None, start=None, length=None):
    p = PRESETS[preset]
    vf = [f"scale={_inopt(info)},format=gbrp16le", f"lut3d=file='{cube}':interp=tetrahedral"]
    if crop == "9:16":
        vf.append("crop=w='min(iw,ih*9/16)':h='min(ih,iw*16/9)'")
    elif crop == "4:5":
        vf.append("crop=w='min(iw,ih*4/5)':h='min(ih,iw*5/4)'")
    vf.append(p["scale"] + "out_color_matrix=bt709:out_range=tv")
    vf.append(f"format={p['pix']}")
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", info["path"]]
    if length is not None:
        cmd += ["-t", str(length)]
    cmd += ["-map", "0:v:0", "-map", "0:a?", "-vf", ",".join(vf), *p["codec"],
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", out]
    run(cmd)


# ----------------------------------------------------------------- commands
def _tweaks(a):
    return {k: getattr(a, k) for k in ("exposure", "temp", "tint", "sat", "contrast", "intensity")}


def _prep(path, a, lut_for):
    info = probe(path)
    frames = grab_frames(info, n=a.frames)
    prof, why = (a.profile, "forced") if a.profile != "auto" else detect_profile(info, frames)
    if prof == "pq":
        raise SystemExit(f"{path}: {why}")
    lut = lut_for(prof)
    note = None
    if prof == "dlog2" and lut is None:
        note = "D-Log 2 graded with the FALLBACK curve: send DJI's official D-Log 2 LUT for exact color"
    return info, frames, prof, why, lut, note


def cmd_analyze(a):
    luts = _lut_loader(a)
    for path in a.clips:
        info, frames, prof, why, lut, note = _prep(path, a, luts)
        res = solve(frames, prof, PROJECTS[a.project], lut, _tweaks(a))
        print(json.dumps(dict(clip=path, profile=prof, why=why, note=note, info={k: v for k, v in info.items()},
                              solution=res), indent=1, default=str))


def _lut_loader(a):
    cache = {}
    def get(prof):
        path = {"dlog2": a.input_lut, "dlog": a.input_lut_dlog}.get(prof)
        if not path:
            return None
        if path not in cache:
            cache[path] = Cube(path)
        return cache[path]
    return get


def cmd_grade(a):
    os.makedirs(a.out, exist_ok=True)
    proj = PROJECTS[a.project]
    look = LOOKS[proj["look"]]
    tw = _tweaks(a)
    luts = _lut_loader(a)
    preps = [_prep(p, a, luts) for p in a.clips]
    wb_ref = None
    if a.wb_ref:
        rp, _, rt = a.wb_ref.rpartition("@")
        rinfo = probe(rp)
        wb_ref = grab(rinfo, float(rt))[None]
    shared = {}
    if a.lock:  # one correction per profile for the batch (same light / same setup): zero shot-to-shot drift
        for prof in sorted({p[2] for p in preps}):
            grp = [p for p in preps if p[2] == prof]
            shared[prof] = solve(np.concatenate([p[1] for p in grp]), prof, proj, grp[0][4], tw, wb_ref)
    summary = []
    for info, frames, prof, why, lut, note in preps:
        sol = shared.get(prof) or solve(frames, prof, proj, lut, tw, wb_ref)
        name = os.path.splitext(os.path.basename(info["path"]))[0]
        chain = Chain(prof, look, lut, sol["gains"], sol["stops"], tw)
        cube = os.path.join(a.out, f"{name}_{a.project}.cube")
        write_cube(cube, chain, 65, f"{name} {prof} {a.project}")
        before = frames[len(frames) // 2]
        after = chain(before)
        lines = [f"profile {prof} ({why})", f"exposure {sol['stops']:+.2f} st by {sol['expose_by']}",
                 f"WB gains {sol['gains']} (conf {sol['wb_confidence']})",
                 f"skin {sol['stats'].get('skin_luma', '-')} @ {sol['stats'].get('skin_angle', '-')} deg"]
        report_image(before, after, f"GRADED: {a.project}", lines, os.path.join(a.out, f"{name}_report.png"))
        outs = []
        if not a.no_render:
            for preset in a.preset.split(","):
                out = os.path.join(a.out, f"{name}_{a.project}_{preset}{PRESETS[preset]['ext']}")
                render(info, cube, out, preset, a.crop, a.start, a.length)
                outs.append(out)
        for i in range(a.stills):
            t = info["duration"] * (i + 0.5) / max(a.stills, 1)
            Image.fromarray(_u8(chain(grab(info, t, width=info["width"])))).save(
                os.path.join(a.out, f"{name}_{a.project}_still{i + 1}.jpg"), quality=95)
        summary.append(dict(clip=info["path"], profile=prof, why=why, note=note, solution=sol, cube=cube, outputs=outs))
        print(f"[ok] {name}: {prof}, {sol['stops']:+.2f} stops, gains {sol['gains']}" + (f"  NOTE: {note}" if note else ""))
    with open(os.path.join(a.out, "grade_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)


def cmd_luts(a):
    os.makedirs(a.out, exist_ok=True)
    d2 = Cube(a.input_lut) if a.input_lut else None
    for pname, proj in PROJECTS.items():
        lk = LOOKS[proj["look"]]
        write_cube(os.path.join(a.out, f"{pname}_look_on_Rec709.cube"), lambda x, lk=lk: apply_look(x, lk), 33,
                   f"{pname} look (after a Rec.709 conversion)")
        write_cube(os.path.join(a.out, f"{pname}_from_DLog.cube"), Chain("dlog", lk), 33, f"{pname} from D-Log")
        if d2:
            write_cube(os.path.join(a.out, f"{pname}_from_DLog2.cube"), Chain("dlog2", lk, d2), 33,
                       f"{pname} from D-Log 2 (DJI conversion + look)")
    write_cube(os.path.join(a.out, "DLog_to_Rec709_neutral.cube"), Chain("dlog"), 33, "D-Log to Rec.709 neutral")
    print("wrote LUTs to", a.out)


def cmd_scopes(a):
    img = np.asarray(Image.open(a.image).convert("RGB"), float) / 255
    wave, parade, vs = scopes(img)
    out = a.out or os.path.splitext(a.image)[0] + "_scopes.png"
    W = wave.shape[1] + parade.shape[1] + vs.shape[1] + 16
    c = Image.new("RGB", (W, 256), (14, 14, 14))
    c.paste(Image.fromarray(wave), (0, 0)); c.paste(Image.fromarray(parade), (wave.shape[1] + 8, 0))
    c.paste(Image.fromarray(vs), (wave.shape[1] + parade.shape[1] + 16, 0))
    c.save(out)
    print(out, json.dumps(image_stats(img)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("clips", nargs="+")
        p.add_argument("--project", default="clean", choices=PROJECTS)
        p.add_argument("--profile", default="auto", choices=["auto", "dlog2", "dlog", "hlg", "normal"])
        p.add_argument("--input-lut", help="DJI official D-Log 2 -> Rec.709 .cube (exact D-Log 2 color)")
        p.add_argument("--input-lut-dlog", help="optional DJI official D-Log -> Rec.709 .cube for tele clips")
        p.add_argument("--frames", type=int, default=6)
        p.add_argument("--exposure", type=float, default=0.0, help="extra stops on top of auto")
        p.add_argument("--temp", type=float, default=0.0, help="+ warmer / - cooler (approx %%)")
        p.add_argument("--tint", type=float, default=0.0, help="+ magenta / - green")
        p.add_argument("--sat", type=float, default=1.0)
        p.add_argument("--contrast", type=float, default=1.0)
        p.add_argument("--intensity", type=float, default=1.0, help="look strength 0..1.5")
    p1 = sp.add_parser("analyze"); common(p1); p1.set_defaults(fn=cmd_analyze)
    p2 = sp.add_parser("grade"); common(p2); p2.set_defaults(fn=cmd_grade)
    p2.add_argument("--out", required=True)
    p2.add_argument("--preset", default="reel", help="reel,master,prores (comma list)")
    p2.add_argument("--crop", choices=["9:16", "4:5"], help="center crop for vertical delivery")
    p2.add_argument("--lock", action="store_true", help="one shared correction for all clips (same setup)")
    p2.add_argument("--wb-ref", help="CLIP@SECONDS of a grey card / white wall filling the center")
    p2.add_argument("--stills", type=int, default=0, help="also export N graded full-res JPEG stills per clip")
    p2.add_argument("--start", type=float); p2.add_argument("--length", type=float)
    p2.add_argument("--no-render", action="store_true", help="LUT + report only")
    p3 = sp.add_parser("luts"); p3.add_argument("--out", required=True); p3.add_argument("--input-lut")
    p3.set_defaults(fn=cmd_luts)
    p4 = sp.add_parser("scopes"); p4.add_argument("image"); p4.add_argument("--out"); p4.set_defaults(fn=cmd_scopes)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
