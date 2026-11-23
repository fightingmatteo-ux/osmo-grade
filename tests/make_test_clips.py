"""Build synthetic Pocket 4P style clips: known scene -> camera log encoding with deliberate errors."""
import subprocess, sys
import numpy as np
import colour
sys.path.insert(0, "/home/claude/osmo_grade")
import osmo_grade as og

W, H = 1920, 1080
rng = np.random.default_rng(7)

def chart_linear_709():
    cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    xyY = np.array(list(cc.data.values()))
    XYZ = colour.xyY_to_XYZ(xyY)
    rgb = colour.XYZ_to_RGB(XYZ, "ITU-R BT.709", illuminant=cc.illuminant, chromatic_adaptation_transform="Bradford")
    return np.clip(rgb, 0, None), list(cc.data.keys())

def scene():
    img = np.zeros((H, W, 3))
    img[:, : W // 2] = 0.12                      # concrete wall (neutral)
    img[:, W // 2:] = [0.035, 0.075, 0.025]      # foliage
    img[: H // 5, W // 2:] = [0.55, 0.75, 1.25]  # bright sky
    chart, names = chart_linear_709()
    pw, ph, x0, y0 = 110, 110, 60, 300
    for i, c in enumerate(chart):
        r, q = divmod(i, 6)
        img[y0 + r * ph: y0 + r * ph + ph - 10, x0 + q * pw: x0 + q * pw + pw - 10] = c
    # three "faces": light, olive/medium (Matteo-ish), deep
    for k, skin in enumerate([[0.42, 0.30, 0.24], [0.30, 0.20, 0.14], [0.12, 0.075, 0.055]]):
        cx, cy = 1050 + k * 260, 620
        yy, xx = np.mgrid[:H, :W]
        m = ((xx - cx) / 100) ** 2 + ((yy - cy) / 140) ** 2 < 1
        img[m] = skin
    # exposure ramp -6..+6 stops around 18% grey
    for s in range(13):
        img[H - 120: H - 40, 60 + s * 140: 60 + s * 140 + 130] = 0.18 * 2.0 ** (s - 6)
    img[60:100, 1500:1560] = 0.18 * 2 ** 7       # specular highlight
    return img

def encode(lin709, profile, cast=(1.25, 1.0, 0.70), stops=-1.0):
    lin = lin709 * np.array(cast) * 2 ** stops
    dg = lin @ np.linalg.inv(og.gamut_matrix("dgamut")).T
    code = og.lin_to_dlog(dg) if profile == "dlog" else og.lin_to_dlog2(np.maximum(dg, 0))
    return np.clip(code, 0, 1)

def write_clip(code, path, frames=25):
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{W}x{H}",
                          "-r", "25", "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p10le",
                          "-c:v", "libx265", "-crf", "8", "-preset", "fast", "-x265-params", "log-level=error",
                          "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
                          "-color_trc", "bt709", path], stdin=subprocess.PIPE)
    for _ in range(frames):
        noisy = np.clip(code + rng.normal(0, 0.004, code.shape), 0, 1)
        p.stdin.write((noisy * 65535 + 0.5).astype("<u2").tobytes())
    p.stdin.close(); p.wait()

if __name__ == "__main__":
    s = scene()
    np.save("/home/claude/tests/scene_lin709.npy", s)
    write_clip(encode(s, "dlog"), "/home/claude/tests/TEST_tele_dlog_warm_under.mp4")
    write_clip(encode(s, "dlog2"), "/home/claude/tests/TEST_main_dlog2_warm_under.mp4")
    write_clip(encode(s, "dlog", cast=(1, 1, 1), stops=0), "/home/claude/tests/TEST_tele_dlog_correct.mp4")
    # stand-in for DJI's official D-Log 2 LUT, to exercise the --input-lut code path
    og.write_cube("/home/claude/tests/FAKE_official_DLog2_to_Rec709.cube", og.Chain("dlog2"), 33, "stand-in")
    print("ok")
