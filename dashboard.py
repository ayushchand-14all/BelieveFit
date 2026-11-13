"""
dashboard.py — Renders the analytics dashboard video.

Layout:
  Left  → annotated pose video (skeleton + live HUD overlaid)
  Right → dark panel with:
    • brand header + live/complete status badge
    • 3 stat cards: clean reps (dot trail), form-score ring, depth gauge
    • knee/elbow angle chart (glow line + gradient fill + rep bands)
    • rep breakdown chips (fade-in cards per rep)
    • animated "Set complete" summary overlay at the end

All drawing is done at 2× resolution then down-sampled for crisp edges on
any display.  Requires Pillow and imageio-ffmpeg (for ffmpeg).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import urllib.request
from collections import Counter
from typing import Optional

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SS = 2          # super-sampling factor (draw at 2×, reduce to 1×)

# ---------------------------------------------------------------------------
# Layout constants (all in 1× pixels)
# ---------------------------------------------------------------------------
H        = 1080
MARGIN   = 36
GAP      = 28
PANEL_W  = 820        # right panel width for portrait/square videos
WIDE_VIDEO_W = 1100   # video width for landscape videos
WIDE_PANEL_W = 720    # right panel for landscape videos
VIDEO_MAX_W  = 900
RADIUS   = 26

# ---------------------------------------------------------------------------
# Colour palette (RGB tuples — PIL uses RGB)
# ---------------------------------------------------------------------------
BG_TOP      = (10,  14,  26)
BG_BOTTOM   = (17,  24,  41)
CARD        = (22,  30,  48)
CARD_EDGE   = (44,  56,  82)
TEXT        = (238, 242, 250)
MUTED       = (128, 140, 165)
FAINT       = (58,  70,  96)
GOOD        = (124, 230, 120)
BAD         = (255,  92, 110)
WARN        = (255, 184,  76)
LINE        = (76,  201, 240)
PERSON_RGB  = [(124, 230, 120), (196, 120, 255), (255, 184, 76), (76, 201, 240)]

STANDING = 160
PARALLEL =  90
ANGLE_TOP    = 185
ANGLE_BOTTOM =  35

EXERCISES = {
    "squat": {
        "title": "AI SQUAT COACH",
        "angle": "Knee angle",
        "refs":  {STANDING: "standing", PARALLEL: "parallel"},
        "tags":  {"depth": "DEPTH", "lean": "CHEST", "heels": "HEELS",
                  "knees": "KNEES", "tempo": "RUSHED", None: "NO REP"},
        "focus": {"depth": "Go a little deeper", "lean": "Keep your chest up",
                  "heels": "Keep your heels down", "knees": "Push your knees out",
                  "tempo": "Slow down on the way down"},
    },
    "pushup": {
        "title": "AI PUSH-UP COACH",
        "angle": "Elbow angle",
        "refs":  {STANDING: "arms locked", PARALLEL: "chest down"},
        "tags":  {"depth": "DEPTH", "sag": "HIPS SAG", "pike": "HIPS UP",
                  "head": "HEAD", "tempo": "RUSHED", "lockout": "LOCKOUT",
                  None: "NO REP"},
        "focus": {"depth": "Get your chest lower", "sag": "Keep your hips up",
                  "pike": "Keep a straight body line", "head": "Keep your head in line",
                  "tempo": "Slow down on the way down", "lockout": "Lock out at the top"},
    },
}

# ---------------------------------------------------------------------------
# Fonts (Poppins downloaded once; falls back to Segoe UI / DejaVu)
# ---------------------------------------------------------------------------
FONT_DIR = "fonts"
POPPINS  = "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-{}.ttf"


def _get_fonts() -> dict:
    os.makedirs(FONT_DIR, exist_ok=True)
    found: dict = {}
    for weight in ("Regular", "Medium", "SemiBold", "Bold"):
        path = os.path.join(FONT_DIR, f"Poppins-{weight}.ttf")
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(POPPINS.format(weight), path)
            except Exception:
                pass
        if os.path.exists(path):
            found[weight] = path
    if len(found) == 4:
        return found
    fallbacks = {
        "Regular":  ["C:/Windows/Fonts/segoeui.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
        "Medium":   ["C:/Windows/Fonts/seguisb.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
        "SemiBold": ["C:/Windows/Fonts/seguisb.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        "Bold":     ["C:/Windows/Fonts/segoeuib.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    }
    for weight, paths in fallbacks.items():
        if weight not in found:
            found[weight] = next((p for p in paths if os.path.exists(p)), None)
    return found


FONT_PATHS = _get_fonts()
_font_cache: dict = {}


def _font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    key = (size, weight)
    if key not in _font_cache:
        path = FONT_PATHS.get(weight)
        _font_cache[key] = (
            ImageFont.truetype(path, size * SS) if path
            else ImageFont.load_default(size * SS)
        )
    return _font_cache[key]


# ---------------------------------------------------------------------------
# Drawing layer (super-sampled transparent surface)
# ---------------------------------------------------------------------------
class Layer:
    """
    A transparent RGBA surface at 2× resolution covering one region
    of the final canvas.  Every coordinate is given in 1× canvas space.
    """

    def __init__(self, x0: int, y0: int, w: int, h: int) -> None:
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
        self.d   = ImageDraw.Draw(self.img)

    def _p(self, x: float, y: float) -> tuple:
        return ((x - self.x0) * SS, (y - self.y0) * SS)

    def _box(self, x0, y0, x1, y1) -> list:
        return [*self._p(x0, y0), *self._p(x1, y1)]

    def rrect(self, x0, y0, x1, y1, r, fill=None, outline=None, width=1) -> None:
        self.d.rounded_rectangle(self._box(x0, y0, x1, y1), r * SS,
                                 fill=fill, outline=outline, width=width * SS)

    def text(self, x, y, s, size, weight="Regular", fill=TEXT, anchor="la") -> None:
        self.d.text(self._p(x, y), s, font=_font(size, weight), fill=fill, anchor=anchor)

    def width_of(self, s, size, weight="Regular") -> float:
        return self.d.textlength(s, font=_font(size, weight)) / SS

    def circle(self, cx, cy, r, fill=None, outline=None, width=1) -> None:
        self.d.ellipse(self._box(cx - r, cy - r, cx + r, cy + r),
                       fill=fill, outline=outline, width=width * SS)

    def arc(self, cx, cy, r, start, end, fill, width) -> None:
        self.d.arc(self._box(cx - r, cy - r, cx + r, cy + r),
                   start, end, fill=fill, width=width * SS)

    def line(self, pts, fill, width, joint="curve") -> None:
        self.d.line([self._p(*q) for q in pts], fill=fill, width=width * SS, joint=joint)

    def polygon(self, pts, fill) -> None:
        self.d.polygon([self._p(*q) for q in pts], fill=fill)

    def onto(self, canvas: Image.Image) -> None:
        """Anti-alias by shrinking 2× → 1× and composite onto canvas."""
        small = self.img.reduce(SS)
        canvas.alpha_composite(small, (self.x0, self.y0))


def _alpha(rgb: tuple, a: float) -> tuple:
    return (*rgb, int(255 * max(0.0, min(1.0, a))))


def _ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def _mix(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
class Dashboard:
    def __init__(self, report: dict, video_w: int, video_h: int) -> None:
        self.report   = report
        self.ex       = EXERCISES.get(report.get("exercise", "squat"), EXERCISES["squat"])
        self.people   = report["people"]
        self.many     = len(self.people) > 1
        self.duration = max(report["duration_s"], 1.0)

        for k, p in enumerate(self.people):
            p["rgb"]    = PERSON_RGB[k % len(PERSON_RGB)]
            p["_times"] = np.array([s[0] for s in p["series"]]) if p["series"] else np.zeros(0)

        y = MARGIN
        self.top   = (y, y + 56)
        self.stats = (y + 76, y + 316)

        if video_w > video_h * 1.1:   # landscape
            self.vw = WIDE_VIDEO_W
            self.vh = int(round(self.vw * video_h / video_w)) // 2 * 2
            if self.vh > H - 2 * MARGIN - GAP - 300:
                self.vh = (H - 2 * MARGIN - GAP - 300) // 2 * 2
                self.vw = int(round(self.vh * video_w / video_h)) // 2 * 2
            self.vx, self.vy = MARGIN, MARGIN
            self.pw = WIDE_PANEL_W
            self.px = MARGIN + WIDE_VIDEO_W + GAP
            self.chart   = (y + 336, self.vy + self.vh)
            self.chipbox = (MARGIN, self.vy + self.vh + GAP, self.px + self.pw, H - MARGIN)
        else:                          # portrait / square
            box_h    = H - 2 * MARGIN
            self.vw  = min(int(round(box_h * video_w / video_h)), VIDEO_MAX_W) // 2 * 2
            self.vh  = int(round(self.vw * video_h / video_w)) // 2 * 2
            self.vx  = MARGIN
            self.vy  = MARGIN + (box_h - self.vh) // 2
            self.pw  = PANEL_W
            self.px  = MARGIN + self.vw + GAP
            self.chart   = (y + 336, y + 676)
            self.chipbox = (self.px, y + 696, self.px + PANEL_W, H - MARGIN)

        self.W   = (self.px + self.pw + MARGIN) // 2 * 2
        self.ax0 = self.px + 118
        self.ax1 = self.px + self.pw - 30
        self.ay0 = self.chart[0] + 100
        self.ay1 = self.chart[1] - 46

        self._base  = self._build_base()
        self._mask  = self._build_video_mask()
        self._grad  = self._build_gradient()

    # ------------------------------------------------------------------ statics
    def _stat_boxes(self) -> list:
        w = (self.pw - 2 * 20) / 3
        return [
            (int(self.px + i * (w + 20)), self.stats[0],
             int(self.px + i * (w + 20) + w), self.stats[1])
            for i in range(3)
        ]

    def _build_base(self) -> Image.Image:
        img = Image.new("RGBA", (self.W, H))
        col = np.linspace(0, 1, H)[:, None, None]
        top, bottom = np.array(BG_TOP), np.array(BG_BOTTOM)
        grad = (top + (bottom - top) * col).astype(np.uint8)
        img.paste(Image.fromarray(np.repeat(grad, self.W, axis=1), "RGB"))

        glow = Image.new("RGBA", (self.W, H), (0, 0, 0, 0))
        g    = ImageDraw.Draw(glow)
        g.ellipse((self.px + 300, -260, self.px + 1100, 400), fill=(76, 201, 240, 40))
        g.ellipse((-200, H - 380, 500, H + 280), fill=(124, 230, 120, 26))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(120)))

        shadow = Image.new("RGBA", (self.W, H), (0, 0, 0, 0))
        s = ImageDraw.Draw(shadow)
        boxes = ([(self.vx, self.vy, self.vx + self.vw, self.vy + self.vh)]
                 + self._stat_boxes()
                 + [(self.px, self.chart[0], self.px + self.pw, self.chart[1]),
                    self.chipbox])
        for b in boxes:
            s.rounded_rectangle((b[0], b[1] + 14, b[2], b[3] + 14), RADIUS, fill=(0, 0, 0, 120))
        img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18)))

        L = Layer(0, 0, self.W, H)
        for b in self._stat_boxes():
            L.rrect(*b, RADIUS, fill=CARD, outline=CARD_EDGE)
        L.rrect(self.px, self.chart[0], self.px + self.pw, self.chart[1],
                RADIUS, fill=CARD, outline=CARD_EDGE)
        L.rrect(*self.chipbox, RADIUS, fill=CARD, outline=CARD_EDGE)
        L.rrect(self.vx - 1, self.vy - 1,
                self.vx + self.vw + 1, self.vy + self.vh + 1,
                RADIUS + 1, outline=CARD_EDGE, width=2)

        L.circle(self.px + 12, self.top[0] + 26, 9, fill=GOOD)
        L.text(self.px + 32, self.top[0] + 26, self.ex["title"], 24, "Bold", anchor="lm")

        for b, label in zip(self._stat_boxes(), ["CLEAN REPS", "FORM SCORE", "DEPTH"]):
            L.text(b[0] + 24, b[1] + 26, label, 15, "SemiBold", MUTED, anchor="lm")
        L.text(self.px + 28, self.chart[0] + 34, self.ex["angle"], 22, "SemiBold", anchor="lm")
        L.text(self.chipbox[0] + 28, self.chipbox[1] + 34, "Rep breakdown", 22, "SemiBold", anchor="lm")

        for a, name in self.ex["refs"].items():
            L.text(self.ax0 - 12, self._Y(a), name, 14, "Medium", MUTED, anchor="rm")
        for sec in range(0, int(self.duration) + 1, 5):
            L.text(self._X(sec), self.ay1 + 22, f"{sec}s", 13, "Medium", MUTED, anchor="mm")

        lx = self.px + self.pw - 28
        if self.many:
            for p in reversed(self.people):
                w = L.width_of(f"Person {p['person']}", 15, "Medium")
                L.text(lx, self.chart[0] + 34, f"Person {p['person']}", 15, "Medium", MUTED, anchor="rm")
                L.circle(lx - w - 14, self.chart[0] + 34, 6, fill=p["rgb"])
                lx -= w + 40
        else:
            for name, colour in (("no rep", BAD), ("good rep", GOOD)):
                w = L.width_of(name, 15, "Medium")
                L.text(lx, self.chart[0] + 34, name, 15, "Medium", MUTED, anchor="rm")
                L.rrect(lx - w - 26, self.chart[0] + 27, lx - w - 12, self.chart[0] + 41,
                        4, fill=_alpha(colour, 0.55))
                lx -= w + 46
        L.onto(img)
        return img

    def _build_video_mask(self) -> Image.Image:
        m  = Image.new("L", (self.vw * SS, self.vh * SS), 0)
        ImageDraw.Draw(m).rounded_rectangle(
            (0, 0, self.vw * SS - 1, self.vh * SS - 1), RADIUS * SS, fill=255)
        return m.reduce(SS)

    def _build_gradient(self) -> Image.Image:
        h    = self.ay1 - self.ay0
        ramp = (np.linspace(1, 0, h) ** 1.4 * 110).astype(np.uint8)
        return Image.fromarray(np.repeat(ramp[:, None], self.ax1 - self.ax0, axis=1), "L")

    # ------------------------------------------------------------------ coords
    def _X(self, t: float) -> float:
        return self.ax0 + (self.ax1 - self.ax0) * t / self.duration

    def _Y(self, a: float) -> float:
        a = float(np.clip(a, ANGLE_BOTTOM, ANGLE_TOP))
        return self.ay0 + (self.ay1 - self.ay0) * (ANGLE_TOP - a) / (ANGLE_TOP - ANGLE_BOTTOM)

    def _live_angle(self, p: dict, t: float) -> Optional[float]:
        if not len(p["_times"]):
            return None
        i = int(np.searchsorted(p["_times"], t, side="right")) - 1
        if i < 0 or t - p["series"][i][0] > 0.25:
            return None
        return float(p["series"][i][1])

    # ------------------------------------------------------------------ render
    def render(self, frame: np.ndarray, t: float, end_progress: Optional[float] = None) -> np.ndarray:
        canvas = self._base.copy()
        video  = Image.fromarray(
            cv2.cvtColor(
                cv2.resize(frame, (self.vw, self.vh), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
        )
        canvas.paste(video, (self.vx, self.vy), self._mask)
        done = [[r for r in p["reps"] if r["time_s"] <= t] for p in self.people]

        L = Layer(self.px, 0, self.pw, H)
        self._draw_status(L, t, end_progress is not None)
        self._draw_stats(L, t, done)
        L.onto(canvas)

        x0, y0, x1, y1 = self.chipbox
        C = Layer(x0, y0, x1 - x0, y1 - y0)
        self._draw_chips(C, t, done)
        C.onto(canvas)

        self._draw_chart(canvas, t, done)

        V = Layer(self.vx, self.vy, self.vw, self.vh)
        self._draw_view_badge(V)
        self._draw_toast(V, t, done)
        if end_progress is not None:
            self._draw_summary(V, end_progress)
        V.onto(canvas)

        return cv2.cvtColor(np.asarray(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------ widgets
    def _draw_status(self, L: Layer, t: float, done: bool) -> None:
        right, cy = self.px + self.pw, self.top[0] + 26
        label, colour = ("SET COMPLETE", GOOD) if done else (f"LIVE  {int(t) // 60:02d}:{int(t) % 60:02d}", BAD)
        w     = L.width_of(label, 16, "SemiBold") + 50
        L.rrect(right - w, cy - 19, right, cy + 19, 19,
                fill=_alpha(colour, 0.14), outline=_alpha(colour, 0.5))
        pulse = 0.55 + 0.45 * math.sin(t * 6) if not done else 1.0
        L.circle(right - w + 20, cy, 6, fill=_alpha(colour, pulse))
        L.text(right - 16, cy, label, 16, "SemiBold", colour, anchor="rm")

    def _draw_stats(self, L: Layer, t: float, done: list) -> None:
        a, b, c    = self._stat_boxes()
        all_reps   = sorted((r for reps in done for r in reps), key=lambda r: r["time_s"])
        good_count = sum(r["valid"] for r in all_reps)
        recent     = bool(all_reps and t - all_reps[-1]["time_s"] < 0.35)

        # --- card A: clean reps
        x0, y0, x1, y1 = a
        if self.many:
            for k, (p, reps) in enumerate(zip(self.people, done)):
                yy = y0 + 92 + k * 72
                L.circle(x0 + 34, yy, 8, fill=p["rgb"])
                g = sum(r["valid"] for r in reps)
                L.text(x0 + 52, yy, str(g), 52, "Bold", anchor="lm")
                L.text(x0 + 56 + L.width_of(str(g), 52, "Bold"), yy + 8,
                       f"/ {len(reps)}", 22, "Medium", MUTED, anchor="lm")
        else:
            col = GOOD if recent and all_reps[-1]["valid"] else (BAD if recent else TEXT)
            L.text(x0 + 22, y0 + 120, str(good_count), 92, "Bold", col, anchor="lm")
            L.text(x0 + 30 + L.width_of(str(good_count), 92, "Bold"), y0 + 138,
                   f"/ {len(all_reps)}", 30, "Medium", MUTED, anchor="lm")
        dots = all_reps[-12:]
        for i, r in enumerate(dots):
            cx2 = x0 + 30 + (i % 6) * 34
            cy2 = y1 - 52 + (i // 6) * 26
            L.circle(cx2, cy2, 9, fill=GOOD if r["valid"] else BAD)
        if not dots:
            L.text(x0 + 24, y1 - 52, "waiting for rep 1", 15, "Medium", MUTED, anchor="lm")

        # --- card B: form-score ring
        x0, y0, x1, y1 = b
        cx, cy, R = (x0 + x1) / 2, y0 + 140, 72
        L.arc(cx, cy, R, 0, 360, FAINT, 14)
        if all_reps:
            prev   = float(np.mean([r["score"] for r in all_reps[:-1]])) if len(all_reps) > 1 else 0.0
            target = float(np.mean([r["score"] for r in all_reps]))
            shown  = prev + (target - prev) * _ease((t - all_reps[-1]["time_s"]) / 0.6)
            colour = GOOD if shown >= 85 else WARN if shown >= 65 else BAD
            L.arc(cx, cy, R, -90, -90 + 360 * shown / 100, colour, 14)
            end = math.radians(-90 + 360 * shown / 100)
            L.circle(cx + R * math.cos(end), cy + R * math.sin(end), 7, fill=colour)
            L.text(cx, cy - 6,  f"{shown:.0f}", 46, "Bold", anchor="mm")
            L.text(cx, cy + 32, "avg / 100",    14, "Medium", MUTED, anchor="mm")
        else:
            L.text(cx, cy, "--", 46, "Bold", MUTED, anchor="mm")

        # --- card C: depth gauge
        x0, y0, x1, y1 = c
        cx, cy, R = (x0 + x1) / 2, y0 + 168, 88

        def _pos(a: float) -> float:
            return 180 + 180 * (180 - float(np.clip(a, 40, 180))) / 140

        L.arc(cx, cy, R, 180, 360, FAINT, 16)
        L.arc(cx, cy, R, _pos(PARALLEL), 360, _alpha(GOOD, 0.85), 16)
        L.text(x1 - 24, y0 + 26, "target", 13, "Medium", GOOD, anchor="rm")
        for k, p in enumerate(self.people):
            ang = self._live_angle(p, t)
            if ang is None:
                continue
            th  = math.radians(_pos(ang))
            tip = (cx + (R - 26) * math.cos(th), cy + (R - 26) * math.sin(th))
            col = p["rgb"] if self.many else (GOOD if ang <= PARALLEL else TEXT)
            L.line([(cx, cy), tip], col, 5)
            L.circle(*tip, 6, fill=col)
            if k == 0:
                L.text(cx, cy + 38, f"{ang:.0f}°", 34, "Bold", anchor="mm")
        L.circle(cx, cy, 10, fill=TEXT)

    def _draw_chart(self, canvas: Image.Image, t: float, done: list) -> None:
        x0, y0, x1, y1 = self.ax0, self.ay0, self.ax1, self.ay1
        C = Layer(x0 - 20, y0 - 44, x1 - x0 + 40, y1 - y0 + 60)

        for k, (p, reps) in enumerate(zip(self.people, done)):
            for r in reps:
                colour = GOOD if r["valid"] else BAD
                appear = _ease((t - r["time_s"]) / 0.4)
                bx0, bx1 = self._X(r["start_s"]), self._X(r["time_s"])
                if self.many:
                    ly = y0 + 2 + k * 12
                    C.rrect(bx0, ly, bx1, ly + 8, 4, fill=_alpha(colour, 0.8 * appear))
                else:
                    C.rrect(bx0, y0, bx1, y1, 10, fill=_alpha(colour, 0.13 * appear))
                    mx = (bx0 + bx1) / 2
                    C.rrect(mx - 14, y0 - 34, mx + 14, y0 - 8, 8, fill=_alpha(colour, appear))
                    C.text(mx, y0 - 21, str(r["rep"]), 14, "Bold",
                           _alpha(BG_TOP, appear), anchor="mm")

        for a_ref in (STANDING, PARALLEL):
            y_ref = self._Y(a_ref)
            for xs in np.arange(x0, x1, 16):
                C.line([(xs, y_ref), (min(xs + 7, x1), y_ref)], FAINT, 2, joint=None)
        C.line([(x0, y1), (x1, y1)], FAINT, 2, joint=None)
        C.onto(canvas)

        for p in self.people:
            colour   = LINE if not self.many else p["rgb"]
            segments: list = []
            seg:      list = []
            last_t:   Optional[float] = None
            for s_t, a_val in p["series"]:
                if s_t > t:
                    break
                if last_t is not None and s_t - last_t > 0.25:
                    segments.append(seg)
                    seg = []
                seg.append((self._X(s_t), self._Y(a_val)))
                last_t = s_t
            segments.append(seg)
            segments = [s for s in segments if len(s) > 1]
            if not segments:
                continue

            if not self.many:
                m  = Image.new("L", self._grad.size, 0)
                md = ImageDraw.Draw(m)
                for s in segments:
                    poly = [(px - x0, py - y0) for px, py in s]
                    poly += [(s[-1][0] - x0, y1 - y0), (s[0][0] - x0, y1 - y0)]
                    md.polygon(poly, fill=255)
                fill_img = Image.new("RGBA", self._grad.size, (*colour, 0))
                fill_img.putalpha(ImageChops.multiply(m, self._grad))
                canvas.alpha_composite(fill_img, (int(x0), int(y0)))

            G = Layer(x0 - 20, y0 - 16, x1 - x0 + 40, y1 - y0 + 32)
            for s in segments:
                G.line(s, _alpha(colour, 0.55), 12)
            glow = G.img.reduce(SS).filter(ImageFilter.GaussianBlur(9))
            canvas.alpha_composite(glow, (G.x0, G.y0))

            Ln = Layer(x0 - 20, y0 - 16, x1 - x0 + 40, y1 - y0 + 32)
            for s in segments:
                Ln.line(s, colour, 4)
            if p["series"] and last_t is not None and t - last_t < 0.25:
                head = segments[-1][-1]
                Ln.circle(*head, 12, fill=_alpha(colour, 0.3))
                Ln.circle(*head, 6, fill=TEXT)
            for r in [r for r in p["reps"] if r["time_s"] <= t and "bottom_s" in r]:
                i = int(np.searchsorted(p["_times"], r["bottom_s"]))
                if i < len(p["series"]):
                    bx, by = self._X(p["series"][i][0]), self._Y(p["series"][i][1])
                    Ln.circle(bx, by, 8,
                               fill=GOOD if r["valid"] else BAD, outline=CARD, width=3)
            Ln.onto(canvas)

    def _draw_chips(self, L: Layer, t: float, done: list) -> None:
        bx0, by0, bx1, by1 = self.chipbox
        cols = 6 if bx1 - bx0 > 1200 else 5
        gap  = 14
        rows = sorted(
            ((p, r) for p, reps in zip(self.people, done) for r in reps),
            key=lambda pr: pr[1]["time_s"],
        )[-cols * 2:]
        x0, y0 = bx0 + 24, by0 + 66
        cw = (bx1 - bx0 - 48 - gap * (cols - 1)) / cols
        ch = (by1 - 20 - y0 - gap) / 2

        all_reps = [r for reps in done for r in reps]
        if all_reps:
            tempo = float(np.mean([r["measured"].get("tempo", 1) for r in all_reps]))
            L.text(bx1 - 28, by0 + 34, f"avg lowering {tempo:.1f}s",
                   15, "Medium", MUTED, anchor="rm")

        for slot in range(cols * 2):
            cx0 = x0 + (slot % cols) * (cw + gap)
            cy0 = y0 + (slot // cols) * (ch + gap)
            if slot >= len(rows):
                L.rrect(cx0, cy0, cx0 + cw, cy0 + ch, 18,
                        outline=_alpha(FAINT, 0.7), width=2)
                continue
            p, r  = rows[slot]
            a     = _ease((t - r["time_s"]) / 0.35)
            lift  = (1 - a) * 14
            colour    = GOOD if r["valid"] else BAD
            score_col = GOOD if r["score"] >= 85 else WARN if r["score"] >= 65 else BAD
            L.rrect(cx0, cy0 + lift, cx0 + cw, cy0 + ch + lift, 18,
                    fill=_alpha(_mix(CARD, colour, 0.16), a),
                    outline=_alpha(colour, 0.85 * a), width=2)
            tag = f"P{p['person']} · #{r['rep']}" if self.many else f"REP {r['rep']}"
            L.text(cx0 + 16, cy0 + 24 + lift, tag, 13, "SemiBold", _alpha(MUTED, a), anchor="lm")
            L.text(cx0 + cw / 2, cy0 + ch / 2 + 4 + lift, str(r["score"]), 38, "Bold",
                   _alpha(score_col if r["valid"] else TEXT, a), anchor="mm")
            word = "GOOD" if r["valid"] else self.ex["tags"].get(r["main_fault"], "NO REP")
            L.text(cx0 + cw / 2, cy0 + ch - 22 + lift, word, 14, "Bold",
                   _alpha(colour, a), anchor="mm")

    def _draw_view_badge(self, V: Layer) -> None:
        views = {r["view"] for p in self.people for r in p["reps"]}
        label = (" + ".join(sorted(views)) if views else "side").upper() + " VIEW"
        w     = V.width_of(label, 14, "SemiBold") + 32
        V.rrect(self.vx + 18, self.vy + 18, self.vx + 18 + w, self.vy + 52, 17,
                fill=(10, 14, 26, 150))
        V.text(self.vx + 18 + w / 2, self.vy + 35, label, 14, "SemiBold", TEXT, anchor="mm")

    def _draw_toast(self, V: Layer, t: float, done: list) -> None:
        recent = [(p, r) for p, reps in zip(self.people, done)
                  for r in reps if 0 <= t - r["time_s"] < 1.5][-2:]
        for k, (p, r) in enumerate(recent):
            age    = t - r["time_s"]
            a      = _ease(age / 0.25) * (1 - _ease((age - 1.2) / 0.3))
            colour = GOOD if r["valid"] else BAD
            title  = f"Rep {r['rep']}" if r["valid"] else "No rep"
            sub    = f"{r['score']} / 100" if r["valid"] else self.ex["focus"].get(r["main_fault"], "")
            if self.many:
                title = f"P{p['person']} · {title}"
            w = max(V.width_of(title, 24, "Bold"), V.width_of(sub, 15, "Medium")) + 84
            x = self.vx + (self.vw - w) / 2
            y = self.vy + self.vh - 120 - k * 92 + (1 - a) * 20
            V.rrect(x, y, x + w, y + 76, 22,
                    fill=(10, 14, 26, int(200 * a)), outline=_alpha(colour, a), width=2)
            V.circle(x + 34, y + 38, 14, fill=_alpha(colour, a))
            V.text(x + 60, y + 26, title, 24, "Bold", _alpha(TEXT, a), anchor="lm")
            V.text(x + 60, y + 54, sub,   15, "Medium", _alpha(MUTED, a), anchor="lm")

    def _draw_summary(self, V: Layer, progress: float) -> None:
        a    = _ease(progress)
        V.rrect(self.vx, self.vy, self.vx + self.vw, self.vy + self.vh,
                RADIUS, fill=(6, 9, 18, int(170 * a)))
        reps = [r for p in self.people for r in p["reps"]]
        if not reps:
            return
        good   = sum(r["valid"] for r in reps)
        best   = max(reps, key=lambda r: r["score"])
        faults = Counter(r["main_fault"] for r in reps if r["main_fault"])
        focus  = (self.ex["focus"].get(faults.most_common(1)[0][0], "")
                  if faults else "Great set. Keep it up!")
        avg    = float(np.mean([r["score"] for r in reps]))

        w  = min(self.vw - 48, 520)
        h  = 400
        x  = self.vx + (self.vw - w) / 2
        y  = self.vy + (self.vh - h) / 2 + (1 - a) * 60
        A  = lambda c, k=1.0: _alpha(c, a * k)

        V.rrect(x, y, x + w, y + h, 28, fill=A(CARD, 0.97), outline=A(CARD_EDGE), width=2)
        cx2 = x + w / 2
        V.text(cx2, y + 46,  "SET COMPLETE",          16, "Bold",   A(GOOD),  anchor="mm")
        V.text(cx2, y + 128, f"{good}/{len(reps)}",   92, "Bold",   A(TEXT),  anchor="mm")
        V.text(cx2, y + 196, "clean reps",            18, "Medium", A(MUTED), anchor="mm")
        for i, (label, value, colour) in enumerate((
            ("AVG SCORE", f"{avg:.0f}", GOOD if avg >= 85 else WARN if avg >= 65 else BAD),
            ("BEST REP",  f"#{best['rep']} · {best['score']}", TEXT),
        )):
            mx = x + w * (0.27 + 0.46 * i)
            V.text(mx, y + 244, label, 13, "SemiBold", A(MUTED), anchor="mm")
            V.text(mx, y + 276, value, 28, "Bold",     A(colour), anchor="mm")
        V.rrect(x + 28, y + h - 88, x + w - 28, y + h - 32, 18,
                fill=A(_mix(CARD, WARN if faults else GOOD, 0.18)))
        V.text(cx2, y + h - 60,
               f"Next set: {focus}" if faults else focus,
               17, "SemiBold", A(WARN if faults else GOOD), anchor="mm")


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------
def _media_duration(path: str) -> float:
    out = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", path],
        capture_output=True, text=True,
    ).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def render_dashboard(video_path: str) -> str:
    """
    Read the skeleton video and rep JSON produced by the analysis pipeline,
    render the full analytics dashboard, and return the path to the output file.
    """
    name         = os.path.splitext(os.path.basename(video_path))[0]
    report_path  = f"output/{name}_reps.json"
    skeleton     = f"output/{name}_skeleton.mp4"
    voice        = f"output/{name}_voice.m4a"
    out_path     = f"output/{name}_dashboard.mp4"

    for p in (report_path, skeleton):
        if not os.path.exists(p):
            sys.exit(f"Missing {p} — run the analysis pipeline first.")

    with open(report_path) as f:
        report = json.load(f)
    if not report["people"]:
        sys.exit("No person was detected in the video.")

    has_voice = os.path.exists(voice)
    if not has_voice:
        print("No voice track found — dashboard will be silent.")

    cap = cv2.VideoCapture(skeleton)
    fps_value = float(cap.get(cv2.CAP_PROP_FPS))
    report_fps = float(report.get("fps", 30.0) or 30.0)
    fps = fps_value if math.isfinite(fps_value) and fps_value > 0 else report_fps
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    width_value = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_value = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (not math.isfinite(width_value) or not math.isfinite(height_value)
            or width_value < 2 or height_value < 2):
        cap.release()
        raise RuntimeError("The analyzed skeleton video has invalid frame dimensions; dashboard rendering cannot continue.")
    dash = Dashboard(report, int(width_value), int(height_value))

    hold_s = 3.0
    if has_voice:
        hold_s = float(np.clip(_media_duration(voice) - report["duration_s"] + 0.6, 3.0, 6.0))

    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{dash.W}x{H}", "-r", f"{fps}", "-i", "-",
    ]
    if has_voice:
        cmd += ["-i", voice, "-map", "0:v", "-map", "1:a",
                "-af", "apad", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "19", "-preset", "medium",
            "-shortest", "-movflags", "+faststart", out_path]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    total_value = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = int(total_value) if math.isfinite(total_value) and total_value > 0 else 0
    i, last = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
        enc.stdin.write(dash.render(frame, i / fps).tobytes())
        i += 1
        if i % 30 == 0:
            print(f"  rendered {i}/{total} frames", end="\r")

    end_t = i / fps
    if last is not None:
        for k in range(int(hold_s * fps)):
            enc.stdin.write(dash.render(last, end_t, end_progress=k / (0.6 * fps)).tobytes())

    cap.release()
    enc.stdin.close()
    enc.wait()
    print(f"\nDashboard saved → {os.path.abspath(out_path)}  ({dash.W}×{H})")
    return out_path
