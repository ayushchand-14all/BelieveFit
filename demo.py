"""
demo.py — Offline demo that works WITHOUT a real workout video.

It synthesises a short fake angle series (a squat set and a push-up set),
runs the form scorer directly on the data, and renders a static dashboard
PNG so you can verify the full visual pipeline runs correctly.

Run:
    python demo.py

This will:
  1. Simulate a 5-rep squat session with varied form quality
  2. Simulate a 4-rep push-up session
  3. Write  output/demo_squat_reps.json
  4. Write  output/demo_pushup_reps.json
  5. Print  a text summary of every rep's score and fault
  6. Render output/demo_squat_dashboard_frame.png  (1 static dashboard frame)
  7. Render output/demo_pushup_dashboard_frame.png

No video camera, no internet, and no MediaPipe model download needed.
"""
from __future__ import annotations

import json
import math
import os
import sys

# cv2 lazily imported

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

def _fallback_sine(t_total, fps, start, bottom, n_reps):
    steps = int(t_total * fps)
    series = []
    reps = []
    rep_duration = t_total / n_reps
    for i in range(steps):
        t = i / fps
        cycle = (t % rep_duration) / rep_duration
        # Sine curve between start and bottom
        ang = bottom + (start - bottom) * (0.5 + 0.5 * math.cos(cycle * 2 * math.pi))
        series.append([round(t, 3), round(ang, 1)])
    for r in range(1, n_reps + 1):
        reps.append({
            'rep': r,
            'start_s': round((r - 1) * rep_duration, 2),
            'bottom_s': round(((r - 1) + 0.5) * rep_duration, 2),
            'time_s': round(r * rep_duration, 2),
            'view': 'side',
            'score': 85 + (r % 10),
            'valid': True,
            'measured': {'depth': bottom + 5, 'tempo': 1.8},
            'points_lost': {},
            'main_fault': None,
            'cue': 'Good rep. Strong lockout.',
            'live_faults': []
        })
    return series, reps

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
def _sine_wave(
    t_arr:    np.ndarray,
    start:    float,
    bottom:   float,
    t_start:  float,
    t_bottom: float,
    t_end:    float,
) -> np.ndarray:
    """Smooth angle curve: starts at *start*, dips to *bottom*, returns to *start*."""
    out = np.full_like(t_arr, start, dtype=float)
    desc  = (t_arr >= t_start) & (t_arr < t_bottom)
    ascen = (t_arr >= t_bottom) & (t_arr <= t_end)
    phase_d = (t_arr[desc]  - t_start)  / (t_bottom - t_start)
    phase_a = (t_arr[ascen] - t_bottom) / (t_end - t_bottom)
    out[desc]  = start  + (bottom - start) * (1 - np.cos(phase_d  * math.pi)) / 2
    out[ascen] = bottom + (start  - bottom) * (1 - np.cos(phase_a * math.pi)) / 2
    return out


def build_squat_series(fps: float = 30.0) -> tuple[list, list]:
    if not HAVE_NUMPY:
        return _fallback_sine(20.0, fps, 170.0, 85.0, 5)
    """
    Simulate 5 squats with varying quality:
      Rep 1 — perfect (depth 85°, slow, chest up)
      Rep 2 — good    (depth 92°, decent tempo)
      Rep 3 — shallow (depth 130°) → no-rep
      Rep 4 — good    (depth 88°)
      Rep 5 — perfect (depth 82°)

    Returns (angle_series, rep_dicts).
    """
    t_total = 20.0
    t_arr   = np.arange(0, t_total, 1 / fps)
    angles  = np.full(len(t_arr), 170.0)

    reps_def = [
        # (t_start, t_bottom, t_end, min_angle, tempo_ok)
        (1.0,  2.5,  4.0,  85,  True),   # perfect
        (5.0,  5.9,  7.2,  92,  True),   # good
        (8.0,  8.6,  9.5, 130,  True),   # half-squat → no rep
        (10.5, 11.5, 13.0, 88,  True),   # good
        (14.0, 15.5, 17.0, 82,  True),   # perfect
    ]
    for t_s, t_b, t_e, bot, _ in reps_def:
        mask = (t_arr >= t_s) & (t_arr <= t_e)
        angles[mask] = _sine_wave(t_arr[mask], 170.0, float(bot), t_s, t_b, t_e)

    series = [[round(float(t), 3), round(float(a), 1)] for t, a in zip(t_arr, angles)]
    return series, _score_squat_series(series, fps, reps_def)


def _score_squat_series(series, fps, reps_def):
    """Build rep dicts directly from the simulation data (no MediaPipe needed)."""
    from form_scorer import SquatCoach
    coach = SquatCoach(fps)
    for t, a in series:
        frame = int(round(t * fps))
        coach.update(a, frame, "side", lean=12.0, heel=0.02)
    return coach.reps


def build_pushup_series(fps: float = 30.0) -> tuple[list, list]:
    if not HAVE_NUMPY:
        return _fallback_sine(16.0, fps, 165.0, 75.0, 4)
    """
    Simulate 4 push-ups:
      Rep 1 — perfect (elbow 85°, good body line)
      Rep 2 — shallow (elbow 108°) → no-rep
      Rep 3 — good    (elbow 92°)
      Rep 4 — perfect (elbow 80°)
    """
    t_total = 16.0
    t_arr   = np.arange(0, t_total, 1 / fps)
    angles  = np.full(len(t_arr), 165.0)

    reps_def = [
        (1.0,  2.0,  3.2,  85),
        (4.5,  5.1,  5.9, 108),   # half → no-rep
        (7.0,  7.9,  9.0,  92),
        (10.5, 11.5, 12.8, 80),
    ]
    for t_s, t_b, t_e, bot in reps_def:
        mask = (t_arr >= t_s) & (t_arr <= t_e)
        angles[mask] = _sine_wave(t_arr[mask], 165.0, float(bot), t_s, t_b, t_e)

    series = [[round(float(t), 3), round(float(a), 1)] for t, a in zip(t_arr, angles)]
    return series, _score_pushup_series(series, fps)


def _score_pushup_series(series, fps):
    from form_scorer import PushupCoach, in_plank
    coach = PushupCoach(fps)
    for t, a in series:
        frame = int(round(t * fps))
        # Simulate being in plank for all frames where the angle is below 160
        if a < 160:
            coach.update(a, 0.01, 15.0, frame)   # small offset, normal head
    return coach.reps


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def print_summary(title: str, reps: list) -> None:
    print(f"\n{'─'*56}")
    print(f"  {title}")
    print(f"{'─'*56}")
    good = sum(r["valid"] for r in reps)
    scores = [r["score"] for r in reps]
    avg = round(sum(scores) / len(scores)) if scores else 0
    print(f"  Good reps: {good}/{len(reps)}    Average score: {avg}/100\n")
    for r in reps:
        tag    = "✓ GOOD " if r["valid"] else "✗ NOREP"
        m_str  = "  ".join(f"{k}={v}" for k, v in r["measured"].items())
        fault  = r["main_fault"] or "none"
        print(f"  Rep {r['rep']:>2}  [{tag}]  {r['score']:>3}/100  "
              f"fault={fault:<8}  {m_str}")
        print(f"          → {r['cue']}")
    print()


# ---------------------------------------------------------------------------
# Static dashboard frame
# ---------------------------------------------------------------------------
def render_static_frame(report: dict, out_png: str) -> None:
    """Render one dashboard frame at t=duration (all reps visible) → PNG."""
    from dashboard import Dashboard

    # Build a tiny black frame the right aspect ratio (16:9)
    vw, vh = 640, 360
    black_frame = np.zeros((vh, vw, 3), dtype=np.uint8)

    dash  = Dashboard(report, vw, vh)
    t     = report["duration_s"]
    frame = dash.render(black_frame, t, end_progress=1.0)

    os.makedirs("output", exist_ok=True)
    import cv2
    cv2.imwrite(out_png, frame)
    print(f"  Dashboard frame saved → {os.path.abspath(out_png)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 56)
    print("  AI Gym Coach — Offline Demo")
    print("=" * 56)
    print("  Simulating reps (no camera or model needed) …\n")

    os.makedirs("output", exist_ok=True)

    # ---- Squats
    squat_series, squat_reps = build_squat_series()
    squat_report = {
        "exercise":   "squat",
        "angle_name": "Knee angle",
        "video":      "demo_squat",
        "fps":        30.0,
        "duration_s": squat_series[-1][0] if squat_series else 20.0,
        "people": [{
            "person":           1,
            "colour_bgr":       [80, 200, 120],
            "reps":             squat_reps,
            "average_score":    (round(sum(r["score"] for r in squat_reps) / len(squat_reps))
                                 if squat_reps else None),
            "series":           squat_series,
            "horizontal_share": 0.0,
        }],
        "cues": [],
    }
    with open("output/demo_squat_reps.json", "w") as f:
        json.dump(squat_report, f, indent=2)
    print_summary("SQUAT SESSION", squat_reps)

    # ---- Push-ups
    pushup_series, pushup_reps = build_pushup_series()
    pushup_report = {
        "exercise":   "pushup",
        "angle_name": "Elbow angle",
        "video":      "demo_pushup",
        "fps":        30.0,
        "duration_s": pushup_series[-1][0] if pushup_series else 16.0,
        "people": [{
            "person":           1,
            "colour_bgr":       [80, 200, 120],
            "reps":             pushup_reps,
            "average_score":    (round(sum(r["score"] for r in pushup_reps) / len(pushup_reps))
                                 if pushup_reps else None),
            "series":           pushup_series,
            "horizontal_share": 0.9,
        }],
        "cues": [],
    }
    with open("output/demo_pushup_reps.json", "w") as f:
        json.dump(pushup_report, f, indent=2)
    print_summary("PUSH-UP SESSION", pushup_reps)

    # ---- Dashboard frames
    print("Rendering static dashboard frames …")
    render_static_frame(squat_report,  "output/demo_squat_dashboard_frame.png")
    render_static_frame(pushup_report, "output/demo_pushup_dashboard_frame.png")

    print("\n" + "=" * 56)
    print("  Demo complete!  Check the output/ folder:")
    print("    output/demo_squat_reps.json")
    print("    output/demo_pushup_reps.json")
    print("    output/demo_squat_dashboard_frame.png")
    print("    output/demo_pushup_dashboard_frame.png")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
