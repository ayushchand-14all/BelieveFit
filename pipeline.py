"""
pipeline.py — Core analysis pipeline.

Runs MediaPipe pose estimation on every frame, feeds measurements into the
exercise-specific coach, draws the annotated skeleton + live HUD, writes the
annotated video and a clean skeleton video, and saves the per-rep JSON report.

Supports:
  • Squats — SIDE view (depth, lean, heels, tempo) and FRONT view (depth, knees, tempo)
  • Push-ups — SIDE view (depth, body-line, tempo, lockout, head)
  • Up to 2 people tracked simultaneously

Usage (usually called via coach.py, but can run standalone):
    python pipeline.py videos/squat.mp4
    python pipeline.py videos/pushups.mp4 --exercise pushup --people 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from form_scorer import (
    BAD, GOOD, NEUTRAL, ORANGE,
    HipDrop, PushupCoach, SquatCoach, ViewDetector,
    SQUAT_DOWN_ANGLE, SQUAT_UP_ANGLE,
    LIVE_HEEL_LIMIT, LIVE_KNEE_LIMIT, LIVE_LEAN_LIMIT,
    PLANK_TILT, PLANK_KNEE, EARS,
    in_plank, is_horizontal, is_real_person_pushup, is_real_person_squat,
    _body_offset, _head_drop, _heel_rise, _knee_ratio, _torso_lean,
)
from pose_utils import (
    PERSON_COLOURS, SIDES,
    SideChooser, Smoother,
    angle, create_landmarker, draw_label, joint, landmarks_to_array,
)

MAX_PEOPLE       = 2
MIN_SEEN_SECONDS = 1.0   # a person's HUD card appears only after they've been visible this long


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _bone(frame, a, b, colour, t):
    a2, b2 = tuple(int(v) for v in a[:2]), tuple(int(v) for v in b[:2])
    cv2.line(frame, a2, b2, (255, 255, 255), t + 3, cv2.LINE_AA)
    cv2.line(frame, a2, b2, colour, t + 1, cv2.LINE_AA)


def _dot(frame, p, colour, t):
    p2 = tuple(int(v) for v in p[:2])
    cv2.circle(frame, p2, t + 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, p2, t + 2, colour, -1, cv2.LINE_AA)


def _score_colour(score: int) -> tuple:
    return GOOD if score >= 85 else ORANGE if score >= 65 else BAD


def _draw_squat_skeleton_side(frame, pts, side, person, lean, heel_up):
    t = max(2, frame.shape[1] // 300)
    j = lambda n: joint(pts, side, n)
    legs  = GOOD if person.knee is not None and person.knee <= 90 else NEUTRAL
    torso = BAD if lean > LIVE_LEAN_LIMIT else GOOD
    feet  = BAD if heel_up else GOOD
    _bone(frame, j("shoulder"), j("elbow"), person.colour, t)
    _bone(frame, j("elbow"),   j("wrist"), person.colour, t)
    _bone(frame, j("shoulder"), j("hip"),  torso,         t)
    _bone(frame, j("hip"),     j("knee"),  legs,          t)
    _bone(frame, j("knee"),    j("ankle"), legs,          t)
    for a, b in (("ankle", "heel"), ("heel", "foot"), ("ankle", "foot")):
        _bone(frame, j(a), j(b), feet, t)
    for n in SIDES[side]:
        _dot(frame, j(n), person.colour, t)


def _draw_squat_skeleton_front(frame, pts, person, knees_in):
    t    = max(2, frame.shape[1] // 300)
    deep = person.knee is not None and person.knee <= 90
    legs = BAD if knees_in else (GOOD if deep else NEUTRAL)
    L2   = lambda n: joint(pts, "left", n)
    R2   = lambda n: joint(pts, "right", n)
    _bone(frame, L2("shoulder"), R2("shoulder"), person.colour, t)
    _bone(frame, L2("hip"),      R2("hip"),       person.colour, t)
    for jj in (L2, R2):
        _bone(frame, jj("shoulder"), jj("elbow"), person.colour, t)
        _bone(frame, jj("elbow"),    jj("wrist"), person.colour, t)
        _bone(frame, jj("shoulder"), jj("hip"),   person.colour, t)
        _bone(frame, jj("hip"),      jj("knee"),  legs,          t)
        _bone(frame, jj("knee"),     jj("ankle"), legs,          t)
        _bone(frame, jj("ankle"),    jj("foot"),  person.colour, t)
        for n in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "foot"):
            _dot(frame, jj(n), person.colour, t)
    if knees_in:
        k = (L2("knee")[:2] + R2("knee")[:2]) / 2
        draw_label(frame, "KNEES IN", (k[0], k[1] - 50), BAD)


def _draw_pushup_skeleton(frame, pts, side, person, offset, head):
    t    = max(2, frame.shape[1] // 300)
    j    = lambda n: joint(pts, side, n)
    arms = GOOD if person.knee is not None and person.knee <= 90 else NEUTRAL
    body = BAD if abs(offset) > 0.07 else GOOD
    neck = BAD if head > 40.0 else person.colour
    _bone(frame, pts[EARS[side]], j("shoulder"), neck,         t)
    _bone(frame, j("shoulder"),   j("elbow"),    arms,         t)
    _bone(frame, j("elbow"),      j("wrist"),    arms,         t)
    _bone(frame, j("shoulder"),   j("hip"),      body,         t)
    _bone(frame, j("hip"),        j("knee"),     body,         t)
    _bone(frame, j("knee"),       j("ankle"),    body,         t)
    _bone(frame, j("ankle"),      j("foot"),     person.colour, t)
    for n in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "foot"):
        _dot(frame, j(n), person.colour, t)
    _dot(frame, pts[EARS[side]], person.colour, t)


def _draw_hud_card(frame, person, x, y, card_w):
    s      = card_w / 380
    h_card = int(250 * s)
    c      = person.colour
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + card_w, y + h_card), (30, 30, 30), -1)
    alpha = 0.85 if person.flash else 0.70
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + int(10 * s), y + h_card), c, -1)
    if person.flash:
        cv2.rectangle(frame, (x, y), (x + card_w, y + h_card), c, max(2, int(4 * s)))

    f   = cv2.FONT_HERSHEY_SIMPLEX
    put = lambda txt, px, py, sc, col, th=2: cv2.putText(
        frame, txt, (x + int(px * s), y + int(py * s)), f, sc * s, col,
        max(1, int(th * s)), cv2.LINE_AA,
    )
    coach = person.coach
    put(f"PERSON {person.index + 1}", 28, 40, 0.8, c)
    put(person.view.upper(),          200, 40, 0.6, (190, 190, 190))
    put(coach.stage,                  300, 40, 0.8, (255, 255, 255))
    put(str(coach.good_reps),          24, 140, 3.0, (255, 255, 255), 6)
    put(f"GOOD / {len(coach.reps)}",   28, 170, 0.7, (190, 190, 190))
    last = coach.reps[-1]["score"] if coach.reps else None
    put("LAST REP", 190, 75, 0.6, (190, 190, 190))
    put(str(last) if last is not None else "--", 190, 130, 1.8,
        _score_colour(last) if last is not None else (190, 190, 190), 4)
    avg = coach.average
    put(f"AVG {avg}" if avg is not None else "AVG --", 190, 170, 0.7, (230, 230, 230))
    # depth bar
    bx, by = x + int(28 * s), y + int(200 * s)
    bw, bh = card_w - int(56 * s), max(6, int(16 * s))
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
    if person.knee is not None:
        fill = float(np.clip((180 - person.knee) / (180 - 70), 0, 1))
        cv2.rectangle(frame, (bx, by), (bx + int(bw * fill), by + bh),
                      GOOD if person.knee <= 90 else ORANGE, -1)
    target = bx + int(bw * (180 - 90) / (180 - 70))
    cv2.line(frame, (target, by - 6), (target, by + bh + 6), (255, 255, 255), 2)
    put("DEPTH", 28, 238, 0.55, (190, 190, 190))


def _draw_feedback_strip(frame, person, x, y, card_w):
    if not person.feedback_frames:
        return
    rep    = person.feedback
    s      = card_w / 380
    h_strip = int(56 * s)
    colour = _score_colour(rep["score"]) if rep["valid"] else BAD
    cv2.rectangle(frame, (x, y), (x + card_w, y + h_strip), colour, -1)
    text   = f"{rep['score']}  {rep['cue']}" if rep["valid"] else rep["cue"].upper()
    scale  = 0.75 * s
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    if tw > card_w - 24:
        scale *= (card_w - 24) / tw
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(frame, text,
                (x + (card_w - tw) // 2, y + (h_strip + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    person.feedback_frames -= 1


# ---------------------------------------------------------------------------
# Person state
# ---------------------------------------------------------------------------
class _PersonState:
    def __init__(self, index: int, fps: float, exercise: str) -> None:
        self.index        = index
        self.colour       = PERSON_COLOURS[index % len(PERSON_COLOURS)]
        self.smoother     = Smoother(alpha=0.5)
        self.side_chooser = SideChooser()
        self.view_detector = ViewDetector()
        self.hip_drop     = HipDrop(fps)
        self.coach        = SquatCoach(fps) if exercise == "squat" else PushupCoach(fps)
        self.knee: Optional[float] = None   # for push-ups this is actually the elbow angle
        self.view         = "side"
        self.seen         = 0
        self.horizontal   = 0
        self.series: list = []
        self.flash        = 0
        self.feedback: Optional[dict] = None
        self.feedback_frames = 0


# ---------------------------------------------------------------------------
# Video writer helper (retries if file is locked)
# ---------------------------------------------------------------------------
def _open_writer(path: str, fps: float, w: int, h: int):
    import time
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    candidates = [path] + [path.replace(".mp4", f"_{n}.mp4") for n in range(2, 10)]
    for candidate in candidates:
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except (PermissionError, OSError):
            continue
        writer = cv2.VideoWriter(candidate, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, candidate
    # Fallback to unique timestamped file
    ts_candidate = path.replace(".mp4", f"_{int(time.time() * 1000)}.mp4")
    writer = cv2.VideoWriter(ts_candidate, fourcc, fps, (w, h))
    if writer.isOpened():
        return writer, ts_candidate
    raise RuntimeError(f"Could not initialize video writer for {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_analysis(
    video_path:  str,
    exercise:    str = "squat",
    max_people:  int = MAX_PEOPLE,
    progress_callback = None,
) -> str:
    """
    Analyse *video_path*, write annotated + skeleton videos and a JSON report,
    and return the path to the JSON report.
    Optimised for large video files: decodes frames in stream, resizes for pose inference,
    decimates ultra-high frame rates, and manages CPU/memory safely.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Some phone recordings/container combinations report NaN/Inf metadata.
    # Never pass those values to int()/round(); use safe defaults or fail clearly.
    raw_fps_value = float(cap.get(cv2.CAP_PROP_FPS))
    raw_fps = raw_fps_value if math.isfinite(raw_fps_value) and raw_fps_value > 0 else 30.0

    width_value = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_value = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (not math.isfinite(width_value) or not math.isfinite(height_value)
            or width_value < 2 or height_value < 2):
        cap.release()
        raise RuntimeError(
            "The uploaded video has invalid frame dimensions. Please re-export it as MP4 (H.264) and upload again."
        )
    w, h = int(width_value), int(height_value)

    frame_count_value = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = int(frame_count_value) if math.isfinite(frame_count_value) and frame_count_value > 0 else 0

    # Intelligent downscale for huge 4K/2K resolution videos to avoid CPU & disk overload
    max_out_dim = 1280
    out_w, out_h = w, h
    scale_out = 1.0
    if max(w, h) > max_out_dim:
        scale_out = max_out_dim / max(w, h)
        out_w = int(w * scale_out)
        out_h = int(h * scale_out)
        out_w -= out_w % 2
        out_h -= out_h % 2

    # High frame rate decimation (e.g. 60fps/120fps -> ~30fps target)
    frame_step = max(1, int(round(raw_fps / 30.0))) if raw_fps > 35.0 else 1
    fps = raw_fps / frame_step

    # Pose inference resolution (MediaPipe optimal at ~640p)
    max_infer_dim = 640
    infer_scale = min(1.0, max_infer_dim / max(w, h))
    infer_w = int(w * infer_scale)
    infer_h = int(h * infer_scale)

    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.environ.get("BELIEVFIT_DATA_DIR", BASE_DIR)
    output_dir  = os.path.join(data_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    name        = os.path.splitext(os.path.basename(video_path))[0]
    out_path    = os.path.join(output_dir, f"{name}_form.mp4")
    skel_path   = os.path.join(output_dir, f"{name}_skeleton.mp4")
    json_path   = os.path.join(output_dir, f"{name}_reps.json")

    writer,       out_path  = _open_writer(out_path,  fps, out_w, out_h)
    skel_writer, _skel_path = _open_writer(skel_path, fps, out_w, out_h)

    landmarker = create_landmarker(max_people)
    people     = [_PersonState(i, fps, exercise) for i in range(max_people)]
    cues: list = []
    # Timestamped normalized landmarks for browser-side, frame-synchronized overlays.
    pose_frames: list = []
    frame_idx  = 0
    analysed_idx = 0
    card_w     = int(out_w * 0.2) if out_w >= out_h else int(out_w * (0.42 if max_people > 1 else 0.5))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Frame decimation for ultra-high FPS videos
            if frame_step > 1 and (frame_idx % frame_step != 0):
                frame_idx += 1
                continue

            # Scale frame for output writer if original was 4K
            if scale_out < 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Downscale for MediaPipe pose estimation (runs 4x faster with identical normalized output)
            if infer_scale < 1.0:
                infer_rgb = cv2.resize(rgb, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
            else:
                infer_rgb = rgb

            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb),
                int(analysed_idx * 1000 / fps),
            )

            if exercise == "squat":
                bodies = [landmarks_to_array(lms, out_w, out_h) for lms in result.pose_landmarks]
                bodies = [b for b in bodies if is_real_person_squat(b, out_h)]
            else:
                bodies = [landmarks_to_array(lms, out_w, out_h) for lms in result.pose_landmarks]
                bodies = [b for b in bodies if is_real_person_pushup(b, out_w, out_h)]

            bodies.sort(key=lambda b: (b[23, 0] + b[24, 0]) / 2)   # left → right

            labels: list = []
            primary_pose_frame = None

            for i, person in enumerate(people):
                if i >= len(bodies):
                    person.smoother.reset()
                    person.knee = None
                    continue

                pts = person.smoother(bodies[i])
                person.seen += 1
                person.horizontal += int(is_horizontal(pts))

                rep: Optional[dict] = None

                if exercise == "squat":
                    view = person.view = person.view_detector(pts)
                    if view == "side":
                        side  = person.side_chooser(pts)
                        knee  = angle(joint(pts, side, "hip"), joint(pts, side, "knee"),
                                      joint(pts, side, "ankle"))
                        lean  = _torso_lean(pts, side)
                        heel  = _heel_rise(pts, side)
                        person.knee = knee
                        rep = person.coach.update(knee, analysed_idx, "side", lean=lean, heel=heel)
                        heel_up = (person.coach.stage == "DOWN" and
                                   heel - person.coach._baseline > LIVE_HEEL_LIMIT)
                        _draw_squat_skeleton_side(frame, pts, side, person, lean, heel_up)
                        k = joint(pts, side, "knee")
                        dx = 110 if joint(pts, side, "foot")[0] > joint(pts, side, "heel")[0] else -110
                        labels.append((f"{int(knee)} deg", (k[0] + dx, k[1]), person.colour))
                    else:
                        knee = person.hip_drop(pts)
                        kr   = _knee_ratio(pts)
                        person.knee = knee
                        rep  = person.coach.update(knee, analysed_idx, "front", kratio=kr)
                        _draw_squat_skeleton_front(
                            frame, pts, person,
                            knees_in=(knee < 140 and kr < LIVE_KNEE_LIMIT),
                        )
                else:
                    side   = person.side_chooser(pts)
                    elbow  = angle(joint(pts, side, "shoulder"), joint(pts, side, "elbow"),
                                   joint(pts, side, "wrist"))
                    offset = _body_offset(pts, side)
                    head   = _head_drop(pts, side)
                    person.knee = elbow
                    person.view = "side"

                    d    = joint(pts, side, "ankle")[:2] - joint(pts, side, "shoulder")[:2]
                    person.horizontal += int(
                        float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-6))) < PLANK_TILT
                    )
                    if in_plank(pts, side):
                        rep = person.coach.update(elbow, offset, head, analysed_idx)
                    else:
                        person.coach.abandon()

                    _draw_pushup_skeleton(frame, pts, side, person, offset, head)
                    e = joint(pts, side, "elbow")
                    labels.append((f"{int(elbow)} deg", (e[0], e[1] + 70), person.colour))

                # Keep the first detected athlete's pose for synchronized browser overlay.
                # Coordinates are normalized to the analyzed output frame dimensions.
                if i == 0:
                    if exercise == "squat":
                        if person.view == "side":
                            segment_state = {
                                "arms": "neutral",
                                "torso": "bad" if lean > LIVE_LEAN_LIMIT else "good",
                                "legs": "good" if knee <= 100 else "neutral",
                                "feet": "bad" if heel_up else "good",
                            }
                        else:
                            segment_state = {
                                "arms": "neutral",
                                "torso": "good",
                                "legs": "bad" if (knee < 140 and kr < LIVE_KNEE_LIMIT) else ("good" if knee <= 100 else "neutral"),
                                "feet": "neutral",
                            }
                    else:
                        segment_state = {
                            "arms": "good" if elbow <= 100 else "neutral",
                            "torso": "bad" if abs(offset) > 0.07 else "good",
                            "legs": "bad" if abs(offset) > 0.07 else "good",
                            "feet": "neutral",
                            "neck": "bad" if head > 40.0 else "good",
                        }
                    primary_pose_frame = {
                        "t": round(analysed_idx / fps, 4),
                        "landmarks": [
                            [round(float(pt[0]) / max(out_w, 1), 6),
                             round(float(pt[1]) / max(out_h, 1), 6),
                             round(float(pt[2]), 4)]
                            for pt in pts
                        ],
                        "segments": segment_state,
                    }

                if rep:
                    person.flash          = int(fps * 0.4)
                    person.feedback       = rep
                    person.feedback_frames = int(fps * 1.5)
                    cues.append({
                        "time_s": rep["time_s"], "person": i + 1,
                        "rep": rep["rep"], "score": rep["score"],
                        "valid": rep["valid"], "text": rep["cue"],
                    })

                person.series.append([round(analysed_idx / fps, 3), round(float(person.knee or 90), 1)])

                head_pt = pts[0]
                if len(bodies) > 1 or exercise == "squat":
                    labels.append((f"Person {i + 1}", (head_pt[0], head_pt[1] - 60), person.colour))

            if primary_pose_frame is not None:
                pose_frames.append(primary_pose_frame)

            # write clean skeleton frame, then add labels
            skel_writer.write(frame)

            for text, pos, colour in labels:
                draw_label(frame, text, pos, colour)

            shown = [p for p in people if p.seen >= fps * MIN_SEEN_SECONDS]
            for slot, person in enumerate(shown):
                x = 30 if slot == 0 else out_w - card_w - 30
                _draw_hud_card(frame, person, x, 30, card_w)
                _draw_feedback_strip(frame, person, x, 30 + int(250 * card_w / 380) + 8, card_w)
                person.flash = max(0, person.flash - 1)

            writer.write(frame)
            frame_idx += 1
            analysed_idx += 1

            if progress_callback and (frame_idx % 15 == 0 or frame_idx >= total):
                try:
                    pct = min(0.99, frame_idx / total)
                    progress_callback(pct, f"Analyzing frame {frame_idx}/{total} ({int(pct * 100)}%)...")
                except Exception:
                    pass

            if frame_idx % 30 == 0:
                print(f"  analysed {frame_idx}/{total} frames", end="\r")

    finally:
        for resource in (cap, writer, skel_writer):
            if resource is not None:
                try:
                    resource.release()
                except Exception:
                    pass
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass

    real = [p for p in people if p.seen >= fps * MIN_SEEN_SECONDS]
    report = {
        "exercise":   exercise,
        "angle_name": "Knee angle" if exercise == "squat" else "Elbow angle",
        "video":      video_path,
        "fps":        fps,
        "duration_s": round(analysed_idx / fps, 2),
        "pose_frames": pose_frames,
        "people": [
            {
                "person":           p.index + 1,
                "colour_bgr":       list(p.colour),
                "reps":             p.coach.reps,
                "average_score":    p.coach.average,
                "series":           p.series,
                "horizontal_share": round(p.horizontal / max(p.seen, 1), 2),
            }
            for p in real
        ],
        "cues": [c for c in cues if any(p.index + 1 == c["person"] for p in real)],
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    if progress_callback:
        try:
            progress_callback(1.0, "Analysis complete!")
        except Exception:
            pass

    size_mb = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0
    print(f"\nAnnotated video → {os.path.abspath(out_path)} ({size_mb:.1f} MB)")
    print(f"Rep report      → {os.path.abspath(json_path)}")

    # Helpful hints when no reps were detected
    if not any(p.coach.reps for p in real):
        if any(p.horizontal > p.seen / 2 for p in real) and exercise == "squat":
            print("\nHint: person is horizontal most of the time — "
                  f"try:  python coach.py {video_path} --exercise pushup")
        elif exercise == "pushup" and real and all(p.horizontal < p.seen / 4 for p in real):
            print("\nHint: person is upright — "
                  f"try:  python coach.py {video_path}")

    for p in real:
        print(f"\nPerson {p.index + 1}: "
              f"{p.coach.good_reps} good reps / {len(p.coach.reps)} total  |  "
              f"avg score {p.coach.average}")
        for r in p.coach.reps:
            tag = "GOOD  " if r["valid"] else "NOREP"
            m   = "  ".join(f"{k} {v}" for k, v in r["measured"].items())
            print(f"  [{tag}] rep {r['rep']:>2}  {r['score']:>3}/100  {m}  → {r['cue']}")

    return json_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyse a workout video and produce rep data.")
    ap.add_argument("video")
    ap.add_argument("--exercise", choices=["squat", "pushup"], default="squat")
    ap.add_argument("--people",   type=int, default=MAX_PEOPLE)
    a = ap.parse_args()
    run_analysis(a.video, a.exercise, a.people)
