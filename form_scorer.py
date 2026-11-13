"""
form_scorer.py — Squat & push-up form analysis engine.

For squats it handles both SIDE view (depth, torso lean, heels, tempo)
and FRONT view (depth via hip-drop, knee cave, tempo).
For push-ups it covers SIDE view (elbow depth, body line, tempo, lockout, head).

Each exercise exposes a *Coach* class whose `.update()` method is called once
per frame and returns a finished-rep dict when a rep completes.

Rep dict schema:
  rep         int      1-based rep number
  start_s     float    time the rep started
  bottom_s    float    time the deepest point was reached
  time_s      float    time the rep finished
  view        str      "side" | "front"
  score       int      0–100
  valid       bool     True if the rep counts
  measured    dict     raw measured values
  points_lost dict     penalty per rule
  main_fault  str|None the most costly fault
  cue         str      spoken coaching cue
  live_faults list     faults detected mid-rep (for the voice coach)
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from pose_utils import SIDES, angle, joint

# ---------------------------------------------------------------------------
# Shared colours (BGR)
# ---------------------------------------------------------------------------
GOOD    = (90, 210, 110)
BAD     = (60, 60, 235)
NEUTRAL = (255, 255, 255)
ORANGE  = (40, 170, 255)

# ---------------------------------------------------------------------------
# Rep counting thresholds
# ---------------------------------------------------------------------------
SQUAT_UP_ANGLE   = 160   # degrees — standing, rep can finish
SQUAT_DOWN_ANGLE = 100   # degrees — deep enough to count

PUSHUP_UP_ANGLE   = 150  # elbow — arms extended
PUSHUP_DOWN_ANGLE = 100  # elbow — chest low enough


# ===========================================================================
# Scoring helpers
# ===========================================================================
def _penalty(value: float, rule: tuple, lower_is_worse: bool = False) -> float:
    """Points lost, linearly scaled between the rule's start and end thresholds."""
    points, start, end = rule
    if lower_is_worse:
        value, start, end = -value, -start, -end
    return points * float(np.clip((value - start) / (end - start), 0.0, 1.0))


# ===========================================================================
# View detection (squat only)
# ===========================================================================
class ViewDetector:
    """
    Classifies the camera as SIDE or FRONT view.

    One side clearly hidden → side.  Wide shoulder-to-torso ratio → front.
    15-frame median + hysteresis keeps the label stable.
    """

    def __init__(self) -> None:
        self.view: Optional[str] = None
        self.history: deque = deque(maxlen=15)

    def __call__(self, pts: np.ndarray) -> str:
        vis = [
            float(np.mean([pts[SIDES[s][n], 2]
                           for n in ("shoulder", "hip", "knee", "ankle")]))
            for s in SIDES
        ]
        hidden_side = abs(vis[0] - vis[1]) > 0.3

        def _mid(name: str) -> np.ndarray:
            return (joint(pts, "left", name)[:2] + joint(pts, "right", name)[:2]) / 2

        if hidden_side:
            self.history.append(0.0)
        else:
            width = abs(joint(pts, "left", "shoulder")[0] - joint(pts, "right", "shoulder")[0])
            torso = float(np.linalg.norm(_mid("shoulder") - _mid("hip")))
            self.history.append(width / (torso + 1e-6))

        r = float(np.median(self.history))
        if self.view is None:
            self.view = "front" if r > 0.5 else "side"
        elif self.view == "side" and r > 0.6:
            self.view = "front"
        elif self.view == "front" and r < 0.4:
            self.view = "side"
        return self.view


class HipDrop:
    """
    Estimate squat depth from the front by tracking how far the hips descend.

    Converts to an equivalent knee-angle (180° = standing, 90° = parallel).
    """

    def __init__(self, fps: float) -> None:
        self.recent: deque = deque(maxlen=int(fps * 4))

    def __call__(self, pts: np.ndarray) -> float:
        def _mid(name: str) -> np.ndarray:
            return (joint(pts, "left", name)[:2] + joint(pts, "right", name)[:2]) / 2

        height = float(_mid("ankle")[1] - _mid("hip")[1])
        self.recent.append(height)
        standing = max(self.recent)
        ratio = height / (standing + 1e-6)
        return float(np.clip(180 - 200 * (1 - ratio), 40, 180))


# ===========================================================================
# Squat form scorer
# ===========================================================================
# Scoring rules: (max_points, threshold_where_losing_starts, threshold_all_lost)
_SQUAT_SIDE_RULES = {
    "depth": (35, 90.0, 115.0),   # knee angle at bottom (deg)
    "lean":  (25, 45.0, 65.0),    # max torso lean from vertical (deg)
    "heels": (15, 0.06, 0.18),    # heel rise fraction of shin length
    "tempo": (25, 1.0,  0.4),     # lowering time (s); less = worse
}
_SQUAT_FRONT_RULES = {
    "depth": (40, 90.0, 115.0),
    "knees": (35, 0.90, 0.65),    # knee gap / ankle gap; less = caving
    "tempo": (25, 1.0,  0.4),
}
_SQUAT_LOWER_IS_WORSE = {"tempo", "knees"}

_SQUAT_NO_REP_SCORE  = 70
_SQUAT_NO_REP_LEAN   = 60.0
_SQUAT_NO_REP_HEELS  = 0.15
_SQUAT_NO_REP_KNEES  = 0.75
_SQUAT_SHALLOW_ANGLE = 135   # a dip below this that never reaches depth = half squat

LIVE_LEAN_LIMIT  = 50.0
LIVE_HEEL_LIMIT  = 0.10
LIVE_KNEE_LIMIT  = 0.80

SQUAT_CUES = {
    "depth": "Go a little deeper",
    "lean":  "Keep your chest up",
    "heels": "Keep your heels down",
    "knees": "Push your knees out",
    "tempo": "Slow down on the way down",
    None:    "Great rep",
}


def _torso_lean(pts: np.ndarray, side: str) -> float:
    v = joint(pts, side, "shoulder")[:2] - joint(pts, side, "hip")[:2]
    cos = -v[1] / (np.linalg.norm(v) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _heel_rise(pts: np.ndarray, side: str) -> float:
    heel, toe = joint(pts, side, "heel"), joint(pts, side, "foot")
    shin = float(np.linalg.norm(
        joint(pts, side, "knee")[:2] - joint(pts, side, "ankle")[:2]
    ))
    return float((toe[1] - heel[1]) / (shin + 1e-6))


def _knee_ratio(pts: np.ndarray) -> float:
    knees  = abs(joint(pts, "left", "knee")[0]  - joint(pts, "right", "knee")[0])
    ankles = abs(joint(pts, "left", "ankle")[0] - joint(pts, "right", "ankle")[0])
    return float(knees / (ankles + 1e-6))


class SquatCoach:
    """
    State machine that counts squat reps and scores each one.

    Call `.update()` on every frame; it returns a rep dict when a rep finishes.
    """

    def __init__(self, fps: float) -> None:
        self.fps  = max(float(fps), 1.0)
        self.stage = "UP"
        self.reps: list[dict] = []
        self._heel_baseline: list[float] = []
        self._reset_rep()

    def _reset_rep(self) -> None:
        self._rep_start:    Optional[int]   = None
        self._min_knee:     float           = 180.0
        self._bottom_frame: Optional[int]   = None
        self._max_lean:     float           = 0.0
        self._max_heel:     float           = 0.0
        self._min_kratio:   float           = 9.9
        self._front_frames  = 0
        self._side_frames   = 0
        self._live_faults:  dict[str, int]  = {}

    @property
    def _baseline(self) -> float:
        return float(np.median(self._heel_baseline)) if self._heel_baseline else 0.0

    def update(
        self,
        knee: float,
        frame_idx: int,
        view: str,
        lean: float = 0.0,
        heel: float = 0.0,
        kratio: Optional[float] = None,
    ) -> Optional[dict]:
        """Return finished-rep dict or None."""
        # --- settled back at the top (between reps)
        if self.stage == "UP" and knee >= SQUAT_UP_ANGLE:
            half = None
            if self._rep_start is not None and self._min_knee < _SQUAT_SHALLOW_ANGLE:
                half = self._score(frame_idx)
                self.reps.append(half)
            if view == "side":
                self._heel_baseline = (self._heel_baseline + [heel])[-60:]
            self._reset_rep()
            return half

        # --- descending or at the bottom
        if self._rep_start is None:
            self._rep_start = frame_idx
        if knee < self._min_knee:
            self._min_knee, self._bottom_frame = knee, frame_idx

        if view == "front":
            self._front_frames += 1
            if kratio is not None and knee < 140:
                self._min_kratio = min(self._min_kratio, kratio)
                if kratio < LIVE_KNEE_LIMIT:
                    self._live_faults.setdefault("knees", frame_idx)
        else:
            self._side_frames += 1
            self._max_lean = max(self._max_lean, lean)
            self._max_heel = max(self._max_heel, heel - self._baseline)
            if lean > LIVE_LEAN_LIMIT:
                self._live_faults.setdefault("lean", frame_idx)
            if self.stage == "DOWN" and heel - self._baseline > LIVE_HEEL_LIMIT:
                self._live_faults.setdefault("heels", frame_idx)

        # --- state transitions
        if self.stage == "UP" and knee < SQUAT_DOWN_ANGLE:
            self.stage = "DOWN"
        elif self.stage == "DOWN" and knee >= SQUAT_UP_ANGLE:
            self.stage = "UP"
            rep = self._score(frame_idx)
            self.reps.append(rep)
            self._reset_rep()
            return rep
        return None

    def _score(self, frame_idx: int) -> dict:
        view    = "front" if self._front_frames > self._side_frames else "side"
        descent = ((self._bottom_frame or frame_idx) - (self._rep_start or frame_idx)) / self.fps

        measured: dict = {"depth": self._min_knee, "tempo": descent}
        if view == "side":
            rules = _SQUAT_SIDE_RULES
            measured.update(lean=self._max_lean, heels=self._max_heel)
        else:
            rules = _SQUAT_FRONT_RULES
            measured["knees"] = self._min_kratio if self._min_kratio < 9 else 1.0

        lost    = {k: _penalty(measured[k], rules[k], k in _SQUAT_LOWER_IS_WORSE) for k in rules}
        worst_k = max(lost, key=lost.get)  # type: ignore[arg-type]
        worst   = worst_k if lost[worst_k] >= 3 else None
        score   = max(0, min(100, int(round(100 - sum(lost.values())))))

        broken: list[str] = []
        if self._min_knee >= SQUAT_DOWN_ANGLE:
            broken.append("depth")
        if view == "front" and measured["knees"] < _SQUAT_NO_REP_KNEES:
            broken.append("knees")
        if view == "side"  and self._max_lean  > _SQUAT_NO_REP_LEAN:
            broken.append("lean")
        if view == "side"  and self._max_heel  > _SQUAT_NO_REP_HEELS:
            broken.append("heels")
        valid = not broken and score >= _SQUAT_NO_REP_SCORE
        if broken:
            worst = broken[0] if broken[0] != "depth" or len(broken) == 1 else broken[1]

        return {
            "rep":         len(self.reps) + 1,
            "start_s":     round((self._rep_start or 0) / self.fps, 2),
            "bottom_s":    round((self._bottom_frame or frame_idx) / self.fps, 2),
            "time_s":      round(frame_idx / self.fps, 2),
            "live_faults": [
                {"fault": k, "time_s": round(f / self.fps, 2)}
                for k, f in sorted(self._live_faults.items(), key=lambda kv: kv[1])
            ],
            "view":         view,
            "score":        score,
            "valid":        valid,
            "measured":     {k: round(v, 2) for k, v in measured.items()},
            "points_lost":  {k: round(v, 1) for k, v in lost.items()},
            "main_fault":   worst,
            "cue":          SQUAT_CUES[worst] if valid else f"No rep. {SQUAT_CUES[worst or 'depth']}",
        }

    @property
    def good_reps(self) -> int:
        return sum(r["valid"] for r in self.reps)

    @property
    def average(self) -> Optional[int]:
        return int(round(float(np.mean([r["score"] for r in self.reps])))) if self.reps else None


# ===========================================================================
# Push-up form scorer
# ===========================================================================
_PUSHUP_RULES = {
    "depth":   (35, 90.0,  115.0),
    "body":    (30, 0.05,  0.10),
    "tempo":   (15, 0.5,   0.2),
    "lockout": (10, 160.0, 140.0),
    "head":    (10, 30.0,  50.0),
}
_PUSHUP_LOWER_IS_WORSE  = {"tempo", "lockout"}
_PUSHUP_NO_REP_SCORE    = 70
_PUSHUP_NO_REP_BODY     = 0.09
_PUSHUP_BODY_PERCENTILE = 75
_PUSHUP_WORKING_ANGLE   = 140
_PUSHUP_LIVE_BODY_LIMIT = 0.07
_PUSHUP_LIVE_HEAD_LIMIT = 40.0
_PUSHUP_HOLD_FRAMES     = 5
_PUSHUP_SHALLOW_ANGLE   = 140

PLANK_KNEE = 140   # knees must be straighter than this to count as a plank
PLANK_TILT = 40    # body must be within this many degrees of horizontal

EARS = {"left": 7, "right": 8}

PUSHUP_CUES = {
    "depth":   "Chest closer to the floor",
    "sag":     "Hips up — squeeze your glutes",
    "pike":    "Hips down — straight line",
    "tempo":   "Slow down on the way down",
    "lockout": "Lock out your arms at the top",
    "head":    "Keep your head in line",
    None:      "Great rep",
}


def _body_offset(pts: np.ndarray, side: str) -> float:
    """
    Fraction of body length the hips sit away from the shoulder–ankle line.
    Positive = sagging below the line; negative = piking up.
    """
    sh  = joint(pts, side, "shoulder")[:2]
    hip = joint(pts, side, "hip")[:2]
    an  = joint(pts, side, "ankle")[:2]
    length = float(np.linalg.norm(an - sh)) + 1e-6
    if abs(an[0] - sh[0]) < 1e-3:
        return 0.0
    line_y = sh[1] + (an[1] - sh[1]) * (hip[0] - sh[0]) / (an[0] - sh[0])
    return float((hip[1] - line_y) / length)


def _head_drop(pts: np.ndarray, side: str) -> float:
    """Angle between torso direction and shoulder→ear direction (degrees)."""
    sh  = joint(pts, side, "shoulder")[:2]
    hip = joint(pts, side, "hip")[:2]
    ear = pts[EARS[side], :2]
    u = sh - hip
    v = ear - sh
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def in_plank(pts: np.ndarray, side: str) -> bool:
    """True when the person is in a horizontal plank position."""
    k = angle(joint(pts, side, "hip"), joint(pts, side, "knee"), joint(pts, side, "ankle"))
    d = joint(pts, side, "ankle")[:2] - joint(pts, side, "shoulder")[:2]
    tilt = float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-6)))
    return k > PLANK_KNEE and tilt < PLANK_TILT


class PushupCoach:
    """State machine that counts push-up reps and scores each one."""

    def __init__(self, fps: float) -> None:
        self.fps   = max(float(fps), 1.0)
        self.stage = "UP"
        self.reps: list[dict] = []
        self._top_angle = 0.0
        self._reset_rep()

    def _reset_rep(self) -> None:
        self._rep_start:    Optional[int]  = None
        self._min_elbow:    float          = 180.0
        self._bottom_frame: Optional[int]  = None
        self._offsets:      list[float]    = []
        self._max_head:     float          = 0.0
        self._live_faults:  dict[str, int] = {}
        self._streak:       dict[str, int] = {"sag": 0, "pike": 0, "head": 0}

    def abandon(self) -> None:
        """Person left the plank mid-rep — discard without counting."""
        self.stage      = "UP"
        self._top_angle = 0.0
        self._reset_rep()

    def update(
        self,
        elbow: float,
        offset: float,
        head: float,
        frame_idx: int,
    ) -> Optional[dict]:
        if self.stage == "UP" and elbow >= PUSHUP_UP_ANGLE:
            half = None
            if self._rep_start is not None and self._min_elbow < _PUSHUP_SHALLOW_ANGLE:
                half = self._score(frame_idx)
                self.reps.append(half)
                self._top_angle = 0.0
            self._reset_rep()
            self._top_angle = max(self._top_angle, elbow)
            return half

        if self._rep_start is None:
            self._rep_start = frame_idx
        if elbow < self._min_elbow:
            self._min_elbow, self._bottom_frame = elbow, frame_idx

        if elbow < _PUSHUP_WORKING_ANGLE:
            self._offsets.append(offset)
            self._max_head = max(self._max_head, head)
            for fault, on in (
                ("sag",  offset  >  _PUSHUP_LIVE_BODY_LIMIT),
                ("pike", -offset >  _PUSHUP_LIVE_BODY_LIMIT),
                ("head", head    >  _PUSHUP_LIVE_HEAD_LIMIT),
            ):
                self._streak[fault] = self._streak[fault] + 1 if on else 0
                if self._streak[fault] == _PUSHUP_HOLD_FRAMES:
                    self._live_faults.setdefault(fault, frame_idx - _PUSHUP_HOLD_FRAMES + 1)

        if self.stage == "UP" and elbow < PUSHUP_DOWN_ANGLE:
            self.stage = "DOWN"
        elif self.stage == "DOWN" and elbow >= PUSHUP_UP_ANGLE:
            self.stage = "UP"
            rep = self._score(frame_idx)
            self.reps.append(rep)
            self._reset_rep()
            self._top_angle = elbow
            return rep
        return None

    def _score(self, frame_idx: int) -> dict:
        descent = ((self._bottom_frame or frame_idx) - (self._rep_start or frame_idx)) / self.fps
        offs    = np.array(self._offsets or [0.0])
        sag     = max(0.0, float(np.percentile(offs,  _PUSHUP_BODY_PERCENTILE)))
        pike    = max(0.0, float(np.percentile(-offs, _PUSHUP_BODY_PERCENTILE)))
        hips    = max(sag, pike)
        measured = {
            "depth":   self._min_elbow,
            "body":    hips,
            "tempo":   descent,
            "lockout": self._top_angle or PUSHUP_UP_ANGLE,
            "head":    self._max_head,
        }
        lost    = {k: _penalty(measured[k], _PUSHUP_RULES[k], k in _PUSHUP_LOWER_IS_WORSE) for k in _PUSHUP_RULES}
        score   = max(0, min(100, int(round(100 - sum(lost.values())))))

        body_fault = "sag" if sag >= pike else "pike"
        worst_k    = max(lost, key=lost.get)  # type: ignore[arg-type]
        worst      = worst_k if lost[worst_k] >= 3 else None
        if worst == "body":
            worst = body_fault

        broken: list[str] = []
        if hips > _PUSHUP_NO_REP_BODY:
            broken.append(body_fault)
        if self._min_elbow >= PUSHUP_DOWN_ANGLE:
            broken.append("depth")
        valid = not broken and score >= _PUSHUP_NO_REP_SCORE
        if broken:
            worst = broken[0]

        return {
            "rep":         len(self.reps) + 1,
            "start_s":     round((self._rep_start or 0) / self.fps, 2),
            "bottom_s":    round((self._bottom_frame or frame_idx) / self.fps, 2),
            "time_s":      round(frame_idx / self.fps, 2),
            "live_faults": [
                {"fault": k, "time_s": round(f / self.fps, 2)}
                for k, f in sorted(self._live_faults.items(), key=lambda kv: kv[1])
            ],
            "view":        "side",
            "score":       score,
            "valid":       valid,
            "measured":    {k: round(v, 3) for k, v in measured.items()},
            "points_lost": {k: round(v, 1) for k, v in lost.items()},
            "main_fault":  worst,
            "cue":         PUSHUP_CUES[worst] if valid else f"No rep. {PUSHUP_CUES[worst or 'depth']}",
        }

    @property
    def good_reps(self) -> int:
        return sum(r["valid"] for r in self.reps)

    @property
    def average(self) -> Optional[int]:
        return int(round(float(np.mean([r["score"] for r in self.reps])))) if self.reps else None


# ===========================================================================
# Body validation helpers (used by the pipeline)
# ===========================================================================
_MIN_VIS    = 0.6
_MIN_HEIGHT = 0.25   # shoulder-to-ankle as fraction of frame height (squats)
_MIN_LENGTH = 0.25   # shoulder-to-ankle as fraction of max(w, h) (push-ups)


def is_real_person_squat(pts: np.ndarray, frame_h: int) -> bool:
    best = max(
        float(np.mean([pts[SIDES[s][n], 2] for n in ("shoulder", "hip", "knee")]))
        for s in SIDES
    )
    if best < _MIN_VIS:
        return False
    height = max(pts[27, 1], pts[28, 1]) - min(pts[11, 1], pts[12, 1])
    return bool(height >= _MIN_HEIGHT * frame_h)


def is_real_person_pushup(pts: np.ndarray, frame_w: int, frame_h: int) -> bool:
    best = max(
        float(np.mean([pts[SIDES[s][n], 2] for n in ("shoulder", "hip", "knee")]))
        for s in SIDES
    )
    if best < _MIN_VIS:
        return False
    side = "left" if pts[11, 2] + pts[23, 2] >= pts[12, 2] + pts[24, 2] else "right"
    length = float(np.linalg.norm(
        joint(pts, side, "ankle")[:2] - joint(pts, side, "shoulder")[:2]
    ))
    return bool(length >= _MIN_LENGTH * max(frame_w, frame_h))


def is_horizontal(pts: np.ndarray) -> bool:
    def _mid(name: str) -> np.ndarray:
        return (joint(pts, "left", name)[:2] + joint(pts, "right", name)[:2]) / 2
    d = _mid("ankle") - _mid("shoulder")
    return bool(float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-6))) < 40)
