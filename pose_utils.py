"""
pose_utils.py — Shared helpers for BelievFit AI Gym Coach.

Covers: model loading & caching, landmark smoothing, near-side selection,
joint-angle geometry, and CV drawing primitives used by every pipeline step.
"""
from __future__ import annotations

import os
import threading
import time
import urllib.request
from typing import Optional

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# MediaPipe pose model
# ---------------------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = os.environ.get(
    "BELIEVFIT_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "pose_landmarker_full.task"),
)
MODEL_LOCK = threading.Lock()

# One colour per tracked person (BGR): green, purple, orange, cyan
PERSON_COLOURS = [
    (80, 200, 120),
    (220, 90, 180),
    (40, 150, 255),
    (220, 200, 40),
]

# MediaPipe landmark indices for the two sides of the body
SIDES: dict[str, dict[str, int]] = {
    "left": {
        "shoulder": 11, "elbow": 13, "wrist": 15,
        "hip": 23, "knee": 25, "ankle": 27, "heel": 29, "foot": 31,
    },
    "right": {
        "shoulder": 12, "elbow": 14, "wrist": 16,
        "hip": 24, "knee": 26, "ankle": 28, "heel": 30, "foot": 32,
    },
}

# Bone pairs drawn for the near (camera-facing) side
SIDE_BONES = [
    ("shoulder", "elbow"), ("elbow", "wrist"),
    ("shoulder", "hip"),
    ("hip", "knee"), ("knee", "ankle"),
    ("ankle", "heel"), ("heel", "foot"), ("ankle", "foot"),
]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def ensure_model() -> str:
    """
    Ensure the MediaPipe pose model exists and is valid (>1MB).
    Downloads with retry logic if missing or corrupt.
    Returns the absolute path to the model file.
    """
    with MODEL_LOCK:
        models_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(models_dir, exist_ok=True)

        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
            return MODEL_PATH

        print("Downloading BelievFit pose model (~9 MB) …")
        max_retries = 3
        last_err = None

        for attempt in range(1, max_retries + 1):
            temp_path = f"{MODEL_PATH}.tmp_{attempt}"
            try:
                urllib.request.urlretrieve(MODEL_URL, temp_path)
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024 * 1024:
                    if os.path.exists(MODEL_PATH):
                        os.remove(MODEL_PATH)
                    os.rename(temp_path, MODEL_PATH)
                    print(f"Pose model verified and saved → {MODEL_PATH}")
                    return MODEL_PATH
                else:
                    raise IOError("Downloaded model file is incomplete or too small.")
            except Exception as e:
                last_err = e
                print(f"Download attempt {attempt}/{max_retries} failed: {e}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                time.sleep(1.5 * attempt)

        raise RuntimeError(f"Failed to download MediaPipe pose model after {max_retries} attempts: {last_err}")


def create_landmarker(max_people: int = 2) -> vision.PoseLandmarker:
    """Return a PoseLandmarker configured for VIDEO mode."""
    model_path = ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=max_people,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def landmarks_to_array(landmarks, w: int, h: int) -> np.ndarray:
    """Convert 33 MediaPipe landmarks → float32 array (33, 3): px_x, px_y, visibility."""
    return np.array(
        [[lm.x * w, lm.y * h, lm.visibility] for lm in landmarks],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
class Smoother:
    """
    Exponential moving average on (x, y) landmark positions.

    alpha near 1 → tracks the signal quickly but may be noisy.
    alpha near 0 → very smooth but adds lag.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.state: Optional[np.ndarray] = None

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = pts.copy()
        else:
            self.state[:, :2] = (
                self.alpha * pts[:, :2] + (1 - self.alpha) * self.state[:, :2]
            )
            self.state[:, 2] = pts[:, 2]  # keep raw visibility
        return self.state

    def reset(self) -> None:
        self.state = None


# ---------------------------------------------------------------------------
# Near-side selection
# ---------------------------------------------------------------------------
class SideChooser:
    """
    Choose which body half faces the camera in a side-view shot.

    The MediaPipe model guesses the hidden side's landmarks, which are
    unreliable. A hysteresis margin stops the choice flickering every frame.
    """

    KEY_POINTS = ("shoulder", "hip", "knee", "ankle")

    def __init__(self, margin: float = 0.1) -> None:
        self.margin = margin
        self.side: Optional[str] = None

    def __call__(self, pts: np.ndarray) -> str:
        score = {
            s: float(np.mean([pts[SIDES[s][k], 2] for k in self.KEY_POINTS]))
            for s in SIDES
        }
        best = max(score, key=score.get)  # type: ignore[arg-type]
        if self.side is None or score[best] > score[self.side] + self.margin:
            self.side = best
        return self.side  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at vertex b (degrees) for the triangle a–b–c.

    Works on 2-D or 3-D arrays; only the first two coordinates are used.
    """
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 180.0
    cos_val = np.dot(ba, bc) / (norm_ba * norm_bc + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def joint(pts: np.ndarray, side: str, name: str) -> np.ndarray:
    """Return the landmark row for a named joint on the given side."""
    return pts[SIDES[side][name]]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_side_skeleton(
    frame: np.ndarray,
    pts: np.ndarray,
    side: str,
    colour: tuple,
    min_visibility: float = 0.3,
) -> None:
    """Draw the near-side skeleton: bones then joint dots, white outline."""
    h, w = frame.shape[:2]
    t = max(2, w // 300)
    idx = SIDES[side]
    visible = lambda n: pts[idx[n], 2] >= min_visibility
    p = lambda n: (int(pts[idx[n], 0]), int(pts[idx[n], 1]))

    for a, b in SIDE_BONES:
        if visible(a) and visible(b):
            cv2.line(frame, p(a), p(b), (255, 255, 255), t + 3, cv2.LINE_AA)
            cv2.line(frame, p(a), p(b), colour, t + 1, cv2.LINE_AA)

    for n in idx:
        if visible(n):
            cv2.circle(frame, p(n), t + 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p(n), t + 2, colour, -1, cv2.LINE_AA)


def draw_label(
    frame: np.ndarray,
    text: str,
    centre: tuple,
    colour: tuple,
    scale: Optional[float] = None,
    pad: int = 10,
) -> None:
    """Filled coloured tag with white centred text."""
    if scale is None:
        scale = max(0.6, frame.shape[1] / 1600)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = int(centre[0] - tw / 2)
    y = int(centre[1] + th / 2)
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + pad),
        colour, -1,
    )
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
