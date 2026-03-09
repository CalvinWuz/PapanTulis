# =============================================================================
# hand_tracker.py — MediaPipe Hands wrapper (mediapipe 0.10.30+ Tasks API)
#
# Tracking pipeline per hand:
#   RAW landmark → Multi-frame buffer average → Kalman Filter predict+correct
#   → Adaptive EMA (alpha scales with velocity) → Dead-zone jitter gate
#
# Rendering:
#   draw_skeleton(frame, hand_results) — futuristic glow skeleton, per-finger
#   colours, joint dots, fingertip labels.
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import deque
import urllib.request
import os
import tempfile

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)
import numpy as np

import config

# ── Model download ─────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")

def _ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe hand model (~8 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return MODEL_PATH


# ── Tuning knobs ──────────────────────────────────────────────────────────────
GRACE_FRAMES      = 12
BUFFER_SIZE       = 3
KALMAN_Q          = 1e-3
KALMAN_R          = 5e-3
ALPHA_MIN         = 0.25
ALPHA_MAX         = 0.90
VELOCITY_SCALE    = 80.0
DEAD_ZONE         = 0.003


# ── Skeleton visual config ────────────────────────────────────────────────────
# BGR colours per finger
FINGER_COLORS: Dict[str, Tuple[int,int,int]] = {
    "thumb":  (255, 180,  50),   # gold
    "index":  ( 50, 255, 180),   # cyan-green
    "middle": (180,  50, 255),   # violet
    "ring":   ( 50, 150, 255),   # blue
    "pinky":  (255,  80, 150),   # pink
    "palm":   (200, 200, 200),   # white-grey for palm connections
}
BONE_THICKNESS   = 2     # base line thickness
GLOW_LAYERS      = 3     # number of progressively thicker blurred layers
JOINT_RADIUS     = 4
TIP_RADIUS       = 7
LABEL_FONT_SCALE = 0.38
LABEL_THICKNESS  = 1

# MediaPipe landmark indices grouped by finger
_FINGER_BONES: Dict[str, List[Tuple[int,int]]] = {
    "thumb":  [(0,1),(1,2),(2,3),(3,4)],
    "index":  [(0,5),(5,6),(6,7),(7,8)],
    "middle": [(0,9),(9,10),(10,11),(11,12)],
    "ring":   [(0,13),(13,14),(14,15),(15,16)],
    "pinky":  [(0,17),(17,18),(18,19),(19,20)],
    "palm":   [(5,9),(9,13),(13,17)],
}
# Fingertip indices and their labels
_TIP_LABELS: Dict[int, str] = {
    4:  "Thumb",
    8:  "Index",
    12: "Middle",
    16: "Ring",
    20: "Pinky",
}


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class Landmark:
    x: float; y: float; z: float
    px: int = 0; py: int = 0


@dataclass
class HandResult:
    handedness: str
    score: float
    landmarks: List[Landmark] = field(default_factory=list)

    index_tip:  Optional[Landmark] = None
    thumb_tip:  Optional[Landmark] = None
    middle_tip: Optional[Landmark] = None
    ring_tip:   Optional[Landmark] = None
    pinky_tip:  Optional[Landmark] = None
    index_pip:  Optional[Landmark] = None
    middle_pip: Optional[Landmark] = None
    ring_pip:   Optional[Landmark] = None
    pinky_pip:  Optional[Landmark] = None
    index_mcp:  Optional[Landmark] = None
    thumb_mcp:  Optional[Landmark] = None
    wrist:      Optional[Landmark] = None


_LANDMARK_MAP = {
    "wrist": 0, "thumb_mcp": 2, "thumb_tip": 4,
    "index_mcp": 5, "index_pip": 6, "index_tip": 8,
    "middle_pip": 10, "middle_tip": 12,
    "ring_pip": 14, "ring_tip": 16,
    "pinky_pip": 18, "pinky_tip": 20,
}


# ── Skeleton renderer ─────────────────────────────────────────────────────────

def _glow_line(frame: np.ndarray,
               p1: Tuple[int,int], p2: Tuple[int,int],
               color: Tuple[int,int,int],
               thickness: int = BONE_THICKNESS,
               layers: int = GLOW_LAYERS) -> None:
    """
    Draw a line with a multi-layer glow effect.
    Outer layers are thicker and dimmer, inner layer is bright.
    """
    for i in range(layers, 0, -1):
        alpha  = 0.15 + 0.25 * (layers - i)           # dimmer outer layers
        t      = thickness + (i - 1) * 3              # thicker outer layers
        c      = tuple(int(ch * alpha) for ch in color)
        cv2.line(frame, p1, p2, c, t, cv2.LINE_AA)
    # Bright core line
    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def _glow_circle(frame: np.ndarray,
                 center: Tuple[int,int],
                 radius: int,
                 color: Tuple[int,int,int],
                 layers: int = GLOW_LAYERS) -> None:
    for i in range(layers, 0, -1):
        alpha = 0.15 + 0.25 * (layers - i)
        r     = radius + (i - 1) * 3
        c     = tuple(int(ch * alpha) for ch in color)
        cv2.circle(frame, center, r, c, -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray,
                  hands: List[HandResult],
                  show_labels: bool = True) -> None:
    """
    Draw a futuristic glowing skeleton overlay for all detected hands.

    Call this after drawing_manager.render() so the skeleton appears
    on top of the whiteboard strokes.
    """
    for hand in hands:
        lms = hand.landmarks
        if len(lms) < 21:
            continue

        pts = [(lm.px, lm.py) for lm in lms]

        # ── Draw bones ────────────────────────────────────────────────
        for finger_name, bones in _FINGER_BONES.items():
            color = FINGER_COLORS[finger_name]
            for (a, b) in bones:
                _glow_line(frame, pts[a], pts[b], color)

        # ── Draw joints (all 21 landmarks) ───────────────────────────
        for i, pt in enumerate(pts):
            # Determine which finger this landmark belongs to
            if i == 0:
                color = FINGER_COLORS["palm"]
            elif i in (1, 2, 3, 4):
                color = FINGER_COLORS["thumb"]
            elif i in (5, 6, 7, 8):
                color = FINGER_COLORS["index"]
            elif i in (9, 10, 11, 12):
                color = FINGER_COLORS["middle"]
            elif i in (13, 14, 15, 16):
                color = FINGER_COLORS["ring"]
            else:
                color = FINGER_COLORS["pinky"]

            r = TIP_RADIUS if i in _TIP_LABELS else JOINT_RADIUS
            _glow_circle(frame, pt, r, color)

            # White dot in center for clarity
            cv2.circle(frame, pt, max(1, r // 3), (255, 255, 255), -1, cv2.LINE_AA)

        # ── Fingertip labels ─────────────────────────────────────────
        if show_labels:
            for tip_idx, label in _TIP_LABELS.items():
                pt  = pts[tip_idx]
                col = FINGER_COLORS[label.lower()] if label.lower() in FINGER_COLORS \
                      else (255, 255, 255)

                # Offset label above the fingertip
                tx = pt[0] - 28
                ty = pt[1] - 14

                # Dark background pill for readability
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX,
                    LABEL_FONT_SCALE, LABEL_THICKNESS
                )
                cv2.rectangle(frame,
                              (tx - 2, ty - th - 2),
                              (tx + tw + 2, ty + 2),
                              (20, 20, 20), -1)
                cv2.putText(frame, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            LABEL_FONT_SCALE, col,
                            LABEL_THICKNESS, cv2.LINE_AA)

        # ── Handedness badge ─────────────────────────────────────────
        wrist_pt = pts[0]
        badge    = hand.handedness[0]   # "L" or "R"
        cv2.putText(frame, badge,
                    (wrist_pt[0] - 8, wrist_pt[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1, cv2.LINE_AA)


# ── Kalman Filter ─────────────────────────────────────────────────────────────

class _KalmanLandmark:
    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * KALMAN_Q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_R
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self._initialised = False

    def update(self, x: float, y: float) -> Tuple[float, float]:
        meas = np.array([[x], [y]], dtype=np.float32)
        if not self._initialised:
            self.kf.statePre  = np.array([[x],[y],[0],[0]], dtype=np.float32)
            self.kf.statePost = np.array([[x],[y],[0],[0]], dtype=np.float32)
            self._initialised = True
        self.kf.predict()
        corrected = self.kf.correct(meas)
        return float(corrected[0][0]), float(corrected[1][0])

    def reset(self) -> None:
        self._initialised = False


# ── Per-hand smoother ─────────────────────────────────────────────────────────

class _HandSmoother:
    def __init__(self, n_landmarks: int = 21) -> None:
        self.n = n_landmarks
        self._buf: deque = deque(maxlen=BUFFER_SIZE)
        self._kalmans: List[_KalmanLandmark] = [_KalmanLandmark() for _ in range(n_landmarks)]
        self._ema:      Optional[np.ndarray] = None
        self._prev_ema: Optional[np.ndarray] = None

    def _buffered_average(self, raw: np.ndarray) -> np.ndarray:
        self._buf.append(raw)
        return np.mean(self._buf, axis=0)

    def _kalman_pass(self, pts: np.ndarray) -> np.ndarray:
        out = pts.copy()
        for i, kf in enumerate(self._kalmans):
            kx, ky = kf.update(pts[i, 0], pts[i, 1])
            out[i, 0] = kx
            out[i, 1] = ky
        return out

    def _adaptive_ema(self, kalman_out: np.ndarray) -> np.ndarray:
        if self._ema is None:
            self._ema = self._prev_ema = kalman_out.copy()
            return self._ema
        vel   = np.linalg.norm(kalman_out[:, :2] - self._prev_ema[:, :2], axis=1)
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * np.tanh(vel * VELOCITY_SCALE)
        alpha = alpha[:, np.newaxis]
        candidate = alpha * kalman_out + (1 - alpha) * self._ema
        delta  = np.linalg.norm(candidate[:, :2] - self._ema[:, :2], axis=1)
        freeze = (delta < DEAD_ZONE)[:, np.newaxis]
        self._prev_ema = self._ema.copy()
        self._ema = np.where(freeze, self._ema, candidate)
        return self._ema

    def update(self, raw_landmarks: list) -> np.ndarray:
        raw      = np.array([[lm.x, lm.y, lm.z] for lm in raw_landmarks], dtype=np.float32)
        buffered = self._buffered_average(raw)
        kalman   = self._kalman_pass(buffered)
        return self._adaptive_ema(kalman)

    def reset(self) -> None:
        self._buf.clear()
        for kf in self._kalmans:
            kf.reset()
        self._ema = self._prev_ema = None


# ── HandTracker ───────────────────────────────────────────────────────────────

class HandTracker:
    def __init__(self) -> None:
        model_path = _ensure_model()
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=config.MP_MAX_HANDS,
            min_hand_detection_confidence=config.MP_DETECTION_CONF,
            min_hand_presence_confidence=config.MP_TRACKING_CONF,
            min_tracking_confidence=config.MP_TRACKING_CONF,
        )
        self._landmarker     = HandLandmarker.create_from_options(options)
        self._smoothers:     Dict[str, _HandSmoother] = {}
        self._last_results:  List[HandResult]         = []
        self._grace_counter: int                      = 0

    def process(self, bgr_frame: np.ndarray) -> List[HandResult]:
        h, w     = bgr_frame.shape[:2]
        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self._landmarker.detect(mp_image)

        if not detection.hand_landmarks:
            if self._grace_counter < GRACE_FRAMES and self._last_results:
                self._grace_counter += 1
                return self._last_results
            self._smoothers.clear()
            self._last_results  = []
            self._grace_counter = 0
            return []

        self._grace_counter = 0
        results: List[HandResult] = []

        for i, hand_lms in enumerate(detection.hand_landmarks):
            label = "Right"
            score = 1.0
            if detection.handedness and i < len(detection.handedness):
                cat   = detection.handedness[i][0]
                label = cat.category_name
                score = cat.score

            if score < config.MP_DETECTION_CONF * 0.75:
                continue

            if label not in self._smoothers:
                self._smoothers[label] = _HandSmoother()

            smoothed = self._smoothers[label].update(hand_lms)

            hand = HandResult(handedness=label, score=score)
            for j in range(21):
                sx, sy, sz = smoothed[j]
                hand.landmarks.append(Landmark(
                    x=float(sx), y=float(sy), z=float(sz),
                    px=int(sx * w), py=int(sy * h),
                ))

            for attr, idx in _LANDMARK_MAP.items():
                setattr(hand, attr, hand.landmarks[idx])

            results.append(hand)

        active = {r.handedness for r in results}
        for lbl in list(self._smoothers.keys()):
            if lbl not in active:
                self._smoothers[lbl].reset()
                del self._smoothers[lbl]

        self._last_results = results
        return results

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()