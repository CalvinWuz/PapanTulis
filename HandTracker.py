# =============================================================================
# hand_tracker.py — MediaPipe Hands wrapper
#
# Responsibilities:
#   • Initialise MediaPipe Hands once and keep it alive for the session.
#   • Accept a BGR frame, convert to RGB, run inference.
#   • Return a lightweight HandResult list (one entry per detected hand).
#   • Provide convenience accessors: landmark arrays, handedness label.
#
# Why a wrapper?
#   Isolates MediaPipe from the rest of the app so swapping backends is easy.
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class Landmark:
    """Normalised (0-1) landmark position plus pixel coords for a given frame."""
    x: float          # normalised x
    y: float          # normalised y
    z: float          # normalised depth (rough)
    px: int = 0       # pixel x (filled in after scaling)
    py: int = 0       # pixel y (filled in after scaling)


@dataclass
class HandResult:
    """All data for one detected hand in a single frame."""
    handedness: str                        # "Left" or "Right"
    score: float                           # detection confidence
    landmarks: List[Landmark] = field(default_factory=list)

    # Quick landmark accessors (populated by HandTracker.process)
    index_tip:  Optional[Landmark] = None  # landmark 8
    thumb_tip:  Optional[Landmark] = None  # landmark 4
    middle_tip: Optional[Landmark] = None  # landmark 12
    ring_tip:   Optional[Landmark] = None  # landmark 16
    pinky_tip:  Optional[Landmark] = None  # landmark 20

    index_pip:  Optional[Landmark] = None  # landmark 6  (proximal joint)
    middle_pip: Optional[Landmark] = None  # landmark 10
    ring_pip:   Optional[Landmark] = None  # landmark 14
    pinky_pip:  Optional[Landmark] = None  # landmark 18

    index_mcp:  Optional[Landmark] = None  # landmark 5  (knuckle)
    thumb_mcp:  Optional[Landmark] = None  # landmark 2
    wrist:      Optional[Landmark] = None  # landmark 0


# ── HandTracker ───────────────────────────────────────────────────────────────

class HandTracker:
    """
    Thin wrapper around mediapipe.solutions.hands.Hands.

    Usage:
        tracker = HandTracker()
        while True:
            frame = cap.read()[1]
            hands = tracker.process(frame)   # List[HandResult], len 0-2
    """

    # MediaPipe landmark indices we care about
    _LANDMARK_MAP = {
        "wrist":      0,
        "thumb_mcp":  2,
        "thumb_tip":  4,
        "index_mcp":  5,
        "index_pip":  6,
        "index_tip":  8,
        "middle_pip": 10,
        "middle_tip": 12,
        "ring_pip":   14,
        "ring_tip":   16,
        "pinky_pip":  18,
        "pinky_tip":  20,
    }

    def __init__(self) -> None:
        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MP_MAX_HANDS,
            model_complexity=config.MP_MODEL_COMPLEXITY,
            min_detection_confidence=config.MP_DETECTION_CONF,
            min_tracking_confidence=config.MP_TRACKING_CONF,
        )

    # ------------------------------------------------------------------
    def process(self, bgr_frame: np.ndarray) -> List[HandResult]:
        """
        Run MediaPipe inference on a BGR frame.

        Returns a list of HandResult objects (0, 1 or 2 items).
        Pixel coordinates are computed from the *full* frame dimensions
        so callers can draw directly on the original frame.
        """
        h, w = bgr_frame.shape[:2]

        # MediaPipe expects RGB; writeable=False avoids an internal copy
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_result = self._hands.process(rgb)

        if not mp_result.multi_hand_landmarks:
            return []

        results: List[HandResult] = []

        for mp_lm, mp_hand in zip(
            mp_result.multi_hand_landmarks,
            mp_result.multi_handedness,
        ):
            classification = mp_hand.classification[0]
            hand = HandResult(
                handedness=classification.label,
                score=classification.score,
            )

            # Build full landmark list
            for lm in mp_lm.landmark:
                l = Landmark(
                    x=lm.x, y=lm.y, z=lm.z,
                    px=int(lm.x * w), py=int(lm.y * h),
                )
                hand.landmarks.append(l)

            # Attach named shortcuts
            for attr, idx in self._LANDMARK_MAP.items():
                setattr(hand, attr, hand.landmarks[idx])

            results.append(hand)

        return results

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()