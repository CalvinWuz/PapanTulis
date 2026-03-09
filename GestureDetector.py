# =============================================================================
# gesture_detector.py — Gesture classification from HandResult objects
#
# Each gesture is detected by analysing the relative positions of MediaPipe
# landmarks.  The logic is intentionally expressed as small, named functions
# so it's easy to understand, test and extend.
#
# Gesture vocabulary
# ──────────────────
# Single hand:
#   DRAW        — only index finger extended upward
#   ERASE       — open palm (4+ fingers up)
#   PAUSE       — closed fist  (0 fingers up)
#   COLOR_TOGGLE— thumbs up (thumb well above palm, other fingers curled)
#   LASER       — index + middle up, ring + pinky down
#
# Two hands:
#   CLEAR       — both palms open
#   DRAW_LEFT   — left palm open, right index up  → draw with right hand
#
# Algorithm: "finger is up" test
# ────────────────────────────────
# A finger is considered UP when its tip y-coordinate is *above* (smaller y
# in image space) its PIP (proximal-interphalangeal) joint by more than a
# small threshold.  The threshold is expressed in normalised MediaPipe coords
# (range 0-1) so it scales correctly regardless of hand distance from camera.
#
# For the thumb we compare its tip x-position relative to its MCP joint
# because the thumb moves laterally when extended.  We also flip the
# left/right comparison for the left hand.
# =============================================================================

from __future__ import annotations
from enum import Enum, auto
from typing import List, Optional

from hand_tracker import HandResult
import config


# ── Gesture enum ─────────────────────────────────────────────────────────────

class Gesture(Enum):
    DRAW         = auto()   # 1 hand: only index up
    ERASE        = auto()   # 1 hand: open palm
    PAUSE        = auto()   # 1 hand: fist
    COLOR_TOGGLE = auto()   # 1 hand: thumbs-up
    LASER        = auto()   # 1 hand: index + middle up
    CLEAR        = auto()   # 2 hands: both open palms
    DRAW_TWO     = auto()   # 2 hands: left open, right draws
    UNKNOWN      = auto()


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _finger_up(tip_y: float, pip_y: float) -> bool:
    """
    True when the finger tip is above the PIP joint.
    In image coordinates y increases downward, so 'above' means smaller y.
    """
    return (pip_y - tip_y) > config.FINGER_UP_THRESHOLD


def _thumb_up_extended(hand: HandResult) -> bool:
    """
    True when the thumb tip is clearly to the outside of its MCP joint.
    'Outside' is left for a Right hand and right for a Left hand.
    We also require the thumb to be higher than the index MCP (the classic
    thumbs-up position).
    """
    if hand.thumb_tip is None or hand.thumb_mcp is None or hand.index_mcp is None:
        return False

    # Thumb must be higher (smaller y) than the index knuckle
    if hand.thumb_tip.y >= hand.index_mcp.y:
        return False

    # Lateral check: right hand thumb extends to the left (lower x)
    if hand.handedness == "Right":
        return hand.thumb_tip.x < hand.thumb_mcp.x
    else:
        return hand.thumb_tip.x > hand.thumb_mcp.x


def _count_fingers_up(hand: HandResult) -> int:
    """
    Return the number of non-thumb fingers that are extended (0-4).
    """
    pairs = [
        (hand.index_tip,  hand.index_pip),
        (hand.middle_tip, hand.middle_pip),
        (hand.ring_tip,   hand.ring_pip),
        (hand.pinky_tip,  hand.pinky_pip),
    ]
    return sum(
        1 for tip, pip_ in pairs
        if tip is not None and pip_ is not None and _finger_up(tip.y, pip_.y)
    )


def _index_up(hand: HandResult) -> bool:
    return (
        hand.index_tip is not None and hand.index_pip is not None
        and _finger_up(hand.index_tip.y, hand.index_pip.y)
    )


def _middle_up(hand: HandResult) -> bool:
    return (
        hand.middle_tip is not None and hand.middle_pip is not None
        and _finger_up(hand.middle_tip.y, hand.middle_pip.y)
    )


def _ring_up(hand: HandResult) -> bool:
    return (
        hand.ring_tip is not None and hand.ring_pip is not None
        and _finger_up(hand.ring_tip.y, hand.ring_pip.y)
    )


def _pinky_up(hand: HandResult) -> bool:
    return (
        hand.pinky_tip is not None and hand.pinky_pip is not None
        and _finger_up(hand.pinky_tip.y, hand.pinky_pip.y)
    )


# ── Single-hand gesture classifier ───────────────────────────────────────────

def classify_single_hand(hand: HandResult) -> Gesture:
    """
    Classify the gesture for one hand.

    Priority order (most specific first):
      1. Laser       (index + middle up, ring + pinky down)
      2. Draw        (only index up)
      3. Color toggle(thumbs-up)
      4. Erase       (open palm, 4 fingers up)
      5. Pause       (fist, 0 fingers up)
      6. Unknown
    """
    fingers = _count_fingers_up(hand)
    idx_up  = _index_up(hand)
    mid_up  = _middle_up(hand)
    rng_up  = _ring_up(hand)
    pnk_up  = _pinky_up(hand)

    # ── Laser pointer: index + middle up, ring + pinky down ──────────────
    if idx_up and mid_up and not rng_up and not pnk_up:
        return Gesture.LASER

    # ── Draw: only index up ───────────────────────────────────────────────
    if idx_up and not mid_up and not rng_up and not pnk_up:
        return Gesture.DRAW

    # ── Thumbs up (color toggle) ──────────────────────────────────────────
    if _thumb_up_extended(hand) and fingers == 0:
        return Gesture.COLOR_TOGGLE

    # ── Open palm: 4 fingers extended ────────────────────────────────────
    if fingers >= config.OPEN_PALM_MIN_FINGERS:
        return Gesture.ERASE

    # ── Fist: no fingers extended ─────────────────────────────────────────
    if fingers <= config.FIST_MAX_FINGERS:
        return Gesture.PAUSE

    return Gesture.UNKNOWN


# ── Two-hand gesture classifier ───────────────────────────────────────────────

def classify_two_hands(
    hand_a: HandResult, hand_b: HandResult
) -> Optional[Gesture]:
    """
    Classify gestures that require two hands.
    Returns None if no two-hand gesture is detected.

    Two-hand gestures:
      CLEAR    — both palms open
      DRAW_TWO — left palm open + right hand index up
    """
    fingers_a = _count_fingers_up(hand_a)
    fingers_b = _count_fingers_up(hand_b)

    # Determine which is left and which is right
    # MediaPipe reports handedness from the user's perspective (mirrored view)
    left  = hand_a if hand_a.handedness == "Left"  else hand_b
    right = hand_a if hand_a.handedness == "Right" else hand_b

    left_fingers  = _count_fingers_up(left)
    right_fingers = _count_fingers_up(right)

    # ── Both palms open → clear board ────────────────────────────────────
    if (fingers_a >= config.CLEAR_GESTURE_MIN_FINGERS and
            fingers_b >= config.CLEAR_GESTURE_MIN_FINGERS):
        return Gesture.CLEAR

    # ── Left open + right drawing ────────────────────────────────────────
    if (left is not right and
            left_fingers >= config.OPEN_PALM_MIN_FINGERS and
            _index_up(right) and not _middle_up(right)):
        return Gesture.DRAW_TWO

    return None


# ── Public API ────────────────────────────────────────────────────────────────

class GestureDetector:
    """
    Stateless gesture classifier.
    Call detect() with the list of HandResult objects from HandTracker.
    """

    def detect(self, hands: List[HandResult]) -> dict:
        """
        Analyse detected hands and return a result dict:

        {
            "gesture":    Gesture,           # primary gesture
            "draw_hand":  HandResult | None, # hand to use for drawing/erasing
            "num_hands":  int,
        }
        """
        num = len(hands)

        if num == 0:
            return {"gesture": Gesture.PAUSE, "draw_hand": None, "num_hands": 0}

        if num == 1:
            gesture = classify_single_hand(hands[0])
            return {
                "gesture":   gesture,
                "draw_hand": hands[0],
                "num_hands": 1,
            }

        # Two hands
        two_hand_gesture = classify_two_hands(hands[0], hands[1])
        if two_hand_gesture == Gesture.CLEAR:
            return {"gesture": Gesture.CLEAR, "draw_hand": None, "num_hands": 2}

        if two_hand_gesture == Gesture.DRAW_TWO:
            # The drawing hand is the Right hand
            right = next((h for h in hands if h.handedness == "Right"), hands[0])
            return {"gesture": Gesture.DRAW, "draw_hand": right, "num_hands": 2}

        # Fall back to primary hand (index 0) single-hand classification
        gesture = classify_single_hand(hands[0])
        return {
            "gesture":   gesture,
            "draw_hand": hands[0],
            "num_hands": 2,
        }