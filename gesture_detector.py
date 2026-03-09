# =============================================================================
# gesture_detector.py — Robust gesture classification with debounce + scoring
#
# Problems solved vs previous version:
#   • DRAW triggers without index finger   → stricter index isolation check
#   • ERASE fires when trying to DRAW      → ERASE requires ALL 4 fingers clearly up
#   • PAUSE unstable (flickers on/off)     → debounce: gesture must hold N frames
#   • COLOR_TOGGLE hard to trigger         → relaxed thumb check + hold requirement
#   • Random DRAW triggers                 → confidence score gate + debounce
#
# Core improvements
# ─────────────────
# 1. Confidence scoring
#    Each gesture gets a float score 0.0–1.0 based on how clearly the
#    landmarks match the expected pose.  Gestures below SCORE_THRESHOLD
#    are rejected.
#
# 2. Debounce
#    A gesture must be detected consistently for DEBOUNCE_FRAMES consecutive
#    frames before it is "committed" and reported to the caller.
#    Exception: PAUSE commits immediately (so pen-up is never delayed).
#
# 3. Hysteresis on DRAW vs ERASE
#    Once in DRAW mode the system requires a higher ERASE score to switch,
#    preventing accidental erase when slightly spreading fingers while drawing.
#
# 4. Stricter finger isolation
#    DRAW: index up AND middle/ring/pinky clearly DOWN (below MCP level)
#    ERASE: all 4 fingers clearly up with strong margin
# =============================================================================

from __future__ import annotations
from enum import Enum, auto
from typing import List, Optional, Deque
from collections import deque

from hand_tracker import HandResult
import config

# feature: gesture erase detection
# ── Tuning ────────────────────────────────────────────────────────────────────
DEBOUNCE_FRAMES   = 4     # frames a gesture must hold before committing
                           # PAUSE always commits instantly
SCORE_THRESHOLD   = 0.55  # minimum confidence score to consider a gesture
DRAW_ERASE_HYST   = 0.15  # extra score needed to switch from DRAW → ERASE

# How far above PIP a tip must be (normalised) to count as "clearly up"
FINGER_UP_MARGIN  = 0.03
# How far BELOW PIP a tip must be to count as "clearly down" (for isolation)
FINGER_DOWN_MARGIN = 0.01


# ── Gesture enum ─────────────────────────────────────────────────────────────

class Gesture(Enum):
    DRAW         = auto()
    ERASE        = auto()
    PAUSE        = auto()
    COLOR_TOGGLE = auto()
    LASER        = auto()
    CLEAR        = auto()
    DRAW_TWO     = auto()
    UNKNOWN      = auto()


# ── Low-level scored helpers ──────────────────────────────────────────────────

def _tip_pip_margin(tip_y: float, pip_y: float) -> float:
    """
    Returns how far the tip is above (+) or below (-) the PIP joint.
    Positive = finger is up.
    """
    return pip_y - tip_y   # positive when tip is above pip in image coords


def _finger_up_score(tip_y: float, pip_y: float) -> float:
    """
    Continuous score 0–1 for how extended a finger is.
    0 = clearly down, 1 = clearly up, values in between = ambiguous.
    """
    margin = _tip_pip_margin(tip_y, pip_y)
    # Map [−0.05 .. +0.08] → [0 .. 1]
    score = (margin + 0.05) / 0.13
    return float(max(0.0, min(1.0, score)))


def _finger_down_score(tip_y: float, pip_y: float) -> float:
    """Inverse of _finger_up_score — how clearly the finger is curled."""
    return 1.0 - _finger_up_score(tip_y, pip_y)


def _finger_clearly_up(tip_y: float, pip_y: float) -> bool:
    return _tip_pip_margin(tip_y, pip_y) > FINGER_UP_MARGIN


def _finger_clearly_down(tip_y: float, pip_y: float) -> bool:
    return _tip_pip_margin(tip_y, pip_y) < -FINGER_DOWN_MARGIN


# ── Per-finger accessors ──────────────────────────────────────────────────────

def _idx_margin(h: HandResult) -> float:
    if h.index_tip is None or h.index_pip is None: return -1.0
    return _tip_pip_margin(h.index_tip.y, h.index_pip.y)

def _mid_margin(h: HandResult) -> float:
    if h.middle_tip is None or h.middle_pip is None: return -1.0
    return _tip_pip_margin(h.middle_tip.y, h.middle_pip.y)

def _rng_margin(h: HandResult) -> float:
    if h.ring_tip is None or h.ring_pip is None: return -1.0
    return _tip_pip_margin(h.ring_tip.y, h.ring_pip.y)

def _pnk_margin(h: HandResult) -> float:
    if h.pinky_tip is None or h.pinky_pip is None: return -1.0
    return _tip_pip_margin(h.pinky_tip.y, h.pinky_pip.y)


# ── Gesture scorers ───────────────────────────────────────────────────────────

def _score_draw(hand: HandResult) -> float:
    """
    DRAW: index clearly up, middle/ring/pinky clearly down.
    Score = index_up_score * avg(other three down scores).
    Extra penalty if middle finger is ambiguously up (common false trigger).
    """
    if hand.index_tip is None or hand.index_pip is None:
        return 0.0

    idx_score = _finger_up_score(hand.index_tip.y, hand.index_pip.y)

    # Middle must be clearly down — strongest isolation requirement
    mid_down = _finger_down_score(hand.middle_tip.y, hand.middle_pip.y) \
               if hand.middle_tip and hand.middle_pip else 0.5
    rng_down = _finger_down_score(hand.ring_tip.y, hand.ring_pip.y) \
               if hand.ring_tip and hand.ring_pip else 0.5
    pnk_down = _finger_down_score(hand.pinky_tip.y, hand.pinky_pip.y) \
               if hand.pinky_tip and hand.pinky_pip else 0.5

    isolation = (mid_down * 0.5 + rng_down * 0.25 + pnk_down * 0.25)
    return float(idx_score * isolation)


def _score_erase(hand: HandResult) -> float:
    """
    ERASE: all 4 fingers clearly up.
    Score = geometric mean of all four up-scores.
    """
    scores = []
    for tip, pip_ in [
        (hand.index_tip,  hand.index_pip),
        (hand.middle_tip, hand.middle_pip),
        (hand.ring_tip,   hand.ring_pip),
        (hand.pinky_tip,  hand.pinky_pip),
    ]:
        if tip is None or pip_ is None:
            scores.append(0.0)
        else:
            scores.append(_finger_up_score(tip.y, pip_.y))
    # Geometric mean penalises any one finger being ambiguous
    product = 1.0
    for s in scores:
        product *= max(s, 1e-6)
    return float(product ** 0.25)


def _score_pause(hand: HandResult) -> float:
    """
    PAUSE / fist: all 4 fingers down.
    """
    scores = []
    for tip, pip_ in [
        (hand.index_tip,  hand.index_pip),
        (hand.middle_tip, hand.middle_pip),
        (hand.ring_tip,   hand.ring_pip),
        (hand.pinky_tip,  hand.pinky_pip),
    ]:
        if tip is None or pip_ is None:
            scores.append(0.5)
        else:
            scores.append(_finger_down_score(tip.y, pip_.y))
    return float(sum(scores) / len(scores))


def _score_laser(hand: HandResult) -> float:
    """
    LASER: index + middle up, ring + pinky down.
    """
    if any(x is None for x in [
        hand.index_tip, hand.index_pip,
        hand.middle_tip, hand.middle_pip,
        hand.ring_tip, hand.ring_pip,
        hand.pinky_tip, hand.pinky_pip,
    ]):
        return 0.0

    idx_up  = _finger_up_score(hand.index_tip.y,  hand.index_pip.y)
    mid_up  = _finger_up_score(hand.middle_tip.y, hand.middle_pip.y)
    rng_dn  = _finger_down_score(hand.ring_tip.y,  hand.ring_pip.y)
    pnk_dn  = _finger_down_score(hand.pinky_tip.y, hand.pinky_pip.y)

    return float((idx_up + mid_up + rng_dn + pnk_dn) / 4.0)


def _score_color_toggle(hand: HandResult) -> float:
    """
    COLOR_TOGGLE: thumb tip well above index MCP, all 4 fingers down.
    Relaxed: only needs thumb clearly raised + 3 of 4 fingers down.
    """
    if hand.thumb_tip is None or hand.index_mcp is None:
        return 0.0

    # Thumb height above index knuckle (normalised)
    thumb_raise = hand.index_mcp.y - hand.thumb_tip.y   # positive = thumb up
    thumb_score = min(1.0, max(0.0, thumb_raise / 0.08))

    # Lateral extension of thumb
    if hand.thumb_mcp is not None:
        if hand.handedness == "Right":
            lateral = hand.thumb_mcp.x - hand.thumb_tip.x
        else:
            lateral = hand.thumb_tip.x - hand.thumb_mcp.x
        lateral_score = min(1.0, max(0.0, lateral / 0.04))
    else:
        lateral_score = 0.5

    # Fingers down (at least 3 of 4)
    down_scores = []
    for tip, pip_ in [
        (hand.index_tip,  hand.index_pip),
        (hand.middle_tip, hand.middle_pip),
        (hand.ring_tip,   hand.ring_pip),
        (hand.pinky_tip,  hand.pinky_pip),
    ]:
        if tip and pip_:
            down_scores.append(_finger_down_score(tip.y, pip_.y))
    fingers_down = sorted(down_scores, reverse=True)[:3]  # best 3
    finger_score = sum(fingers_down) / 3.0 if fingers_down else 0.0

    return float(thumb_score * 0.4 + lateral_score * 0.2 + finger_score * 0.4)


# ── Classify single hand (returns gesture + score) ───────────────────────────

def _classify_scored(hand: HandResult) -> tuple[Gesture, float]:
    """
    Score all gestures and return the best match above threshold.
    Priority order breaks ties when scores are close.
    """
    candidates = [
        (Gesture.LASER,        _score_laser(hand),        0),
        (Gesture.DRAW,         _score_draw(hand),         1),
        (Gesture.COLOR_TOGGLE, _score_color_toggle(hand), 2),
        (Gesture.ERASE,        _score_erase(hand),        3),
        (Gesture.PAUSE,        _score_pause(hand),        4),
    ]

    # Filter by threshold
    valid = [(g, s, p) for g, s, p in candidates if s >= SCORE_THRESHOLD]
    if not valid:
        return Gesture.UNKNOWN, 0.0

    # Pick highest score; use priority to break ties
    best = max(valid, key=lambda x: (x[1], -x[2]))
    return best[0], best[1]


# ── Debounce state machine ────────────────────────────────────────────────────

class _Debouncer:
    """
    Requires a gesture to be stable for DEBOUNCE_FRAMES before committing.
    PAUSE is always committed instantly (we never want delayed pen-up).
    """

    def __init__(self) -> None:
        self._candidate:   Gesture = Gesture.UNKNOWN
        self._hold_count:  int     = 0
        self._committed:   Gesture = Gesture.PAUSE
        self._current_mode: Gesture = Gesture.PAUSE  # for hysteresis

    def update(self, raw: Gesture, raw_score: float) -> Gesture:
        # PAUSE always wins immediately — critical for pen-up safety
        if raw == Gesture.PAUSE:
            self._candidate  = Gesture.PAUSE
            self._hold_count = DEBOUNCE_FRAMES
            self._committed  = Gesture.PAUSE
            self._current_mode = Gesture.PAUSE
            return Gesture.PAUSE

        # Hysteresis: switching from DRAW → ERASE needs extra score margin
        if (self._current_mode == Gesture.DRAW and
                raw == Gesture.ERASE and
                raw_score < SCORE_THRESHOLD + DRAW_ERASE_HYST):
            return self._committed

        if raw == self._candidate:
            self._hold_count += 1
        else:
            self._candidate  = raw
            self._hold_count = 1

        if self._hold_count >= DEBOUNCE_FRAMES:
            self._committed    = self._candidate
            self._current_mode = self._candidate

        return self._committed


# ── Two-hand classification ───────────────────────────────────────────────────

def _classify_two_hands(hand_a: HandResult, hand_b: HandResult) -> Optional[tuple[Gesture, HandResult]]:
    """Returns (Gesture, draw_hand) or None."""
    score_a_erase = _score_erase(hand_a)
    score_b_erase = _score_erase(hand_b)

    # Both palms open → CLEAR
    if score_a_erase >= SCORE_THRESHOLD and score_b_erase >= SCORE_THRESHOLD:
        return Gesture.CLEAR, None

    # Left open + right draws
    left  = hand_a if hand_a.handedness == "Left"  else hand_b
    right = hand_a if hand_a.handedness == "Right" else hand_b
    if left is not right:
        left_erase  = _score_erase(left)
        right_draw  = _score_draw(right)
        if left_erase >= SCORE_THRESHOLD and right_draw >= SCORE_THRESHOLD:
            return Gesture.DRAW_TWO, right

    return None


# ── Public API ────────────────────────────────────────────────────────────────

class GestureDetector:
    """
    Stateful gesture classifier with debounce and confidence scoring.

    Usage:
        detector = GestureDetector()
        result   = detector.detect(hands)   # List[HandResult] from HandTracker
        # result = {"gesture": Gesture, "draw_hand": HandResult|None, "num_hands": int}
    """

    def __init__(self) -> None:
        # One debouncer per hand slot (keyed by handedness)
        self._debouncers: dict[str, _Debouncer] = {
            "Left":  _Debouncer(),
            "Right": _Debouncer(),
            "single": _Debouncer(),
        }

    def detect(self, hands: List[HandResult]) -> dict:
        num = len(hands)

        # ── No hands ──────────────────────────────────────────────────
        if num == 0:
            # Immediately commit PAUSE on no detection
            for db in self._debouncers.values():
                db.update(Gesture.PAUSE, 1.0)
            return {"gesture": Gesture.PAUSE, "draw_hand": None, "num_hands": 0}

        # ── Two hands ─────────────────────────────────────────────────
        if num >= 2:
            two = _classify_two_hands(hands[0], hands[1])
            if two is not None:
                gesture, draw_hand = two
                return {"gesture": gesture, "draw_hand": draw_hand, "num_hands": 2}
            # Fall through to single-hand on primary hand
            primary = hands[0]
        else:
            primary = hands[0]

        # ── Single hand ───────────────────────────────────────────────
        raw_gesture, raw_score = _classify_scored(primary)

        # Pick debouncer by handedness
        label = primary.handedness if primary.handedness in self._debouncers else "single"
        committed = self._debouncers[label].update(raw_gesture, raw_score)

        return {
            "gesture":   committed,
            "draw_hand": primary,
            "num_hands": num,
            # Debug info — can be printed during development:
            # "raw_gesture": raw_gesture,
            # "raw_score":   raw_score,
        }
    