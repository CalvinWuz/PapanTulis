# =============================================================================
# drawing_utils.py — Stroke storage, smoothing, erasing and rendering
#
# Design
# ──────
# A *stroke* is a list of (x, y) pixel tuples that represent a single pen-down
# to pen-up sequence.  Strokes are stored as a list-of-lists so each can be
# managed independently (undo, erase, shape-replace).
#
# Smoothing
# ─────────
# We use Exponential Moving Average (EMA):
#
#   smoothed_x = α · raw_x + (1 − α) · prev_smoothed_x
#
# α is read from config.SMOOTHING_ALPHA.  Lower α = smoother but laggier.
# The smoother keeps its own state between frames; it resets on pen-up.
#
# Eraser
# ──────
# Erasing works by removing individual *segments* (pairs of adjacent points)
# from all strokes if either endpoint falls within the erase circle.
# Strokes that become too short are deleted entirely.
#
# Rendering
# ─────────
# All strokes are blended onto the camera frame each tick.  We redraw from
# scratch every frame (no persistent canvas) so the overlay is always
# pixel-perfect.  For thick strokes this is fast because OpenCV's polylines
# is implemented in C++.
# =============================================================================

from __future__ import annotations
from typing import List, Optional, Tuple
import math

import cv2
import numpy as np

import config


# Type aliases
Point  = Tuple[int, int]
Color  = Tuple[int, int, int]


# ── EMA Smoother ─────────────────────────────────────────────────────────────

class EMAPoint:
    """
    Exponential Moving Average smoother for (x, y) coordinates.
    Call reset() when a new stroke begins so there's no jump from the
    last position.
    """

    def __init__(self, alpha: float = config.SMOOTHING_ALPHA) -> None:
        self.alpha = alpha
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None

    def update(self, x: int, y: int) -> Point:
        if self._sx is None:
            self._sx, self._sy = float(x), float(y)
        else:
            a = self.alpha
            self._sx = a * x + (1 - a) * self._sx
            self._sy = a * y + (1 - a) * self._sy
        return (int(self._sx), int(self._sy))

    def reset(self) -> None:
        self._sx = None
        self._sy = None

    @property
    def current(self) -> Optional[Point]:
        if self._sx is None:
            return None
        return (int(self._sx), int(self._sy))


# ── Stroke ────────────────────────────────────────────────────────────────────

class Stroke:
    """One continuous pen-down segment with its own colour and thickness."""

    def __init__(self, color: Color, thickness: int) -> None:
        self.color:     Color      = color
        self.thickness: int        = thickness
        self.points:    List[Point] = []

    def add_point(self, p: Point) -> None:
        self.points.append(p)

    @property
    def is_empty(self) -> bool:
        return len(self.points) < 2

    def draw(self, frame: np.ndarray) -> None:
        """Render this stroke onto frame in-place."""
        if len(self.points) < 2:
            return
        pts = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            frame, [pts],
            isClosed=False,
            color=self.color,
            thickness=self.thickness,
            lineType=cv2.LINE_AA,
        )


# ── DrawingManager ────────────────────────────────────────────────────────────

class DrawingManager:
    """
    Manages the complete drawing state for a session.

    Public interface:
        begin_stroke(color, thickness)  — start a new stroke (pen down)
        add_point(x, y)                 — extend current stroke (pen move)
        end_stroke()                    — finalise stroke (pen up)
        erase(cx, cy, radius)           — erase points near (cx, cy)
        undo()                          — remove last finished stroke
        clear()                         — remove all strokes
        replace_last_stroke(stroke)     — used by shape detector
        render(frame)                   — draw all strokes onto frame
    """

    def __init__(self) -> None:
        self._strokes:     List[Stroke]     = []  # completed strokes
        self._current:     Optional[Stroke] = None
        self._undo_stack:  List[Stroke]     = []  # same as _strokes, separate ref
        self.smoother      = EMAPoint()

    # ── Stroke lifecycle ─────────────────────────────────────────────────

    def begin_stroke(self, color: Color, thickness: int) -> None:
        """Start a brand-new stroke.  Resets the EMA smoother."""
        self.smoother.reset()
        self._current = Stroke(color, thickness)

    def add_point(self, raw_x: int, raw_y: int) -> None:
        """
        Smooth the raw fingertip position and append to the active stroke.
        No-op if no stroke is active.
        """
        if self._current is None:
            return
        sx, sy = self.smoother.update(raw_x, raw_y)
        self._current.add_point((sx, sy))

    def end_stroke(self) -> Optional[Stroke]:
        """
        Finalise the current stroke and push it onto the stack.
        Returns the stroke (may be used by shape detector).
        """
        if self._current is None or self._current.is_empty:
            self._current = None
            return None

        stroke = self._current
        self._strokes.append(stroke)
        # Keep undo stack bounded
        if len(self._strokes) > config.MAX_UNDO_STEPS:
            self._strokes.pop(0)
        self._current = None
        return stroke

    # ── Mutation ─────────────────────────────────────────────────────────

    def erase(self, cx: int, cy: int, radius: int = config.ERASE_RADIUS) -> None:
        """
        Remove any stroke points within `radius` pixels of (cx, cy).
        Strokes that become too short are removed entirely.
        After erasure a stroke may split; we keep both halves if long enough.
        """
        r2 = radius * radius
        survivors: List[Stroke] = []

        for stroke in self._strokes:
            # Split stroke at erased points
            segment: List[Point] = []
            for pt in stroke.points:
                dx = pt[0] - cx
                dy = pt[1] - cy
                if dx * dx + dy * dy <= r2:
                    # This point is in the erase zone
                    if len(segment) >= 2:
                        s = Stroke(stroke.color, stroke.thickness)
                        s.points = segment
                        survivors.append(s)
                    segment = []
                else:
                    segment.append(pt)
            if len(segment) >= 2:
                s = Stroke(stroke.color, stroke.thickness)
                s.points = segment
                survivors.append(s)

        self._strokes = survivors

    def undo(self) -> None:
        """Remove the last completed stroke."""
        if self._strokes:
            self._strokes.pop()

    def clear(self) -> None:
        """Remove all strokes and reset the active stroke."""
        self._strokes.clear()
        self._current = None
        self.smoother.reset()

    def replace_last_stroke(self, new_stroke: Stroke) -> None:
        """
        Replace the most recently completed stroke with a clean shape.
        Used by the shape detector after shape recognition.
        """
        if self._strokes:
            self._strokes[-1] = new_stroke

    # ── Rendering ────────────────────────────────────────────────────────

    def render(self, frame: np.ndarray) -> None:
        """
        Draw all strokes (plus the in-progress stroke) onto `frame` in-place.
        This is called every tick so the overlay is always up-to-date.
        """
        for stroke in self._strokes:
            stroke.draw(frame)
        if self._current is not None:
            self._current.draw(frame)

    # ── Helpers ──────────────────────────────────────────────────────────

    @property
    def current_stroke(self) -> Optional[Stroke]:
        return self._current

    @property
    def stroke_count(self) -> int:
        return len(self._strokes)

    def get_last_stroke(self) -> Optional[Stroke]:
        return self._strokes[-1] if self._strokes else None