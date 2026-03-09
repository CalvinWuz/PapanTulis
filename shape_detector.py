# =============================================================================
# shape_detector.py — Automatic shape recognition and replacement
#
# Algorithm overview
# ──────────────────
# 1. Collect the (x, y) points of a completed stroke.
# 2. Smooth the polyline with a Gaussian kernel (reduces jitter).
# 3. Compute the Ramer-Douglas-Peucker (RDP) approximation via
#    cv2.approxPolyDP, which reduces the curve to its essential vertices.
# 4. Check vertex count and geometry to classify as:
#      • Circle    — roughly round; aspect ratio ≈ 1; extent high
#      • Rectangle — 4 vertices, right-ish angles
#      • Triangle  — 3 vertices
# 5. If a shape is detected, build a *clean* replacement Stroke with the
#    ideal geometric shape (fitted to the bounding box of the original).
# 6. Return the replacement Stroke or None.
#
# Shape classification heuristics
# ────────────────────────────────
# Circle
#   • After RDP the polygon has many vertices (>= 8)  OR
#     the contour is nearly circular:
#       aspect_ratio = min(w,h)/max(w,h)  close to 1
#       extent = contour_area / bounding_rect_area  high
#
# Rectangle
#   • RDP yields 4 vertices
#   • Each interior angle is within ±20° of 90°
#
# Triangle
#   • RDP yields 3 vertices
# =============================================================================

from __future__ import annotations
from typing import List, Optional, Tuple
import math

import cv2
import numpy as np

import config
from drawing_utils import Stroke, Point, Color


def _smooth_points(pts: List[Point], sigma: int = 3) -> List[Point]:
    """
    Apply a 1-D Gaussian blur to x and y separately.
    sigma controls how much the curve is smoothed before shape fitting.
    """
    if len(pts) < 5:
        return pts
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    k  = max(3, sigma * 2 + 1)
    if k % 2 == 0:
        k += 1
    xs = cv2.GaussianBlur(xs.reshape(1, -1), (k, 1), sigma).flatten()
    ys = cv2.GaussianBlur(ys.reshape(1, -1), (k, 1), sigma).flatten()
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle in degrees between two 2-D vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos = np.clip(cos, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def _is_rectangle(approx: np.ndarray) -> bool:
    """
    True if the 4-vertex polygon has all angles within ±25° of 90°.
    """
    if len(approx) != 4:
        return False
    for i in range(4):
        p0 = approx[(i - 1) % 4][0].astype(float)
        p1 = approx[i][0].astype(float)
        p2 = approx[(i + 1) % 4][0].astype(float)
        v1 = p0 - p1
        v2 = p2 - p1
        angle = _angle_between(v1, v2)
        if abs(angle - 90.0) > 25.0:
            return False
    return True


def _build_circle_stroke(
    cx: int, cy: int, rx: int, ry: int,
    color: Color, thickness: int, n_pts: int = 72,
) -> Stroke:
    """Build a Stroke that approximates an ellipse."""
    stroke = Stroke(color, thickness)
    for i in range(n_pts + 1):
        angle = 2 * math.pi * i / n_pts
        x = cx + int(rx * math.cos(angle))
        y = cy + int(ry * math.sin(angle))
        stroke.points.append((x, y))
    return stroke


def _build_poly_stroke(
    vertices: np.ndarray, color: Color, thickness: int
) -> Stroke:
    """Build a Stroke that connects the given vertices (closed polygon)."""
    stroke = Stroke(color, thickness)
    pts = [tuple(v[0]) for v in vertices]
    pts.append(pts[0])  # close the shape
    stroke.points = pts
    return stroke


# ── Public API ────────────────────────────────────────────────────────────────

class ShapeDetector:
    """
    Stateless shape recogniser.

    Usage:
        detector = ShapeDetector()
        replacement = detector.detect(stroke)
        if replacement:
            drawing_manager.replace_last_stroke(replacement)
    """

    def detect(self, stroke: Stroke) -> Optional[Stroke]:
        """
        Analyse a completed stroke and attempt shape recognition.

        Returns a clean replacement Stroke if a shape is detected, else None.
        Requires at least config.SHAPE_MIN_POINTS points in the stroke.
        """
        if len(stroke.points) < config.SHAPE_MIN_POINTS:
            return None

        # Step 1: smooth the raw points
        pts = _smooth_points(stroke.points)

        # Step 2: convert to numpy contour format expected by OpenCV
        contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

        # Step 3: RDP approximation
        arc_len = cv2.arcLength(contour, closed=True)
        epsilon = config.SHAPE_APPROX_EPSILON * arc_len
        approx  = cv2.approxPolyDP(contour, epsilon, closed=True)

        n_vertices = len(approx)

        # Step 4: bounding box for shape fitting
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2

        # ── Circle ────────────────────────────────────────────────────────
        # Criterion: many vertices after RDP, near-square bounding box,
        # and the contour fills most of its bounding rect.
        if n_vertices >= 8:
            aspect  = min(w, h) / (max(w, h) + 1e-6)
            area    = cv2.contourArea(contour)
            extent  = area / (w * h + 1e-6)
            if (aspect > (1 - config.CIRCLE_ASPECT_RATIO_TOL) and
                    extent > config.CIRCLE_EXTENT_MIN):
                return _build_circle_stroke(
                    cx, cy, w // 2, h // 2,
                    stroke.color, stroke.thickness
                )

        # ── Rectangle ─────────────────────────────────────────────────────
        if _is_rectangle(approx):
            return _build_poly_stroke(approx, stroke.color, stroke.thickness)

        # ── Triangle ──────────────────────────────────────────────────────
        if n_vertices == 3:
            return _build_poly_stroke(approx, stroke.color, stroke.thickness)

        return None