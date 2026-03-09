# =============================================================================
# main.py — Air Whiteboard entry point
#
# Architecture
# ────────────
#  ┌─────────────┐     frames      ┌──────────────┐
#  │  Webcam      │ ─────────────► │  HandTracker  │
#  └─────────────┘                 └──────┬───────┘
#                                         │ HandResult list
#                                  ┌──────▼───────┐
#                                  │ GestureDetect│
#                                  └──────┬───────┘
#                                         │ Gesture + draw_hand
#                           ┌─────────────▼─────────────┐
#                           │       State Machine        │
#                           │  (draw / erase / pause …)  │
#                           └─────────────┬─────────────┘
#                                         │
#                          ┌──────────────▼──────────────┐
#                          │  DrawingManager.render(frame) │
#                          └──────────────┬──────────────┘
#                                         │
#                           ┌─────────────▼─────────────┐
#                           │         UI overlay         │
#                           └─────────────┬─────────────┘
#                                         │
#                                  cv2.imshow()
#
# State machine transitions
# ─────────────────────────
#   PAUSE  → DRAW       : gesture == DRAW and index_tip not None
#   PAUSE  → ERASE      : gesture == ERASE
#   PAUSE  → LASER      : gesture == LASER
#   DRAW   → PAUSE      : gesture != DRAW  (pen-up, commits stroke)
#   ERASE  → PAUSE      : gesture != ERASE
#   LASER  → PAUSE      : gesture != LASER
#   ANY    → CLEAR      : gesture == CLEAR (clears everything)
#   ANY    → toggle     : gesture == COLOR_TOGGLE (one-shot, no held state)
# =============================================================================

import sys
import time
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from hand_tracker    import HandTracker
from gesture_detector import GestureDetector, Gesture
from drawing_utils   import DrawingManager
from shape_detector  import ShapeDetector


# ── App state ─────────────────────────────────────────────────────────────────

class AppMode(Enum):
    DRAW  = auto()
    ERASE = auto()
    PAUSE = auto()
    LASER = auto()


# Colour names for the UI label
COLOR_NAMES = ["White", "Black", "Green", "Cyan", "Red", "Yellow"]
THICKNESS_NAMES = ["Thin (2px)", "Medium (5px)", "Thick (8px)"]


# ── UI helpers ────────────────────────────────────────────────────────────────

def _draw_ui(
    frame: np.ndarray,
    mode: AppMode,
    color_idx: int,
    thickness_idx: int,
    fps: float,
) -> None:
    """
    Render a semi-transparent status pill in the top-left corner.
    Lines shown:
      Mode | Color | Thickness | FPS
    Keyboard shortcuts shown as a compact reminder bar.
    """
    lines = [
        f"Mode  : {mode.name}",
        f"Color : {COLOR_NAMES[color_idx]}",
        f"Pen   : {THICKNESS_NAMES[thickness_idx]}",
        f"FPS   : {fps:.0f}",
        "─" * 22,
        "1/2/3 thickness  Z undo",
        "C clear  Q quit",
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    fscale     = config.UI_FONT_SCALE
    fthick     = config.UI_FONT_THICKNESS
    pad        = 8
    line_h     = 20
    box_w      = 220
    box_h      = len(lines) * line_h + pad * 2

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, config.UI_BG_ALPHA,
                    frame,   1 - config.UI_BG_ALPHA, 0, frame)

    # Colour swatch
    swatch_color = config.PEN_COLORS[color_idx]
    cv2.rectangle(frame,
                  (8 + box_w - 28, 8 + pad + line_h - 14),
                  (8 + box_w - 8,  8 + pad + line_h + 2),
                  swatch_color, -1)

    # Mode indicator strip
    mode_colors = {
        AppMode.DRAW:  (0, 200, 100),
        AppMode.ERASE: (0, 140, 255),
        AppMode.PAUSE: (60, 60, 200),
        AppMode.LASER: (0, 0, 220),
    }
    cv2.rectangle(frame, (8, 8), (14, 8 + box_h), mode_colors[mode], -1)

    for i, line in enumerate(lines):
        y = 8 + pad + (i + 1) * line_h
        cv2.putText(
            frame, line, (20, y),
            font, fscale, config.UI_TEXT_COLOR, fthick, cv2.LINE_AA
        )


def _draw_cursor(
    frame: np.ndarray,
    pt: Tuple[int, int],
    mode: AppMode,
    color: Tuple[int, int, int],
    erase_radius: int,
) -> None:
    """Draw a visual cursor at the fingertip position."""
    if mode == AppMode.DRAW:
        cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 7, (200, 200, 200), 1, cv2.LINE_AA)
    elif mode == AppMode.ERASE:
        cv2.circle(frame, pt, erase_radius, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, (0, 200, 255), -1, cv2.LINE_AA)
    elif mode == AppMode.LASER:
        cv2.circle(frame, pt, config.LASER_RADIUS, config.LASER_COLOR, -1, cv2.LINE_AA)
        # Glow ring
        cv2.circle(frame, pt, config.LASER_RADIUS + 4,
                   (100, 100, 255), 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Initialise subsystems ────────────────────────────────────────────
    tracker   = HandTracker()
    detector  = GestureDetector()
    drawing   = DrawingManager()
    shapes    = ShapeDetector()

    # ── Open webcam ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)          # request 60; camera will honour what it can
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # reduce latency by keeping buffer tiny

    if not cap.isOpened():
        sys.exit("ERROR: Cannot open camera index "
                 f"{config.CAMERA_INDEX}.  Check CAMERA_INDEX in config.py.")

    # ── App state ────────────────────────────────────────────────────────
    mode:          AppMode = AppMode.PAUSE
    color_idx:     int     = config.DEFAULT_COLOR_INDEX
    thickness_idx: int     = config.DEFAULT_THICKNESS

    # Gesture edge-detection helpers
    prev_gesture:      Gesture  = Gesture.UNKNOWN
    color_toggle_lock: bool     = False  # prevents repeated toggles while held

    # FPS tracking
    fps_timer    = time.perf_counter()
    fps_counter  = 0
    fps_display  = 0.0

    # Frame-skip counter
    skip_count   = 0
    last_hands   = []   # cached hand results when skipping

    print("Air Whiteboard started.  Press Q to quit.")

    # ── Main loop ────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Lost camera feed.")
            break

        # Mirror horizontally so movement feels natural (like a mirror)
        frame = cv2.flip(frame, 1)

        # ── Hand tracking (with optional frame skip) ──────────────────
        skip_count += 1
        if config.SKIP_FRAMES == 0 or skip_count > config.SKIP_FRAMES:
            # Scale down for faster inference
            small = cv2.resize(
                frame,
                (config.PROCESS_WIDTH, config.PROCESS_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            last_hands = tracker.process(small)

            # Scale landmark pixel coords back to full resolution
            sx = config.FRAME_WIDTH  / config.PROCESS_WIDTH
            sy = config.FRAME_HEIGHT / config.PROCESS_HEIGHT
            for h in last_hands:
                for lm in h.landmarks:
                    lm.px = int(lm.px * sx)
                    lm.py = int(lm.py * sy)
                # Refresh named shortcuts after scaling
                if h.index_tip:
                    h.index_tip.px  = h.landmarks[8].px
                    h.index_tip.py  = h.landmarks[8].py
                if h.middle_tip:
                    h.middle_tip.px = h.landmarks[12].px
                    h.middle_tip.py = h.landmarks[12].py

            skip_count = 0

        hands = last_hands

        # ── Gesture classification ────────────────────────────────────
        result   = detector.detect(hands)
        gesture  = result["gesture"]
        draw_h   = result["draw_hand"]

        # Finger tip pixel position (landmark 8)
        tip_pt: Optional[Tuple[int, int]] = None
        if draw_h is not None and draw_h.index_tip is not None:
            tip_pt = (draw_h.index_tip.px, draw_h.index_tip.py)

        # ── State machine transitions ─────────────────────────────────

        # One-shot: CLEAR (two open palms) resets canvas immediately
        if gesture == Gesture.CLEAR:
            drawing.clear()
            mode = AppMode.PAUSE

        # One-shot: COLOR_TOGGLE (thumbs up)
        elif gesture == Gesture.COLOR_TOGGLE:
            if not color_toggle_lock:
                color_idx = (color_idx + 1) % len(config.PEN_COLORS)
                color_toggle_lock = True
        else:
            color_toggle_lock = False

        # Transitions that depend on current mode
        if gesture == Gesture.DRAW and tip_pt is not None:
            if mode != AppMode.DRAW:
                # Pen down — start a new stroke
                mode = AppMode.DRAW
                drawing.begin_stroke(
                    config.PEN_COLORS[color_idx],
                    config.PEN_THICKNESSES[thickness_idx],
                )
            drawing.add_point(*tip_pt)

        elif gesture == Gesture.ERASE and tip_pt is not None:
            if mode == AppMode.DRAW:
                # Pen up before switching to erase
                stroke = drawing.end_stroke()
                if stroke:
                    replacement = shapes.detect(stroke)
                    if replacement:
                        drawing.replace_last_stroke(replacement)
            mode = AppMode.ERASE
            drawing.erase(*tip_pt, config.ERASE_RADIUS)

        elif gesture == Gesture.LASER:
            if mode == AppMode.DRAW:
                stroke = drawing.end_stroke()
                if stroke:
                    replacement = shapes.detect(stroke)
                    if replacement:
                        drawing.replace_last_stroke(replacement)
            mode = AppMode.LASER

        else:
            # Pen up (any gesture that isn't DRAW/ERASE/LASER)
            if mode == AppMode.DRAW:
                stroke = drawing.end_stroke()
                if stroke:
                    replacement = shapes.detect(stroke)
                    if replacement:
                        drawing.replace_last_stroke(replacement)
            if gesture not in (Gesture.CLEAR, Gesture.COLOR_TOGGLE):
                mode = AppMode.PAUSE

        prev_gesture = gesture

        # ── Render drawing onto frame ─────────────────────────────────
        drawing.render(frame)

        # ── Cursor visual ─────────────────────────────────────────────
        if tip_pt is not None and mode != AppMode.PAUSE:
            _draw_cursor(
                frame, tip_pt, mode,
                config.PEN_COLORS[color_idx],
                config.ERASE_RADIUS,
            )

        # ── Skeleton overlay ───────────────────────────────────────────
        from hand_tracker import draw_skeleton
        draw_skeleton(frame, hands, show_labels=True)

        # ── FPS counter ───────────────────────────────────────────────
        fps_counter += 1
        now = time.perf_counter()
        elapsed = now - fps_timer
        if elapsed >= 0.5:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = now

        # ── UI overlay ────────────────────────────────────────────────
        _draw_ui(frame, mode, color_idx, thickness_idx, fps_display)

        # ── Display ───────────────────────────────────────────────────
        cv2.imshow("Air Whiteboard", frame)

        # ── Keyboard shortcuts ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:          # Q or Esc → quit
            break
        elif key == ord('z'):                      # Z → undo
            if mode == AppMode.DRAW:
                drawing.end_stroke()               # discard in-progress stroke
                mode = AppMode.PAUSE
            drawing.undo()
        elif key == ord('c'):                      # C → clear
            drawing.clear()
            mode = AppMode.PAUSE
        elif key == ord('1'):                      # 1 → thin
            thickness_idx = 0
        elif key == ord('2'):                      # 2 → medium
            thickness_idx = 1
        elif key == ord('3'):                      # 3 → thick
            thickness_idx = 2

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("Air Whiteboard closed.")


if __name__ == "__main__":
    main()