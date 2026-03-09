# =============================================================================
# config.py — Central configuration for Air Whiteboard
# All tunable parameters live here so you never have to hunt through source.
# =============================================================================

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0          # 0 = built-in MacBook webcam
FRAME_WIDTH         = 1280       # Capture width  (pixels)
FRAME_HEIGHT        = 720        # Capture height (pixels)
PROCESS_WIDTH       = 640        # Internal processing width (half res = 2× speed)
PROCESS_HEIGHT      = 360        # Internal processing height

# ── MediaPipe Hands ───────────────────────────────────────────────────────────
MP_MAX_HANDS           = 2       # Track up to 2 hands
MP_DETECTION_CONF      = 0.7     # Minimum detection confidence
MP_TRACKING_CONF       = 0.6     # Minimum tracking confidence
MP_MODEL_COMPLEXITY    = 0       # 0 = lite (faster), 1 = full

# ── Smoothing ─────────────────────────────────────────────────────────────────
# Exponential Moving Average: smoothed = α·current + (1-α)·previous
# Lower α → smoother but laggier; higher α → more responsive but jittery
SMOOTHING_ALPHA     = 0.40

# ── Pen ───────────────────────────────────────────────────────────────────────
PEN_COLORS = [
    (255, 255, 255),   # White  (index 0)
    (0,   0,   0),     # Black  (index 1)
    (0,   255, 100),   # Green  (index 2)
    (0,   180, 255),   # Cyan   (index 3)
    (255, 80,  80),    # Red    (index 4)
    (255, 200, 0),     # Yellow (index 5)
]
DEFAULT_COLOR_INDEX = 0          # Start with white

PEN_THICKNESSES     = [2, 5, 8]  # Thin / medium / thick
DEFAULT_THICKNESS   = 1          # Index into PEN_THICKNESSES (5 px)

# ── Eraser ────────────────────────────────────────────────────────────────────
ERASE_RADIUS        = 40         # Pixels; circular erase zone around fingertip

# ── Laser pointer ─────────────────────────────────────────────────────────────
LASER_COLOR         = (0, 0, 255)   # BGR red dot
LASER_RADIUS        = 12

# ── Gesture detection thresholds ─────────────────────────────────────────────
# "Finger is UP" when its tip y-coord is above its pip y-coord by this fraction
# of the palm height.  Increase to make detection less sensitive.
FINGER_UP_THRESHOLD = 0.02      # Normalised units (MediaPipe uses 0-1 space)

# Open palm: all 4 fingers up
OPEN_PALM_MIN_FINGERS = 4

# Two-open-palms clear gesture: both hands need ≥ this many fingers up
CLEAR_GESTURE_MIN_FINGERS = 4

# Thumbs-up: thumb tip clearly above index knuckle, all other fingers down
THUMB_UP_FINGER_RATIO = 1.3     # thumb must be this × higher than index MCP

# Fist: all fingers DOWN (0 fingers up)
FIST_MAX_FINGERS = 0

# ── Shape detection ───────────────────────────────────────────────────────────
SHAPE_MIN_POINTS        = 20    # Ignore very short strokes
SHAPE_APPROX_EPSILON    = 0.04  # Ramer-Douglas-Peucker tolerance (×arc length)
CIRCLE_ASPECT_RATIO_TOL = 0.35  # How non-square a bounding rect can be for circle
CIRCLE_EXTENT_MIN       = 0.65  # Min ratio of contour area to bounding-rect area

# ── Undo stack ────────────────────────────────────────────────────────────────
MAX_UNDO_STEPS      = 50        # Keep at most this many strokes in history

# ── UI overlay ────────────────────────────────────────────────────────────────
UI_FONT_SCALE       = 0.55
UI_FONT_THICKNESS   = 1
UI_TEXT_COLOR       = (220, 220, 220)
UI_BG_ALPHA         = 0.45      # Translucency of the status pill background

# ── Performance ───────────────────────────────────────────────────────────────
# Number of frames to skip hand-tracking on (0 = every frame).
# Setting to 1 processes every other frame → ~2× CPU saving with slight lag.
SKIP_FRAMES         = 0