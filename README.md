# ✋ Air Whiteboard

Draw in the air with your finger — writing appears live on your camera feed.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.10 + |
| opencv-python | 4.8 + |
| mediapipe | 0.10 + |
| numpy | 1.24 + |

---

## Installation

```bash
# 1. Clone or copy the project folder
cd air_whiteboard

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

> **macOS camera permission** — the first run will trigger a camera permission
> dialog.  Allow access and the window will open automatically.

---

## Gesture Reference

| Gesture | Action |
|---|---|
| ☝️ Index finger up | **Draw** mode |
| 🖐 Open palm (4+ fingers) | **Erase** mode |
| ✊ Closed fist | **Pause** (lift pen) |
| 👍 Thumbs up | Cycle **pen colour** |
| ✌️ Index + middle up | **Laser pointer** (no drawing) |
| 🖐🖐 Both palms open | **Clear** entire board |
| 🖐 Left palm + ☝️ Right index | **Draw** (two-hand mode) |

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `1` | Thin pen (2 px) |
| `2` | Medium pen (5 px) |
| `3` | Thick pen (8 px) |
| `Z` | Undo last stroke |
| `C` | Clear board |
| `Q` / `Esc` | Quit |

---

## Pen Colours

Cycle through 6 colours with the **Thumbs Up** gesture:

White → Black → Green → Cyan → Red → Yellow → (repeat)

---

## Shape Detection

When you finish drawing a stroke, the app automatically checks whether it
resembles a **circle**, **rectangle**, or **triangle**.  If it does, the
freehand stroke is silently replaced with a clean geometric shape fitted to
the same bounding box.

To disable shape detection, open `config.py` and set:

```python
SHAPE_MIN_POINTS = 999999
```

---

## Project Structure

```
air_whiteboard/
├── main.py            # Application loop and state machine
├── hand_tracker.py    # MediaPipe Hands wrapper → HandResult objects
├── gesture_detector.py# Finger-position heuristics → Gesture enum
├── drawing_utils.py   # Stroke storage, EMA smoothing, eraser, renderer
├── shape_detector.py  # Contour approximation → shape replacement
├── config.py          # All tunable parameters in one place
└── requirements.txt
```

---

## Tuning Tips

| Parameter | Location | Effect |
|---|---|---|
| `SMOOTHING_ALPHA` | config.py | Lower = smoother but laggier cursor |
| `ERASE_RADIUS` | config.py | Larger = bigger eraser circle |
| `PROCESS_WIDTH/HEIGHT` | config.py | Lower = faster but less accurate tracking |
| `SKIP_FRAMES` | config.py | 1 = process every other frame (saves CPU) |
| `FINGER_UP_THRESHOLD` | config.py | Raise if gestures trigger too easily |
| `SHAPE_APPROX_EPSILON` | config.py | Higher = more aggressive shape snapping |

---

## Troubleshooting

**Camera not opening**  
Change `CAMERA_INDEX` in `config.py` (try `1` or `2` for external cameras).

**Low FPS**  
- Reduce `PROCESS_WIDTH` / `PROCESS_HEIGHT` in `config.py`
- Set `SKIP_FRAMES = 1`
- Set `MP_MODEL_COMPLEXITY = 0` (already default)

**Gestures mis-detected**  
Adjust `FINGER_UP_THRESHOLD` in `config.py`.  Good lighting and a plain
background improve accuracy significantly.