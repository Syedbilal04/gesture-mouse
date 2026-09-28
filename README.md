# Gaze & Gesture Mouse (HCI)

> Control your computer's mouse hands-free: **your eyes move the cursor** and **pinch gestures click**. It runs in real time on an ordinary webcam with MediaPipe Face Mesh + Hands.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-0097A7?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![PyAutoGUI](https://img.shields.io/badge/PyAutoGUI-mouse_control-555?style=flat-square)

Human-Computer Interaction (HCI) project demo.

## ✨ Features

- 👁️ **Iris-based cursor control:** iris centres (Face Mesh refined landmarks) are projected along the eye-corner axis and mapped to screen coordinates
- 🧭 **Head-movement compensation:** subtracts nose-bridge motion from the iris position, so small head movements don't throw the cursor off
- 🎯 **Adjustable sensitivity:** an inner-margin setting (`eye_margin`) maps a small central range of eye motion to the full screen
- 😌 **Blink pause:** the cursor freezes while you blink (left-eye openness ratio), so blinks don't cause jumps
- 🤏 **Gesture clicks (MediaPipe Hands):**
  - thumb + index pinch → **left click**
  - thumb + middle pinch → **right click**
  - debounced with hysteresis (press/release thresholds) to avoid double clicks
  - gestures are read only from a hand that MediaPipe classifies as `Right`
- 🧵 **Smooth motion:** a background mouse thread with EMA smoothing and a pixel dead-zone
- 🛑 **Safety:** PyAutoGUI fail-safe (flick the mouse to the top-left corner) and **Esc** to quit
- Live mirrored preview window with status text ("Blink", "Left Click Detected", ...)

## 🛠️ Tech Stack

Python · MediaPipe (legacy `solutions` API: Face Mesh with refined iris landmarks, Hands) · OpenCV · NumPy · PyAutoGUI · `threading`

## 🚀 Getting Started

```bash
# Python 3.9–3.12 (mediapipe 0.10.14 has no wheels for 3.13)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

- The code uses the legacy `mp.solutions` API, which newer MediaPipe releases no longer ship. `requirements.txt` pins `mediapipe==0.10.14`, which still includes it, plus NumPy 1.x and a compatible OpenCV.
- `test_mp.py` is a quick check that the `mediapipe.solutions` import works in your environment.
- The webcam opens at 1280×720 (camera index 0).
- On macOS, grant Camera and Accessibility permissions to your terminal so PyAutoGUI can move the mouse.

### Tuning (in `main.py`)

| Variable | Default | Effect |
| --- | --- | --- |
| `eye_margin` | `0.35` | Higher values mean more sensitive cursor movement |
| `compensation_strength` | `0.8` | Fraction of head (nose) motion that gets subtracted |
| `blink_ratio_threshold` | `0.03` | Eye-openness ratio below which the cursor pauses |
| `pinch_threshold` / `pinch_release` | `0.08` / `0.1` | Click press/release distances |
| `MouseController(deadzone_px, ema_alpha)` | `1`, `0.25` | Jitter dead-zone and smoothing factor |

## 📁 Project Structure

```
main.py           # tracking loop, cursor mapping, gesture clicks, MouseController thread
requirements.txt  # pinned dependencies
test_mp.py        # mediapipe import sanity check
```

## ⚠️ Limitations

- Accuracy depends on lighting, camera quality and head stability; there is no per-user calibration step yet
- Tracks one face and one hand at a time

## 📄 License

No licence file has been added yet, so all rights are reserved by default.
