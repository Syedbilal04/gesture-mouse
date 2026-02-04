import threading
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui

try:
    import mediapipe as mp
    # Use standard legacy solutions API available in mediapipe 0.10.11
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except ImportError:
    raise SystemExit("mediapipe not installed. Run: pip install mediapipe opencv-python numpy pyautogui")


# =============================
# PyAutoGUI runtime configuration
# =============================
pyautogui.FAILSAFE = True  # Move mouse to top-left to raise FailSafeException
pyautogui.PAUSE = 0        # No pause between actions
try:
    # Some environments expose MINIMUM_SLEEP; set to 0 if available
    pyautogui.MINIMUM_SLEEP = 0
except Exception:
    pass


# =============================
# MediaPipe setup
# =============================
# mp_face_mesh, mp_hands, mp_drawing, mp_drawing_styles already imported above


# =============================
# Constants and indices
# =============================
# Iris indices (right and left) for FaceMesh with refine_landmarks=True
RIGHT_IRIS_IDXS = [468, 469, 470, 471]
LEFT_IRIS_IDXS = [473, 474, 475, 476]

# Eye corner indices (as requested: 33, 133)
EYE_CORNER_RIGHT = 33
EYE_CORNER_LEFT = 133

# Eyelid indices for blink detection
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


# =============================
# Utility functions
# =============================
def _avg_points(landmarks, idxs) -> Tuple[float, float]:
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in idxs], dtype=np.float32)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def compute_blink_ratio(landmarks) -> float:
    """Compute eye open ratio based on vertical vs horizontal distance of left eye.

    ratio = vertical_distance / horizontal_distance
    Smaller ratio implies blink/closed eye.
    """
    left_top = (landmarks[LEFT_EYE_TOP].x, landmarks[LEFT_EYE_TOP].y)
    left_bot = (landmarks[LEFT_EYE_BOTTOM].x, landmarks[LEFT_EYE_BOTTOM].y)
    eye_left_corner = (landmarks[EYE_CORNER_LEFT].x, landmarks[EYE_CORNER_LEFT].y)
    eye_right_corner = (landmarks[EYE_CORNER_RIGHT].x, landmarks[EYE_CORNER_RIGHT].y)

    vertical = _dist(left_top, left_bot)
    horizontal = _dist(eye_left_corner, eye_right_corner)
    if horizontal <= 1e-6:
        return 1.0
    return vertical / horizontal


def project_ratio_along_axis(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Project point p onto axis a->b and return ratio along [a,b]."""
    a_v = np.array(a, dtype=np.float32)
    b_v = np.array(b, dtype=np.float32)
    p_v = np.array(p, dtype=np.float32)
    ab = b_v - a_v
    ap = p_v - a_v
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 <= 1e-9:
        return 0.5
    t = float(np.dot(ap, ab) / ab_len2)
    return t


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def apply_inner_margin(r: float, margin: float = 0.2) -> float:
    """Map [margin, 1-margin] to [0,1] and clamp."""
    return clamp01((r - margin) / max(1e-6, (1.0 - 2.0 * margin)))


class EMAFilter:
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self.initialized = False
        self.x = 0.0
        self.y = 0.0

    def update(self, x: float, y: float) -> Tuple[float, float]:
        if not self.initialized:
            self.x, self.y = x, y
            self.initialized = True
            return self.x, self.y
        self.x = self.alpha * x + (1.0 - self.alpha) * self.x
        self.y = self.alpha * y + (1.0 - self.alpha) * self.y
        return self.x, self.y


class MouseController(threading.Thread):
    """Worker thread that moves the mouse and handles click events."""

    def __init__(self, deadzone_px: int = 3, ema_alpha: float = 0.15):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._target: Optional[Tuple[int, int]] = None
        self._running = True
        self._ema = EMAFilter(alpha=ema_alpha)
        self._deadzone = deadzone_px
        self._click_queue: deque[str] = deque()

    def set_target(self, x: int, y: int):
        with self._lock:
            self._target = (x, y)

    def enqueue_click(self, button: str):
        with self._lock:
            self._click_queue.append(button)

    def stop(self):
        self._running = False

    def run(self):
        last_pos = pyautogui.position()
        while self._running:
            try:
                # Handle clicks first to ensure responsiveness
                with self._lock:
                    if self._click_queue:
                        btn = self._click_queue.popleft()
                        pyautogui.click(button='left' if btn == 'left' else 'right')

                    target = self._target

                if target is not None:
                    sx, sy = self._ema.update(target[0], target[1])
                    # Deadzone to avoid micro tremors
                    if abs(sx - last_pos.x) >= self._deadzone or abs(sy - last_pos.y) >= self._deadzone:
                        pyautogui.moveTo(int(sx), int(sy))
                        last_pos = pyautogui.position()

                # A tiny sleep prevents CPU spin without adding perceptible latency
                time.sleep(0.002)
            except pyautogui.FailSafeException:
                # User moved mouse to top-left; exit immediately
                self._running = False
            except Exception:
                # Ignore transient input errors to keep loop alive
                time.sleep(0.005)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    screen_w, screen_h = pyautogui.size()
    status_text = "Initializing"

    mouse_worker = MouseController(deadzone_px=1, ema_alpha=0.25)
    mouse_worker.start()

    # Head pose compensation baseline (nose bridge landmark 1)
    nose_baseline: Optional[Tuple[float, float]] = None
    compensation_strength = 0.8  # fraction of nose delta to subtract

    # Click debouncing flags
    pinch_index_active = False
    pinch_middle_active = False
    pinch_threshold = 0.08  # Increased threshold for easier clicking
    pinch_release = 0.1     # Hysteresis
    
    # Sensitivity margin (0.4 means the center 20% of eye movement maps to 100% screen)
    # Adjust this value to control sensitivity. Higher = more sensitive.
    eye_margin = 0.35
    blink_ratio_threshold = 0.03

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh, mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as hands:
        cv2.namedWindow("Eye & Hand Tracking", cv2.WINDOW_NORMAL)
        status_color = (0, 255, 0)

        while True:
            ok, frame = cap.read()
            if not ok:
                status_text = "Camera Read Failed"
                status_color = (0, 0, 255)
                break

            # Mirror view for more intuitive feedback
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_res = face_mesh.process(rgb)
            hand_res = hands.process(rgb)

            # ================= Face / Eye tracking =================
            face_ok = face_res.multi_face_landmarks is not None
            blink_paused = False
            cursor_updated = False

            if face_ok:
                lm = face_res.multi_face_landmarks[0].landmark
                # Compute blink ratio (left eye)
                ratio = compute_blink_ratio(lm)
                if ratio < blink_ratio_threshold:
                    blink_paused = True

                # Nose baseline update
                nose = (lm[1].x, lm[1].y)
                if nose_baseline is None:
                    nose_baseline = nose

                # Iris centers
                right_iris = _avg_points(lm, RIGHT_IRIS_IDXS)
                left_iris = _avg_points(lm, LEFT_IRIS_IDXS)
                # Average both for stability
                iris_center = ((right_iris[0] + left_iris[0]) * 0.5, (right_iris[1] + left_iris[1]) * 0.5)

                # Head pose compensation using nose delta
                nose_dx = (nose[0] - nose_baseline[0]) if nose_baseline else 0.0
                nose_dy = (nose[1] - nose_baseline[1]) if nose_baseline else 0.0
                compensated = (iris_center[0] - compensation_strength * nose_dx,
                               iris_center[1] - compensation_strength * nose_dy)

                # Horizontal ratio along the axis defined by corners 33->133
                corner_r = (lm[EYE_CORNER_RIGHT].x, lm[EYE_CORNER_RIGHT].y)
                corner_l = (lm[EYE_CORNER_LEFT].x, lm[EYE_CORNER_LEFT].y)
                r_horiz = project_ratio_along_axis(compensated, corner_r, corner_l)
                r_horiz = apply_inner_margin(r_horiz, margin=eye_margin)

                # Vertical ratio: use left eye top/bottom bounds and compensated y
                eye_top = (lm[LEFT_EYE_TOP].y)
                eye_bot = (lm[LEFT_EYE_BOTTOM].y)
                # Normalize so that top -> 0, bottom -> 1 (image y grows downward)
                denom = max(1e-6, (eye_bot - eye_top))
                r_vert = (compensated[1] - eye_top) / denom
                r_vert = apply_inner_margin(r_vert, margin=eye_margin)

                # Map to screen pixels
                tx = int(clamp01(r_horiz) * screen_w)
                ty = int(clamp01(r_vert) * screen_h)

                # Robustness: stop movement on loss or blink
                if not blink_paused:
                    mouse_worker.set_target(tx, ty)
                    cursor_updated = True

                # Draw iris point for feedback
                ix = int(compensated[0] * w)
                iy = int(compensated[1] * h)
                # cv2.circle(frame, (ix, iy), 5, (0, 255, 0), -1)

                if blink_paused:
                    status_text = "Blink"
                    status_color = (0, 255, 255)
                elif cursor_updated:
                    status_text = "Tracking"
                    status_color = (0, 255, 0)
            else:
                status_text = "Face Lost"
                status_color = (0, 0, 255)

            # ================= Hand gestures / clicks =================
            if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
                handedness = hand_res.multi_handedness[0].classification[0].label
                if handedness == 'Right':
                    h_lm = hand_res.multi_hand_landmarks[0].landmark
                    # Index-thumb pinch for left click
                    idx_tip = (h_lm[8].x, h_lm[8].y)
                    thumb_tip = (h_lm[4].x, h_lm[4].y)
                    dist_idx_thumb = _dist(idx_tip, thumb_tip)

                    # Middle-thumb pinch for right click
                    mid_tip = (h_lm[12].x, h_lm[12].y)
                    dist_mid_thumb = _dist(mid_tip, thumb_tip)

                    # Draw lines for feedback
                    cv2.line(frame,
                             (int(idx_tip[0] * w), int(idx_tip[1] * h)),
                             (int(thumb_tip[0] * w), int(thumb_tip[1] * h)),
                             (255, 0, 0), 2)

                    cv2.line(frame,
                             (int(mid_tip[0] * w), int(mid_tip[1] * h)),
                             (int(thumb_tip[0] * w), int(thumb_tip[1] * h)),
                             (0, 0, 255), 2)

                    # Debounced left click
                    if dist_idx_thumb < pinch_threshold and not pinch_index_active:
                        mouse_worker.enqueue_click('left')
                        pinch_index_active = True
                        status_text = "Left Click Detected"
                        status_color = (0, 200, 0)
                    elif dist_idx_thumb > pinch_release and pinch_index_active:
                        pinch_index_active = False

                    # Debounced right click
                    if dist_mid_thumb < pinch_threshold and not pinch_middle_active:
                        mouse_worker.enqueue_click('right')
                        pinch_middle_active = True
                        status_text = "Right Click Detected"
                        status_color = (0, 200, 0)
                    elif dist_mid_thumb > pinch_release and pinch_middle_active:
                        pinch_middle_active = False

            # ================= UI overlay =================
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
            cv2.imshow("Eye & Hand Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                break

        # Cleanup
        mouse_worker.stop()
        mouse_worker.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()