import os
import time
import warnings
from pathlib import Path
from collections import deque

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def euclidean(p1, p2):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return float(np.linalg.norm(p1 - p2))


def eye_aspect_ratio(pts6):
    p1, p2, p3, p4, p5, p6 = pts6
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_open_ratio(mouth_left, mouth_right, upper_lip_inner, lower_lip_inner, min_width_px=20.0):
    mouth_w = euclidean(mouth_left, mouth_right)
    if mouth_w < min_width_px:
        return 0.0
    mouth_h = euclidean(upper_lip_inner, lower_lip_inner)
    return mouth_h / mouth_w


class TimeWindowPerclos:
    def __init__(self, window_sec=30.0):
        self.window_sec = float(window_sec)
        self.q = deque()  # (t_end, dt, closed)
        self.total_t = 0.0
        self.closed_t = 0.0

    def add(self, t_end, dt, closed):
        if dt <= 0:
            return
        self.q.append((t_end, dt, bool(closed)))
        self.total_t += dt
        if closed:
            self.closed_t += dt
        self._trim(t_end)

    def _trim(self, now):
        while self.q and (now - self.q[0][0]) > self.window_sec:
            t_end, dt, closed = self.q.popleft()
            self.total_t -= dt
            if closed:
                self.closed_t -= dt
        self.total_t = max(0.0, self.total_t)
        self.closed_t = max(0.0, self.closed_t)

    def value(self):
        if self.total_t <= 1e-6:
            return 0.0
        return self.closed_t / self.total_t


def main():
    # ==========================
    # BASELINE CONFIG (Step 1)
    # ==========================
    CAMERA_INDEX = 0
    BACKEND = cv2.CAP_DSHOW

    CAP_W, CAP_H = 640, 480
    CAP_FPS = 30

    DET_W, DET_H = 320, 320
    DETECT_EVERY_N = 1

    SCORE_THRESH = 0.85
    NMS_THRESH = 0.30
    TOP_K = 5000

    # ==========================
    # FaceMesh CONFIG (Step 2)
    # ==========================
    USE_FACE_MESH = True
    FACE_MESH_MAX_FACES = 1
    FACE_MESH_REFINE = True
    FACE_MESH_DET_CONF = 0.5
    FACE_MESH_TRK_CONF = 0.5

    ROI_SCALE = 1.25
    ROI_MIN_W = 140
    ROI_MIN_H = 140

    DRAW_YUNET_LANDMARKS = True
    DRAW_FACE_MESH_DEBUG = False
    DRAW_METRICS = True

    # ==========================
    # DROWSINESS LOGIC (Step 3)
    # ==========================
    EMA_ALPHA = 0.25

    EAR_CLOSED_TH = 0.18
    EAR_OPEN_TH = 0.21

    MOR_YAWN_ON = 0.45
    MOR_YAWN_OFF = 0.35

    YAWN_MIN_DUR_SEC = 0.60
    YAWN_COOLDOWN_SEC = 2.0

    PERCLOS_WINDOW_SEC = 30.0
    PERCLOS_DROWSY_TH = 0.25

    # ==========================
    # PATHS
    # ==========================
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "face_detector" / "face_detection_yunet_2023mar.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"YuNet model not found at: {model_path}")

    if USE_FACE_MESH and mp is None:
        raise ImportError("mediapipe is not installed. Install it with: pip install mediapipe")

    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold("error")
    except Exception:
        pass

    # ==========================
    # OPEN CAMERA
    # ==========================
    cap = cv2.VideoCapture(CAMERA_INDEX, BACKEND)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Try changing CAMERA_INDEX or BACKEND.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read from camera.")
        cap.release()
        return

    H_full, W_full = frame.shape[:2]
    fps_reported = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Camera negotiated: {W_full}x{H_full} @ {fps_reported:.2f} FPS (reported)")
    print(f"[INFO] DET frame: {DET_W}x{DET_H}, detect every {DETECT_EVERY_N} frame(s)")
    print("[INFO] Press 'q' to quit.")

    # ==========================
    # INIT YUNET
    # ==========================
    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (DET_W, DET_H),
        score_threshold=SCORE_THRESH,
        nms_threshold=NMS_THRESH,
        top_k=TOP_K
    )

    # ==========================
    # INIT FACEMESH (LAZY)
    # ==========================
    face_mesh = None
    mp_face_mesh = mp.solutions.face_mesh if (USE_FACE_MESH and mp is not None) else None

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP_INNER = 13
    LOWER_LIP_INNER = 14

    # ==========================
    # STATE
    # ==========================
    ear_s = None
    mor_s = None

    last_ear_raw = None
    last_mor_raw = None

    eye_closed_state = False

    yawn_active = False
    yawn_start_t = 0.0
    yawn_count = 0
    yawn_cooldown_until = 0.0

    perclos = TimeWindowPerclos(window_sec=PERCLOS_WINDOW_SEC)

    last_faces = None
    frame_idx = 0
    fps_smooth = 0.0
    t_prev = time.perf_counter()
    t_prev_logic = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ret, frame_full = cap.read()
        t1 = time.perf_counter()

        if not ret or frame_full is None:
            continue

        H_full, W_full = frame_full.shape[:2]
        frame_det = cv2.resize(frame_full, (DET_W, DET_H), interpolation=cv2.INTER_LINEAR)

        t2 = time.perf_counter()
        if frame_idx % DETECT_EVERY_N == 0:
            detector.setInputSize((DET_W, DET_H))
            _, faces = detector.detect(frame_det)
            last_faces = faces
        else:
            faces = last_faces
        t3 = time.perf_counter()

        sx = W_full / float(DET_W)
        sy = H_full / float(DET_H)

        best_face = None
        if faces is not None and len(faces) > 0:
            best_i = int(np.argmax(faces[:, -1]))
            best_face = faces[best_i]

        ear_val = None
        mor_val = None
        have_face = False

        if best_face is not None:
            have_face = True
            x, y, w_box, h_box = best_face[:4]
            score = float(best_face[-1])

            x1 = clamp(int(x * sx), 0, W_full - 1)
            y1 = clamp(int(y * sy), 0, H_full - 1)
            x2 = clamp(int((x + w_box) * sx), 0, W_full - 1)
            y2 = clamp(int((y + h_box) * sy), 0, H_full - 1)

            cv2.rectangle(frame_full, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_full, f"{score:.2f}", (x1, max(0, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            if DRAW_YUNET_LANDMARKS:
                lms = best_face[4:14].reshape(-1, 2)
                for (lx, ly) in lms:
                    px = int(lx * sx)
                    py = int(ly * sy)
                    if 0 <= px < W_full and 0 <= py < H_full:
                        cv2.circle(frame_full, (px, py), 2, (0, 0, 255), -1)

            if USE_FACE_MESH and mp_face_mesh is not None:
                if face_mesh is None:
                    face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=FACE_MESH_MAX_FACES,
                        refine_landmarks=FACE_MESH_REFINE,
                        min_detection_confidence=FACE_MESH_DET_CONF,
                        min_tracking_confidence=FACE_MESH_TRK_CONF,
                    )

                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                bw = (x2 - x1) * ROI_SCALE
                bh = (y2 - y1) * ROI_SCALE

                rx1 = clamp(int(cx - bw * 0.5), 0, W_full - 1)
                ry1 = clamp(int(cy - bh * 0.5), 0, H_full - 1)
                rx2 = clamp(int(cx + bw * 0.5), 0, W_full - 1)
                ry2 = clamp(int(cy + bh * 0.5), 0, H_full - 1)

                roi_w = rx2 - rx1
                roi_h = ry2 - ry1

                if roi_w >= ROI_MIN_W and roi_h >= ROI_MIN_H:
                    roi = frame_full[ry1:ry2, rx1:rx2]
                    if roi.size > 0:
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        res = face_mesh.process(roi_rgb)

                        if res.multi_face_landmarks:
                            lm = res.multi_face_landmarks[0].landmark

                            def lm_xy(idx):
                                x_px = int(lm[idx].x * roi_w) + rx1
                                y_px = int(lm[idx].y * roi_h) + ry1
                                return (x_px, y_px)

                            left_eye_pts = [lm_xy(i) for i in LEFT_EYE]
                            right_eye_pts = [lm_xy(i) for i in RIGHT_EYE]
                            ear_left = eye_aspect_ratio(left_eye_pts)
                            ear_right = eye_aspect_ratio(right_eye_pts)
                            ear_val = 0.5 * (ear_left + ear_right)

                            mouth_left = lm_xy(MOUTH_LEFT)
                            mouth_right = lm_xy(MOUTH_RIGHT)
                            upper_lip = lm_xy(UPPER_LIP_INNER)
                            lower_lip = lm_xy(LOWER_LIP_INNER)
                            mor_val = mouth_open_ratio(mouth_left, mouth_right, upper_lip, lower_lip, min_width_px=20.0)

                            if DRAW_FACE_MESH_DEBUG:
                                for p in left_eye_pts + right_eye_pts:
                                    cv2.circle(frame_full, p, 1, (255, 255, 0), -1)
                                for p in [mouth_left, mouth_right, upper_lip, lower_lip]:
                                    cv2.circle(frame_full, p, 2, (255, 255, 0), -1)

        # ==========================
        # Step 3: Update logic (robust against missing raw metrics)
        # ==========================
        now_t = time.perf_counter()
        dt_logic = now_t - t_prev_logic
        t_prev_logic = now_t

        have_metrics = (ear_val is not None) and (mor_val is not None)

        if have_metrics:
            last_ear_raw = ear_val
            last_mor_raw = mor_val

            ear_s = ear_val if ear_s is None else ((1.0 - EMA_ALPHA) * ear_s + EMA_ALPHA * ear_val)
            mor_s = mor_val if mor_s is None else ((1.0 - EMA_ALPHA) * mor_s + EMA_ALPHA * mor_val)

            if not eye_closed_state:
                if ear_s < EAR_CLOSED_TH:
                    eye_closed_state = True
            else:
                if ear_s > EAR_OPEN_TH:
                    eye_closed_state = False

            perclos.add(now_t, dt_logic, eye_closed_state)

            if not yawn_active:
                if now_t >= yawn_cooldown_until and mor_s > MOR_YAWN_ON:
                    yawn_active = True
                    yawn_start_t = now_t
            else:
                if mor_s < MOR_YAWN_OFF:
                    dur = now_t - yawn_start_t
                    if dur >= YAWN_MIN_DUR_SEC:
                        yawn_count += 1
                    yawn_active = False
                    yawn_cooldown_until = now_t + YAWN_COOLDOWN_SEC

        # ==========================
        # Overlay (never crash)
        # ==========================
        dt = now_t - t_prev
        t_prev = now_t
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

        cap_ms = (t1 - t0) * 1000.0
        det_ms = (t3 - t2) * 1000.0

        cv2.putText(frame_full, f"FPS:{fps_smooth:.1f}  CAP:{W_full}x{H_full}  DET:{DET_W}x{DET_H}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_full, f"ms: cap {cap_ms:.1f} | det {det_ms:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

        y0 = 95
        if DRAW_METRICS:
            if ear_s is not None and mor_s is not None:
                p = perclos.value()
                drowsy = (p >= PERCLOS_DROWSY_TH)

                # raw may be missing some frames; show last raw if available
                ear_raw_disp = last_ear_raw if last_ear_raw is not None else ear_s
                mor_raw_disp = last_mor_raw if last_mor_raw is not None else mor_s

                cv2.putText(frame_full, f"EAR(raw/s): {ear_raw_disp:.3f}/{ear_s:.3f}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_full, f"MOR(raw/s): {mor_raw_disp:.3f}/{mor_s:.3f}", (10, y0 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.putText(frame_full, f"EYE: {'CLOSED' if eye_closed_state else 'OPEN'}", (10, y0 + 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_full, f"YAWN: {'ACTIVE' if yawn_active else '---'}  count={yawn_count}", (10, y0 + 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_full, f"PERCLOS({int(PERCLOS_WINDOW_SEC)}s): {p*100:.1f}%  drowsy={'YES' if drowsy else 'no'}",
                            (10, y0 + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame_full, "Metrics: waiting for FaceMesh...", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if not have_face:
            cv2.putText(frame_full, "No face detected", (10, H_full - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("DrowSAFE - YuNet + FaceMesh EAR/MOR + PERCLOS/Yawn", frame_full)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
