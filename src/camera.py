"""
DrowSAFE — Camera module.

Wraps OpenCV VideoCapture for the Raspberry Pi Camera Module 3 NoIR.
On Pi 5 Bookworm, uses the libcamera Python bindings via picamera2,
which is the officially supported path replacing the old GStreamer pipeline.
Falls back to SimulatedCamera if no camera is connected.
"""

import cv2
import numpy as np
import logging

log = logging.getLogger("drowsafe.camera")


class SimulatedCamera:
    """
    Generates synthetic BGR frames for dashboard testing.
    Draws a grey frame with a centered face-placeholder so the
    pipeline runs end-to-end without any physical camera.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._frame_count = 0
        log.info("SimulatedCamera ready (%dx%d) — no physical camera needed.", width, height)

    def read(self):
        self._frame_count += 1
        frame = np.full((self.height, self.width, 3), 40, dtype=np.uint8)

        cx, cy = self.width // 2, self.height // 2
        cv2.ellipse(frame, (cx, cy), (160, 200), 0, 0, 360, (100, 100, 100), 2)
        cv2.ellipse(frame, (cx - 60, cy - 40), (30, 20), 0, 0, 360, (120, 120, 120), 2)
        cv2.ellipse(frame, (cx + 60, cy - 40), (30, 20), 0, 0, 360, (120, 120, 120), 2)
        cv2.circle(frame, (cx - 60, cy - 40), 8, (150, 150, 150), -1)
        cv2.circle(frame, (cx + 60, cy - 40), 8, (150, 150, 150), -1)
        cv2.ellipse(frame, (cx, cy + 60), (50, 25), 0, 0, 180, (100, 100, 100), 2)
        cv2.putText(
            frame, "SIMULATION MODE — awaiting camera",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 80, 80), 1,
        )
        # Output RGB to match picamera2 native format used throughout pipeline
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        log.info("SimulatedCamera released.")


class Picamera2Camera:
    """
    Camera backend using picamera2 — the officially supported
    camera library for Raspberry Pi OS Bookworm on Pi 5.
    Picamera2 uses libcamera natively without GStreamer.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30, flip: bool = False):
        from picamera2 import Picamera2
        self.width  = width
        self.height = height
        self._flip  = flip
        self._cam   = Picamera2()

        config = self._cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self._cam.configure(config)
        self._cam.start()
        log.info("Picamera2 started: %dx%d @ %d fps", width, height, fps)

    def read(self):
        frame = self._cam.capture_array("main")
        if frame is None:
            return None

        # picamera2 RGB888 delivers native RGB — keep as-is.
        # Handle edge cases only.
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        # 3-channel RGB888 — no conversion needed

        if self._flip:
            frame = cv2.flip(frame, -1)
        return frame

    def release(self):
        self._cam.stop()
        self._cam.close()
        log.info("Picamera2 released.")


class Camera:
    """
    Unified camera interface for DrowSAFE.

    Priority order:
      1. picamera2  — preferred on Pi 5 Bookworm (libcamera native)
      2. OpenCV VideoCapture(0) — fallback for USB cameras / testing
      3. SimulatedCamera — fallback when no camera hardware is present
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30,
                 simulate: bool = False, flip: bool = False):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._backend = None

        if simulate:
            self._backend = SimulatedCamera(width, height, fps)
            return

        self._open(width, height, fps, flip)

    def _open(self, width, height, fps, flip):
        # --- Try picamera2 first (Pi 5 Bookworm native path) ---
        try:
            self._backend = Picamera2Camera(width, height, fps, flip=flip)
            log.info("Camera backend: picamera2")
            return
        except Exception as e:
            log.warning("picamera2 failed (%s) — trying OpenCV VideoCapture...", e)

        # --- Try OpenCV direct capture ---
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,          fps)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._backend = _OpenCVCamera(cap)
                    log.info("Camera backend: OpenCV VideoCapture")
                    return
            cap.release()
        except Exception as e:
            log.warning("OpenCV VideoCapture failed (%s)", e)

        # --- Final fallback: simulation ---
        log.warning(
            "No physical camera available — falling back to SimulatedCamera. "
            "Connect the Camera Module 3 and ensure picamera2 is installed."
        )
        self._backend = SimulatedCamera(width, height, fps)

    def read(self):
        return self._backend.read() if self._backend else None

    @property
    def is_simulated(self) -> bool:
        return isinstance(self._backend, SimulatedCamera)

    def release(self):
        if self._backend:
            self._backend.release()
        log.info("Camera released.")

    def __del__(self):
        self.release()


class _OpenCVCamera:
    """Thin wrapper to make OpenCV VideoCapture fit the same interface."""
    def __init__(self, cap):
        self._cap = cap

    def read(self):
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        self._cap.release()
