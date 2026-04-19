"""
DrowSAFE — Camera module.

Wraps OpenCV VideoCapture for the Raspberry Pi Camera Module 3.
Includes a SimulatedCamera for testing the dashboard without hardware.
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

        # Draw a simple face placeholder so the UI looks realistic
        cx, cy = self.width // 2, self.height // 2

        # Head oval
        cv2.ellipse(frame, (cx, cy), (160, 200), 0, 0, 360, (100, 100, 100), 2)

        # Eyes
        cv2.ellipse(frame, (cx - 60, cy - 40), (30, 20), 0, 0, 360, (120, 120, 120), 2)
        cv2.ellipse(frame, (cx + 60, cy - 40), (30, 20), 0, 0, 360, (120, 120, 120), 2)

        # Pupils
        cv2.circle(frame, (cx - 60, cy - 40), 8, (150, 150, 150), -1)
        cv2.circle(frame, (cx + 60, cy - 40), 8, (150, 150, 150), -1)

        # Mouth
        cv2.ellipse(frame, (cx, cy + 60), (50, 25), 0, 0, 180, (100, 100, 100), 2)

        # Label
        cv2.putText(
            frame, "SIMULATION MODE — awaiting camera",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 200), 1,
        )

        return frame

    def release(self):
        log.info("SimulatedCamera released.")


class Camera:
    """
    Manages frame capture from the Pi Camera Module 3.

    On Raspberry Pi OS (Bookworm), the recommended way to access
    the camera via OpenCV is through the libcamera GStreamer pipeline.
    Falls back to direct VideoCapture index if GStreamer is unavailable.
    Falls back to SimulatedCamera if no camera is connected at all.
    """

    GSTREAMER_PIPELINE = (
        "libcamerasrc ! "
        "video/x-raw,width={w},height={h},framerate={fps}/1,format=RGBx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1"
    )

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30,
                 simulate: bool = False):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._cap   = None
        self._sim   = None

        if simulate:
            self._sim = SimulatedCamera(width, height, fps)
            return

        self._open()

    def _open(self):
        pipeline = self.GSTREAMER_PIPELINE.format(
            w=self.width, h=self.height, fps=self.fps
        )

        log.info("Opening camera via GStreamer libcamera pipeline...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            log.warning(
                "GStreamer pipeline failed — trying VideoCapture(0)..."
            )
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS,          self.fps)

        if not cap.isOpened():
            log.warning(
                "No physical camera found — falling back to SimulatedCamera. "
                "Connect the Camera Module 3 to enable real detection."
            )
            self._sim = SimulatedCamera(self.width, self.height, self.fps)
            return

        self._cap = cap
        log.info("Camera opened: %dx%d @ %d fps", self.width, self.height, self.fps)

    def read(self):
        if self._sim is not None:
            return self._sim.read()

        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            log.error("Frame read failed.")
            return None

        return frame

    @property
    def is_simulated(self) -> bool:
        return self._sim is not None

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._sim is not None:
            self._sim.release()
        log.info("Camera released.")

    def __del__(self):
        self.release()
