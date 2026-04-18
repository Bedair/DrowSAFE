"""
DrowSAFE — Camera module.

Wraps OpenCV VideoCapture for the Raspberry Pi Camera Module 3.
Uses the libcamera GStreamer backend on Pi OS Bookworm.
"""

import cv2
import logging

log = logging.getLogger("drowsafe.camera")


class Camera:
    """
    Manages frame capture from the Pi Camera Module 3.

    On Raspberry Pi OS (Bookworm), the recommended way to access
    the camera via OpenCV is through the libcamera GStreamer pipeline.
    Falls back to direct VideoCapture index if GStreamer is unavailable.
    """

    # libcamera GStreamer pipeline for Pi Camera Module 3
    # Outputs BGR frames directly usable by OpenCV / MediaPipe
    GSTREAMER_PIPELINE = (
        "libcamerasrc ! "
        "video/x-raw,width={w},height={h},framerate={fps}/1,format=RGBx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1"
    )

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._cap   = None
        self._open()

    def _open(self):
        """Open the camera — tries GStreamer pipeline first, then fallback."""
        pipeline = self.GSTREAMER_PIPELINE.format(
            w=self.width, h=self.height, fps=self.fps
        )

        log.info("Opening camera via GStreamer libcamera pipeline...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            log.warning(
                "GStreamer pipeline failed — falling back to VideoCapture(0). "
                "This is normal when testing without a physical camera."
            )
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS,          self.fps)

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open any camera source. "
                "Check that the camera is connected and enabled in raspi-config."
            )

        self._cap = cap
        log.info(
            "Camera opened: %dx%d @ %d fps",
            self.width, self.height, self.fps,
        )

    def read(self):
        """
        Read the next frame.

        Returns
        -------
        numpy.ndarray or None
            BGR frame if successful, None on read failure.
        """
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            log.error("Frame read failed.")
            return None

        return frame

    def release(self):
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Camera released.")

    def __del__(self):
        self.release()
