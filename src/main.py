"""
DrowSAFE — main entry point.

Starts the full detection pipeline:
  Camera → Detector → Features → Scoring → State Machine → Dashboard + Alert
"""

import sys
import os
import time
import signal
import logging

# Make project root importable regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import (
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE,
    DISPLAY_WIDTH, DISPLAY_HEIGHT, FULLSCREEN,
    SHOW_LANDMARKS, SHOW_FPS,
)
from src.camera       import Camera
from src.detector     import FaceDetector
from src.features     import FeatureExtractor
from src.scoring      import FatigueScorer
from src.state_machine import AlertStateMachine
from src.alert        import AlertController
from src.dashboard    import Dashboard
from src.logger       import EventLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("drowsafe.main")


def main():
    log.info("DrowSAFE starting...")

    # ------------------------------------------------------------------
    # Initialise subsystems
    # ------------------------------------------------------------------
    camera        = Camera(FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE)
    detector      = FaceDetector()
    features      = FeatureExtractor()
    scorer        = FatigueScorer()
    state_machine = AlertStateMachine()
    alert         = AlertController()
    dashboard     = Dashboard(DISPLAY_WIDTH, DISPLAY_HEIGHT, FULLSCREEN)
    event_logger  = EventLogger()

    # ------------------------------------------------------------------
    # Graceful shutdown on Ctrl-C or SIGTERM
    # ------------------------------------------------------------------
    running = True

    def shutdown(sig, frame):
        nonlocal running
        log.info("Shutdown signal received.")
        running = False

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    log.info("Entering main loop. Press Ctrl-C to stop.")

    fps_timer  = time.perf_counter()
    fps_count  = 0
    fps_display = 0.0

    while running:
        loop_start = time.perf_counter()

        # 1. Grab frame
        frame = camera.read()
        if frame is None:
            log.warning("No frame received from camera — skipping.")
            continue

        # 2. Run face detection + landmark extraction
        landmarks, annotated_frame = detector.process(frame)

        # 3. Extract drowsiness features
        if landmarks:
            feat = features.extract(landmarks)
        else:
            feat = None  # No face visible — keep last score, don't reset

        # 4. Compute fatigue score
        score = scorer.update(feat)

        # 5. Update alert state machine
        alert_level = state_machine.update(score)

        # 6. Trigger physical alerts (buzzer / LED)
        alert.update(alert_level)

        # 7. Log events on state change
        event_logger.log(alert_level, score, feat)

        # 8. FPS counter
        fps_count += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_timer   = time.perf_counter()

        # 9. Render dashboard
        display_frame = annotated_frame if SHOW_LANDMARKS else frame
        dashboard.render(
            frame       = display_frame,
            score       = score,
            alert_level = alert_level,
            features    = feat,
            fps         = fps_display if SHOW_FPS else None,
        )

        # 10. Check if user closed the dashboard window
        if not dashboard.is_running():
            running = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    log.info("Shutting down subsystems...")
    alert.stop()
    camera.release()
    dashboard.quit()
    event_logger.close()
    log.info("DrowSAFE stopped cleanly.")


if __name__ == "__main__":
    main()
