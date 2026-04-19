"""
DrowSAFE — main entry point.

Starts the full detection pipeline:
  Camera → Detector → Features → Scoring → State Machine → Dashboard + Alert

Run modes
---------
  Normal   : python src/main.py
  Simulate : python src/main.py --simulate
             Runs the full dashboard with synthetic fatigue data.
             No camera or MediaPipe required. Use while awaiting hardware.
"""

import sys
import os
import time
import math
import signal
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import (
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE,
    DISPLAY_WIDTH, DISPLAY_HEIGHT, FULLSCREEN,
    SHOW_LANDMARKS, SHOW_FPS,
)
from src.camera        import Camera
from src.scoring       import FatigueScorer
from src.state_machine import AlertStateMachine
from src.alert         import AlertController
from src.dashboard     import Dashboard
from src.logger        import EventLogger
from src.features      import Features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("drowsafe.main")


# ---------------------------------------------------------------------------
# Synthetic feature generator for simulation mode
# ---------------------------------------------------------------------------
def _simulate_features(t: float) -> Features:
    """
    Generate realistic-looking drowsiness features that slowly cycle
    through alert → warning → critical → recovery over ~90 seconds.
    Useful for demoing the full dashboard without a camera.
    """
    cycle = t % 90.0  # 90-second demo cycle

    if cycle < 30:
        # Phase 1: alert driver (0–30s)
        ear   = 0.30 + 0.04 * math.sin(t * 2)
        mar   = 0.20 + 0.05 * math.sin(t * 1.3)
        pitch = 2.0  * math.sin(t * 0.5)
    elif cycle < 55:
        # Phase 2: growing drowsiness (30–55s)
        p     = (cycle - 30) / 25.0
        ear   = 0.30 - p * 0.12 + 0.02 * math.sin(t * 3)
        mar   = 0.20 + p * 0.30 + 0.05 * math.sin(t * 0.8)
        pitch = p * 15.0 + 2.0 * math.sin(t * 0.4)
    elif cycle < 70:
        # Phase 3: critical drowsiness (55–70s)
        ear   = 0.14 + 0.04 * math.sin(t * 4)
        mar   = 0.65 + 0.10 * math.sin(t * 0.6)
        pitch = 22.0 + 5.0 * math.sin(t * 0.3)
    else:
        # Phase 4: recovery (70–90s)
        p     = (cycle - 70) / 20.0
        ear   = 0.14 + p * 0.16 + 0.02 * math.sin(t * 2)
        mar   = 0.65 - p * 0.45
        pitch = 22.0 - p * 20.0

    return Features(
        ear        = max(0.05, min(0.45, ear)),
        ear_left   = max(0.05, min(0.45, ear + 0.01)),
        ear_right  = max(0.05, min(0.45, ear - 0.01)),
        mar        = max(0.10, min(0.90, mar)),
        head_pitch = pitch,
        head_yaw   = 3.0 * math.sin(t * 0.2),
        head_roll  = 1.5 * math.sin(t * 0.3),
        face_visible = True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DrowSAFE driver drowsiness detection")
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run in simulation mode (no camera or MediaPipe required)"
    )
    args = parser.parse_args()

    simulate = args.simulate

    log.info("DrowSAFE starting... (mode=%s)", "SIMULATE" if simulate else "LIVE")

    # ------------------------------------------------------------------
    # Initialise subsystems
    # ------------------------------------------------------------------
    camera        = Camera(FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE, simulate=simulate)

    # In simulation mode, skip detector import (MediaPipe not needed)
    detector = None
    if not simulate:
        from src.detector import FaceDetector
        from src.features import FeatureExtractor
        detector  = FaceDetector()
        extractor = FeatureExtractor(FRAME_WIDTH, FRAME_HEIGHT)
    else:
        extractor = None
        log.info("Simulation mode: MediaPipe detector bypassed.")

    scorer        = FatigueScorer()
    state_machine = AlertStateMachine()
    alert         = AlertController()
    dashboard     = Dashboard(DISPLAY_WIDTH, DISPLAY_HEIGHT, FULLSCREEN)
    event_logger  = EventLogger()

    # ------------------------------------------------------------------
    # Graceful shutdown
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
    log.info("Entering main loop. Press Ctrl-C or ESC to stop.")

    fps_timer   = time.perf_counter()
    fps_count   = 0
    fps_display = 0.0
    start_time  = time.perf_counter()

    while running:
        loop_start = time.perf_counter()
        t = loop_start - start_time

        # 1. Grab frame
        frame = camera.read()
        if frame is None:
            continue

        # 2. Extract features
        if simulate:
            feat           = _simulate_features(t)
            annotated_frame = frame
        else:
            landmarks, annotated_frame = detector.process(frame)
            feat = extractor.extract(landmarks) if landmarks else None

        # 3. Score + state machine
        score       = scorer.update(feat)
        alert_level = state_machine.update(score)

        # 4. Physical alert
        alert.update(alert_level)

        # 5. Log
        event_logger.log(alert_level, score, feat)

        # 6. FPS counter
        fps_count += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_timer   = time.perf_counter()

        # 7. Render
        display_frame = annotated_frame if (SHOW_LANDMARKS and not simulate) else frame
        dashboard.render(
            frame       = display_frame,
            score       = score,
            alert_level = alert_level,
            features    = feat,
            fps         = fps_display if SHOW_FPS else None,
            simulated   = simulate,
        )

        if not dashboard.is_running():
            running = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    log.info("Shutting down...")
    alert.stop()
    camera.release()
    dashboard.quit()
    event_logger.close()
    if detector:
        detector.close()
    log.info("DrowSAFE stopped cleanly.")


if __name__ == "__main__":
    main()
