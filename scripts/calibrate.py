"""
DrowSAFE — Calibration helper.

Prints live EAR and MAR values to the terminal so you can
find the right thresholds for your face.

Usage:
    python scripts/calibrate.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE, CAMERA_FLIP
from src.camera   import Camera
from src.detector import FaceDetector
from src.features import FeatureExtractor

print("=" * 55)
print("  DrowSAFE Calibration Helper")
print("=" * 55)
print("Sit in your normal position and look straight ahead.")
print("Keep mouth CLOSED — note the MAR range.")
print("Open mouth wide (yawn) — note the MAR range.")
print("Press Ctrl-C to stop.")
print("-" * 55)
print(f"  {'EAR-L':>7}  {'EAR-R':>7}  {'EAR-AVG':>7}  {'MAR':>7}  {'PITCH':>7}")
print("-" * 55)

camera   = Camera(FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE, flip=CAMERA_FLIP)
detector = FaceDetector()
extractor= FeatureExtractor(FRAME_WIDTH, FRAME_HEIGHT)

try:
    while True:
        frame = camera.read()
        if frame is None:
            continue

        landmarks, _ = detector.process(frame)
        if landmarks:
            f = extractor.extract(landmarks)
            print(
                f"  {f.ear_left:7.3f}  {f.ear_right:7.3f}  "
                f"{f.ear:7.3f}  {f.mar:7.3f}  {f.head_pitch:7.1f}°",
                end="\r",
            )
        else:
            print("  -- no face detected --                           ", end="\r")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\nCalibration stopped.")
    camera.release()
    detector.close()
