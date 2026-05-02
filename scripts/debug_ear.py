"""
DrowSAFE — EAR blink debugger.

Prints every frame's EAR value, counter states, and warn decision.
Run this, blink normally, and paste the output — it will show exactly
why the warning is triggering.

Usage:
    source venv/bin/activate
    python scripts/debug_ear.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE, CAMERA_FLIP, EAR_THRESHOLD, EAR_CONSEC_FRAMES
from src.camera   import Camera
from src.detector import FaceDetector
from src.features import FeatureExtractor

print("=" * 70)
print(f"  EAR_THRESHOLD    = {EAR_THRESHOLD}")
print(f"  EAR_CONSEC_FRAMES= {EAR_CONSEC_FRAMES}")
print("=" * 70)
print(f"  {'Frame':>6}  {'EAR':>7}  {'low_cnt':>7}  {'hi_cnt':>7}  {'WARN':>6}")
print("-" * 70)

camera    = Camera(FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE, flip=CAMERA_FLIP)
detector  = FaceDetector()
extractor = FeatureExtractor(FRAME_WIDTH, FRAME_HEIGHT)

ear_low  = 0
ear_high = 0
frame_n  = 0

try:
    while True:
        frame = camera.read()
        if frame is None:
            continue

        landmarks, _ = detector.process(frame)
        if not landmarks:
            print(f"  {'--':>6}  {'no face':>7}")
            continue

        f = extractor.extract(landmarks)
        frame_n += 1

        if f.ear < EAR_THRESHOLD:
            ear_low  += 1
            ear_high  = 0
        else:
            ear_high += 1
            if ear_high >= EAR_CONSEC_FRAMES:
                ear_low = 0

        warn = ear_low >= EAR_CONSEC_FRAMES

        # Only print when EAR is low OR just recovered (ear_high < 5)
        if f.ear < EAR_THRESHOLD or ear_high < 5 or warn:
            print(
                f"  {frame_n:>6}  {f.ear:>7.3f}  {ear_low:>7}  {ear_high:>7}  {'RED' if warn else '':>6}"
            )

        time.sleep(1 / FRAME_RATE)

except KeyboardInterrupt:
    print("\nDone.")
    camera.release()
    detector.close()
