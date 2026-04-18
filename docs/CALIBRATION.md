# DrowSAFE — Threshold Calibration Guide

Every face is different. The default thresholds in `config/config.py` are based on published research averages but **should be tuned to the specific driver** for best accuracy.

---

## Which thresholds to calibrate

| Parameter | Default | What it controls |
|---|---|---|
| `EAR_THRESHOLD` | 0.22 | Eye considered "closed" below this |
| `MAR_THRESHOLD` | 0.60 | Mouth considered "yawning" above this |
| `HEAD_PITCH_THRESHOLD` | 20° | Forward nod angle |
| `PERCLOS_THRESHOLD` | 0.15 | Fraction of closed frames → drowsy |

---

## Step 1 — Measure your baseline EAR

Run the system with `SHOW_LANDMARKS = True` in `config.py`. The dashboard displays the live EAR value.

1. Sit in your normal driving position in front of the camera.
2. Keep your eyes **fully open** and look straight ahead for 30 seconds. Note the EAR range (e.g. 0.28–0.35).
3. **Blink naturally** — note how low EAR drops during a blink (e.g. 0.05–0.12).
4. **Squint or look sideways** — note the EAR at partial closure.

Set `EAR_THRESHOLD` to a value **between** your natural open range and your blink floor. A value about 60–70% of your natural open EAR usually works well.

**Example:** natural open EAR = 0.30, blink floor = 0.08 → set threshold to `0.22`.

---

## Step 2 — Measure your baseline MAR

1. Keep your mouth **closed normally** — note the MAR (usually 0.15–0.35).
2. Open your mouth wide as if yawning — note the MAR (usually 0.65–0.85).

Set `MAR_THRESHOLD` to a value clearly above your resting MAR but below your yawn MAR.

---

## Step 3 — Measure your head nod angle

The dashboard shows live `Head pitch` in degrees.

1. Sit straight — note the pitch (should be near 0°, ±5°).
2. Let your head droop forward as it would when nodding off — note the pitch (typically 15°–35°).

Set `HEAD_PITCH_THRESHOLD` to a value between your upright reading and your nod reading.

---

## Step 4 — Validate

After updating `config.py`:

1. Run DrowSAFE for 5 minutes while **fully alert**. The fatigue score should stay below 20.
2. Deliberately blink slowly and repeatedly for 30 seconds. The score should rise noticeably.
3. Deliberately yawn 3–4 times. The score should rise.
4. Deliberately nod your head forward for 5 seconds. The score should rise.

If any of these don't behave as expected, adjust the relevant threshold and repeat.

---

## Lighting considerations

- The Camera Module 3 NoIR performs well in low light but landmark confidence drops in very dark conditions.
- Avoid strong backlight (e.g. direct sunlight behind the driver) — it causes under-exposure on the face.
- If driving at night without IR illumination, increase `min_detection_confidence` in `detector.py` to reduce false positives from low-confidence landmarks.
