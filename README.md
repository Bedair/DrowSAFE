# DrowSAFE 🚗💤

**Real-Time Driver Drowsiness Detection System — Powered by Edge AI on Raspberry Pi 5**

> Submitted to the [Hackster Best of 2025 Competition](https://www.hackster.io/contests/best-of-2025-competition) — SBC Category

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()

---

## Overview

DrowSAFE is an open-source, fully offline driver drowsiness detection system that runs entirely on a **Raspberry Pi 5**. It uses a camera to monitor the driver's face in real time and detects early signs of fatigue — prolonged eye closure, excessive blinking, yawning, and head nodding — then triggers audible and visual alerts before an accident can happen.

No cloud. No subscription. No internet required. Just edge AI running at the dashboard of any vehicle.

---

## Hardware

| Component | Model | Notes |
|---|---|---|
| SBC | Raspberry Pi 5 4GB | Main compute unit |
| Camera | Raspberry Pi Camera Module 3 NoIR 12MP | No IR cut filter — works in low light |
| Display | Raspberry Pi Touch Display v1.1 7" | DSI connection, shows dashboard UI |
| Buzzer | Active piezo buzzer | GPIO-driven alert |
| Cable | RPi Camera Cable Standard→Mini 300mm | Required for Pi 5 mini CSI port |

---

## How It Works

DrowSAFE runs a continuous detection pipeline on the Pi 5 at ~30 fps:

```
Camera Frame
    │
    ▼
MediaPipe Face Mesh  ──►  468 facial landmarks
    │
    ▼
Feature Extraction
    ├── EAR  (Eye Aspect Ratio)       — detects eye closure / blinks
    ├── MAR  (Mouth Aspect Ratio)     — detects yawning
    └── Head Pose (PnP solve)         — detects forward head nod
    │
    ▼
Fatigue Scoring
    └── PERCLOS (% eye closure / 60s rolling window)
    │
    ▼
Alert State Machine
    ├── Level 0 — Alert    (green)
    ├── Level 1 — Warning  (amber)  → soft beep
    └── Level 2 — Critical (red)    → sustained alarm
    │
    ▼
Dashboard UI  +  GPIO Buzzer
```

All processing is **on-device**. No data leaves the vehicle.

---

## Project Structure

```
drowsafe/
├── src/
│   ├── main.py              # Entry point — starts the full pipeline
│   ├── camera.py            # Camera capture and frame management
│   ├── detector.py          # MediaPipe face mesh + landmark extraction
│   ├── features.py          # EAR, MAR, head pose calculations
│   ├── scoring.py           # PERCLOS and fatigue score computation
│   ├── state_machine.py     # Alert level logic with hysteresis
│   ├── alert.py             # Buzzer and display alert triggers
│   ├── dashboard.py         # Pygame UI — live fatigue dashboard
│   └── logger.py            # CSV event logger
├── config/
│   └── config.py            # All tunable thresholds and settings
├── tests/
│   ├── test_features.py     # Unit tests for EAR/MAR/head pose
│   ├── test_scoring.py      # Unit tests for PERCLOS scoring
│   └── test_state_machine.py# Unit tests for alert state machine
├── scripts/
│   ├── setup_drowsafe.sh    # One-shot environment setup script
│   └── run.sh               # Launch script (activates venv + starts main.py)
├── assets/
│   ├── sounds/              # Alert audio files (.wav)
│   └── fonts/               # UI fonts
├── docs/
│   ├── images/              # Architecture diagrams, demo screenshots
│   ├── CALIBRATION.md       # How to tune thresholds for your face
│   └── WIRING.md            # Buzzer GPIO wiring diagram and instructions
├── logs/                    # Runtime event logs (CSV, gitignored)
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/drowsafe.git
cd drowsafe
```

### 2. Run the setup script (first time only)

```bash
bash scripts/setup_drowsafe.sh
```

This installs all system and Python dependencies and creates the virtual environment.

### 3. Activate the virtual environment

```bash
source venv/bin/activate
```

### 4. Launch DrowSAFE

```bash
bash scripts/run.sh
```

Or directly:

```bash
python src/main.py
```

---

## Configuration

All thresholds are in `config/config.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | 0.22 | Eye aspect ratio below this = closed |
| `MAR_THRESHOLD` | 0.6 | Mouth aspect ratio above this = yawning |
| `PERCLOS_THRESHOLD` | 0.15 | >15% eye closure per minute = drowsy |
| `WARNING_SCORE` | 40 | Fatigue score to trigger level 1 alert |
| `CRITICAL_SCORE` | 70 | Fatigue score to trigger level 2 alert |
| `FRAME_WIDTH/HEIGHT` | 1280×720 | Camera capture resolution |

See `docs/CALIBRATION.md` for guidance on tuning these to your face and lighting conditions.

---

## GPIO Wiring

See `docs/WIRING.md` for the full buzzer wiring diagram.

Quick reference:
- Buzzer **+** → GPIO 18 (BCM)
- Buzzer **−** → GND

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Face landmarks | MediaPipe Face Mesh |
| Inference runtime | ONNX Runtime |
| Camera I/O | OpenCV + libcamera |
| Dashboard UI | Pygame |
| GPIO control | RPi.GPIO |
| OS | Raspberry Pi OS 64-bit (Bookworm) |

---

## Roadmap

- [x] Project architecture design
- [x] Environment setup script
- [ ] Camera pipeline + live frame capture
- [ ] MediaPipe landmark extraction
- [ ] EAR / MAR / head pose features
- [ ] PERCLOS fatigue scoring
- [ ] Alert state machine
- [ ] Dashboard UI
- [ ] GPIO buzzer integration
- [ ] End-to-end integration test
- [ ] Threshold calibration
- [ ] Final demo video

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built for the [Hackster Best of 2025 Competition](https://www.hackster.io/contests/best-of-2025-competition).
