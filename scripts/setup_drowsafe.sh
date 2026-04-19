#!/bin/bash
# =============================================================================
# DrowSAFE — Environment Setup Script for Raspberry Pi 5
# Run once after fresh Raspberry Pi OS (64-bit Bookworm) install.
# Usage: bash scripts/setup_drowsafe.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

echo "============================================="
echo "  DrowSAFE Environment Setup"
echo "  Raspberry Pi 5 | Python 3 | 64-bit OS"
echo "============================================="

# -----------------------------------------------------------------------------
# 1. System update
# -----------------------------------------------------------------------------
echo ""
echo "[1/5] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# -----------------------------------------------------------------------------
# 2. System-level dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-numpy \
    libcap-dev \
    libcamera-apps \
    libopencv-dev \
    python3-opencv \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libegl1 \
    libgl1 \
    fonts-dejavu-core \
    git \
    curl \
    python3-lgpio \
    python3-rpi-lgpio

# -----------------------------------------------------------------------------
# 3. Python virtual environment
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Creating Python virtual environment..."

# --system-site-packages exposes system-level OpenCV inside the venv
python3 -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# -----------------------------------------------------------------------------
# 4. Python packages
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Installing Python packages..."
pip install -r "$PROJECT_DIR/requirements.txt"

# -----------------------------------------------------------------------------
# 5. Verify
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Verifying installations..."

python3 - << 'PYCHECK'
import sys
print(f"Python: {sys.version.split()[0]}")

modules = [
    ("cv2",         "OpenCV"),
    ("mediapipe",   "MediaPipe"),
    ("onnxruntime", "ONNX Runtime"),
    ("pygame",      "Pygame"),
    ("numpy",       "NumPy"),
    ("scipy",       "SciPy"),
    ("PIL",         "Pillow"),
    ("RPi.GPIO",    "RPi.GPIO"),
]

all_ok = True
print("\n--- Package Check ---")
for mod, name in modules:
    try:
        m   = __import__(mod)
        ver = getattr(m, "__version__", "ok")
        print(f"  ✓  {name:20s} {ver}")
    except ImportError as e:
        print(f"  ✗  {name:20s} MISSING — {e}")
        all_ok = False

print()
print("All packages OK!" if all_ok else "WARNING: Some packages missing.")
PYCHECK

echo ""
echo "============================================="
echo "  Setup complete!"
echo ""
echo "  To launch DrowSAFE:"
echo "    bash scripts/run.sh"
echo ""
echo "  To run unit tests:"
echo "    source venv/bin/activate"
echo "    pytest tests/ -v"
echo "============================================="
