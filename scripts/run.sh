#!/bin/bash
# =============================================================================
# DrowSAFE — Launch Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run scripts/setup_drowsafe.sh first."
    exit 1
fi

# --- Display setup for Pygame ---
# When running via SSH, DISPLAY must point to the Pi's local X session.
# The RPi Touch Display v1.1 runs on :0 by default.
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo "DISPLAY set to :0 (SSH session detected)"
fi

# Allow the current user to connect to the X display
xhost +local: 2>/dev/null || true

# Disable screen blanking during session
xset s off -dpms 2>/dev/null || true

echo "Starting DrowSAFE..."
cd "$PROJECT_DIR"
python src/main.py "$@"
