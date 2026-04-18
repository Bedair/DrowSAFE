#!/bin/bash
# =============================================================================
# DrowSAFE — Launch Script
# Activates the virtual environment and starts the main pipeline.
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

# Set display for Pygame on the DSI touchscreen
export DISPLAY=:0

# Disable DPMS / screen blanking during session
xset s off -dpms 2>/dev/null || true

echo "Starting DrowSAFE..."
cd "$PROJECT_DIR"
python src/main.py "$@"
