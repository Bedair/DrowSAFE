#!/bin/bash
# DrowSAFE — Calibration helper launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$PROJECT_DIR/venv/bin/activate"

if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi

cd "$PROJECT_DIR"
python scripts/calibrate.py
