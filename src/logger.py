"""
DrowSAFE — Event logger.

Writes drowsiness events to a timestamped CSV file in the logs/ directory.
A new file is created each time DrowSAFE starts.
"""

import csv
import os
import time
import logging
from datetime import datetime

from config.config import LOG_DIR, LOG_EVENTS
from src.state_machine import LEVEL_NAMES

log = logging.getLogger("drowsafe.logger")


class EventLogger:
    """
    Logs alert state transitions and periodic fatigue score snapshots
    to a CSV file for post-session analysis.

    CSV columns
    -----------
    timestamp, elapsed_s, alert_level, alert_name,
    fatigue_score, ear, mar, head_pitch
    """

    SNAPSHOT_INTERVAL = 5.0  # Write a row every N seconds regardless of state

    def __init__(self):
        self._file    = None
        self._writer  = None
        self._last_level = -1
        self._last_snap  = 0.0
        self._start_time = time.monotonic()

        if LOG_EVENTS:
            self._open_file()

    def _open_file(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(LOG_DIR, f"drowsafe_{ts}.csv")

        self._file   = open(filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "timestamp", "elapsed_s", "alert_level", "alert_name",
            "fatigue_score", "ear", "mar", "head_pitch",
        ])
        log.info("Event log: %s", filepath)

    def log(self, alert_level: int, score: float, features):
        """
        Write a row on state change or periodic snapshot.

        Parameters
        ----------
        alert_level : int
        score       : float
        features    : Features | None
        """
        if not LOG_EVENTS or self._writer is None:
            return

        now     = time.monotonic()
        elapsed = round(now - self._start_time, 2)

        state_changed = alert_level != self._last_level
        snapshot_due  = (now - self._last_snap) >= self.SNAPSHOT_INTERVAL

        if not (state_changed or snapshot_due):
            return

        ear   = round(features.ear,        3) if features else ""
        mar   = round(features.mar,        3) if features else ""
        pitch = round(features.head_pitch, 1) if features else ""

        self._writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            elapsed,
            alert_level,
            LEVEL_NAMES[alert_level],
            score,
            ear, mar, pitch,
        ])
        self._file.flush()

        self._last_level = alert_level
        self._last_snap  = now

    def close(self):
        if self._file:
            self._file.close()
            log.info("Event log closed.")
