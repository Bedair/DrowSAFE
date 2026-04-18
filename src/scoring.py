"""
DrowSAFE — Fatigue scoring module.

Computes a composite fatigue score (0–100) from:
  - PERCLOS (primary signal)
  - Instantaneous EAR
  - Yawn frequency (MAR)
  - Head pose (pitch / nod)

The score drives the alert state machine.
"""

import time
import logging
from collections import deque
from config.config import (
    EAR_THRESHOLD,
    MAR_THRESHOLD,
    HEAD_PITCH_THRESHOLD,
    PERCLOS_WINDOW_SEC,
    PERCLOS_THRESHOLD,
    FRAME_RATE,
    SCORE_WEIGHT_PERCLOS,
    SCORE_WEIGHT_EAR,
    SCORE_WEIGHT_MAR,
    SCORE_WEIGHT_HEAD_POSE,
)

log = logging.getLogger("drowsafe.scoring")


class FatigueScorer:
    """
    Maintains a rolling window of per-frame signals and produces a
    single fatigue score in the range [0, 100].

    Score interpretation
    --------------------
    0  – 39  : Alert
    40 – 69  : Warning
    70 – 100 : Critical
    """

    def __init__(self):
        # PERCLOS rolling window: stores (timestamp, eye_closed) tuples
        window_frames = int(PERCLOS_WINDOW_SEC * FRAME_RATE)
        self._eye_history: deque = deque(maxlen=window_frames)

        # Yawn counter over the same window
        self._yawn_history: deque = deque(maxlen=window_frames)

        # Head nod counter
        self._nod_history: deque  = deque(maxlen=window_frames)

        self._last_score: float   = 0.0
        self._yawn_in_progress    = False
        self._nod_in_progress     = False

        log.info("FatigueScorer ready (PERCLOS window=%ds)", PERCLOS_WINDOW_SEC)

    def update(self, features) -> float:
        """
        Update rolling history with the latest frame's features and
        return the current fatigue score.

        Parameters
        ----------
        features : Features | None
            Extracted features from the current frame.
            If None (no face visible), the scorer holds its last score.

        Returns
        -------
        float
            Fatigue score in [0, 100].
        """
        now = time.monotonic()

        if features is None:
            # No face detected — don't penalise immediately but don't reset
            return self._last_score

        # --- Eye closed? ---
        eye_closed = features.ear < EAR_THRESHOLD
        self._eye_history.append((now, eye_closed))

        # --- Yawning? (rising edge detection) ---
        yawning = features.mar > MAR_THRESHOLD
        if yawning and not self._yawn_in_progress:
            self._yawn_in_progress = True
        elif not yawning and self._yawn_in_progress:
            self._yawn_in_progress = False
            self._yawn_history.append(now)  # Record completed yawn timestamp

        # --- Nodding? ---
        nodding = features.head_pitch > HEAD_PITCH_THRESHOLD
        if nodding and not self._nod_in_progress:
            self._nod_in_progress = True
        elif not nodding and self._nod_in_progress:
            self._nod_in_progress = False
            self._nod_history.append(now)

        # --- Prune expired history entries ---
        cutoff = now - PERCLOS_WINDOW_SEC
        while self._yawn_history and self._yawn_history[0] < cutoff:
            self._yawn_history.popleft()
        while self._nod_history and self._nod_history[0] < cutoff:
            self._nod_history.popleft()

        # --- PERCLOS ---
        perclos = self._compute_perclos()

        # --- Normalised sub-scores (each 0–1) ---
        # EAR: invert and normalise so 0 = fully open, 1 = fully closed
        ear_norm      = max(0.0, min(1.0, 1.0 - (features.ear / EAR_THRESHOLD)))

        # PERCLOS: normalise relative to threshold (1.0 = at threshold, >1 = above)
        perclos_norm  = min(1.0, perclos / max(PERCLOS_THRESHOLD, 1e-6))

        # MAR / yawn frequency: yawns per minute, capped at 1.0 above 6/min
        yawns_per_min = len(self._yawn_history) / (PERCLOS_WINDOW_SEC / 60.0)
        mar_norm      = min(1.0, yawns_per_min / 6.0)

        # Head pose: nods per minute, capped at 1.0 above 10/min
        nods_per_min  = len(self._nod_history) / (PERCLOS_WINDOW_SEC / 60.0)
        pose_norm     = min(1.0, nods_per_min / 10.0)

        # --- Composite weighted score ---
        raw = (
            SCORE_WEIGHT_PERCLOS   * perclos_norm +
            SCORE_WEIGHT_EAR       * ear_norm     +
            SCORE_WEIGHT_MAR       * mar_norm     +
            SCORE_WEIGHT_HEAD_POSE * pose_norm
        )

        score = round(min(100.0, max(0.0, raw * 100.0)), 1)
        self._last_score = score
        return score

    def _compute_perclos(self) -> float:
        """
        Compute PERCLOS from the rolling eye closure history.

        Returns fraction of frames where eyes were closed, in [0, 1].
        """
        if not self._eye_history:
            return 0.0
        closed = sum(1 for _, c in self._eye_history if c)
        return closed / len(self._eye_history)

    @property
    def perclos(self) -> float:
        """Current PERCLOS value (read-only)."""
        return self._compute_perclos()

    def reset(self):
        """Clear all rolling history (e.g. driver change)."""
        self._eye_history.clear()
        self._yawn_history.clear()
        self._nod_history.clear()
        self._last_score = 0.0
        log.info("FatigueScorer reset.")
