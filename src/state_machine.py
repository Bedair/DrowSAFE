"""
DrowSAFE — Alert state machine.

Three-level state machine with hysteresis to prevent flickering.

States
------
  0 — ALERT    : driver is awake, no action
  1 — WARNING  : early drowsiness signs, soft alert
  2 — CRITICAL : severe drowsiness, sustained alarm

Transitions
-----------
  Score rises:   0 → 1 when score ≥ WARNING_SCORE
                 1 → 2 when score ≥ CRITICAL_SCORE
  Score falls:   2 → 1 when score < CRITICAL_HYSTERESIS
                 1 → 0 when score < WARNING_HYSTERESIS
"""

import logging
from config.config import (
    WARNING_SCORE, CRITICAL_SCORE,
    WARNING_HYSTERESIS, CRITICAL_HYSTERESIS,
)

log = logging.getLogger("drowsafe.state_machine")

ALERT    = 0
WARNING  = 1
CRITICAL = 2

LEVEL_NAMES = {ALERT: "ALERT", WARNING: "WARNING", CRITICAL: "CRITICAL"}


class AlertStateMachine:
    """Hysteresis-based alert level state machine."""

    def __init__(self):
        self._level = ALERT
        log.info("AlertStateMachine ready. Initial state: ALERT")

    def update(self, score: float) -> int:
        """
        Evaluate the current fatigue score and return the new alert level.

        Parameters
        ----------
        score : float
            Fatigue score in [0, 100] from FatigueScorer.

        Returns
        -------
        int
            Alert level: 0 (ALERT), 1 (WARNING), 2 (CRITICAL).
        """
        prev = self._level

        if self._level == ALERT:
            if score >= WARNING_SCORE:
                self._level = WARNING

        elif self._level == WARNING:
            if score >= CRITICAL_SCORE:
                self._level = CRITICAL
            elif score < WARNING_HYSTERESIS:
                self._level = ALERT

        elif self._level == CRITICAL:
            if score < CRITICAL_HYSTERESIS:
                self._level = WARNING

        if self._level != prev:
            log.info(
                "State transition: %s → %s (score=%.1f)",
                LEVEL_NAMES[prev], LEVEL_NAMES[self._level], score,
            )

        return self._level

    @property
    def level(self) -> int:
        return self._level

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES[self._level]

    def reset(self):
        """Reset to ALERT state."""
        self._level = ALERT
        log.info("AlertStateMachine reset to ALERT.")
