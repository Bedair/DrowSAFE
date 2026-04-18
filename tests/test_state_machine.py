"""
Unit tests for DrowSAFE alert state machine.
Validates all transitions and hysteresis behaviour.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.state_machine import AlertStateMachine, ALERT, WARNING, CRITICAL


class TestAlertStateMachine:

    def test_initial_state_is_alert(self):
        sm = AlertStateMachine()
        assert sm.level == ALERT

    # ------------------------------------------------------------------
    # Upward transitions
    # ------------------------------------------------------------------
    def test_alert_to_warning(self):
        sm = AlertStateMachine()
        level = sm.update(40.0)   # WARNING_SCORE = 40
        assert level == WARNING

    def test_alert_to_warning_below_threshold(self):
        sm = AlertStateMachine()
        level = sm.update(39.9)
        assert level == ALERT

    def test_warning_to_critical(self):
        sm = AlertStateMachine()
        sm.update(40.0)           # → WARNING
        level = sm.update(70.0)   # CRITICAL_SCORE = 70
        assert level == CRITICAL

    def test_alert_cannot_jump_to_critical_directly(self):
        """State machine must pass through WARNING first."""
        sm = AlertStateMachine()
        level = sm.update(95.0)   # Above CRITICAL_SCORE but starts in ALERT
        assert level == WARNING   # Not CRITICAL — must pass through WARNING first

    # ------------------------------------------------------------------
    # Downward transitions (hysteresis)
    # ------------------------------------------------------------------
    def test_warning_to_alert_via_hysteresis(self):
        sm = AlertStateMachine()
        sm.update(40.0)           # → WARNING
        level = sm.update(29.9)   # WARNING_HYSTERESIS = 30 → should drop to ALERT
        assert level == ALERT

    def test_warning_does_not_drop_above_hysteresis(self):
        sm = AlertStateMachine()
        sm.update(40.0)           # → WARNING
        level = sm.update(35.0)   # Above WARNING_HYSTERESIS(30), below WARNING_SCORE(40)
        assert level == WARNING   # Stays in WARNING

    def test_critical_to_warning_via_hysteresis(self):
        sm = AlertStateMachine()
        sm.update(40.0)           # → WARNING
        sm.update(70.0)           # → CRITICAL
        level = sm.update(54.9)   # CRITICAL_HYSTERESIS = 55 → drops to WARNING
        assert level == WARNING

    def test_critical_does_not_drop_above_hysteresis(self):
        sm = AlertStateMachine()
        sm.update(40.0)
        sm.update(70.0)           # → CRITICAL
        level = sm.update(60.0)   # Above CRITICAL_HYSTERESIS(55) → stays CRITICAL
        assert level == CRITICAL

    def test_critical_cannot_jump_to_alert_directly(self):
        """Must step down: CRITICAL → WARNING, then WARNING → ALERT."""
        sm = AlertStateMachine()
        sm.update(40.0)
        sm.update(70.0)           # → CRITICAL
        level = sm.update(0.0)    # Score = 0, but must pass through WARNING first
        assert level == WARNING   # Not ALERT directly

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def test_reset_returns_to_alert(self):
        sm = AlertStateMachine()
        sm.update(40.0)
        sm.update(70.0)
        sm.reset()
        assert sm.level == ALERT

    def test_level_name(self):
        sm = AlertStateMachine()
        assert sm.level_name == "ALERT"
        sm.update(40.0)
        assert sm.level_name == "WARNING"
        sm.update(70.0)
        assert sm.level_name == "CRITICAL"

    # ------------------------------------------------------------------
    # Stability — no flickering on boundary scores
    # ------------------------------------------------------------------
    def test_no_flicker_near_warning_boundary(self):
        """Rapid oscillation near threshold should not flip state."""
        sm = AlertStateMachine()
        sm.update(40.0)  # → WARNING

        # Oscillate between 35 (above hysteresis) and 42
        levels = []
        for i in range(20):
            score  = 35.0 if i % 2 == 0 else 42.0
            levels.append(sm.update(score))

        # Should stay in WARNING throughout — never drops to ALERT (hysteresis=30)
        assert all(l == WARNING for l in levels)
