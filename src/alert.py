"""
DrowSAFE — Alert controller.

Drives the GPIO buzzer based on the current alert level.
Uses a background thread so buzzer PWM never blocks the main pipeline.

Alert behaviours
----------------
  Level 0 (ALERT)    : buzzer off
  Level 1 (WARNING)  : intermittent beep at BUZZER_WARNING_HZ
  Level 2 (CRITICAL) : rapid continuous beep at BUZZER_CRITICAL_HZ
"""

import threading
import time
import logging

log = logging.getLogger("drowsafe.alert")

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    _GPIO_AVAILABLE = False
    log.warning(
        "RPi.GPIO not available — running in simulation mode. "
        "Buzzer alerts will be logged only."
    )

from config.config import BUZZER_PIN, BUZZER_WARNING_HZ, BUZZER_CRITICAL_HZ
from src.state_machine import ALERT, WARNING, CRITICAL


class AlertController:
    """
    Background-threaded buzzer controller.

    Safe to call `update()` every frame — it only changes hardware
    state when the alert level changes.
    """

    def __init__(self):
        self._level    = ALERT
        self._running  = True
        self._lock     = threading.Lock()
        self._thread   = threading.Thread(
            target=self._buzzer_loop, daemon=True, name="buzzer"
        )

        if _GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUZZER_PIN, GPIO.OUT)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            log.info("GPIO initialised on BCM pin %d.", BUZZER_PIN)

        self._thread.start()
        log.info("AlertController started.")

    def update(self, alert_level: int):
        """Update the target alert level (thread-safe)."""
        with self._lock:
            self._level = alert_level

    def _set_buzzer(self, state: bool):
        """Drive the buzzer pin HIGH or LOW."""
        if _GPIO_AVAILABLE:
            GPIO.output(BUZZER_PIN, GPIO.HIGH if state else GPIO.LOW)

    def _buzzer_loop(self):
        """
        Background thread: generates buzzer patterns based on alert level.
        """
        while self._running:
            with self._lock:
                level = self._level

            if level == ALERT:
                self._set_buzzer(False)
                time.sleep(0.1)

            elif level == WARNING:
                # Intermittent beep: on for half period, off for half period
                period = 1.0 / BUZZER_WARNING_HZ
                self._set_buzzer(True)
                time.sleep(period / 2)
                self._set_buzzer(False)
                time.sleep(period / 2)

            elif level == CRITICAL:
                # Rapid beep
                period = 1.0 / BUZZER_CRITICAL_HZ
                self._set_buzzer(True)
                time.sleep(period / 2)
                self._set_buzzer(False)
                time.sleep(period / 2)

        # Ensure buzzer is off on exit
        self._set_buzzer(False)

    def stop(self):
        """Stop the buzzer thread and clean up GPIO."""
        self._running = False
        self._thread.join(timeout=2.0)

        if _GPIO_AVAILABLE:
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            GPIO.cleanup(BUZZER_PIN)
            log.info("GPIO cleaned up.")

        log.info("AlertController stopped.")
