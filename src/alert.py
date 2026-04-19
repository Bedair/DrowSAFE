"""
DrowSAFE — Alert controller.

Drives the GPIO buzzer based on the current alert level.
Uses a background thread so buzzer PWM never blocks the main pipeline.

Uses lgpio (the correct GPIO library for Raspberry Pi 5 on Bookworm).
RPi.GPIO is not supported on Pi 5 — lgpio is its replacement.

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
    import lgpio
    _GPIO_CHIP = lgpio.gpiochip_open(0)
    _GPIO_AVAILABLE = True
    log.info("lgpio initialised successfully.")
except (ImportError, RuntimeError, Exception) as e:
    _GPIO_AVAILABLE = False
    _GPIO_CHIP = None
    log.warning(
        "lgpio not available (%s) — running in simulation mode. "
        "Buzzer alerts will be logged only.", e
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
        self._level   = ALERT
        self._running = True
        self._lock    = threading.Lock()
        self._thread  = threading.Thread(
            target=self._buzzer_loop, daemon=True, name="buzzer"
        )

        if _GPIO_AVAILABLE:
            lgpio.gpio_claim_output(_GPIO_CHIP, BUZZER_PIN, 0)
            log.info("GPIO pin %d (BCM) claimed as output.", BUZZER_PIN)

        self._thread.start()
        log.info("AlertController started.")

    def update(self, alert_level: int):
        """Update the target alert level (thread-safe)."""
        with self._lock:
            self._level = alert_level

    def _set_buzzer(self, state: bool):
        """Drive the buzzer pin HIGH (True) or LOW (False)."""
        if _GPIO_AVAILABLE and _GPIO_CHIP is not None:
            lgpio.gpio_write(_GPIO_CHIP, BUZZER_PIN, 1 if state else 0)

    def _buzzer_loop(self):
        """Background thread: generates buzzer patterns based on alert level."""
        while self._running:
            with self._lock:
                level = self._level

            if level == ALERT:
                self._set_buzzer(False)
                time.sleep(0.1)

            elif level == WARNING:
                period = 1.0 / BUZZER_WARNING_HZ
                self._set_buzzer(True)
                time.sleep(period / 2)
                self._set_buzzer(False)
                time.sleep(period / 2)

            elif level == CRITICAL:
                period = 1.0 / BUZZER_CRITICAL_HZ
                self._set_buzzer(True)
                time.sleep(period / 2)
                self._set_buzzer(False)
                time.sleep(period / 2)

        self._set_buzzer(False)

    def stop(self):
        """Stop the buzzer thread and clean up GPIO."""
        self._running = False
        self._thread.join(timeout=2.0)

        if _GPIO_AVAILABLE and _GPIO_CHIP is not None:
            lgpio.gpio_write(_GPIO_CHIP, BUZZER_PIN, 0)
            lgpio.gpio_free(_GPIO_CHIP, BUZZER_PIN)
            lgpio.gpiochip_close(_GPIO_CHIP)
            log.info("lgpio GPIO cleaned up.")

        log.info("AlertController stopped.")
