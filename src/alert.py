"""
DrowSAFE — Alert controller.

Drives the GPIO buzzer based on the current alert level.
Uses lgpio (the correct GPIO library for Raspberry Pi 5 on Bookworm).
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
    log.warning("lgpio not available (%s) — buzzer disabled.", e)

from config.config import BUZZER_PIN, BUZZER_WARNING_HZ, BUZZER_CRITICAL_HZ
from src.state_machine import ALERT, WARNING, CRITICAL


def _claim_pin(chip, pin):
    """Claim GPIO output pin, freeing it first if already busy."""
    try:
        lgpio.gpio_claim_output(chip, pin, 0)
        return True
    except Exception:
        pass
    # Pin busy — try to free it and reclaim
    try:
        lgpio.gpio_free(chip, pin)
        lgpio.gpio_claim_output(chip, pin, 0)
        log.info("GPIO pin %d reclaimed after busy state.", pin)
        return True
    except Exception as e:
        log.warning("Could not claim GPIO pin %d: %s — buzzer disabled.", pin, e)
        return False


class AlertController:
    def __init__(self):
        self._level    = ALERT
        self._running  = True
        self._lock     = threading.Lock()
        self._pin_ok   = False
        self._thread   = threading.Thread(
            target=self._buzzer_loop, daemon=True, name="buzzer"
        )

        if _GPIO_AVAILABLE and _GPIO_CHIP is not None:
            self._pin_ok = _claim_pin(_GPIO_CHIP, BUZZER_PIN)

        self._thread.start()
        log.info("AlertController started (buzzer=%s).", "enabled" if self._pin_ok else "disabled")

    def update(self, alert_level: int):
        with self._lock:
            self._level = alert_level

    def _set_buzzer(self, state: bool):
        if self._pin_ok and _GPIO_CHIP is not None:
            try:
                lgpio.gpio_write(_GPIO_CHIP, BUZZER_PIN, 1 if state else 0)
            except Exception:
                pass

    def _buzzer_loop(self):
        while self._running:
            with self._lock:
                level = self._level

            if level == ALERT:
                self._set_buzzer(False)
                time.sleep(0.1)
            elif level == WARNING:
                period = 1.0 / BUZZER_WARNING_HZ
                self._set_buzzer(True);  time.sleep(period / 2)
                self._set_buzzer(False); time.sleep(period / 2)
            elif level == CRITICAL:
                period = 1.0 / BUZZER_CRITICAL_HZ
                self._set_buzzer(True);  time.sleep(period / 2)
                self._set_buzzer(False); time.sleep(period / 2)

        self._set_buzzer(False)

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        if self._pin_ok and _GPIO_CHIP is not None:
            try:
                lgpio.gpio_write(_GPIO_CHIP, BUZZER_PIN, 0)
                lgpio.gpio_free(_GPIO_CHIP, BUZZER_PIN)
                lgpio.gpiochip_close(_GPIO_CHIP)
            except Exception:
                pass
        log.info("AlertController stopped.")
