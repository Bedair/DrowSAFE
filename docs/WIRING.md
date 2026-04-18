# DrowSAFE — GPIO Wiring Guide

## Buzzer

DrowSAFE uses a **5V active piezo buzzer** driven from a GPIO pin via a transistor or direct connection.

### Components needed

| Component | Specification |
|---|---|
| Active piezo buzzer | 5V, any standard module |
| NPN transistor | 2N2222 or BC547 (optional but recommended) |
| Resistor | 1kΩ (base resistor, if using transistor) |
| Jumper wires | Female–female |

---

### Option A — Direct GPIO (3.3V buzzer)

If your buzzer is rated for 3.3V:

```
GPIO 18 (BCM pin 18) ──────── Buzzer +
GND                  ──────── Buzzer −
```

> ⚠ Most active buzzers are rated 5V. A 3.3V GPIO signal may produce a quieter buzz — acceptable for a POC.

---

### Option B — Transistor-switched 5V (recommended)

For a louder, reliable 5V buzzer:

```
GPIO 18 ──[1kΩ]── Base (NPN transistor)
                   Collector ── Buzzer +
                   Emitter  ── GND

5V pin  ──────────────────── Buzzer + (via collector)
GND     ──────────────────── Emitter
```

Raspberry Pi 5 GPIO pinout reference:

```
 Pin 1  [3.3V]    [5V]     Pin 2
 Pin 3  [SDA1]    [5V]     Pin 4
 Pin 5  [SCL1]    [GND]    Pin 6
 Pin 7  [GPIO4]   [TX]     Pin 8
 Pin 9  [GND]     [RX]     Pin 10
 Pin 11 [GPIO17]  [GPIO18] Pin 12  ← GPIO 18 used here
 Pin 13 [GPIO27]  [GND]    Pin 14
 ...
```

BCM GPIO 18 = Physical pin 12.

---

### Changing the GPIO pin

To use a different GPIO pin, update `config/config.py`:

```python
BUZZER_PIN = 18   # Change to your preferred BCM pin number
```

---

### Testing the buzzer

With DrowSAFE running, the buzzer can be tested by temporarily forcing an alert level. From a Python shell with the venv active:

```python
from src.alert import AlertController
from src.state_machine import WARNING, CRITICAL

alert = AlertController()
alert.update(WARNING)   # Soft intermittent beep
import time; time.sleep(5)
alert.update(CRITICAL)  # Rapid alarm
time.sleep(3)
alert.stop()
```
