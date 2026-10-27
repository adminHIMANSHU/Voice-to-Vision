import time
import threading
from typing import Optional

from app.core.config import (
    GPIO_PIN_LEFT_HAPTIC,
    GPIO_PIN_RIGHT_HAPTIC,
    GPIO_PIN_BUZZER,
    ENABLE_HARDWARE_HAPTICS
)


class HardwareBridge:
    """
    IoT Hardware Abstraction Layer (HAL).
    Supports plug-and-play physical haptic feedback (vibration motors, buzzers)
    for Raspberry Pi, NVIDIA Jetson, and embedded wearable smart canes/glasses.
    Falls back gracefully to simulated logging when running on Windows/PC.
    """

    def __init__(self):
        self.hardware_available = False
        self.gpio_lib = None
        self._lock = threading.Lock()
        self.last_haptic_event = ""
        self.last_haptic_time = 0.0

        if ENABLE_HARDWARE_HAPTICS:
            self._init_hardware()
        else:
            print("[HardwareBridge] Running in IoT Simulation Mode (PC/Laptop).")

    def _init_hardware(self):
        """Attempts to initialize physical GPIO drivers (RPi.GPIO or Jetson.GPIO)."""
        try:
            import RPi.GPIO as GPIO
            self.gpio_lib = GPIO
            self.gpio_lib.setmode(self.gpio_lib.BCM)
            self.gpio_lib.setwarnings(False)

            self.gpio_lib.setup(GPIO_PIN_LEFT_HAPTIC, self.gpio_lib.OUT, initial=self.gpio_lib.LOW)
            self.gpio_lib.setup(GPIO_PIN_RIGHT_HAPTIC, self.gpio_lib.OUT, initial=self.gpio_lib.LOW)
            self.gpio_lib.setup(GPIO_PIN_BUZZER, self.gpio_lib.OUT, initial=self.gpio_lib.LOW)

            self.hardware_available = True
            print("[HardwareBridge] Physical Raspberry Pi GPIO initialized successfully.")
            return
        except ImportError:
            pass

        try:
            import Jetson.GPIO as GPIO
            self.gpio_lib = GPIO
            self.gpio_lib.setmode(self.gpio_lib.BCM)
            self.gpio_lib.setwarnings(False)

            self.gpio_lib.setup(GPIO_PIN_LEFT_HAPTIC, self.gpio_lib.OUT, initial=self.gpio_lib.LOW)
            self.gpio_lib.setup(GPIO_PIN_RIGHT_HAPTIC, self.gpio_lib.OUT, initial=self.gpio_lib.LOW)
            self.gpio_lib.setup(GPIO_PIN_BUZZER, self.gpio_lib.OUT, initial=self.gpio_lib.LOW)

            self.hardware_available = True
            print("[HardwareBridge] Physical NVIDIA Jetson GPIO initialized successfully.")
            return
        except ImportError:
            pass

        print("[HardwareBridge] GPIO drivers not found. Running in IoT Simulation Mode.")
        self.hardware_available = False

    def trigger_haptic(self, action: str, direction: str):
        """
        Triggers directional vibration patterns in a non-blocking background thread.
        - Emergency STOP: Double pulse on both motors + buzzer.
        - Hazard LEFT: Vibration pulse on Left motor.
        - Hazard RIGHT: Vibration pulse on Right motor.
        """
        if not ENABLE_HARDWARE_HAPTICS:
            return

        now = time.time()
        # Prevent haptic spam within 0.8s
        if now - self.last_haptic_time < 0.8:
            return

        self.last_haptic_time = now
        self.last_haptic_event = f"{action}_{direction}"

        thread = threading.Thread(
            target=self._haptic_worker,
            args=(action, direction),
            daemon=True,
            name="HapticWorker"
        )
        thread.start()

    def _haptic_worker(self, action: str, direction: str):
        with self._lock:
            if self.hardware_available and self.gpio_lib:
                try:
                    if action == "STOP":
                        # Double pulse emergency stop
                        for _ in range(2):
                            self.gpio_lib.output(GPIO_PIN_LEFT_HAPTIC, self.gpio_lib.HIGH)
                            self.gpio_lib.output(GPIO_PIN_RIGHT_HAPTIC, self.gpio_lib.HIGH)
                            self.gpio_lib.output(GPIO_PIN_BUZZER, self.gpio_lib.HIGH)
                            time.sleep(0.18)
                            self.gpio_lib.output(GPIO_PIN_LEFT_HAPTIC, self.gpio_lib.LOW)
                            self.gpio_lib.output(GPIO_PIN_RIGHT_HAPTIC, self.gpio_lib.LOW)
                            self.gpio_lib.output(GPIO_PIN_BUZZER, self.gpio_lib.LOW)
                            time.sleep(0.10)
                    elif direction == "LEFT" or action == "MOVE RIGHT":
                        # Pulse left motor
                        self.gpio_lib.output(GPIO_PIN_LEFT_HAPTIC, self.gpio_lib.HIGH)
                        time.sleep(0.25)
                        self.gpio_lib.output(GPIO_PIN_LEFT_HAPTIC, self.gpio_lib.LOW)
                    elif direction == "RIGHT" or action == "MOVE LEFT":
                        # Pulse right motor
                        self.gpio_lib.output(GPIO_PIN_RIGHT_HAPTIC, self.gpio_lib.HIGH)
                        time.sleep(0.25)
                        self.gpio_lib.output(GPIO_PIN_RIGHT_HAPTIC, self.gpio_lib.LOW)
                except Exception as e:
                    print(f"[HardwareBridge] GPIO output error: {e}")
            else:
                # Simulated Haptic pulse logged to console
                sim_tag = f"HAPTIC VIBRATION >> [{action}] | Direction: {direction}"
                if action == "STOP":
                    sim_tag += " (BOTH MOTORS + BUZZER PULSED)"
                elif direction == "LEFT" or action == "MOVE RIGHT":
                    sim_tag += " (LEFT MOTOR PULSED)"
                elif direction == "RIGHT" or action == "MOVE LEFT":
                    sim_tag += " (RIGHT MOTOR PULSED)"
                print(f"[IoT HAL] {sim_tag}")

    def is_hardware_active(self) -> bool:
        return self.hardware_available

    def get_last_event(self) -> str:
        if time.time() - self.last_haptic_time > 3.0:
            return ""
        return self.last_haptic_event

    def cleanup(self):
        if self.hardware_available and self.gpio_lib:
            try:
                self.gpio_lib.cleanup()
                print("[HardwareBridge] GPIO cleaned up.")
            except Exception:
                pass

