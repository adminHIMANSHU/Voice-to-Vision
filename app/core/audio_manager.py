import time
import queue
import threading
from typing import Optional, List
import pyttsx3

from app.core.config import (
    SPEECH_RATE,
    SPEECH_VOLUME,
    ALERT_COOLDOWN_SEC,
    PRIORITY_CRITICAL,
    PRIORITY_WARNING,
    PRIORITY_INFO,
)


class AudioManager:
    """
    Non-blocking, thread-safe speech synthesis manager.
    Queues utterances by priority and speaks in a dedicated daemon thread.
    Prevents any interruption or stuttering of the primary video feed.
    """

    def __init__(self, rate: int = SPEECH_RATE, volume: float = SPEECH_VOLUME):
        self.rate = rate
        self.volume = volume

        # Priority queue item format: (priority, timestamp, message)
        self.queue: queue.PriorityQueue = queue.PriorityQueue()
        self.lock = threading.Lock()
        self.running = False
        self.muted = False

        # Announcement tracking for debouncing and HUD closed captions
        self.last_spoken_time = {}
        self.current_caption = "System Ready"
        self.caption_timestamp = time.time()
        self.history: List[str] = []

        self.worker_thread: Optional[threading.Thread] = None

    def start(self) -> "AudioManager":
        if self.running:
            return self

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._speech_worker,
            daemon=True,
            name="AudioWorker"
        )
        self.worker_thread.start()
        return self

    def _speech_worker(self):
        """Worker loop that initializes TTS within the dedicated thread for COM stability."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
        except Exception as e:
            print(f"[AudioManager] Error initializing pyttsx3: {e}")
            engine = None

        while self.running:
            try:
                # Wait for up to 0.2s for an utterance
                priority, ts, message = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self.muted:
                self.queue.task_done()
                continue

            # Update live caption for HUD display
            with self.lock:
                self.current_caption = message
                self.caption_timestamp = time.time()
                self.history.append(message)
                if len(self.history) > 20:
                    self.history.pop(0)

            print(f"[VOICE] >> {message}")

            if engine is not None:
                try:
                    engine.say(message)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[AudioManager] Speech output error: {e}")

            self.queue.task_done()

    def speak(self, message: str, priority: int = PRIORITY_INFO, debounce_sec: float = ALERT_COOLDOWN_SEC, force: bool = False):
        """
        Submits an utterance to the speech queue.
        Applies cooldown logic to prevent redundant speech spam.
        """
        if not message or not message.strip():
            return

        message = message.strip()
        now = time.time()

        # Check debouncer unless forced
        if not force:
            last_time = self.last_spoken_time.get(message, 0)
            if now - last_time < debounce_sec:
                return

        self.last_spoken_time[message] = now

        # High priority clears backlog if critically urgent
        if priority == PRIORITY_CRITICAL:
            with self.lock:
                # Purge lower priority messages if a critical stop alert arrives
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                        self.queue.task_done()
                    except queue.Empty:
                        break

        self.queue.put((priority, now, message))

    def speak_urgent(self, message: str):
        """Preempts current queue to deliver an immediate hazard alert."""
        self.speak(message, priority=PRIORITY_CRITICAL, debounce_sec=1.5, force=True)

    def speak_warning(self, message: str, debounce_sec: float = ALERT_COOLDOWN_SEC):
        self.speak(message, priority=PRIORITY_WARNING, debounce_sec=debounce_sec)

    def speak_info(self, message: str, debounce_sec: float = 3.0):
        self.speak(message, priority=PRIORITY_INFO, debounce_sec=debounce_sec)

    def get_live_caption(self) -> str:
        """Returns the current spoken message for the HUD subtitle bar."""
        with self.lock:
            # Clear caption if older than 5 seconds
            if time.time() - self.caption_timestamp > 5.0:
                return ""
            return self.current_caption

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def stop(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
