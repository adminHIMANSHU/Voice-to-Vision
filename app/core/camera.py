import cv2
import time
import threading
from typing import Optional, Tuple
import numpy as np

from app.core.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


class CameraStream:
    """
    High-performance, threaded camera capture stream.
    Eliminates frame buffering and decoupling video capture from inference.
    """

    def __init__(self, src: int = CAMERA_INDEX, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        self.src = src
        self.target_width = width
        self.target_height = height

        self.cap = cv2.VideoCapture(self.src)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        self.grabbed, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Performance telemetry
        self.fps = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def start(self) -> "CameraStream":
        if self.running:
            return self

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True, name="CameraWorker")
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue

            grabbed, frame = self.cap.read()
            if grabbed:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame

                # FPS accounting
                self._fps_counter += 1
                now = time.time()
                elapsed = now - self._fps_timer
                if elapsed >= 1.0:
                    self.fps = round(self._fps_counter / elapsed, 1)
                    self._fps_counter = 0
                    self._fps_timer = now
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def get_fps(self) -> float:
        return self.fps

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
