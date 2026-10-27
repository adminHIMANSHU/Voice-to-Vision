from typing import Optional
import numpy as np

from app.core.config import DEFAULT_LANGUAGE
from app.core.audio_manager import AudioManager
from app.ai.vlm_engine import VlmEngine


class ReaderMode:
    """
    Document, Label & Signboard Reading Mode.
    Allows visually impaired individuals to read medicine labels, book pages,
    room signs, or documents placed before the camera in Hindi or English.
    """

    def __init__(self, audio_manager: AudioManager, vlm_engine: VlmEngine, language: str = DEFAULT_LANGUAGE):
        self.audio_manager = audio_manager
        self.vlm_engine = vlm_engine
        self.language = language.upper()
        self.reading = False

    def set_language(self, language: str):
        self.language = language.upper()

    def on_enter(self):
        msg = "Text reader mode chalu hai. Text samne layein aur Space dabayein." if self.language == "HINDI" else "Text Reader Mode active. Hold text in front of camera and press Space."
        self.audio_manager.speak_info(msg, debounce_sec=0.5)

    def on_exit(self):
        self.reading = False

    def trigger_read(self, frame: np.ndarray):
        """Captures frame and triggers text transcription."""
        if self.reading or self.vlm_engine.is_busy:
            return

        self.reading = True
        self.audio_manager.speak_info("Reading visible text. Please hold still.", debounce_sec=0.2)
        read_msg = "Text padha ja raha hai, kripya sthir rahein." if self.language == "HINDI" else "Reading visible text. Please hold still."
        self.audio_manager.speak_info(read_msg, debounce_sec=0.2)

        self.vlm_engine.request_text_read(
            frame,
            callback=self._on_text_extracted
        )

    def _on_text_extracted(self, text_result: str):
        self.reading = False
        print(f"[ReaderMode] Extracted Text: {text_result}")

        if not text_result or len(text_result.strip()) < 3 or "no text" in text_result.lower() or "koi text" in text_result.lower():
            no_txt = "Koi saaf text nahi dikha. Kripya dhang se dikhayein." if self.language == "HINDI" else "No clear text detected. Please adjust position."
            self.audio_manager.speak_info(no_txt, debounce_sec=0.5)
        else:
            prefix = "Text me likha hai: " if self.language == "HINDI" else "Text says: "
            self.audio_manager.speak_info(f"{prefix}{text_result}", debounce_sec=0.2)
