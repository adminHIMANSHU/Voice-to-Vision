import time
from typing import Optional
import numpy as np

from app.core.config import DEFAULT_LANGUAGE
from app.core.audio_manager import AudioManager
from app.ai.vlm_engine import VlmEngine
from app.ai.speech_engine import SpeechEngine


class AssistantMode:
    """
    Interactive Conversational Vision Assistant Mode.
    Allows visually impaired users to ask questions verbally about what is in front of them.
    """

    def __init__(self, audio_manager: AudioManager, vlm_engine: VlmEngine, speech_engine: SpeechEngine, language: str = DEFAULT_LANGUAGE):
        self.audio_manager = audio_manager
        self.vlm_engine = vlm_engine
        self.speech_engine = speech_engine
        self.language = language.upper()

        self.listening = False
        self.latest_frame: Optional[np.ndarray] = None
        self.last_query = ""
        self.last_response = ""

    def set_language(self, language: str):
        self.language = language.upper()

    def on_enter(self):
        msg = "Voice Assistant mode chalu hai. Sawal poochne ke liye Space dabayein." if self.language == "HINDI" else "Voice Assistant Mode active. Press Space to ask a question."
        self.audio_manager.speak_info(msg, debounce_sec=0.5)

    def on_exit(self):
        self.listening = False

    def trigger_listening(self, frame: np.ndarray):
        """Initiates voice query recording and visual analysis."""
        if self.listening or self.vlm_engine.is_busy:
            return

        self.listening = True
        self.latest_frame = frame.copy()
        # Give the spoken prompt time to play before opening the microphone.
        prompt_msg = "Sun raha hoon, apna sawal boliye." if self.language == "HINDI" else "Listening. Please ask your question."
        self.audio_manager.speak_info(prompt_msg, debounce_sec=0.1)

        time.sleep(0.4)
        self.speech_engine.record_and_transcribe(
            duration_sec=4.0,
            callback=self._on_speech_recognized,
            language=self.language,
        )

    def _on_speech_recognized(self, query: str):
        self.listening = False
        if not query or len(query.strip()) < 2:
            self.audio_manager.speak_info("I didn't hear a question. Press space to try again.", debounce_sec=0.5)
            retry_msg = "Aapki aawaz nahi suni. Dobara koshish karne ke liye space dabayein." if self.language == "HINDI" else "I didn't hear a question. Press space to try again."
            self.audio_manager.speak_info(retry_msg, debounce_sec=0.5)
            return

        self.last_query = query
        print(f"[AssistantMode] User asked: '{query}'")
        self.audio_manager.speak_info("Analyzing visual scene...", debounce_sec=0.2)

        wait_msg = "Drishya dekha ja raha hai..." if self.language == "HINDI" else "Analyzing visual scene..."
        self.audio_manager.speak_info(wait_msg, debounce_sec=0.2)

        if self.latest_frame is not None:
            self.vlm_engine.request_question_answer(
                self.latest_frame,
                question=query,
                callback=self._on_vlm_answer
            )
        else:
            self.audio_manager.speak_info("No camera frame available to answer.", debounce_sec=0.5)
            no_cam = "Camera frame uplabdh nahi hai." if self.language == "HINDI" else "No camera frame available."
            self.audio_manager.speak_info(no_cam, debounce_sec=0.5)

    def _on_vlm_answer(self, answer: str):
        self.last_response = answer
        print(f"[AssistantMode] Answer: '{answer}'")
        self.audio_manager.speak_info(answer, debounce_sec=0.1)

