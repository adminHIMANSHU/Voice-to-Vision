import time
import queue
import threading
from typing import Optional, Callable
import numpy as np
import sounddevice as sd
import whisper

from app.core.config import DEFAULT_LANGUAGE, WHISPER_MODEL_NAME


class SpeechEngine:
    """
    Speech Recognition Engine using OpenAI Whisper.
    Captures microphone input and transcribes spoken queries for conversational vision.
    """

    def __init__(self, model_name: str = WHISPER_MODEL_NAME, sample_rate: int = 16000):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.model = None

        self.recording = False
        self.audio_buffer = []
        self.transcription_thread: Optional[threading.Thread] = None

    def load_model(self):
        if self.model is None:
            print(f"[SpeechEngine] Loading Whisper '{self.model_name}' model...")
            self.model = whisper.load_model(self.model_name)
            print("[SpeechEngine] Whisper model loaded.")

    def record_and_transcribe(
        self,
        duration_sec: float = 3.5,
        callback: Optional[Callable[[str], None]] = None,
        language: str = DEFAULT_LANGUAGE,
    ):
        """
        Records audio from microphone for `duration_sec` asynchronously,
        then transcribes and invokes `callback(transcription)`.
        """
        thread = threading.Thread(
            target=self._record_worker,
            args=(duration_sec, callback, language),
            daemon=True,
            name="SpeechWorker"
        )
        thread.start()

    def _record_worker(
        self,
        duration_sec: float,
        callback: Optional[Callable[[str], None]],
        language: str,
    ):
        try:
            self.load_model()
            print(f"[SpeechEngine] Listening for {duration_sec}s...")

            num_samples = int(duration_sec * self.sample_rate)
            # Record 1-channel 16kHz audio
            audio_data = sd.rec(
                num_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            # Flatten to 1D float32 array
            audio_np = audio_data.flatten()

            # Check if there is actual audio signal above noise floor
            energy = np.sqrt(np.mean(audio_np ** 2))
            if energy < 0.005:
                print("[SpeechEngine] Audio too faint / silent.")
                if callback:
                    callback("")
                return

            print("[SpeechEngine] Transcribing audio with Whisper...")
            # Transcribe with fp16=False for CPU compatibility
            result = self.model.transcribe(
                audio_np,
                fp16=False,
                language="hi" if language.upper() == "HINDI" else "en"
            )

            text = result.get("text", "").strip()
            print(f"[SpeechEngine] Heard: '{text}'")

            if callback:
                callback(text)

        except Exception as e:
            print(f"[SpeechEngine] Speech recognition error: {e}")
            if callback:
                callback("")
