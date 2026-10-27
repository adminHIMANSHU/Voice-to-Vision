import time
import queue
import threading
from typing import Optional, Callable
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from app.core.config import VLM_MODEL_ID, VLM_MAX_NEW_TOKENS, DEFAULT_LANGUAGE


class VlmEngine:
    """
    Asynchronous Visual Language Model (SmolVLM) reasoning engine.
    Executes scene descriptions, conversational visual question answering,
    and text reading in a background thread without dropping video frames.
    Understands:
    - Who is in front of the camera? (Identity & Person count)
    - What are they doing? (Action, activity, sitting/standing/walking)
    - How are they positioned? (Posture, direction, surroundings)
    Generates bilingual descriptions in Hindi and English.
    """

    def __init__(self, model_id: str = VLM_MODEL_ID, default_lang: str = DEFAULT_LANGUAGE):
        self.model_id = model_id
        self.language = default_lang.upper()
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.job_queue: queue.Queue = queue.Queue(maxsize=3)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        self.latest_context = ""
        self.latest_context_time = 0.0
        self.lock = threading.Lock()
        self.is_busy = False

    def set_language(self, lang: str):
        self.language = lang.upper()

    def load_model(self):
        """Loads SmolVLM weights."""
        print(f"[VlmEngine] Loading SmolVLM on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        print("[VlmEngine] SmolVLM loaded and ready.")

    def start(self) -> "VlmEngine":
        if self.running:
            return self

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="VlmWorker"
        )
        self.worker_thread.start()
        return self

    def _worker_loop(self):
        # Lazy load if not already loaded
        if self.model is None:
            self.load_model()

        while self.running:
            try:
                task = self.job_queue.get(timeout=0.3)
            except queue.Empty:
                continue

            frame, prompt, callback = task
            self.is_busy = True

            try:
                response = self._infer(frame, prompt)
                with self.lock:
                    self.latest_context = response
                    self.latest_context_time = time.time()

                if callback:
                    callback(response)

            except Exception as e:
                print(f"[VlmEngine] Inference error: {e}")
                err_msg = "Maaf kijiye, drishya samajhne me samasya aayi." if self.language == "HINDI" else "I encountered an issue processing the scene."
                if callback:
                    callback(err_msg)
            finally:
                self.is_busy = False
                self.job_queue.task_done()

    def _infer(self, frame: np.ndarray, prompt: str) -> str:
        """Executes model generation with chat template."""
        # Convert BGR frame to RGB PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=VLM_MAX_NEW_TOKENS
            )

        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract assistant response if conversational tags exist
        # Extract assistant response
        if "Assistant:" in output:
            output = output.split("Assistant:", 1)[1].strip()
        elif "assistant\n" in output:
            output = output.split("assistant\n", 1)[1].strip()

        return output.strip().replace("\n", " ")

    def request_scene_summary(self, frame: np.ndarray, callback: Optional[Callable[[str], None]] = None):
        """Queues a general scene description request."""
        if self.language == "HINDI":
            prompt = (
                "Describe in simple, natural romanized Hindi (Hinglish like 'Saamne ek vyakti bed par baitha hai aur camera ki taraf dekh raha hai'): "
                "Who is in front of the camera, what is their exact action or activity (sitting, standing, lying down, looking), "
                "and what is their surroundings in 1 to 2 short sentences. "
                "If two roads or paths split left and right, do not choose for the user; ask exactly: 'Aapko baayein jaana hai ya daayein?'"
            )
        else:
            prompt = (
                "For a visually impaired person, describe concisely in 1 to 2 sentences: "
                "Who is in front of the camera, what is their exact action or activity (e.g. sitting on the bed, standing, walking, looking at camera), "
                "their posture, and their surroundings. If two roads or paths split left and right, ask: 'Which way do you want to go, left or right?'"
            )
        self.submit_job(frame, prompt, callback)

    def request_question_answer(self, frame: np.ndarray, question: str, callback: Optional[Callable[[str], None]] = None):
        """Queues a question regarding the current visual field."""
        if self.language == "HINDI":
            prompt = (
                f"A visually impaired user is asking in Hindi: '{question}'. "
                "Look at the image carefully and answer concisely in simple romanized Hindi (Hinglish) in 1 sentence."
            )
        else:
            prompt = (
                f"A visually impaired user is asking: '{question}'. "
                "Answer directly and concisely based strictly on what is visible in the image in 1 sentence."
            )
        self.submit_job(frame, prompt, callback)

    def request_text_read(self, frame: np.ndarray, callback: Optional[Callable[[str], None]] = None):
        """Queues an OCR text extraction request."""
        if self.language == "HINDI":
            prompt = (
                "Transcribe any visible printed or handwritten text, signboards, book text, or package labels in this image. "
                "If no text is found, reply 'Koi text nahi mila'."
            )
        else:
            prompt = (
                "Read and transcribe any visible printed or handwritten text, signboards, labels, "
                "or document text in this image clearly for a blind person. If no text is visible, say 'No text detected'."
            )
        self.submit_job(frame, prompt, callback)

    def submit_job(self, frame: np.ndarray, prompt: str, callback: Optional[Callable[[str], None]] = None):
        """Submits a job if queue is not clogged."""
        if self.job_queue.full():
            # Drop older pending requests to keep it fresh
            try:
                self.job_queue.get_nowait()
                self.job_queue.task_done()
            except queue.Empty:
                pass

        self.job_queue.put((frame.copy(), prompt, callback))

    def get_latest_context(self) -> str:
        with self.lock:
            return self.latest_context

    def stop(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
