import time
from typing import List, Dict, Any, Optional
import numpy as np

from app.core.config import VLM_AUTO_INTERVAL, ALERT_COOLDOWN_SEC, DEFAULT_LANGUAGE
from app.core.safety_evaluator import SafetyEvaluator
from app.core.audio_manager import AudioManager
from app.ai.vlm_engine import VlmEngine


class NavigationMode:
    """
    Primary Operating Mode: Autonomous Real-Time Obstacle Navigation.
    Provides continuous spatial tracking, collision avoidance audio cues,
    voice alerts, and background scene context fusion.
    """

    def __init__(self, audio_manager: AudioManager, vlm_engine: VlmEngine, language: str = DEFAULT_LANGUAGE):
        self.audio_manager = audio_manager
        self.vlm_engine = vlm_engine
        self.language = language.upper()
        self.safety_evaluator = SafetyEvaluator(language=self.language)

        self.last_vlm_trigger = time.time()
        self.active = True

    def set_language(self, language: str):
        self.language = language.upper()
        self.safety_evaluator.set_language(self.language)
        self.vlm_engine.set_language(self.language)

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        active_track_ids: set
    ) -> List[Dict[str, Any]]:
        """
        Evaluates risk for all detections, triggers spoken warnings for hazards,
        and manages periodic VLM scene context analysis.
        """
        evaluations: List[Dict[str, Any]] = []

        for det in detections:
            eval_result = self.safety_evaluator.evaluate_object(det)
            if eval_result:
                evaluations.append(eval_result)

        # Cleanup disappeared tracks
        self.safety_evaluator.cleanup_tracks(active_track_ids)

        # Identify the highest-priority hazard for a spoken alert.
        priority_hazard = self.safety_evaluator.get_priority_hazard(evaluations)
        if priority_hazard:
            self._handle_hazard_alert(priority_hazard)

        # Announce additional hazards too; the most severe object must not
        # hide other nearby objects from the user.
        for evaluation in evaluations:
            if priority_hazard and evaluation["track_id"] == priority_hazard["track_id"]:
                continue
            if evaluation["risk_level"] == "HIGH":
                self.audio_manager.speak_warning(evaluation["alert_phrase"], debounce_sec=5.0)
            elif evaluation["risk_level"] == "MEDIUM":
                self.audio_manager.speak_info(evaluation["alert_phrase"], debounce_sec=6.0)
            elif evaluation.get("motion_state") == "APPROACHING":
                self.audio_manager.speak_info(evaluation["alert_phrase"], debounce_sec=6.0)
            else:
                self.audio_manager.speak_info(evaluation["alert_phrase"], debounce_sec=8.0)

        now = time.time()
        if now - self.last_vlm_trigger >= VLM_AUTO_INTERVAL:
            self.last_vlm_trigger = now
            if not self.vlm_engine.is_busy:
                self.vlm_engine.request_scene_summary(
                    frame,
                    callback=self._on_scene_description
                )

        return evaluations

    def _handle_hazard_alert(self, hazard: Dict[str, Any]):
        phrase = hazard["alert_phrase"]
        level = hazard["risk_level"]

        if not phrase:
            return

        # Voice warning with smart debouncing to prevent repetitive spam.
        if level == "CRITICAL":
            self.audio_manager.speak_urgent(phrase)
        elif level == "HIGH":
            self.audio_manager.speak_warning(phrase, debounce_sec=5.0)
        elif level == "MEDIUM":
            self.audio_manager.speak_info(phrase, debounce_sec=6.0)
        else:
            self.audio_manager.speak_info(phrase, debounce_sec=8.0)

    def _on_scene_description(self, description: str):
        if description and len(description) > 10:
            print(f"[NavigationMode] VLM Scene Context: {description}")
            self.audio_manager.speak_info(description, debounce_sec=5.0)

    def on_enter(self):
        welcome = "Navigation mode chalu hai. Rasta scan ho raha hai." if self.language == "HINDI" else "Navigation mode active. Scanning path."
        self.audio_manager.speak_info(welcome, debounce_sec=0.5)

    def on_exit(self):
        pass
