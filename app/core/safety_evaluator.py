from collections import defaultdict, deque
from typing import Dict, List, Any, Optional

from app.core.config import (
    CRITICAL_ENTER,
    CRITICAL_EXIT,
    HIGH_ENTER,
    HIGH_EXIT,
    MEDIUM_ENTER,
    MEDIUM_EXIT,
    WEIGHT_DISTANCE,
    WEIGHT_DIRECTION,
    WEIGHT_CONFIDENCE,
    STABILITY_FRAMES,
    RISK_WINDOW_SIZE,
    DEFAULT_LANGUAGE,
    HINDI_OBJECT_NAMES
)


class SafetyEvaluator:
    """
    Evaluates multi-factor risk scores, manages hysteresis state transitions,
    and formats unambiguous bilingual navigation directives.
    """

    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self.language = language.upper()
        self.stability_counts = defaultdict(int)
        self.risk_histories = defaultdict(lambda: deque(maxlen=RISK_WINDOW_SIZE))
        self.current_risk_levels = {}
        self.last_announced_actions = {}

    def set_language(self, language: str):
        self.language = language.upper()

    def evaluate_object(self, detection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluates risk for an individual detected and tracked object.
        Returns None if the object hasn't met the temporal stability threshold yet.
        """
        track_id = detection.get("track_id", "unknown")
        object_name = detection["name"]
        confidence = detection["confidence"]
        direction = detection["direction"]
        distance = detection["distance"]
        motion_state = detection.get("motion_state", "STATIONARY")

        # 1. Temporal Stability Check
        self.stability_counts[track_id] += 1
        if self.stability_counts[track_id] < STABILITY_FRAMES:
            return None

        # 2. Base Factor Scoring
        distance_score = {"NEAR": 75, "MEDIUM": 40, "FAR": 10}.get(distance, 15)
        direction_score = {"CENTER": 80, "LEFT": 40, "RIGHT": 40}.get(direction, 20)
        confidence_score = confidence * 100

        # Weighted base risk
        raw_risk = (
            WEIGHT_DISTANCE * distance_score
            + WEIGHT_DIRECTION * direction_score
            + WEIGHT_CONFIDENCE * confidence_score
        )

        # Dynamic motion amplification: approaching objects pose immediate hazard
        if motion_state == "APPROACHING":
            raw_risk += 15.0
        elif motion_state == "RECEDING":
            raw_risk -= 10.0

        raw_risk = max(0.0, min(100.0, raw_risk))

        # 3. Rolling Window Smoothing
        self.risk_histories[track_id].append(raw_risk)
        smoothed_risk = round(sum(self.risk_histories[track_id]) / len(self.risk_histories[track_id]), 1)

        # 4. Hysteresis Risk Level Calculation
        prev_level = self.current_risk_levels.get(track_id, "LOW")
        new_level = prev_level

        if prev_level == "CRITICAL":
            if smoothed_risk < CRITICAL_EXIT:
                new_level = "HIGH" if smoothed_risk >= HIGH_EXIT else ("MEDIUM" if smoothed_risk >= MEDIUM_EXIT else "LOW")
        elif prev_level == "HIGH":
            if smoothed_risk >= CRITICAL_ENTER:
                new_level = "CRITICAL"
            elif smoothed_risk < HIGH_EXIT:
                new_level = "MEDIUM" if smoothed_risk >= MEDIUM_EXIT else "LOW"
        elif prev_level == "MEDIUM":
            if smoothed_risk >= HIGH_ENTER:
                new_level = "HIGH"
            elif smoothed_risk < MEDIUM_EXIT:
                new_level = "LOW"
        else:  # LOW
            if smoothed_risk >= CRITICAL_ENTER:
                new_level = "CRITICAL"
            elif smoothed_risk >= HIGH_ENTER:
                new_level = "HIGH"
            elif smoothed_risk >= MEDIUM_ENTER:
                new_level = "MEDIUM"

        self.current_risk_levels[track_id] = new_level

        # 5. Navigational Action Determination
        if new_level == "CRITICAL":
            action = "STOP"
        elif new_level == "HIGH":
            if direction == "CENTER":
                action = "STOP"
            elif direction == "LEFT":
                action = "MOVE RIGHT"
            elif direction == "RIGHT":
                action = "MOVE LEFT"
            else:
                action = "CAUTION"
        elif new_level == "MEDIUM":
            action = "CAUTION"
        else:
            action = "SAFE"

        # 6. Audible phrasing generation
        alert_phrase = self._build_alert_phrase(object_name, action, direction, distance, motion_state, new_level)

        return {
            "track_id": track_id,
            "object": object_name,
            "confidence": confidence,
            "direction": direction,
            "distance": distance,
            "motion_state": motion_state,
            "raw_risk": raw_risk,
            "smoothed_risk": smoothed_risk,
            "risk_level": new_level,
            "action": action,
            "alert_phrase": alert_phrase,
            "bbox": detection["bbox"]
        }

    def _build_alert_phrase(self, obj: str, action: str, direction: str, distance: str, motion: str, level: str) -> str:
        """Construct concise, natural language voice instructions."""
        hindi_obj = HINDI_OBJECT_NAMES.get(obj.lower(), obj)

        if self.language == "HINDI":
            dir_text = "saamne" if direction == "CENTER" else ("baayein" if direction == "LEFT" else "daayein")
            motion_text = ", paas aa raha hai" if motion == "APPROACHING" else ""

            if level == "CRITICAL":
                return f"Ruk jaaiye! Saamne {hindi_obj} hai{motion_text}."
            elif level == "HIGH":
                if action == "STOP":
                    return f"Ruk jaaiye. {hindi_obj.title()} bilkul saamne hai."
                elif action == "MOVE RIGHT":
                    return f"Daayein mudiye. Baayein taraf {hindi_obj} hai."
                elif action == "MOVE LEFT":
                    return f"Baayein mudiye. Daayein taraf {hindi_obj} hai."
                return f"Savdhaan. {hindi_obj.title()} paas me hai."
            elif level == "MEDIUM":
                return f"Savdhaan. {dir_text.title()} taraf {hindi_obj} hai."
            elif motion == "APPROACHING":
                return f"Savdhaan. {hindi_obj} paas aa raha hai."
            if direction == "CENTER":
                return f"Saamne {hindi_obj} hai."
            return f"{dir_text.title()} taraf {hindi_obj} hai."

        dir_text = "ahead" if direction == "CENTER" else f"on your {direction.lower()}"
        motion_text = ", approaching" if motion == "APPROACHING" else ""
        if level == "CRITICAL":
            return f"Stop! {obj.title()} {dir_text}{motion_text}."
        if level == "HIGH":
            if action == "STOP":
                return f"Stop. {obj.title()} directly ahead."
            if action in ["MOVE RIGHT", "MOVE LEFT"]:
                return f"{action.capitalize()}. {obj.title()} {dir_text}."
            return f"Caution. {obj.title()} close {dir_text}."
        if level == "MEDIUM":
            return f"Caution. {obj.title()} {dir_text}."
        if motion == "APPROACHING":
            return f"Caution. {obj.title()} is approaching."
        return f"{obj.title()} is {dir_text}."

    def get_priority_hazard(self, evaluations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Sorts active evaluations and selects the single most critical obstacle
        demanding immediate user attention.
        """
        if not evaluations:
            return None

        level_weights = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        # Sort by level severity first, then smoothed risk score descending
        sorted_evals = sorted(
            evaluations,
            key=lambda x: (level_weights.get(x["risk_level"], 0), x["smoothed_risk"]),
            reverse=True
        )
        return sorted_evals[0]

    def cleanup_tracks(self, active_track_ids: set):
        """Cleans up internal state for vanished tracks."""
        for tid in list(self.stability_counts.keys()):
            if tid not in active_track_ids:
                del self.stability_counts[tid]
                if tid in self.risk_histories:
                    del self.risk_histories[tid]
                if tid in self.current_risk_levels:
                    del self.current_risk_levels[tid]
                if tid in self.last_announced_actions:
                    del self.last_announced_actions[tid]
