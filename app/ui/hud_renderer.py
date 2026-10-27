import cv2
import time
import math
from typing import List, Dict, Any, Optional
import numpy as np

from app.core.config import (
    COLOR_BG_DARK,
    COLOR_SAFE,
    COLOR_CAUTION,
    COLOR_WARNING,
    COLOR_CRITICAL,
    COLOR_ACCENT,
    COLOR_WHITE,
    COLOR_GRAY,
)


class HudRenderer:
    """
    Modern Assistive Heads-Up Display (HUD) renderer for VoiceToVision.
    Renders top status telemetry, bounding chips, directional mini-radar,
    risk gauges, and spoken subtitles with high aesthetic polish.
    risk gauges, IoT status, and spoken subtitles with high aesthetic polish.
    """

    def __init__(self):
        self.radar_radius = 85
        self.radar_margin = 20

    def render(
        self,
        frame: np.ndarray,
        mode_name: str,
        evaluations: List[Dict[str, Any]],
        priority_hazard: Optional[Dict[str, Any]],
        fps: float,
        caption: str,
        language: str = "HINDI",
        iot_active: bool = False,
        vlm_busy: bool = False,
        speech_busy: bool = False,
        is_muted: bool = False
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Render Bounding Boxes & Tags
        self._render_bounding_boxes(canvas, evaluations)

        # 2. Render Top Telemetry Banner
        self._render_top_bar(canvas, w, mode_name, fps, priority_hazard, language, iot_active, vlm_busy, speech_busy, is_muted)

        # 3. Render Spatial Mini-Radar in Bottom-Right
        self._render_mini_radar(canvas, w, h, evaluations)

        # 4. Render Bottom Subtitle Bar
        self._render_subtitles(canvas, w, h, caption)

        # 5. Render Quick Shortcut Key Guide
        self._render_footer_keys(canvas, w, h)

        return canvas

    def _render_bounding_boxes(self, canvas: np.ndarray, evaluations: List[Dict[str, Any]]):
        for ev in evaluations:
            x1, y1, x2, y2 = ev["bbox"]
            level = ev["risk_level"]
            obj_name = ev["object"].upper()
            conf = ev["confidence"]
            dist = ev["distance"]
            direction = ev["direction"]
            motion = ev.get("motion_state", "STATIONARY")

            color = self._get_level_color(level)

            # Draw main bounding box with corner highlights
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Corner accents
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 4:
                # Top-Left
                cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, 4)
                cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, 4)
                # Top-Right
                cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), color, 4)
                cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), color, 4)
                # Bottom-Left
                cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), color, 4)
                cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), color, 4)
                # Bottom-Right
                cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, 4)
                cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, 4)

            # Chip text
            motion_tag = " [!] APPROACHING" if motion == "APPROACHING" else ""
            label_text = f"{obj_name} ({conf:.2f}) | {direction} - {dist}{motion_tag}"

            # Chip background
            font_scale = 0.48
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            tag_y1 = max(0, y1 - th - 12)
            tag_y2 = y1

            # Draw label banner
            cv2.rectangle(canvas, (x1, tag_y1), (x1 + tw + 14, tag_y2), color, -1)
            cv2.putText(
                canvas,
                label_text,
                (x1 + 6, tag_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (10, 10, 10),
                thickness,
                cv2.LINE_AA
            )

    def _render_top_bar(
        self,
        canvas: np.ndarray,
        width: int,
        mode_name: str,
        fps: float,
        priority_hazard: Optional[Dict[str, Any]],
        language: str,
        iot_active: bool,
        vlm_busy: bool,
        speech_busy: bool,
        is_muted: bool
    ):
        bar_height = 55
        # Semi-transparent dark header banner
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_height), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, bar_height), (width, bar_height), (50, 50, 60), 1)

        # 1. Project Title & Active Mode Badge
        title = "VOICE TO VISION 2.0"
        cv2.putText(canvas, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_ACCENT, 2, cv2.LINE_AA)

        # Mode Badge
        mode_x = 215
        mode_str = f"MODE: {mode_name.upper()}"
        cv2.rectangle(canvas, (mode_x, 14), (mode_x + 180, 42), (45, 45, 55), -1)
        cv2.rectangle(canvas, (mode_x, 14), (mode_x + 180, 42), COLOR_ACCENT, 1)
        cv2.putText(canvas, mode_str, (mode_x + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_WHITE, 1, cv2.LINE_AA)

        # 2. System Status Badges
        # Language Badge
        lang_x = mode_x + 195
        lang_str = f"LANG: {language}"
        cv2.rectangle(canvas, (lang_x, 14), (lang_x + 115, 42), (45, 45, 55), -1)
        cv2.rectangle(canvas, (lang_x, 14), (lang_x + 115, 42), (100, 200, 100), 1)
        cv2.putText(canvas, lang_str, (lang_x + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 255, 150), 1, cv2.LINE_AA)

        # Voice-only alert status badge
        iot_x = lang_x + 130
        iot_str = "ALERT: VOICE"
        iot_color = (100, 220, 100)
        cv2.rectangle(canvas, (iot_x, 14), (iot_x + 135, 42), (35, 35, 45), -1)
        cv2.putText(canvas, iot_str, (iot_x + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, iot_color, 1, cv2.LINE_AA)

        # System Activity Badges
        status_x = iot_x + 150
        if speech_busy:
            cv2.putText(canvas, "[LISTENING...]", (status_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_CAUTION, 1, cv2.LINE_AA)
        elif vlm_busy:
            cv2.putText(canvas, "[VLM REASONING...]", (status_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 180, 50), 1, cv2.LINE_AA)
        else:
            status_color = COLOR_GRAY if is_muted else COLOR_SAFE
            status_text = "[MUTED]" if is_muted else "[ACTIVE MONITORING]"
            cv2.putText(canvas, status_text, (status_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1, cv2.LINE_AA)

        # 3. Overall Safety / Hazard Status on Right
        # Hazard Status on Far Right
        if priority_hazard:
            hz_level = priority_hazard["risk_level"]
            hz_score = priority_hazard["smoothed_risk"]
            hz_color = self._get_level_color(hz_level)

            hz_text = f"{hz_level} ({hz_score:.0f}%)"
            (htw, _), _ = cv2.getTextSize(hz_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            hz_start_x = width - htw - 105

            cv2.rectangle(canvas, (hz_start_x - 8, 14), (width - 95, 42), hz_color, 1)
            cv2.putText(canvas, hz_text, (hz_start_x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.50, hz_color, 2, cv2.LINE_AA)
        else:
            safe_text = "SAFE"
            (stw, _), _ = cv2.getTextSize(safe_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            cv2.putText(canvas, safe_text, (width - stw - 105, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_SAFE, 2, cv2.LINE_AA)

        # FPS counter
        fps_text = f"{fps:.1f} FPS"
        cv2.putText(canvas, fps_text, (width - 80, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_GRAY, 1, cv2.LINE_AA)

    def _render_mini_radar(self, canvas: np.ndarray, width: int, height: int, evaluations: List[Dict[str, Any]]):
        """Renders an assistive top-down bird's-eye spatial radar in bottom-right."""
        cx = width - self.radar_radius - self.radar_margin
        cy = height - self.radar_margin - 40
        radius = self.radar_radius

        # Dark radar circular backdrop
        overlay = canvas.copy()
        cv2.circle(overlay, (cx, cy), radius, (15, 18, 22), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.circle(canvas, (cx, cy), radius, (60, 65, 75), 1)

        # Semicircle distance range arcs
        cv2.ellipse(canvas, (cx, cy), (int(radius * 0.35), int(radius * 0.35)), 0, 180, 360, (50, 55, 65), 1)
        cv2.ellipse(canvas, (cx, cy), (int(radius * 0.70), int(radius * 0.70)), 0, 180, 360, (50, 55, 65), 1)

        # Field-of-View dividing rays (left 30 deg, right 30 deg)
        ang_left = math.radians(225)
        ang_right = math.radians(315)
        cv2.line(canvas, (cx, cy), (int(cx + radius * math.cos(ang_left)), int(cy + radius * math.sin(ang_left))), (50, 55, 65), 1)
        cv2.line(canvas, (cx, cy), (int(cx + radius * math.cos(ang_right)), int(cy + radius * math.sin(ang_right))), (50, 55, 65), 1)

        # User center indicator
        cv2.circle(canvas, (cx, cy), 5, COLOR_ACCENT, -1)
        cv2.putText(canvas, "RADAR", (cx - 20, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_GRAY, 1, cv2.LINE_AA)

        # Plot obstacle blips on radar
        for ev in evaluations:
            direction = ev["direction"]
            distance = ev["distance"]
            level = ev["risk_level"]
            color = self._get_level_color(level)

            # Distance radial factor
            r_factor = 0.35 if distance == "NEAR" else (0.65 if distance == "MEDIUM" else 0.90)
            target_r = radius * r_factor

            # Angular position
            if direction == "LEFT":
                angle_deg = 215
            elif direction == "RIGHT":
                angle_deg = 325
            else:  # CENTER
                angle_deg = 270

            rad = math.radians(angle_deg)
            blip_x = int(cx + target_r * math.cos(rad))
            blip_y = int(cy + target_r * math.sin(rad))

            # Draw pulsing blip
            cv2.circle(canvas, (blip_x, blip_y), 5, color, -1)
            cv2.circle(canvas, (blip_x, blip_y), 8, color, 1)

    def _render_subtitles(self, canvas: np.ndarray, width: int, height: int, caption: str):
        if not caption:
            return

        box_height = 36
        box_y = height - box_height - 35
        display_text = f"VOICE: \"{caption}\""

        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        box_w = min(width - 40, tw + 40)
        box_x = (width - box_w) // 2

        overlay = canvas.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_height), (12, 14, 18), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_height), COLOR_ACCENT, 1)

        cv2.putText(
            canvas,
            display_text,
            (box_x + 18, box_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            COLOR_WHITE,
            1,
            cv2.LINE_AA
        )

    def _render_footer_keys(self, canvas: np.ndarray, width: int, height: int):
        footer_y = height - 10
        shortcuts = "[TAB] Switch Mode   |   [SPACE] Voice Query / Read Text   |   [M] Mute Audio   |   [Q] Exit"
        shortcuts = "[TAB] Mode   |   [L] Switch Lang (Hindi/Eng)   |   [SPACE] Voice / Scan   |   [M] Mute   |   [Q] Quit"
        cv2.putText(canvas, shortcuts, (20, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 175, 185), 1, cv2.LINE_AA)

    def _get_level_color(self, level: str):
        if level == "CRITICAL":
            return COLOR_CRITICAL
        elif level == "HIGH":
            return COLOR_WARNING
        elif level == "MEDIUM":
            return COLOR_CAUTION
        else:
            return COLOR_SAFE
