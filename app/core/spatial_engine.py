from collections import defaultdict, deque
from typing import Dict, Tuple, Any

from app.core.config import (
    DIST_NEAR_RATIO,
    DIST_MEDIUM_RATIO,
    ZONE_LEFT_SPLIT,
    ZONE_RIGHT_SPLIT,
    APPROACH_THRESHOLD_RATIO,
)


class SpatialEngine:
    """
    Spatial reasoning engine that analyzes 2D bounding boxes and tracking IDs
    to extract 3D-approximate positional metrics:
    - 3-zone horizontal orientation (LEFT, CENTER, RIGHT)
    - Proximity categorization (NEAR, MEDIUM, FAR)
    - Relative motion dynamics (APPROACHING, RECEDING, STATIONARY)
    """

    def __init__(self, history_len: int = 6):
        # Maps track_id -> deque of area_ratios over recent frames
        self.area_history: Dict[Any, deque] = defaultdict(lambda: deque(maxlen=history_len))

    def evaluate_spatial_attributes(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        track_id: Any = None
    ) -> Dict[str, Any]:
        """
        Extracts direction, distance category, area ratio, and dynamic approach vector.
        """
        x1, y1, x2, y2 = bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        box_area = box_width * box_height
        frame_area = max(1, frame_width * frame_height)

        area_ratio = round(box_area / frame_area, 4)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 1. Direction Partitioning (3 zones)
        norm_center_x = center_x / frame_width
        if norm_center_x < ZONE_LEFT_SPLIT:
            direction = "LEFT"
        elif norm_center_x < ZONE_RIGHT_SPLIT:
            direction = "CENTER"
        else:
            direction = "RIGHT"

        # 2. Distance Categorization
        if area_ratio >= DIST_NEAR_RATIO:
            distance = "NEAR"
        elif area_ratio >= DIST_MEDIUM_RATIO:
            distance = "MEDIUM"
        else:
            distance = "FAR"

        # 3. Dynamic Velocity / Approach Vector
        motion_state = "STATIONARY"
        approach_rate = 0.0

        if track_id is not None:
            history = self.area_history[track_id]
            history.append(area_ratio)

            if len(history) >= 3:
                # delta between newest and oldest in history window
                delta_area = history[-1] - history[0]
                approach_rate = round(delta_area / len(history), 4)

                if approach_rate > APPROACH_THRESHOLD_RATIO:
                    motion_state = "APPROACHING"
                elif approach_rate < -APPROACH_THRESHOLD_RATIO:
                    motion_state = "RECEDING"

        return {
            "center": (center_x, center_y),
            "norm_center": (norm_center_x, center_y / frame_height),
            "direction": direction,
            "distance": distance,
            "area_ratio": area_ratio,
            "motion_state": motion_state,
            "approach_rate": approach_rate,
            "bbox": bbox
        }

    def cleanup_tracks(self, active_track_ids: set):
        """Removes stale tracks that are no longer visible to free memory."""
        all_ids = list(self.area_history.keys())
        for tid in all_ids:
            if tid not in active_track_ids:
                del self.area_history[tid]
