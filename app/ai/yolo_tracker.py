from typing import List, Dict, Any, Tuple
import numpy as np
from ultralytics import YOLO

from app.core.config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD
from app.core.spatial_engine import SpatialEngine


class YoloTracker:
    """
    High-performance YOLOv11 tracker wrapper with spatial reasoning integration.
    """

    def __init__(self, model_path: str = YOLO_MODEL_PATH, conf_thresh: float = CONFIDENCE_THRESHOLD):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        print(f"[YoloTracker] Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.spatial_engine = SpatialEngine()
        print("[YoloTracker] YOLO model ready.")

    def track(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], set]:
        """
        Executes tracking on the frame and returns enriched detection records
        plus the set of visible track IDs.
        """
        height, width = frame.shape[:2]
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf_thresh
        )

        detections: List[Dict[str, Any]] = []
        active_track_ids = set()

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.conf_thresh:
                    continue

                class_id = int(box.cls[0])
                object_name = self.model.names[class_id]

                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = f"{object_name}_{int(box.xyxy[0][0])}"

                active_track_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Spatial attributes (distance, direction, approach velocity)
                spatial = self.spatial_engine.evaluate_spatial_attributes(
                    bbox=(x1, y1, x2, y2),
                    frame_width=width,
                    frame_height=height,
                    track_id=track_id
                )

                detection = {
                    "track_id": track_id,
                    "name": object_name,
                    "confidence": round(confidence, 2),
                    "bbox": (x1, y1, x2, y2),
                    "center": spatial["center"],
                    "direction": spatial["direction"],
                    "distance": spatial["distance"],
                    "area_ratio": spatial["area_ratio"],
                    "motion_state": spatial["motion_state"],
                    "approach_rate": spatial["approach_rate"],
                }
                detections.append(detection)

        # Prune dead tracks in spatial engine
        self.spatial_engine.cleanup_tracks(active_track_ids)

        return detections, active_track_ids
