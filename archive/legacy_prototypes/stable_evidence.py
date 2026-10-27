import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

# -----------------------------
# Configuration
# -----------------------------

CONFIDENCE_THRESHOLD = 0.50
STABILITY_FRAMES = 3

# Stores consecutive observations
evidence_memory = {}

print("Stable Evidence System started.")
print("Press Q to quit.")

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    results = model(frame, verbose=False)

    height, width = frame.shape[:2]

    # Track objects detected in current frame
    current_objects = set()

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            # -----------------------------
            # Confidence Filtering
            # -----------------------------

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounding box center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # -----------------------------
            # Direction
            # -----------------------------

            if center_x < width / 3:
                direction = "LEFT"

            elif center_x < 2 * width / 3:
                direction = "CENTER"

            else:
                direction = "RIGHT"

            # -----------------------------
            # Approximate visual distance
            # -----------------------------

            box_area = (x2 - x1) * (y2 - y1)
            frame_area = width * height

            area_ratio = box_area / frame_area

            if area_ratio > 0.20:
                distance = "NEAR"

            elif area_ratio > 0.07:
                distance = "MEDIUM"

            else:
                distance = "FAR"

            # Object identity for temporal stability
            object_key = (object_name, direction)

            current_objects.add(object_key)

            # -----------------------------
            # Temporal Stability
            # -----------------------------

            if object_key not in evidence_memory:

                evidence_memory[object_key] = {
                    "count": 1,
                    "last_distance": distance,
                    "confidence": confidence,
                    "center": (center_x, center_y),
                    "bbox": (x1, y1, x2, y2)
                }

            else:

                evidence_memory[object_key]["count"] += 1
                evidence_memory[object_key]["last_distance"] = distance
                evidence_memory[object_key]["confidence"] = confidence
                evidence_memory[object_key]["center"] = (
                    center_x,
                    center_y
                )
                evidence_memory[object_key]["bbox"] = (
                    x1,
                    y1,
                    x2,
                    y2
                )

            # -----------------------------
            # Stable Evidence
            # -----------------------------

            evidence = evidence_memory[object_key]

            if evidence["count"] >= STABILITY_FRAMES:

                stable_evidence = {
                    "object": object_name,
                    "confidence": round(evidence["confidence"], 2),
                    "direction": direction,
                    "visual_distance": evidence["last_distance"],
                    "center": evidence["center"],
                    "bounding_box": evidence["bbox"],
                    "stable_frames": evidence["count"]
                }

                print("STABLE EVIDENCE:")
                print(stable_evidence)
                print("-" * 60)

    # -----------------------------
    # Remove objects that disappeared
    # -----------------------------

    disappeared_objects = set(evidence_memory.keys()) - current_objects

    for object_key in disappeared_objects:
        del evidence_memory[object_key]

    # Display detections
    annotated_frame = results[0].plot()

    cv2.imshow(
        "VoiceToVision - Stable Evidence",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()