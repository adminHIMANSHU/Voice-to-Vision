import cv2
from ultralytics import YOLO

from safety_engine import calculate_risk
from risk_stabilization import RiskStabilizer


# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()


# -----------------------------
# Configuration
# -----------------------------
CONFIDENCE_THRESHOLD = 0.50
STABILITY_FRAMES = 3

risk_stabilizer = RiskStabilizer(window_size=5)

evidence_count = {}
announced_state = {}


print("Stable Safety Integration started.")
print("Press Q to quit.")


# -----------------------------
# Main Loop
# -----------------------------
while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    results = model(frame, verbose=False)

    height, width = frame.shape[:2]

    current_objects = set()

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # -----------------------------
            # Direction
            # -----------------------------
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

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

            # Object tracking key
            object_key = object_name

            current_objects.add(object_key)

            # -----------------------------
            # Temporal Stability
            # -----------------------------
            evidence_count[object_key] = (
                evidence_count.get(object_key, 0) + 1
            )

            if evidence_count[object_key] < STABILITY_FRAMES:
                continue

            evidence = {
                "object": object_name,
                "confidence": round(confidence, 2),
                "direction": direction,
                "visual_distance": distance,
                "center": (center_x, center_y),
                "bounding_box": (x1, y1, x2, y2)
            }

            # -----------------------------
            # Safety Engine
            # -----------------------------
            risk_result = calculate_risk(evidence)

            raw_risk_score = risk_result["risk_score"]

            # -----------------------------
            # Risk Stabilization
            # -----------------------------
            stable_risk = risk_stabilizer.update(
                raw_risk_score
            )

            average_score = stable_risk["average_risk_score"]
            risk_level = stable_risk["risk_level"]

            # -----------------------------
            # Final Action
            # -----------------------------
            if risk_level == "CRITICAL":

                action = "STOP"

            elif risk_level == "HIGH":

                if direction == "CENTER":
                    action = "STOP"

                elif direction == "LEFT":
                    action = "MOVE RIGHT"

                elif direction == "RIGHT":
                    action = "MOVE LEFT"

                else:
                    action = "CAUTION"

            elif risk_level == "MEDIUM":

                action = "CAUTION"

            else:

                action = "SAFE"

            # -----------------------------
            # State Change Detection
            # -----------------------------
            state = (
                risk_level,
                action,
                direction,
                distance
            )

            if announced_state.get(object_key) != state:

                print("\nSTABLE SAFETY DECISION")
                print("-" * 50)
                print(f"Object: {object_name}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Direction: {direction}")
                print(f"Distance: {distance}")
                print(f"Raw Risk Score: {raw_risk_score}")
                print(f"Stable Risk Score: {average_score}")
                print(f"Risk Level: {risk_level}")
                print(f"Action: {action}")

                announced_state[object_key] = state


    # -----------------------------
    # Remove disappeared objects
    # -----------------------------
    disappeared = set(evidence_count.keys()) - current_objects

    for object_key in disappeared:

        evidence_count.pop(object_key, None)
        announced_state.pop(object_key, None)


    # -----------------------------
    # Display camera
    # -----------------------------
    annotated_frame = results[0].plot()

    cv2.imshow(
        "VoiceToVision - Stable Safety",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()