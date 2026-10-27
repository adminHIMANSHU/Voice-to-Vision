import cv2
from ultralytics import YOLO
from safety_engine import calculate_risk


# -----------------------------
# Configuration
# -----------------------------

CONFIDENCE_THRESHOLD = 0.50
STABILITY_FRAMES = 3


# -----------------------------
# Load YOLO
# -----------------------------

model = YOLO("yolo11n.pt")


# -----------------------------
# Camera
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()


# -----------------------------
# Evidence Memory
# -----------------------------

evidence_memory = {}

# Last announced safety state
announced_state = {}


print("VoiceToVision Safety Integration started.")
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


    # -----------------------------
    # Process detections
    # -----------------------------

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            # Confidence filtering
            if confidence < CONFIDENCE_THRESHOLD:
                continue


            class_id = int(box.cls[0])
            object_name = model.names[class_id]


            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])


            # -----------------------------
            # Object Center
            # -----------------------------

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
            # Visual Distance
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


            # -----------------------------
            # Object Memory
            # -----------------------------

            object_key = object_name

            current_objects.add(object_key)


            if object_key not in evidence_memory:

                evidence_memory[object_key] = {
                    "count": 1,
                    "confidence": confidence,
                    "direction": direction,
                    "distance": distance,
                    "center": (center_x, center_y),
                    "bbox": (x1, y1, x2, y2)
                }

            else:

                evidence_memory[object_key]["count"] += 1

                evidence_memory[object_key]["confidence"] = confidence

                evidence_memory[object_key]["direction"] = direction

                evidence_memory[object_key]["distance"] = distance

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


            evidence = evidence_memory[object_key]


            # -----------------------------
            # Stable Evidence
            # -----------------------------

            if evidence["count"] >= STABILITY_FRAMES:

                stable_evidence = {

                    "object": object_name,

                    "confidence": round(
                        evidence["confidence"], 2
                    ),

                    "direction": evidence["direction"],

                    "visual_distance": evidence["distance"],

                    "center": evidence["center"],

                    "bounding_box": evidence["bbox"]
                }


                # -----------------------------
                # Safety Engine
                # -----------------------------

                safety_result = calculate_risk(
                    stable_evidence
                )


                # -----------------------------
                # State for Debouncing
                # -----------------------------

                state = (
                    safety_result["risk_level"],
                    safety_result["action"],
                    safety_result["direction"],
                    safety_result["distance"]
                )


                previous_state = announced_state.get(
                    object_key
                )


                # Only display meaningful changes
                if previous_state != state:

                    print()
                    print("=" * 60)

                    print("SAFETY DECISION")

                    print("=" * 60)

                    print(
                        f"Object       : "
                        f"{safety_result['object']}"
                    )

                    print(
                        f"Confidence   : "
                        f"{safety_result['confidence']}"
                    )

                    print(
                        f"Direction    : "
                        f"{safety_result['direction']}"
                    )

                    print(
                        f"Distance     : "
                        f"{safety_result['distance']}"
                    )

                    print(
                        f"Risk Score   : "
                        f"{safety_result['risk_score']}"
                    )

                    print(
                        f"Risk Level   : "
                        f"{safety_result['risk_level']}"
                    )

                    print(
                        f"Action       : "
                        f"{safety_result['action']}"
                    )

                    print("=" * 60)


                    announced_state[object_key] = state


    # -----------------------------
    # Remove disappeared objects
    # -----------------------------

    disappeared_objects = (
        set(evidence_memory.keys())
        - current_objects
    )


    for object_key in disappeared_objects:

        del evidence_memory[object_key]

        if object_key in announced_state:
            del announced_state[object_key]


    # -----------------------------
    # Display
    # -----------------------------

    annotated_frame = results[0].plot()

    cv2.imshow(
        "VoiceToVision - Safety Integration",
        annotated_frame
    )


    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()