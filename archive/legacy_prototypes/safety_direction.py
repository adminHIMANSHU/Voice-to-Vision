import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

obstacle_classes = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "chair",
    "bench",
    "backpack",
    "suitcase"
}

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    frame_height, frame_width = frame.shape[:2]

    left_blocked = False
    center_blocked = False
    right_blocked = False

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence < 0.50:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            object_name = model.names[class_id]

            if object_name not in obstacle_classes:
                continue

            center_x = (x1 + x2) // 2

            object_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_width * frame_height
            area_ratio = object_area / frame_area

            if area_ratio > 0.20:
                distance = "NEAR"
            elif area_ratio > 0.07:
                distance = "MEDIUM"
            else:
                distance = "FAR"

            if center_x < frame_width / 3:
                direction = "LEFT"

                if distance in {"NEAR", "MEDIUM"}:
                    left_blocked = True

            elif center_x < (frame_width * 2) / 3:
                direction = "CENTER"

                if distance in {"NEAR", "MEDIUM"}:
                    center_blocked = True

            else:
                direction = "RIGHT"

                if distance in {"NEAR", "MEDIUM"}:
                    right_blocked = True

            label = f"{object_name} {confidence:.2f} {direction} {distance}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    if center_blocked:
        if not left_blocked:
            guidance = "CENTER BLOCKED - MOVE LEFT"
        elif not right_blocked:
            guidance = "CENTER BLOCKED - MOVE RIGHT"
        else:
            guidance = "ALL DIRECTIONS BLOCKED - STOP"

    elif left_blocked and right_blocked:
        guidance = "LEFT AND RIGHT BLOCKED - MOVE STRAIGHT"

    elif left_blocked:
        guidance = "LEFT BLOCKED - MOVE RIGHT"

    elif right_blocked:
        guidance = "RIGHT BLOCKED - MOVE LEFT"

    else:
        guidance = "PATH CLEAR - MOVE FORWARD"

    cv2.putText(
        frame,
        guidance,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    print(guidance)

    cv2.imshow("VoiceToVision - Safety Guidance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()