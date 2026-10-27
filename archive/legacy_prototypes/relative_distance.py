import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    frame_height, frame_width = frame.shape[:2]

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            object_name = model.names[class_id]

            object_width = x2 - x1
            object_height = y2 - y1
            object_area = object_width * object_height
            frame_area = frame_width * frame_height

            area_ratio = object_area / frame_area

            if area_ratio > 0.20:
                distance = "NEAR"
            elif area_ratio > 0.07:
                distance = "MEDIUM"
            else:
                distance = "FAR"

            center_x = (x1 + x2) // 2

            if center_x < frame_width / 3:
                direction = "LEFT"
            elif center_x < (frame_width * 2) / 3:
                direction = "CENTER"
            else:
                direction = "RIGHT"

            print(
                f"Object: {object_name} | "
                f"Confidence: {confidence:.2f} | "
                f"Direction: {direction} | "
                f"Distance: {distance}"
            )

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

    cv2.imshow("VoiceToVision - Spatial Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()