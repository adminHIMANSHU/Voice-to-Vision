import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

print("Evidence Fusion started.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    results = model(frame, verbose=False)

    height, width = frame.shape[:2]

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            object_name = model.names[class_id]

            # Bounding-box center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Direction
            if center_x < width / 3:
                direction = "LEFT"
            elif center_x < 2 * width / 3:
                direction = "CENTER"
            else:
                direction = "RIGHT"

            # Approximate visual distance
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = width * height
            area_ratio = box_area / frame_area

            if area_ratio > 0.20:
                distance = "NEAR"
            elif area_ratio > 0.07:
                distance = "MEDIUM"
            else:
                distance = "FAR"

            # Structured evidence
            evidence = {
                "object": object_name,
                "confidence": round(confidence, 2),
                "direction": direction,
                "visual_distance": distance,
                "center": (center_x, center_y),
                "bounding_box": (x1, y1, x2, y2)
            }

            print(evidence)

    annotated_frame = results[0].plot()

    cv2.imshow(
        "VoiceToVision - Evidence Fusion",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()