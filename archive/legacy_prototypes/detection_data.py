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

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            object_name = model.names[class_id]

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            print(
                f"Object: {object_name} | "
                f"Confidence: {confidence:.2f} | "
                f"Center: ({center_x}, {center_y})"
            )

    annotated_frame = results[0].plot()

    cv2.imshow("VoiceToVision - Detection Data", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()