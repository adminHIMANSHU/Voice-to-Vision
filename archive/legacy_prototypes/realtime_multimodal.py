import cv2
import time
import torch

from collections import defaultdict, deque
from ultralytics import YOLO

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)

from safety_engine import calculate_risk
from multimodal_fusion import calculate_fusion


# ============================================================
# CONFIGURATION
# ============================================================

YOLO_MODEL = "yolo11n.pt"
VLM_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

CONFIDENCE_THRESHOLD = 0.50

STABILITY_FRAMES = 3
RISK_WINDOW_SIZE = 5

# VLM will run after this many seconds
VLM_INTERVAL = 5.0

# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 70)
print("VoiceToVision - REAL-TIME MULTIMODAL PIPELINE")
print("=" * 70)

print("\nLoading YOLO...")

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded successfully.")

# ============================================================
# LOAD VLM
# ============================================================

print("\nLoading SmolVLM...")

processor = AutoProcessor.from_pretrained(VLM_MODEL)

vlm = AutoModelForImageTextToText.from_pretrained(
    VLM_MODEL,
    dtype=torch.float32
)

vlm.eval()

print("SmolVLM loaded successfully.")

# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

# ============================================================
# TRACK MEMORY
# ============================================================

stability_counter = defaultdict(int)

risk_history = defaultdict(
    lambda: deque(maxlen=RISK_WINDOW_SIZE)
)

latest_evidence = {}

latest_vlm_context = ""

last_vlm_time = 0

# ============================================================
# VLM FUNCTION
# ============================================================

def analyze_scene(frame):

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    prompt = (
        "Describe this scene for a visually impaired person. "
        "Focus only on important objects, obstacles, people, "
        "doors, stairs, vehicles and anything that may affect "
        "safe movement. Keep the description short and practical."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        generated_ids = vlm.generate(
            **inputs,
            max_new_tokens=100
        )

    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    # --------------------------------------------------------
    # Clean VLM output
    # --------------------------------------------------------

    if "Assistant:" in result:

        result = result.split(
            "Assistant:",
            1
        )[1].strip()

    return result

# ============================================================
# MAIN LOOP
# ============================================================

print("\nCamera started.")
print("Real-time multimodal processing started.")
print("Press Q to quit.")
print("=" * 70)

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame could not be read.")
            continue


        height, width = frame.shape[:2]

        # ====================================================
        # YOLO TRACKING
        # ====================================================

        results = yolo.track(
            frame,
            persist=True,
            verbose=False
        )

        current_tracks = set()

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                if confidence < CONFIDENCE_THRESHOLD:
                    continue


                class_id = int(
                    box.cls[0]
                )

                object_name = yolo.names[class_id]

                # --------------------------------------------
                # TRACK ID
                # --------------------------------------------

                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = f"{object_name}_unknown"


                current_tracks.add(track_id)

                # --------------------------------------------
                # BOUNDING BOX
                # --------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                center_x = (
                    x1 + x2
                ) // 2

                # --------------------------------------------
                # DIRECTION
                # --------------------------------------------

                if center_x < width / 3:

                    direction = "LEFT"

                elif center_x < 2 * width / 3:

                    direction = "CENTER"

                else:

                    direction = "RIGHT"

                # --------------------------------------------
                # APPROXIMATE DISTANCE
                # --------------------------------------------

                box_area = (
                    (x2 - x1)
                    *
                    (y2 - y1)
                )

                frame_area = (
                    width * height
                )

                area_ratio = (
                    box_area / frame_area
                )

                if area_ratio > 0.20:

                    distance = "NEAR"

                elif area_ratio > 0.07:

                    distance = "MEDIUM"

                else:

                    distance = "FAR"

                # --------------------------------------------
                # TEMPORAL STABILITY
                # --------------------------------------------

                stability_counter[track_id] += 1

                if (
                    stability_counter[track_id]
                    < STABILITY_FRAMES
                ):
                    continue

                # --------------------------------------------
                # YOLO EVIDENCE
                # --------------------------------------------

                evidence = {

                    "object": object_name,

                    "confidence": round(
                        confidence,
                        2
                    ),

                    "direction": direction,

                    "visual_distance": distance

                }

                latest_evidence[
                    track_id
                ] = evidence

                # --------------------------------------------
                # SAFETY ENGINE
                # --------------------------------------------

                safety = calculate_risk(
                    evidence
                )

                # --------------------------------------------
                # RISK SMOOTHING
                # --------------------------------------------

                risk_history[
                    track_id
                ].append(
                    safety["risk_score"]
                )

                stable_risk = round(
                    sum(
                        risk_history[
                            track_id
                        ]
                    )
                    /
                    len(
                        risk_history[
                            track_id
                        ]
                    ),
                    2
                )

                # --------------------------------------------
                # DISPLAY
                # --------------------------------------------

                label = (
                    f"{object_name} "
                    f"{confidence:.2f} "
                    f"{direction} "
                    f"{distance}"
                )

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
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                # --------------------------------------------
                # CONSOLE SAFETY DATA
                # --------------------------------------------

                print(
                    f"\nYOLO SAFETY"
                )

                print(
                    f"Track      : {track_id}"
                )

                print(
                    f"Object     : {object_name}"
                )

                print(
                    f"Direction  : {direction}"
                )

                print(
                    f"Distance   : {distance}"
                )

                print(
                    f"Risk       : {stable_risk}"
                )

        # ====================================================
        # VLM PROCESSING
        # ====================================================

        current_time = time.time()

        if (
            current_time - last_vlm_time
            >= VLM_INTERVAL
        ):

            print(
                "\nAnalyzing scene with SmolVLM..."
            )

            try:

                latest_vlm_context = (
                    analyze_scene(frame)
                )

                last_vlm_time = current_time

                print(
                    "\nVLM CONTEXT"
                )

                print("-" * 70)

                print(
                    latest_vlm_context
                )

                print("-" * 70)


            except Exception as error:

                print(
                    f"VLM error: {error}"
                )

        # ====================================================
        # MULTIMODAL FUSION
        # ====================================================
        if latest_vlm_context:

            for track_id, evidence in latest_evidence.items():

                fusion = calculate_fusion(
                    evidence,
                    latest_vlm_context
                )


                print(
                    "\nMULTIMODAL FUSION"
                )

                print("-" * 70)

                print(
                    f"Track ID          : {track_id}"
                )

                print(
                    f"Object            : "
                    f"{fusion['object']}"
                )

                print(
                    f"YOLO Confidence   : "
                    f"{fusion['yolo_confidence']}"
                )

                print(
                    f"VLM Quality       : "
                    f"{fusion['vlm_quality']}"
                )

                print(
                    f"Object Agreement  : "
                    f"{fusion['object_agreement']}"
                )

                print(
                    f"Agreement Score   : "
                    f"{fusion['agreement_score']}"
                )

                print(
                    f"Final Confidence  : "
                    f"{fusion['final_confidence']}"
                )

                print(
                    f"Confidence Level  : "
                    f"{fusion['confidence_level']}"
                )

                print(
                    f"FINAL ACTION      : "
                    f"{fusion['action']}"
                )

                print("-" * 70)

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "VoiceToVision - Real-Time Multimodal",
            frame
        )

        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:

    print(
        "\nPipeline interrupted by user."
    )

finally:

    camera.release()

    cv2.destroyAllWindows()

    print(
        "\nVoiceToVision Real-Time Multimodal Pipeline stopped."
    )