import cv2
import pyttsx3
from collections import defaultdict, deque

from ultralytics import YOLO
from safety_engine import calculate_risk


# =========================================================
# CONFIGURATION
# =========================================================

CONFIDENCE_THRESHOLD = 0.50
STABILITY_FRAMES = 3

RISK_WINDOW_SIZE = 5

# Frames between repeated voice reminders
VOICE_COOLDOWN_FRAMES = 90

# Hysteresis
CRITICAL_ENTER = 82
CRITICAL_EXIT = 77

HIGH_ENTER = 60
HIGH_EXIT = 55

MEDIUM_ENTER = 35
MEDIUM_EXIT = 30


# =========================================================
# VOICE ENGINE
# =========================================================

voice_engine = pyttsx3.init()

voice_engine.setProperty("rate", 155)
voice_engine.setProperty("volume", 1.0)


def speak(message):

    print(f"VOICE: {message}")

    voice_engine.say(message)
    voice_engine.runAndWait()


# =========================================================
# LOAD YOLO
# =========================================================

model = YOLO("yolo11n.pt")


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Camera could not be opened.")
    raise SystemExit


# =========================================================
# STATE
# =========================================================

stability_count = defaultdict(int)

risk_history = defaultdict(
    lambda: deque(maxlen=RISK_WINDOW_SIZE)
)

risk_levels = {}

last_voice_state = {}

voice_cooldown = defaultdict(int)


# =========================================================
# DIRECTION
# =========================================================

def get_direction(center_x, width):

    if center_x < width / 3:

        return "LEFT"

    elif center_x < 2 * width / 3:

        return "CENTER"

    return "RIGHT"


# =========================================================
# VISUAL DISTANCE
# =========================================================

def get_visual_distance(
    x1,
    y1,
    x2,
    y2,
    width,
    height
):

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    box_area = box_width * box_height
    frame_area = width * height

    area_ratio = box_area / frame_area


    if area_ratio > 0.20:

        return "NEAR"

    elif area_ratio > 0.07:

        return "MEDIUM"

    return "FAR"


# =========================================================
# RISK SMOOTHING
# =========================================================

def get_stable_score(track_id, raw_score):

    history = risk_history[track_id]

    history.append(raw_score)

    average = sum(history) / len(history)

    return round(average, 2)


# =========================================================
# RISK HYSTERESIS
# =========================================================

def get_risk_level(track_id, score):

    previous = risk_levels.get(
        track_id,
        "LOW"
    )


    if previous == "CRITICAL":

        if score >= CRITICAL_EXIT:
            return "CRITICAL"

        elif score >= HIGH_ENTER:
            return "HIGH"

        elif score >= MEDIUM_ENTER:
            return "MEDIUM"

        return "LOW"


    if previous == "HIGH":

        if score >= CRITICAL_ENTER:
            return "CRITICAL"

        elif score >= HIGH_EXIT:
            return "HIGH"

        elif score >= MEDIUM_ENTER:
            return "MEDIUM"

        return "LOW"


    if previous == "MEDIUM":

        if score >= CRITICAL_ENTER:
            return "CRITICAL"

        elif score >= HIGH_ENTER:
            return "HIGH"

        elif score >= MEDIUM_EXIT:
            return "MEDIUM"

        return "LOW"


    # LOW

    if score >= CRITICAL_ENTER:
        return "CRITICAL"

    elif score >= HIGH_ENTER:
        return "HIGH"

    elif score >= MEDIUM_ENTER:
        return "MEDIUM"

    return "LOW"


# =========================================================
# SAFETY ACTION
# =========================================================

def get_action(risk_level, direction):

    if risk_level == "CRITICAL":

        return "STOP"


    if risk_level == "HIGH":

        if direction == "CENTER":
            return "STOP"

        elif direction == "LEFT":
            return "MOVE RIGHT"

        elif direction == "RIGHT":
            return "MOVE LEFT"

        return "CAUTION"


    if risk_level == "MEDIUM":

        return "CAUTION"


    return "SAFE"


# =========================================================
# VOICE MESSAGE
# =========================================================

def create_voice_message(
    object_name,
    direction,
    distance,
    action
):

    if action == "STOP":

        if direction == "CENTER":

            return f"Stop. {object_name} ahead."

        elif direction == "LEFT":

            return f"Stop. {object_name} on the left."

        elif direction == "RIGHT":

            return f"Stop. {object_name} on the right."


    if action == "MOVE RIGHT":

        return f"{object_name} on the left. Move right."


    if action == "MOVE LEFT":

        return f"{object_name} on the right. Move left."


    if action == "CAUTION":

        if direction == "CENTER":

            return f"Caution. {object_name} ahead."

        elif direction == "LEFT":

            return f"Caution. {object_name} on the left."

        elif direction == "RIGHT":

            return f"Caution. {object_name} on the right."


    return f"{object_name} detected."


# =========================================================
# VOICE DEBOUNCER
# =========================================================

def should_speak(
    track_id,
    action,
    direction,
    distance
):

    current_state = (
        action,
        direction,
        distance
    )


    # First alert
    if track_id not in last_voice_state:

        last_voice_state[
            track_id
        ] = current_state

        voice_cooldown[
            track_id
        ] = 0

        return True


    previous_state = last_voice_state[
        track_id
    ]


    # Meaningful change
    if current_state != previous_state:

        last_voice_state[
            track_id
        ] = current_state

        voice_cooldown[
            track_id
        ] = 0

        return True


    # Same state
    voice_cooldown[
        track_id
    ] += 1


    # Repeat only dangerous alerts
    if (
        action in (
            "STOP",
            "MOVE LEFT",
            "MOVE RIGHT"
        )
        and
        voice_cooldown[
            track_id
        ] >= VOICE_COOLDOWN_FRAMES
    ):

        voice_cooldown[
            track_id
        ] = 0

        return True


    return False


# =========================================================
# CLEANUP
# =========================================================

def cleanup_track(track_id):

    stability_count.pop(
        track_id,
        None
    )

    risk_history.pop(
        track_id,
        None
    )

    risk_levels.pop(
        track_id,
        None
    )

    last_voice_state.pop(
        track_id,
        None
    )

    voice_cooldown.pop(
        track_id,
        None
    )


# =========================================================
# START
# =========================================================

print()
print("=" * 65)
print("VoiceToVision - VOICE SAFETY PIPELINE")
print("=" * 65)

print("YOLO Detection            : ENABLED")
print("Object Tracking           : ENABLED")
print("Temporal Stability        : ENABLED")
print("Risk Smoothing            : ENABLED")
print("Risk Hysteresis           : ENABLED")
print("Voice Debouncing          : ENABLED")
print("Text-to-Speech            : ENABLED")

print()
print("Press Q to quit.")
print("=" * 65)


# =========================================================
# MAIN LOOP
# =========================================================

try:

    while True:

        success, frame = camera.read()


        if not success:

            print("Could not read camera frame.")
            break


        # YOLO tracking
        results = model.track(
            frame,
            persist=True,
            verbose=False
        )


        height, width = frame.shape[:2]

        current_tracks = set()


        # =================================================
        # DETECTIONS
        # =================================================

        for result in results:

            boxes = result.boxes


            if boxes is None:
                continue


            for box in boxes:

                confidence = float(
                    box.conf[0]
                )


                if confidence < CONFIDENCE_THRESHOLD:
                    continue


                class_id = int(
                    box.cls[0]
                )

                object_name = model.names[
                    class_id
                ]


                # -----------------------------------------
                # TRACK ID
                # -----------------------------------------

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                else:

                    track_id = (
                        f"{object_name}_fallback"
                    )


                current_tracks.add(
                    track_id
                )


                # -----------------------------------------
                # BOUNDING BOX
                # -----------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                # -----------------------------------------
                # TEMPORAL STABILITY
                # -----------------------------------------

                stability_count[
                    track_id
                ] += 1


                if (
                    stability_count[
                        track_id
                    ] < STABILITY_FRAMES
                ):

                    continue


                # -----------------------------------------
                # POSITION
                # -----------------------------------------

                center_x = (
                    x1 + x2
                ) // 2


                center_y = (
                    y1 + y2
                ) // 2


                direction = get_direction(
                    center_x,
                    width
                )


                # -----------------------------------------
                # DISTANCE
                # -----------------------------------------

                distance = get_visual_distance(
                    x1,
                    y1,
                    x2,
                    y2,
                    width,
                    height
                )


                # -----------------------------------------
                # EVIDENCE
                # -----------------------------------------

                evidence = {

                    "object": object_name,

                    "confidence": round(
                        confidence,
                        2
                    ),

                    "direction": direction,

                    "visual_distance": distance,

                    "center": (
                        center_x,
                        center_y
                    ),

                    "bounding_box": (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                }


                # -----------------------------------------
                # SAFETY ENGINE
                # -----------------------------------------

                risk_result = calculate_risk(
                    evidence
                )


                raw_score = risk_result[
                    "risk_score"
                ]


                # -----------------------------------------
                # STABLE SCORE
                # -----------------------------------------

                stable_score = get_stable_score(
                    track_id,
                    raw_score
                )


                # -----------------------------------------
                # HYSTERESIS
                # -----------------------------------------

                risk_level = get_risk_level(
                    track_id,
                    stable_score
                )


                risk_levels[
                    track_id
                ] = risk_level


                # -----------------------------------------
                # ACTION
                # -----------------------------------------

                action = get_action(
                    risk_level,
                    direction
                )


                # -----------------------------------------
                # VOICE DECISION
                # -----------------------------------------

                if should_speak(
                    track_id,
                    action,
                    direction,
                    distance
                ):

                    message = create_voice_message(
                        object_name,
                        direction,
                        distance,
                        action
                    )


                    print()
                    print("=" * 65)
                    print("VOICE SAFETY ALERT")
                    print("=" * 65)

                    print(
                        f"Track ID: {track_id}"
                    )

                    print(
                        f"Object: {object_name}"
                    )

                    print(
                        f"Direction: {direction}"
                    )

                    print(
                        f"Distance: {distance}"
                    )

                    print(
                        f"Risk Score: {stable_score}"
                    )

                    print(
                        f"Risk Level: {risk_level}"
                    )

                    print(
                        f"Action: {action}"
                    )

                    print(
                        f"Message: {message}"
                    )

                    print("=" * 65)


                    # Actual speech
                    speak(message)


        # =================================================
        # REMOVE DISAPPEARED TRACKS
        # =================================================

        known_tracks = set(
            stability_count.keys()
        )


        disappeared = (
            known_tracks
            - current_tracks
        )


        for track_id in disappeared:

            cleanup_track(
                track_id
            )


        # =================================================
        # DISPLAY
        # =================================================

        annotated_frame = results[0].plot()


        cv2.imshow(
            "VoiceToVision - Voice Safety",
            annotated_frame
        )


        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break


except KeyboardInterrupt:

    print()
    print("Voice safety pipeline interrupted by user.")


finally:

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("VoiceToVision Voice Safety Pipeline stopped.")