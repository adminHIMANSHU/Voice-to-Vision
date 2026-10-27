import cv2
from collections import defaultdict, deque
from ultralytics import YOLO

from safety_engine import calculate_risk


# =========================================================
# CONFIGURATION
# =========================================================

CONFIDENCE_THRESHOLD = 0.50

# Object must be visible for these many frames
# before safety processing starts.
STABILITY_FRAMES = 3

# Number of risk scores used for smoothing.
RISK_WINDOW_SIZE = 5

# Same safety alert can be repeated after this many frames.
ALERT_COOLDOWN_FRAMES = 60

# Hysteresis thresholds.
#
# Enter CRITICAL only at >= 82.
# Once CRITICAL, remain CRITICAL until < 77.
CRITICAL_ENTER = 82
CRITICAL_EXIT = 77

# Enter HIGH at >= 60.
# Leave HIGH only when < 55.
HIGH_ENTER = 60
HIGH_EXIT = 55

# Enter MEDIUM at >= 35.
# Leave MEDIUM only when < 30.
MEDIUM_ENTER = 35
MEDIUM_EXIT = 30


# =========================================================
# LOAD YOLO MODEL
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
# PER-TRACK STATE
# =========================================================

# Number of consecutive frames for each tracked object.
stability_count = defaultdict(int)


# Individual risk history for every tracked object.
risk_history = defaultdict(
    lambda: deque(maxlen=RISK_WINDOW_SIZE)
)


# Current hysteresis risk level for each object.
risk_levels = {}


# Last safety state announced for each object.
#
# Format:
# (action, direction, distance)
#
last_alert_state = {}


# Number of frames since last alert.
alert_cooldown = defaultdict(int)


# Track IDs currently visible.
current_tracks = set()


# =========================================================
# HELPER: DIRECTION
# =========================================================

def get_direction(center_x, width):

    if center_x < width / 3:

        return "LEFT"

    elif center_x < 2 * width / 3:

        return "CENTER"

    return "RIGHT"


# =========================================================
# HELPER: VISUAL DISTANCE
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
# HELPER: PER-TRACK RISK SMOOTHING
# =========================================================

def get_average_risk(track_id, raw_risk_score):

    history = risk_history[track_id]

    history.append(raw_risk_score)

    average_score = sum(history) / len(history)

    return round(average_score, 2)


# =========================================================
# HELPER: RISK HYSTERESIS
# =========================================================

def get_hysteresis_risk(track_id, score):

    previous_level = risk_levels.get(
        track_id,
        "LOW"
    )


    # -----------------------------------------------------
    # CRITICAL STATE
    # -----------------------------------------------------

    if previous_level == "CRITICAL":

        if score >= CRITICAL_EXIT:

            return "CRITICAL"

        elif score >= HIGH_ENTER:

            return "HIGH"

        elif score >= MEDIUM_ENTER:

            return "MEDIUM"

        else:

            return "LOW"


    # -----------------------------------------------------
    # HIGH STATE
    # -----------------------------------------------------

    if previous_level == "HIGH":

        if score >= CRITICAL_ENTER:

            return "CRITICAL"

        elif score >= HIGH_EXIT:

            return "HIGH"

        elif score >= MEDIUM_ENTER:

            return "MEDIUM"

        else:

            return "LOW"


    # -----------------------------------------------------
    # MEDIUM STATE
    # -----------------------------------------------------

    if previous_level == "MEDIUM":

        if score >= CRITICAL_ENTER:

            return "CRITICAL"

        elif score >= HIGH_ENTER:

            return "HIGH"

        elif score >= MEDIUM_EXIT:

            return "MEDIUM"

        else:

            return "LOW"


    # -----------------------------------------------------
    # LOW STATE
    # -----------------------------------------------------

    if score >= CRITICAL_ENTER:

        return "CRITICAL"

    elif score >= HIGH_ENTER:

        return "HIGH"

    elif score >= MEDIUM_ENTER:

        return "MEDIUM"

    return "LOW"


# =========================================================
# HELPER: SAFETY ACTION
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
# HELPER: ALERT DEBOUNCING
# =========================================================

def should_alert(
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


    # -----------------------------------------------------
    # First alert for this tracked object
    # -----------------------------------------------------

    if track_id not in last_alert_state:

        last_alert_state[track_id] = current_state

        alert_cooldown[track_id] = 0

        return True


    previous_state = last_alert_state[track_id]


    # -----------------------------------------------------
    # Meaningful safety change
    # -----------------------------------------------------

    if current_state != previous_state:

        last_alert_state[track_id] = current_state

        alert_cooldown[track_id] = 0

        return True


    # -----------------------------------------------------
    # Same state
    # -----------------------------------------------------

    alert_cooldown[track_id] += 1


    # Periodic reminder for persistent danger.
    #
    # This is important for an assistive system:
    # a continuously dangerous object should eventually
    # remind the user again.
    #

    if (
        action in ("STOP", "MOVE LEFT", "MOVE RIGHT")
        and
        alert_cooldown[track_id]
        >= ALERT_COOLDOWN_FRAMES
    ):

        alert_cooldown[track_id] = 0

        return True


    return False


# =========================================================
# HELPER: CLEAN DISAPPEARED TRACKS
# =========================================================

def cleanup_tracks(current_tracks):

    known_tracks = set(stability_count.keys())

    disappeared_tracks = (
        known_tracks - current_tracks
    )


    for track_id in disappeared_tracks:

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

        last_alert_state.pop(
            track_id,
            None
        )

        alert_cooldown.pop(
            track_id,
            None


        )


# =========================================================
# START SYSTEM
# =========================================================

print()
print("=" * 65)
print("VoiceToVision - FINAL SAFETY PIPELINE")
print("=" * 65)

print("YOLO Detection             : ENABLED")
print("Object Tracking            : ENABLED")
print("Temporal Stability         : ENABLED")
print("Per-Object Risk Smoothing  : ENABLED")
print("Risk Hysteresis            : ENABLED")
print("Alert Debouncing           : ENABLED")

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


        # =================================================
        # YOLO TRACKING
        # =================================================
        #
        # persist=True maintains object identities
        # across frames.
        #

        results = model.track(
            frame,
            persist=True,
            verbose=False
        )


        height, width = frame.shape[:2]

        current_tracks = set()


        # =================================================
        # PROCESS RESULTS
        # =================================================

        for result in results:

            boxes = result.boxes


            if boxes is None:

                continue


            for box in boxes:

                confidence = float(
                    box.conf[0]
                )


                # -----------------------------------------
                # CONFIDENCE FILTER
                # -----------------------------------------

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

                    # Fallback if tracker temporarily
                    # has no ID.
                    #
                    # This should be rare.
                    #

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
                    stability_count[track_id]
                    < STABILITY_FRAMES
                ):

                    continue


                # -----------------------------------------
                # CENTER
                # -----------------------------------------

                center_x = (
                    x1 + x2
                ) // 2

                center_y = (
                    y1 + y2
                ) // 2


                # -----------------------------------------
                # DIRECTION
                # -----------------------------------------

                direction = get_direction(
                    center_x,
                    width
                )


                # -----------------------------------------
                # VISUAL DISTANCE
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


                raw_risk_score = risk_result[
                    "risk_score"
                ]


                # -----------------------------------------
                # PER-OBJECT SMOOTHING
                # -----------------------------------------

                stable_score = get_average_risk(
                    track_id,
                    raw_risk_score
                )


                # -----------------------------------------
                # HYSTERESIS
                # -----------------------------------------

                risk_level = get_hysteresis_risk(
                    track_id,
                    stable_score
                )


                risk_levels[
                    track_id
                ] = risk_level


                # -----------------------------------------
                # SAFETY ACTION
                # -----------------------------------------

                action = get_action(
                    risk_level,
                    direction
                )


                # -----------------------------------------
                # ALERT DEBOUNCER
                # -----------------------------------------

                speak_alert = should_alert(
                    track_id,
                    action,
                    direction,
                    distance
                )


                # -----------------------------------------
                # FINAL OUTPUT
                # -----------------------------------------

                if speak_alert:

                    print()
                    print("=" * 65)
                    print("FINAL SAFETY ALERT")
                    print("=" * 65)

                    print(
                        f"Track ID: {track_id}"
                    )

                    print(
                        f"Object: {object_name}"
                    )

                    print(
                        f"Confidence: {confidence:.2f}"
                    )

                    print(
                        f"Direction: {direction}"
                    )

                    print(
                        f"Distance: {distance}"
                    )

                    print(
                        f"Raw Risk Score: "
                        f"{raw_risk_score}"
                    )

                    print(
                        f"Stable Risk Score: "
                        f"{stable_score}"
                    )

                    print(
                        f"Risk Level: "
                        f"{risk_level}"
                    )

                    print(
                        f"Action: {action}"
                    )

                    print("=" * 65)


        # =================================================
        # CLEANUP
        # =================================================

        cleanup_tracks(
            current_tracks
        )


        # =================================================
        # DISPLAY
        # =================================================

        annotated_frame = results[0].plot()


        cv2.imshow(
            "VoiceToVision - Final Safety Pipeline",
            annotated_frame
        )


        # =================================================
        # QUIT
        # =================================================

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break


except KeyboardInterrupt:

    print()
    print("Pipeline interrupted by user.")


finally:

    camera.release()

    cv2.destroyAllWindows()

    print()
    print("VoiceToVision Final Safety Pipeline stopped.")