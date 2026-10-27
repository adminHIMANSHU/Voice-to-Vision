from typing import Dict


def calculate_fusion(yolo_evidence: Dict, vlm_context: str) -> Dict:
    """
    Combines structured YOLO safety evidence
    with VLM scene context.
    """

    object_name = yolo_evidence["object"]
    direction = yolo_evidence["direction"]
    distance = yolo_evidence["visual_distance"]
    yolo_confidence = yolo_evidence["confidence"]

    context = vlm_context.lower().strip()

    # --------------------------------------------------
    # 1. Check whether VLM provides meaningful context
    # --------------------------------------------------

    weak_vlm_outputs = {
        "",
        "picture.",
        "picture",
        "image.",
        "image",
        "photo.",
        "photo"
    }

    if context in weak_vlm_outputs or len(context) < 15:
        vlm_quality = 0.0
    else:
        vlm_quality = 1.0

    # --------------------------------------------------
    # 2. Check object agreement
    # --------------------------------------------------

    object_words = object_name.lower().replace("_", " ")

    object_agreement = 0.0

    if object_words in context:
        object_agreement = 1.0

    # --------------------------------------------------
    # 3. Safety evidence
    # --------------------------------------------------

    visual_safety_score = yolo_confidence * 100

    if distance == "NEAR":
        visual_safety_score += 20

    elif distance == "MEDIUM":
        visual_safety_score += 10

    if direction == "CENTER":
        visual_safety_score += 15

    visual_safety_score = min(100, visual_safety_score)

    # --------------------------------------------------
    # 4. Multimodal agreement
    # --------------------------------------------------

    if vlm_quality == 0:
        agreement_score = 0

    elif object_agreement:
        agreement_score = 100

    else:
        agreement_score = 50

    # --------------------------------------------------
    # 5. Final evidence confidence
    # --------------------------------------------------

    if vlm_quality == 0:
        final_confidence = visual_safety_score

    else:
        final_confidence = (
            0.60 * visual_safety_score
            + 0.40 * agreement_score
        )

    final_confidence = round(
        min(100, final_confidence),
        2
    )

    # --------------------------------------------------
    # 6. Final safety decision
    # --------------------------------------------------

    if direction == "CENTER" and distance == "NEAR":
        action = "STOP"

    elif direction == "LEFT" and distance == "NEAR":
        action = "MOVE RIGHT"

    elif direction == "RIGHT" and distance == "NEAR":
        action = "MOVE LEFT"

    elif distance == "MEDIUM":
        action = "CAUTION"

    else:
        action = "SAFE"

    # --------------------------------------------------
    # 7. Evidence confidence level
    # --------------------------------------------------

    if final_confidence >= 80:
        confidence_level = "HIGH"

    elif final_confidence >= 60:
        confidence_level = "MEDIUM"

    else:
        confidence_level = "LOW"

    return {
        "object": object_name,
        "direction": direction,
        "distance": distance,
        "yolo_confidence": yolo_confidence,
        "vlm_quality": vlm_quality,
        "object_agreement": object_agreement,
        "agreement_score": agreement_score,
        "final_confidence": final_confidence,
        "confidence_level": confidence_level,
        "action": action,
        "vlm_context": vlm_context
    }


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    print("VoiceToVision - MULTIMODAL EVIDENCE FUSION")
    print("=" * 60)

    yolo_evidence = {
        "object": "person",
        "confidence": 0.90,
        "direction": "CENTER",
        "visual_distance": "NEAR"
    }

    vlm_context = (
        "A person is standing directly in front of "
        "the camera in a room."
    )

    result = calculate_fusion(
        yolo_evidence,
        vlm_context
    )

    print("\nYOLO EVIDENCE")
    print("-" * 60)

    print(f"Object       : {yolo_evidence['object']}")
    print(f"Confidence   : {yolo_evidence['confidence']}")
    print(f"Direction    : {yolo_evidence['direction']}")
    print(f"Distance     : {yolo_evidence['visual_distance']}")

    print("\nVLM CONTEXT")
    print("-" * 60)

    print(vlm_context)

    print("\nFUSION RESULT")
    print("-" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)