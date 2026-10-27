from typing import Dict


def calculate_risk(evidence: Dict) -> Dict:

    confidence = evidence["confidence"]
    direction = evidence["direction"]
    distance = evidence["visual_distance"]

    # -----------------------------
    # Confidence Score
    # -----------------------------

    confidence_score = confidence * 100

    # -----------------------------
    # Distance Risk
    # -----------------------------

    distance_score = {
        "FAR": 10,
        "MEDIUM": 40,
        "NEAR": 75
    }.get(distance, 20)

    # -----------------------------
    # Direction Risk
    # -----------------------------

    direction_score = {
        "LEFT": 40,
        "CENTER": 80,
        "RIGHT": 40
    }.get(direction, 20)

    # -----------------------------
    # Combined Risk Score
    # -----------------------------

    risk_score = (
        0.40 * distance_score
        + 0.35 * direction_score
        + 0.25 * confidence_score
    )

    risk_score = round(risk_score, 2)

    # -----------------------------
    # Risk Level
    # -----------------------------

    if risk_score >= 80:
        risk_level = "CRITICAL"

    elif risk_score >= 60:
        risk_level = "HIGH"

    elif risk_score >= 35:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # -----------------------------
    # Safety Decision
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
    # Final Result
    # -----------------------------

    return {
        "object": evidence["object"],
        "confidence": confidence,
        "direction": direction,
        "distance": distance,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "action": action
    }


# ---------------------------------
# Test Evidence
# ---------------------------------

if __name__ == "__main__":

    test_evidence = {
        "object": "person",
        "confidence": 0.88,
        "direction": "CENTER",
        "visual_distance": "NEAR"
    }

    result = calculate_risk(test_evidence)

    print("SAFETY ENGINE RESULT")
    print("-" * 50)

    for key, value in result.items():
        print(f"{key}: {value}")