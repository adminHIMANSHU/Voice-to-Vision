import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.spatial_engine import SpatialEngine
from app.core.safety_evaluator import SafetyEvaluator
from app.core.diagnostics import CheckResult, format_preflight


def test_spatial_engine():
    print("Running Spatial Engine tests...")
    engine = SpatialEngine()
    frame_w, frame_h = 1280, 720

    # Test center obstacle near
    bbox_near = (400, 150, 880, 650) # large area
    res_near = engine.evaluate_spatial_attributes(bbox_near, frame_w, frame_h, track_id=1)
    assert res_near["direction"] == "CENTER", f"Expected CENTER, got {res_near['direction']}"
    assert res_near["distance"] == "NEAR", f"Expected NEAR, got {res_near['distance']}"

    # Test left obstacle far
    bbox_left = (50, 100, 150, 200) # small area on left
    res_left = engine.evaluate_spatial_attributes(bbox_left, frame_w, frame_h, track_id=2)
    assert res_left["direction"] == "LEFT", f"Expected LEFT, got {res_left['direction']}"
    assert res_left["distance"] == "FAR", f"Expected FAR, got {res_left['distance']}"

    print("Spatial Engine tests passed!")


def test_safety_evaluator():
    print("Running Safety Evaluator tests...")
    evaluator = SafetyEvaluator()

    det_critical = {
        "track_id": 10,
        "name": "person",
        "confidence": 0.92,
        "direction": "CENTER",
        "distance": "NEAR",
        "motion_state": "APPROACHING",
        "bbox": (400, 100, 880, 680)
    }

    # Simulate stability frames
    res = None
    for _ in range(4):
        res = evaluator.evaluate_object(det_critical)

    assert res is not None, "Evaluation should produce result after stability frames"
    assert res["risk_level"] in ["HIGH", "CRITICAL"], f"Expected high risk, got {res['risk_level']}"
    assert res["action"] == "STOP", f"Expected STOP action, got {res['action']}"
    print(f"Safety decision: {res['action']} ({res['risk_level']}, {res['smoothed_risk']}%)")
    print("Safety Evaluator tests passed!")


def test_safety_evaluator_hindi_alert():
    evaluator = SafetyEvaluator(language="HINDI")
    detection = {
        "track_id": 20,
        "name": "person",
        "confidence": 0.95,
        "direction": "CENTER",
        "distance": "NEAR",
        "motion_state": "APPROACHING",
        "bbox": (400, 100, 880, 680)
    }

    result = None
    for _ in range(4):
        result = evaluator.evaluate_object(detection)

    assert result is not None
    assert result["risk_level"] in ["HIGH", "CRITICAL"]
    assert "Ruk jaaiye" in result["alert_phrase"]


def test_spatial_engine_cleans_stale_tracks():
    engine = SpatialEngine()
    engine.evaluate_spatial_attributes((0, 0, 100, 100), 640, 480, track_id=7)

    assert 7 in engine.area_history
    engine.cleanup_tracks(set())
    assert 7 not in engine.area_history


def test_preflight_format_is_actionable():
    report = format_preflight([
        CheckResult("package:test", True, "installed"),
        CheckResult("model:test", False, "missing: test.pt"),
    ])

    assert "VoiceToVision preflight" in report
    assert "[OK" in report
    assert "missing: test.pt" in report


if __name__ == "__main__":
    test_spatial_engine()
    test_safety_evaluator()
    print("\nALL TEST SUITES PASSED SUCCESSFULLY!")

