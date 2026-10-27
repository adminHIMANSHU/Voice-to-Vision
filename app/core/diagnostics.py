"""Runtime checks used before starting the assistive pipeline."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Optional

from app.core.config import CAMERA_INDEX, YOLO_MODEL_PATH


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def check_python_package(package_name: str, required: bool = True) -> CheckResult:
    try:
        import_module(package_name)
    except Exception as exc:
        return CheckResult(
            name=f"package:{package_name}",
            ok=False,
            detail=f"import failed: {type(exc).__name__}: {exc}",
            required=required,
        )
    return CheckResult(
        name=f"package:{package_name}",
        ok=True,
        detail="installed and importable",
        required=required,
    )


def check_model_file(model_path: Optional[str] = None) -> CheckResult:
    path = Path(model_path or YOLO_MODEL_PATH)
    return CheckResult(
        name="model:yolo",
        ok=path.is_file(),
        detail=str(path) if path.is_file() else f"missing: {path}",
    )


def check_camera(camera_index: int = CAMERA_INDEX) -> CheckResult:
    try:
        import cv2
    except ImportError:
        return CheckResult("camera", False, "OpenCV is not installed")

    capture = cv2.VideoCapture(camera_index)
    try:
        opened = capture.isOpened()
        return CheckResult(
            name="camera",
            ok=opened,
            detail=f"index {camera_index} available" if opened else f"index {camera_index} unavailable",
        )
    finally:
        capture.release()


def run_preflight(include_camera: bool = False) -> list[CheckResult]:
    """Run checks that do not initialize AI models or speech engines."""
    results = [
        check_python_package("cv2"),
        check_python_package("numpy"),
        check_python_package("torch"),
        check_python_package("ultralytics"),
        check_python_package("transformers"),
        check_python_package("whisper"),
        check_python_package("sounddevice"),
        check_python_package("pyttsx3"),
        check_model_file(),
    ]
    if include_camera:
        results.append(check_camera())
    return results


def format_preflight(results: list[CheckResult]) -> str:
    lines = ["VoiceToVision preflight", "=" * 24]
    for result in results:
        state = "OK" if result.ok else ("MISSING" if result.required else "OPTIONAL")
        lines.append(f"[{state:7}] {result.name}: {result.detail}")
    return "\n".join(lines)