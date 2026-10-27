import sys
import time
from pathlib import Path

# Add project root to sys.path to ensure clean package imports
# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2

from app.core.config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS_TARGET,
    DEFAULT_LANGUAGE
)
from app.core.camera import CameraStream
from app.core.audio_manager import AudioManager
from app.ai.yolo_tracker import YoloTracker
from app.ai.vlm_engine import VlmEngine
from app.ai.speech_engine import SpeechEngine
from app.ui.hud_renderer import HudRenderer

from app.modes.navigation_mode import NavigationMode
from app.modes.assistant_mode import AssistantMode
from app.modes.reader_mode import ReaderMode


def print_banner(current_lang: str):
    print("=" * 75)
    print("         VOICE TO VISION 2.0 - MAJOR ASSISTIVE AI PLATFORM         ")
    print("   Real-Time Spatial Navigation | Conversational Vision | Text Reader   ")
    print("=" * 75)
    print(f" [ACTIVE LANGUAGE] : {current_lang}")
    print(" [CONTROLS]                                                       ")
    print("   [TAB]   : Switch Mode (Navigation -> Assistant -> Reader)       ")
    print("   [L]     : Switch Language (HINDI <-> ENGLISH)                   ")
    print("   [SPACE] : Context Action (Voice Query / Read Text / Scene Scan) ")
    print("   [M]     : Toggle Audio Mute                                     ")
    print("   [Q/ESC] : Quit & Stop Application                               ")
    print("=" * 75)


def main():
    current_language = DEFAULT_LANGUAGE.upper()
    print_banner(current_language)

    print("[Init] Initializing Audio Engine...")
    audio_mgr = AudioManager().start()
    init_msg = "Voice To Vision platform shuru ho raha hai. Kripya prateeksha karein." if current_language == "HINDI" else "Initializing Voice To Vision Platform. Please wait."
    audio_mgr.speak_info(init_msg, debounce_sec=0.1)

    print("[Init] Starting High-Performance Threaded Camera...")
    camera = CameraStream(width=CAMERA_WIDTH, height=CAMERA_HEIGHT).start()
    if not camera.is_opened():
        print("[Error] Could not open camera. Please check camera connection.")
        err_msg = "Camera ki kharabi. Device nahi mila." if current_language == "HINDI" else "Camera error. Device not detected."
        audio_mgr.speak_urgent(err_msg)
        time.sleep(2)
        return

    print("[Init] Loading YOLOv11 Spatial Tracking Network...")
    tracker = YoloTracker()

    print("[Init] Starting Asynchronous SmolVLM Background Engine...")
    vlm_engine = VlmEngine(default_lang=current_language).start()

    print("[Init] Preparing Whisper Speech Recognition Engine...")
    speech_engine = SpeechEngine()

    hud = HudRenderer()

    modes = [
        ("Navigation", NavigationMode(audio_mgr, vlm_engine, language=current_language)),
        ("Voice Assistant", AssistantMode(audio_mgr, vlm_engine, speech_engine, language=current_language)),
        ("Text Reader", ReaderMode(audio_mgr, vlm_engine, language=current_language))
    ]
    current_mode_idx = 0

    mode_name, current_mode = modes[current_mode_idx]
    current_mode.on_enter()

    print("\n" + "=" * 75)
    print("[Ready] All systems operational. Starting main visual loop...")
    print("=" * 75 + "\n")

    # Main Processing Loop
    window_name = "VoiceToVision 2.0 - Major Assistive AI Platform"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CAMERA_WIDTH, CAMERA_HEIGHT)

    prev_frame_time = time.time()

    try:
        while True:
            grabbed, frame = camera.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            # Calculate main thread FPS
            now = time.time()
            fps = 1.0 / max(1e-5, now - prev_frame_time)
            prev_frame_time = now

            detections, active_track_ids = tracker.track(frame)

            evaluations = []
            priority_hazard = None

            if mode_name == "Navigation":
                evaluations = current_mode.process_frame(frame, detections, active_track_ids)
                if evaluations:
                    priority_hazard = current_mode.safety_evaluator.get_priority_hazard(evaluations)

            elif mode_name == "Voice Assistant":
                # Keep tracking active for visual awareness
                if detections:
                    evaluations = [
                        {
                            "track_id": d["track_id"],
                            "object": d["name"],
                            "confidence": d["confidence"],
                            "direction": d["direction"],
                            "distance": d["distance"],
                            "risk_level": "LOW",
                            "action": "SAFE",
                            "bbox": d["bbox"]
                        } for d in detections
                    ]

            elif mode_name == "Text Reader":
                # In reader mode, draw focus frame guide
                h, w = frame.shape[:2]
                rx1, ry1 = int(w * 0.20), int(h * 0.20)
                rx2, ry2 = int(w * 0.80), int(h * 0.80)
                evaluations = [{
                    "track_id": "document_target",
                    "object": "DOCUMENT TARGET",
                    "confidence": 1.0,
                    "direction": "CENTER",
                    "distance": "NEAR",
                    "risk_level": "LOW",
                    "action": "SCAN",
                    "bbox": (rx1, ry1, rx2, ry2)
                }]

            caption = audio_mgr.get_live_caption()
            vlm_busy = vlm_engine.is_busy
            speech_busy = False
            if hasattr(current_mode, "listening"):
                speech_busy = current_mode.listening

            display_frame = hud.render(
                frame=frame,
                mode_name=mode_name,
                evaluations=evaluations,
                priority_hazard=priority_hazard,
                fps=fps,
                caption=caption,
                language=current_language,
                iot_active=False,
                vlm_busy=vlm_busy,
                speech_busy=speech_busy,
                is_muted=audio_mgr.muted
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:
                print("\n[System] Shutdown requested by user.")
                break

            elif key == 9:  # TAB -> Cycle Mode
                current_mode.on_exit()
                current_mode_idx = (current_mode_idx + 1) % len(modes)
                mode_name, current_mode = modes[current_mode_idx]
                current_mode.set_language(current_language)
                current_mode.on_enter()

            elif key in [ord("l"), ord("L")]:  # L -> Toggle Language
                current_language = "ENGLISH" if current_language == "HINDI" else "HINDI"
                print(f"\n[Language] Switched to: {current_language}")

                # Update language in all subsystem modes
                for _, m in modes:
                    m.set_language(current_language)
                vlm_engine.set_language(current_language)

                # Voice confirmation of language change
                confirm_msg = "Bhasha badal kar Hindi kar di gayi hai." if current_language == "HINDI" else "Language switched to English."
                audio_mgr.speak_info(confirm_msg, debounce_sec=0.1)

            elif key == 32:  # SPACE -> Contextual Action
                if mode_name == "Navigation":
                    audio_mgr.speak_info("Scanning surroundings...", debounce_sec=0.1)
                    scan_prompt = "Aas paas ka drishya dekha ja raha hai..." if current_language == "HINDI" else "Scanning surroundings..."
                    audio_mgr.speak_info(scan_prompt, debounce_sec=0.1)
                    vlm_engine.request_scene_summary(
                        frame,
                        callback=lambda desc: audio_mgr.speak_info(desc, debounce_sec=0.1)
                    )
                elif mode_name == "Voice Assistant":
                    current_mode.trigger_listening(frame)
                elif mode_name == "Text Reader":
                    current_mode.trigger_read(frame)

            elif key in [ord("m"), ord("M")]:  # Toggle Mute
                is_muted = audio_mgr.toggle_mute()
                state_str = "Muted" if is_muted else "Unmuted"
                print(f"[System] Audio {state_str}.")

    except KeyboardInterrupt:
        print("\n[System] Interrupted by keyboard.")

    finally:
        print("\n[System] Releasing resources and shutting down...")
        camera.stop()
        audio_mgr.speak_info("System shutting down. Goodbye.", debounce_sec=0.1)
        bye_msg = "System band ho raha hai. Namaste." if current_language == "HINDI" else "System shutting down. Goodbye."
        audio_mgr.speak_info(bye_msg, debounce_sec=0.1)
        time.sleep(1.0)
        audio_mgr.stop()
        vlm_engine.stop()
        cv2.destroyAllWindows()
        print("[System] VoiceToVision Platform stopped cleanly.")


if __name__ == "__main__":
    main()
