import os
from pathlib import Path

# ============================================================
# PROJECT ROOT & PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for folder in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Model paths
YOLO_MODEL_PATH = str(BASE_DIR / "yolo11n.pt")
VLM_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
WHISPER_MODEL_NAME = "tiny"

# ============================================================
# LANGUAGE CONFIGURATION (BILINGUAL SUPPORT)
# ============================================================
DEFAULT_LANGUAGE = "HINDI"  # "HINDI" or "ENGLISH"
AVAILABLE_LANGUAGES = ["HINDI", "ENGLISH"]

# Common object name phonetic translations for clear Hindi speech
HINDI_OBJECT_NAMES = {
    "person": "vyakti",
    "chair": "kursi",
    "table": "table",
    "bed": "bistar",
    "door": "darwaaza",
    "bottle": "botal",
    "car": "gaadi",
    "bicycle": "cycle",
    "motorcycle": "motorcycle",
    "cell phone": "mobile",
    "laptop": "laptop",
    "backpack": "bag",
    "dog": "kutta",
    "cat": "billi"
}

# ============================================================
# IoT & HARDWARE CONFIGURATION (RASPBERRY PI / JETSON)
# ============================================================
ENABLE_HARDWARE_HAPTICS = False  # Voice-only navigation; no motors or buzzer
GPIO_PIN_LEFT_HAPTIC = 18       # BCM Pin 18 (Left vibration motor)
GPIO_PIN_RIGHT_HAPTIC = 23      # BCM Pin 23 (Right vibration motor)
GPIO_PIN_BUZZER = 24            # BCM Pin 24 (Emergency hazard buzzer)

# ============================================================
# CAMERA CONFIGURATION
# ============================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS_TARGET = 30

# ============================================================
# DETECTION & SPATIAL PARAMETERS
# ============================================================
CONFIDENCE_THRESHOLD = 0.30
STABILITY_FRAMES = 3             # Track must persist across N frames to be active
RISK_WINDOW_SIZE = 5             # Rolling window for risk score averaging
APPROACH_THRESHOLD_RATIO = 0.015 # Δ(area ratio)/frame to classify as "approaching"

# Distance Area Ratios (Object Box Area / Frame Area)
DIST_NEAR_RATIO = 0.18
DIST_MEDIUM_RATIO = 0.06

# Spatial Thirds (Fraction of Frame Width)
ZONE_LEFT_SPLIT = 0.33
ZONE_RIGHT_SPLIT = 0.67

# ============================================================
# SAFETY & HYSTERESIS THRESHOLDS
# ============================================================
CRITICAL_ENTER = 80
CRITICAL_EXIT = 75

HIGH_ENTER = 60
HIGH_EXIT = 55

MEDIUM_ENTER = 35
MEDIUM_EXIT = 30

# Weightings for Risk Calculation
WEIGHT_DISTANCE = 0.40
WEIGHT_DIRECTION = 0.35
WEIGHT_CONFIDENCE = 0.25

# ============================================================
# AUDIO & SPEECH SYNTHESIS (TTS)
# ============================================================
SPEECH_RATE = 155         # Natural speaking rate
SPEECH_VOLUME = 1.0       # Max volume
ALERT_COOLDOWN_SEC = 5.0  # Increased to prevent repetitive warning spam
PRIORITY_CRITICAL = 1
PRIORITY_WARNING = 2
PRIORITY_INFO = 3

# ============================================================
# VLM INTELLIGENCE PARAMETERS
# ============================================================
VLM_AUTO_INTERVAL = 6.0   # Background scene update interval in seconds
VLM_MAX_NEW_TOKENS = 120

# ============================================================
# USER INTERFACE / HUD STYLES
# ============================================================
COLOR_BG_DARK = (18, 18, 22)
COLOR_SAFE = (76, 217, 100)      # Green (BGR)
COLOR_CAUTION = (50, 205, 255)   # Yellow/Amber (BGR)
COLOR_WARNING = (0, 140, 255)    # Orange (BGR)
COLOR_CRITICAL = (50, 50, 255)   # Red (BGR)
COLOR_ACCENT = (255, 191, 0)     # Cyan/Blue Accent (BGR)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (140, 140, 140)
