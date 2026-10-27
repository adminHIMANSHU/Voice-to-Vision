# VoiceToVision 2.0: Multimodal Assistive AI Platform for the Visually Impaired

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-brightgreen.svg)](https://docs.ultralytics.com/)
[![SmolVLM](https://img.shields.io/badge/SmolVLM-256M--Instruct-orange.svg)](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
[![Whisper](https://img.shields.io/badge/Whisper-tiny-red.svg)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VoiceToVision 2.0** is an intelligent, multi-threaded assistive computer vision and conversational AI platform designed to empower visually impaired individuals with independent spatial navigation, hazard awareness, and contextual scene comprehension.

---

## 🌟 Key Highlights & Major Project Features

| Feature | Legacy Prototype | **VoiceToVision 2.0 (Major Upgrade)** |
| :--- | :--- | :--- |
| **Pipeline Architecture** | Single-threaded (Frame freeze during VLM/Voice) | **Asynchronous Multi-Threaded** (Smooth 30 FPS) |
| **Interaction Model** | One-way passive warning alerts | **Two-Way Conversational Vision** (Whisper + SmolVLM) |
| **Operating Modes** | Single hardcoded loop | **3 Specialized Modes** (Navigation, Assistant, Reader) |
| **Spatial Awareness** | Static bounding box area heuristic | **3-Zone Orientation + Approach Rate Velocity Vector** |
| **Audio Synthesizer** | Blocking `runAndWait()` | **Priority Queue + Subtitle Telemetry + Anti-Spam** |
| **Visual Telemetry** | Plain OpenCV text | **Cyber HUD with Top-Down Spatial Mini-Radar & Subtitles** |
| **Text Reading** | None | **On-Demand OCR / Document Reader Mode** |

---

## 🏗️ System Architecture

```
                                    ┌────────────────────────┐
                                    │  Camera Video Stream   │
                                    └───────────┬────────────┘
                                                │ (Zero-Lag Frame Buffer)
                         ┌──────────────────────┼──────────────────────┐
                         ▼                      ▼                      ▼
               ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
               │  YOLOv11 Tracker  │  │  SmolVLM Worker   │  │   Whisper STT     │
               │   (Spatial +      │  │ (Background Scene │  │  (Microphone VAD  │
               │    Velocity)      │  │  & Question QA)   │  │   Transcriber)    │
               └─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
                         │                      │                      │
                         ▼                      │                      │
               ┌───────────────────┐            │                      │
               │ Safety Evaluator  │◄───────────┘                      │
               │ (Hysteresis Risk) │                                   │
               └─────────┬─────────┘                                   │
                         │                                             │
                         ├─────────────────────────────────────────────┘
                         ▼
               ┌───────────────────┐                 ┌───────────────────┐
               │    Audio Queue    │                 │   Modern UI HUD   │
               │  (Non-blocking    │◄───────────────►│ (Mini-Radar, Risk │
               │   Voice Output)   │                 │  Gauge, Captions) │
               └───────────────────┘                 └───────────────────┘
```

---

## 🕹️ Assistive Operating Modes

### 1. Navigation Mode (Default)
* Real-time spatial tracking of people, furniture, stairs, doors, and vehicles.
* Classifies position into **`LEFT`**, **`CENTER`**, or **`RIGHT`**.
* Estimates proximity (**`NEAR`**, **`MEDIUM`**, **`FAR`**) and dynamically detects if an obstacle is **`APPROACHING`**.
* Dual-threshold **hysteresis engine** eliminates flickering state changes.
* Highest-risk obstacle is prioritized for spoken directives (*"Stop! Person directly ahead, approaching."*).
* Alerts are voice-only; navigation does not activate vibration motors or a buzzer.

### 2. Conversational Voice Assistant Mode
* Press **`[SPACE]`** or use voice query.
* User asks natural questions aloud:
  * *"Is my walkway clear?"*
  * *"Where is my backpack?"*
  * *"Is anyone standing in front of me?"*
* **OpenAI Whisper** transcribes the user's speech; **SmolVLM** contextually inspects the live camera frame and speaks back a natural, accurate answer.

### 3. Document & Text Reader Mode
* Designed for reading medication packaging, book pages, door numbers, and street signboards.
* User presents text in front of the camera and taps **`[SPACE]`**.
* High-resolution visual snapshot is parsed and read aloud.

---

## 📊 Safety Evaluation & Risk Mathematics

The safety engine computes a multi-factor normalized risk score $R \in [0, 100]$:

$$R = w_d \cdot S_{\text{dist}} + w_p \cdot S_{\text{pos}} + w_c \cdot S_{\text{conf}} + M_{\text{approach}}$$

* **Distance Factor ($S_{\text{dist}}$)**: `NEAR` = 75, `MEDIUM` = 40, `FAR` = 10 ($w_d = 0.40$).
* **Positional Factor ($S_{\text{pos}}$)**: `CENTER` = 80, `LEFT`/`RIGHT` = 40 ($w_p = 0.35$).
* **Confidence Factor ($S_{\text{conf}}$)**: $C \times 100$ ($w_c = 0.25$).
* **Dynamic Approach Bonus ($M_{\text{approach}}$)**: $+15$ if area delta $\Delta A / \Delta t > \tau$.

### Hysteresis State Transition Table
To eliminate rapid audio fluttering:
* **Enter `CRITICAL`**: Risk $\ge 80$ | **Exit `CRITICAL`**: Risk $< 75$
* **Enter `HIGH`**: Risk $\ge 60$ | **Exit `HIGH`**: Risk $< 55$
* **Enter `MEDIUM`**: Risk $\ge 35$ | **Exit `MEDIUM`**: Risk $< 30$

---

## ⌨️ Controls & Keybindings

| Key | Function |
| :--- | :--- |
| **`[TAB]`** | Cycle Modes: `Navigation` ➔ `Voice Assistant` ➔ `Text Reader` |
| **`[SPACE]`** | Context Action: Ask Voice Question / Read Text / Trigger Scene Scan |
| **`[M]`** | Toggle Audio Mute / Unmute |
| **`[Q]` or `[ESC]`** | Clean Shutdown and Stop Application |

---

## 🚀 Quick Start Guide

### Option 1: One-Click Windows Launcher (Recommended)
Double-click `run.bat` in the project root.

### Option 2: Command Line
Activate your conda environment and run:
```bash
conda activate voicevision
python app/main.py
```

### Development Validation

Run the deterministic unit tests before using camera, microphone, or model-backed smoke tests:
```bash
python -m compileall -q app tests
pytest -q
```

The automated suite covers spatial reasoning, risk evaluation, bilingual alerts, and stale-track cleanup. Hardware and model smoke tests remain separate because they require local devices and large model dependencies.

### Environment Diagnostics

Before launching the application, run:
```bash
python scripts/doctor.py
python scripts/doctor.py --camera
```

The doctor command checks the Python dependencies and YOLO weight without loading large models. The `--camera` option also checks camera access. Install the dependencies in the same environment used to launch the application:
```bash
python -m pip install -r requirements.txt
```

For a research or embedded deployment, record the output of the doctor command with the experiment results. This makes hardware and software conditions reproducible.

### Core Performance Benchmark

Measure the CPU-only spatial and safety decision layers with:
```bash
python scripts/benchmark_core.py
```

This benchmark does not load AI models. End-to-end FPS, YOLO latency, VLM latency, speech latency, and camera throughput must be measured separately on the target hardware.

---

## 📁 Repository Structure

```
VoiceToVision/
├── app/
│   ├── core/                      # Asynchronous Infrastructure
│   │   ├── config.py              # Global configurations & thresholds
│   │   ├── camera.py              # Threaded camera stream
│   │   ├── audio_manager.py       # Non-blocking priority audio worker
│   │   ├── spatial_engine.py      # 3-Zone & approach velocity math
│   │   └── safety_evaluator.py    # Multi-factor risk & hysteresis
│   ├── ai/                        # Deep Learning Engines
│   │   ├── yolo_tracker.py        # YOLOv11 tracker with persistent IDs
│   │   ├── vlm_engine.py          # Asynchronous SmolVLM worker
│   │   └── speech_engine.py       # Whisper speech recognition
│   ├── ui/                        # Assistive Telemetry & Display
│   │   └── hud_renderer.py        # Glassmorphism HUD & Mini-Radar
│   ├── modes/                     # Operating State Machines
│   │   ├── navigation_mode.py     # Real-time obstacle avoidance
│   │   ├── assistant_mode.py      # Two-way conversational Q&A
│   │   └── reader_mode.py         # Text & document reader
│   └── main.py                    # Master application entrypoint
├── data/                          # Audio & test images
├── models/                        # Saved weights & checkpoints
├── run.bat                        # One-click Windows runner
├── requirements.txt               # Dependency specifications
└── README.md                      # Comprehensive documentation
```

---

## 🎓 Viva & Presentation Q&A Preparation

**Q1: Why did you separate YOLO and VLM instead of using only VLM?**
> *Answer:* VLMs like SmolVLM provide rich semantic understanding but have higher latency (typically 1–3 seconds on CPU/edge). YOLOv11 runs at 30+ FPS, providing instant real-time collision detection. Our dual-tiered architecture achieves the best of both worlds: zero-latency reflex alerts via YOLO + rich contextual understanding via background VLM.

**Q2: How do you prevent audio alert spam when an obstacle stays in front of the user?**
> *Answer:* We implemented dual stabilization: (1) An **Alert Debouncer** with cooldown timestamps that suppresses duplicate messages, and (2) **Hysteresis Thresholds** that require the risk score to drop significantly below the entry threshold before changing state.

**Q3: How do you measure distance without a physical depth sensor (LiDAR)?**
> *Answer:* We employ monocular relative distance estimation based on bounding box area ratio and temporal rate of expansion ($\Delta \text{area} / \Delta t$). When an object's bounding box expands rapidly across successive frames, our spatial engine flags it as `APPROACHING` and amplifies the risk score.
