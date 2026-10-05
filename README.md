# Assistive Real-Time Meeting Tool

[![Architecture: Modular Python](https://img.shields.io/badge/Architecture-Modular%20Python%203.10-blue.svg)](#architecture)
[![Docker Ready](https://img.shields.io/badge/Docker-Multi--Stage%20X11-green.svg)](#docker-deployment)
[![AI Pipelines](https://img.shields.io/badge/AI-Whisper%20%7C%20MediaPipe%20%7C%20PyTorch-orange.svg)](#system-features)

A production-grade, modular edge application designed to bridge communication barriers in virtual and hybrid meetings for hearing-impaired and speech-impaired individuals.

---

## Abstract & System Features

1. **Speech-to-Text (STT) Subtitles**:
   - Captures microphone audio streams asynchronously.
   - Converts speech into live subtitles using OpenAI Whisper or Google SpeechRecognition.
   - Delivers real-time subtitles via a non-blocking HUD banner with automatic word wrapping.

2. **Sign Language Recognition (SLR) - ISL & ASL**:
   - **Indian Sign Language (ISL)**: Recognizes canonical dual-handed (bimanual) and single-handed signs (*Namaste/Hello*, *Thank You*, *Help*, *Yes*, *No*, *Need Water*, *Meeting/Together*, etc.).
   - **American Sign Language (ASL)**: Recognizes unimanual signs and finger spelling (*Hello*, *Thank You*, *I Love You*, *Peace/Victory*, *OK*, *Help*, letters *A, B, C, L, V, Y*).
   - Real-time MediaPipe hand landmark extraction with temporal hysteresis smoothing.
   - Extensible support for custom-trained ML models (`.onnx` or `.pt`).

3. **Text-to-Speech (TTS) Voice Synthesis**:
   - Converts confirmed sign language gestures into synthetic voice using `pyttsx3` (offline) or `gTTS`.
   - Incorporates time-window debouncing to prevent stuttering or repeated utterances.

4. **HUD Video Overlay**:
   - Low-overhead, 30+ FPS translucent glassmorphic HUD.
   - Displays live telemetry (FPS, audio indicators, active mode), hand tracking bounding boxes, confidence meters, and live subtitle bars.

---

## Architecture Blueprint

```
+-----------------------------------------------------------------------------+
|                          Host Operating System                              |
|   Webcam (/dev/video0)  |  Microphone (/dev/snd)  |  Display (/tmp/.X11-unix) |
+-----------------------------------------------------------------------------+
                                      |
                              (Docker Devices)
                                      v
+-----------------------------------------------------------------------------+
|                      Containerized Application Boundary                     |
|                                                                             |
|   +-------------------+        +----------------------------------------+   |
|   | STT Audio Worker  | -----> | Subtitle Queue (queue.Queue)           |   |
|   | (modules/stt.py)  |        +----------------------------------------+   |
|   +-------------------+                            |                        |
|                                                    v                        |
|   +-------------------+        +----------------------------------------+   |
|   | Video Capture     | -----> | Main Controller & HUD Overlay Pipeline |   |
|   | (modules/isl.py,  |        | (main.py, modules/ui_overlay.py)       |   |
|   |  modules/asl.py)  |        +----------------------------------------+   |
|   +-------------------+                            |                        |
|             | (Confirmed Gesture)                  v                        |
|             v                              X11 Composite Output             |
|   +-------------------+                                                     |
|   | TTS Audio Worker  | -----> Spoken Voice Output                          |
|   | (modules/tts.py)  |                                                     |
|   +-------------------+                                                     |
+-----------------------------------------------------------------------------+
```

---

## Directory Structure

```
AI_meeting_Assistant/
├── .env.example                     # Environment configuration template
├── .env                             # Active configuration
├── Dockerfile                       # Multi-stage production container with X11 & ALSA
├── docker-compose.yml               # Service definitions with device bindings
├── requirements.txt                 # Locked dependency specifications
├── README.md                        # Project documentation & operational manual
├── src/
│   ├── __init__.py
│   ├── config.py                    # Pydantic Settings & environment manager
│   ├── main.py                      # Main controller, video loop, & CLI
│   └── modules/
│       ├── __init__.py              # Namespace exports
│       ├── base.py                  # Abstract base classes & spatial landmark math
│       ├── stt.py                   # Speech-to-Text streaming worker
│       ├── isl_detection.py         # Indian Sign Language recognition engine
│       ├── asl_detection.py         # American Sign Language recognition engine
│       ├── tts.py                   # Non-blocking Text-to-Speech worker
│       └── ui_overlay.py            # Real-time HUD overlay renderer
└── tests/
    ├── __init__.py
    ├── test_config.py               # Settings validation tests
    ├── test_stt.py                  # Audio queue and subtitle lifecycle tests
    ├── test_isl_asl.py              # Gesture classification & landmark math tests
    ├── test_tts.py                  # Audio debouncing and synthesis tests
    └── test_ui_overlay.py           # HUD compositing & text wrapping tests
```

---

## Quick Start with Docker

### macOS Users (Docker Desktop)

> [!NOTE]
> On macOS, `xhost` and Linux V4L2 device nodes (`/dev/video0`) do not exist natively because Docker runs inside a lightweight Linux VM. 
> To provide an instant, zero-friction experience without needing XQuartz or X11 configurations, the application includes an **embedded Web Stream HUD**.

#### 1. Launch with Web Stream HUD
```bash
# Build and launch the container
docker compose -f docker-compose.mac.yml up --build
```
*(Alternatively, run `docker compose up --build meeting-tool-web`)*

#### 2. View in Your Browser
Open your browser to:
👉 **[http://localhost:5050](http://localhost:5050)**

You will see:
- Real-time video stream (camera / animated meeting emulation)
- Live Speech-to-Text (STT) subtitles
- Indian Sign Language (ISL) & American Sign Language (ASL) gesture translation card
- Telemetry stats (FPS, Audio sync, active mode)
- Interactive toggle button to switch between ISL and ASL

#### 3. Container Lifecycle Commands (Stop, Restart, Logs)
```bash
# Restart the container
docker compose -f docker-compose.mac.yml restart meeting-tool

# Stop the container
docker compose -f docker-compose.mac.yml stop

# Start again in background
docker compose -f docker-compose.mac.yml up -d meeting-tool

# View live application logs
docker logs -f assistive_meeting_assistant_mac
```

#### 4. Run Automated Tests Inside Docker
```bash
docker compose -f docker-compose.mac.yml run --rm test-runner
```

---

### How to Test ASL Symbols (Hands in Front of Camera)

1. Open **[http://localhost:5050](http://localhost:5050)** in your browser.
2. Ensure mode is set to **`ASL`** (click the **"Toggle ISL / ASL"** button if it currently says `ISL`).
3. Click the green button **`[📷 Enable Real Webcam]`** and grant camera permission.
4. Hold your hand in front of the camera and form any of these signs steady for about half a second:

| Symbol / Gesture | Hand Posture Instructions | Expected HUD Output |
|:---:|:---|:---:|
| **🤟 "I Love You"** | Extend **Thumb**, **Index finger**, and **Pinky**. Fold **Middle** and **Ring** fingers. | `I Love You` (96%) |
| **✌️ "Peace / 'V'"** | Extend **Index** and **Middle** fingers in a V-shape. Curl **Thumb**, **Ring**, and **Pinky** down. | `Peace / 'V'` (92%) |
| **👌 "OK / Agreement"** | Touch tips of **Thumb** and **Index** in a circle. Extend **Middle**, **Ring**, and **Pinky**. | `OK / Agreement` (94%) |
| **👆 "Letter 'L'"** | Extend **Thumb** and **Index** at a 90° angle (forming an 'L'). Fold other fingers. | `Letter 'L'` (92%) |
| **🤙 "Letter 'Y'"** | Extend only **Thumb** and **Pinky** outwards (shaka sign). Curl others down. | `Letter 'Y'` (92%) |
| **👍 "Thumbs Up / Good"** | Make a closed fist with **Thumb** pointing straight up vertically. | `Thumbs Up / Good` (91%) |
| **👋 "Hello"** | Hold an open, flat hand (all 4 fingers pointing up) near your face/temple level. | `Hello` (90%) |
| **✊ on 🖐️ "Help"** | Place one hand flat (palm up), and rest a thumbs-up fist on top of the flat palm. | `Help` (94%) |

---

### Linux Users (Native Hardware Passthrough)

#### 1. Enable Host Display Access (X11)
```bash
# Allow local container connections to X11 display
xhost +local:root

# Ensure user belongs to video and audio groups
sudo usermod -aG video,audio $USER
```

#### 2. Launch with Physical Webcam & Microphone
```bash
docker compose up --build meeting-tool
```

#### 3. Run Automated Tests Inside Docker
```bash
docker compose run --rm test-runner
```

---

## Local Development (Without Docker)

### Prerequisites
- Python 3.10
- PortAudio & eSpeak libraries:
  - **Ubuntu/Debian**: `sudo apt install libasound2-dev portaudio19-dev espeak libespeak-dev ffmpeg`
  - **macOS**: `brew install portaudio espeak ffmpeg`

### Setup Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Launch Locally
```bash
# Run with default settings (or modify .env)
python3 -m src.main

# CLI Overrides:
python3 -m src.main --mode ASL --camera 0
python3 -m src.main --mock      # Run with synthetic video/audio emulation
python3 -m src.main --headless  # Run in headless mode without window
```

### Run Tests Locally
```bash
pytest tests/ -v
```

---

## Testing & Integrating with Google Meet, Zoom & Teams

There are two primary ways to test this application in a live meeting:

### Method 1: Chrome Tab Sharing (Zero Installation — Fastest & Best Audio)
1. Launch the application:
   ```bash
   docker compose -f docker-compose.mac.yml up -d meeting-tool
   ```
2. Open **[http://localhost:5050](http://localhost:5050)** in Google Chrome.
3. Click anywhere on the webpage or click **`[🔊 Test Voice]`** to unlock browser audio playback.
4. Click **`[📷 Enable Real Webcam]`** to activate live hand tracking.
5. In your **Google Meet** call:
   - Click the **Present now** button (Screen Share icon).
   - Select **A Chrome Tab** $\rightarrow$ choose **`Assistive Meeting Tool - Real-Time Subtitles & Voice`**.
   - **Crucial:** Ensure the checkbox **"Also share tab audio"** is checked!
6. Whenever you form a sign gesture (e.g. 👍 *Good*, 🤟 *I Love You*, or 🙏 *Namaste*), the high-fidelity synthesized voice will speak aloud through your speakers AND stream directly into the Google Meet call for everyone to hear!

---

### Method 2: Virtual Camera (Pro Method — Appears as Your Google Meet Webcam)
To have Google Meet use the assistive HUD feed directly as your camera tile without screen sharing:

1. Install OBS Studio (free):
   ```bash
   brew install --cask obs
   ```
2. Open OBS Studio:
   - Under **Sources**, click **`+`** $\rightarrow$ select **Browser Source**.
   - Set URL to: `http://localhost:5050/video_feed` (Width: 1280, Height: 720).
   - In the bottom right of OBS, click **Start Virtual Camera**.
3. Open **Google Meet**:
   - Go to **Settings (three dots)** $\rightarrow$ **Video**.
   - Change your camera from FaceTime HD to **OBS Virtual Camera**.
4. Your video box in Google Meet will now display your live video feed with sign language translations and HUD subtitles!

---

## Keyboard Controls

| Key | Action |
|:---:|:---|
| **`M`** | Toggle Sign Language mode (**ISL** $\leftrightarrow$ **ASL**) |
| **`L`** | Toggle MediaPipe skeletal landmarks & hand connections |
| **`Q`** / **`ESC`** | Gracefully terminate the application |

---

## Configuration Reference (`.env`)

| Variable | Default | Description |
|:---|:---:|:---|
| `SIGN_LANGUAGE_MODE` | `ISL` | Active sign language interpreter (`ISL` or `ASL`) |
| `CAMERA_INDEX` | `0` | Camera device index (`0` for `/dev/video0`) |
| `CAMERA_WIDTH` / `HEIGHT` | `1280` / `720` | Video frame resolution |
| `STT_BACKEND` | `whisper` | Speech-to-Text engine (`whisper` or `speech_recognition`) |
| `WHISPER_MODEL_SIZE` | `tiny` | Whisper model size (`tiny`, `base`, `small`, etc.) |
| `TTS_ENGINE` | `pyttsx3` | Text-to-Speech engine (`pyttsx3` or `gtts`) |
| `TTS_DEBOUNCE_SECONDS` | `2.5` | Minimum seconds before re-voicing the same gesture |
| `GESTURE_CONFIRMATION_FRAMES`| `6` | Rolling frames required to debounce & confirm sign |
| `HEADLESS_MODE` | `false` | Run without opening an X11 window (for CI/servers) |
| `MOCK_HARDWARE` | `false` | Emulate camera/mic with synthetic feeds |
