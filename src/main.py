"""
Main Application Controller for Assistive Real-Time Meeting Tool.

Orchestrates concurrent pipelines:
1. Speech-to-Text (STT) audio worker
2. Sign Language Recognition (ISL / ASL) vision pipeline
3. Text-to-Speech (TTS) non-blocking audio engine
4. HUD Overlay & Video Compositing pipeline
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import time
from typing import Optional

import cv2
import numpy as np

from src.config import AppSettings, get_settings
from src.modules.asl_detection import AmericanSignLanguageRecognizer
from src.modules.base import GestureResult, SubtitleItem
from src.modules.isl_detection import IndianSignLanguageRecognizer
from src.modules.stt import SpeechToTextWorker
from src.modules.tts import TextToSpeechWorker
from src.modules.ui_overlay import MeetingHUDOverlay
from src.modules.web_stream import WebStreamServer

logger = logging.getLogger("assistive_meeting_tool")


class MeetingAssistantController:
    """
    Central controller managing lifecycle, hardware capture, module coordination,
    and user input events.
    """

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        """
        Initializes the application controller.

        :param settings: Application configuration instance.
        """
        self.settings = settings or get_settings()
        self._configure_logging()

        self.running = False
        self.active_mode: str = self.settings.SIGN_LANGUAGE_MODE.upper()
        self.display_landmarks: bool = self.settings.DISPLAY_LANDMARKS

        # Communication queues
        self.subtitle_queue: queue.Queue[SubtitleItem] = queue.Queue(maxsize=100)
        self.speech_queue: queue.Queue[str] = queue.Queue(maxsize=50)

        # Active subtitle tracker
        self.active_subtitle: Optional[SubtitleItem] = None

        # Core Modules
        logger.info("Initializing core AI and assistive modules...")
        self.stt_worker = SpeechToTextWorker(self.settings, self.subtitle_queue)
        self.isl_recognizer = IndianSignLanguageRecognizer(self.settings)
        self.asl_recognizer = AmericanSignLanguageRecognizer(self.settings)
        self.tts_worker = TextToSpeechWorker(self.settings, self.speech_queue)
        self.hud_overlay = MeetingHUDOverlay(self.settings)
        self.web_server: Optional[WebStreamServer] = None

        # Hardware capture state
        self.cap: Optional[cv2.VideoCapture] = None
        self._mock_frame_counter = 0
        self._display_failed = False

    def _configure_logging(self) -> None:
        """Configures standard formatted logging."""
        numeric_level = getattr(logging, self.settings.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _toggle_mode(self) -> str:
        """Toggles between ISL and ASL and returns new mode."""
        self.active_mode = "ASL" if self.active_mode == "ISL" else "ISL"
        logger.info("Switched active Sign Language mode to: %s", self.active_mode)
        self.isl_recognizer.reset()
        self.asl_recognizer.reset()
        return self.active_mode

    def start(self) -> None:
        """Starts worker threads and begins the main video processing loop."""
        self.running = True
        self._setup_signal_handlers()

        # Start web stream server if enabled
        if self.settings.ENABLE_WEB_STREAM:
            self.web_server = WebStreamServer(
                host="0.0.0.0",
                port=self.settings.WEB_STREAM_PORT,
                mode_toggle_callback=self._toggle_mode,
            )
            self.web_server.start()
            logger.info("Live Browser HUD feed available at: http://localhost:%d/", self.settings.WEB_STREAM_PORT)

        # Start background workers
        self.stt_worker.start()
        self.tts_worker.start()

        # Initialize Video Capture
        self._init_video_capture()

        logger.info("Starting main video loop. Press 'Q' to quit, 'M' to switch mode, 'L' to toggle landmarks.")
        self._run_loop()

    def stop(self) -> None:
        """Gracefully halts all pipelines and cleans up hardware resources."""
        if not self.running:
            return

        logger.info("Initiating graceful application shutdown...")
        self.running = False

        # Stop web server
        if self.web_server:
            self.web_server.stop()
            self.web_server = None

        # Stop workers
        self.stt_worker.stop()
        self.tts_worker.stop()

        # Release camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("Assistive Meeting Tool shutdown complete.")

    # -------------------------------------------------------------------------
    # Hardware & Video Pipeline
    # -------------------------------------------------------------------------

    def _init_video_capture(self) -> None:
        """Initializes OpenCV video capture device or falls back to synthetic stream."""
        if self.settings.MOCK_HARDWARE:
            logger.info("MOCK_HARDWARE is enabled. Bypassing physical camera initialization.")
            return

        cam_idx = self.settings.CAMERA_INDEX
        logger.info("Opening camera device index %d...", cam_idx)
        self.cap = cv2.VideoCapture(cam_idx)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, self.settings.CAMERA_FPS)
            logger.info(
                "Camera initialized successfully at %dx%d @ %d FPS",
                self.settings.CAMERA_WIDTH,
                self.settings.CAMERA_HEIGHT,
                self.settings.CAMERA_FPS,
            )
        else:
            logger.warning(
                "Could not open camera device %d. Operating in synthetic/mock video mode.", cam_idx
            )
            self.cap = None

    def _get_frame(self) -> np.ndarray:
        """Captures a frame from uploaded browser webcam, physical camera, or synthetic frame."""
        # 1. Live browser webcam stream (ideal for Docker on macOS/Windows)
        if self.web_server:
            incoming = self.web_server.get_incoming_frame()
            if incoming is not None:
                return incoming

        # 2. Hardware camera (V4L2 on Linux or AVFoundation on macOS native)
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            logger.warning("Frame grab failed from camera. Using synthetic fallback.")

        # Synthetic camera frame generator (useful for headless/containerized testing)
        self._mock_frame_counter += 1
        h, w = self.settings.CAMERA_HEIGHT, self.settings.CAMERA_WIDTH
        synth_frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Subtle animated gradient background
        shift = (self._mock_frame_counter * 2) % 255
        synth_frame[:, :] = (30, (shift // 4) + 20, 40)

        cv2.putText(
            synth_frame,
            "[SYNTHETIC VIDEO STREAM - CAMERA EMULATION]",
            (w // 2 - 320, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return synth_frame

    def _poll_subtitles(self) -> None:
        """Pulls newly arrived subtitles from the STT worker queue."""
        while not self.subtitle_queue.empty():
            try:
                item = self.subtitle_queue.get_nowait()
                self.active_subtitle = item
                logger.debug("Received live subtitle: %s", item.text)
            except queue.Empty:
                break

    @staticmethod
    def _format_speech_text(label: str) -> str:
        """Translates technical gesture labels into natural spoken phrases."""
        mapping = {
            "Yes / Agree": "Yes",
            "No / Disagree": "No",
            "Stop / Wait": "Stop",
            "Namaste / Hello": "Hello",
            "Meeting / Together": "Meeting",
            "Thumbs Up / Good": "Good",
            "Letter 'A' / Yes": "Yes",
            "OK / Agreement": "OK",
            "Peace / 'V'": "Peace",
            "Letter 'B'": "B",
            "Letter 'L'": "L",
            "Letter 'Y'": "Y",
            "Need Water": "Need Water",
            "Help Needed": "Help",
        }
        if label in mapping:
            return mapping[label]

        clean = label.replace("Letter", "").replace("'", "").strip()
        if "/" in clean:
            parts = [p.strip() for p in clean.split("/")]
            return parts[0] if parts[0] else clean
        return clean

    def _run_loop(self) -> None:
        """Core visual processing, inference, and rendering loop."""
        if not self.settings.HEADLESS_MODE:
            cv2.namedWindow(self.settings.WINDOW_TITLE, cv2.WINDOW_NORMAL)

        try:
            while self.running:
                loop_start = time.time()
                frame = self._get_frame()

                # Poll subtitles from audio worker
                self._poll_subtitles()

                # Dispatch frame to active sign language recognizer
                gesture_result: Optional[GestureResult] = None
                annotated_frame = frame

                if self.active_mode == "ISL":
                    annotated_frame, gesture_result = self.isl_recognizer.process_frame(
                        frame, draw_landmarks=self.display_landmarks
                    )
                else:
                    annotated_frame, gesture_result = self.asl_recognizer.process_frame(
                        frame, draw_landmarks=self.display_landmarks
                    )

                # If gesture is newly confirmed, send to Text-to-Speech & Web Speech API
                if gesture_result and gesture_result.is_confirmed:
                    spoken_text = self._format_speech_text(gesture_result.label)
                    enqueued = self.tts_worker.enqueue_speech(spoken_text)
                    if enqueued and self.web_server:
                        self.web_server.trigger_speech(spoken_text)

                # Render HUD Overlay onto the frame
                composite_frame = self.hud_overlay.render(
                    frame=annotated_frame,
                    active_mode=self.active_mode,
                    current_gesture=gesture_result,
                    active_subtitle=self.active_subtitle,
                    stt_active=self.stt_worker.is_running(),
                    tts_active=self.tts_worker.is_running(),
                )

                # Push frame to web browser streamer
                if self.web_server:
                    self.web_server.update_frame(
                        composite_frame,
                        telemetry={
                            "mode": self.active_mode,
                            "fps": round(self.hud_overlay._current_fps, 1),
                            "gesture": gesture_result.label if gesture_result else "None",
                            "subtitle": self.active_subtitle.text if self.active_subtitle else "",
                        },
                    )

                # Display or headless spin
                if not self.settings.HEADLESS_MODE and not self._display_failed:
                    try:
                        cv2.imshow(self.settings.WINDOW_TITLE, composite_frame)
                        key = cv2.waitKey(1) & 0xFF
                        self._handle_key(key)
                    except Exception as cv_err:
                        logger.warning(
                            "X11/GUI display unavailable (%s). Switching to browser web stream at http://localhost:%d",
                            cv_err,
                            self.settings.WEB_STREAM_PORT,
                        )
                        self._display_failed = True
                else:
                    # In headless mode or web-only mode, regulate loop frequency to target FPS
                    elapsed = time.time() - loop_start
                    delay = max(0.001, (1.0 / self.settings.CAMERA_FPS) - elapsed)
                    time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received.")
        finally:
            self.stop()

    def _handle_key(self, key: int) -> None:
        """Processes keyboard hotkeys."""
        if key == ord("q") or key == ord("Q") or key == 27:  # 27 = ESC
            logger.info("Exit hotkey pressed.")
            self.running = False
        elif key == ord("m") or key == ord("M"):
            self._toggle_mode()
        elif key == ord("l") or key == ord("L"):
            # Toggle landmark skeletal rendering
            self.display_landmarks = not self.display_landmarks
            logger.info("Toggled skeletal landmarks rendering: %s", self.display_landmarks)

    def _setup_signal_handlers(self) -> None:
        """Configures OS signal traps for clean termination."""
        def handler(sig: int, frame: Any) -> None:
            logger.info("Signal %d caught. Terminating cleanly...", sig)
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


def parse_arguments() -> argparse.Namespace:
    """Parses optional command-line overrides."""
    parser = argparse.ArgumentParser(
        description="Assistive Real-Time Meeting Tool: Live STT Subtitles & ISL/ASL Translation."
    )
    parser.add_argument(
        "--mode",
        choices=["ISL", "ASL"],
        default=None,
        help="Initial sign language interpreter mode (ISL or ASL)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="V4L2 camera device index",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without X11 GUI window",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with synthetic camera and microphone feeds",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=None,
        help="Enable embedded web streaming server (default: enabled)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for embedded web streaming server (default: 5000)",
    )
    return parser.parse_args()


def main() -> None:
    """Application CLI entry point."""
    args = parse_arguments()
    settings = get_settings()

    # Apply CLI overrides if provided
    if args.mode:
        settings.SIGN_LANGUAGE_MODE = args.mode
    if args.camera is not None:
        settings.CAMERA_INDEX = args.camera
    if args.headless:
        settings.HEADLESS_MODE = True
    if args.mock:
        settings.MOCK_HARDWARE = True
    if args.web is not None:
        settings.ENABLE_WEB_STREAM = args.web
    if args.port is not None:
        settings.WEB_STREAM_PORT = args.port

    controller = MeetingAssistantController(settings)
    controller.start()


if __name__ == "__main__":
    main()
