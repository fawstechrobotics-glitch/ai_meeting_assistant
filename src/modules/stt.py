"""
Speech-to-Text (STT) Module for Assistive Real-Time Meeting Tool.

Handles asynchronous microphone audio capture, silence/voice activity detection,
and transcribes speech into subtitles using OpenAI Whisper or SpeechRecognition.
Transcriptions are delivered via a thread-safe queue for zero-latency HUD rendering.
"""

from __future__ import annotations

import io
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from src.config import AppSettings, get_settings
from src.modules.base import SubtitleItem

logger = logging.getLogger(__name__)


class SpeechToTextWorker:
    """
    Background worker that continuously streams audio from the configured microphone,
    performs speech recognition, and enqueues SubtitleItem instances.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        subtitle_queue: Optional[queue.Queue[SubtitleItem]] = None,
    ) -> None:
        """
        Initializes the STT worker.

        :param settings: Application configuration instance.
        :param subtitle_queue: Thread-safe queue to publish subtitles to HUD.
        """
        self.settings = settings or get_settings()
        self.subtitle_queue = subtitle_queue if subtitle_queue is not None else queue.Queue(maxsize=100)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Backend state
        self._recognizer: Optional[Any] = None
        self._whisper_model: Optional[Any] = None
        self._backend_initialized = False

    def start(self) -> None:
        """Starts the audio capture and transcription background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("STT Worker is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="STTWorkerThread", daemon=True)
        self._thread.start()
        logger.info(
            "STT Worker thread launched using backend: '%s' (language: '%s')",
            self.settings.STT_BACKEND,
            self.settings.STT_LANGUAGE,
        )

    def stop(self, timeout: float = 3.0) -> None:
        """
        Signals the worker thread to terminate and waits for join.

        :param timeout: Maximum seconds to wait for worker thread shutdown.
        """
        logger.info("Stopping STT Worker...")
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            logger.info("STT Worker stopped gracefully.")
        self._thread = None

    def is_running(self) -> bool:
        """Returns True if the background audio thread is active."""
        return self._thread is not None and self._thread.is_alive()

    # -------------------------------------------------------------------------
    # Internal Engine Initialization & Worker Loop
    # -------------------------------------------------------------------------

    def _initialize_backend(self) -> bool:
        """
        Lazily initializes the selected STT engine (Whisper or SpeechRecognition).
        Returns True if successful, False if fallback/mock is required.
        """
        backend = self.settings.STT_BACKEND.lower()
        try:
            if "whisper" in backend:
                logger.info(
                    "Loading Whisper model '%s' for STT processing...", self.settings.WHISPER_MODEL_SIZE
                )
                import whisper  # type: ignore[import-untyped]

                self._whisper_model = whisper.load_model(self.settings.WHISPER_MODEL_SIZE)
                logger.info("Whisper model loaded successfully.")
                self._backend_initialized = True
                return True

            elif backend == "speech_recognition":
                import speech_recognition as sr  # type: ignore[import-untyped]

                self._recognizer = sr.Recognizer()
                self._recognizer.energy_threshold = self.settings.ENERGY_THRESHOLD
                self._recognizer.dynamic_energy_threshold = self.settings.DYNAMIC_ENERGY_THRESHOLD
                self._backend_initialized = True
                logger.info("SpeechRecognition backend initialized.")
                return True

            else:
                logger.error("Unsupported STT_BACKEND: '%s'", self.settings.STT_BACKEND)
                return False

        except Exception as exc:
            logger.error("Failed to initialize STT backend '%s': %s", backend, exc)
            return False

    def _run(self) -> None:
        """Main loop executed inside the background thread."""
        if self.settings.MOCK_HARDWARE:
            logger.info("MOCK_HARDWARE enabled. Entering synthetic subtitle generation loop.")
            self._run_mock_loop()
            return

        initialized = self._initialize_backend()
        if not initialized:
            logger.warning("Falling back to synthetic subtitle test generator due to init failure.")
            self._run_mock_loop()
            return

        backend = self.settings.STT_BACKEND.lower()
        if "whisper" in backend:
            self._run_whisper_stream()
        else:
            self._run_speech_recognition_stream()

    def _run_whisper_stream(self) -> None:
        """
        Streams audio via SpeechRecognition / PyAudio microphone input and decodes
        using OpenAI Whisper.
        """
        try:
            import speech_recognition as sr  # type: ignore[import-untyped]

            rec = sr.Recognizer()
            rec.energy_threshold = self.settings.ENERGY_THRESHOLD
            rec.dynamic_energy_threshold = self.settings.DYNAMIC_ENERGY_THRESHOLD

            device_index = self.settings.AUDIO_DEVICE_INDEX

            with sr.Microphone(
                device_index=device_index, sample_rate=self.settings.AUDIO_SAMPLE_RATE
            ) as source:
                logger.info("Calibrating microphone for ambient noise...")
                rec.adjust_for_ambient_noise(source, duration=1.0)
                logger.info("Microphone calibrated. Listening for live meeting speech...")

                while not self._stop_event.is_set():
                    try:
                        # Listen with short timeout to allow responsive stop_event checks
                        audio_data = rec.listen(source, timeout=1.5, phrase_time_limit=8.0)
                        self._transcribe_whisper_chunk(audio_data)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as loop_err:
                        if not self._stop_event.is_set():
                            logger.error("Audio capture iteration error: %s", loop_err)
                            time.sleep(0.5)

        except Exception as mic_err:
            logger.error(
                "Fatal error in microphone stream: %s. Falling back to synthetic STT.", mic_err
            )
            self._run_mock_loop()

    def _transcribe_whisper_chunk(self, audio_data: Any) -> None:
        """Processes an AudioData chunk using the loaded Whisper model."""
        if self._whisper_model is None:
            return

        try:
            # Convert WAV byte payload into float32 numpy audio array for Whisper
            wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
            import wave

            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                raw_frames = wf.readframes(wf.getnframes())
                audio_np = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

            result = self._whisper_model.transcribe(
                audio_np,
                language=self.settings.STT_LANGUAGE,
                fp16=False,  # CPU friendly inside generic Docker
                without_timestamps=True,
            )

            text = result.get("text", "").strip()
            if text:
                logger.info("[STT Whisper] Transcribed: '%s'", text)
                self._enqueue_subtitle(text)

        except Exception as trans_err:
            logger.warning("Whisper transcription failed for chunk: %s", trans_err)

    def _run_speech_recognition_stream(self) -> None:
        """Streams audio using standard SpeechRecognition Google Web Speech API."""
        try:
            import speech_recognition as sr  # type: ignore[import-untyped]

            rec = self._recognizer or sr.Recognizer()
            device_index = self.settings.AUDIO_DEVICE_INDEX

            with sr.Microphone(device_index=device_index) as source:
                logger.info("SpeechRecognition listening on audio device %s...", device_index)
                rec.adjust_for_ambient_noise(source, duration=1.0)

                while not self._stop_event.is_set():
                    try:
                        audio = rec.listen(source, timeout=1.5, phrase_time_limit=6.0)
                        text = rec.recognize_google(audio, language=self.settings.STT_LANGUAGE)
                        if text:
                            logger.info("[STT Google] Transcribed: '%s'", text)
                            self._enqueue_subtitle(text)
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        # Audio unintelligible / silence
                        continue
                    except sr.RequestError as req_err:
                        logger.warning("SpeechRecognition API request failed: %s", req_err)
                        time.sleep(1.0)
                    except Exception as err:
                        if not self._stop_event.is_set():
                            logger.error("SpeechRecognition error: %s", err)

        except Exception as init_err:
            logger.error("Failed to run SpeechRecognition stream: %s. Using mock fallback.", init_err)
            self._run_mock_loop()

    def _enqueue_subtitle(self, text: str) -> None:
        """Pushes a new SubtitleItem into the thread-safe queue."""
        item = SubtitleItem(
            text=text,
            timestamp=time.time(),
            duration=self.settings.SUBTITLE_DURATION_SECONDS,
            is_final=True,
        )
        try:
            self.subtitle_queue.put_nowait(item)
        except queue.Full:
            # Discard oldest to prioritize current live subtitles
            try:
                self.subtitle_queue.get_nowait()
                self.subtitle_queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _run_mock_loop(self) -> None:
        """Simulates periodic speech subtitles when hardware/audio devices are unavailable."""
        mock_phrases = [
            "Welcome to the quarterly design sync meeting.",
            "Can everyone see my shared screen and camera feed?",
            "Let's review the sign language interpreter outputs.",
            "Meeting audio and subtitles are streaming synchronously.",
            "Thank you everyone for attending today's call.",
        ]
        phrase_idx = 0
        while not self._stop_event.is_set():
            time.sleep(5.0)
            if self._stop_event.is_set():
                break
            phrase = mock_phrases[phrase_idx % len(mock_phrases)]
            phrase_idx += 1
            logger.debug("[STT Mock] Injected subtitle: '%s'", phrase)
            self._enqueue_subtitle(f"[Live] {phrase}")
