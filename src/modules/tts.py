"""
Text-to-Speech (TTS) Module for Assistive Real-Time Meeting Tool.

Provides an asynchronous, non-blocking audio synthesis worker thread. Converts
interpreted sign language gestures into synthetic voice using pyttsx3 (offline)
or gTTS (online), incorporating debouncing and rate/volume controls.
"""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
from typing import Optional

from src.config import AppSettings, get_settings

logger = logging.getLogger(__name__)


class TextToSpeechWorker:
    """
    Background worker that consumes translated sign phrases from a queue,
    debounces rapid repetitions, and voices them aloud via pyttsx3 or gTTS.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        speech_queue: Optional[queue.Queue[str]] = None,
    ) -> None:
        """
        Initializes the TTS worker.

        :param settings: Application configuration instance.
        :param speech_queue: Thread-safe queue containing strings to synthesize.
        """
        self.settings = settings or get_settings()
        self.speech_queue = speech_queue if speech_queue is not None else queue.Queue(maxsize=50)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Debouncing tracker: {phrase: last_spoken_timestamp}
        self._last_spoken_times: dict[str, float] = {}

        # Engine instance
        self._pyttsx_engine: Optional[Any] = None

    def start(self) -> None:
        """Starts the background audio synthesis consumer thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("TTS Worker thread is already active.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="TTSWorkerThread", daemon=True)
        self._thread.start()
        logger.info(
            "TTS Worker thread launched using engine: '%s' (rate: %d, vol: %.1f)",
            self.settings.TTS_ENGINE,
            self.settings.TTS_SPEECH_RATE,
            self.settings.TTS_VOLUME,
        )

    def stop(self, timeout: float = 2.0) -> None:
        """Signals the TTS worker thread to terminate and cleans up resources."""
        logger.info("Stopping TTS Worker...")
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            logger.info("TTS Worker stopped.")
        self._thread = None

    def enqueue_speech(self, text: str, force: bool = False) -> bool:
        """
        Enqueues text for spoken synthesis if it satisfies debounce constraints.

        :param text: Text phrase to speak aloud.
        :param force: If True, bypasses debounce cooldown timer.
        :return: True if enqueued, False if rejected by debounce.
        """
        normalized_text = text.strip()
        if not normalized_text:
            return False

        current_time = time.time()
        last_time = self._last_spoken_times.get(normalized_text, 0.0)

        # Check debounce window
        if not force and (current_time - last_time) < self.settings.TTS_DEBOUNCE_SECONDS:
            logger.debug("[TTS Debounce] Skipping recently voiced phrase: '%s'", normalized_text)
            return False

        try:
            self.speech_queue.put_nowait(normalized_text)
            self._last_spoken_times[normalized_text] = current_time
            return True
        except queue.Full:
            logger.warning("TTS queue is full. Discarding speech request for: '%s'", normalized_text)
            return False

    def is_running(self) -> bool:
        """Returns True if the worker thread is active."""
        return self._thread is not None and self._thread.is_alive()

    # -------------------------------------------------------------------------
    # Internal Synthesis Pipeline
    # -------------------------------------------------------------------------

    def _init_pyttsx3(self) -> bool:
        """Initializes the pyttsx3 engine with configured rate and volume."""
        try:
            import pyttsx3  # type: ignore[import-untyped]

            self._pyttsx_engine = pyttsx3.init()
            self._pyttsx_engine.setProperty("rate", self.settings.TTS_SPEECH_RATE)
            self._pyttsx_engine.setProperty("volume", self.settings.TTS_VOLUME)
            return True
        except Exception as err:
            logger.warning("Failed to initialize pyttsx3: %s. Will fallback to gTTS/mock.", err)
            self._pyttsx_engine = None
            return False

    def _run(self) -> None:
        """Main loop consuming phrases from the speech queue."""
        if self.settings.TTS_ENGINE.lower() == "pyttsx3":
            self._init_pyttsx3()

        while not self._stop_event.is_set():
            try:
                # 0.5s timeout allows responsive check of _stop_event
                text = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._speak(text)
            except Exception as speak_err:
                logger.error("Error during speech synthesis of '%s': %s", text, speak_err)
            finally:
                self.speech_queue.task_done()

    def _speak(self, text: str) -> None:
        """Synthesizes text using the designated engine or mock fallback."""
        if self.settings.MOCK_HARDWARE:
            logger.info("[TTS Mock Speak] Voice output: '%s'", text)
            return

        engine_type = self.settings.TTS_ENGINE.lower()

        # Engine 1: pyttsx3 (offline, fast native speech)
        if engine_type == "pyttsx3" and self._pyttsx_engine is not None:
            try:
                logger.info("[TTS Speak pyttsx3] '%s'", text)
                self._pyttsx_engine.say(text)
                self._pyttsx_engine.runAndWait()
                return
            except Exception as p_err:
                logger.warning("pyttsx3 runtime error: %s. Falling back to gTTS.", p_err)

        # Engine 2: gTTS (Google Text-to-Speech)
        try:
            from gtts import gTTS  # type: ignore[import-untyped]

            logger.info("[TTS Speak gTTS] Generating audio for: '%s'", text)
            tts = gTTS(text=text, lang="en", slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                temp_filename = tf.name
                tts.save(temp_filename)

            # Play mp3 file using available system utility
            self._play_audio_file(temp_filename)

            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        except Exception as g_err:
            logger.error("[TTS Fallback Log] Spoken gesture: '%s' (Error: %s)", text, g_err)

    @staticmethod
    def _play_audio_file(filepath: str) -> None:
        """Attempts to play an audio file using available Linux/Mac players."""
        import subprocess

        for player in ["mpg123", "ffplay", "afplay", "aplay"]:
            try:
                if player == "ffplay":
                    cmd = ["ffplay", "-nodisp", "-autoexit", filepath]
                elif player == "afplay":
                    cmd = ["afplay", filepath]
                else:
                    cmd = [player, filepath]

                subprocess.run(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )
                return
            except FileNotFoundError:
                continue
            except Exception:
                continue
