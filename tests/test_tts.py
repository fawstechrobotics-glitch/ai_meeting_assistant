"""
Unit tests for Text-to-Speech (TTS) worker and debounce mechanics.
"""

import queue
import time

import pytest

from src.config import AppSettings
from src.modules.tts import TextToSpeechWorker


def test_tts_enqueue_and_debounce() -> None:
    """Verifies that speech requests within debounce cooldown are suppressed."""
    settings = AppSettings(TTS_DEBOUNCE_SECONDS=2.0)
    speech_queue: queue.Queue[str] = queue.Queue(maxsize=10)
    worker = TextToSpeechWorker(settings=settings, speech_queue=speech_queue)

    # First utterance should succeed
    enqueued = worker.enqueue_speech("Namaste")
    assert enqueued is True
    assert speech_queue.qsize() == 1

    # Immediate second utterance of same phrase should be rejected by debounce
    enqueued_repeat = worker.enqueue_speech("Namaste")
    assert enqueued_repeat is False
    assert speech_queue.qsize() == 1

    # Utterance of different phrase should succeed
    enqueued_diff = worker.enqueue_speech("Thank You")
    assert enqueued_diff is True
    assert speech_queue.qsize() == 2

    # Forced speech should bypass debounce
    enqueued_forced = worker.enqueue_speech("Namaste", force=True)
    assert enqueued_forced is True
    assert speech_queue.qsize() == 3


def test_tts_worker_lifecycle_in_mock_mode() -> None:
    """Verifies starting, synthesis, and stopping of TTS worker thread."""
    settings = AppSettings(MOCK_HARDWARE=True)
    speech_queue: queue.Queue[str] = queue.Queue(maxsize=10)
    worker = TextToSpeechWorker(settings=settings, speech_queue=speech_queue)

    assert not worker.is_running()
    worker.start()
    assert worker.is_running()

    worker.enqueue_speech("Testing audio output")
    time.sleep(0.1)

    worker.stop(timeout=1.0)
    assert not worker.is_running()
