"""
Unit tests for Speech-to-Text (STT) worker.
"""

import queue
import time
from unittest.mock import MagicMock, patch

import pytest

from src.config import AppSettings
from src.modules.base import SubtitleItem
from src.modules.stt import SpeechToTextWorker


def test_stt_worker_enqueue_subtitle() -> None:
    """Tests enqueueing subtitles and expiration checks."""
    sub_queue: queue.Queue[SubtitleItem] = queue.Queue(maxsize=10)
    settings = AppSettings(SUBTITLE_DURATION_SECONDS=2.0)
    worker = SpeechToTextWorker(settings=settings, subtitle_queue=sub_queue)

    worker._enqueue_subtitle("Test subtitle for hearing-impaired users")

    assert not sub_queue.empty()
    item = sub_queue.get_nowait()
    assert item.text == "Test subtitle for hearing-impaired users"
    assert not item.is_expired
    assert item.is_final is True


def test_stt_worker_mock_loop_lifecycle() -> None:
    """Tests starting and stopping the STT worker in mock mode."""
    sub_queue: queue.Queue[SubtitleItem] = queue.Queue(maxsize=10)
    settings = AppSettings(MOCK_HARDWARE=True)
    worker = SpeechToTextWorker(settings=settings, subtitle_queue=sub_queue)

    assert not worker.is_running()
    worker.start()
    assert worker.is_running()

    time.sleep(0.1)
    worker.stop(timeout=1.0)
    assert not worker.is_running()


def test_stt_queue_overflow_discard_oldest() -> None:
    """Verifies that full queues discard older items to keep live subtitles current."""
    sub_queue: queue.Queue[SubtitleItem] = queue.Queue(maxsize=2)
    settings = AppSettings()
    worker = SpeechToTextWorker(settings=settings, subtitle_queue=sub_queue)

    worker._enqueue_subtitle("First")
    worker._enqueue_subtitle("Second")
    worker._enqueue_subtitle("Third")

    assert sub_queue.qsize() == 2
    item1 = sub_queue.get_nowait()
    item2 = sub_queue.get_nowait()
    assert item1.text == "Second"
    assert item2.text == "Third"
