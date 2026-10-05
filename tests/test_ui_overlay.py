"""
Unit tests for Meeting HUD Overlay compositing and text rendering.
"""

import numpy as np
import pytest

from src.config import AppSettings
from src.modules.base import GestureResult, SubtitleItem
from src.modules.ui_overlay import MeetingHUDOverlay


def test_hud_overlay_render_clean_frame() -> None:
    """Tests that overlay renders without raising errors on empty inputs."""
    settings = AppSettings()
    overlay = MeetingHUDOverlay(settings=settings)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = overlay.render(
        frame=frame,
        active_mode="ISL",
        current_gesture=None,
        active_subtitle=None,
        stt_active=True,
        tts_active=True,
    )

    assert rendered.shape == (720, 1280, 3)
    # Ensure pixels have been modified (e.g. top bar rendered)
    assert not np.array_equal(frame, rendered)


def test_hud_overlay_render_with_gesture_and_subtitles() -> None:
    """Tests that overlay properly incorporates gesture cards, bounding boxes, and subtitles."""
    settings = AppSettings()
    overlay = MeetingHUDOverlay(settings=settings)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    gesture = GestureResult(
        label="Namaste / Hello",
        confidence=0.95,
        is_confirmed=True,
        language="ISL",
        bounding_box=(200, 200, 400, 400),
        landmarks_detected=True,
    )

    subtitle = SubtitleItem(
        text="Welcome to the quarterly design sync meeting for accessibility.",
        duration=5.0,
    )

    rendered = overlay.render(
        frame=frame,
        active_mode="ISL",
        current_gesture=gesture,
        active_subtitle=subtitle,
        stt_active=True,
        tts_active=True,
    )

    assert rendered.shape == (720, 1280, 3)


def test_text_wrapping() -> None:
    """Verifies that long subtitle strings are wrapped across multiple lines."""
    long_text = "This is a very long meeting subtitle sentence designed to test whether the automatic word wrapping function will properly partition lines without overflowing the video display width boundary."
    wrapped = MeetingHUDOverlay._wrap_text(
        text=long_text,
        font=0,  # cv2.FONT_HERSHEY_SIMPLEX
        font_scale=0.6,
        thickness=1,
        max_width_px=300,
    )

    assert len(wrapped) > 1
    # Check that reconstructed string preserves all words
    reconstructed = " ".join(wrapped)
    assert reconstructed == long_text
