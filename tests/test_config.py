"""
Unit tests for configuration validation and environment parsing.
"""

import os
from unittest.mock import patch

import pytest

from src.config import AppSettings, get_settings


def test_default_config() -> None:
    """Verifies that default settings instantiate with expected values."""
    settings = AppSettings()
    assert settings.APP_NAME == "Assistive Meeting Tool"
    assert settings.CAMERA_WIDTH == 1280
    assert settings.CAMERA_HEIGHT == 720
    assert settings.SIGN_LANGUAGE_MODE in ["ISL", "ASL"]
    assert settings.TTS_DEBOUNCE_SECONDS > 0
    assert 0.0 <= settings.OVERLAY_OPACITY <= 1.0


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that environment variables correctly override default settings."""
    monkeypatch.setenv("CAMERA_INDEX", "3")
    monkeypatch.setenv("SIGN_LANGUAGE_MODE", "ASL")
    monkeypatch.setenv("HEADLESS_MODE", "true")
    monkeypatch.setenv("TTS_SPEECH_RATE", "220")

    settings = AppSettings()
    assert settings.CAMERA_INDEX == 3
    assert settings.SIGN_LANGUAGE_MODE == "ASL"
    assert settings.HEADLESS_MODE is True
    assert settings.TTS_SPEECH_RATE == 220


def test_get_settings_cached() -> None:
    """Verifies lru_cache returns singleton configuration object."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
