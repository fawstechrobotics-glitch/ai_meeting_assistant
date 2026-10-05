"""
Configuration Module for Assistive Real-Time Meeting Tool.

Provides strongly-typed, validated configuration management using Pydantic Settings
with automatic fallback to python-dotenv and environment variable parsing.
Adheres to 12-factor application standards: no hardcoded ports, devices, or paths.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

# Preload .env if it exists in the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
else:
    # Also check current working directory
    load_dotenv()

try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class AppSettings(BaseSettings):
        """
        Validated Application Settings schema using Pydantic BaseSettings.
        """
        model_config = SettingsConfigDict(
            env_file=str(ENV_FILE) if ENV_FILE.exists() else None,
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )

        # Application Execution Settings
        APP_NAME: str = Field(default="Assistive Meeting Tool", description="Human-readable application name")
        LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
            default="INFO", description="Standard logging verbosity"
        )
        HEADLESS_MODE: bool = Field(
            default=False, description="If true, disables graphical GUI display loops (useful in CI or test)"
        )
        MOCK_HARDWARE: bool = Field(
            default=False, description="If true, generates synthetic video and audio streams when devices are absent"
        )

        # Video & Camera Device Settings
        CAMERA_INDEX: int = Field(default=0, description="V4L2 camera device index or video stream source")
        CAMERA_WIDTH: int = Field(default=1280, ge=320, le=3840, description="Frame capture width")
        CAMERA_HEIGHT: int = Field(default=720, ge=240, le=2160, description="Frame capture height")
        CAMERA_FPS: int = Field(default=30, ge=1, le=120, description="Target capture frames per second")
        SIGN_LANGUAGE_MODE: Literal["ISL", "ASL"] = Field(
            default="ISL", description="Active sign language interpreter (ISL or ASL)"
        )

        # Audio & Microphone Device Settings
        AUDIO_DEVICE_INDEX: Optional[int] = Field(
            default=None, description="System audio input device index (None for system default)"
        )
        AUDIO_SAMPLE_RATE: int = Field(default=16000, description="Audio sampling rate in Hertz")
        AUDIO_CHUNK_SIZE: int = Field(default=1024, description="Audio buffer chunk size")
        ENERGY_THRESHOLD: int = Field(default=300, description="SpeechRecognition ambient energy threshold")
        DYNAMIC_ENERGY_THRESHOLD: bool = Field(
            default=True, description="Enable dynamic background noise adaptation"
        )

        # Speech-to-Text (STT) Settings
        STT_BACKEND: Literal["whisper", "faster-whisper", "speech_recognition"] = Field(
            default="whisper", description="STT inference backend"
        )
        WHISPER_MODEL_SIZE: Literal["tiny", "base", "small", "medium", "large-v3"] = Field(
            default="tiny", description="Model footprint for Whisper STT"
        )
        STT_LANGUAGE: str = Field(default="en", description="Target language code for STT engine")
        SUBTITLE_DURATION_SECONDS: float = Field(
            default=4.0, ge=1.0, le=30.0, description="Time in seconds to retain a subtitle banner on HUD"
        )

        # Text-to-Speech (TTS) Settings
        TTS_ENGINE: Literal["pyttsx3", "gtts"] = Field(
            default="pyttsx3", description="Audio synthesis backend"
        )
        TTS_SPEECH_RATE: int = Field(default=160, ge=80, le=300, description="Speech rate in words per minute")
        TTS_VOLUME: float = Field(default=1.0, ge=0.0, le=1.0, description="Audio output gain")
        TTS_DEBOUNCE_SECONDS: float = Field(
            default=2.5, ge=0.5, le=10.0, description="Cooldown before repeating identical gesture speech"
        )

        # MediaPipe & Gesture Recognition Thresholds
        MIN_DETECTION_CONFIDENCE: float = Field(
            default=0.7, ge=0.1, le=1.0, description="MediaPipe minimum landmark detection confidence"
        )
        MIN_TRACKING_CONFIDENCE: float = Field(
            default=0.6, ge=0.1, le=1.0, description="MediaPipe minimum tracking confidence"
        )
        GESTURE_CONFIRMATION_FRAMES: int = Field(
            default=6, ge=1, le=30, description="Consecutive frames required to finalize a gesture"
        )
        CUSTOM_ISL_MODEL_PATH: Optional[str] = Field(
            default=None, description="Optional path to custom-trained ISL model weights (.onnx or .pt)"
        )
        CUSTOM_ASL_MODEL_PATH: Optional[str] = Field(
            default=None, description="Optional path to custom-trained ASL model weights (.onnx or .pt)"
        )

        # Display & Overlay Settings
        WINDOW_TITLE: str = Field(
            default="Assistive Meeting Assistant - Live Subtitles & Sign Translation",
            description="OpenCV window caption",
        )
        DISPLAY_LANDMARKS: bool = Field(
            default=True, description="Render skeletal landmarks and hand connections"
        )
        OVERLAY_OPACITY: float = Field(
            default=0.75, ge=0.0, le=1.0, description="Alpha blending opacity for UI overlay rectangles"
        )
        FONT_SCALE: float = Field(default=0.7, ge=0.3, le=2.0, description="OpenCV font scale multiplier")

        # Web Streaming Server Settings (for Browser viewing on macOS / Docker)
        ENABLE_WEB_STREAM: bool = Field(
            default=True, description="Serve live video and HUD overlay over HTTP for browser viewing"
        )
        WEB_STREAM_PORT: int = Field(
            default=5000, ge=1024, le=65535, description="Port for the embedded web stream server"
        )

        @field_validator("CUSTOM_ISL_MODEL_PATH", "CUSTOM_ASL_MODEL_PATH", "AUDIO_DEVICE_INDEX", mode="before")
        @classmethod
        def empty_str_to_none(cls, v: Any) -> Optional[Any]:
            if isinstance(v, str) and not v.strip():
                return None
            return v

except ImportError:
    # Graceful dataclass fallback if pydantic / pydantic_settings are not yet installed
    from dataclasses import dataclass

    @dataclass
    class AppSettings:  # type: ignore[no-redef]
        """Fallback AppSettings for lightweight environments."""
        APP_NAME: str = os.getenv("APP_NAME", "Assistive Meeting Tool")
        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        HEADLESS_MODE: bool = os.getenv("HEADLESS_MODE", "false").lower() in ("true", "1")
        MOCK_HARDWARE: bool = os.getenv("MOCK_HARDWARE", "false").lower() in ("true", "1")
        CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
        CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1280"))
        CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "720"))
        CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", "30"))
        SIGN_LANGUAGE_MODE: str = os.getenv("SIGN_LANGUAGE_MODE", "ISL")
        AUDIO_DEVICE_INDEX: Optional[int] = (
            int(os.getenv("AUDIO_DEVICE_INDEX")) if os.getenv("AUDIO_DEVICE_INDEX") else None
        )
        AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        AUDIO_CHUNK_SIZE: int = int(os.getenv("AUDIO_CHUNK_SIZE", "1024"))
        ENERGY_THRESHOLD: int = int(os.getenv("ENERGY_THRESHOLD", "300"))
        DYNAMIC_ENERGY_THRESHOLD: bool = os.getenv("DYNAMIC_ENERGY_THRESHOLD", "true").lower() in ("true", "1")
        STT_BACKEND: str = os.getenv("STT_BACKEND", "whisper")
        WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "tiny")
        STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "en")
        SUBTITLE_DURATION_SECONDS: float = float(os.getenv("SUBTITLE_DURATION_SECONDS", "4.0"))
        TTS_ENGINE: str = os.getenv("TTS_ENGINE", "pyttsx3")
        TTS_SPEECH_RATE: int = int(os.getenv("TTS_SPEECH_RATE", "160"))
        TTS_VOLUME: float = float(os.getenv("TTS_VOLUME", "1.0"))
        TTS_DEBOUNCE_SECONDS: float = float(os.getenv("TTS_DEBOUNCE_SECONDS", "2.5"))
        MIN_DETECTION_CONFIDENCE: float = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.7"))
        MIN_TRACKING_CONFIDENCE: float = float(os.getenv("MIN_TRACKING_CONFIDENCE", "0.6"))
        GESTURE_CONFIRMATION_FRAMES: int = int(os.getenv("GESTURE_CONFIRMATION_FRAMES", "6"))
        CUSTOM_ISL_MODEL_PATH: Optional[str] = os.getenv("CUSTOM_ISL_MODEL_PATH") or None
        CUSTOM_ASL_MODEL_PATH: Optional[str] = os.getenv("CUSTOM_ASL_MODEL_PATH") or None
        WINDOW_TITLE: str = os.getenv(
            "WINDOW_TITLE", "Assistive Meeting Assistant - Live Subtitles & Sign Translation"
        )
        DISPLAY_LANDMARKS: bool = os.getenv("DISPLAY_LANDMARKS", "true").lower() in ("true", "1")
        OVERLAY_OPACITY: float = float(os.getenv("OVERLAY_OPACITY", "0.75"))
        FONT_SCALE: float = float(os.getenv("FONT_SCALE", "0.7"))
        ENABLE_WEB_STREAM: bool = os.getenv("ENABLE_WEB_STREAM", "true").lower() in ("true", "1")
        WEB_STREAM_PORT: int = int(os.getenv("WEB_STREAM_PORT", "5000"))


@lru_cache()
def get_settings() -> AppSettings:
    """
    Returns a cached singleton instance of validated AppSettings.
    """
    return AppSettings()
