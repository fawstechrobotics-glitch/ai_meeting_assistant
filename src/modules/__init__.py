"""
Assistive Real-Time Meeting Tool - Core Modules.
Exposes modular engines for STT, ISL, ASL, TTS, and UI Overlay.
"""

from src.modules.base import BaseSignLanguageRecognizer, GestureResult, SubtitleItem
from src.modules.stt import SpeechToTextWorker
from src.modules.isl_detection import IndianSignLanguageRecognizer
from src.modules.asl_detection import AmericanSignLanguageRecognizer
from src.modules.tts import TextToSpeechWorker
from src.modules.ui_overlay import MeetingHUDOverlay

__all__ = [
    "BaseSignLanguageRecognizer",
    "GestureResult",
    "SubtitleItem",
    "SpeechToTextWorker",
    "IndianSignLanguageRecognizer",
    "AmericanSignLanguageRecognizer",
    "TextToSpeechWorker",
    "MeetingHUDOverlay",
]
