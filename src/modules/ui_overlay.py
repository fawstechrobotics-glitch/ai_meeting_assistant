"""
Meeting HUD Overlay Module for Assistive Real-Time Meeting Tool.

Renders translucent glassmorphic HUD telemetry, real-time speech subtitles,
gesture translation cards, and status badges directly onto OpenCV video frames.
Designed for low-overhead, 30+ FPS compositing using optimized NumPy array blending.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import AppSettings, get_settings
from src.modules.base import GestureResult, SubtitleItem


class MeetingHUDOverlay:
    """
    Renders high-contrast, accessibility-optimized HUD elements onto camera frames:
    1. Top Status & Telemetry Bar (FPS, Active Mode, Audio Status)
    2. Real-Time Spoken Subtitles (STT banner for hearing-impaired users)
    3. Interpreted Sign Language Card (Confidence, Confirmation, Handedness)
    4. Hotkey Navigation Overlay
    """

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        """
        Initializes the HUD Overlay renderer.

        :param settings: Application configuration instance.
        """
        self.settings = settings or get_settings()

        # Visual theme palette (BGR format for OpenCV)
        self.COLOR_BG_DARK = (20, 20, 24)        # Deep Charcoal
        self.COLOR_PRIMARY = (255, 140, 0)       # Accent Blue / Orange
        self.COLOR_SUCCESS = (70, 200, 60)       # Emerald Green
        self.COLOR_WARNING = (0, 180, 255)       # Amber Gold
        self.COLOR_TEXT = (250, 250, 250)        # Crisp White
        self.COLOR_SUBTEXT = (180, 180, 190)     # Muted Silver
        self.COLOR_BOX = (255, 200, 50)          # Hand Box Accent
        self.COLOR_ISL = (240, 120, 40)          # Indian Sign Language Badge
        self.COLOR_ASL = (60, 160, 240)          # American Sign Language Badge

        # FPS calculation state
        self._last_frame_time = time.time()
        self._fps_history: List[float] = []
        self._current_fps: float = 30.0

    def render(
        self,
        frame: np.ndarray,
        active_mode: str,
        current_gesture: Optional[GestureResult] = None,
        active_subtitle: Optional[SubtitleItem] = None,
        stt_active: bool = True,
        tts_active: bool = True,
    ) -> np.ndarray:
        """
        Composites all UI layers onto the camera frame.

        :param frame: Raw or annotated BGR frame from camera pipeline.
        :param active_mode: Current active sign language ("ISL" or "ASL").
        :param current_gesture: Most recent GestureResult (if any).
        :param active_subtitle: Active spoken SubtitleItem (if any).
        :param stt_active: Whether STT audio capture is alive.
        :param tts_active: Whether TTS audio synthesis is alive.
        :return: Fully composited BGR frame ready for display.
        """
        self._update_fps()
        h, w, _ = frame.shape
        composite = frame.copy()

        # 1. Draw Hand Bounding Box & Target Highlight if present
        if current_gesture and current_gesture.bounding_box:
            self._draw_gesture_bounding_box(composite, current_gesture)

        # 2. Render Top Telemetry & Status Bar
        composite = self._render_top_bar(
            composite, active_mode, stt_active, tts_active, w
        )

        # 3. Render Recognized Sign Translation Card (Top Left)
        if current_gesture and current_gesture.label:
            composite = self._render_gesture_card(composite, current_gesture)

        # 4. Render Spoken Subtitle Banner (Bottom Center)
        if active_subtitle and not active_subtitle.is_expired:
            composite = self._render_subtitle_banner(composite, active_subtitle, w, h)

        # 5. Render Hotkey Guide (Bottom Right)
        self._render_hotkey_guide(composite, w, h)

        return composite

    # -------------------------------------------------------------------------
    # UI Components Rendering
    # -------------------------------------------------------------------------

    def _draw_translucent_rect(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        alpha: float,
    ) -> np.ndarray:
        """Draws a semi-transparent filled rectangle using alpha blending."""
        x1, y1 = max(0, pt1[0]), max(0, pt1[1])
        x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])

        if x1 >= x2 or y1 >= y2:
            return img

        sub_img = img[y1:y2, x1:x2]
        rect = np.full(sub_img.shape, color, dtype=np.uint8)
        blended = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 0)
        img[y1:y2, x1:x2] = blended
        return img

    def _render_top_bar(
        self,
        frame: np.ndarray,
        active_mode: str,
        stt_active: bool,
        tts_active: bool,
        width: int,
    ) -> np.ndarray:
        """Renders the top translucent application header bar."""
        bar_height = 50
        self._draw_translucent_rect(
            frame, (0, 0), (width, bar_height), self.COLOR_BG_DARK, self.settings.OVERLAY_OPACITY
        )

        # App Title
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            frame,
            self.settings.APP_NAME,
            (16, 32),
            font,
            0.65,
            self.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        # Sign Language Mode Badge
        badge_color = self.COLOR_ISL if active_mode == "ISL" else self.COLOR_ASL
        badge_x1 = 280
        badge_x2 = badge_x1 + 105
        cv2.rectangle(frame, (badge_x1, 10), (badge_x2, 40), badge_color, -1)
        cv2.putText(
            frame,
            f"MODE: {active_mode}",
            (badge_x1 + 10, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Telemetry Stats (FPS, Mic, Speaker)
        fps_text = f"FPS: {self._current_fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (badge_x2 + 25, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.COLOR_SUBTEXT,
            1,
            cv2.LINE_AA,
        )

        # Audio indicators
        mic_color = self.COLOR_SUCCESS if stt_active else (80, 80, 80)
        cv2.circle(frame, (width - 150, 25), 6, mic_color, -1)
        cv2.putText(
            frame,
            "MIC (STT)",
            (width - 138, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        spk_color = self.COLOR_SUCCESS if tts_active else (80, 80, 80)
        cv2.circle(frame, (width - 60, 25), 6, spk_color, -1)
        cv2.putText(
            frame,
            "TTS",
            (width - 48, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        return frame

    def _draw_gesture_bounding_box(
        self, frame: np.ndarray, gesture: GestureResult
    ) -> None:
        """Renders bounding box and confidence tag around detected hands."""
        if not gesture.bounding_box:
            return
        xmin, ymin, xmax, ymax = gesture.bounding_box
        box_color = self.COLOR_SUCCESS if gesture.is_confirmed else self.COLOR_WARNING

        # Bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)

        # Corner accents for a sleek HUD look
        line_len = 15
        for corner_x, corner_y, dx, dy in [
            (xmin, ymin, 1, 1),
            (xmax, ymin, -1, 1),
            (xmin, ymax, 1, -1),
            (xmax, ymax, -1, -1),
        ]:
            cv2.line(frame, (corner_x, corner_y), (corner_x + dx * line_len, corner_y), box_color, 3)
            cv2.line(frame, (corner_x, corner_y), (corner_x, corner_y + dy * line_len), box_color, 3)

    def _render_gesture_card(
        self, frame: np.ndarray, gesture: GestureResult
    ) -> np.ndarray:
        """Renders the gesture translation badge on the left side."""
        card_w = 340
        card_h = 100
        x1, y1 = 20, 65
        x2, y2 = x1 + card_w, y1 + card_h

        # Translucent backdrop
        self._draw_translucent_rect(
            frame, (x1, y1), (x2, y2), self.COLOR_BG_DARK, self.settings.OVERLAY_OPACITY
        )
        accent_color = self.COLOR_SUCCESS if gesture.is_confirmed else self.COLOR_WARNING
        cv2.rectangle(frame, (x1, y1), (x1 + 6, y2), accent_color, -1)

        # Header: Sign Language Gesture
        header_text = f"GESTURE ({gesture.language})"
        cv2.putText(
            frame,
            header_text,
            (x1 + 16, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.COLOR_SUBTEXT,
            1,
            cv2.LINE_AA,
        )

        # Gesture Label
        label_text = gesture.label
        cv2.putText(
            frame,
            label_text,
            (x1 + 16, y1 + 55),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            self.COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )

        # Confidence Bar
        bar_x = x1 + 16
        bar_y = y1 + 72
        bar_w = 200
        bar_h = 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        filled_w = int(bar_w * min(1.0, max(0.0, gesture.confidence)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), accent_color, -1)

        # Confidence % Text
        conf_str = f"{int(gesture.confidence * 100)}%"
        cv2.putText(
            frame,
            conf_str,
            (bar_x + bar_w + 10, bar_y + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.COLOR_SUBTEXT,
            1,
            cv2.LINE_AA,
        )

        # Status text (Confirmed vs Tracking)
        status_str = "CONFIRMED" if gesture.is_confirmed else "TRACKING..."
        cv2.putText(
            frame,
            status_str,
            (x2 - 100, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            accent_color,
            1,
            cv2.LINE_AA,
        )

        return frame

    def _render_subtitle_banner(
        self,
        frame: np.ndarray,
        subtitle: SubtitleItem,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Renders live spoken subtitles in a high-contrast banner at the bottom.
        Supports automatic word wrapping.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.68
        thickness = 2

        # Wrap text to fit width
        max_width_px = width - 120
        wrapped_lines = self._wrap_text(subtitle.text, font, font_scale, thickness, max_width_px)

        line_height = 32
        banner_h = max(64, len(wrapped_lines) * line_height + 34)
        y1 = height - banner_h - 40
        y2 = height - 40
        x1 = 40
        x2 = width - 40

        # Translucent glassmorphic banner
        self._draw_translucent_rect(
            frame, (x1, y1), (x2, y2), self.COLOR_BG_DARK, self.settings.OVERLAY_OPACITY + 0.1
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_PRIMARY, 1)

        # Subtitle icon / label
        cv2.putText(
            frame,
            "SUBTITLES (LIVE STT):",
            (x1 + 18, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.COLOR_PRIMARY,
            1,
            cv2.LINE_AA,
        )

        # Render wrapped subtitle lines
        text_y = y1 + 50
        for line in wrapped_lines:
            cv2.putText(
                frame,
                line,
                (x1 + 18, text_y),
                font,
                font_scale,
                self.COLOR_TEXT,
                thickness,
                cv2.LINE_AA,
            )
            text_y += line_height

        return frame

    def _render_hotkey_guide(self, frame: np.ndarray, width: int, height: int) -> None:
        """Renders keyboard shortcut assistance at bottom-right."""
        guide_text = "[M] Switch ISL/ASL | [L] Toggle Landmarks | [Q] Quit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            guide_text,
            (width - 440, height - 12),
            font,
            0.40,
            self.COLOR_SUBTEXT,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _wrap_text(
        text: str,
        font: int,
        font_scale: float,
        thickness: int,
        max_width_px: int,
    ) -> List[str]:
        """Splits a string into lines that fit within max_width_px."""
        words = text.split()
        if not words:
            return []

        lines: List[str] = []
        current_line = words[0]

        for word in words[1:]:
            test_line = f"{current_line} {word}"
            (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if w <= max_width_px:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)
        return lines

    def _update_fps(self) -> None:
        """Maintains a rolling FPS calculation."""
        current_time = time.time()
        dt = current_time - self._last_frame_time
        self._last_frame_time = current_time

        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps_history.append(instant_fps)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            self._current_fps = sum(self._fps_history) / len(self._fps_history)
