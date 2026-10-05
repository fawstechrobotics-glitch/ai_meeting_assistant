"""
Abstract Base Classes and Data Structures for Assistive Meeting Tool.

Defines standardized data contracts, base recognizers, and mathematical utilities
shared across ISL, ASL, and audio/video pipelines.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GestureResult:
    """
    Encapsulates a recognized sign language gesture and its detection metadata.
    """
    label: str
    confidence: float
    is_confirmed: bool
    timestamp: float = field(default_factory=time.time)
    language: str = "ISL"  # "ISL" or "ASL"
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (xmin, ymin, xmax, ymax) in pixels
    landmarks_detected: bool = False
    handedness: List[str] = field(default_factory=list)  # e.g., ["Left", "Right"]


@dataclass
class SubtitleItem:
    """
    Encapsulates real-time transcribed speech text for HUD rendering.
    """
    text: str
    timestamp: float = field(default_factory=time.time)
    duration: float = 4.0  # Duration in seconds to keep subtitle visible
    is_final: bool = True
    speaker_id: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Determines if the subtitle has exceeded its visibility lifespan."""
        return (time.time() - self.timestamp) > self.duration


class BaseSignLanguageRecognizer(ABC):
    """
    Abstract Base Class for Sign Language Recognizers (ISL and ASL).
    Standardizes feature extraction, landmark spatial normalization, and frame processing.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        confirmation_frames: int = 6,
    ) -> None:
        """
        Initializes the base sign recognizer.

        :param min_detection_confidence: Confidence threshold for MediaPipe hand/pose detection.
        :param min_tracking_confidence: Confidence threshold for MediaPipe landmark tracking.
        :param confirmation_frames: Rolling consecutive frames required to debounce & confirm a gesture.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.confirmation_frames = confirmation_frames

        # Temporal smoothing buffer
        self._history_labels: List[str] = []
        self._last_confirmed_label: Optional[str] = None
        self._last_confirmed_time: float = 0.0

    @abstractmethod
    def process_frame(
        self, frame: np.ndarray, draw_landmarks: bool = True
    ) -> Tuple[np.ndarray, Optional[GestureResult]]:
        """
        Processes a raw BGR frame, extracts landmarks, runs classification, and returns
        the annotated frame along with the recognized gesture result.

        :param frame: Raw input BGR image from camera (OpenCV format).
        :param draw_landmarks: If True, draws skeleton and landmarks on the frame.
        :return: (annotated_frame, GestureResult)
        """
        raise NotImplementedError("Subclasses must implement process_frame()")

    @abstractmethod
    def reset(self) -> None:
        """Clears temporal state and smoothing buffers."""
        self._history_labels.clear()
        self._last_confirmed_label = None
        self._last_confirmed_time = 0.0

    # -------------------------------------------------------------------------
    # Shared Mathematical & Landmark Feature Extraction Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def euclidean_distance(
        pt1: Tuple[float, float, float] | Tuple[float, float],
        pt2: Tuple[float, float, float] | Tuple[float, float],
    ) -> float:
        """
        Computes 2D or 3D Euclidean distance between two landmark coordinates.
        """
        if len(pt1) == 2 or len(pt2) == 2:
            return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
        return math.sqrt(
            (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2
        )

    @staticmethod
    def normalize_landmarks_to_wrist(landmarks_list: List[Any]) -> np.ndarray:
        """
        Normalizes a set of 21 hand landmarks relative to the wrist (landmark 0),
        scaling by the palm size to ensure translation- and scale-invariance.

        :param landmarks_list: List of 21 MediaPipe hand landmark points.
        :return: (21, 3) normalized NumPy array.
        """
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list], dtype=np.float32)
        wrist = coords[0]
        # Translate wrist to origin
        translated = coords - wrist
        # Scale by distance between wrist (0) and middle finger MCP (9)
        palm_size = np.linalg.norm(translated[9])
        if palm_size > 1e-6:
            normalized = translated / palm_size
        else:
            normalized = translated
        return normalized

    @classmethod
    def get_finger_extension_states(
        cls, hand_landmarks: Any, handedness_label: str = "Right"
    ) -> Dict[str, bool]:
        """
        Determines whether each finger (Thumb, Index, Middle, Ring, Pinky) is extended or curled.

        MediaPipe Hand Landmark Indices:
        - Wrist: 0
        - Thumb: [1, 2, 3, 4]
        - Index: [5, 6, 7, 8]
        - Middle: [9, 10, 11, 12]
        - Ring: [13, 14, 15, 16]
        - Pinky: [17, 18, 19, 20]

        :param hand_landmarks: MediaPipe NormalizedLandmarkList.
        :param handedness_label: "Left" or "Right".
        :return: Dict mapping finger name to boolean (True if extended, False if folded).
        """
        lm = hand_landmarks.landmark

        # For fingers other than thumb: Tip (y) is above PIP (y) in screen space (y increases downward)
        index_extended = lm[8].y < lm[6].y
        middle_extended = lm[12].y < lm[10].y
        ring_extended = lm[16].y < lm[14].y
        pinky_extended = lm[20].y < lm[18].y

        # Thumb extension: depends on handedness. For right hand facing camera, thumb tip is left of IP
        if handedness_label == "Right":
            thumb_extended = lm[4].x < lm[3].x
        else:
            thumb_extended = lm[4].x > lm[3].x

        return {
            "thumb": thumb_extended,
            "index": index_extended,
            "middle": middle_extended,
            "ring": ring_extended,
            "pinky": pinky_extended,
        }

    def update_temporal_smoothing(self, current_label: str) -> Tuple[bool, str]:
        """
        Maintains a sliding window of recent predictions to reject transient noise.

        :param current_label: Detected gesture label in current frame.
        :return: (is_confirmed, smoothed_label)
        """
        self._history_labels.append(current_label)
        if len(self._history_labels) > self.confirmation_frames:
            self._history_labels.pop(0)

        # Count occurrences in the sliding window
        counts: Dict[str, int] = {}
        for l in self._history_labels:
            counts[l] = counts.get(l, 0) + 1

        most_common_label, count = max(counts.items(), key=lambda item: item[1])

        # Require majority agreement in window
        if count >= (self.confirmation_frames * 0.7):
            return True, most_common_label

        return False, current_label
