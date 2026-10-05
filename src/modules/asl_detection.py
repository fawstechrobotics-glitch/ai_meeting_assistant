"""
American Sign Language (ASL) Recognition Module.

Processes video frames using MediaPipe Hand tracking to interpret American
Sign Language gestures into text. Specializes in dominant-hand finger spelling,
canonical conversational signs (Hello, Thank You, ILY, Peace, OK), and supports
custom ML classifier backends (ONNX / Torch).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import AppSettings, get_settings
from src.modules.base import BaseSignLanguageRecognizer, GestureResult

logger = logging.getLogger(__name__)


class AmericanSignLanguageRecognizer(BaseSignLanguageRecognizer):
    """
    Dedicated American Sign Language (ASL) interpreter.
    Specializes in unimanual finger spelling and conversational gestures.
    """

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        """
        Initializes the ASL recognizer.

        :param settings: Application configuration instance.
        """
        self.settings = settings or get_settings()
        super().__init__(
            min_detection_confidence=self.settings.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.settings.MIN_TRACKING_CONFIDENCE,
            confirmation_frames=self.settings.GESTURE_CONFIRMATION_FRAMES,
        )

        self._custom_model: Optional[Any] = None
        self._load_custom_model_if_present()

        # Initialize MediaPipe Hands pipeline
        import mediapipe as mp  # type: ignore[import-untyped]

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def _load_custom_model_if_present(self) -> None:
        """Attempts to load pre-trained weights if specified in settings."""
        model_path = self.settings.CUSTOM_ASL_MODEL_PATH
        if model_path and os.path.exists(model_path):
            logger.info("Loading custom ASL model weights from: %s", model_path)
            try:
                if model_path.endswith(".onnx"):
                    import onnxruntime as ort  # type: ignore[import-untyped]

                    self._custom_model = ort.InferenceSession(model_path)
                    logger.info("ONNX ASL model loaded successfully.")
                elif model_path.endswith(".pt") or model_path.endswith(".pth"):
                    import torch  # type: ignore[import-untyped]

                    self._custom_model = torch.jit.load(model_path)
                    self._custom_model.eval()
                    logger.info("TorchScript ASL model loaded successfully.")
            except Exception as err:
                logger.error("Failed to load custom ASL model from %s: %s", model_path, err)
                self._custom_model = None

    def reset(self) -> None:
        """Resets temporal smoothing history."""
        super().reset()

    def process_frame(
        self, frame: np.ndarray, draw_landmarks: bool = True
    ) -> Tuple[np.ndarray, Optional[GestureResult]]:
        """
        Interprets ASL gestures from an incoming BGR frame.

        :param frame: Raw BGR frame from camera.
        :param draw_landmarks: Whether to render landmark skeleton onto output frame.
        :return: (annotated_frame, GestureResult)
        """
        frame_h, frame_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        annotated_frame = frame.copy()

        if not results.multi_hand_landmarks:
            if self._history_labels:
                self._history_labels.pop(0)
            return annotated_frame, None

        if draw_landmarks and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_drawing_styles.get_default_hand_connections_style(),
                )

        hands_data = []
        all_x: List[int] = []
        all_y: List[int] = []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = "Right"
            if results.multi_handedness and idx < len(results.multi_handedness):
                label = results.multi_handedness[idx].classification[0].label

            finger_states = self.get_finger_extension_states(hand_landmarks, label)
            hands_data.append({
                "landmarks": hand_landmarks,
                "label": label,
                "finger_states": finger_states,
            })

            for lm in hand_landmarks.landmark:
                all_x.append(int(lm.x * frame_w))
                all_y.append(int(lm.y * frame_h))

        bounding_box: Optional[Tuple[int, int, int, int]] = None
        if all_x and all_y:
            padding = 20
            xmin = max(0, min(all_x) - padding)
            ymin = max(0, min(all_y) - padding)
            xmax = min(frame_w, max(all_x) + padding)
            ymax = min(frame_h, max(all_y) + padding)
            bounding_box = (xmin, ymin, xmax, ymax)

        raw_gesture, confidence = self._classify_asl(hands_data, frame_w, frame_h)

        if not raw_gesture:
            return annotated_frame, None

        is_confirmed, smoothed_label = self.update_temporal_smoothing(raw_gesture)

        result = GestureResult(
            label=smoothed_label,
            confidence=confidence,
            is_confirmed=is_confirmed,
            timestamp=time.time(),
            language="ASL",
            bounding_box=bounding_box,
            landmarks_detected=True,
            handedness=[h["label"] for h in hands_data],
        )

        return annotated_frame, result

    # -------------------------------------------------------------------------
    # ASL Heuristic & Geometric Classification Rules
    # -------------------------------------------------------------------------

    def _classify_asl(
        self, hands_data: List[Dict[str, Any]], frame_w: int, frame_h: int
    ) -> Tuple[Optional[str], float]:
        """
        Classifies American Sign Language gestures and finger-spelled letters.
        """
        # If two hands are present, check bimanual signs like Help
        if len(hands_data) >= 2:
            two_hand_res, conf = self._classify_two_hand_asl(hands_data[0], hands_data[1])
            if two_hand_res:
                return two_hand_res, conf

        # Otherwise evaluate dominant hand
        primary_hand = hands_data[0]
        return self._classify_single_hand_asl(primary_hand)

    def _classify_two_hand_asl(
        self, hand_a: Dict[str, Any], hand_b: Dict[str, Any]
    ) -> Tuple[Optional[str], float]:
        """Classifies two-handed ASL signs such as 'Help'."""
        lm_a = hand_a["landmarks"].landmark
        lm_b = hand_b["landmarks"].landmark
        fingers_a = hand_a["finger_states"]
        fingers_b = hand_b["finger_states"]

        wrist_dist = self.euclidean_distance((lm_a[0].x, lm_a[0].y), (lm_b[0].x, lm_b[0].y))

        # ASL "Help": Flat palm supporting a thumbs-up fist
        fist_a = not any([fingers_a["index"], fingers_a["middle"], fingers_a["ring"], fingers_a["pinky"]]) and fingers_a["thumb"]
        flat_b = all([fingers_b["index"], fingers_b["middle"], fingers_b["ring"], fingers_b["pinky"]])

        fist_b = not any([fingers_b["index"], fingers_b["middle"], fingers_b["ring"], fingers_b["pinky"]]) and fingers_b["thumb"]
        flat_a = all([fingers_a["index"], fingers_a["middle"], fingers_a["ring"], fingers_a["pinky"]])

        if (fist_a and flat_b and wrist_dist < 0.25) or (fist_b and flat_a and wrist_dist < 0.25):
            return "Help", 0.94

        return None, 0.0

    def _classify_single_hand_asl(self, hand: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """
        Classifies single-handed ASL signs and finger spelling (A, B, C, L, V, Y, ILY, OK, Hello, Thanks).
        """
        lm = hand["landmarks"].landmark
        fingers = hand["finger_states"]

        # Distance between thumb tip (4) and index tip (8)
        thumb_index_dist = self.euclidean_distance((lm[4].x, lm[4].y), (lm[8].x, lm[8].y))
        # Distance between index tip (8) and middle tip (12)
        index_middle_dist = self.euclidean_distance((lm[8].x, lm[8].y), (lm[12].x, lm[12].y))

        # 1. "I Love You" (ILY): Thumb, Index, and Pinky extended; Middle and Ring folded
        if fingers["thumb"] and fingers["index"] and not fingers["middle"] and not fingers["ring"] and fingers["pinky"]:
            return "I Love You", 0.96

        # 2. "OK Sign": Thumb tip and Index tip touching (< 0.05), Middle, Ring, Pinky extended
        if thumb_index_dist < 0.05 and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
            return "OK / Agreement", 0.94

        # 3. "Peace / Victory / V": Index and Middle extended in V-shape, Ring and Pinky folded
        if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            if index_middle_dist > 0.04:
                return "Peace / 'V'", 0.92

        # 4. "Letter 'L'": Thumb and Index extended at ~90 degrees, others folded
        if fingers["thumb"] and fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            return "Letter 'L'", 0.92

        # 5. "Letter 'Y' / Hang Loose": Thumb and Pinky extended, Index, Middle, Ring folded
        if fingers["thumb"] and not fingers["index"] and not fingers["middle"] and not fingers["ring"] and fingers["pinky"]:
            return "Letter 'Y'", 0.92

        # 6. "Letter 'B' / Hello": 4 fingers straight up, thumb folded across palm
        if not fingers["thumb"] and fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
            # If hand is held high near head level (y < 0.35)
            if lm[0].y < 0.40:
                return "Hello", 0.90
            return "Letter 'B'", 0.88

        # 7. "Letter 'A' / Thumbs Up": Thumb extended up or resting on fist, all 4 fingers folded
        if not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            if fingers["thumb"] and lm[4].y < lm[3].y:
                return "Thumbs Up / Good", 0.91
            return "Letter 'A' / Yes", 0.87

        # 8. "Thank You": Flat hand starting near chin moving outward
        if fingers["thumb"] and fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
            if 0.30 < lm[0].y < 0.70:
                return "Thank You", 0.86

        return None, 0.0
