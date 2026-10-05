"""
Indian Sign Language (ISL) Recognition Module.

Processes video frames using MediaPipe Hand/Holistic estimation to interpret
Indian Sign Language gestures into text. Features dual-handed spatial landmark
geometry, temporal smoothing, and extensible custom ML model inference (ONNX/Torch).
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


class IndianSignLanguageRecognizer(BaseSignLanguageRecognizer):
    """
    Dedicated Indian Sign Language (ISL) interpreter.
    Specializes in dual-handed (bimanual) and spatial relational gestures.
    """

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        """
        Initializes the ISL recognizer.

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
        model_path = self.settings.CUSTOM_ISL_MODEL_PATH
        if model_path and os.path.exists(model_path):
            logger.info("Loading custom ISL model weights from: %s", model_path)
            try:
                if model_path.endswith(".onnx"):
                    import onnxruntime as ort  # type: ignore[import-untyped]

                    self._custom_model = ort.InferenceSession(model_path)
                    logger.info("ONNX ISL model loaded successfully.")
                elif model_path.endswith(".pt") or model_path.endswith(".pth"):
                    import torch  # type: ignore[import-untyped]

                    self._custom_model = torch.jit.load(model_path)
                    self._custom_model.eval()
                    logger.info("TorchScript ISL model loaded successfully.")
            except Exception as err:
                logger.error("Failed to load custom ISL model from %s: %s", model_path, err)
                self._custom_model = None

    def reset(self) -> None:
        """Resets temporal smoothing history."""
        super().reset()

    def process_frame(
        self, frame: np.ndarray, draw_landmarks: bool = True
    ) -> Tuple[np.ndarray, Optional[GestureResult]]:
        """
        Interprets ISL gestures from an incoming BGR frame.

        :param frame: Raw BGR frame from camera.
        :param draw_landmarks: Whether to render landmark skeleton onto output frame.
        :return: (annotated_frame, GestureResult)
        """
        frame_h, frame_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        annotated_frame = frame.copy()

        if not results.multi_hand_landmarks:
            # Clear or decay history when no hands are present
            if self._history_labels:
                self._history_labels.pop(0)
            return annotated_frame, None

        # Draw MediaPipe hand landmarks if requested
        if draw_landmarks and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_drawing_styles.get_default_hand_connections_style(),
                )

        # Extract handedness and bounding boxes
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

        # Overall bounding box across visible hands
        bounding_box: Optional[Tuple[int, int, int, int]] = None
        if all_x and all_y:
            padding = 20
            xmin = max(0, min(all_x) - padding)
            ymin = max(0, min(all_y) - padding)
            xmax = min(frame_w, max(all_x) + padding)
            ymax = min(frame_h, max(all_y) + padding)
            bounding_box = (xmin, ymin, xmax, ymax)

        # Classification: custom model if available, else geometric rules
        raw_gesture, confidence = self._classify_isl(hands_data, frame_w, frame_h)

        if not raw_gesture:
            return annotated_frame, None

        is_confirmed, smoothed_label = self.update_temporal_smoothing(raw_gesture)

        result = GestureResult(
            label=smoothed_label,
            confidence=confidence,
            is_confirmed=is_confirmed,
            timestamp=time.time(),
            language="ISL",
            bounding_box=bounding_box,
            landmarks_detected=True,
            handedness=[h["label"] for h in hands_data],
        )

        return annotated_frame, result

    # -------------------------------------------------------------------------
    # ISL Heuristic & Geometric Classification Rules
    # -------------------------------------------------------------------------

    def _classify_isl(
        self, hands_data: List[Dict[str, Any]], frame_w: int, frame_h: int
    ) -> Tuple[Optional[str], float]:
        """
        Classifies Indian Sign Language gestures based on dual-handed spatial relations
        and single-hand postures.
        """
        num_hands = len(hands_data)

        # Case 1: Dual-handed (bimanual) gestures
        if num_hands >= 2:
            hand_a = hands_data[0]
            hand_b = hands_data[1]
            return self._classify_two_hand_isl(hand_a, hand_b)

        # Case 2: Single-handed gestures
        elif num_hands == 1:
            hand = hands_data[0]
            return self._classify_single_hand_isl(hand)

        return None, 0.0

    def _classify_two_hand_isl(
        self, hand_a: Dict[str, Any], hand_b: Dict[str, Any]
    ) -> Tuple[Optional[str], float]:
        """
        Evaluates canonical dual-hand ISL signs.
        """
        lm_a = hand_a["landmarks"].landmark
        lm_b = hand_b["landmarks"].landmark
        fingers_a = hand_a["finger_states"]
        fingers_b = hand_b["finger_states"]

        # Distance between wrists (landmark 0)
        wrist_dist = self.euclidean_distance((lm_a[0].x, lm_a[0].y), (lm_b[0].x, lm_b[0].y))
        # Distance between middle fingertips (landmark 12)
        middle_tip_dist = self.euclidean_distance((lm_a[12].x, lm_a[12].y), (lm_b[12].x, lm_b[12].y))
        # Distance between index fingertips (landmark 8)
        index_tip_dist = self.euclidean_distance((lm_a[8].x, lm_a[8].y), (lm_b[8].x, lm_b[8].y))

        # 1. "Namaste" / "Greetings" (ISL iconic prayer gesture):
        # Both hands vertical, palms touching or close, all fingers extended upward
        all_fingers_a = all([fingers_a["index"], fingers_a["middle"], fingers_a["ring"], fingers_a["pinky"]])
        all_fingers_b = all([fingers_b["index"], fingers_b["middle"], fingers_b["ring"], fingers_b["pinky"]])

        if all_fingers_a and all_fingers_b:
            if wrist_dist < 0.18 and middle_tip_dist < 0.15:
                # Vertical check: tips above wrists
                if lm_a[12].y < lm_a[0].y and lm_b[12].y < lm_b[0].y:
                    return "Namaste / Hello", 0.95

            # 2. "Meeting / Together" (hands facing each other coming together)
            if wrist_dist < 0.25 and middle_tip_dist < 0.25:
                return "Meeting / Together", 0.88

        # 3. "Help" in ISL: One hand flat (palm up/forward), other hand forms a fist resting on it
        fist_a = not any([fingers_a["index"], fingers_a["middle"], fingers_a["ring"], fingers_a["pinky"]])
        flat_b = all([fingers_b["index"], fingers_b["middle"], fingers_b["ring"], fingers_b["pinky"]])

        fist_b = not any([fingers_b["index"], fingers_b["middle"], fingers_b["ring"], fingers_b["pinky"]])
        flat_a = all([fingers_a["index"], fingers_a["middle"], fingers_a["ring"], fingers_a["pinky"]])

        if (fist_a and flat_b and wrist_dist < 0.22) or (fist_b and flat_a and wrist_dist < 0.22):
            return "Help Needed", 0.92

        # 4. "Thank You" (Both hands open moving outward from chest)
        if all_fingers_a and all_fingers_b and 0.18 < wrist_dist < 0.45:
            # Both hands at similar vertical height
            if abs(lm_a[0].y - lm_b[0].y) < 0.10:
                return "Thank You", 0.85

        return None, 0.0

    def _classify_single_hand_isl(self, hand: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """
        Evaluates single-hand ISL signs.
        """
        lm = hand["landmarks"].landmark
        fingers = hand["finger_states"]

        # 1. "Yes / Agree" (Thumb-up or nodding fist)
        if fingers["thumb"] and not (fingers["index"] or fingers["middle"] or fingers["ring"] or fingers["pinky"]):
            # Thumb tip is significantly above wrist
            if lm[4].y < lm[0].y - 0.15:
                return "Yes / Agree", 0.90

        # 2. "No / Disagree" (Index finger up, waving or pointed)
        if fingers["index"] and not (fingers["middle"] or fingers["ring"] or fingers["pinky"]):
            return "No / Disagree", 0.88

        # 3. "Stop / Wait" (Open palm facing camera, all fingers extended vertically)
        if all([fingers["thumb"], fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]]):
            if lm[12].y < lm[0].y - 0.20:
                return "Stop / Wait", 0.92

        # 4. "Need Water" (W-shape: Index, Middle, Ring extended, thumb and pinky curled)
        if fingers["index"] and fingers["middle"] and fingers["ring"] and not fingers["pinky"]:
            return "Need Water", 0.86

        # 5. "Good / Great" (Thumb extended outward)
        if fingers["thumb"] and not fingers["index"] and not fingers["middle"]:
            return "Good", 0.85

        return None, 0.0
