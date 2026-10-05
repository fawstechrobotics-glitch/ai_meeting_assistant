"""
Unit tests for ISL and ASL sign language recognizers and landmark math.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import AppSettings
from src.modules.asl_detection import AmericanSignLanguageRecognizer
from src.modules.base import BaseSignLanguageRecognizer, GestureResult
from src.modules.isl_detection import IndianSignLanguageRecognizer


class DummyRecognizer(BaseSignLanguageRecognizer):
    """Concrete implementation for testing base math and smoothing methods."""

    def process_frame(self, frame: np.ndarray, draw_landmarks: bool = True):
        return frame, None

    def reset(self) -> None:
        super().reset()


def test_base_euclidean_distance() -> None:
    """Validates 2D and 3D Euclidean distance calculations."""
    dist_2d = DummyRecognizer.euclidean_distance((0.0, 0.0), (3.0, 4.0))
    assert pytest.approx(dist_2d, 0.001) == 5.0

    dist_3d = DummyRecognizer.euclidean_distance((0.0, 0.0, 0.0), (1.0, 2.0, 2.0))
    assert pytest.approx(dist_3d, 0.001) == 3.0


def test_temporal_smoothing_hysteresis() -> None:
    """Verifies that majority voting confirms stable gestures over time."""
    recognizer = DummyRecognizer(confirmation_frames=5)

    # Feed transient noisy frame
    confirmed, label = recognizer.update_temporal_smoothing("Noise")
    assert not confirmed

    # Feed repeated target frame
    for _ in range(4):
        confirmed, label = recognizer.update_temporal_smoothing("Namaste")

    assert confirmed is True
    assert label == "Namaste"


def test_isl_two_hand_namaste_logic() -> None:
    """Tests the ISL bimanual Namaste classification logic with mock landmark coordinates."""
    settings = AppSettings()
    isl = IndianSignLanguageRecognizer(settings=settings)

    # Create mock landmarks for two hands placed together in prayer gesture
    class MockPoint:
        def __init__(self, x: float, y: float, z: float = 0.0):
            self.x, self.y, self.z = x, y, z

    class MockLandmarkList:
        def __init__(self, points):
            self.landmark = points

    # Hand A (Right): Wrist at (0.48, 0.6), middle tip at (0.49, 0.3)
    pts_a = [MockPoint(0.48, 0.6)] * 21
    pts_a[12] = MockPoint(0.49, 0.3)
    hand_a = {
        "landmarks": MockLandmarkList(pts_a),
        "label": "Right",
        "finger_states": {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True},
    }

    # Hand B (Left): Wrist at (0.52, 0.6), middle tip at (0.51, 0.3)
    pts_b = [MockPoint(0.52, 0.6)] * 21
    pts_b[12] = MockPoint(0.51, 0.3)
    hand_b = {
        "landmarks": MockLandmarkList(pts_b),
        "label": "Left",
        "finger_states": {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True},
    }

    label, conf = isl._classify_two_hand_isl(hand_a, hand_b)
    assert label == "Namaste / Hello"
    assert conf >= 0.90


def test_asl_single_hand_ily_logic() -> None:
    """Tests the ASL single-hand I Love You (ILY) classification logic."""
    settings = AppSettings()
    asl = AmericanSignLanguageRecognizer(settings=settings)

    class MockPoint:
        def __init__(self, x: float, y: float, z: float = 0.0):
            self.x, self.y, self.z = x, y, z

    pts = [MockPoint(0.5, 0.5)] * 21
    pts[4] = MockPoint(0.3, 0.3)  # Thumb tip
    pts[8] = MockPoint(0.5, 0.2)  # Index tip

    class MockLandmarkList:
        def __init__(self, points):
            self.landmark = points

    hand = {
        "landmarks": MockLandmarkList(pts),
        "label": "Right",
        # Thumb, Index, Pinky extended; Middle, Ring folded
        "finger_states": {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": True},
    }

    label, conf = asl._classify_single_hand_asl(hand)
    assert label == "I Love You"
    assert conf >= 0.95
