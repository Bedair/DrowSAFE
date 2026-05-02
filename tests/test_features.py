"""
Unit tests for DrowSAFE feature extraction (EAR, MAR).

These tests use synthetic landmark data so they run without
a camera or MediaPipe — pure math validation.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import _aspect_ratio, _euclidean, _mouth_opening_ratio, FeatureExtractor, LEFT_EYE


# ---------------------------------------------------------------------------
# Helper — synthetic landmark
# ---------------------------------------------------------------------------
class FakeLandmark:
    """Mimics a MediaPipe NormalizedLandmark."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def make_landmarks(n: int = 468):
    """Return a list of n landmarks at (0.5, 0.5)."""
    return [FakeLandmark(0.5, 0.5) for _ in range(n)]


# ---------------------------------------------------------------------------
# EAR tests
# ---------------------------------------------------------------------------
class TestEuclidean:
    def test_zero_distance(self):
        assert _euclidean((0, 0), (0, 0)) == pytest.approx(0.0)

    def test_horizontal(self):
        assert _euclidean((0, 0), (3, 0)) == pytest.approx(3.0)

    def test_diagonal(self):
        assert _euclidean((0, 0), (3, 4)) == pytest.approx(5.0)


class TestAspectRatio:
    def _make_eye_landmarks(self, ear_target: float):
        v = ear_target
        lms = [FakeLandmark(0.5, 0.5)] * 468
        lms[362] = FakeLandmark(0.0, 0.5)
        lms[385] = FakeLandmark(0.5, 0.5 - v)
        lms[387] = FakeLandmark(0.5, 0.5 - v)
        lms[263] = FakeLandmark(1.0, 0.5)
        lms[373] = FakeLandmark(0.5, 0.5 + v)
        lms[380] = FakeLandmark(0.5, 0.5 + v)
        return lms

    def test_ear_open_eye(self):
        lms = self._make_eye_landmarks(0.3)
        ear = _aspect_ratio(lms, LEFT_EYE, 1, 1)
        assert ear == pytest.approx(0.3, abs=1e-6)

    def test_ear_closed_eye(self):
        lms = self._make_eye_landmarks(0.05)
        ear = _aspect_ratio(lms, LEFT_EYE, 1, 1)
        assert ear == pytest.approx(0.05, abs=1e-6)

    def test_ear_zero_horizontal_span(self):
        lms = make_landmarks()
        ear = _aspect_ratio(lms, LEFT_EYE, 1, 1)
        assert ear == pytest.approx(0.0)


class TestMouthOpeningRatio:
    def _make_mouth_landmarks(self, opening: float, width: float = 1.0):
        """MAR = opening / width — both controllable."""
        lms = [FakeLandmark(0.5, 0.5)] * 468
        lms[13]  = FakeLandmark(0.5, 0.5 - opening / 2)  # top
        lms[14]  = FakeLandmark(0.5, 0.5 + opening / 2)  # bottom
        lms[78]  = FakeLandmark(0.5 - width / 2, 0.5)    # left
        lms[308] = FakeLandmark(0.5 + width / 2, 0.5)    # right
        return lms

    def test_closed_mouth_near_zero(self):
        lms = self._make_mouth_landmarks(opening=0.01)
        mar = _mouth_opening_ratio(lms, 1, 1)
        assert mar < 0.05

    def test_yawning_mouth_high(self):
        lms = self._make_mouth_landmarks(opening=0.6)
        mar = _mouth_opening_ratio(lms, 1, 1)
        assert mar == pytest.approx(0.6, abs=1e-6)

    def test_ratio_scales_with_opening(self):
        lms_small = self._make_mouth_landmarks(opening=0.1)
        lms_large = self._make_mouth_landmarks(opening=0.6)
        assert _mouth_opening_ratio(lms_large, 1, 1) > _mouth_opening_ratio(lms_small, 1, 1)

    def test_zero_width_returns_zero(self):
        lms = make_landmarks()  # all at (0.5, 0.5) — zero width
        mar = _mouth_opening_ratio(lms, 1, 1)
        assert mar == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FeatureExtractor smoke test
# ---------------------------------------------------------------------------
class TestFeatureExtractor:
    def test_instantiation(self):
        fe = FeatureExtractor(1280, 720)
        assert fe.frame_w == 1280
        assert fe.frame_h == 720

    def test_update_frame_size(self):
        fe = FeatureExtractor()
        fe.update_frame_size(640, 480)
        assert fe.frame_w == 640
        assert fe.frame_h == 480
