"""
Unit tests for DrowSAFE feature extraction (EAR, MAR).

These tests use synthetic landmark data so they run without
a camera or MediaPipe — pure math validation.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import _aspect_ratio, _euclidean, FeatureExtractor, LEFT_EYE


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
    """
    Build a synthetic eye where EAR is deterministic:
    Horizontal span = 1.0, each vertical span = 0.4 → EAR = 0.4
    """

    def _make_eye_landmarks(self, ear_target: float):
        """
        Create a minimal landmark list where the 6 eye points yield ear_target.
        Indices used: LEFT_EYE = [362, 385, 387, 263, 373, 380]
        Layout:
          p0=left, p1=top-left, p2=top-right, p3=right, p4=bot-right, p5=bot-left
        EAR = (|p1-p5| + |p2-p4|) / (2*|p0-p3|)
        Choose: p0=(0,0.5), p3=(1,0.5), p1=p2=(0.5, 0.5-v), p4=p5=(0.5, 0.5+v)
        → EAR = 2v / (2*1) = v
        """
        v = ear_target
        lms = [FakeLandmark(0.5, 0.5)] * 468
        lms[362] = FakeLandmark(0.0, 0.5)   # p0 left
        lms[385] = FakeLandmark(0.5, 0.5 - v)  # p1 top-left
        lms[387] = FakeLandmark(0.5, 0.5 - v)  # p2 top-right
        lms[263] = FakeLandmark(1.0, 0.5)   # p3 right
        lms[373] = FakeLandmark(0.5, 0.5 + v)  # p4 bot-right
        lms[380] = FakeLandmark(0.5, 0.5 + v)  # p5 bot-left
        return lms

    def test_ear_open_eye(self):
        lms = self._make_eye_landmarks(0.3)
        # Frame 1×1 px so normalised coords equal pixel coords
        ear = _aspect_ratio(lms, LEFT_EYE, 1, 1)
        assert ear == pytest.approx(0.3, abs=1e-6)

    def test_ear_closed_eye(self):
        lms = self._make_eye_landmarks(0.05)
        ear = _aspect_ratio(lms, LEFT_EYE, 1, 1)
        assert ear == pytest.approx(0.05, abs=1e-6)

    def test_ear_zero_horizontal_span(self):
        """All 6 points coincident → should return 0 (no div-by-zero)."""
        lms = make_landmarks()
        ear = _aspect_ratio(lms, LEFT_EYE, 1, 1)
        assert ear == pytest.approx(0.0)


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
