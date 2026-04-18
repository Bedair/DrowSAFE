"""
Unit tests for DrowSAFE fatigue scoring (PERCLOS, composite score).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scoring import FatigueScorer
from src.features import Features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_features(ear=0.30, mar=0.30, pitch=0.0):
    """Return a Features object with sensible defaults."""
    return Features(
        ear=ear, ear_left=ear, ear_right=ear,
        mar=mar, head_pitch=pitch,
        head_yaw=0.0, head_roll=0.0,
        face_visible=True,
    )


AWAKE_FEAT   = make_features(ear=0.32, mar=0.20, pitch=0.0)
DROWSY_FEAT  = make_features(ear=0.14, mar=0.25, pitch=5.0)  # eyes closed
YAWNING_FEAT = make_features(ear=0.28, mar=0.75, pitch=0.0)  # yawning
NODDING_FEAT = make_features(ear=0.28, mar=0.20, pitch=25.0) # head down


# ---------------------------------------------------------------------------
# PERCLOS
# ---------------------------------------------------------------------------
class TestPERCLOS:
    def test_zero_at_start(self):
        scorer = FatigueScorer()
        assert scorer.perclos == pytest.approx(0.0)

    def test_all_open_eyes(self):
        scorer = FatigueScorer()
        for _ in range(100):
            scorer.update(AWAKE_FEAT)
        assert scorer.perclos == pytest.approx(0.0)

    def test_all_closed_eyes(self):
        scorer = FatigueScorer()
        for _ in range(100):
            scorer.update(DROWSY_FEAT)
        assert scorer.perclos == pytest.approx(1.0, abs=0.01)

    def test_half_closed(self):
        scorer = FatigueScorer()
        for _ in range(50):
            scorer.update(AWAKE_FEAT)
        for _ in range(50):
            scorer.update(DROWSY_FEAT)
        # PERCLOS should be ~0.5
        assert 0.40 <= scorer.perclos <= 0.60

    def test_reset_clears_history(self):
        scorer = FatigueScorer()
        for _ in range(100):
            scorer.update(DROWSY_FEAT)
        scorer.reset()
        assert scorer.perclos == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------
class TestFatigueScore:
    def test_awake_score_is_low(self):
        scorer = FatigueScorer()
        score = 0.0
        for _ in range(30):
            score = scorer.update(AWAKE_FEAT)
        assert score < 20.0, f"Awake score should be low, got {score}"

    def test_drowsy_score_rises(self):
        scorer = FatigueScorer()
        scores = [scorer.update(DROWSY_FEAT) for _ in range(120)]
        # Score should climb as PERCLOS window fills
        assert scores[-1] > scores[0], "Score should increase with sustained eye closure"

    def test_score_bounded_0_100(self):
        scorer = FatigueScorer()
        for feat in [AWAKE_FEAT, DROWSY_FEAT, YAWNING_FEAT, NODDING_FEAT] * 50:
            score = scorer.update(feat)
            assert 0.0 <= score <= 100.0

    def test_none_features_holds_score(self):
        scorer = FatigueScorer()
        for _ in range(60):
            scorer.update(DROWSY_FEAT)
        held = scorer.update(None)
        # Score should not reset to 0 when face disappears briefly
        assert held > 0.0

    def test_yawn_contributes_to_score(self):
        scorer_base  = FatigueScorer()
        scorer_yawn  = FatigueScorer()

        base_score = 0.0
        yawn_score = 0.0

        # Simulate a yawn event (MAR rises then falls) 3 times
        for _ in range(3):
            for _ in range(20):
                yawn_score = scorer_yawn.update(YAWNING_FEAT)
            for _ in range(10):
                yawn_score = scorer_yawn.update(AWAKE_FEAT)

        for _ in range(90):
            base_score = scorer_base.update(AWAKE_FEAT)

        assert yawn_score > base_score, (
            f"Yawning scorer ({yawn_score:.1f}) should exceed "
            f"baseline ({base_score:.1f})"
        )
