from __future__ import annotations
from agentforge.anti_oscillation import AntiOscillation


class TestPlateau:
    def test_no_plateau(self):
        assert AntiOscillation.check_plateau([52, 67, 74, 77, 82, 85]) is False

    def test_plateau_detected(self):
        assert AntiOscillation.check_plateau([52, 67, 74, 82, 82.1, 82.2, 82.1], window=3) is True

    def test_too_few_rounds(self):
        assert AntiOscillation.check_plateau([52, 67]) is False


class TestSeedPolicy:
    def test_same_fingerprint_same_seed(self):
        s1 = AntiOscillation.compute_seed("fp_abc", 42)
        s2 = AntiOscillation.compute_seed("fp_abc", 42)
        assert s1 == s2

    def test_different_fingerprint_different_seed(self):
        s1 = AntiOscillation.compute_seed("fp_abc", 42)
        s2 = AntiOscillation.compute_seed("fp_xyz", 42)
        assert s1 != s2
