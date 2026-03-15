from __future__ import annotations
import hashlib


class AntiOscillation:
    @staticmethod
    def check_plateau(trajectory, threshold=0.005, window=3):
        if len(trajectory) < window + 1:
            return False
        best_before = max(trajectory[:-window])
        best_recent = max(trajectory[-window:])
        if best_before == 0:
            return False
        improvement = (best_recent - best_before) / abs(best_before)
        return improvement < threshold

    @staticmethod
    def compute_seed(fingerprint, base_seed=42):
        h = hashlib.sha256(fingerprint.encode()).hexdigest()
        return base_seed + int(h[:8], 16) % 10000
