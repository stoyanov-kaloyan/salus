from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deterministic_analysis import StaticRiskEvaluator
from recognizers import DeepFakeRecognizer


class _FakePipe:
    def __call__(self, image: Image.Image):
        return [
            {"label": "Deepfake", "score": 0.70},
            {"label": "Real", "score": 0.30},
        ]


class TestHybridRiskPipeline(unittest.TestCase):
    def test_static_evaluator_multiview_has_extended_features(self) -> None:
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        evaluator = StaticRiskEvaluator(use_multiview=True, parallel=True, profile="full")
        result = evaluator.evaluate(frame)

        scores = result["scores"]
        self.assertIn("jpeg_blocking", scores)
        self.assertIn("edge_ringing", scores)
        self.assertIn("texture_inconsistency", scores)

        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        self.assertGreaterEqual(result["risk"], 0.0)
        self.assertLessEqual(result["risk"], 1.0)

    def test_static_evaluator_fast_profile_skips_slowest_metrics(self) -> None:
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        evaluator = StaticRiskEvaluator(profile="fast", parallel=False)
        result = evaluator.evaluate(frame)

        scores = result["scores"]
        self.assertIn("dft", scores)
        self.assertIn("edge_ringing", scores)
        self.assertNotIn("fft", scores)
        self.assertNotIn("hist", scores)
        self.assertNotIn("noise_residual", scores)

        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_deepfake_recognizer_fuses_neural_and_deterministic(self) -> None:
        image = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8), mode="RGB")

        with patch("recognizers.pipeline", return_value=_FakePipe()):
            recognizer = DeepFakeRecognizer(
                threshold=0.50,
                deterministic_weight=0.40,
                deterministic_enabled=True,
            )

            with patch.object(
                recognizer,
                "_run_deterministic_risk",
                return_value={"risk": 0.80, "scores": {"laplacian": 0.2}},
            ):
                decision = recognizer.evaluate(image)

        self.assertTrue(decision.should_change)
        self.assertEqual(decision.label.casefold(), "deepfake")
        self.assertGreater(decision.score, 0.70)
        self.assertLess(decision.score, 0.80)

        labels = {str(pred.get("label", "")) for pred in decision.predictions}
        self.assertIn("neural_ensemble_risk", labels)
        self.assertIn("deterministic_risk", labels)
        self.assertIn("fused_deepfake_risk", labels)


if __name__ == "__main__":
    unittest.main()
