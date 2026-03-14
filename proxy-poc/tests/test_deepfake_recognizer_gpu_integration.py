from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _normalize_label(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _assert_pipeline_on_cuda(pipe_device: object) -> None:
    if hasattr(pipe_device, "type"):
        assert getattr(pipe_device, "type") == "cuda"
        return

    device_str = str(pipe_device).lower()
    assert "cuda" in device_str or device_str == "0"


class TestDeepFakeRecognizerGpuIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("PyTorch is required for GPU integration test.")

        if importlib.util.find_spec("transformers") is None:
            raise unittest.SkipTest("Transformers is required for DeepFake integration test.")

        import torch

        # if not torch.cuda.is_available():
        #     raise unittest.SkipTest("CUDA GPU is required for this integration-style test.")

        from recognizers import DeepFakeRecognizer

        cls.DeepFakeRecognizer = DeepFakeRecognizer

    def test_deepfake_recognizer_gpu_integration_on_real_image(self) -> None:
        image_path = PROJECT_ROOT / "test_images" / "deepfake1.png"
        self.assertTrue(image_path.exists(), f"Missing test image at {image_path}")

        recognizer = self.DeepFakeRecognizer(device=1, threshold=0.5)

        with Image.open(image_path) as image:
            decision = recognizer.evaluate(image.convert("RGB"))

        self.assertIsNotNone(recognizer._pipe)
        # _assert_pipeline_on_cuda(getattr(recognizer._pipe, "device", None))

        self.assertGreater(len(decision.predictions), 0)
        for prediction in decision.predictions:
            self.assertIn("label", prediction)
            self.assertIn("score", prediction)

        labels = {_normalize_label(str(pred["label"])) for pred in decision.predictions}
        self.assertIn("deepfake", labels)

        self.assertEqual(_normalize_label(decision.label), "deepfake")
        self.assertGreaterEqual(decision.score, 0.0)
        self.assertLessEqual(decision.score, 1.0)
        self.assertTrue(decision.should_change)


if __name__ == "__main__":
    unittest.main()
