"""
Convert prithivMLmods/Deep-Fake-Detector-v2-Model to ONNX format
Requirements:
    pip install transformers torch optimum[exporters] onnx onnxruntime pillow
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

# ── 1. Export to ONNX using Optimum ──────────────────────────────────────────
def export_model(output_dir: str = "deepfake_detector_onnx"):
    from optimum.exporters.onnx import main_export

    print("Exporting model to ONNX via Optimum …")
    main_export(
        model_name_or_path="prithivMLmods/Deep-Fake-Detector-v2-Model",
        output=output_dir,
        task="image-classification",
        opset=17,          # ONNX opset; 17 is broadly supported
        optimize="O2",     # optional graph optimisation (O1–O4)
    )
    print(f"✅  Model saved to: {output_dir}/")
    return output_dir


# ── 2. Run inference with ONNX Runtime ───────────────────────────────────────
class DeepFakeDetectorONNX:
    """
    Drop-in ONNX Runtime replacement for the HuggingFace pipeline.
    """

    def __init__(self, model_dir: str = "deepfake_detector_onnx"):
        import onnxruntime as ort
        from transformers import AutoFeatureExtractor
        import json

        self.model_dir = Path(model_dir)

        # Load feature extractor (image pre-processing)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_dir)

        # Load label mapping
        config_path = self.model_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        self.id2label = config.get("id2label", {})

        # Create ONNX Runtime session
        onnx_path = self._find_onnx_model()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # Use CUDAExecutionProvider for GPU; falls back to CPU automatically
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options, providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        print(f"✅  ONNX session ready  |  model: {onnx_path.name}")
        print(f"    Active provider: {self.session.get_providers()[0]}")

    def _find_onnx_model(self) -> Path:
        """Locate the main ONNX model file inside the export directory."""
        candidates = list(self.model_dir.glob("model.onnx")) + \
                     list(self.model_dir.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(
                f"No .onnx file found in {self.model_dir}. "
                "Run export_model() first."
            )
        return candidates[0]

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def __call__(self, image_path: str, top_k: int = 5):
        """
        Parameters
        ----------
        image_path : str | PIL.Image
            Path to an image file or a PIL Image object.
        top_k : int
            Number of top predictions to return.

        Returns
        -------
        list[dict]  – [{"label": "...", "score": 0.99}, …]
        """
        # Pre-process
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        inputs = self.extractor(images=image, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)

        # Inference
        outputs = self.session.run(None, {self.input_name: pixel_values})
        logits = outputs[0]          # shape: (1, num_classes)

        # Post-process
        probs = self._softmax(logits)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_indices:
            label = self.id2label.get(str(idx), str(idx))
            results.append({"label": label, "score": float(probs[idx])})

        return results


# ── 3. Optional: benchmark latency ───────────────────────────────────────────
def benchmark(detector: DeepFakeDetectorONNX, image_path: str, runs: int = 50):
    import time

    image = Image.open(image_path).convert("RGB")
    # Warm-up
    for _ in range(5):
        detector(image)

    start = time.perf_counter()
    for _ in range(runs):
        detector(image)
    elapsed = time.perf_counter() - start

    print(f"Benchmark ({runs} runs): {elapsed*1000/runs:.1f} ms / image")


# ── 4. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepFake Detector ONNX")
    parser.add_argument(
        "--export", action="store_true",
        help="Download & export the HuggingFace model to ONNX (run once)."
    )
    parser.add_argument(
        "--model-dir", default="deepfake_detector_onnx",
        help="Directory containing the exported ONNX model."
    )
    parser.add_argument(
        "--image", default=None,
        help="Path to an image to classify."
    )
    parser.add_argument(
        "--bench", action="store_true",
        help="Run a latency benchmark (requires --image)."
    )
    args = parser.parse_args()

    if args.export:
        export_model(args.model_dir)

    if args.image:
        detector = DeepFakeDetectorONNX(model_dir=args.model_dir)
        results = detector(args.image)
        print("\nPredictions:")
        for r in results:
            bar = "█" * int(r["score"] * 40)
            print(f"  {r['label']:<20} {r['score']:.4f}  {bar}")

        if args.bench:
            benchmark(detector, args.image)