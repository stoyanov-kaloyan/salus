"""
Pure ONNX Runtime inference for DeepFake Detector v2.
No HuggingFace dependency – only onnxruntime, Pillow, numpy.

Requirements:
    pip install onnxruntime pillow numpy
    (use onnxruntime-gpu for CUDA support)

Usage:
    python infer_onnx.py --image path/to/image.jpg
    python infer_onnx.py --image path/to/image.jpg --model-dir deepfake_detector_onnx
"""

import argparse
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image

# ── Pre-processing constants (ViT / ImageNet standard) ───────────────────────
INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path: str) -> np.ndarray:
    """Load an image and return a (1, 3, 224, 224) float32 numpy array."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)  values in [0, 1]
    arr = (arr - MEAN) / STD                          # normalise
    arr = arr.transpose(2, 0, 1)                      # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)                 # add batch dim -> (1, 3, H, W)
    return arr


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def predict(image_path: str, session: ort.InferenceSession,
            id2label: dict, top_k: int = 2) -> list[dict]:

    pixel_values = preprocess(image_path)

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: pixel_values})[0][0]  # (num_classes,)

    probs = softmax(logits)
    top_indices = np.argsort(probs)[::-1][:top_k]

    return [
        {"label": id2label.get(str(i), str(i)), "score": float(probs[i])}
        for i in top_indices
    ]


def load_session(model_dir: str):
    model_dir = Path(model_dir)

    # Label map from config.json
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    id2label: dict = config.get("id2label", {})

    # ONNX Runtime session
    onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"model.onnx not found in '{model_dir}'.")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path), opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"Provider : {session.get_providers()[0]}")
    print(f"Input    : {session.get_inputs()[0].name}  {session.get_inputs()[0].shape}")

    return session, id2label


def main():
    parser = argparse.ArgumentParser(description="DeepFake Detector - ONNX inference")
    parser.add_argument("--image",     required=True,                    help="Path to input image")
    parser.add_argument("--model-dir", default="deepfake_detector_onnx", help="ONNX model directory")
    parser.add_argument("--top-k",     type=int, default=2,              help="Number of top predictions")
    args = parser.parse_args()

    session, id2label = load_session(args.model_dir)
    results = predict(args.image, session, id2label, top_k=args.top_k)

    print(f"\nImage : {args.image}")
    print("-" * 44)
    for r in results:
        bar = "#" * int(r["score"] * 36)
        print(f"  {r['label']:<10}  {r['score']:.4f}  {bar}")
    print("-" * 44)
    top = results[0]
    print(f"  -> {top['label']} ({top['score'] * 100:.1f}% confidence)\n")


if __name__ == "__main__":
    main()