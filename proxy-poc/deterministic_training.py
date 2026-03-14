from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from deterministic_analysis import ANALYZERS, run_multiview_analyzers


@dataclass(frozen=True)
class LabeledImage:
    path: Path
    label: int


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def load_manifest(manifest_path: Path) -> list[LabeledImage]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    rows: list[LabeledImage] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Manifest CSV is empty.")

        expected = {"path", "label"}
        if not expected.issubset({field.strip().lower() for field in reader.fieldnames}):
            raise ValueError("Manifest must contain columns: path,label")

        for row in reader:
            raw_path = str(row.get("path", "")).strip()
            raw_label = str(row.get("label", "")).strip()
            if not raw_path or raw_label == "":
                continue

            full_path = (manifest_path.parent / raw_path).resolve()
            if not full_path.exists():
                continue

            label = int(float(raw_label))
            if label not in (0, 1):
                continue

            rows.append(LabeledImage(path=full_path, label=label))

    if len(rows) < 10:
        raise ValueError("Need at least 10 valid samples to calibrate deterministic model.")

    return rows


def _extract_feature_vector(sample: LabeledImage, feature_order: list[str]) -> tuple[np.ndarray, int]:
    image = cv2.imread(str(sample.path))
    if image is None:
        raise ValueError(f"Could not read image: {sample.path}")

    scores = run_multiview_analyzers(image, parallel=True)
    vector = np.array([float(scores.get(name, 0.0)) for name in feature_order], dtype=np.float32)
    return vector, sample.label


def extract_dataset(
    samples: list[LabeledImage],
    feature_order: list[str],
    max_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []

    workers = max(1, min(max_workers, len(samples)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_extract_feature_vector, sample, feature_order): sample
            for sample in samples
        }
        for future in as_completed(futures):
            sample = futures[future]
            try:
                vector, label = future.result()
                vectors.append(vector)
                labels.append(label)
            except Exception as exc:
                print(f"Skipping {sample.path.name}: {exc}")

    if len(vectors) < 10:
        raise ValueError("Not enough valid images for calibration after feature extraction.")

    x = np.vstack(vectors).astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    return x, y


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    num_features = x_train.shape[1]
    weights = np.zeros(num_features, dtype=np.float32)
    bias = 0.0

    for _ in range(epochs):
        logits = x_train @ weights + bias
        probs = _sigmoid(logits)
        error = probs - y_train

        grad_w = (x_train.T @ error) / x_train.shape[0] + l2 * weights
        grad_b = float(error.mean())

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, float(bias)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float(np.sum((y_true == 1.0) & (y_pred == 1.0)))
    tn = float(np.sum((y_true == 0.0) & (y_pred == 0.0)))
    fp = float(np.sum((y_true == 0.0) & (y_pred == 1.0)))
    fn = float(np.sum((y_true == 1.0) & (y_pred == 0.0)))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def select_best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = _metrics(labels, (probabilities >= 0.5).astype(np.float32))
    best_f1 = best_metrics["f1"]

    for threshold in np.linspace(0.10, 0.90, 161):
        predicted = (probabilities >= threshold).astype(np.float32)
        metrics = _metrics(labels, predicted)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def split_train_validation(
    x: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)

    val_count = max(1, int(round(x.shape[0] * validation_fraction)))
    val_count = min(val_count, x.shape[0] - 1)

    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def save_calibration(
    output_path: Path,
    feature_order: list[str],
    weights: np.ndarray,
    bias: float,
    means: np.ndarray,
    scales: np.ndarray,
    threshold: float,
    metrics: dict[str, float],
    train_size: int,
    validation_size: int,
) -> None:
    payload = {
        "feature_order": feature_order,
        "weights": [float(v) for v in weights],
        "bias": float(bias),
        "means": [float(v) for v in means],
        "scales": [float(v) for v in scales],
        "threshold": float(threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "train_size": int(train_size),
        "validation_size": int(validation_size),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deterministic deepfake calibration weights.")
    parser.add_argument("--manifest", required=True, help="CSV with columns: path,label")
    parser.add_argument("--output", required=True, help="Output calibration JSON path")
    parser.add_argument("--epochs", type=int, default=1200, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="Gradient descent learning rate")
    parser.add_argument("--l2", type=float, default=0.001, help="L2 regularization")
    parser.add_argument("--validation-fraction", type=float, default=0.20, help="Validation split fraction")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers for feature extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()

    samples = load_manifest(manifest_path)
    feature_order = list(ANALYZERS.keys())
    x, y = extract_dataset(samples, feature_order, max_workers=args.max_workers)

    x_train, y_train, x_val, y_val = split_train_validation(
        x,
        y,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)

    x_train_std = (x_train - means) / scales
    x_val_std = (x_val - means) / scales

    weights, bias = train_logistic_regression(
        x_train=x_train_std,
        y_train=y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )

    val_probs = _sigmoid(x_val_std @ weights + bias)
    threshold, metrics = select_best_threshold(val_probs, y_val)

    save_calibration(
        output_path=output_path,
        feature_order=feature_order,
        weights=weights,
        bias=bias,
        means=means,
        scales=scales,
        threshold=threshold,
        metrics=metrics,
        train_size=x_train.shape[0],
        validation_size=x_val.shape[0],
    )

    print("Calibration written:", output_path)
    print("Validation metrics:", metrics)
    print("Recommended threshold:", round(threshold, 4))


if __name__ == "__main__":
    main()
