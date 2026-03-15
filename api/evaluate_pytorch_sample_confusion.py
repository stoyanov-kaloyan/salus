from __future__ import annotations

# python .\evaluate_pytorch_sample_confusion.py --sample-per-class 20000 --batch-size 128 --num-workers 8 --device cuda

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int  # 0=real, 1=ai


class ImageRecordDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record.label, str(record.path)


def _collect_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Image folder was not found: {folder}")

    files = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not files:
        raise ValueError(f"No images found in {folder}")

    return sorted(files)


def _sample_records(ai_dir: Path, real_dir: Path, sample_per_class: int, seed: int) -> list[ImageRecord]:
    ai_images = _collect_images(ai_dir)
    real_images = _collect_images(real_dir)

    if len(ai_images) < sample_per_class:
        raise ValueError(
            f"Requested {sample_per_class} AI images but only found {len(ai_images)} in {ai_dir}."
        )
    if len(real_images) < sample_per_class:
        raise ValueError(
            f"Requested {sample_per_class} real images but only found {len(real_images)} in {real_dir}."
        )

    rng = random.Random(seed)
    sampled_ai = rng.sample(ai_images, sample_per_class)
    sampled_real = rng.sample(real_images, sample_per_class)

    records = [ImageRecord(path=path, label=1) for path in sampled_ai]
    records.extend(ImageRecord(path=path, label=0) for path in sampled_real)
    rng.shuffle(records)
    return records


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
    threshold_override: float | None,
) -> tuple[torch.nn.Module, int, list[float], list[float], float]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file was not found: {checkpoint_path}")

    payload = torch.load(str(checkpoint_path), map_location="cpu")

    metadata: dict[str, object] = {}
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        metadata = payload
        state_dict = dict(payload["state_dict"])
    elif isinstance(payload, dict):
        state_dict = dict(payload)
    else:
        raise TypeError("Unsupported checkpoint format. Expected a state_dict mapping.")

    state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}

    image_size = DEFAULT_IMAGE_SIZE
    raw_image_size = metadata.get("image_size")
    if raw_image_size is not None:
        try:
            image_size = int(raw_image_size)
        except (TypeError, ValueError):
            image_size = DEFAULT_IMAGE_SIZE

    mean = list(DEFAULT_MEAN)
    std = list(DEFAULT_STD)
    normalization = metadata.get("normalization")
    if isinstance(normalization, dict):
        raw_mean = normalization.get("mean")
        raw_std = normalization.get("std")
        if (
            isinstance(raw_mean, (list, tuple))
            and isinstance(raw_std, (list, tuple))
            and len(raw_mean) == 3
            and len(raw_std) == 3
        ):
            mean = [float(value) for value in raw_mean]
            std = [float(value) for value in raw_std]

    if threshold_override is None:
        saved_threshold = metadata.get("threshold")
        if saved_threshold is None:
            threshold = 0.5
        else:
            try:
                threshold = float(saved_threshold)
            except (TypeError, ValueError):
                threshold = 0.5
    else:
        threshold = float(threshold_override)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, image_size, mean, std, threshold


def _save_confusion_matrix_plot(cm: np.ndarray, threshold: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})", fontsize=13, weight="bold")
    ax.set_xticks([0, 1], labels=["Pred Real", "Pred AI"])
    ax.set_yticks([0, 1], labels=["True Real", "True AI"])

    max_value = int(cm.max()) if cm.size else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > max_value * 0.5 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=13, color=color, weight="bold")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_root = script_dir / "data" / "real-vs-ai-mbilal" / "my_real_vs_ai_dataset" / "my_real_vs_ai_dataset"

    parser = argparse.ArgumentParser(
        description="Run PyTorch deepfake checkpoint on a balanced image sample and output confusion matrix."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=script_dir / "deepfake_resnet18_best.pt",
        help="Path to .pt checkpoint (default: proxy-poc/deepfake_resnet18_best.pt).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="Root directory containing class folders 'ai_images' and 'real'.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=20000,
        help="Number of images to sample per class (default: 20000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size (default: 128).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 2),
        help="DataLoader workers (default: min(8, CPU count)).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold override (default: use checkpoint threshold or 0.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs",
        help="Output directory for confusion matrix and prediction files.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--skip-predictions-csv",
        action="store_true",
        help="Skip writing per-image predictions CSV.",
    )

    args = parser.parse_args()
    if args.sample_per_class <= 0:
        parser.error("--sample-per-class must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")

    return args


def main() -> None:
    args = parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ai_dir = args.data_root / "ai_images"
    real_dir = args.data_root / "real"
    records = _sample_records(
        ai_dir=ai_dir,
        real_dir=real_dir,
        sample_per_class=int(args.sample_per_class),
        seed=int(args.seed),
    )

    model, image_size, mean, std, threshold = _load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        threshold_override=args.threshold,
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    dataset = ImageRecordDataset(records, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )

    all_true: list[int] = []
    all_pred: list[int] = []
    all_scores: list[float] = []
    all_paths: list[str] = []

    with torch.no_grad():
        if tqdm is not None:
            iterator = tqdm(loader, total=len(loader), desc="Inference", unit="batch")
        else:
            iterator = loader

        for images, labels, paths in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images).squeeze(1)
            scores = torch.sigmoid(logits)
            preds = (scores >= threshold).to(dtype=torch.int64)

            all_true.extend(labels.detach().cpu().numpy().astype(np.int32).tolist())
            all_pred.extend(preds.detach().cpu().numpy().astype(np.int32).tolist())
            all_scores.extend(scores.detach().cpu().numpy().astype(np.float32).tolist())
            all_paths.extend(paths)

    y_true = np.array(all_true, dtype=np.int32)
    y_pred = np.array(all_pred, dtype=np.int32)
    y_score = np.array(all_scores, dtype=np.float32)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.sample_per_class}_per_class"
    cm_plot_path = args.output_dir / f"pytorch_confusion_matrix_{suffix}.png"
    metrics_path = args.output_dir / f"pytorch_confusion_metrics_{suffix}.json"
    predictions_path = args.output_dir / f"pytorch_predictions_{suffix}.csv"

    _save_confusion_matrix_plot(cm, threshold=threshold, out_path=cm_plot_path)

    payload = {
        "sample_per_class": int(args.sample_per_class),
        "total_samples": int(y_true.size),
        "threshold": float(threshold),
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "device": str(device),
        "image_size": int(image_size),
        "normalization": {"mean": mean, "std": std},
        "confusion_matrix": {
            "labels": ["real(0)", "ai(1)"],
            "matrix": cm.astype(int).tolist(),
        },
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.skip_predictions_csv:
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "score", "pred"])
            writer.writeheader()
            for path, y, score, pred in zip(all_paths, all_true, all_scores, all_pred):
                writer.writerow(
                    {
                        "path": path,
                        "y_true": int(y),
                        "score": float(score),
                        "pred": int(pred),
                    }
                )

    print("\n=== Evaluation Summary ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Sampled: {args.sample_per_class} AI + {args.sample_per_class} real")
    print(f"Threshold: {threshold:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}")
    print(json.dumps(metrics, indent=2))
    print(f"Saved confusion matrix image: {cm_plot_path}")
    print(f"Saved metrics JSON: {metrics_path}")
    if not args.skip_predictions_csv:
        print(f"Saved predictions CSV: {predictions_path}")


if __name__ == "__main__":
    main()