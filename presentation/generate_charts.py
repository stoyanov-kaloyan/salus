from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    scores: list[float] = []
    y_pred: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row is None:
                continue
            raw_y = row.get("y_true")
            raw_score = row.get("score")
            raw_pred = row.get("pred")
            if raw_y is None or raw_score is None:
                continue

            try:
                y = int(float(raw_y))
                s = float(raw_score)
            except ValueError:
                continue

            if raw_pred is None or raw_pred == "":
                p = int(s >= 0.5)
            else:
                try:
                    p = int(float(raw_pred))
                except ValueError:
                    p = int(s >= 0.5)

            y_true.append(y)
            scores.append(s)
            y_pred.append(p)

    if not y_true:
        raise ValueError(f"No valid rows found in {csv_path}")

    return np.array(y_true, dtype=np.int32), np.array(scores, dtype=np.float32), np.array(y_pred, dtype=np.int32)


def plot_class_balance(y_true: np.ndarray, out_path: Path) -> None:
    real_count = int(np.sum(y_true == 0))
    ai_count = int(np.sum(y_true == 1))
    total = max(1, real_count + ai_count)

    labels = ["Real", "AI"]
    values = [real_count, ai_count]
    colors = ["#2D6A4F", "#D62828"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#1B1B1B", linewidth=1.0)
    ax.set_title("Баланс на класовете в evaluation набора", fontsize=14, weight="bold")
    ax.set_ylabel("Брой изображения")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        pct = 100.0 * value / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{value} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_score_distribution(y_true: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    real_scores = scores[y_true == 0]
    ai_scores = scores[y_true == 1]

    bins = np.linspace(0.0, 1.0, 24)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(real_scores, bins=bins, alpha=0.65, color="#1D3557", label="Real (y=0)", density=True)
    ax.hist(ai_scores, bins=bins, alpha=0.65, color="#E63946", label="AI/Deepfake (y=1)", density=True)
    ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1.5, label="Праг 0.5")
    ax.set_title("Разпределение на Deepfake score", fontsize=14, weight="bold")
    ax.set_xlabel("Score за класа Deepfake")
    ax.set_ylabel("Плътност")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int32)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (Deepfake)", fontsize=14, weight="bold")
    ax.set_xticks([0, 1], labels=["Pred: Real", "Pred: AI"])
    ax.set_yticks([0, 1], labels=["True: Real", "True: AI"])

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14, color=color, weight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_threshold_curves(y_true: np.ndarray, scores: np.ndarray, out_path: Path) -> dict[str, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    precision_vals: list[float] = []
    recall_vals: list[float] = []
    f1_vals: list[float] = []

    for thr in thresholds:
        y_pred_thr = (scores >= thr).astype(np.int32)
        metrics = compute_metrics(y_true, y_pred_thr)
        precision_vals.append(metrics["precision"])
        recall_vals.append(metrics["recall"])
        f1_vals.append(metrics["f1"])

    precision_arr = np.array(precision_vals)
    recall_arr = np.array(recall_vals)
    f1_arr = np.array(f1_vals)

    best_idx = int(np.argmax(f1_arr))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1_arr[best_idx])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thresholds, precision_arr, color="#4C6EF5", linewidth=2.0, label="Precision")
    ax.plot(thresholds, recall_arr, color="#2B8A3E", linewidth=2.0, label="Recall")
    ax.plot(thresholds, f1_arr, color="#D9480F", linewidth=2.4, label="F1")

    ax.scatter([best_thr], [best_f1], color="#D9480F", s=80, zorder=4)
    ax.annotate(
        f"Best F1={best_f1:.3f} @ thr={best_thr:.2f}",
        xy=(best_thr, best_f1),
        xytext=(best_thr + 0.06, min(0.98, best_f1 + 0.07)),
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.2},
        fontsize=10,
    )

    ax.set_title("Trade-off при избор на праг", fontsize=14, weight="bold")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Стойност")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {"best_threshold": best_thr, "best_f1": best_f1}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "proxy-poc" / "outputs" / "deepfake_ai_only_predictions.csv"
    out_dir = root / "presentation" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true, scores, y_pred = load_predictions(csv_path)

    metrics_default = compute_metrics(y_true, y_pred)
    threshold_info = plot_threshold_curves(y_true, scores, out_dir / "threshold_tradeoff.png")
    y_pred_best = (scores >= threshold_info["best_threshold"]).astype(np.int32)
    metrics_best = compute_metrics(y_true, y_pred_best)

    plot_class_balance(y_true, out_dir / "class_balance.png")
    plot_score_distribution(y_true, scores, out_dir / "score_distribution.png")
    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred_best, out_dir / "confusion_matrix_best_threshold.png")

    summary = {
        "sample_count": int(y_true.size),
        "real_count": int(np.sum(y_true == 0)),
        "ai_count": int(np.sum(y_true == 1)),
        "metrics_current_threshold": metrics_default,
        "metrics_best_threshold": metrics_best,
        "threshold_sweep": threshold_info,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Generated charts in:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
