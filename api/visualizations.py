from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


BBox = tuple[int, int, int, int]
PaddedBBox = tuple[int, int, int, int]


def _load_rgb_image(image_or_path: Image.Image | str | Path) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")

    with Image.open(image_or_path) as img:
        return img.convert("RGB")


def detect_face_bboxes(
    image: Image.Image,
    *,
    scale_factor: float = 1.1,
    min_neighbors: int = 4,
    cascade_path: str | Path | None = None,
) -> list[BBox]:
    """Return all detected face boxes as (x, y, w, h)."""
    import cv2

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    cascade_file = str(cascade_path) if cascade_path is not None else (cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    cascade = cv2.CascadeClassifier(cascade_file)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_file}")

    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        return []

    return [tuple(int(v) for v in face) for face in faces]


def largest_face_bbox(face_bboxes: list[BBox]) -> BBox | None:
    if not face_bboxes:
        return None
    return max(face_bboxes, key=lambda bbox: bbox[2] * bbox[3])


def expand_bbox_with_padding(
    bbox: BBox,
    image_size: tuple[int, int],
    *,
    padding: float = 0.15,
) -> PaddedBBox:
    """Expand bbox with fractional padding and clamp it to image bounds."""
    x, y, w, h = bbox
    img_w, img_h = image_size

    safe_padding = max(0.0, float(padding))
    pad_x = int(w * safe_padding)
    pad_y = int(h * safe_padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return x1, y1, x2, y2


def crop_with_padded_bbox(
    image: Image.Image,
    bbox: BBox,
    *,
    padding: float = 0.15,
) -> tuple[Image.Image, PaddedBBox]:
    padded_bbox = expand_bbox_with_padding(bbox, image.size, padding=padding)
    return image.crop(padded_bbox), padded_bbox


def visualize_face_detection_and_crop(
    image_or_path: Image.Image | str | Path,
    *,
    padding: float = 0.15,
    scale_factor: float = 1.1,
    min_neighbors: int = 4,
    figsize: tuple[float, float] = (15.0, 5.0),
    show: bool = True,
) -> dict[str, Any]:
    """Visualize detected face box and padded crop in a 3-panel Matplotlib figure.

    Returns a dict with figure/axes plus detection metadata for programmatic use.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = _load_rgb_image(image_or_path)
    face_bboxes = detect_face_bboxes(
        image,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
    )
    bbox = largest_face_bbox(face_bboxes)

    cropped = image
    padded_bbox: PaddedBBox | None = None
    if bbox is not None:
        cropped, padded_bbox = crop_with_padded_bbox(image, bbox, padding=padding)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: original image with all detections
    axes[0].imshow(image)
    axes[0].set_title(f"Detected faces: {len(face_bboxes)}")
    axes[0].axis("off")
    for x, y, w, h in face_bboxes:
        axes[0].add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2, edgecolor="yellow"))

    # Panel 2: largest face + padded crop region
    axes[1].imshow(image)
    axes[1].set_title("Largest face and padded crop")
    axes[1].axis("off")
    if bbox is not None:
        x, y, w, h = bbox
        axes[1].add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2, edgecolor="cyan"))
        if padded_bbox is not None:
            x1, y1, x2, y2 = padded_bbox
            axes[1].add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2, edgecolor="lime")
            )
    else:
        axes[1].text(
            0.5,
            0.5,
            "No face detected",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="red",
        )

    # Panel 3: cropped output used for model inference
    axes[2].imshow(cropped)
    axes[2].set_title("Cropped face (padded)")
    axes[2].axis("off")

    fig.tight_layout()
    if show:
        plt.show()

    return {
        "figure": fig,
        "axes": axes,
        "face_count": len(face_bboxes),
        "all_bboxes": face_bboxes,
        "largest_bbox": bbox,
        "padded_bbox": padded_bbox,
        "cropped_image": cropped,
    }

if __name__ == "__main__":
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "face_detection_chart.png"
    crop_path = output_dir / "cropped_face.png"

    result = visualize_face_detection_and_crop(
        "test_images/deepfake1.png",
        padding=0.15,
    )

    print("Faces:", result["face_count"])
    print("Largest bbox:", result["largest_bbox"])
    print("Padded bbox:", result["padded_bbox"])

    result["figure"].savefig(chart_path, dpi=180, bbox_inches="tight")
    result["cropped_image"].save(crop_path)

    print("Saved chart:", chart_path)
    print("Saved crop:", crop_path)