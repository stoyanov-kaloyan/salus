from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image
from transformers import pipeline


DEFAULT_NSFW_MODEL = "Falconsai/nsfw_image_detection"
DEFAULT_FLUX_MODEL = "prithivMLmods/OpenSDI-Flux.1-SigLIP2"
DEFAULT_FLUX_TARGET_LABEL = "Flux.1_Generated"
DEFAULT_PT_DEEPFAKE_CHECKPOINT = "deepfake_resnet18_best.pt"
DEFAULT_PT_IMAGE_SIZE = 224
DEFAULT_PT_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_PT_NORMALIZE_STD = (0.229, 0.224, 0.225)


def crop_face(
    image: Image.Image,
    padding: float = 0.15,
    *,
    scale_factor: float = 1.1,
    min_neighbors: int = 4,
) -> Image.Image:
    """Detect the largest face in *image* and return it cropped with padding.

    Parameters
    ----------
    image:
        Input PIL image (any mode; converted to RGB internally).
    padding:
        Fractional padding added around the detected face bounding box on
        each side (0.15 = 15 %).  Applied relative to the bounding-box
        dimension (width for left/right, height for top/bottom).
    scale_factor:
        Haar cascade ``scaleFactor`` parameter.
    min_neighbors:
        Haar cascade ``minNeighbors`` parameter (higher = fewer false positives).

    Returns
    -------
    PIL.Image.Image
        Cropped face region, or the original image when no face is detected.
    """
    import cv2 as _cv2

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    bgr = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)
    gray = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)

    cascade = _cv2.CascadeClassifier(_cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    if not isinstance(faces, np.ndarray) or len(faces) == 0:
        return image

    # Pick the largest face by area.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    img_w, img_h = image.size
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return image.convert("RGB").crop((x1, y1, x2, y2))


@dataclass(frozen=True)
class RecognitionDecision:
    should_change: bool
    label: str
    score: float
    predictions: list[dict[str, Any]]


class ImageRecognizer(Protocol):
    def evaluate(self, image: Image.Image) -> RecognitionDecision:
        ...


class _BasePipelineRecognizer:
    def __init__(
        self,
        model_name: str,
        device: int,
        target_label: str,
        threshold: float,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.target_label = target_label
        self.threshold = threshold
        self._pipe = None

    def _predict(self, image: Image.Image) -> list[dict[str, Any]]:
        if self._pipe is None:
            self._pipe = pipeline(
                "image-classification",
                model=self.model_name,
                device=self.device,
            )

        raw_predictions = self._pipe(image)
        if not isinstance(raw_predictions, list):
            raise TypeError("Unexpected prediction output type.")

        normalized: list[dict[str, Any]] = []
        for item in raw_predictions:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "unknown"))
            score = float(item.get("score", 0.0))
            normalized.append({"label": label, "score": score})

        return normalized

    def _select_target_score(self, predictions: list[dict[str, Any]]) -> tuple[str, float]:
        if not predictions:
            return "unknown", 0.0

        for item in predictions:
            label = str(item.get("label", "unknown"))
            score = float(item.get("score", 0.0))
            if label.casefold() == self.target_label.casefold():
                return label, score

        top = predictions[0]
        return str(top.get("label", "unknown")), float(top.get("score", 0.0))


class PytorchDeepFakeRecognizer:
    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] | None = None,
        device: int = 0,
        target_label: str = "Deepfake",
        threshold: float | None = None,
    ) -> None:
        self.device = device
        self.target_label = target_label
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else _default_pt_checkpoint_path()
        self.model_name = str(self._checkpoint_path)

        self._torch = None
        self._model = None
        self._preprocess = None
        self._torch_device = None

        self.image_size = DEFAULT_PT_IMAGE_SIZE
        self.normalization_mean = list(DEFAULT_PT_NORMALIZE_MEAN)
        self.normalization_std = list(DEFAULT_PT_NORMALIZE_STD)
        self.threshold = float(threshold) if threshold is not None else 0.5
        self._load_checkpoint(override_threshold=threshold)

    def _load_checkpoint(self, override_threshold: float | None) -> None:
        try:
            import torch
            import torch.nn as nn
            from torchvision import models as tv_models
            from torchvision import transforms as tv_transforms
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch checkpoint backend requires torch and torchvision to be installed."
            ) from exc

        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file was not found: {self._checkpoint_path}")

        self._torch_device = (
            torch.device(f"cuda:{self.device}")
            if self.device >= 0 and torch.cuda.is_available()
            else torch.device("cpu")
        )

        payload = torch.load(str(self._checkpoint_path), map_location="cpu")

        metadata: dict[str, Any] = {}
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            metadata = payload
            state_dict = dict(payload["state_dict"])
        elif isinstance(payload, dict):
            state_dict = dict(payload)
        else:
            raise TypeError("Unsupported checkpoint format. Expected a state_dict mapping.")

        state_dict = {
            str(key).removeprefix("module."): value
            for key, value in state_dict.items()
        }

        if "image_size" in metadata:
            try:
                self.image_size = int(metadata["image_size"])
            except (TypeError, ValueError):
                self.image_size = DEFAULT_PT_IMAGE_SIZE

        normalization = metadata.get("normalization")
        if isinstance(normalization, dict):
            mean = normalization.get("mean")
            std = normalization.get("std")
            if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)) and len(mean) == 3 and len(std) == 3:
                self.normalization_mean = [float(v) for v in mean]
                self.normalization_std = [float(v) for v in std]

        if override_threshold is None:
            saved_threshold = metadata.get("threshold")
            if saved_threshold is not None:
                try:
                    self.threshold = float(saved_threshold)
                except (TypeError, ValueError):
                    self.threshold = 0.5

        model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(state_dict)
        model = model.to(self._torch_device)
        model.eval()

        self._preprocess = tv_transforms.Compose(
            [
                tv_transforms.Resize((self.image_size, self.image_size)),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std),
            ]
        )

        self._torch = torch
        self._model = model

    def evaluate(self, image: Image.Image) -> RecognitionDecision:
        if self._torch is None or self._model is None or self._preprocess is None or self._torch_device is None:
            raise RuntimeError("PyTorch deepfake recognizer is not initialized.")

        rgb = crop_face(image)
        tensor = self._preprocess(rgb).unsqueeze(0).to(self._torch_device)

        with self._torch.no_grad():
            logits = self._model(tensor).squeeze(1)
            deepfake_score = float(self._torch.sigmoid(logits)[0].item())

        real_score = float(1.0 - deepfake_score)
        label = self.target_label if deepfake_score >= 0.5 else "Real"
        should_change = (
            label.casefold() == self.target_label.casefold()
            and deepfake_score >= self.threshold
        )

        predictions = [
            {"label": "Real", "score": real_score},
            {"label": self.target_label, "score": deepfake_score},
        ]

        return RecognitionDecision(
            should_change=should_change,
            label=label,
            score=deepfake_score,
            predictions=predictions,
        )


def _default_pt_checkpoint_path() -> Path:
    return Path(__file__).resolve().parent / DEFAULT_PT_DEEPFAKE_CHECKPOINT


def create_deepfake_recognizer(
    *,
    checkpoint_path: str | os.PathLike[str] | None = None,
    device: int = 0,
    target_label: str = "Deepfake",
    threshold: float | None = None,
    **_kwargs: object,
) -> ImageRecognizer:
    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path else _default_pt_checkpoint_path()
    return PytorchDeepFakeRecognizer(
        checkpoint_path=resolved_checkpoint,
        device=device,
        target_label=target_label,
        threshold=threshold,
    )


class FluxDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_FLUX_MODEL,
        device: int = 0,
        target_label: str = DEFAULT_FLUX_TARGET_LABEL,
        threshold: float = 0.5,
        id2label: dict[int | str, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.target_label = target_label
        self.threshold = float(threshold)
        self.id2label = id2label or {
            0: "Real_Image",
            1: DEFAULT_FLUX_TARGET_LABEL,
        }

        self._torch = None
        self._processor = None
        self._model = None
        self._torch_device = None
        self._load_model()

    def _label_for_index(self, idx: int) -> str:
        if idx in self.id2label:
            return str(self.id2label[idx])
        key = str(idx)
        if key in self.id2label:
            return str(self.id2label[key])
        return f"class_{idx}"

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, SiglipForImageClassification
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "FluxDetector requires torch and a recent transformers build with SigLIP support."
            ) from exc

        self._torch_device = (
            torch.device(f"cuda:{self.device}")
            if self.device >= 0 and torch.cuda.is_available()
            else torch.device("cpu")
        )

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = SiglipForImageClassification.from_pretrained(self.model_name)
        self._model = self._model.to(self._torch_device)
        self._model.eval()
        self._torch = torch

    def evaluate(self, image: Image.Image) -> RecognitionDecision:
        if self._torch is None or self._processor is None or self._model is None or self._torch_device is None:
            raise RuntimeError("FluxDetector is not initialized.")

        rgb = image.convert("RGB")
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {name: tensor.to(self._torch_device) for name, tensor in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs)
            probs = self._torch.nn.functional.softmax(outputs.logits, dim=1).squeeze(0)

        prob_list = probs.detach().cpu().tolist()
        predictions = [
            {
                "label": self._label_for_index(idx),
                "score": float(score),
            }
            for idx, score in enumerate(prob_list)
        ]

        target_score = 0.0
        for pred in predictions:
            if str(pred["label"]).casefold() == self.target_label.casefold():
                target_score = float(pred["score"])
                break

        top_idx = int(probs.argmax().item())
        top_label = self._label_for_index(top_idx)
        final_label = self.target_label if target_score >= 0.5 else top_label

        should_change = (
            final_label.casefold() == self.target_label.casefold()
            and target_score >= self.threshold
        )

        return RecognitionDecision(
            should_change=should_change,
            label=final_label,
            score=target_score,
            predictions=predictions,
        )


def create_flux_detector(
    *,
    model_name: str = DEFAULT_FLUX_MODEL,
    device: int = 0,
    target_label: str = DEFAULT_FLUX_TARGET_LABEL,
    threshold: float = 0.5,
    id2label: dict[int | str, str] | None = None,
) -> FluxDetector:
    return FluxDetector(
        model_name=model_name,
        device=device,
        target_label=target_label,
        threshold=threshold,
        id2label=id2label,
    )


class NsfwRecognizer(_BasePipelineRecognizer):
    def __init__(
        self,
        model_name: str = DEFAULT_NSFW_MODEL,
        device: int = 0,
        target_label: str = "nsfw",
        threshold: float = 0.25,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            target_label=target_label,
            threshold=threshold,
        )

    def evaluate(self, image: Image.Image) -> RecognitionDecision:
        predictions = self._predict(image)
        label, score = self._select_target_score(predictions)

        should_change = (
            label.casefold() == self.target_label.casefold()
            and score >= self.threshold
        )

        return RecognitionDecision(
            should_change=should_change,
            label=label,
            score=score,
            predictions=predictions,
        )
