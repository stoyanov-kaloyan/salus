from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image
from transformers import pipeline


DEFAULT_DEEPFAKE_MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEFAULT_NSFW_MODEL = "Falconsai/nsfw_image_detection"


@dataclass(frozen=True)
class RecognitionDecision:
    should_change: bool
    label: str
    score: float
    predictions: list[dict[str, Any]]


class ImageRecognizer(Protocol):
    def evaluate(self, image: Image.Image) -> RecognitionDecision:
        ...


class DeepFakeRecognizer:
    def __init__(
        self,
        model_name: str = DEFAULT_DEEPFAKE_MODEL,
        device: int = 0,
        target_label: str = "Deepfake",
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.target_label = target_label
        self.threshold = threshold
        self._pipe = None

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


class NsfwRecognizer:
    def __init__(
        self,
        model_name: str = DEFAULT_NSFW_MODEL,
        device: int = 0,
        target_label: str = "nsfw",
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.target_label = target_label
        self.threshold = threshold
        self._pipe = None

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
