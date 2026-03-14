from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from transformers import pipeline

from deterministic_analysis import DEFAULT_ANALYSIS_PROFILE, StaticRiskEvaluator


DEFAULT_DEEPFAKE_MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEFAULT_NSFW_MODEL = "Falconsai/nsfw_image_detection"

DEFAULT_NEURAL_VARIANT_WEIGHTS: dict[str, float] = {
    "original": 0.55,
    "autocontrast": 0.20,
    "equalized": 0.15,
    "sharpened": 0.10,
}


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


class DeepFakeRecognizer(_BasePipelineRecognizer):
    def __init__(
        self,
        model_name: str = DEFAULT_DEEPFAKE_MODEL,
        device: int = 0,
        target_label: str = "Deepfake",
        threshold: float = 0.8,
        deterministic_weight: float = 0.35,
        deterministic_enabled: bool = True,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            target_label=target_label,
            threshold=threshold,
        )

        env_det_weight = os.getenv("DEEPFAKE_DETERMINISTIC_WEIGHT")
        if env_det_weight is not None:
            deterministic_weight = float(env_det_weight)

        self.deterministic_weight = float(np.clip(deterministic_weight, 0.0, 0.9))
        self.neural_weight = float(np.clip(1.0 - self.deterministic_weight, 0.1, 1.0))
        self.neural_variant_weights = dict(DEFAULT_NEURAL_VARIANT_WEIGHTS)

        env_det_toggle = os.getenv("DEEPFAKE_USE_DETERMINISTIC")
        if env_det_toggle is not None:
            deterministic_enabled = env_det_toggle.strip().lower() not in {"0", "false", "no"}

        calibration_path = os.getenv("DETERMINISTIC_CALIBRATION_PATH")
        deterministic_threshold = float(os.getenv("DETERMINISTIC_THRESHOLD", "0.55"))
        deterministic_profile = os.getenv("DETERMINISTIC_PROFILE", DEFAULT_ANALYSIS_PROFILE)
        raw_max_side = os.getenv("DETERMINISTIC_MAX_SIDE")
        try:
            deterministic_max_side = int(raw_max_side) if raw_max_side else None
        except ValueError:
            deterministic_max_side = None

        self._deterministic_evaluator = (
            StaticRiskEvaluator(
                threshold=deterministic_threshold,
                use_multiview=None,
                parallel=True,
                calibration_path=calibration_path,
                profile=deterministic_profile,
                max_image_side=deterministic_max_side,
            )
            if deterministic_enabled
            else None
        )

    def _build_neural_variants(self, image: Image.Image) -> dict[str, Image.Image]:
        rgb = image.convert("RGB")
        return {
            "original": rgb,
            "autocontrast": ImageOps.autocontrast(rgb, cutoff=1),
            "equalized": ImageOps.equalize(rgb),
            "sharpened": rgb.filter(ImageFilter.UnsharpMask(radius=1, percent=130, threshold=2)),
        }

    def _aggregate_variant_score(self, scores: dict[str, float]) -> float:
        weighted_sum = 0.0
        weight_sum = 0.0
        for name, score in scores.items():
            weight = float(self.neural_variant_weights.get(name, 0.0))
            weighted_sum += weight * float(score)
            weight_sum += weight

        if weight_sum <= 0.0:
            if not scores:
                return 0.0
            return float(sum(scores.values()) / len(scores))

        return float(weighted_sum / weight_sum)

    def _run_neural_variant_voting(
        self,
        image: Image.Image,
    ) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]]]:
        variants = self._build_neural_variants(image)

        variant_scores: dict[str, float] = {}
        variant_predictions: list[dict[str, Any]] = []
        base_predictions: list[dict[str, Any]] = []
        base_label = "unknown"

        for name, variant in variants.items():
            predictions = self._predict(variant)
            label, score = self._select_target_score(predictions)
            variant_scores[name] = score

            variant_predictions.append(
                {
                    "label": f"neural_variant:{name}",
                    "score": float(score),
                }
            )

            if name == "original":
                base_predictions = predictions
                base_label = label

        ensemble_score = self._aggregate_variant_score(variant_scores)
        return base_label, ensemble_score, base_predictions, variant_predictions

    def _run_deterministic_risk(self, image: Image.Image) -> dict[str, Any]:
        if self._deterministic_evaluator is None:
            return {"risk": 0.0, "scores": {}}

        try:
            import cv2

            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            result = self._deterministic_evaluator.evaluate(bgr)
            return {
                "risk": float(result.get("risk", 0.0)),
                "scores": dict(result.get("scores", {})),
            }
        except Exception:
            return {"risk": 0.0, "scores": {}}

    def _fuse_scores(self, neural_score: float, deterministic_score: float) -> float:
        if self._deterministic_evaluator is None:
            return float(np.clip(neural_score, 0.0, 1.0))

        fused = self.neural_weight * neural_score + self.deterministic_weight * deterministic_score

        # If signals diverge strongly, trust neural slightly more while keeping deterministic influence.
        disagreement = abs(neural_score - deterministic_score)
        if disagreement > 0.55:
            fused = 0.70 * neural_score + 0.30 * fused

        return float(np.clip(fused, 0.0, 1.0))

    def evaluate(self, image: Image.Image) -> RecognitionDecision:
        det_future = None
        with ThreadPoolExecutor(max_workers=1) as executor:
            if self._deterministic_evaluator is not None:
                det_future = executor.submit(self._run_deterministic_risk, image.copy())

            base_label, neural_score, base_predictions, variant_predictions = self._run_neural_variant_voting(image)

            deterministic = det_future.result() if det_future is not None else {"risk": 0.0, "scores": {}}

        deterministic_score = float(deterministic.get("risk", 0.0))
        fused_score = self._fuse_scores(neural_score, deterministic_score)

        final_label = self.target_label if fused_score >= 0.5 else base_label

        should_change = (
            final_label.casefold() == self.target_label.casefold()
            and fused_score >= self.threshold
        )

        predictions: list[dict[str, Any]] = list(base_predictions)
        predictions.extend(variant_predictions)
        predictions.append({"label": "neural_ensemble_risk", "score": float(neural_score)})

        if self._deterministic_evaluator is not None:
            predictions.append({"label": "deterministic_risk", "score": deterministic_score})

            det_scores = deterministic.get("scores", {})
            if isinstance(det_scores, dict) and det_scores:
                top_feature = max(det_scores.items(), key=lambda kv: float(kv[1]))
                predictions.append(
                    {
                        "label": "deterministic_top_feature",
                        "score": float(top_feature[1]),
                        "feature": str(top_feature[0]),
                    }
                )

        predictions.append({"label": "fused_deepfake_risk", "score": float(fused_score)})

        return RecognitionDecision(
            should_change=should_change,
            label=final_label,
            score=float(fused_score),
            predictions=predictions,
        )


class NsfwRecognizer(_BasePipelineRecognizer):
    def __init__(
        self,
        model_name: str = DEFAULT_NSFW_MODEL,
        device: int = 0,
        target_label: str = "nsfw",
        threshold: float = 0.5,
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
