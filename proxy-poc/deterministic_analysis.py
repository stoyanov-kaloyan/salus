from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping

import cv2
import numpy as np


AnalyzerFn = Callable[[np.ndarray], float]


def _soft_two_tailed_score(
    value: float,
    low: float,
    high: float,
    outer_width: float,
) -> float:
    """
    Convert a scalar metric to a [0, 1] anomaly score with a soft two-tailed profile.

    - Inside [low, high], the score grows smoothly from 0 at band center
      to 0.35 at the band edges.
    - Outside the band, the score ramps to 1.0 over outer_width.
    """
    if high <= low:
        return 0.0

    mid = 0.5 * (low + high)
    half_range = 0.5 * (high - low) + 1e-9
    in_band_distance = abs(value - mid) / half_range

    if low <= value <= high:
        return float(np.clip(0.35 * in_band_distance, 0.0, 1.0))

    if value < low:
        outside_distance = (low - value) / (outer_width + 1e-9)
    else:
        outside_distance = (value - high) / (outer_width + 1e-9)

    return float(np.clip(0.35 + 0.65 * outside_distance, 0.0, 1.0))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def _default_workers(cap: int) -> int:
    cpu = os.cpu_count() or 4
    return max(2, min(cap, cpu))


def _center_crop_region(bgr: np.ndarray) -> tuple[int, int, int, int]:
    h, w = bgr.shape[:2]
    margin_y, margin_x = h // 4, w // 4
    return margin_y, h - margin_y, margin_x, w - margin_x


def analyze_laplacian(bgr: np.ndarray) -> float:
    """
    Blur/oversharpening signal from Laplacian variance.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())

    log_variance = float(np.log10(variance + 1.0))
    log_low = float(np.log10(80.0 + 1.0))
    log_high = float(np.log10(550.0 + 1.0))
    score = _soft_two_tailed_score(log_variance, low=log_low, high=log_high, outer_width=0.25)
    return float(np.clip(score, 0.0, 1.0))


def analyze_sobel(bgr: np.ndarray) -> float:
    """
    Gradient spike ratio around seams.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    mean, std = float(magnitude.mean()), float(magnitude.std())
    spike_mask = magnitude > (mean + 2.0 * std)
    spike_ratio = float(spike_mask.sum() / magnitude.size)
    score = np.clip((spike_ratio - 0.02) / 0.08, 0.0, 1.0)
    return float(score)


def analyze_dft(bgr: np.ndarray) -> float:
    """
    High-frequency energy ratio in DFT domain.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft[:, :, 0] + 1j * dft[:, :, 1])
    magnitude = np.abs(dft_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    hf_mask = dist > 0.6 * max_dist

    hf_energy = float(magnitude[hf_mask].sum())
    total_energy = float(magnitude.sum()) + 1e-9
    hf_ratio = hf_energy / total_energy
    score = np.clip((hf_ratio - 0.10) / 0.15, 0.0, 1.0)
    return float(score)


def analyze_fft(bgr: np.ndarray) -> float:
    """
    Angular anisotropy in FFT spectrum.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    num_sectors = 8
    sector_energies = []
    y_grid, x_grid = np.ogrid[:h, :w]
    angles = np.arctan2(y_grid - cy, x_grid - cx)

    for i in range(num_sectors):
        lo = -np.pi + i * (2.0 * np.pi / num_sectors)
        hi = lo + (2.0 * np.pi / num_sectors)
        sector_mask = (angles >= lo) & (angles < hi)
        sector_energies.append(float(magnitude[sector_mask].mean()))

    sectors = np.array(sector_energies, dtype=np.float32)
    cv_score = float(sectors.std() / (sectors.mean() + 1e-9))
    return float(np.clip(cv_score / 1.0, 0.0, 1.0))


def analyze_hist(bgr: np.ndarray) -> float:
    """
    Face-center vs outer-region color histogram mismatch.
    """
    h, w = bgr.shape[:2]
    y0, y1, x0, x1 = _center_crop_region(bgr)
    face_region = bgr[y0:y1, x0:x1]

    bg_mask = np.ones((h, w), dtype=np.uint8)
    bg_mask[y0:y1, x0:x1] = 0
    background = bgr[bg_mask == 1].reshape(-1, 1, 3)
    if background.shape[0] < 100:
        return 0.0

    distances = []
    for ch in range(3):
        h_face = cv2.calcHist([face_region], [ch], None, [64], [0, 256])
        bg_img = bgr.copy()
        bg_img[bg_mask == 0] = 0
        h_bg = cv2.calcHist([bg_img], [ch], bg_mask, [64], [0, 256])

        cv2.normalize(h_face, h_face)
        cv2.normalize(h_bg, h_bg)
        distances.append(float(cv2.compareHist(h_face, h_bg, cv2.HISTCMP_BHATTACHARYYA)))

    avg_distance = float(np.mean(distances))
    score = _soft_two_tailed_score(avg_distance, low=0.20, high=0.38, outer_width=0.22)
    return float(score)


def analyze_hsv_skin(bgr: np.ndarray) -> float:
    """
    Skin-region saturation consistency.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_pixels = hsv[:, :, 1][skin_mask > 0].astype(np.float32)

    if skin_pixels.size < 100:
        return 0.0

    mean_sat = float(skin_pixels.mean())
    std_sat = float(skin_pixels.std())
    cv_sat = std_sat / (mean_sat + 1e-9)
    p10 = float(np.percentile(skin_pixels, 10))
    p90 = float(np.percentile(skin_pixels, 90))
    sat_span = (p90 - p10) / (mean_sat + 1e-9)

    cv_score = _soft_two_tailed_score(cv_sat, low=0.30, high=0.52, outer_width=0.20)
    span_score = _soft_two_tailed_score(sat_span, low=0.75, high=1.15, outer_width=0.35)
    score = 0.60 * cv_score + 0.40 * span_score
    return float(np.clip(score, 0.0, 1.0))


def analyze_noise_residual(bgr: np.ndarray) -> float:
    """
    Residual kurtosis after denoising.
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    residual = bgr.astype(np.float32) - denoised.astype(np.float32)
    flat = residual.flatten()
    mean_r = float(flat.mean())
    std_r = float(flat.std()) + 1e-9
    kurtosis = float(np.mean(((flat - mean_r) / std_r) ** 4) - 3.0)
    score = np.clip(abs(kurtosis) / 4.0, 0.0, 1.0)
    return float(score)


def analyze_jpeg_blocking(bgr: np.ndarray) -> float:
    """
    Detect JPEG block boundary energy imbalance.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    if h < 24 or w < 24:
        return 0.0

    diff_v = np.abs(np.diff(gray, axis=1))
    diff_h = np.abs(np.diff(gray, axis=0))

    cols = np.arange(diff_v.shape[1])
    rows = np.arange(diff_h.shape[0])
    v_boundary_mask = (cols % 8) == 7
    h_boundary_mask = (rows % 8) == 7

    if not np.any(v_boundary_mask) or not np.any(h_boundary_mask):
        return 0.0

    boundary_energy = float(diff_v[:, v_boundary_mask].mean() + diff_h[h_boundary_mask, :].mean())
    interior_energy = float(diff_v[:, ~v_boundary_mask].mean() + diff_h[~h_boundary_mask, :].mean()) + 1e-9
    ratio = boundary_energy / interior_energy

    return _soft_two_tailed_score(ratio, low=0.95, high=1.18, outer_width=0.40)


def analyze_edge_ringing(bgr: np.ndarray) -> float:
    """
    Approximate halo/ringing around high-contrast edges.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    high_pass = np.abs(gray_f - cv2.GaussianBlur(gray_f, (0, 0), 1.2))

    edges = cv2.Canny(gray, 80, 180)
    if int(edges.sum()) == 0:
        return 0.0

    dilate_inner = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    dilate_outer = cv2.dilate(edges, np.ones((7, 7), dtype=np.uint8), iterations=1) > 0
    ring_band = dilate_outer & (~dilate_inner)
    edge_band = dilate_inner

    if int(ring_band.sum()) < 50 or int(edge_band.sum()) < 50:
        return 0.0

    ring_strength = float(high_pass[ring_band].mean())
    edge_strength = float(high_pass[edge_band].mean()) + 1e-9
    ratio = ring_strength / edge_strength

    return _soft_two_tailed_score(ratio, low=0.20, high=0.55, outer_width=0.45)


def analyze_texture_inconsistency(bgr: np.ndarray) -> float:
    """
    Local texture variance mismatch between center and outer region.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(gray, (0, 0), 3.0)
    sq_mean = cv2.GaussianBlur(gray * gray, (0, 0), 3.0)
    local_var = np.clip(sq_mean - (mean * mean), 0.0, None)

    y0, y1, x0, x1 = _center_crop_region(bgr)
    face_var = float(local_var[y0:y1, x0:x1].mean())

    bg_mask = np.ones(gray.shape, dtype=bool)
    bg_mask[y0:y1, x0:x1] = False
    bg_var = float(local_var[bg_mask].mean()) + 1e-9
    ratio = face_var / bg_var

    return _soft_two_tailed_score(ratio, low=0.65, high=1.35, outer_width=0.75)


ANALYZERS: Dict[str, AnalyzerFn] = {
    "laplacian": analyze_laplacian,
    "sobel": analyze_sobel,
    "dft": analyze_dft,
    "fft": analyze_fft,
    "hist": analyze_hist,
    "hsv_skin": analyze_hsv_skin,
    "noise_residual": analyze_noise_residual,
    "jpeg_blocking": analyze_jpeg_blocking,
    "edge_ringing": analyze_edge_ringing,
    "texture_inconsistency": analyze_texture_inconsistency,
}


DEFAULT_WEIGHTS: Dict[str, float] = {
    "laplacian": 0.08,
    "sobel": 0.08,
    "dft": 0.14,
    "fft": 0.14,
    "hist": 0.10,
    "hsv_skin": 0.08,
    "noise_residual": 0.10,
    "jpeg_blocking": 0.10,
    "edge_ringing": 0.09,
    "texture_inconsistency": 0.09,
}


DEFAULT_VARIANT_WEIGHTS: Dict[str, float] = {
    "original": 0.55,
    "denoised": 0.15,
    "sharpened": 0.15,
    "clahe": 0.15,
}


@dataclass(frozen=True)
class CalibrationConfig:
    feature_order: tuple[str, ...]
    weights: np.ndarray
    bias: float
    means: np.ndarray | None
    scales: np.ndarray | None
    threshold: float | None


def load_calibration(calibration_path: str | None) -> CalibrationConfig | None:
    if not calibration_path:
        return None

    path = Path(calibration_path)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_order = tuple(str(name) for name in payload.get("feature_order", []))
    weights = np.array(payload.get("weights", []), dtype=np.float32)
    bias = float(payload.get("bias", 0.0))

    if not feature_order or weights.size != len(feature_order):
        return None

    means_raw = payload.get("means")
    scales_raw = payload.get("scales")
    means = np.array(means_raw, dtype=np.float32) if isinstance(means_raw, list) else None
    scales = np.array(scales_raw, dtype=np.float32) if isinstance(scales_raw, list) else None

    if means is not None and means.size != len(feature_order):
        means = None
    if scales is not None and scales.size != len(feature_order):
        scales = None

    threshold_raw = payload.get("threshold")
    threshold = float(threshold_raw) if threshold_raw is not None else None

    return CalibrationConfig(
        feature_order=feature_order,
        weights=weights,
        bias=bias,
        means=means,
        scales=scales,
        threshold=threshold,
    )


def calibrate_risk(scores: Mapping[str, float], calibration: CalibrationConfig) -> float:
    x = np.array([float(scores.get(name, 0.0)) for name in calibration.feature_order], dtype=np.float32)

    if calibration.means is not None and calibration.scales is not None:
        x = (x - calibration.means) / (calibration.scales + 1e-9)

    logit = float(np.dot(calibration.weights, x) + calibration.bias)
    return _sigmoid(logit)


def _safe_run_analyzer(name: str, fn: AnalyzerFn, bgr: np.ndarray) -> tuple[str, float]:
    try:
        value = float(np.clip(fn(bgr), 0.0, 1.0))
        return name, value
    except Exception:
        return name, 0.0


def run_all_analyzers(
    bgr: np.ndarray,
    parallel: bool = True,
    max_workers: int | None = None,
) -> Dict[str, float]:
    """
    Run every static analyzer and return analyzer_name -> score.
    """
    if not parallel:
        return {
            name: _safe_run_analyzer(name, fn, bgr)[1]
            for name, fn in ANALYZERS.items()
        }

    workers = max_workers or _default_workers(len(ANALYZERS))
    results: Dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_safe_run_analyzer, name, fn, bgr): name
            for name, fn in ANALYZERS.items()
        }
        for future in as_completed(futures):
            name, value = future.result()
            results[name] = value

    return results


def build_image_variants(bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Build deterministic-friendly filtered variants for robust scoring.
    """
    variants: Dict[str, np.ndarray] = {"original": bgr}

    denoised = cv2.bilateralFilter(bgr, d=7, sigmaColor=50, sigmaSpace=50)
    variants["denoised"] = denoised

    blurred = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    sharpened = cv2.addWeighted(bgr, 1.35, blurred, -0.35, 0)
    variants["sharpened"] = np.clip(sharpened, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    clahe_bgr = cv2.cvtColor(cv2.merge([l_eq, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    variants["clahe"] = clahe_bgr

    return variants


def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    positive = {k: float(v) for k, v in weights.items() if float(v) > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        count = len(weights)
        if count == 0:
            return {}
        uniform = 1.0 / count
        return {k: uniform for k in weights}

    return {k: v / total for k, v in positive.items()}


def run_multiview_analyzers(
    bgr: np.ndarray,
    variant_weights: Mapping[str, float] = DEFAULT_VARIANT_WEIGHTS,
    parallel: bool = True,
    max_workers: int | None = None,
) -> Dict[str, float]:
    """
    Run analyzers across filtered views and aggregate feature-wise scores.
    """
    variants = build_image_variants(bgr)
    weights = _normalize_weights({name: variant_weights.get(name, 0.0) for name in variants})

    if not weights:
        weights = {name: 1.0 / len(variants) for name in variants}

    per_variant_scores: Dict[str, Dict[str, float]] = {}

    if parallel:
        workers = max_workers or _default_workers(len(variants))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_all_analyzers, img, False, None): name
                for name, img in variants.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                per_variant_scores[name] = future.result()
    else:
        for name, image in variants.items():
            per_variant_scores[name] = run_all_analyzers(image, parallel=False)

    aggregated: Dict[str, float] = {}
    for feature in ANALYZERS:
        weighted_sum = 0.0
        weight_sum = 0.0
        for variant_name, scores in per_variant_scores.items():
            weight = float(weights.get(variant_name, 0.0))
            weighted_sum += weight * float(scores.get(feature, 0.0))
            weight_sum += weight
        aggregated[feature] = float(weighted_sum / weight_sum) if weight_sum > 0 else 0.0

    return aggregated


def aggregate_risk(
    scores: Mapping[str, float],
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> float:
    """
    Weighted average risk from feature scores.
    """
    total = sum(float(weights[k]) * float(scores[k]) for k in weights if k in scores)
    weight_sum = sum(float(weights[k]) for k in weights if k in scores)
    return float(total / weight_sum) if weight_sum > 0 else 0.0


class StaticRiskEvaluator:
    """
    Deterministic deepfake risk evaluator with optional multiview and calibration.
    """

    def __init__(
        self,
        threshold: float = 0.55,
        weights: Mapping[str, float] | None = None,
        use_multiview: bool = True,
        parallel: bool = True,
        max_workers: int | None = None,
        calibration_path: str | None = None,
    ):
        self.threshold = float(threshold)
        self.weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
        self.use_multiview = bool(use_multiview)
        self.parallel = bool(parallel)
        self.max_workers = max_workers
        self.calibration = load_calibration(calibration_path)

        if self.calibration and self.calibration.threshold is not None:
            self.decision_threshold = float(self.calibration.threshold)
        else:
            self.decision_threshold = self.threshold

    def evaluate(self, bgr: np.ndarray) -> dict:
        if self.use_multiview:
            scores = run_multiview_analyzers(
                bgr,
                variant_weights=DEFAULT_VARIANT_WEIGHTS,
                parallel=self.parallel,
                max_workers=self.max_workers,
            )
        else:
            scores = run_all_analyzers(bgr, parallel=self.parallel, max_workers=self.max_workers)

        if self.calibration is not None:
            risk = calibrate_risk(scores, self.calibration)
        else:
            risk = aggregate_risk(scores, self.weights)

        return {
            "scores": scores,
            "risk": risk,
            "is_deepfake": risk >= self.decision_threshold,
            "threshold": self.decision_threshold,
        }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Could not read image: {path}")
            sys.exit(1)
    else:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    calibration_path = os.getenv("DETERMINISTIC_CALIBRATION_PATH")
    evaluator = StaticRiskEvaluator(
        threshold=0.55,
        use_multiview=True,
        parallel=True,
        calibration_path=calibration_path,
    )
    result = evaluator.evaluate(frame)

    print("\n-- Static Analysis Results -------------------")
    for name, score in result["scores"].items():
        bar = "#" * int(score * 20)
        print(f"  {name:<20} {score:.3f}  {bar}")
    print(f"\n  Aggregate Risk : {result['risk']:.3f}")
    print(f"  Is Deepfake    : {result['is_deepfake']}")
    print(f"  Threshold      : {result['threshold']}")
    print("---------------------------------------------\n")