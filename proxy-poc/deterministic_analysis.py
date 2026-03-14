from __future__ import annotations

import cv2
import numpy as np
from typing import Dict


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
    - Outside the band, the score ramps to 1.0 over ``outer_width``.
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


# Laplacian sharpness detects edge sharpness
def analyze_laplacian(bgr: np.ndarray) -> float:
    """
    Deepfakes often show *unnatural* sharpness — either too crisp at
    face boundaries (hard blend seam) or too blurry (over-smoothed skin).

    Returns a score where values near 0 or near 1 are suspicious:

    Uses a log-domain variance metric with a soft two-tailed mapping so
    the feature keeps signal in typical ranges instead of collapsing to 0.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())

    # Typical portrait sharpness spans a broad range; log scaling stabilizes
    # camera/resize differences while preserving blur/oversharpening outliers.
    log_variance = float(np.log10(variance + 1.0))
    log_low = float(np.log10(80.0 + 1.0))
    log_high = float(np.log10(550.0 + 1.0))

    score = _soft_two_tailed_score(
        log_variance,
        low=log_low,
        high=log_high,
        outer_width=0.25,
    )

    return float(np.clip(score, 0.0, 1.0))


def analyze_sobel(bgr: np.ndarray) -> float:
    """
    Detects gradient discontinuities using Sobel operators on X and Y axes.

    Face-swap boundaries produce localised gradient spikes that are
    inconsistent with the gradient distribution of authentic images.

    Score = ratio of "spike" pixels (gradient > 2-sigma) to total pixels.
    A high ratio indicates sharp, synthetic-looking edges.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    mean, std = magnitude.mean(), magnitude.std()
    spike_mask = magnitude > (mean + 2 * std)
    spike_ratio = spike_mask.sum() / magnitude.size

    # Natural images: ~2–5 % spike ratio. -> 10 % is suspicious.
    score = np.clip((spike_ratio - 0.02) / 0.08, 0.0, 1.0)
    return float(score)


def analyze_dft(bgr: np.ndarray) -> float:
    """
    Detects frequency domain artifacts
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft[:, :, 0] + 1j * dft[:, :, 1])
    magnitude = np.abs(dft_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    hf_mask = dist > 0.6 * max_dist

    hf_energy = magnitude[hf_mask].sum()
    total_energy = magnitude.sum() + 1e-9

    hf_ratio = hf_energy / total_energy

    # Natural images: hf_ratio ~0.05–0.15. ; 0.20 - suspicious.
    score = np.clip((hf_ratio - 0.10) / 0.15, 0.0, 1.0)
    return float(score)


def analyze_fft(bgr: np.ndarray) -> float:
    """
    Uses NumPy's FFT to detect periodic spectral peaks introduced by
    up-sampling layers in GAN decoders (the "checkerboard" artefact).

    Measures the coefficient of variation of peak energies across angular
    sectors of the magnitude spectrum.  A high CV indicates periodic,
    non-natural spectral structure.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    num_sectors = 8
    sector_energies = []
    Y, X = np.ogrid[:h, :w]
    angles = np.arctan2(Y - cy, X - cx) 

    for i in range(num_sectors):
        lo = -np.pi + i * (2 * np.pi / num_sectors)
        hi = lo + (2 * np.pi / num_sectors)
        sector_mask = (angles >= lo) & (angles < hi)
        sector_energies.append(magnitude[sector_mask].mean())

    sector_energies = np.array(sector_energies)
    cv_score = sector_energies.std() / (sector_energies.mean() + 1e-9)

    # Natural images: low angular variance.  cv_score > 0.5 is suspicious.
    score = np.clip(cv_score / 1.0, 0.0, 1.0)
    return float(score)



def analyze_hist(bgr: np.ndarray) -> float:
    """
    Compares the colour histogram distributions across B, G, R channels.

    Face-swap composites often introduce colour temperature mismatches
    between the swapped face and the original background.  We split the
    image into a central face region and an outer background region, then
    compute the Bhattacharyya distance between their histograms.

    A large distance signals a colour discontinuity consistent with
    face-swap post-processing.
    """
    h, w = bgr.shape[:2]

    margin_y, margin_x = h // 4, w // 4
    face_region = bgr[margin_y: h - margin_y, margin_x: w - margin_x]
    bg_mask = np.ones((h, w), dtype=np.uint8)
    bg_mask[margin_y: h - margin_y, margin_x: w - margin_x] = 0
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
        dist = cv2.compareHist(h_face, h_bg, cv2.HISTCMP_BHATTACHARYYA)
        distances.append(dist)

    avg_distance = np.mean(distances)

    # We treat both unusually low and unusually high face-vs-background
    # distances as potentially suspicious, with a smooth two-tailed mapping.
    score = _soft_two_tailed_score(
        avg_distance,
        low=0.20,
        high=0.38,
        outer_width=0.22,
    )
    return float(score)



def analyze_hsv_skin(bgr: np.ndarray) -> float:
    """
    Converts to HSV and analyses the consistency of skin-tone hue and
    saturation within the detected skin region.

    Face swaps often produce skin pixels with an inconsistent saturation
    distribution (either too uniform — over-processed — or too scattered).
    We measure the coefficient of variation of saturation in the skin mask.

    High CV means unnaturally varied skin saturation (suspicious).
    Very low CV means artificially uniform skin (also suspicious).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    skin_pixels = hsv[:, :, 1][skin_mask > 0].astype(np.float32)  # Saturation channel

    if skin_pixels.size < 100:
        return 0.0  # No significant skin region found

    mean_sat = float(skin_pixels.mean())
    std_sat = float(skin_pixels.std())
    cv_sat = std_sat / (mean_sat + 1e-9)

    p10 = float(np.percentile(skin_pixels, 10))
    p90 = float(np.percentile(skin_pixels, 90))
    sat_span = (p90 - p10) / (mean_sat + 1e-9)

    # Blend two complementary saturation-shape cues.
    cv_score = _soft_two_tailed_score(
        cv_sat,
        low=0.30,
        high=0.52,
        outer_width=0.20,
    )
    span_score = _soft_two_tailed_score(
        sat_span,
        low=0.75,
        high=1.15,
        outer_width=0.35,
    )
    score = 0.60 * cv_score + 0.40 * span_score

    return float(np.clip(score, 0.0, 1.0))


def analyze_noise_residual(bgr: np.ndarray) -> float:
    """
    Extracts the noise residual by subtracting a denoised version of the
    image from the original.

    GAN generators produce images with a distinctive noise pattern that
    differs from camera sensor noise.  The key signal is the spatial
    variance of the noise residual: GAN noise tends to be more spatially
    structured (lower local variance, higher global pattern).

    We also compute the kurtosis of the residual histogram — authentic
    camera noise is approximately Gaussian (kurtosis ≈ 0), while GAN
    noise is often heavier-tailed or flatter.
    """
    # h=3, hColor=3 → light denoising to preserve subtle noise patterns
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h=3, hColor=3,
                                               templateWindowSize=7,
                                               searchWindowSize=21)
    residual = bgr.astype(np.float32) - denoised.astype(np.float32)
    flat = residual.flatten()

    # Kurtosis of residual (excess kurtosis: 0 = Gaussian)
    mean_r = flat.mean()
    std_r = flat.std() + 1e-9
    kurtosis = np.mean(((flat - mean_r) / std_r) ** 4) - 3.0

    # Authentic noise: |kurtosis| < 1.  GAN noise: often > 2 or < -1.
    score = np.clip(abs(kurtosis) / 4.0, 0.0, 1.0)
    return float(score)


# Aggregator - this would be plugged into the risk evaluator for the detection model

# Default weights per analyser (must sum to 1.0).
# Tune these based on your validation dataset.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "laplacian":     0.10,
    "sobel":         0.10,
    "dft":           0.20,
    "fft":           0.20,
    "hist":          0.15,
    "hsv_skin":      0.10,
    "noise_residual":0.15,
}


def run_all_analyzers(bgr: np.ndarray) -> Dict[str, float]:
    """
    Run every static analyser and return a dict of individual scores.

    Parameters
    ----------
    bgr : np.ndarray
        A BGR image as returned by cv2.imread or cv2.VideoCapture.read().

    Returns
    -------
    Dict[str, float]
        Mapping of analyser name → score in [0.0, 1.0].
    """
    return {
        "laplacian":      analyze_laplacian(bgr),
        "sobel":          analyze_sobel(bgr),
        "dft":            analyze_dft(bgr),
        "fft":            analyze_fft(bgr),
        "hist":           analyze_hist(bgr),
        "hsv_skin":       analyze_hsv_skin(bgr),
        "noise_residual": analyze_noise_residual(bgr),
    }


def aggregate_risk(
    scores: Dict[str, float],
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> float:
    """
    Weighted average of individual analyser scores.

    Parameters
    ----------
    scores  : output of run_all_analyzers()
    weights : per-analyser weights (must sum to 1.0)

    Returns
    -------
    float
        Aggregate risk score in [0.0, 1.0].
    """
    total = sum(weights[k] * scores[k] for k in weights if k in scores)
    weight_sum = sum(weights[k] for k in weights if k in scores)
    return float(total / weight_sum) if weight_sum > 0 else 0.0


class StaticRiskEvaluator:
    """
    Parameters
    ----------
    threshold : float
        Risk score above which a frame is flagged (0.0–1.0).
    weights : dict, optional
        Per-analyser weights.  Defaults to DEFAULT_WEIGHTS.
    """

    def __init__(self, threshold: float = 0.55,
                 weights: Dict[str, float] = None):
        self.threshold = threshold
        self.weights = weights or DEFAULT_WEIGHTS

    def evaluate(self, bgr: np.ndarray) -> dict:
        """
        Returns
        -------
        dict with keys:
            scores      – individual analyser scores
            risk        – aggregate risk float
            is_deepfake – bool, True if risk >= threshold
        """
        scores = run_all_analyzers(bgr)
        risk = aggregate_risk(scores, self.weights)
        return {
            "scores": scores,
            "risk": risk,
            "is_deepfake": risk >= self.threshold,
        }



# run this evaluator on an image with `python deterministic_analysis.py [path_to_image]`
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Could not read image: {path}")
            sys.exit(1)
    else:
        # Synthetic test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    evaluator = StaticRiskEvaluator(threshold=0.55)
    result = evaluator.evaluate(frame)

    print("\n── Static Analysis Results ──────────────────")
    for name, score in result["scores"].items():
        bar = "█" * int(score * 20)
        print(f"  {name:<20} {score:.3f}  {bar}")
    print(f"\n  Aggregate Risk : {result['risk']:.3f}")
    print(f"  Is Deepfake    : {result['is_deepfake']}")
    print(f"  Threshold      : {evaluator.threshold}")
    print("─────────────────────────────────────────────\n")