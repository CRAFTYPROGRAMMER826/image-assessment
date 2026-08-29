from __future__ import annotations

import cv2
import numpy as np


FEATURE_NAMES = [
    "laplacian_variance",
    "tenengrad_score",
    "edge_density",
    "high_frequency_energy",
    "mean_luminance",
    "std_luminance",
    "black_clip_ratio",
    "white_clip_ratio",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "rms_contrast",
    "dynamic_range",
    "noise_sigma_estimate",
    "local_variance_mean",
    "entropy",
    "gradient_mean",
    "gradient_std",
    "mean_saturation",
    "saturation_std",
]


def _entropy(gray: np.ndarray) -> float:
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def extract_features(image: np.ndarray) -> dict[str, float]:
    """Extract fixed-order, interpretable quality features from a BGR image."""
    if image is None or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("A non-empty grayscale or BGR image is required")

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    gray_f = gray.astype(np.float32)

    laplacian = cv2.Laplacian(gray_f, cv2.CV_32F)
    grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    edges = cv2.Canny(gray, 100, 200)

    blurred = cv2.GaussianBlur(gray_f, (5, 5), 0)
    residual = gray_f - blurred
    local_mean = cv2.boxFilter(gray_f, cv2.CV_32F, (7, 7))
    local_sq_mean = cv2.boxFilter(gray_f * gray_f, cv2.CV_32F, (7, 7))
    local_variance = np.maximum(local_sq_mean - local_mean * local_mean, 0)

    p01, p99 = np.percentile(gray_f, [1, 99])
    luminance_mean = float(gray_f.mean())
    luminance_std = float(gray_f.std())
    saturation = (
        np.zeros_like(gray_f)
        if image.ndim == 2
        else cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    )

    values = {
        "laplacian_variance": float(laplacian.var()),
        "tenengrad_score": float(np.mean(gradient * gradient)),
        "edge_density": float(np.mean(edges > 0)),
        "high_frequency_energy": float(np.mean(np.abs(laplacian))),
        "mean_luminance": luminance_mean,
        "std_luminance": luminance_std,
        "black_clip_ratio": float(np.mean(gray <= 5)),
        "white_clip_ratio": float(np.mean(gray >= 250)),
        "dark_pixel_ratio": float(np.mean(gray <= 40)),
        "bright_pixel_ratio": float(np.mean(gray >= 215)),
        "rms_contrast": luminance_std,
        "dynamic_range": float(p99 - p01),
        "noise_sigma_estimate": float(np.median(np.abs(residual)) / 0.6745),
        "local_variance_mean": float(local_variance.mean()),
        "entropy": _entropy(gray),
        "gradient_mean": float(gradient.mean()),
        "gradient_std": float(gradient.std()),
        "mean_saturation": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
    }
    return {name: values[name] for name in FEATURE_NAMES}
