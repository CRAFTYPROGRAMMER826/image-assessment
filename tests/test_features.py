import cv2
import numpy as np

from app.core.features import FEATURE_NAMES, extract_features


def test_features_have_fixed_order_and_finite_values():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(image, (12, 12), (52, 52), (200, 100, 30), -1)
    values = extract_features(image)
    assert list(values) == FEATURE_NAMES
    assert np.isfinite(list(values.values())).all()


def test_blur_reduces_laplacian_variance():
    checker = (np.indices((96, 96)).sum(axis=0) % 2 * 255).astype(np.uint8)
    sharp = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)
    assert extract_features(blurred)["laplacian_variance"] < extract_features(sharp)["laplacian_variance"]
