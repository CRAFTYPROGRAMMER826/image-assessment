from __future__ import annotations

import os

import cv2
import numpy as np

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = (4 if os.getenv("VERCEL") else 10) * 1024 * 1024
MIN_DIMENSION = 16
MAX_PIXELS = 40_000_000


class ImageValidationError(ValueError):
    pass


def decode_image(payload: bytes) -> np.ndarray:
    if not payload:
        raise ImageValidationError("Uploaded file is empty")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("File is not a readable JPEG or PNG image")
    height, width = image.shape[:2]
    if min(height, width) < MIN_DIMENSION:
        raise ImageValidationError(f"Image dimensions must be at least {MIN_DIMENSION}x{MIN_DIMENSION}")
    if height * width > MAX_PIXELS:
        raise ImageValidationError("Decoded image dimensions are too large")
    return image
