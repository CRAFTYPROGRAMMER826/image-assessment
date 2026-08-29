from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABELS = ["blur", "underexposure", "overexposure", "noise", "severe_degradation"]


def split_for(source_id: str) -> str:
    """Stable source-level split; every derivative of one source stays together."""
    bucket = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def apply_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def apply_underexposure(image: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_overexposure(image: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def apply_severe_degradation(image: np.ndarray, level: int) -> np.ndarray:
    scale = (6, 12, 20)[level]
    small = cv2.resize(
        image,
        (max(8, image.shape[1] // scale), max(8, image.shape[0] // scale)),
        interpolation=cv2.INTER_AREA,
    )
    pixelated = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    quality = (24, 12, 5)[level]
    ok, encoded = cv2.imencode(".jpg", pixelated, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return pixelated if not ok else cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def transform(image: np.ndarray, kind: str, level: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "clean":
        return image.copy()
    if kind == "blur":
        sigma = (1.0, 2.0, 4.0)[level]
        return apply_gaussian_blur(image, sigma)
    if kind == "underexposure":
        return apply_underexposure(image, (0.7, 0.4, 0.2)[level])
    if kind == "overexposure":
        return apply_overexposure(image, (1.3, 1.7, 2.2)[level])
    if kind == "noise":
        return apply_gaussian_noise(image, (8, 18, 35)[level], rng)
    if kind == "severe_degradation":
        return apply_severe_degradation(image, level)
    if kind == "blur_noise":
        return apply_gaussian_noise(apply_gaussian_blur(image, 3), 25, rng)
    if kind == "under_noise":
        return apply_gaussian_noise(apply_underexposure(image, 0.35), 18, rng)
    if kind == "over_blur":
        return apply_gaussian_blur(apply_overexposure(image, 1.8), 2)
    raise ValueError(f"Unknown degradation: {kind}")


def labels_for(kind: str) -> dict[str, int]:
    labels = {name: 0 for name in LABELS}
    mappings = {
        "blur": ["blur"],
        "underexposure": ["underexposure"],
        "overexposure": ["overexposure"],
        "noise": ["noise"],
        "severe_degradation": ["severe_degradation"],
        "blur_noise": ["blur", "noise"],
        "under_noise": ["underexposure", "noise"],
        "over_blur": ["overexposure", "blur"],
    }
    for name in mappings.get(kind, []):
        labels[name] = 1
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Create source-separated synthetic image degradations")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = sorted(path for path in args.input.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(sources) < 20:
        raise SystemExit("At least 20 diverse clean source images are required; 100+ is strongly recommended.")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    random.Random(args.seed).shuffle(sources)
    rows: list[dict[str, str | int]] = []
    variants = ["clean", "blur", "underexposure", "overexposure", "noise", "severe_degradation"]
    combinations = ["blur_noise", "under_noise", "over_blur"]
    for source in sources:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < 32:
            continue
        source_id = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        split = split_for(source_id)
        for kind in variants:
            levels = range(1) if kind == "clean" else range(3)
            for level in levels:
                result = transform(image, kind, level, rng)
                destination = args.output / split / f"{source_id}_{kind}_{level}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(destination), result, [cv2.IMWRITE_JPEG_QUALITY, 94])
                rows.append({"path": str(destination), "source_id": source_id, "split": split, **labels_for(kind)})
        for kind in combinations:
            destination = args.output / split / f"{source_id}_{kind}_0.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), transform(image, kind, 0, rng), [cv2.IMWRITE_JPEG_QUALITY, 94])
            rows.append({"path": str(destination), "source_id": source_id, "split": split, **labels_for(kind)})
    if not rows:
        raise SystemExit("No readable source images found")
    with (args.output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} examples from {len(set(row['source_id'] for row in rows))} sources")


if __name__ == "__main__":
    main()
