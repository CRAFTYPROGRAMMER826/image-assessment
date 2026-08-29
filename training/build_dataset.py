from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from app.core.features import FEATURE_NAMES, extract_features
from training.generate_degradations import LABELS, labels_for, transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEVERITIES = ("low", "medium", "high")
SPLIT_FILES = {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}


def sanity_check(path: Path) -> tuple[bool, str]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return False, "unreadable"
    height, width = image.shape[:2]
    if min(height, width) < 64:
        return False, "tiny_dimensions"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray <= 5)) > 0.85:
        return False, "extreme_black_clipping"
    if float(np.mean(gray >= 250)) > 0.85:
        return False, "extreme_white_clipping"
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    edge_density = float(np.mean(cv2.Canny(gray, 100, 200) > 0))
    if laplacian_variance < 2.0 and edge_density < 0.002:
        return False, "extreme_low_sharpness"
    return True, "accepted"


def _write_split_files(assignments: dict[str, list[Path]], split_dir: Path, source_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, paths in assignments.items():
        relative = [str(path.relative_to(source_dir)) for path in paths]
        (split_dir / SPLIT_FILES[split]).write_text("\n".join(relative) + "\n")


def create_splits(source_dir: Path, split_dir: Path, seed: int, limit: int | None) -> tuple[dict[str, list[Path]], list[dict[str, str]]]:
    candidates = sorted(path for path in source_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if limit:
        candidates = candidates[:limit]
    accepted: list[Path] = []
    rejected: list[dict[str, str]] = []
    for index, path in enumerate(candidates, 1):
        valid, reason = sanity_check(path)
        if valid:
            accepted.append(path)
        else:
            rejected.append({"source_image": str(path.relative_to(source_dir)), "reason": reason})
        if index % 500 == 0:
            print(f"Sanity checked {index}/{len(candidates)} source images", flush=True)
    if len(accepted) < 20:
        raise SystemExit("Fewer than 20 source images passed the conservative sanity filter")
    random.Random(seed).shuffle(accepted)
    train_end = int(len(accepted) * 0.70)
    validation_end = train_end + int(len(accepted) * 0.15)
    assignments = {
        "train": accepted[:train_end],
        "validation": accepted[train_end:validation_end],
        "test": accepted[validation_end:],
    }
    _write_split_files(assignments, split_dir, source_dir)
    return assignments, rejected


def load_splits(source_dir: Path, split_dir: Path) -> dict[str, list[Path]]:
    assignments: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for split, filename in SPLIT_FILES.items():
        entries = [line.strip() for line in (split_dir / filename).read_text().splitlines() if line.strip()]
        paths = [(source_dir / entry).resolve() for entry in entries]
        missing = [path for path in paths if not path.exists()]
        overlap = seen.intersection(paths)
        if missing:
            raise SystemExit(f"Persisted {split} split contains {len(missing)} missing files; use --reset-splits")
        if overlap:
            raise SystemExit(f"Source leakage in persisted splits: {sorted(map(str, overlap))[:3]}")
        seen.update(paths)
        assignments[split] = paths
    return assignments


def variant_rows(image: np.ndarray, source_name: str, split: str, seed: int) -> list[dict]:
    source_hash = int(hashlib.sha256(source_name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed ^ source_hash)
    rows: list[dict] = []
    variants = [("clean", 0, "none")]
    variants.extend((kind, level, SEVERITIES[level]) for kind in LABELS for level in range(3))
    variants.extend((kind, 0, "mixed") for kind in ("blur_noise", "under_noise", "over_blur"))
    for kind, level, severity in variants:
        degraded = transform(image, kind, level, rng)
        rows.append(
            {
                "source_image": source_name,
                "source_id": Path(source_name).stem,
                "split": split,
                "degradation": kind,
                "severity": severity,
                **extract_features(degraded),
                **labels_for(kind),
            }
        )
    return rows


def process_source(job: tuple[str, str, int]) -> list[dict]:
    path_string, split, seed = job
    cv2.setNumThreads(1)
    path = Path(path_string)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    return variant_rows(image, path.name, split, seed)


def save_inspection_samples(source: Path, output: Path, seed: int) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        return
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "clean.jpg"), image)
    for kind in LABELS:
        for level, severity in enumerate(SEVERITIES):
            cv2.imwrite(str(output / f"{kind}_{severity}.jpg"), transform(image, kind, level, rng))
    for kind in ("blur_noise", "under_noise", "over_blur"):
        cv2.imwrite(str(output / f"{kind}.jpg"), transform(image, kind, 0, rng))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an in-memory synthetic feature dataset from clean source images")
    parser.add_argument("--input", type=Path, default=Path("val2017"))
    parser.add_argument("--output", type=Path, default=Path("data/features.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--report", type=Path, default=Path("data/dataset_report.json"))
    parser.add_argument("--samples", type=Path, default=Path("samples/generated"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--limit", type=int, default=None, help="Development-only source image limit")
    parser.add_argument("--reset-splits", action="store_true")
    args = parser.parse_args()

    split_paths = [args.splits / filename for filename in SPLIT_FILES.values()]
    if all(path.exists() for path in split_paths) and not args.reset_splits and args.limit is None:
        assignments = load_splits(args.input.resolve(), args.splits)
        rejected: list[dict[str, str]] = []
        print("Reusing persisted source-level splits", flush=True)
    else:
        assignments, rejected = create_splits(args.input.resolve(), args.splits, args.seed, args.limit)

    jobs = [(str(path), split, args.seed) for split, paths in assignments.items() for path in paths]
    header = ["source_image", "source_id", "split", "degradation", "severity", *FEATURE_NAMES, *LABELS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    row_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {split: Counter() for split in assignments}
    processed = 0
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for rows in executor.map(process_source, jobs, chunksize=4):
                writer.writerows(rows)
                if rows:
                    split = rows[0]["split"]
                    row_counts[split] += len(rows)
                    for row in rows:
                        for label in LABELS:
                            label_counts[split][label] += int(row[label])
                processed += 1
                if processed % 100 == 0 or processed == len(jobs):
                    print(f"Extracted variants for {processed}/{len(jobs)} sources", flush=True)
    temporary.replace(args.output)
    first_train = assignments["train"][0]
    save_inspection_samples(first_train, args.samples, args.seed)

    rejection_counts = Counter(item["reason"] for item in rejected)
    report = {
        "source_directory": str(args.input.resolve()),
        "seed": args.seed,
        "variants_per_source": 19,
        "source_counts": {split: len(paths) for split, paths in assignments.items()},
        "feature_row_counts": dict(row_counts),
        "positive_label_counts": {split: dict(counts) for split, counts in label_counts.items()},
        "rejected_source_count": len(rejected),
        "rejection_reasons": dict(rejection_counts),
        "feature_dataset": str(args.output.resolve()),
        "inspection_samples": str(args.samples.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    if rejected:
        with (args.report.parent / "rejected_sources.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_image", "reason"])
            writer.writeheader()
            writer.writerows(rejected)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
