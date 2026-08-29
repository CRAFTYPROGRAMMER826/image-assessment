from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from app.core.features import FEATURE_NAMES, extract_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tabular quality features from generated images")
    parser.add_argument("--manifest", type=Path, default=Path("data/generated/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/features.csv"))
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    rows: list[dict] = []
    for record in manifest.to_dict(orient="records"):
        image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {record['path']}")
            continue
        rows.append({**record, **extract_features(image)})
    if not rows:
        raise SystemExit("No features were extracted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ordered = ["path", "source_id", "split", *FEATURE_NAMES, "blur", "underexposure", "overexposure", "noise", "severe_degradation"]
    pd.DataFrame(rows)[ordered].to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
