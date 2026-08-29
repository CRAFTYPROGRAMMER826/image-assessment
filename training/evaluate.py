"""Evaluation is performed by training/train_model.py on the untouched source-level test split.

This module provides a convenient report printer for an existing metrics artifact.
"""
from __future__ import annotations

import json
from pathlib import Path


if __name__ == "__main__":
    path = Path("evaluation/metrics.json")
    if not path.exists():
        raise SystemExit("Run `uv run python -m training.train_model` first.")
    print(json.dumps(json.loads(path.read_text()), indent=2))
