from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

from app.core.features import FEATURE_NAMES
from app.core.scoring import TARGET_NAMES


def positive_probabilities(model: MultiOutputClassifier, x: np.ndarray) -> np.ndarray:
    columns = []
    for estimator, matrix in zip(model.estimators_, model.predict_proba(x), strict=True):
        positive = np.flatnonzero(estimator.classes_ == 1)
        columns.append(np.zeros(len(x)) if positive.size == 0 else matrix[:, positive[0]])
    return np.column_stack(columns)


def tune_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.arange(0.30, 0.76, 0.05)
    scores = [precision_recall_fscore_support(y_true, probability >= value, average="binary", zero_division=0)[2] for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate(y_true: np.ndarray, probability: np.ndarray, thresholds: list[float]) -> dict:
    report: dict[str, dict] = {}
    for index, target in enumerate(TARGET_NAMES):
        predicted = probability[:, index] >= thresholds[index]
        precision, recall, f1, _ = precision_recall_fscore_support(y_true[:, index], predicted, average="binary", zero_division=0)
        values = {
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "accuracy": float(accuracy_score(y_true[:, index], predicted)),
            "confusion_matrix": confusion_matrix(y_true[:, index], predicted, labels=[0, 1]).tolist(),
        }
        if np.unique(y_true[:, index]).size == 2:
            values["roc_auc"] = float(roc_auc_score(y_true[:, index], probability[:, index]))
        report[target] = values
    return report


def save_confusion_figures(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for target, values in report.items():
        matrix = np.asarray(values["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(4, 3.5))
        image = axis.imshow(matrix, cmap="Blues")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["negative", "positive"], yticklabels=["negative", "positive"], xlabel="Predicted", ylabel="Actual", title=target.replace("_", " ").title())
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(output / f"{target}_confusion_matrix.png", dpi=160)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the multilabel image-quality model")
    parser.add_argument("--features", type=Path, default=Path("data/features.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/image_quality_rf.joblib"))
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.features)
    missing = {"split", *FEATURE_NAMES, *TARGET_NAMES} - set(frame.columns)
    if missing:
        raise SystemExit(f"Feature dataset is missing columns: {sorted(missing)}")
    partitions = {name: frame[frame["split"] == name] for name in ("train", "validation", "test")}
    if any(part.empty for part in partitions.values()):
        raise SystemExit("Train, validation, and test partitions must all be non-empty")
    if set(partitions["train"].source_id) & set(partitions["test"].source_id):
        raise SystemExit("Source leakage detected between train and test partitions")

    x_train = partitions["train"][FEATURE_NAMES].to_numpy()
    y_train = partitions["train"][TARGET_NAMES].to_numpy(dtype=int)
    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
        )
    )
    model.fit(x_train, y_train)

    validation_probability = positive_probabilities(model, partitions["validation"][FEATURE_NAMES].to_numpy())
    validation_y = partitions["validation"][TARGET_NAMES].to_numpy(dtype=int)
    thresholds = [tune_threshold(validation_y[:, index], validation_probability[:, index]) for index in range(len(TARGET_NAMES))]

    test_probability = positive_probabilities(model, partitions["test"][FEATURE_NAMES].to_numpy())
    test_y = partitions["test"][TARGET_NAMES].to_numpy(dtype=int)
    test_report = evaluate(test_y, test_probability, thresholds)
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_counts": {name: len(part) for name, part in partitions.items()},
        "thresholds_tuned_on": "validation",
        "test": test_report,
    }

    clean_train = partitions["train"][partitions["train"][TARGET_NAMES].sum(axis=1) == 0][FEATURE_NAMES].to_numpy()
    anomaly_model = IsolationForest(n_estimators=200, contamination="auto", random_state=args.seed).fit(clean_train)
    anomaly_raw = -anomaly_model.decision_function(clean_train)
    anomaly_center = float(np.percentile(anomaly_raw, 95))
    anomaly_scale = float(max(np.std(anomaly_raw), 1e-3))

    importances = np.mean([estimator.feature_importances_ for estimator in model.estimators_], axis=0)
    feature_importance = dict(sorted(zip(FEATURE_NAMES, map(float, importances), strict=True), key=lambda item: item[1], reverse=True))
    bundle = {
        "model": model, "anomaly_model": anomaly_model, "anomaly_center": anomaly_center, "anomaly_scale": anomaly_scale,
        "feature_names": FEATURE_NAMES, "target_names": TARGET_NAMES,
        "thresholds": {**dict(zip(TARGET_NAMES, thresholds, strict=True)), "potential_visual_defect": 0.5},
        "version": "1.0", "trained_at": metrics["generated_at"],
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model, compress=3)
    (args.evaluation / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.evaluation / "feature_importance.json").write_text(json.dumps(feature_importance, indent=2))
    save_confusion_figures(test_report, args.evaluation / "figures")
    identifier_columns = [
        name for name in ("source_image", "source_id", "degradation", "severity") if name in partitions["test"].columns
    ]
    predictions = partitions["test"][identifier_columns].reset_index(drop=True).copy()
    for index, target in enumerate(TARGET_NAMES):
        predictions[f"{target}_actual"] = test_y[:, index]
        predictions[f"{target}_probability"] = test_probability[:, index]
        predictions[f"{target}_predicted"] = (test_probability[:, index] >= thresholds[index]).astype(int)
    predictions.to_csv(args.evaluation / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
