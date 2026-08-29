# AI-powered image quality assessment

A local, deployable full-stack application that combines deterministic computer-vision features with learned classical ML decisions. It detects blur, under/overexposure, noise, severe degradation, and a cautiously named potential visual anomaly. No external vision or AI APIs are used.

The repository now contains a real trained artifact at `models/image_quality_rf.joblib`. It was trained from engineered features over source-separated synthetic variants of COCO val2017 images; no evaluation values were fabricated.

## Architecture and methodology

One FastAPI process serves the REST API and static frontend. OpenCV extracts 19 interpretable sharpness, exposure, contrast, noise, texture, and colour features. A scikit-learn multi-output Random Forest produces five independent issue probabilities. An Isolation Forest fitted only to clean training rows produces the **potential visual defect/anomaly** signal; it must not be interpreted as proof that an object is physically defective.

The final 0–100 score is not presented as a learned prediction. It is a transparent probability-weighted penalty using weights in `app/core/scoring.py`. Random-Forest probabilities are converted to detections using per-target thresholds tuned on the validation split.

## Quick start

Requirements: Python 3.12 and `uv`.

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Open <http://localhost:8000>. API documentation is at <http://localhost:8000/docs>.

The committed model loads once during application startup. Upload an image in the browser, or use the `curl` example below. SQLite history is created automatically at `data/app.db`.

## Training and evaluation

The current source pool is expected at `val2017/` in the repository root. It contains the 5,000 COCO val2017 JPEGs; COCO annotations are not used. The source images are intentionally ignored by Git. Record the dataset version, license, and download source in the final submission.

```bash
# First run, or whenever dependencies change
uv sync

# Build/rebuild features only when the source data or degradation code changes
uv run python -m training.build_dataset

# Train/retrain only when data, features, or model settings change
uv run python -m training.train_model

# Start the application (does not retrain)
uv run uvicorn app.main:app --reload
```

`build_dataset` conservatively rejects only unreadable, tiny, almost completely clipped, or extremely unsharp sources. It then saves the fixed source assignments in `data/splits/{train,val,test}.txt`, generates transformations in memory, and writes features directly to CSV. Use `--reset-splits` only when intentionally rebuilding the split. Consequently, no degraded version of a training photo can appear in validation or test. Each accepted source produces 19 rows: one clean row, three intensity levels for each of five individual degradation types, and three multilabel combination cases. Only one inspection set is saved under `samples/generated/`.

Current reproducible run (seed 42): 4,998 accepted sources, split into 3,498 train / 749 validation / 751 test originals. These produced 66,462 / 14,231 / 14,269 feature rows respectively. Two sources were rejected for extreme black clipping; exact details are in `data/dataset_report.json` and `data/rejected_sources.csv`.

Artifacts:

- `data/features.csv`: ordered features and targets
- `data/dataset_report.json`: accepted/rejected source, row, and positive-label counts
- `models/image_quality_rf.joblib`: model, feature order, target order, thresholds, anomaly detector, and version metadata
- `evaluation/metrics.json`: actual untouched-test precision, recall, F1, accuracy, ROC-AUC, and confusion matrices
- `evaluation/feature_importance.json`: global mean Random-Forest feature importance
- `evaluation/figures/`: one rendered confusion matrix per target
- `evaluation/test_predictions.csv`: unseen-test probabilities for finding errors and uncertain cases

Do not publish evaluation claims until you have inspected actual metrics, representative errors, and uncertain predictions. Synthetic labels teach recognition of the generator's transformations, so performance on real camera failures and domain-specific defects is a key limitation. Strong assessment evidence should add a small manually reviewed external test set.

## REST API

```bash
curl http://localhost:8000/health
curl -F "file=@samples/blurry/example.jpg" http://localhost:8000/api/analyze
curl http://localhost:8000/api/history
curl http://localhost:8000/api/history/1
```

Health returns `status`, `model_loaded`, and `model_version`. Analysis accepts one JPEG/PNG up to 10 MiB and returns the persisted ID, score, label, issue confidences/severities, statistics, evidence statements, and timestamp. Expected errors are 400 (unreadable), 413 (too large), 415 (unsupported type), and 503 (model not trained). SQLite is created automatically at `data/app.db`; images are not stored.

### How inference is stitched together

1. FastAPI startup creates the SQLite table and loads the joblib bundle once.
2. `POST /api/analyze` validates the MIME type/size and decodes bytes with OpenCV.
3. `extract_features()` calculates the exact same 19 ordered numbers used during training.
4. The saved Random Forest produces five independent probabilities. The saved Isolation Forest adds the cautious potential-anomaly probability.
5. Saved validation-tuned thresholds convert probabilities into detections; the transparent scoring layer derives severity, score, and label.
6. The complete result is committed to SQLite and returned as JSON.
7. The frontend renders that JSON, while `/api/history` and `/api/history/{id}` retrieve persisted results.

## Docker

Train first so the saved model is included in the image, then:

```bash
docker build -t image-quality-assessment .
docker run --rm -p 8000:8000 -v image-quality-data:/app/data image-quality-assessment
```

Open <http://localhost:8000>. The named volume persists SQLite history. Configuration variables are `MODEL_PATH` and `DATABASE_PATH`.

## Explainability and limitations

The response exposes the raw statistics supplied to the model and explains the two strongest probabilities with relevant measurements. `evaluation/feature_importance.json` provides global importance. These explain associations rather than causality.

- The defect output is image-feature anomaly detection, not arbitrary product-defect diagnosis or localization.
- Synthetic degradation may not reproduce motion blur, sensor-pattern noise, complex HDR failures, or corrupt containers. Unreadable files are rejected before inference.
- Global measurements can confuse intentional bokeh, low-key/high-key photography, and texture with degradation.
- Random-Forest probability is not automatically calibrated; confidence calibration is future work.
- The score and severity bands are application policy, not scientific ground truth.

## Tests

```bash
uv run pytest
```

Tests cover feature shape, source split stability, scoring, upload validation, health, and history. After training, also upload representative unseen images through the browser and Docker container.

## Project layout

```text
app/                 FastAPI, inference, CV features, scoring, SQLite
training/            degradation, tabular feature, training/evaluation pipeline
frontend/            one-page plain HTML/CSS/JS interface
models/              saved inference artifact (generated)
data/                source/generated/features/SQLite (generated data ignored)
evaluation/          actual model reports (generated)
samples/             submission demonstration images
tests/               automated core/API checks
```
