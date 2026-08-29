# AI-Powered Image Quality Assessment

A deployable full-stack system that determines whether an uploaded JPEG or PNG is **acceptable**, **degraded**, or **potentially defective**. It detects blur, underexposure, overexposure, noise, severe degradation, and unusual visual-quality patterns without external vision APIs.

The core is a hybrid pipeline: OpenCV converts pixels into 19 interpretable measurements, a trained multilabel Random Forest converts those measurements into issue probabilities, and an Isolation Forest adds a cautious anomaly signal. The browser receives a score, detected issues, confidence and severity, plain-English reasoning, and the underlying statistics.

## Problem and Smart City relevance

Image analytics are only as reliable as their input. A blurred, clipped, noisy, or corrupted camera frame can silently damage downstream tasks such as crowd estimation, incident review, traffic observation, infrastructure inspection, and safety monitoring. This project acts as an input-quality gate: it can reject or flag unreliable imagery before another vision system or human operator trusts it.

This maps naturally to the [IIIT Hyderabad Smart City Living Lab](https://smartcitylivinglab.iiit.ac.in/), a real-world smart-campus test bed for IoT research, pilot runs, and proofs of concept. IIITH describes live sensing infrastructure across domains including smart spaces and public safety, while its smart-campus work includes video analytics and crowd monitoring. In that setting, this tool could support:

- camera-health and frame-quality checks before crowd or safety analytics;
- explainable quality telemetry for control-room dashboards;
- lightweight edge inference where external AI services are undesirable;
- persisted evidence for maintenance and model-monitoring workflows.

The fit is architectural, not an affiliation claim: this is an independent assessment project and is not presented as an official IIITH deployment. The design follows the Living Lab's emphasis on testable, data-driven smart-city components and real-world validation. See the official [Living Lab overview](https://smartcitylivinglab.iiit.ac.in/about_us/overview/) and [smart-campus description](https://smartcityresearch.iiit.ac.in/living_lab/smartcampus/).

## What the application returns

For each image, the API and UI provide:

- a transparent quality score from 0 to 100;
- `ACCEPTABLE`, `DEGRADED`, or `POTENTIALLY_DEFECTIVE`;
- blur, underexposure, overexposure, noise, severe-degradation, and anomaly probabilities;
- thresholded detections with low, medium, or high severity;
- a three-sentence plain-English assessment;
- all 19 image statistics and short explanations of the strongest signals;
- SQLite-backed analysis history in local and Docker deployments.

## System design

```text
Browser upload
    -> FastAPI validation and OpenCV decode
    -> deterministic 19-feature extraction
    -> saved multilabel Random Forest probabilities
    -> saved Isolation Forest anomaly probability
    -> thresholds + transparent score aggregation
    -> SQLite persistence + JSON response
    -> one-page HTML/CSS/JavaScript UI
```

The model is trained offline and loaded once during FastAPI startup. Inference never retrains the model and never calls an external AI service. The joblib bundle stores the estimator, anomaly model, exact feature order, exact target order, validation-tuned thresholds, version, and training timestamp.

## Image statistics and why they are used

No single statistic reliably defines image quality. For example, a naturally dark photograph is not necessarily underexposed, and high-frequency content can be either useful detail or random noise. The learned model considers correlated evidence across these groups.

| Evidence group | Features | Technical purpose |
|---|---|---|
| Sharpness | Laplacian variance, Tenengrad score, edge density, high-frequency energy | Blur suppresses coherent edges and gradients. Laplacian variance measures second-derivative activity; Tenengrad uses squared Sobel-gradient magnitude; Canny edge density measures the proportion of strong boundaries. |
| Exposure | Mean/std luminance, black/white clipping, dark/bright pixel ratios | Mean brightness provides context, while clipping is stronger evidence of lost information. Black and white clipping are measured near 0 and 255 rather than treating every dark or bright scene as defective. |
| Contrast | RMS contrast, percentile dynamic range | RMS contrast is luminance standard deviation. Dynamic range uses the 1st-to-99th percentile span, reducing sensitivity to a few extreme pixels. |
| Noise and local variation | Residual noise sigma, mean local variance | The image is Gaussian-smoothed and subtracted from the original. A median absolute residual scaled by `1 / 0.6745` estimates noise magnitude; 7x7 local variance supplies complementary texture evidence. |
| Information and structure | Entropy, gradient mean/std | Shannon entropy measures tonal information. Gradient distribution describes the amount and variability of spatial structure. |
| Colour | HSV saturation mean/std | These capture overall colour intensity and variation without assuming vivid colour is automatically a defect. |

Feature extraction is implemented in `app/core/features.py`. The returned dictionary is explicitly ordered, and the predictor rejects a model artifact whose saved feature order differs from the application order.

## Why classical ML instead of a CNN

The assessment requires a genuine learned decision component but does not require deep learning. A Random Forest was selected because the problem is naturally tabular after feature extraction and because it:

- learns nonlinear interactions between exposure, texture, edges, noise, and colour;
- supports small-to-medium datasets and CPU-only training/inference;
- handles mixed feature scales without neural-network preprocessing;
- provides probability estimates and global feature importance;
- is faster to reproduce and easier to explain than a large CNN.

Five targets are learned independently through `MultiOutputClassifier` because defects can coexist, such as blur plus noise or overexposure plus blur. The current forest uses 250 trees per target, depth 18, balanced class weights, square-root feature sampling, and a fixed seed.

An Isolation Forest is fitted only on clean training rows. It compares a new feature pattern with nominal training behaviour and emits `potential_visual_defect`. This is deliberately described as an anomaly—not proof of a physical product defect or defect localization.

### Transparent scoring layer

The 0-100 score is not misrepresented as a learned regression output. It is an application-level aggregation:

```text
score = clamp(100 - sum(issue_probability x issue_penalty), 0, 100)
```

Blur, underexposure, and overexposure each carry a maximum 20-point penalty; noise 15; severe degradation 25; and anomaly 20. Labels are `ACCEPTABLE` at 80+, `DEGRADED` at 50-79.9, and `POTENTIALLY_DEFECTIVE` below 50. These weights and severity bands are explainable policy choices in `app/core/scoring.py`, not scientific constants.

## Training methodology

### Source data and leakage control

The current model uses the 5,000 images from COCO `val2017` as a varied nominal source pool; COCO object annotations are not used. A conservative sanity filter rejected two extremely black-clipped sources, leaving 4,998 images.

Sources are split **before** degradation using seed 42. Every derivative of a source remains in the same partition, preventing the model from seeing one photograph during training and a modified version of it during testing.

| Partition | Source images | Generated feature rows |
|---|---:|---:|
| Train | 3,498 | 66,462 |
| Validation | 749 | 14,231 |
| Test | 751 | 14,269 |
| **Total** | **4,998** | **94,962** |

Exact assignments are persisted in `data/splits/`. The builder verified 19 rows per source, zero nulls, and zero source overlap between partitions.

### Synthetic supervision

Each source produces one clean row, three severity levels for five individual degradations, and three mixed-label examples—19 rows in total. Processing occurs in memory; the project does not duplicate roughly 95,000 derivative images on disk.

| Condition | Controlled transformation |
|---|---|
| Blur | Gaussian sigma 1, 2, and 4 |
| Underexposure | Intensity factors 0.7, 0.4, and 0.2 |
| Overexposure | Gains 1.3, 1.7, and 2.2 with progressive clipping |
| Noise | Gaussian sigma 8, 18, and 35 |
| Severe degradation | Increasing pixelation plus JPEG quality 24, 12, and 5 |
| Mixed labels | Blur + noise, underexposure + noise, overexposure + blur |

The training partition fits the forest. Validation probabilities select a separate F1-maximizing threshold for each target over 0.30-0.75. Only after model and threshold selection is the untouched test partition evaluated.

## Evaluation

These are measured results from synthetically degraded derivatives of unseen COCO source images—not fabricated values and not a claim of universal real-world accuracy.

| Target | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Blur | 0.999 | 0.999 | 0.999 | 1.000 |
| Underexposure | 0.988 | 0.999 | 0.994 | 1.000 |
| Overexposure | 0.902 | 0.909 | 0.905 | 0.990 |
| Noise | 0.970 | 0.981 | 0.975 | 0.999 |
| Severe degradation | 0.978 | 0.965 | 0.972 | 0.999 |

Reproducible outputs are stored in `evaluation/metrics.json`, `evaluation/feature_importance.json`, `evaluation/figures/`, and `evaluation/test_predictions.csv`. The next meaningful evaluation step is a manually reviewed external set containing real motion blur, sensor noise, difficult low/high-key scenes, camera compression, and domain-specific imagery.

## Run locally

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repository-url>
cd image-assessment
uv sync
uv run uvicorn app.main:app --reload
```

Open:

- UI: <http://localhost:8000>
- API documentation: <http://localhost:8000/docs>
- health check: <http://localhost:8000/health>

The committed `models/image_quality_rf.joblib` is sufficient for inference; COCO and `data/features.csv` are not required to run the application. Local SQLite history is created automatically at `data/app.db`.

### Example API calls

```bash
curl http://localhost:8000/health
curl -F "file=@path/to/image.jpg" http://localhost:8000/api/analyze
curl http://localhost:8000/api/history
curl http://localhost:8000/api/history/1
```

Uploads accept JPEG/PNG files up to 10 MiB locally. Expected failures are 400 for unreadable images, 413 for excessive size, 415 for unsupported media, and 503 when the model artifact is unavailable.

## Run with Docker

```bash
docker build -t image-quality-assessment .
docker run --rm -p 8000:8000 \
  -v image-quality-data:/app/data \
  image-quality-assessment
```

The named volume persists SQLite history. The container runs as a non-root user, exposes port 8000, includes a health check, and loads the committed model at startup.

## Vercel deployment status and procedure

The repository contains a Vercel-compatible root `index.py`, runtime-only requirements, and `vercel.json` exclusions for training data and evaluation artifacts. Vercel uses temporary `/tmp` storage, so history there is best-effort and can disappear across cold starts or instances; local/Docker remains the persistence-complete deployment.

The latest Vercel production build is currently **blocked**, not live: its 440.82 MB scientific-Python bundle exceeded that project's 225 MB function limit. Sample images were not the cause. OpenCV, NumPy, SciPy, scikit-learn, and the 32 MB model dominate the bundle.

To retry on a Vercel plan/runtime that supports the bundle:

1. Import the Git repository with the FastAPI preset and root directory `./`.
2. Leave Build Command and Output Directory unset.
3. Leave Install Command on Vercel's default; a custom `pip install` disables Python bundle optimization.
4. Do not add environment variables. Vercel automatically sets `VERCEL`, selecting `/tmp/image-quality-app.db` and a conservative 4 MiB upload limit.
5. Deploy and verify `/health` reports `model_loaded: true`.

If the optimized function still exceeds the plan limit, use Docker on a container host with a persistent volume, or deliberately replace the heaviest runtime dependency/model representation. Do not delete evaluation evidence merely to conceal a dependency-size problem.

## Retrain, fine-tune, or increase the dataset

For this classical model, “fine-tuning” means rebuilding features and retraining the forest rather than continuing gradient-based neural-network weights.

1. Place COCO `val2017` at `val2017/`, or supply another directory of diverse, licensed clean images.
2. When changing the source pool, rebuild exact source assignments:

   ```bash
   uv run python -m training.build_dataset \
     --input path/to/clean/images \
     --reset-splits
   ```

3. To reuse the current sources and split after changing feature/degradation code, omit `--reset-splits`:

   ```bash
   uv run python -m training.build_dataset
   ```

4. Train, tune thresholds on validation, and evaluate on test:

   ```bash
   uv run python -m training.train_model
   ```

5. Review `evaluation/test_predictions.csv` for errors/uncertain cases and inspect every confusion matrix. Restart FastAPI after replacing the model artifact.

Useful extension points:

- increase clean-source diversity rather than only producing more variants of the same photos;
- add realistic motion blur, Poisson/sensor noise, defocus, and codec artifacts in `training/generate_degradations.py`;
- change forest capacity in `training/train_model.py`, then compare on the unchanged test set;
- add a new target only by updating degradation labels, training targets, scoring, schemas, and UI together;
- calibrate probabilities on validation data before describing them as calibrated confidence;
- add a manually labeled domain test set without mixing it into model selection.

When source data grows, the builder supports `--workers N` and streams rows to `data/features.csv`. It creates 19 rows per accepted source, so expected dataset size and runtime grow approximately linearly.

## Limitations

- COCO sources are diverse but not guaranteed pristine; the sanity filter only removes extreme cases.
- Synthetic transformations may be easier to recognize than real camera failures, explaining the high test scores.
- Global features can confuse intentional bokeh, low-key/high-key photography, repetitive texture, and grain with defects.
- Random-Forest probabilities are not currently calibrated.
- The anomaly detector is global and non-localizing; it does not identify a physical defect region.
- Score weights and severity bands are product-policy choices, not learned scientific ground truth.
- Vercel's ephemeral filesystem does not satisfy durable-history requirements; Docker/local SQLite does.

## Technology stack

Python 3.12, FastAPI, Uvicorn, OpenCV, NumPy, scikit-learn, joblib, SQLite, SQLAlchemy, Pydantic, plain HTML/CSS/JavaScript, pytest, uv, and Docker. No external vision API, cloud model, CNN, React, or PostgreSQL is required.

## Repository map

```text
app/          API, feature extraction, inference, scoring, schemas, SQLite
training/     source splitting, synthetic data, feature building, training
frontend/     one-page upload/results/history interface
models/       versioned inference artifact
data/splits/  persisted source-level train/validation/test assignments
evaluation/   metrics, feature importance, confusion matrices, predictions
samples/      small visual inspection set
tests/        automated feature, split, scoring, and API checks
```

Run the test suite with:

```bash
uv run pytest
```
