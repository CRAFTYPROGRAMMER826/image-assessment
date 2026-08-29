from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.features import extract_features
from app.core.image_validation import ALLOWED_CONTENT_TYPES, MAX_UPLOAD_BYTES, ImageValidationError, decode_image
from app.core.predictor import ModelNotReadyError
from app.core.scoring import aggregate
from app.db.database import get_db
from app.db.models import Analysis
from app.schemas.analysis import AnalysisResponse, AnalysisSummary, HealthResponse

router = APIRouter()


def _explanations(features: dict[str, float], probabilities: dict[str, float]) -> list[str]:
    strongest = sorted(probabilities, key=probabilities.get, reverse=True)[:2]
    messages: list[str] = []
    evidence = {
        "blur": f"sharpness evidence: Laplacian variance {features['laplacian_variance']:.1f}, edge density {features['edge_density']:.3f}",
        "underexposure": f"exposure evidence: mean luminance {features['mean_luminance']:.1f}, dark pixels {features['dark_pixel_ratio']:.1%}",
        "overexposure": f"exposure evidence: white clipping {features['white_clip_ratio']:.1%}, bright pixels {features['bright_pixel_ratio']:.1%}",
        "noise": f"noise evidence: estimated sigma {features['noise_sigma_estimate']:.1f}, local variance {features['local_variance_mean']:.1f}",
        "severe_degradation": f"information evidence: entropy {features['entropy']:.2f}, dynamic range {features['dynamic_range']:.1f}",
        "potential_visual_defect": "anomaly evidence: feature pattern differs from clean training images; this is not a diagnosis of a physical defect",
    }
    for name in strongest:
        messages.append(f"{name.replace('_', ' ').title()} confidence {probabilities[name]:.0%}; {evidence[name]}.")
    return messages


def _full_response(row: Analysis) -> AnalysisResponse:
    return AnalysisResponse(
        id=row.id,
        filename=row.filename,
        quality_score=row.quality_score,
        quality_label=row.quality_label,
        created_at=row.created_at,
        issues=json.loads(row.issues_json),
        statistics=json.loads(row.statistics_json),
        explanations=_stored_explanations(row),
    )


def _stored_explanations(row: Analysis) -> list[str]:
    features = json.loads(row.statistics_json)
    probabilities = {item["type"]: item["confidence"] for item in json.loads(row.issues_json)}
    return _explanations(features, probabilities)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor = request.app.state.predictor
    return HealthResponse(
        status="ok" if predictor.loaded else "degraded",
        model_loaded=predictor.loaded,
        model_version=predictor.version,
    )


@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only JPEG and PNG images are supported")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image exceeds the 10 MiB upload limit")
    try:
        image = decode_image(payload)
        features = extract_features(image)
        probabilities, thresholds = request.app.state.predictor.predict(features)
    except ImageValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    quality_score, quality_label, issues = aggregate(probabilities, thresholds)
    row = Analysis(
        filename=Path(file.filename or "upload").name[:255],
        quality_score=quality_score,
        quality_label=quality_label,
        blur_confidence=probabilities["blur"],
        under_confidence=probabilities["underexposure"],
        over_confidence=probabilities["overexposure"],
        noise_confidence=probabilities["noise"],
        degradation_confidence=probabilities["severe_degradation"],
        defect_confidence=probabilities.get("potential_visual_defect"),
        statistics_json=json.dumps(features),
        issues_json=json.dumps(issues),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _full_response(row)


@router.get("/api/history", response_model=list[AnalysisSummary])
def history(db: Session = Depends(get_db)) -> list[Analysis]:
    return list(db.scalars(select(Analysis).order_by(desc(Analysis.created_at)).limit(100)))


@router.get("/api/history/{analysis_id}", response_model=AnalysisResponse)
def history_item(analysis_id: int, db: Session = Depends(get_db)) -> AnalysisResponse:
    row = db.get(Analysis, analysis_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return _full_response(row)
