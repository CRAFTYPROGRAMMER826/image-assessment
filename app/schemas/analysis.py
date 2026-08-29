from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Issue(BaseModel):
    type: str
    detected: bool
    confidence: float = Field(ge=0, le=1)
    severity: str


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    quality_score: float = Field(ge=0, le=100)
    quality_label: str
    created_at: datetime


class AnalysisResponse(AnalysisSummary):
    issues: list[Issue]
    statistics: dict[str, float]
    quality_summary: list[str]
    explanations: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    history_persistent: bool
