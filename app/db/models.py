from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    quality_score: Mapped[float] = mapped_column(Float)
    quality_label: Mapped[str] = mapped_column(String(32))
    blur_confidence: Mapped[float] = mapped_column(Float)
    under_confidence: Mapped[float] = mapped_column(Float)
    over_confidence: Mapped[float] = mapped_column(Float)
    noise_confidence: Mapped[float] = mapped_column(Float)
    degradation_confidence: Mapped[float] = mapped_column(Float)
    defect_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    statistics_json: Mapped[str] = mapped_column(Text)
    issues_json: Mapped[str] = mapped_column(Text)
