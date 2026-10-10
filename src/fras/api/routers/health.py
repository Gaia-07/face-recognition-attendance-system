"""System health, readiness, and liveness probe router."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from fras.api.schemas import HealthResponse
from fras.db import repository
from fras.db.session import get_db

router = APIRouter(tags=["Health"])

THRESHOLD_FILE = Path(__file__).resolve().parent.parent.parent.parent / "models" / "threshold.json"


@router.get("/healthz", response_model=HealthResponse)
def health_check(
    db: Annotated[Session, Depends(get_db)],
) -> HealthResponse:
    """Readiness and liveness probe checking database connectivity and model configuration."""
    students = repository.list_students(db)
    active_session = repository.get_active_session(db)

    threshold = 0.653
    if THRESHOLD_FILE.exists():
        try:
            with open(THRESHOLD_FILE, encoding="utf-8") as f:
                data = json.load(f)
                threshold = float(data.get("calibrated_threshold", 0.653))
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        backend="avatar_cnn",
        calibrated_threshold=threshold,
        total_enrolled_students=len(students),
        active_sessions=1 if active_session else 0,
        timestamp=datetime.now(UTC).isoformat(),
    )
