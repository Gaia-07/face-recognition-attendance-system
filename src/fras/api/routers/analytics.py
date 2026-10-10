"""Attendance analytics and predictive ML risk endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from fras.attendance.analytics import (
    build_attendance_heatmap_matrix,
    compute_daily_attendance_summary,
    compute_department_breakdown,
    compute_student_punctuality_rankings,
    train_and_predict_absence_risk,
)
from fras.db import repository
from fras.db.session import get_db

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/daily")
def get_daily_summary(
    db: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    """Returns chronological daily attendance and punctuality counts."""
    records = repository.list_attendance_records(db=db)
    return compute_daily_attendance_summary(records)


@router.get("/departments")
def get_department_breakdown(
    db: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    """Returns department-wise attendance statistics."""
    records = repository.list_attendance_records(db=db)
    return compute_department_breakdown(records)


@router.get("/rankings")
def get_student_rankings(
    db: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    """Returns student-level punctuality and attendance ranking statistics."""
    records = repository.list_attendance_records(db=db)
    return compute_student_punctuality_rankings(records)


@router.get("/heatmap")
def get_heatmap_matrix(
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """Returns student x date attendance status matrix for visualization."""
    records = repository.list_attendance_records(db=db)
    return build_attendance_heatmap_matrix(records)


@router.get("/absence-risks")
def get_absence_risk_forecasts(
    db: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    """Runs sequential ML model forecasting student absence risks for the next session."""
    records = repository.list_attendance_records(db=db)
    return train_and_predict_absence_risk(records)
