"""Pydantic schemas for the FRAS FastAPI REST API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# -----------------------------------------------------------------------------
# Auth Schemas
# -----------------------------------------------------------------------------


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class LoginRequest(BaseModel):
    username: str
    password: str


# -----------------------------------------------------------------------------
# Student Schemas
# -----------------------------------------------------------------------------


class StudentCreate(BaseModel):
    external_id: str = Field(..., description="Unique person identifier, e.g. person_001")
    name: str = Field(..., description="Full name of student")
    department: str = Field(..., description="Academic department")


class StudentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: str
    name: str
    department: str
    consent_at: datetime
    created_at: datetime


# -----------------------------------------------------------------------------
# Session Schemas
# -----------------------------------------------------------------------------


class SessionCreate(BaseModel):
    session_name: str
    camera_id: str = "ENTRANCE_CAM_01"


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_name: str
    started_at: datetime
    ended_at: datetime | None = None
    camera_id: str
    created_by: str


# -----------------------------------------------------------------------------
# Attendance Schemas
# -----------------------------------------------------------------------------


class ManualAttendanceRequest(BaseModel):
    student_id: int
    session_id: int
    status: str = "Present"
    reason: str


class AttendanceRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: int
    session_id: int
    date: str
    check_in_time: str | None = None
    check_out_time: str | None = None
    status: str
    confidence: float | None = None
    camera_id: str | None = None
    is_manual_fallback: bool
    manual_reason: str | None = None
    student_name: str | None = None
    external_id: str | None = None
    department: str | None = None


class CheckInResponse(BaseModel):
    status: str  # "checked_in", "already_marked", "unknown_rejected", "quality_rejected", "no_face"
    student_id: int | None = None
    student_name: str | None = None
    external_id: str | None = None
    attendance_status: str | None = None
    confidence: float = 0.0
    message: str = ""
    bounding_box: tuple[int, int, int, int] | None = None


# -----------------------------------------------------------------------------
# Health & Analytics Schemas
# -----------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    backend: str
    calibrated_threshold: float
    total_enrolled_students: int
    active_sessions: int
    timestamp: str
