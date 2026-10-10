"""SQLAlchemy 2.0 database models for biometric attendance and identity management."""

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Student(Base):
    """Enrolled student profile with encrypted biometric template and consent record."""

    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    department: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    consent_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    embedding_encrypted: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # Fernet encrypted string
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    attendance_records: Mapped[list["AttendanceRecord"]] = relationship(
        "AttendanceRecord", back_populates="student", cascade="all, delete-orphan"
    )


class Session(Base):
    """Attendance tracking session (e.g., lecture, laboratory, or daily check-in gate)."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_name: Mapped[str] = mapped_column(String(128), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, default="ENTRANCE_CAM_01")
    created_by: Mapped[str] = mapped_column(String(64), nullable=False, default="system")

    attendance_records: Mapped[list["AttendanceRecord"]] = relationship(
        "AttendanceRecord", back_populates="session", cascade="all, delete-orphan"
    )


class AttendanceRecord(Base):
    """Recorded attendance event with strict student-session uniqueness."""

    __tablename__ = "attendance"
    __table_args__ = (UniqueConstraint("student_id", "session_id", name="uq_student_session"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("students.id"), nullable=False, index=True
    )
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sessions.id"), nullable=False, index=True
    )
    date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    check_in_time: Mapped[str | None] = mapped_column(String(8), nullable=True)  # HH:MM:SS
    check_out_time: Mapped[str | None] = mapped_column(String(8), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # "Present", "Late", "Absent", "Manual"
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    camera_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_manual_fallback: Mapped[bool] = mapped_column(nullable=False, default=False)
    manual_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)

    student: Mapped["Student"] = relationship("Student", back_populates="attendance_records")
    session: Mapped["Session"] = relationship("Session", back_populates="attendance_records")


class UnknownEvent(Base):
    """Audit record for unverified faces (captures metadata only, no raw biometric photos)."""

    __tablename__ = "unknown_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("sessions.id"), nullable=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False)


class AuditLog(Base):
    """Immutable security audit trail of administrative modifications and biometric erasures."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(64), nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=False)


class User(Base):
    """System users with hashed credentials and role-based access control."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="faculty"
    )  # admin, faculty, auditor
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
