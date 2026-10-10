"""Attendance business rules: late cutoffs, deduplication, and manual overrides."""

from datetime import datetime
from enum import StrEnum


class AttendanceStatus(StrEnum):
    PRESENT = "Present"
    LATE = "Late"
    ABSENT = "Absent"
    MANUAL = "Manual"


class AttendanceRuleViolation(Exception):
    """Raised when attendance processing violates operational business rules."""

    pass


class DuplicateAttendanceError(AttendanceRuleViolation):
    """Raised when duplicate attendance is attempted for an already checked-in student."""

    pass


class SessionClosedError(AttendanceRuleViolation):
    """Raised when an attendance check-in is submitted to an expired or ended session."""

    pass


def classify_attendance_status(
    check_in_time_str: str,
    late_cutoff_time_str: str = "09:15:00",
) -> AttendanceStatus:
    """Classifies check-in timestamp as Present or Late.

    Args:
        check_in_time_str: Format "HH:MM:SS" (e.g. "09:14:32")
        late_cutoff_time_str: Cutoff threshold (default "09:15:00")

    Returns:
        AttendanceStatus.PRESENT if check_in_time <= cutoff, else AttendanceStatus.LATE
    """
    t_check = datetime.strptime(check_in_time_str, "%H:%M:%S").time()
    t_cutoff = datetime.strptime(late_cutoff_time_str, "%H:%M:%S").time()

    if t_check <= t_cutoff:
        return AttendanceStatus.PRESENT
    return AttendanceStatus.LATE


def validate_check_in_prerequisites(
    session_ended_at: datetime | None,
    already_checked_in: bool,
    student_id: int,
    session_id: int,
) -> None:
    """Enforces active session status and duplicate prevention."""
    if session_ended_at is not None:
        raise SessionClosedError(
            f"Attendance session {session_id} ended at {session_ended_at.isoformat()} and is no longer accepting entries."
        )

    if already_checked_in:
        raise DuplicateAttendanceError(
            f"Duplicate attendance blocked: Student {student_id} is already recorded in session {session_id}."
        )
