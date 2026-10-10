"""Attendance service orchestrating detection, quality verification, matching, and deduplication."""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from fras.attendance.rules import (
    DuplicateAttendanceError,
    classify_attendance_status,
)
from fras.core.detection import FaceDetector
from fras.core.quality import QualityGate
from fras.db import repository
from fras.db.models import AttendanceRecord, Student
from fras.recognition.avatar_cnn import AvatarCNNBackend
from fras.recognition.base import EmbeddingBackend
from fras.recognition.matcher import BiometricMatcher
from fras.recognition.voting import TemporalVotingEngine

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_FILE = (
    Path(__file__).resolve().parent.parent.parent.parent / "models" / "threshold.json"
)


@dataclass(frozen=True)
class AttendanceEventResult:
    """Outcome of an attendance verification attempt on a video frame or image."""

    status: str  # "checked_in", "already_marked", "unknown_rejected", "quality_rejected", "no_face", "pending_votes"
    student_id: int | None = None
    student_name: str | None = None
    external_id: str | None = None
    attendance_status: str | None = None  # "Present", "Late"
    confidence: float = 0.0
    message: str = ""
    bounding_box: tuple[int, int, int, int] | None = None


class AttendanceService:
    """Production attendance orchestration service."""

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        detector: FaceDetector | None = None,
        quality_gate: QualityGate | None = None,
        threshold: float | None = None,
        temporal_window: int = 1,
        temporal_consensus: int = 1,
        late_cutoff: str = "09:15:00",
    ) -> None:
        self.backend = backend or AvatarCNNBackend()
        self.detector = detector or FaceDetector(backend_name="avatar_cnn")
        self.quality_gate = quality_gate or QualityGate()
        self.threshold = threshold if threshold is not None else self._load_calibrated_threshold()
        self.late_cutoff = late_cutoff

        self.matcher = BiometricMatcher(threshold=self.threshold)
        self.gallery_students: dict[str, Student] = {}
        self.voter = (
            TemporalVotingEngine(k=temporal_consensus, n=temporal_window)
            if temporal_window > 1
            else None
        )

    @staticmethod
    def _load_calibrated_threshold() -> float:
        """Loads calibrated operating threshold from models/threshold.json."""
        if DEFAULT_THRESHOLD_FILE.exists():
            try:
                with open(DEFAULT_THRESHOLD_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                    return float(data.get("calibrated_threshold", 0.65))
            except Exception as e:
                logger.warning(f"Could not load threshold from {DEFAULT_THRESHOLD_FILE}: {e}")
        return 0.65

    def refresh_gallery(self, db: Session) -> None:
        """Reloads enrolled student templates from the database into the vector matcher."""
        students = repository.list_students(db)
        self.matcher.clear()
        self.gallery_students.clear()
        for s in students:
            emb = repository.get_student_embedding(s)
            self.matcher.enroll(s.external_id, emb)
            self.gallery_students[s.external_id] = s
        logger.info(f"Loaded {len(students)} student templates into attendance matcher.")

    def process_frame(
        self,
        image_bgr: np.ndarray,
        session_id: int,
        db: Session,
        camera_id: str = "ENTRANCE_CAM_01",
        timestamp: datetime | None = None,
    ) -> AttendanceEventResult:
        """Processes an incoming video frame to detect, verify, and mark attendance."""
        now = timestamp or datetime.now(UTC)
        current_time_str = now.strftime("%H:%M:%S")
        current_date_str = now.strftime("%Y-%m-%d")

        # 1. Check session existence and state
        session_obj = repository.get_session_by_id(db, session_id)
        if not session_obj:
            return AttendanceEventResult(
                status="error",
                message=f"Session ID {session_id} does not exist.",
            )
        if session_obj.ended_at is not None:
            return AttendanceEventResult(
                status="session_closed",
                message=f"Session {session_id} is closed and no longer accepting check-ins.",
            )

        # 2. Face detection
        detections = self.detector.detect(image_bgr)
        if not detections:
            return AttendanceEventResult(status="no_face", message="No face detected in frame.")

        # Use primary detection (highest confidence)
        det = detections[0]
        x, y, bw, bh = det.bbox
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        h, w = image_bgr.shape[:2]
        crop = image_bgr[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if crop.size == 0:
            return AttendanceEventResult(status="no_face", message="Invalid face bounding box.")

        # 3. Quality gating
        q_res = self.quality_gate.check(crop)
        if not q_res.passed:
            return AttendanceEventResult(
                status="quality_rejected",
                message=f"Quality check rejected: {q_res.failure_reason}",
                bounding_box=det.bbox,
            )

        # 4. Feature extraction
        preprocessed = self.backend.preprocess(crop)
        probe_emb = self.backend.embed(preprocessed)

        # 5. Matching against enrolled biometric gallery
        if not self.gallery_students:
            self.refresh_gallery(db)

        if not self.gallery_students:
            return AttendanceEventResult(
                status="error",
                message="No enrolled students found in biometric gallery.",
                bounding_box=det.bbox,
            )

        match_res = self.matcher.match(probe_emb)
        score = match_res.similarity

        if not match_res.is_match or match_res.matched_id is None:
            # Audit log rejected unknown face
            repository.record_unknown_event(
                db=db,
                similarity_score=score,
                camera_id=camera_id,
                reason_code=match_res.status,
                session_id=session_id,
            )
            return AttendanceEventResult(
                status="unknown_rejected",
                confidence=score,
                message=f"Unknown face rejected ({match_res.status}: similarity {score:.3f} < threshold {self.threshold:.3f}).",
                bounding_box=det.bbox,
            )

        matched_external_id = match_res.matched_id
        student = self.gallery_students.get(matched_external_id)
        if not student:
            return AttendanceEventResult(
                status="error",
                message=f"Student ID {matched_external_id} not found in cache.",
                bounding_box=det.bbox,
            )

        # 6. Temporal consensus voting (if enabled)
        if self.voter is not None:
            decision = self.voter.update(matched_external_id)
            if not decision.is_confirmed:
                return AttendanceEventResult(
                    status="pending_votes",
                    student_id=student.id,
                    student_name=student.name,
                    external_id=student.external_id,
                    confidence=score,
                    message=f"Temporal frame acquired ({decision.votes}/{self.voter.k}) for {student.name}. Awaiting consensus.",
                    bounding_box=det.bbox,
                )

        # 7. Check if already marked (deduplication)
        existing_rec = repository.get_attendance_record(db, student.id, session_id)
        if existing_rec is not None:
            return AttendanceEventResult(
                status="already_marked",
                student_id=student.id,
                student_name=student.name,
                external_id=student.external_id,
                attendance_status=existing_rec.status,
                confidence=score,
                message=f"Attendance already recorded for {student.name} as {existing_rec.status} at {existing_rec.check_in_time}.",
                bounding_box=det.bbox,
            )

        # 8. Classify attendance status according to operational cutoff rule
        att_status = classify_attendance_status(
            check_in_time_str=current_time_str,
            late_cutoff_time_str=self.late_cutoff,
        )

        # 9. Commit attendance record to DB
        try:
            repository.record_attendance(
                db=db,
                student_id=student.id,
                session_id=session_id,
                date=current_date_str,
                check_in_time=current_time_str,
                status=att_status.value,
                confidence=score,
                camera_id=camera_id,
                is_manual_fallback=False,
            )
        except DuplicateAttendanceError:
            return AttendanceEventResult(
                status="already_marked",
                student_id=student.id,
                student_name=student.name,
                external_id=student.external_id,
                attendance_status=att_status.value,
                confidence=score,
                message=f"Attendance already marked for {student.name}.",
                bounding_box=det.bbox,
            )

        return AttendanceEventResult(
            status="checked_in",
            student_id=student.id,
            student_name=student.name,
            external_id=student.external_id,
            attendance_status=att_status.value,
            confidence=score,
            message=f"Attendance successfully recorded: {student.name} marked {att_status.value}.",
            bounding_box=det.bbox,
        )

    def record_manual_fallback(
        self,
        db: Session,
        student_id: int,
        session_id: int,
        status: str,
        reason: str,
        actor: str = "faculty",
        timestamp: datetime | None = None,
    ) -> AttendanceRecord:
        """Manual attendance entry for medical exceptions or camera fallback."""
        now = timestamp or datetime.now(UTC)
        rec = repository.record_attendance(
            db=db,
            student_id=student_id,
            session_id=session_id,
            date=now.strftime("%Y-%m-%d"),
            check_in_time=now.strftime("%H:%M:%S"),
            status=status,
            confidence=1.0,
            camera_id="MANUAL_ENTRY",
            is_manual_fallback=True,
            manual_reason=reason,
        )
        repository.log_audit_event(
            db=db,
            actor=actor,
            action="MANUAL_ATTENDANCE",
            resource_type="AttendanceRecord",
            resource_id=f"student_{student_id}_session_{session_id}",
            details=f"Manual attendance marked ({status}): {reason}",
        )
        return rec
