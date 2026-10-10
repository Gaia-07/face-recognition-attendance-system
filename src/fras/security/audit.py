"""Audit trail service recording immutable security, privacy, and administrative actions."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from fras.db.models import AuditLog

logger = logging.getLogger("fras.security.audit")

DEFAULT_AUDIT_LOG_FILE = (
    Path(__file__).resolve().parent.parent.parent.parent / "logs" / "audit.jsonl"
)


class AuditService:
    """Orchestrates security and privacy audit logging across database and structured file storage."""

    def __init__(self, log_file: Path | str | None = None) -> None:
        self.log_file = Path(log_file) if log_file else DEFAULT_AUDIT_LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        actor: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: str,
        db: Session | None = None,
    ) -> AuditLog:
        """Records an immutable audit event to disk and database."""
        now = datetime.now(UTC)
        record = AuditLog(
            timestamp=now,
            actor=actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
        )

        # 1. Write to database if active session provided
        if db is not None:
            db.add(record)
            db.commit()
            db.refresh(record)

        # 2. Write structured JSON line to immutable audit log file
        log_payload = {
            "timestamp": now.isoformat(),
            "actor": actor,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_payload) + "\n")
        except Exception as e:
            logger.error(f"Failed writing audit record to {self.log_file}: {e}")

        logger.info(f"AUDIT [{action}] by {actor} on {resource_type}:{resource_id}")
        return record

    def log_login_attempt(
        self,
        username: str,
        success: bool,
        ip_address: str = "unknown",
        db: Session | None = None,
    ) -> AuditLog:
        """Audits user authentication attempts."""
        status_str = "SUCCESS" if success else "FAILURE"
        return self.log_event(
            actor=username,
            action=f"LOGIN_{status_str}",
            resource_type="User",
            resource_id=username,
            details=f"Authentication attempt from IP {ip_address}: {status_str}",
            db=db,
        )

    def log_erasure(
        self,
        external_id: str,
        actor: str,
        db: Session,
    ) -> AuditLog:
        """Audits GDPR Article 17 Right-to-Erasure requests."""
        return self.log_event(
            actor=actor,
            action="GDPR_ERASURE",
            resource_type="Student",
            resource_id=external_id,
            details="Permanent erasure of biometric template and profile upon verified request.",
            db=db,
        )

    def log_manual_override(
        self,
        student_id: int,
        session_id: int,
        actor: str,
        reason: str,
        db: Session,
    ) -> AuditLog:
        """Audits manual attendance overrides and exceptions."""
        return self.log_event(
            actor=actor,
            action="MANUAL_OVERRIDE",
            resource_type="AttendanceRecord",
            resource_id=f"student_{student_id}_session_{session_id}",
            details=f"Manual attendance override: {reason}",
            db=db,
        )
