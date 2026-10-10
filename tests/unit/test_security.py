"""Unit tests for biometric template encryption, authentication, RBAC, and audit trails."""

from datetime import timedelta
from pathlib import Path

import jwt
import numpy as np
import pytest
from cryptography.fernet import InvalidToken
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from fras.db import repository
from fras.db.models import Base
from fras.security.audit import AuditService
from fras.security.auth import (
    UserRole,
    create_access_token,
    decode_access_token,
    has_permission,
    hash_password,
    verify_password,
)
from fras.security.crypto import decrypt_embedding, encrypt_embedding


@pytest.fixture
def memory_db() -> Session:
    """Fixture providing clean in-memory database."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def test_embedding_encryption_and_decryption_roundtrip() -> None:
    original = np.random.randn(128).astype(np.float32)
    original /= np.linalg.norm(original)

    encrypted_token = encrypt_embedding(original)
    assert isinstance(encrypted_token, str)
    assert len(encrypted_token) > 0
    assert encrypted_token != str(original.tolist())

    decrypted = decrypt_embedding(encrypted_token)
    assert decrypted.shape == (128,)
    assert np.allclose(original, decrypted, atol=1e-5)


def test_embedding_tampering_rejection() -> None:
    original = np.random.randn(128).astype(np.float32)
    encrypted_token = encrypt_embedding(original)

    # Tamper with cipher string
    tampered_token = encrypted_token[:-5] + "AAAAA"
    with pytest.raises(InvalidToken):
        decrypt_embedding(tampered_token)


def test_password_hashing_and_verification() -> None:
    plain = "SuperSecurePassword2026!"
    hashed = hash_password(plain)

    assert hashed != plain
    assert verify_password(plain, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_jwt_token_generation_and_validation() -> None:
    payload = {"sub": "faculty_user", "role": "faculty", "dept": "Engineering"}
    token = create_access_token(payload, expires_delta=timedelta(minutes=30))

    decoded = decode_access_token(token)
    assert decoded["sub"] == "faculty_user"
    assert decoded["role"] == "faculty"
    assert decoded["dept"] == "Engineering"
    assert "exp" in decoded


def test_jwt_token_expiration() -> None:
    payload = {"sub": "expired_user", "role": "faculty"}
    # Token expired 10 minutes ago
    expired_token = create_access_token(payload, expires_delta=timedelta(minutes=-10))

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(expired_token)


def test_rbac_permission_hierarchy() -> None:
    # Admin can access admin, faculty, auditor scopes
    assert has_permission(UserRole.ADMIN, "admin") is True
    assert has_permission(UserRole.ADMIN, "faculty") is True
    assert has_permission(UserRole.ADMIN, "auditor") is True

    # Faculty can access faculty scope only
    assert has_permission(UserRole.FACULTY, "faculty") is True
    assert has_permission(UserRole.FACULTY, "admin") is False
    assert has_permission(UserRole.FACULTY, "auditor") is False

    # Auditor can access auditor scope only
    assert has_permission(UserRole.AUDITOR, "auditor") is True
    assert has_permission(UserRole.AUDITOR, "admin") is False
    assert has_permission(UserRole.AUDITOR, "faculty") is False


def test_audit_service_logging(tmp_path: Path, memory_db: Session) -> None:
    log_file = tmp_path / "test_audit.jsonl"
    audit_service = AuditService(log_file=log_file)

    # Log events
    audit_service.log_login_attempt("admin", success=True, ip_address="127.0.0.1", db=memory_db)
    audit_service.log_manual_override(
        student_id=1, session_id=1, actor="prof_smith", reason="Medical note", db=memory_db
    )

    # Check file contents
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert "LOGIN_SUCCESS" in lines[0]
    assert "MANUAL_OVERRIDE" in lines[1]

    # Check database contents
    db_logs = repository.list_audit_logs(memory_db)
    assert len(db_logs) == 2
    assert db_logs[0].action in ("LOGIN_SUCCESS", "MANUAL_OVERRIDE")


def test_gdpr_right_to_erasure_workflow(memory_db: Session) -> None:
    # Enroll student
    dummy_emb = np.random.randn(128).astype(np.float32)
    _ = repository.create_student(
        db=memory_db,
        external_id="gdpr_001",
        name="Student For Erasure",
        department="Bioethics",
        embedding=dummy_emb,
    )
    assert repository.get_student_by_external_id(memory_db, "gdpr_001") is not None

    # Execute GDPR Article 17 erasure
    erased = repository.delete_student(memory_db, "gdpr_001", actor="dpo_admin")
    assert erased is True

    # Student and biometric data must be completely gone
    assert repository.get_student_by_external_id(memory_db, "gdpr_001") is None

    # Audit record must prove erasure was fulfilled
    audit_logs = repository.list_audit_logs(memory_db)
    assert any(log.action == "ERASURE" and log.resource_id == "gdpr_001" for log in audit_logs)
