"""Student management and biometric enrollment router."""

from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from fras.api.dependencies import require_role
from fras.api.schemas import StudentResponse
from fras.core.detection import FaceDetector
from fras.core.quality import QualityGate
from fras.db import repository
from fras.db.models import User
from fras.db.session import get_db
from fras.recognition.avatar_cnn import AvatarCNNBackend
from fras.security.audit import AuditService

router = APIRouter(prefix="/api/students", tags=["Students"])
audit_service = AuditService()
quality_gate = QualityGate()
detector = FaceDetector(backend_name="avatar_cnn")
backend = AvatarCNNBackend()


@router.get("", response_model=list[StudentResponse])
def list_students(
    db: Annotated[Session, Depends(get_db)],
    department: str | None = None,
) -> list[StudentResponse]:
    """Retrieves list of enrolled students, optionally filtered by department."""
    students = repository.list_students(db, department=department)
    return [StudentResponse.model_validate(s) for s in students]


@router.get("/{external_id}", response_model=StudentResponse)
def get_student(
    external_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> StudentResponse:
    """Retrieves a single enrolled student profile by external identity ID."""
    student = repository.get_student_by_external_id(db, external_id)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID '{external_id}' not found.",
        )
    return StudentResponse.model_validate(student)


@router.post("", response_model=StudentResponse, status_code=status.HTTP_201_CREATED)
async def enroll_student(
    external_id: Annotated[str, Form(...)],
    name: Annotated[str, Form(...)],
    department: Annotated[str, Form(...)],
    photo: Annotated[UploadFile, File(...)],
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("admin"))],
) -> StudentResponse:
    """Enrolls a new student by validating face quality, extracting embeddings, and encrypting."""
    existing = repository.get_student_by_external_id(db, external_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Student with external ID '{external_id}' is already enrolled.",
        )

    contents = await photo.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        )

    # 1. Detection
    detections = detector.detect(img)
    if not detections:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in enrollment image.",
        )

    # 2. Quality gate
    q_report = quality_gate.check(img)
    if not q_report.passed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Biometric enrollment rejected: {q_report.diagnostic_message}",
        )

    # 3. Extract and normalize embedding
    preprocessed = backend.preprocess(img)
    embedding = backend.embed(preprocessed)[0]

    # 4. Save with Fernet encryption
    student = repository.create_student(
        db=db,
        external_id=external_id,
        name=name,
        department=department,
        embedding=embedding,
    )

    audit_service.log_event(
        actor=current_user.username,
        action="ENROLL_STUDENT",
        resource_type="Student",
        resource_id=external_id,
        details=f"Enrolled student {name} ({department}) with encrypted biometric template.",
        db=db,
    )

    return StudentResponse.model_validate(student)


@router.delete("/{external_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student(
    external_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("admin"))],
) -> None:
    """GDPR Article 17 Right-to-Erasure: permanently deletes student and biometric template."""
    success = repository.delete_student(db, external_id, actor=current_user.username)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID '{external_id}' not found.",
        )
