"""Attendance operations, live check-ins, manual overrides, and report exports."""

import tempfile
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from fras.api.dependencies import require_role
from fras.api.schemas import (
    AttendanceRecordResponse,
    CheckInResponse,
    ManualAttendanceRequest,
)
from fras.attendance.reports import (
    export_attendance_excel,
    export_attendance_pdf,
    generate_attendance_csv,
)
from fras.attendance.service import AttendanceService
from fras.db import repository
from fras.db.models import User
from fras.db.session import get_db

router = APIRouter(prefix="/api/attendance", tags=["Attendance"])
attendance_service = AttendanceService()


@router.post("/check-in", response_model=CheckInResponse)
async def check_in(
    session_id: Annotated[int, Form(...)],
    photo: Annotated[UploadFile, File(...)],
    db: Annotated[Session, Depends(get_db)],
    camera_id: Annotated[str, Form(...)] = "ENTRANCE_CAM_01",
) -> CheckInResponse:
    """Processes a video frame to recognize student identity and record attendance."""
    contents = await photo.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image format.",
        )

    result = attendance_service.process_frame(
        image_bgr=img,
        session_id=session_id,
        db=db,
        camera_id=camera_id,
    )

    return CheckInResponse(
        status=result.status,
        student_id=result.student_id,
        student_name=result.student_name,
        external_id=result.external_id,
        attendance_status=result.attendance_status,
        confidence=result.confidence,
        message=result.message,
        bounding_box=result.bounding_box,
    )


@router.post("/manual", response_model=AttendanceRecordResponse)
def manual_override(
    body: ManualAttendanceRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("faculty"))],
) -> AttendanceRecordResponse:
    """Manual attendance fallback entry for medical, camera, or exception handling."""
    rec = attendance_service.record_manual_fallback(
        db=db,
        student_id=body.student_id,
        session_id=body.session_id,
        status=body.status,
        reason=body.reason,
        actor=current_user.username,
    )

    return AttendanceRecordResponse(
        id=rec.id,
        student_id=rec.student_id,
        session_id=rec.session_id,
        date=rec.date,
        check_in_time=rec.check_in_time,
        check_out_time=rec.check_out_time,
        status=rec.status,
        confidence=rec.confidence,
        camera_id=rec.camera_id,
        is_manual_fallback=rec.is_manual_fallback,
        manual_reason=rec.manual_reason,
        student_name=rec.student.name if rec.student else None,
        external_id=rec.student.external_id if rec.student else None,
        department=rec.student.department if rec.student else None,
    )


@router.get("/records", response_model=list[AttendanceRecordResponse])
def list_records(
    db: Annotated[Session, Depends(get_db)],
    date: str | None = None,
    session_id: int | None = None,
    department: str | None = None,
    student_id: int | None = None,
) -> list[AttendanceRecordResponse]:
    """Retrieves filtered attendance records."""
    records = repository.list_attendance_records(
        db=db,
        date=date,
        session_id=session_id,
        department=department,
        student_id=student_id,
    )
    results = []
    for r in records:
        results.append(
            AttendanceRecordResponse(
                id=r.id,
                student_id=r.student_id,
                session_id=r.session_id,
                date=r.date,
                check_in_time=r.check_in_time,
                check_out_time=r.check_out_time,
                status=r.status,
                confidence=r.confidence,
                camera_id=r.camera_id,
                is_manual_fallback=r.is_manual_fallback,
                manual_reason=r.manual_reason,
                student_name=r.student.name if r.student else None,
                external_id=r.student.external_id if r.student else None,
                department=r.student.department if r.student else None,
            )
        )
    return results


@router.get("/export")
def export_attendance_report(
    db: Annotated[Session, Depends(get_db)],
    format: Literal["csv", "xlsx", "pdf"] = "csv",
    date: str | None = None,
    session_id: int | None = None,
    department: str | None = None,
) -> Response:
    """Exports filtered attendance data in CSV, Excel (.xlsx), or formatted academic PDF."""
    records = repository.list_attendance_records(
        db=db, date=date, session_id=session_id, department=department
    )

    if format == "csv":
        csv_content = generate_attendance_csv(records)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=attendance_report.csv"},
        )

    tmp_dir = tempfile.gettempdir()
    if format == "xlsx":
        file_path = f"{tmp_dir}/attendance_report.xlsx"
        export_attendance_excel(records, file_path)
        return FileResponse(
            path=file_path,
            filename="attendance_report.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # format == "pdf"
    file_path = f"{tmp_dir}/attendance_report.pdf"
    export_attendance_pdf(records, file_path)
    return FileResponse(
        path=file_path,
        filename="attendance_report.pdf",
        media_type="application/pdf",
    )
