"""Unit tests for attendance business rules, reporting, and analytics."""

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from fras.attendance.analytics import (
    build_attendance_heatmap_matrix,
    compute_daily_attendance_summary,
    compute_department_breakdown,
    compute_student_punctuality_rankings,
)
from fras.attendance.reports import (
    export_attendance_csv_file,
    export_attendance_excel,
    export_attendance_pdf,
    generate_attendance_csv,
)
from fras.attendance.rules import (
    AttendanceStatus,
    classify_attendance_status,
)
from fras.attendance.service import AttendanceService
from fras.db import repository
from fras.db.models import Base
from fras.recognition.avatar_cnn import AvatarCNNBackend


@pytest.fixture
def memory_db() -> Session:
    """In-memory database session fixture."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


def test_classify_attendance_status() -> None:
    # 09:15:00 is Present (boundary condition)
    assert classify_attendance_status("09:15:00") == AttendanceStatus.PRESENT
    # 09:14:59 is Present
    assert classify_attendance_status("09:14:59") == AttendanceStatus.PRESENT
    # 09:15:01 is Late
    assert classify_attendance_status("09:15:01") == AttendanceStatus.LATE
    # 09:30:00 is Late
    assert classify_attendance_status("09:30:00") == AttendanceStatus.LATE


def test_attendance_service_flow(memory_db: Session) -> None:
    # Setup enrolled student with realistic image
    img_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "face_attendance_dataset"
        / "images"
        / "person_001"
        / "person_001_01.jpg"
    )
    if not img_path.exists():
        pytest.skip("Dataset raw images not present")

    backend = AvatarCNNBackend()
    img = cv2.imread(str(img_path))
    preprocessed = backend.preprocess(img)
    emb = backend.embed(preprocessed)[0]

    student = repository.create_student(
        db=memory_db,
        external_id="person_001",
        name="Anjali Iyer",
        department="Engineering",
        embedding=emb,
    )
    session = repository.create_session(memory_db, "Morning Session")

    service = AttendanceService(backend=backend, threshold=0.65)
    service.refresh_gallery(memory_db)

    # 1. First presentation -> successfully checked in
    res1 = service.process_frame(
        image_bgr=img,
        session_id=session.id,
        db=memory_db,
        timestamp=datetime.strptime("2026-10-10 09:10:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC),
    )
    assert res1.status == "checked_in"
    assert res1.attendance_status == "Present"
    assert res1.student_id == student.id

    # 2. Immediate second presentation -> deduplicated as already_marked
    res2 = service.process_frame(
        image_bgr=img,
        session_id=session.id,
        db=memory_db,
        timestamp=datetime.strptime("2026-10-10 09:12:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC),
    )
    assert res2.status == "already_marked"
    assert res2.student_id == student.id

    # 3. Unknown face presentation -> unknown_rejected
    np.random.seed(42)
    random_face = np.random.randint(80, 200, (128, 128, 3), dtype=np.uint8)
    res_unknown = service.process_frame(
        image_bgr=random_face,
        session_id=session.id,
        db=memory_db,
    )
    assert res_unknown.status in ("unknown_rejected", "quality_rejected")


def test_attendance_report_generation(tmp_path: Path, memory_db: Session) -> None:
    student = repository.create_student(
        db=memory_db,
        external_id="person_002",
        name="Rohan Nair",
        department="HR",
        embedding=np.zeros(128, dtype=np.float32),
    )
    session = repository.create_session(memory_db, "Test Session")
    rec = repository.record_attendance(
        db=memory_db,
        student_id=student.id,
        session_id=session.id,
        date="2026-10-10",
        check_in_time="09:10:00",
        status="Present",
        confidence=0.98,
        camera_id="CAM_01",
    )

    records = [rec]

    # CSV string
    csv_str = generate_attendance_csv(records)
    assert "Rohan Nair" in csv_str
    assert "Present" in csv_str

    # CSV file
    csv_file = tmp_path / "report.csv"
    export_attendance_csv_file(records, csv_file)
    assert csv_file.exists()
    assert csv_file.stat().st_size > 0

    # Excel file
    xlsx_file = tmp_path / "report.xlsx"
    export_attendance_excel(records, xlsx_file)
    assert xlsx_file.exists()
    assert xlsx_file.stat().st_size > 0

    # PDF file
    pdf_file = tmp_path / "report.pdf"
    export_attendance_pdf(records, pdf_file)
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 0


def test_attendance_analytics(memory_db: Session) -> None:
    student1 = repository.create_student(
        db=memory_db,
        external_id="p1",
        name="Alice",
        department="Engineering",
        embedding=np.zeros(128, dtype=np.float32),
    )
    student2 = repository.create_student(
        db=memory_db,
        external_id="p2",
        name="Bob",
        department="Design",
        embedding=np.zeros(128, dtype=np.float32),
    )
    session = repository.create_session(memory_db, "Session 1")

    rec1 = repository.record_attendance(
        db=memory_db,
        student_id=student1.id,
        session_id=session.id,
        date="2026-10-01",
        check_in_time="09:00:00",
        status="Present",
    )
    rec2 = repository.record_attendance(
        db=memory_db,
        student_id=student2.id,
        session_id=session.id,
        date="2026-10-01",
        check_in_time="09:20:00",
        status="Late",
    )

    records = [rec1, rec2]

    # Daily summary
    daily = compute_daily_attendance_summary(records)
    assert len(daily) == 1
    assert daily[0]["present"] == 1
    assert daily[0]["late"] == 1

    # Department breakdown
    dept = compute_department_breakdown(records)
    assert len(dept) == 2

    # Rankings
    rank = compute_student_punctuality_rankings(records)
    assert len(rank) == 2

    # Heatmap
    hm = build_attendance_heatmap_matrix(records)
    assert len(hm["dates"]) == 1
    assert len(hm["matrix"]) == 2
