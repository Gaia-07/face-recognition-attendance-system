"""Unit tests for FastAPI REST API endpoints using Starlette/FastAPI TestClient."""

import pytest
from starlette.testclient import TestClient

from fras.api.main import app
from fras.db.session import init_db


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provides a TestClient connected to the seeded testing database."""
    init_db()
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Face Recognition Attendance System (FRAS)"
    assert data["health_check"] == "/healthz"


def test_health_check_endpoint(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["backend"] == "avatar_cnn"
    assert "calibrated_threshold" in data
    assert "total_enrolled_students" in data


def test_auth_login_flow(client: TestClient) -> None:
    # 1. Failed login with bad password
    bad_resp = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "wrongpassword"},
    )
    assert bad_resp.status_code == 401

    # 2. Successful login with default seeded credentials
    good_resp = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin123"},
    )
    assert good_resp.status_code == 200
    token_data = good_resp.json()
    assert "access_token" in token_data
    assert token_data["role"] == "admin"
    assert token_data["token_type"] == "bearer"


def test_list_students_endpoint(client: TestClient) -> None:
    response = client.get("/api/students")
    assert response.status_code == 200
    students = response.json()
    assert isinstance(students, list)
    assert len(students) >= 1
    # Check structure
    s = students[0]
    assert "external_id" in s
    assert "name" in s
    assert "department" in s


def test_list_attendance_records(client: TestClient) -> None:
    response = client.get("/api/attendance/records")
    assert response.status_code == 200
    records = response.json()
    assert isinstance(records, list)
    assert len(records) >= 1
    rec = records[0]
    assert "date" in rec
    assert "status" in rec
    assert "student_id" in rec


def test_attendance_csv_export(client: TestClient) -> None:
    response = client.get("/api/attendance/export?format=csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    content = response.text
    assert "Date,Session ID,Student ID" in content or "Date" in content


def test_analytics_endpoints(client: TestClient) -> None:
    # Daily summary
    daily_resp = client.get("/api/analytics/daily")
    assert daily_resp.status_code == 200
    daily_data = daily_resp.json()
    assert isinstance(daily_data, list)
    assert len(daily_data) >= 1

    # Department breakdown
    dept_resp = client.get("/api/analytics/departments")
    assert dept_resp.status_code == 200
    dept_data = dept_resp.json()
    assert isinstance(dept_data, list)
    assert len(dept_data) >= 1
    assert "department" in dept_data[0]


def test_session_lifecycle_and_attendance_flow(client: TestClient) -> None:
    # 1. Login as faculty
    login_resp = client.post(
        "/api/auth/token",
        data={"username": "faculty", "password": "faculty123"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Create session
    sess_resp = client.post(
        "/api/sessions",
        json={"session_name": "API Test Gate Session", "camera_id": "TEST_CAM"},
        headers=headers,
    )
    assert sess_resp.status_code == 201
    sess_id = sess_resp.json()["id"]

    # 3. Get active session
    active_resp = client.get("/api/sessions/active")
    assert active_resp.status_code == 200
    assert active_resp.json()["id"] == sess_id

    # 4. Manual override
    manual_resp = client.post(
        "/api/attendance/manual",
        json={
            "student_id": 1,
            "session_id": sess_id,
            "status": "Present",
            "reason": "Test manual verification",
        },
        headers=headers,
    )
    assert manual_resp.status_code == 200
    assert manual_resp.json()["status"] == "Present"

    # Export reports (xlsx and pdf)
    export_xlsx = client.get("/api/attendance/export?format=xlsx")
    assert export_xlsx.status_code == 200
    assert len(export_xlsx.content) > 0

    export_pdf = client.get("/api/attendance/export?format=pdf")
    assert export_pdf.status_code == 200
    assert len(export_pdf.content) > 0

    # Risk analytics and health
    risk_resp = client.get("/api/analytics/absence-risks")
    assert risk_resp.status_code == 200

    health_resp = client.get("/healthz")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    # 5. End session
    end_resp = client.post(f"/api/sessions/{sess_id}/end", headers=headers)
    assert end_resp.status_code == 200
    assert end_resp.json()["ended_at"] is not None


def test_student_enrollment_checkin_and_erasure_api(client: TestClient) -> None:
    from pathlib import Path

    # 1. Login as admin
    admin_login = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin123"},
    )
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 2. Enroll a student with face photo
    img_path = Path("data/raw/face_attendance_dataset/images/person_001/person_001_01.jpg")
    if not img_path.exists():
        return

    with open(img_path, "rb") as f:
        enroll_resp = client.post(
            "/api/students",
            data={
                "external_id": "test_api_person_999",
                "name": "API Test User",
                "department": "Engineering",
            },
            files={"photo": ("face.jpg", f, "image/jpeg")},
            headers=admin_headers,
        )
    assert enroll_resp.status_code == 201

    # 3. Create session & check-in
    sess_resp = client.post(
        "/api/sessions",
        json={"session_name": "Check-in API Test", "camera_id": "CAM_01"},
        headers=admin_headers,
    )
    sess_id = sess_resp.json()["id"]

    with open(img_path, "rb") as f:
        checkin_resp = client.post(
            "/api/attendance/check-in",
            data={"session_id": str(sess_id), "camera_id": "CAM_01"},
            files={"photo": ("face.jpg", f, "image/jpeg")},
        )
    assert checkin_resp.status_code == 200

    # 4. Close session
    client.post(f"/api/sessions/{sess_id}/end", headers=admin_headers)

    # 5. GDPR erasure
    del_student = client.delete("/api/students/test_api_person_999", headers=admin_headers)
    assert del_student.status_code == 204
