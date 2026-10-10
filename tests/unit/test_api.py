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
