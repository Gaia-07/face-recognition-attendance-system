# REST API Reference & Integration Guide

The **Face Recognition Attendance System (FRAS)** provides an enterprise REST API powered by FastAPI.

- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **OpenAPI JSON Schema**: `http://localhost:8000/openapi.json`

---

## 1. Authentication & RBAC

The API uses OAuth2 Password Bearer authentication with JSON Web Tokens (JWT).

### Role Hierarchy
| Role | Permissions |
| :--- | :--- |
| **Admin** | Full access: student enrollment, GDPR erasure, session control, reporting, audit inspection. |
| **Faculty** | Session management, live check-ins, manual status overrides, report export. |
| **Auditor** | Read-only inspection of attendance records, audit logs, and analytics. |

### Default Credentials
- **Admin**: `admin` / `admin123`
- **Faculty**: `faculty` / `faculty123`
- **Auditor**: `auditor` / `auditor123`

---

## 2. API Endpoints

### 2.1 System Health
- **`GET /healthz`**
  - Public readiness and liveness probe.
  - Returns backend name, model threshold ($\tau^*$), database status, and enrolled student count.

### 2.2 Authentication
- **`POST /api/auth/token`**
  - Form Data: `username`, `password`
  - Response: `{"access_token": "...", "token_type": "bearer", "role": "admin"}`

### 2.3 Student Management
- **`GET /api/students`**
  - Headers: `Authorization: Bearer <token>`
  - Optional Query Params: `department`, `skip`, `limit`
  - Returns list of registered students (with zero photographic data).
- **`POST /api/students`**
  - Headers: `Authorization: Bearer <token>` (Admin required)
  - Multipart Form: `external_id` (string), `name` (string), `department` (string), `photo` (file, JPEG/PNG)
  - Behavior: Extracts 128-d embedding, encrypts with AES-128, discards photo, inserts student record.
- **`GET /api/students/{external_id}`**
  - Returns individual student details.
- **`DELETE /api/students/{external_id}`**
  - Headers: `Authorization: Bearer <token>` (Admin required)
  - Behavior: **GDPR Art. 17 Right-to-Erasure**. Deletes student and encrypted biometric template, cascades attendance records, writes tombstone to audit log. Returns `204 No Content`.

### 2.4 Attendance Sessions
- **`POST /api/sessions`**
  - Headers: `Authorization: Bearer <token>` (Faculty/Admin)
  - JSON Body: `{"session_name": "CS101 Morning Lecture", "camera_id": "ENTRANCE_CAM_01"}`
  - Returns created session with status `active`.
- **`GET /api/sessions/active`**
  - Returns current ongoing active session, if any.
- **`POST /api/sessions/{session_id}/end`**
  - Closes an active attendance session.

### 2.5 Attendance Logging & Verification
- **`POST /api/attendance/check-in`**
  - Multipart Form: `photo` (JPEG/PNG), `session_id` (optional), `camera_id` (optional)
  - Behavior: Runs quality gates $\to$ extracts embedding $\to$ matches enrolled gallery $\to$ checks active session $\to$ applies 09:15 cutoff $\to$ enforces deduplication.
  - Response: Match status, student name, punctuality status (`Present` or `Late`), similarity score.
- **`POST /api/attendance/manual`**
  - Headers: `Authorization: Bearer <token>` (Faculty/Admin)
  - JSON Body: `{"student_id": 1, "session_id": 12, "status": "Present", "reason": "Medical slip"}`
  - Logs supervisory override to audit trail.
- **`GET /api/attendance/records`**
  - Query Params: `date`, `session_id`, `department`, `status`
- **`GET /api/attendance/export`**
  - Query Params: `format` (`csv`, `xlsx`, or `pdf`), `date`, `session_id`, `department`
  - Returns file attachment stream.

### 2.6 Analytics & Risk Forecasting
- **`GET /api/analytics/daily`**: Chronological daily attendance counts and punctuality rates.
- **`GET /api/analytics/departments`**: Department-wise attendance aggregations.
- **`GET /api/analytics/rankings`**: Student punctuality leaderboards.
- **`GET /api/analytics/heatmap`**: Student $\times$ date presence matrix.
- **`GET /api/analytics/absence-risks`**: ML Random Forest predictions of student absence probability.

---

## 3. Example `curl` Workflows

### Step 1: Authenticate as Faculty
```bash
TOKEN=$(curl -s -X POST "http://localhost:8000/api/auth/token" \
  -d "username=faculty&password=faculty123" | jq -r .access_token)
```

### Step 2: Create an Attendance Session
```bash
SESSION_ID=$(curl -s -X POST "http://localhost:8000/api/sessions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_name": "CS204 Data Structures", "camera_id": "ENTRANCE_CAM_01"}' \
  | jq -r .id)
echo "Active Session: $SESSION_ID"
```

### Step 3: Biometric Check-In via Face Photo
```bash
curl -X POST "http://localhost:8000/api/attendance/check-in" \
  -F "session_id=$SESSION_ID" \
  -F "camera_id=ENTRANCE_CAM_01" \
  -F "photo=@data/raw/face_attendance_dataset/images/person_001/person_001_01.jpg"
```

### Step 4: Export Formal Academic PDF Report
```bash
curl -X GET "http://localhost:8000/api/attendance/export?format=pdf" \
  -H "Authorization: Bearer $TOKEN" \
  --output attendance_report.pdf
```

### Step 5: Exercise GDPR Right-to-Erasure (Admin Only)
```bash
ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8000/api/auth/token" \
  -d "username=admin&password=admin123" | jq -r .access_token)

curl -X DELETE "http://localhost:8000/api/students/person_001" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```
