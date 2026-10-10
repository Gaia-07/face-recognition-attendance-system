"""Page 1: Real-time biometric attendance processing and live camera simulation."""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from fras.attendance.service import AttendanceEventResult, AttendanceService
from fras.db import repository
from fras.db.session import SessionLocal

st.set_page_config(page_title="Live Attendance - FRAS", page_icon="🎥", layout="wide")
st.title("🎥 Live Biometric Attendance Gate")
st.caption(
    "Real-time facial verification, late cutoff enforcement (09:15), and session deduplication."
)

RAW_IMAGES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "raw"
    / "face_attendance_dataset"
    / "images"
)

# Initialize Attendance Service
if "attendance_service" not in st.session_state:
    st.session_state.attendance_service = AttendanceService()

service: AttendanceService = st.session_state.attendance_service

with SessionLocal() as db:
    active_session = repository.get_active_session(db)
    students = repository.list_students(db)

# Session management row
col_session, col_actions = st.columns([2, 1])

with col_session:
    if active_session:
        st.success(
            f"🟢 **Active Session:** `{active_session.session_name}` (ID: {active_session.id}) | Camera: `{active_session.camera_id}`"
        )
    else:
        st.warning(
            "⚠️ No active attendance session. Start a new session below to begin recording check-ins."
        )

with col_actions:
    if not active_session:
        session_name = st.text_input(
            "New Session Name", value=f"Lecture Gate - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        if st.button("🚀 Open Gate Session", type="primary"):
            with SessionLocal() as db:
                new_sess = repository.create_session(
                    db, session_name=session_name, camera_id="ENTRANCE_CAM_01"
                )
                st.rerun()
    else:
        if st.button("⏹️ Close Gate Session", type="secondary"):
            with SessionLocal() as db:
                repository.end_session(db, active_session.id)
                st.rerun()

st.markdown("---")

left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown("### 📷 Video Stream Ingestion")
    input_mode = st.radio(
        "Select Video / Frame Source",
        ["Simulated Avatar Stream", "Upload Frame / Photo", "Webcam Snapshot"],
        horizontal=True,
    )

    frame_to_process: np.ndarray | None = None

    if input_mode == "Simulated Avatar Stream":
        st.caption(
            "Demonstration mode: select from benchmark dataset identities or test avatar probes."
        )
        # List available identity folders
        person_folders = sorted([p.name for p in RAW_IMAGES_DIR.glob("person_*")])
        if person_folders:
            sel_pid = st.selectbox("Select Avatar Identity", person_folders, index=0)
            img_files = sorted((RAW_IMAGES_DIR / sel_pid).glob("*.jpg"))
            if img_files:
                sel_img_idx = st.slider("Select Avatar Variation Frame", 1, len(img_files), 1)
                target_path = img_files[sel_img_idx - 1]
                pil_img = Image.open(target_path)
                frame_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                st.image(pil_img, caption=f"Selected Frame: {target_path.name}", width=256)

    elif input_mode == "Upload Frame / Photo":
        uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            pil_img = Image.open(uploaded_file)
            frame_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            st.image(pil_img, caption="Uploaded Frame", width=256)

    elif input_mode == "Webcam Snapshot":
        camera_img = st.camera_input("Capture Live Camera Frame")
        if camera_img:
            pil_img = Image.open(camera_img)
            frame_to_process = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if st.button(
        "⚡ Verify & Mark Attendance",
        type="primary",
        disabled=frame_to_process is None or active_session is None,
    ):
        if frame_to_process is not None and active_session is not None:
            with SessionLocal() as db:
                result: AttendanceEventResult = service.process_frame(
                    image_bgr=frame_to_process,
                    session_id=active_session.id,
                    db=db,
                    camera_id=active_session.camera_id,
                )

            # Display verification outcome
            if result.status == "checked_in":
                st.success(
                    f"✅ **{result.message}**\n\nConfidence: `{result.confidence:.3f}` | Status: `{result.attendance_status}`"
                )
            elif result.status == "already_marked":
                st.warning(
                    f"⚠️ **{result.message}**\n\nDuplicate check-in blocked by operational rules."
                )
            elif result.status == "unknown_rejected":
                st.error(
                    f"❌ **{result.message}**\n\nOpen-set rejection: face template does not match enrolled gallery above calibrated threshold."
                )
            elif result.status == "quality_rejected":
                st.warning(f"🛡️ **{result.message}**")
            else:
                st.info(f"ℹ️ {result.message}")

            # Draw visual annotation box on frame
            annotated_bgr = frame_to_process.copy()
            if result.bounding_box:
                x, y, w, h = result.bounding_box
                box_color = (
                    (0, 200, 0)
                    if result.status == "checked_in"
                    else ((0, 200, 255) if result.status == "already_marked" else (0, 0, 255))
                )
                cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), box_color, 2)
                label = f"{result.student_name or 'Unknown'} ({result.confidence:.2f})"
                cv2.putText(
                    annotated_bgr,
                    label,
                    (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    box_color,
                    1,
                )

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Annotated Biometric Frame", width=300)

with right_col:
    st.markdown("### 📋 Recent Gate Check-Ins")
    if active_session:
        with SessionLocal() as db:
            recent_records = repository.list_attendance_records(db, session_id=active_session.id)

        if recent_records:
            for r in recent_records[:10]:
                badge = "🟢" if r.status == "Present" else ("🟡" if r.status == "Late" else "🔵")
                st.markdown(
                    f"{badge} **{r.student.name if r.student else 'ID ' + str(r.student_id)}** "
                    f"(`{r.student.department if r.student else ''}`) — *{r.status}* at `{r.check_in_time}` "
                    f"(Conf: {r.confidence:.2f})"
                )
        else:
            st.info("No check-ins recorded yet in the current active session.")
    else:
        st.info("Start an active session to view live check-ins.")

st.markdown("---")
st.markdown("### ✍️ Manual Fallback Entry (Exception Handling)")
with st.expander("Submit Manual Attendance Exception"):
    st.caption(
        "Guaranteed manual fallback per Academic Privacy Charter Chapter 16: students cannot be penalized for camera/illumination failures."
    )
    with SessionLocal() as db:
        student_opts = {
            f"{s.external_id} - {s.name} ({s.department})": s.id
            for s in repository.list_students(db)
        }

    if student_opts and active_session:
        sel_student_label = st.selectbox("Select Student", list(student_opts.keys()))
        target_student_id = student_opts[sel_student_label]
        m_status = st.selectbox("Status", ["Present", "Late"])
        m_reason = st.text_input(
            "Reason for Manual Override", value="Camera hardware failure / physical verification"
        )

        if st.button("Submit Manual Override"):
            with SessionLocal() as db:
                rec = service.record_manual_fallback(
                    db=db,
                    student_id=target_student_id,
                    session_id=active_session.id,
                    status=m_status,
                    reason=m_reason,
                    actor="faculty_dashboard",
                )
                st.success(
                    f"Manual attendance recorded for student ID {target_student_id} ({m_status})."
                )
                st.rerun()
    elif not active_session:
        st.warning("Please start an active session first.")
