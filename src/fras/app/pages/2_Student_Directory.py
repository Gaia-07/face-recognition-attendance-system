"""Page 2: Enrolled student directory, biometric enrollment, and GDPR Right-to-Erasure."""

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from fras.core.detection import FaceDetector
from fras.core.quality import QualityGate
from fras.db import repository
from fras.db.session import SessionLocal
from fras.recognition.avatar_cnn import AvatarCNNBackend

st.set_page_config(page_title="Student Directory - FRAS", page_icon="👥", layout="wide")
st.title("👥 Enrolled Student Directory & Biometric Management")
st.caption(
    "Manage student rosters, execute quality-gated face enrollment, and process GDPR erasure requests."
)

quality_gate = QualityGate()
detector = FaceDetector(backend_name="avatar_cnn")
backend = AvatarCNNBackend()

with SessionLocal() as db:
    students = repository.list_students(db)

tab_browse, tab_enroll, tab_gdpr = st.tabs(
    ["📋 Student Roster", "➕ Enroll New Student", "🛡️ GDPR Right-to-Erasure"]
)

with tab_browse:
    c_search, c_dept = st.columns([2, 1])
    with c_dept:
        departments = sorted({s.department for s in students}) if students else []
        selected_dept = st.selectbox("Filter Department", ["All"] + departments)
    with c_search:
        search_query = st.text_input("Search by Name or ID", placeholder="Search...")

    filtered = students
    if selected_dept != "All":
        filtered = [s for s in filtered if s.department == selected_dept]
    if search_query:
        q = search_query.lower()
        filtered = [s for s in filtered if q in s.name.lower() or q in s.external_id.lower()]

    st.markdown(f"**Showing {len(filtered)} of {len(students)} enrolled students**")

    table_data = [
        {
            "External ID": s.external_id,
            "Full Name": s.name,
            "Department": s.department,
            "Consent Timestamp": s.consent_at.strftime("%Y-%m-%d %H:%M"),
            "Biometric Security": "Fernet Encrypted (AES-128)",
            "Enrolled On": s.created_at.strftime("%Y-%m-%d"),
        }
        for s in filtered
    ]
    if table_data:
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info("No matching students found.")

with tab_enroll:
    st.markdown("### ➕ Register New Student with Biometric Photo")
    st.info(
        "Biometric enrollment enforces quality gating (blur variance, face size, illumination) prior to template encryption."
    )

    col_form, col_preview = st.columns([3, 2])
    with col_form:
        new_id = st.text_input("Student Identifier (External ID)", placeholder="e.g. person_031")
        new_name = st.text_input("Full Name", placeholder="e.g. Maya Patel")
        new_dept = st.selectbox(
            "Academic Department",
            [
                "Engineering",
                "HR",
                "Sales",
                "Support",
                "Marketing",
                "Operations",
                "Design",
                "Finance",
            ],
        )
        uploaded_photo = st.file_uploader(
            "Upload Enrollment Face Photo (128x128px or higher)", type=["jpg", "png", "jpeg"]
        )

    with col_preview:
        if uploaded_photo:
            pil_img = Image.open(uploaded_photo)
            st.image(pil_img, caption="Enrollment Photo Preview", width=200)

            # Quality validation
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            q_res = quality_gate.check(img_bgr)
            if q_res.passed:
                st.success(
                    f"✅ Quality Gating Passed! (Blur: {q_res.blur_score:.1f}, Luminance: {q_res.brightness_score:.1f})"
                )
            else:
                st.error(f"❌ Quality Rejected: {q_res.diagnostic_message}")

    if st.button("Complete Biometric Enrollment", type="primary", disabled=uploaded_photo is None):
        if not new_id or not new_name or uploaded_photo is None:
            st.error("Please provide external ID, full name, and enrollment photo.")
        else:
            with SessionLocal() as db:
                if repository.get_student_by_external_id(db, new_id):
                    st.error(f"Student ID '{new_id}' already exists!")
                else:
                    assert uploaded_photo is not None
                    img_bgr = cv2.cvtColor(np.array(Image.open(uploaded_photo)), cv2.COLOR_RGB2BGR)
                    q_res = quality_gate.check(img_bgr)
                    if not q_res.passed:
                        st.error(f"Cannot enroll: {q_res.diagnostic_message}")
                    else:
                        prep = backend.preprocess(img_bgr)
                        emb = backend.embed(prep)[0]
                        student = repository.create_student(
                            db=db,
                            external_id=new_id,
                            name=new_name,
                            department=new_dept,
                            embedding=emb,
                        )
                        st.success(
                            f"Enrolled student {student.name} ({student.external_id}) successfully!"
                        )
                        st.rerun()

with tab_gdpr:
    st.markdown("### 🛡️ GDPR Article 17 Right-to-Erasure Portal")
    st.warning(
        "Executing erasure permanently deletes the student record, all historical check-in logs, and encrypted biometric templates. This operation is irrevocable and recorded in the security audit log."
    )

    if students:
        student_erasure_opts = {
            f"{s.external_id}: {s.name} ({s.department})": s.external_id for s in students
        }
        sel_erase_label = st.selectbox(
            "Select Student for Irrevocable Erasure", list(student_erasure_opts.keys())
        )
        target_ext_id = student_erasure_opts[sel_erase_label]

        confirm_check = st.checkbox(
            "I verify that an official GDPR / DPDP right-to-erasure request has been verified."
        )

        if st.button(
            "🗑️ Permanently Erase Student and Biometrics",
            type="secondary",
            disabled=not confirm_check,
        ):
            with SessionLocal() as db:
                success = repository.delete_student(db, target_ext_id, actor="privacy_officer")
                if success:
                    st.success(
                        f"Student '{target_ext_id}' and all associated biometric templates permanently erased."
                    )
                    st.rerun()
                else:
                    st.error("Failed to delete student.")
    else:
        st.info("No students enrolled.")
