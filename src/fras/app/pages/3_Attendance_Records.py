"""Page 3: Attendance log history, multi-criteria filtering, and multi-format exports."""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from fras.attendance.reports import (
    export_attendance_excel,
    export_attendance_pdf,
    generate_attendance_csv,
)
from fras.db import repository
from fras.db.session import SessionLocal

st.set_page_config(page_title="Attendance Records - FRAS", page_icon="📋", layout="wide")
st.title("📋 Attendance Records & Audit Logs")
st.caption(
    "Inspect 66-day historical attendance, filter by department and status, and export certified reports."
)

with SessionLocal() as db:
    records = repository.list_attendance_records(db)
    students = repository.list_students(db)

# Filters
col_d, col_dept, col_status = st.columns(3)
with col_d:
    dates = sorted({r.date for r in records}, reverse=True) if records else []
    selected_date = st.selectbox("Filter Date", ["All Dates"] + dates)

with col_dept:
    departments = sorted({r.student.department for r in records if r.student}) if records else []
    selected_dept = st.selectbox("Filter Department", ["All Departments"] + departments)

with col_status:
    selected_status = st.selectbox(
        "Filter Status", ["All Statuses", "Present", "Late", "Absent", "Manual"]
    )

# Apply filtering
filtered = records
if selected_date != "All Dates":
    filtered = [r for r in filtered if r.date == selected_date]
if selected_dept != "All Departments":
    filtered = [r for r in filtered if r.student and r.student.department == selected_dept]
if selected_status != "All Statuses":
    filtered = [r for r in filtered if r.status == selected_status]

# Summary metrics
total_f = len(filtered)
present_f = sum(1 for r in filtered if r.status == "Present")
late_f = sum(1 for r in filtered if r.status == "Late")
absent_f = sum(1 for r in filtered if r.status == "Absent")
rate_f = ((present_f + late_f) / total_f * 100) if total_f > 0 else 0.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Records", total_f)
m2.metric("Present", present_f, f"{(present_f / total_f * 100):.1f}%" if total_f else "0%")
m3.metric("Late (After 09:15)", late_f, f"{(late_f / total_f * 100):.1f}%" if total_f else "0%")
m4.metric("Absent", absent_f, f"{(absent_f / total_f * 100):.1f}%" if total_f else "0%")
m5.metric("Attendance Rate", f"{rate_f:.1f}%")

st.markdown("---")

# Export bar
exp_c1, exp_c2, exp_c3 = st.columns(3)

with exp_c1:
    csv_str = generate_attendance_csv(filtered)
    st.download_button(
        label="📥 Download CSV Report",
        data=csv_str,
        file_name=f"fras_attendance_{selected_date}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with exp_c2:
    tmp_xlsx = Path(tempfile.gettempdir()) / "report_temp.xlsx"
    export_attendance_excel(filtered, tmp_xlsx)
    with open(tmp_xlsx, "rb") as f:
        st.download_button(
            label="📊 Download Excel (.xlsx) Report",
            data=f.read(),
            file_name=f"fras_attendance_{selected_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

with exp_c3:
    tmp_pdf = Path(tempfile.gettempdir()) / "report_temp.pdf"
    export_attendance_pdf(filtered, tmp_pdf)
    with open(tmp_pdf, "rb") as f:
        st.download_button(
            label="📄 Download Academic PDF Report",
            data=f.read(),
            file_name=f"fras_attendance_{selected_date}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

st.markdown("---")

# Data table
rows = []
for r in filtered:
    rows.append(
        {
            "Date": r.date,
            "Student ID": r.student.external_id if r.student else str(r.student_id),
            "Full Name": r.student.name if r.student else "Unknown",
            "Department": r.student.department if r.student else "N/A",
            "Status": r.status,
            "Check-In Time": r.check_in_time or "—",
            "Check-Out Time": r.check_out_time or "—",
            "Confidence": f"{r.confidence:.3f}" if r.confidence is not None else "—",
            "Camera ID": r.camera_id or "—",
            "Manual Override": "Yes" if r.is_manual_fallback else "No",
        }
    )

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=500)
else:
    st.info("No attendance records match the selected filter criteria.")
