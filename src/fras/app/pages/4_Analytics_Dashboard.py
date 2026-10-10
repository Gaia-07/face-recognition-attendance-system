"""Page 4: Attendance analytics, punctuality metrics, heatmaps, and predictive ML absence risk."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fras.attendance.analytics import (
    build_attendance_heatmap_matrix,
    compute_daily_attendance_summary,
    compute_department_breakdown,
    compute_student_punctuality_rankings,
    train_and_predict_absence_risk,
)
from fras.db import repository
from fras.db.session import SessionLocal

st.set_page_config(page_title="Analytics - FRAS", page_icon="📊", layout="wide")
st.title("📊 Attendance Analytics & ML Predictive Insights")
st.caption(
    "Punctuality trends across 66 days, department distributions, and sequential machine learning absence forecasting."
)

with SessionLocal() as db:
    records = repository.list_attendance_records(db)

if not records:
    st.warning("No attendance records found in database.")
    st.stop()

daily_summary = compute_daily_attendance_summary(records)
dept_breakdown = compute_department_breakdown(records)
rankings = compute_student_punctuality_rankings(records)

# 1. Daily Trendline
st.markdown("### 📈 Chronological Attendance Trend (66 Sessions)")
if daily_summary:
    df_daily = pd.DataFrame(daily_summary)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values("date").set_index("date")

    col_chart, col_stats = st.columns([3, 1])
    with col_chart:
        st.line_chart(df_daily[["present", "late", "absent"]], use_container_width=True)
    with col_stats:
        avg_rate = df_daily["attendance_rate"].mean()
        avg_late = df_daily["late"].mean()
        avg_absent = df_daily["absent"].mean()
        st.metric("Avg Daily Attendance", f"{avg_rate:.1f}%")
        st.metric("Avg Daily Tardy (Late)", f"{avg_late:.1f} students")
        st.metric("Avg Daily Absences", f"{avg_absent:.1f} students")

st.markdown("---")

# 2. Department Breakdown
st.markdown("### 🏛️ Department Attendance Breakdown")
if dept_breakdown:
    df_dept = pd.DataFrame(dept_breakdown).set_index("department")
    c_bar, c_tbl = st.columns([2, 1])
    with c_bar:
        st.bar_chart(df_dept["attendance_rate"], use_container_width=True)
    with c_tbl:
        st.dataframe(
            df_dept[["total", "present", "late", "absent", "attendance_rate"]].rename(
                columns={"attendance_rate": "Rate (%)"}
            ),
            use_container_width=True,
        )

st.markdown("---")

# 3. Attendance Heatmap Matrix
st.markdown("### 🗺️ Student Attendance Heatmap Matrix (1=Present, 0.5=Late, 0=Absent)")
hm_data = build_attendance_heatmap_matrix(records)
if hm_data["matrix"]:
    matrix_np = np.array(hm_data["matrix"])  # (Students, Dates)
    fig, ax = plt.subplots(figsize=(16, 8))
    cax = ax.imshow(matrix_np, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(hm_data["students"])))
    ax.set_yticklabels(hm_data["students"], fontsize=8)
    # Sample every 5 dates for cleaner x-axis
    date_step = max(1, len(hm_data["dates"]) // 12)
    ax.set_xticks(range(0, len(hm_data["dates"]), date_step))
    ax.set_xticklabels(
        [hm_data["dates"][i] for i in range(0, len(hm_data["dates"]), date_step)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_xlabel("Date (YYYY-MM-DD)", fontsize=10)
    ax.set_ylabel("Enrolled Students", fontsize=10)
    plt.colorbar(
        cax,
        ax=ax,
        orientation="horizontal",
        pad=0.15,
        label="Green = Present | Yellow = Late | Red = Absent",
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# 4. Predictive Machine Learning: Next-Day Absence Risk
st.markdown("### 🤖 Predictive ML: Next-Day Absence Risk Forecaster")
st.info(
    "Trained on rolling 5-day sequential history, day-of-week, and consecutive tardiness using a chronological time-based split. "
    "Flagged students indicate elevated probability of non-attendance in the next scheduled gate session."
)

predictions = train_and_predict_absence_risk(records)
if predictions:
    pred_df = pd.DataFrame(predictions)

    # Highlight high risk in red
    def color_risk(val: str) -> str:
        if val == "High":
            return "background-color: #FED7D7; color: #9B2C2C; font-weight: bold;"
        elif val == "Medium":
            return "background-color: #FEFCBF; color: #975A16;"
        return "background-color: #C6F6D5; color: #22543D;"

    high_risk_count = sum(1 for p in predictions if p["risk_tier"] == "High")
    med_risk_count = sum(1 for p in predictions if p["risk_tier"] == "Medium")

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(
        "High Risk Students", high_risk_count, delta="Requires Outreach", delta_color="inverse"
    )
    kpi2.metric("Medium Risk Students", med_risk_count)
    kpi3.metric("Low Risk Students", len(predictions) - high_risk_count - med_risk_count)

    st.dataframe(
        pred_df[
            [
                "external_id",
                "name",
                "department",
                "risk_tier",
                "risk_score",
                "recent_absence_rate",
                "recent_late_rate",
            ]
        ].rename(
            columns={
                "external_id": "Student ID",
                "name": "Full Name",
                "department": "Department",
                "risk_tier": "Risk Tier",
                "risk_score": "Absence Probability",
                "recent_absence_rate": "Recent Absence %",
                "recent_late_rate": "Recent Late %",
            }
        ),
        use_container_width=True,
    )
else:
    st.info("Insufficient longitudinal data to generate sequential absence forecasts.")
