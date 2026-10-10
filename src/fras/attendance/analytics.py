"""Attendance analytics, punctuality metrics, and ML-based absence risk prediction."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fras.db.models import AttendanceRecord


def compute_daily_attendance_summary(records: Sequence[AttendanceRecord]) -> list[dict[str, Any]]:
    """Aggregates attendance metrics grouped chronologically by date."""
    if not records:
        return []

    data = [
        {
            "date": r.date,
            "status": r.status,
            "student_id": r.student_id,
        }
        for r in records
    ]
    df = pd.DataFrame(data)
    summary = []

    for d, group in df.groupby("date", sort=True):
        total = len(group)
        present = int((group["status"] == "Present").sum())
        late = int((group["status"] == "Late").sum())
        absent = int((group["status"] == "Absent").sum())
        rate = ((present + late) / total * 100.0) if total > 0 else 0.0

        summary.append(
            {
                "date": str(d),
                "total": total,
                "present": present,
                "late": late,
                "absent": absent,
                "attendance_rate": round(rate, 2),
            }
        )

    return summary


def compute_department_breakdown(records: Sequence[AttendanceRecord]) -> list[dict[str, Any]]:
    """Computes aggregated attendance distribution per academic department."""
    if not records:
        return []

    data = [
        {
            "department": r.student.department if r.student else "Unknown",
            "status": r.status,
        }
        for r in records
    ]
    df = pd.DataFrame(data)
    breakdown = []

    for dept, group in df.groupby("department"):
        total = len(group)
        present = int((group["status"] == "Present").sum())
        late = int((group["status"] == "Late").sum())
        absent = int((group["status"] == "Absent").sum())
        rate = ((present + late) / total * 100.0) if total > 0 else 0.0

        breakdown.append(
            {
                "department": str(dept),
                "total": total,
                "present": present,
                "late": late,
                "absent": absent,
                "attendance_rate": round(rate, 2),
            }
        )

    return sorted(breakdown, key=lambda x: x["attendance_rate"], reverse=True)


def compute_student_punctuality_rankings(
    records: Sequence[AttendanceRecord],
) -> list[dict[str, Any]]:
    """Calculates student-level attendance frequency, punctuality, and tardiness."""
    if not records:
        return []

    data = [
        {
            "student_id": r.student_id,
            "external_id": r.student.external_id if r.student else "",
            "name": r.student.name if r.student else "",
            "department": r.student.department if r.student else "",
            "status": r.status,
        }
        for r in records
    ]
    df = pd.DataFrame(data)
    rankings = []

    for sid, group in df.groupby("student_id"):
        total = len(group)
        present = int((group["status"] == "Present").sum())
        late = int((group["status"] == "Late").sum())
        absent = int((group["status"] == "Absent").sum())
        att_rate = ((present + late) / total * 100.0) if total > 0 else 0.0
        punct_rate = (present / total * 100.0) if total > 0 else 0.0

        row_meta = group.iloc[0]
        rankings.append(
            {
                "student_id": int(str(sid)),
                "external_id": str(row_meta["external_id"]),
                "name": str(row_meta["name"]),
                "department": str(row_meta["department"]),
                "total_sessions": total,
                "present": present,
                "late": late,
                "absent": absent,
                "attendance_rate": round(att_rate, 2),
                "punctuality_rate": round(punct_rate, 2),
            }
        )

    return sorted(rankings, key=lambda x: x["attendance_rate"])


def build_attendance_heatmap_matrix(
    records: Sequence[AttendanceRecord],
) -> dict[str, Any]:
    """Constructs a matrix suitable for visualization (students x dates).

    Values:
        1.0: Present
        0.5: Late
        0.0: Absent
    """
    if not records:
        return {"dates": [], "students": [], "matrix": []}

    dates = sorted({r.date for r in records})
    students_map: dict[str, str] = {}
    record_dict: dict[tuple[str, str], float] = {}

    for r in records:
        name = r.student.name if r.student else str(r.student_id)
        pid = r.student.external_id if r.student else str(r.student_id)
        display_label = f"{pid}: {name}"
        students_map[pid] = display_label

        val = 1.0 if r.status == "Present" else (0.5 if r.status == "Late" else 0.0)
        record_dict[(pid, r.date)] = val

    student_keys = sorted(students_map.keys())
    matrix = []
    for pid in student_keys:
        row = [record_dict.get((pid, d), 0.0) for d in dates]
        matrix.append(row)

    return {
        "dates": dates,
        "students": [students_map[k] for k in student_keys],
        "matrix": matrix,
    }


# -----------------------------------------------------------------------------
# Predictive ML Model: Next-Day Absence Risk Estimator
# -----------------------------------------------------------------------------


def train_and_predict_absence_risk(
    records: Sequence[AttendanceRecord],
) -> list[dict[str, Any]]:
    """Trains a ML model on sequential attendance history and forecasts next-day absence risk."""
    if len(records) < 100:
        return []

    # Prepare historical sequences per student
    rows = []
    for r in records:
        rows.append(
            {
                "date": r.date,
                "student_id": r.student_id,
                "external_id": r.student.external_id if r.student else str(r.student_id),
                "name": r.student.name if r.student else "",
                "department": r.student.department if r.student else "",
                "status": r.status,
                "is_absent": 1 if r.status == "Absent" else 0,
                "is_late": 1 if r.status == "Late" else 0,
            }
        )

    df = pd.DataFrame(rows).sort_values(["student_id", "date"])
    unique_dates = sorted(df["date"].unique())
    if len(unique_dates) < 10:
        return []

    # Engineer chronological rolling features per student
    feature_rows = []
    for sid, group in df.groupby("student_id"):
        group = group.sort_values("date").reset_index(drop=True)
        for i in range(5, len(group)):
            prev_window = group.iloc[i - 5 : i]
            rolling_absent_rate = prev_window["is_absent"].mean()
            rolling_late_rate = prev_window["is_late"].mean()
            absent_yesterday = group.iloc[i - 1]["is_absent"]
            late_yesterday = group.iloc[i - 1]["is_late"]
            date_obj = datetime.strptime(group.iloc[i]["date"], "%Y-%m-%d")
            day_of_week = date_obj.weekday()

            feature_rows.append(
                {
                    "student_id": sid,
                    "external_id": group.iloc[i]["external_id"],
                    "name": group.iloc[i]["name"],
                    "department": group.iloc[i]["department"],
                    "date": group.iloc[i]["date"],
                    "rolling_absent_rate": rolling_absent_rate,
                    "rolling_late_rate": rolling_late_rate,
                    "absent_yesterday": absent_yesterday,
                    "late_yesterday": late_yesterday,
                    "day_of_week": day_of_week,
                    "target_absent": group.iloc[i]["is_absent"],
                }
            )

    feat_df = pd.DataFrame(feature_rows)
    if len(feat_df) < 50:
        return []

    feature_cols = [
        "rolling_absent_rate",
        "rolling_late_rate",
        "absent_yesterday",
        "late_yesterday",
        "day_of_week",
    ]

    # Time-based split: earlier 75% for training, latest 25% for evaluation
    cutoff_idx = int(len(unique_dates) * 0.75)
    train_dates = set(unique_dates[:cutoff_idx])

    train_data = feat_df[feat_df["date"].isin(train_dates)]
    clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    clf.fit(train_data[feature_cols], train_data["target_absent"])

    # Forecast risk for next session using each student's most recent 5 records
    predictions = []
    for sid, group in df.groupby("student_id"):
        recent = group.sort_values("date").tail(5)
        rolling_absent_rate = recent["is_absent"].mean()
        rolling_late_rate = recent["is_late"].mean()
        absent_yesterday = recent.iloc[-1]["is_absent"]
        late_yesterday = (
            recent.iloc[-1]["late_yesterday"]
            if "late_yesterday" in recent
            else recent.iloc[-1]["is_late"]
        )
        # Next weekday
        last_date = datetime.strptime(recent.iloc[-1]["date"], "%Y-%m-%d")
        next_dow = (last_date.weekday() + 1) % 5

        x_current = pd.DataFrame(
            [
                {
                    "rolling_absent_rate": rolling_absent_rate,
                    "rolling_late_rate": rolling_late_rate,
                    "absent_yesterday": absent_yesterday,
                    "late_yesterday": late_yesterday,
                    "day_of_week": next_dow,
                }
            ]
        )

        prob_absent = float(clf.predict_proba(x_current)[0, 1])

        # Assign risk tier
        if prob_absent >= 0.35:
            tier = "High"
        elif prob_absent >= 0.15:
            tier = "Medium"
        else:
            tier = "Low"

        meta = group.iloc[0]
        predictions.append(
            {
                "student_id": int(str(sid)),
                "external_id": str(meta["external_id"]),
                "name": str(meta["name"]),
                "department": str(meta["department"]),
                "risk_score": round(prob_absent, 3),
                "risk_tier": tier,
                "recent_absence_rate": round(rolling_absent_rate * 100, 1),
                "recent_late_rate": round(rolling_late_rate * 100, 1),
            }
        )

    return sorted(predictions, key=lambda x: x["risk_score"], reverse=True)
