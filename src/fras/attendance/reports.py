"""Report generation services: CSV, Excel (XLSX), and formatted academic PDF exports."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from fras.db.models import AttendanceRecord


def _records_to_dataframe(records: Sequence[AttendanceRecord]) -> pd.DataFrame:
    """Converts a sequence of AttendanceRecord instances into a structured DataFrame."""
    rows = []
    for r in records:
        rows.append(
            {
                "Date": r.date,
                "Session ID": r.session_id,
                "Student ID": r.student_id,
                "External ID": r.student.external_id if r.student else "",
                "Name": r.student.name if r.student else "",
                "Department": r.student.department if r.student else "",
                "Status": r.status,
                "Check-in Time": r.check_in_time or "N/A",
                "Check-out Time": r.check_out_time or "N/A",
                "Confidence": f"{r.confidence:.3f}" if r.confidence is not None else "N/A",
                "Camera ID": r.camera_id or "N/A",
                "Manual Fallback": "Yes" if r.is_manual_fallback else "No",
                "Manual Reason": r.manual_reason or "",
            }
        )
    return pd.DataFrame(rows)


def generate_attendance_csv(records: Sequence[AttendanceRecord]) -> str:
    """Renders attendance records as a RFC 4180 compliant CSV string."""
    df = _records_to_dataframe(records)
    return df.to_csv(index=False)


def export_attendance_csv_file(
    records: Sequence[AttendanceRecord], output_path: str | Path
) -> Path:
    """Writes attendance records to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _records_to_dataframe(records)
    df.to_csv(path, index=False)
    return path


def export_attendance_excel(records: Sequence[AttendanceRecord], output_path: str | Path) -> Path:
    """Generates an Excel workbook (.xlsx) with attendance data and summary statistics."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _records_to_dataframe(records)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Attendance Records", index=False)

        # Summary sheet
        total = len(df)
        present = int((df["Status"] == "Present").sum())
        late = int((df["Status"] == "Late").sum())
        absent = int((df["Status"] == "Absent").sum())
        rate = ((present + late) / total * 100) if total > 0 else 0.0

        summary_df = pd.DataFrame(
            [
                {"Metric": "Total Records", "Value": total},
                {"Metric": "Present Count", "Value": present},
                {"Metric": "Late Count", "Value": late},
                {"Metric": "Absent Count", "Value": absent},
                {"Metric": "Attendance Rate (%)", "Value": f"{rate:.1f}%"},
            ]
        )
        summary_df.to_excel(writer, sheet_name="Summary Metrics", index=False)

    return path


def export_attendance_pdf(
    records: Sequence[AttendanceRecord],
    output_path: str | Path,
    title: str = "Biometric Attendance Audit Report",
) -> Path:
    """Renders an academic PDF attendance audit report using ReportLab."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12,
    )

    story = [
        Paragraph(title, title_style),
        Paragraph(
            "Govt. Women's Polytechnic, Ranchi — Face Recognition Attendance System (FRAS)",
            subtitle_style,
        ),
        Spacer(1, 10),
    ]

    # Summary metrics block
    total = len(records)
    present = sum(1 for r in records if r.status == "Present")
    late = sum(1 for r in records if r.status == "Late")
    absent = sum(1 for r in records if r.status == "Absent")
    rate = ((present + late) / total * 100) if total > 0 else 0.0

    summary_data = [
        ["Total Records", "Present", "Late", "Absent", "Attendance Rate"],
        [str(total), str(present), str(late), str(absent), f"{rate:.1f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[108] * 5)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EDF2F7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # Detailed Records Table (top 50 if list is long to keep PDF report clean)
    display_records = list(records)[:50]
    table_data = [["Date", "ID", "Name", "Dept", "Status", "Time", "Conf", "Camera"]]
    for r in display_records:
        table_data.append(
            [
                r.date,
                r.student.external_id if r.student else "",
                r.student.name[:16] if r.student else "",
                r.student.department[:12] if r.student else "",
                r.status,
                r.check_in_time or "-",
                f"{r.confidence:.2f}" if r.confidence is not None else "-",
                r.camera_id[:12] if r.camera_id else "-",
            ]
        )

    col_widths = [65, 65, 100, 75, 55, 55, 45, 80]
    rec_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    rec_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3748")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (2, 1), (2, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(rec_table)

    if len(records) > 50:
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                f"Showing first 50 of {len(records)} records. For complete logs, refer to CSV / Excel export.",
                subtitle_style,
            )
        )

    doc.build(story)
    return path
