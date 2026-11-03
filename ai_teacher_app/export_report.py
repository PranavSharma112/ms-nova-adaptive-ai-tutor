"""
export_report.py
-----------------
Export a session's Q&A history as CSV (raw event log) or PDF (readable report)
for a tutor/parent to review offline.
"""
from __future__ import annotations
import csv
import io
from typing import Dict, List

import storage

FIELDS = ["time", "topic", "kind", "difficulty", "correct", "hints_used", "elapsed",
          "error_type", "skill_tag", "reward", "question_source", "policy"]


def session_events(session_id: str) -> List[Dict]:
    return [e for e in storage.read_events() if e.get("session_id") == session_id]


def to_csv_bytes(session_id: str) -> bytes:
    events = session_events(session_id)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(FIELDS)
    if not events:
        w.writerow(["(no questions answered yet in this session)"])
    for e in events:
        import time as _t
        w.writerow([
            _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(e.get("ts", 0))),
            e.get("topic", ""), e.get("kind", ""), e.get("difficulty", ""),
            "yes" if e.get("correct") else "no", e.get("hints_used", 0),
            e.get("elapsed", ""), e.get("error_type", ""), e.get("skill_tag", ""),
            round(e.get("reward", 0) or 0, 3), e.get("question_source", ""), e.get("policy", "tabular"),
        ])
    return buf.getvalue().encode("utf-8")


def to_pdf_bytes(session_id: str, student_name: str = "", subject: str = "") -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    import time as _t

    events = session_events(session_id)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title="Tutoring session report")
    styles = getSampleStyleSheet()
    story = [Paragraph("Tutoring Session Report", styles["Title"]), Spacer(1, 6)]
    who = f"Student: {student_name}" if student_name else "Student: (not set)"
    subj = f" &nbsp;|&nbsp; Subject: {subject}" if subject else ""
    story.append(Paragraph(who + subj, styles["Normal"]))
    story.append(Spacer(1, 4))

    n = len(events)
    n_correct = sum(1 for e in events if e.get("correct"))
    n_hints = sum(e.get("hints_used", 0) or 0 for e in events)
    topics = sorted({e.get("topic", "") for e in events if e.get("topic")})
    summary = (f"Questions answered: {n} &nbsp;|&nbsp; Correct: {n_correct} "
               f"({(100 * n_correct / n):.0f}%)&nbsp;|&nbsp; Hints used: {n_hints}" if n else "No questions answered yet in this session.")
    story.append(Paragraph(summary, styles["Normal"]))
    if topics:
        story.append(Paragraph("Topics covered: " + ", ".join(topics), styles["Normal"]))
    story.append(Spacer(1, 12))

    if events:
        header = ["Time", "Topic", "Diff.", "Correct", "Hints", "Skill / Notes"]
        rows = [header]
        for e in events:
            rows.append([
                _t.strftime("%H:%M:%S", _t.localtime(e.get("ts", 0))),
                (e.get("topic") or "")[:22],
                str(e.get("difficulty", "")),
                "Yes" if e.get("correct") else "No",
                str(e.get("hints_used", 0)),
                (e.get("skill_tag") or e.get("error_type") or "")[:28],
            ])
        tbl = Table(rows, repeatRows=1, colWidths=[55, 130, 35, 45, 35, 150])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fb")]),
        ]))
        story.append(tbl)

    doc.build(story)
    return buf.getvalue()
