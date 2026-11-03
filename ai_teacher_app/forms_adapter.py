"""
forms_adapter.py
----------------
Google Forms (quiz) responses  ->  Ms. Nova event log  ->  deep reports.

Source of responses (pick one):
  * CSV  : Google Sheet linked to the form -> File > Download > CSV
  * Sheet: read live with a Google service account (pip install gspread; share the sheet with the
           service-account email as Viewer). No student data is published to the web.

What a form CAN give: who, when, each answer, a per-question confidence rating (we add it).
What it CANNOT give: time per question, hints. Those detectors switch off automatically.
Diagnosis power comes from the quiz DESIGN: every wrong option is mapped to a named misconception.
"""
from __future__ import annotations
import csv, json, re, time
from datetime import datetime
from typing import Dict, List

from rl_engine import BKTTracker

CONF = {"guessing": 0.0, "not sure": 0.5, "unsure": 0.5, "somewhat sure": 0.5, "sure": 1.0, "very sure": 1.0}


def _norm(x) -> str: return re.sub(r"\s+", " ", str(x or "")).strip().lower()


def _ts(s: str) -> float:
    for fmt in ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try: return datetime.strptime(s.strip(), fmt).timestamp()
        except ValueError: pass
    return time.time()


def read_csv(path: str) -> List[Dict]:
    with open(path, encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def read_sheet(sheet_url: str, creds_json: str) -> List[Dict]:
    import gspread                                              # optional dependency
    return gspread.service_account(filename=creds_json).open_by_url(sheet_url).sheet1.get_all_records()


def to_events(rows: List[Dict], cfg: Dict) -> List[Dict]:
    """One form submission -> one event per question, with BKT replayed per student and topic
    across ALL submissions in time order (so week-2 quizzes show learning or forgetting)."""
    idc, namec, tsc = cfg.get("id_column", "Email Address"), cfg.get("name_column"), cfg.get("timestamp_column", "Timestamp")
    rows = sorted(rows, key=lambda r: _ts(r.get(tsc, "")))
    trackers: Dict[tuple, BKTTracker] = {}; seen: Dict[tuple, int] = {}; events = []
    for r in rows:
        sid = _norm(r.get(idc)) or _norm(r.get(namec)) or "anonymous"
        student = (str(r.get(namec)).strip() if namec and r.get(namec) else sid)
        sub = f"{sid}|{cfg.get('form_id', cfg.get('form_title', 'form'))}|{r.get(tsc, '')}"
        ts0 = _ts(r.get(tsc, ""))
        for k, q in enumerate(cfg["questions"]):
            ans = _norm(r.get(q["header"]))
            key_ok = [_norm(a) for a in (q["answer"] if isinstance(q["answer"], list) else [q["answer"]])]
            correct = int(ans in key_ok)
            dist = {_norm(a): m for a, m in q.get("distractors", {}).items()}
            conf_raw = _norm(r.get(q.get("confidence_header", ""), ""))
            conf = CONF.get(conf_raw)
            if correct: err, note = "none", ""
            elif not ans: err, note = "off_topic", "no answer"
            elif ans in dist: err, note = "conceptual_misconception", dist[ans]
            elif q.get("slip_options") and ans in {_norm(x) for x in q["slip_options"]}: err, note = "arithmetic_slip", ""
            else: err, note = "incomplete_answer", ""
            tkey = (sid, q["topic"])
            tr = trackers.setdefault(tkey, BKTTracker())
            d = int(q.get("difficulty", 2))
            # WHY: a confident correct answer is stronger evidence than a lucky guess.
            pb, pa = tr.observe(d, correct, guess_bonus=0.25 if (correct and conf == 0.0) else 0.0)
            kind = "review" if seen.get(tkey, 0) and seen[tkey] != hash(sub) else "question"
            seen.setdefault(tkey, hash(sub))
            events.append({"ts": ts0 + k, "session_id": sub, "student": student, "subject": cfg.get("subject", ""),
                           "topic": q["topic"], "kind": kind, "difficulty": d, "correct": correct, "elapsed": None,
                           "hints_used": 0, "confidence": conf, "error_type": err, "misconception_note": note,
                           "skill_tag": q.get("skill", q["topic"]), "item_id": q["id"], "answer_given": ans,
                           "p_before": pb, "p_after": pa, "source": "google_form"})
    return events
