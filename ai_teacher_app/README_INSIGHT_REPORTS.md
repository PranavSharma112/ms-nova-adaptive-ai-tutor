# Ms. Nova: Insight Reports module (student + teacher)

New files: `insight_engine.py` (analytics + 14 pattern detectors), `report_builder.py` (HTML reports, inline SVG charts),
`tests/mock_events.py`, `tests/test_insight_engine.py`. Edited: `server.py` (+2 routes), `tutor_service.py` (logs `student`).

Routes
- `GET /api/session/{sid}/report/student.html[?narrative=true]` : student report (open in browser, Ctrl+P to save PDF)
- `GET /api/teacher/report/class.html[?subject=...]` : teacher report (needs teacher PIN header if set)

Test: `python tests/test_insight_engine.py`  -> ALL PASS

Detectors and thresholds (all in insight_engine.py, edit there)
| Pattern | Rule | Source |
|---|---|---|
| wheel_spinning | >=10 attempts on a topic, no mastery and never 3 correct in a row | Beck & Gong 2013 |
| early_wheel_risk | 5-9 attempts, <50% correct | early-detection literature |
| rapid_guessing | answer < 10% of mean time for that level (cap 10 s) | Wise & Ma 2012 |
| hint_abuse | >=2 quick multi-hint answers or >=1.5 hints/question | Aleven et al. 2006 |
| help_avoidance | >=3 wrong with no hint, >=70% of wrong answers | Aleven et al. 2006 |
| persistent_misconception | same skill, conceptual error >=2 | grader error_type |
| careless_slips / misreading / incomplete_answers | error_type counts >=2 | grader error_type |
| fluency_gap | >=2 correct answers slower than max(SLOW_SECONDS, 1.5x class mean) | |
| frustration_streak | >=3 wrong in a row | |
| fatigue_decline | first-third vs last-third accuracy drop >=25 pts (n>=9) | |
| difficulty_ceiling | level k >=70% then level k+1 <=40% | |
| forgetting | review question wrong after mastery >=0.70 | |
| item quality (teacher) | observed level 1+4(1-p) differs from label by >=2 (>=3 unhinted tries) | |

Limits: rules are validated on synthetic logs with injected patterns only. Thresholds need re-tuning once real logs exist.
Events logged before this change have no `student` field; the teacher report then groups by session.

---
# v3: Google Forms link + deep reports

Flow: `forms/create_form.gs` builds the quiz Form (every question + "How sure are you?") -> students answer ->
linked Google Sheet -> CSV export or live read (service account) -> `forms_pipeline.py` -> one report per student + teacher report.

Setup (once per quiz)
1. Write `quiz_config.json` (see `forms/quiz_config_example.json`). Map EVERY wrong option to a named misconception in
   `distractors`; put careless-error options in `slip_options`. This mapping is where diagnosis comes from.
2. script.google.com > new project > paste `forms/create_form.gs` > paste config into CONFIG > Run `createQuiz`.
3. After responses: Sheet > File > Download > CSV, then
   `python forms_pipeline.py --csv responses.csv --config quiz_config.json --out reports/ [--agent]`
   or in the app: POST `/api/teacher/forms/import` (files: responses, config) -> zip of reports.
   Live: `pip install gspread`, share the Sheet with the service-account email, use `--sheet URL --creds key.json`.
4. Re-run the same quiz family weekly into the same sheet: growth and forgetting appear automatically.

AI agent: `--agent` uses ChatGPT if OPENAI_API_KEY is set (OPENAI_MODEL, default gpt-4o-mini), else Gemini.
It writes coach notes, the teacher brief and misconception fixes; numbers not in the computed facts are rejected.

Forms limits: no per-question time or hints, so rapid-guess, hint and fluency detectors switch off; BKT parameters
are low confidence from few snapshots, so forms mode clusters on observed rates (accuracy, growth, slips,
misconceptions, topic gap, overconfidence). Recovery on the simulated cohort: 73% group purity (tutor logs: 77%).
Tests: tests/test_forms.py, tests/test_deep_engine.py, tests/test_insight_engine.py
