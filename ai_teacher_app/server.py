"""
server.py - HTTP API + static front end for Ms. Nova.

Run:   python server.py            (opens http://127.0.0.1:8000)
Key:   GEMINI_API_KEY in the environment, or in .streamlit/secrets.toml  (never sent to the browser)
Optional: TEACHER_PIN (same places) protects the Teacher view.
"""
import json, os, shutil, threading, time, webbrowser
from typing import Dict, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import storage
import teacher_engine as te
import tutor_service as ts
from ope_engine import load_real_trajectories, run_full_ope_analysis
from rl_engine import DEFAULT_TAU, generate_synthetic_logs, train_q

ROOT = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(ROOT, "models", "q_table.npy")
CAND = os.path.join(ROOT, "models", "q_table_candidate.npy")
REPORT = os.path.join(ROOT, "models", "train_report.json")
MAX_UPLOAD = 15 * 1024 * 1024


def get_secret(name: str) -> str:
    v = os.environ.get(name)
    if v: return v.strip()
    p = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(p):
        try:
            import tomllib
            with open(p, "rb") as f: return str(tomllib.load(f).get(name, "")).strip()
        except Exception:
            import re
            m = re.search(rf'^\s*{name}\s*=\s*["\']([^"\']+)["\']', open(p, encoding="utf-8").read(), re.M)
            return m.group(1).strip() if m else ""
    return ""


app = FastAPI(title="Ms. Nova")
SESSIONS: Dict[str, ts.TutorSession] = {}
OPE_STATE: Dict = {"passed": False, "cand_mtime": None, "source": None}
RUNTIME = {"key": ""}      # key typed into the UI: server memory only, never written to disk


def _purge():
    now = time.time()
    for k in [k for k, s in SESSIONS.items() if now - s.created > 12 * 3600]: SESSIONS.pop(k, None)
    if len(SESSIONS) > 300:
        for k in sorted(SESSIONS, key=lambda k: SESSIONS[k].created)[:len(SESSIONS) - 300]: SESSIONS.pop(k, None)

def sess(sid: str) -> ts.TutorSession:
    s = SESSIONS.get(sid)
    if not s: raise HTTPException(404, "Session expired. Start a new session.")
    return s

def ndjson(s: ts.TutorSession, make_gen):
    """Stream events as newline-delimited JSON, one request at a time per session."""
    def gen():
        if not s.lock.acquire(blocking=False):
            yield json.dumps({"type": "error", "message": "Ms. Nova is still working on the last step. Give her a moment."}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"; return
        try:
            for ev in make_gen():
                yield json.dumps(ev, default=str) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": ts.friendly_error(e), "retry": True}) + "\n"
            try: yield json.dumps(s.state_event(), default=str) + "\n"
            except Exception: pass
        finally:
            s.lock.release()
        yield json.dumps({"type": "done"}) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ pages / config
@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))

@app.get("/api/config")
def config():
    return {"has_server_key": bool(get_secret("GEMINI_API_KEY") or RUNTIME["key"]), "pin_required": bool(get_secret("TEACHER_PIN"))}


# ------------------------------------------------------------------ student session
class NewSession(BaseModel):
    api_key: Optional[str] = ""
    student_name: Optional[str] = ""

@app.post("/api/session")
def new_session(body: NewSession):
    _purge()
    key = (body.api_key or "").strip() or RUNTIME["key"] or get_secret("GEMINI_API_KEY")
    s = ts.TutorSession(api_key=key, student_name=(body.student_name or "")[:40])
    SESSIONS[s.id] = s
    return {"session_id": s.id}

@app.get("/api/session/{sid}")
def get_session(sid: str): return sess(sid).snapshot()

@app.post("/api/session/{sid}/start")
def start(sid: str):
    s = sess(sid); return ndjson(s, s.start)

class KeyBody(BaseModel):
    api_key: str

@app.post("/api/session/{sid}/key")
def set_key(sid: str, body: KeyBody):
    s = sess(sid); k = body.api_key.strip()
    if not k: raise HTTPException(400, "Enter a key or type demo.")
    s.api_key = k; RUNTIME["key"] = k
    return {"ok": True, "demo": te.is_demo(k)}

class SettingsBody(BaseModel):
    verify: Optional[bool] = None
    prefetch: Optional[bool] = None
    student_name: Optional[str] = None
    policy: Optional[str] = None
    agent_mode: Optional[bool] = None

@app.post("/api/session/{sid}/settings")
def settings(sid: str, body: SettingsBody):
    s = sess(sid)
    if body.verify is not None: s.settings["verify"] = body.verify
    if body.prefetch is not None: s.settings["prefetch"] = body.prefetch
    if body.student_name is not None: s.student_name = body.student_name[:40]
    if body.policy in ("tabular", "linear"): s.settings["policy"] = body.policy
    if body.agent_mode is not None: s.settings["agent_mode"] = body.agent_mode
    return {"ok": True, "settings": s.settings}

class MessageBody(BaseModel):
    text: str

@app.post("/api/session/{sid}/message")
def message(sid: str, body: MessageBody):
    s = sess(sid); return ndjson(s, lambda: s.handle_message(body.text))

@app.post("/api/session/{sid}/hint")
def hint(sid: str):
    s = sess(sid); return ndjson(s, s.hint)

@app.post("/api/session/{sid}/continue")
def cont(sid: str):
    s = sess(sid); return ndjson(s, s.continue_)

@app.post("/api/session/{sid}/recap")
def recap(sid: str):
    s = sess(sid); return ndjson(s, s.recap)

@app.post("/api/session/{sid}/photo")
async def photo(sid: str, file: UploadFile = File(...)):
    s = sess(sid)
    data = await file.read()
    if len(data) > MAX_UPLOAD: raise HTTPException(413, "Photo is too large (15 MB max).")
    mime = file.content_type or "image/jpeg"
    if not mime.startswith("image/"): raise HTTPException(400, "Please upload an image.")
    if not (s.stage == "quiz" and s.awaiting): raise HTTPException(409, "There's no open question to answer right now.")
    def run():
        yield s.say("student", "📷 Sent a photo of my work", "photo")
        yield from s.grade("", image=(data, mime))
    return ndjson(s, run)

@app.get("/api/session/{sid}/export.csv")
def export_csv(sid: str):
    import export_report as er
    s = sess(sid)
    data = er.to_csv_bytes(s.log_id)
    return StreamingResponse(iter([data]), media_type="text/csv",
                              headers={"Content-Disposition": f'attachment; filename="session_{sid[:8]}.csv"'})

@app.get("/api/session/{sid}/export.pdf")
def export_pdf(sid: str):
    import export_report as er
    s = sess(sid)
    data = er.to_pdf_bytes(s.log_id, s.student_name, s.subject)
    return StreamingResponse(iter([data]), media_type="application/pdf",
                              headers={"Content-Disposition": f'attachment; filename="session_{sid[:8]}.pdf"'})

@app.get("/api/session/{sid}/report/student.html")
def student_insight_report(sid: str, narrative: bool = False):
    # WHY: deep student diagnostics; class time norms make rapid/slow thresholds fair across students.
    import report_builder as rb, insight_engine as ie, export_report as er
    s = sess(sid)
    norms = ie.time_norms(storage.read_events())
    page = rb.student_report(er.session_events(s.log_id), s.student_name or "Student", norms, s.misconceptions,
                             api_key=s.api_key if narrative else None)
    return HTMLResponse(page)

@app.post("/api/session/{sid}/save")
def save(sid: str):
    s = sess(sid)
    if not s.topic_stats: raise HTTPException(400, "Answer a question first, then there's something to save.")
    try: return s.save_memory()
    except Exception as e: raise HTTPException(500, ts.friendly_error(e))

@app.post("/api/session/{sid}/materials")
async def materials(sid: str, file: UploadFile = File(...)):
    s = sess(sid)
    data = await file.read()
    if len(data) > MAX_UPLOAD: raise HTTPException(413, "File is too large (15 MB max).")
    name = (file.filename or "notes.txt")[:80]
    if not name.lower().endswith((".txt", ".md", ".pdf")): raise HTTPException(400, "Use a .txt, .md or .pdf file.")
    try: return s.add_material(name, data)
    except Exception as e: raise HTTPException(400, str(e))



# ------------------------------------------------------------------ Learning Lab (session-scoped)
import learning_lab as lab

@app.get("/api/session/{sid}/lab/progress")
def lab_progress(sid: str):
    s = sess(sid)
    return {"progress": s.progress(), "misconception_count":len(s.misconceptions), "uploaded_notes":s.rag_files}

@app.get("/api/session/{sid}/lab/evidence")
def lab_evidence(sid: str, query: str, k: int = 5):
    if not query.strip(): raise HTTPException(400, "Enter a search query.")
    s = sess(sid)
    with s.lock:
        hits = s.rag.search(query[:180], max(1, min(k, 8)))
        return {"hits": hits, "retrieval": s.rag.status()}

@app.get("/api/session/{sid}/lab/rag-status")
def lab_rag_status(sid: str):
    return sess(sid).rag.status()

class AgentRequest(BaseModel):
    question: str = ""
    mode: str = "lesson"

@app.post("/api/session/{sid}/lab/agent-teach")
def lab_agent_teach(sid: str, body: AgentRequest):
    s = sess(sid)
    if not s.has_key() or te.is_demo(s.api_key):
        raise HTTPException(400, "Connect a real Gemini API key to use the multi-agent tutor.")
    if body.mode not in ("lesson", "question"):
        raise HTTPException(400, "Mode must be 'lesson' or 'question'.")
    import multi_agent
    with s.lock:
        return multi_agent.run(s, body.question, mode=body.mode)

@app.get("/api/session/{sid}/lab/rl-explain")
def lab_rl_explain(sid: str):
    return lab.explain_rl(sess(sid))

class LabRequest(BaseModel):
    topic: str = ""
    count: int = 6
    minutes: int = 45

@app.post("/api/session/{sid}/lab/flashcards")
def lab_flashcards(sid: str, body: LabRequest):
    s = sess(sid)
    if not s.has_key() or te.is_demo(s.api_key):
        raise HTTPException(400, "Connect a real Gemini API key to generate source-grounded flashcards.")
    with s.lock:
        return lab.flashcards(s, body.topic, body.count)

@app.post("/api/session/{sid}/lab/study-plan")
def lab_study_plan(sid: str, body: LabRequest):
    return lab.plan(sess(sid), body.minutes)

@app.post("/api/session/{sid}/lab/misconceptions")
def lab_misconceptions(sid: str):
    s=sess(sid)
    if not s.misconceptions:
        return lab.coach(s)
    if not s.has_key() or te.is_demo(s.api_key):
        raise HTTPException(400, "Connect a real Gemini API key for misconception coaching.")
    with s.lock:
        return lab.coach(s)

@app.post("/api/session/{sid}/lab/concept-map")
def lab_concept_map(sid: str, body: LabRequest):
    s=sess(sid)
    if not s.has_key() or te.is_demo(s.api_key):
        raise HTTPException(400,"Connect a real Gemini key to build a concept map.")
    with s.lock:
        return lab.concept_map(s,body.topic)

@app.post("/api/session/{sid}/lab/practice")
def lab_practice(sid: str, body: LabRequest):
    s=sess(sid)
    if not s.has_key() or te.is_demo(s.api_key):
        raise HTTPException(400,"Connect a real Gemini key to generate practice questions.")
    with s.lock:
        return lab.grounded_practice(s,body.topic,body.count)

@app.get("/learning-lab")
def learning_lab_page():
    return FileResponse(os.path.join(ROOT, "static", "learning_lab.html"))


# ------------------------------------------------------------------ teacher view
def teacher_guard(x_teacher_pin: Optional[str] = Header(default=None)):
    pin = get_secret("TEACHER_PIN")
    if pin and (x_teacher_pin or "") != pin: raise HTTPException(401, "Wrong or missing teacher PIN.")

@app.get("/api/teacher/report", dependencies=[Depends(teacher_guard)])
def teacher_report():
    rep = json.load(open(REPORT)) if os.path.exists(REPORT) else {}
    events = storage.read_events()
    return {"benchmark": rep, "real_trajectories": len(load_real_trajectories(events)), "events": len(events),
            "has_candidate": os.path.exists(CAND), "tau": DEFAULT_TAU, "trained": ts.get_q()[1],
            "gate_passed": OPE_STATE["passed"] and OPE_STATE["cand_mtime"] == (os.path.getmtime(CAND) if os.path.exists(CAND) else None)}

@app.get("/api/teacher/report/class.html", dependencies=[Depends(teacher_guard)])
def teacher_insight_report(subject: str = ""):
    import report_builder as rb
    ev = [e for e in storage.read_events() if not subject or e.get("subject") == subject]
    return HTMLResponse(rb.teacher_report(ev, subject or "All subjects"))

@app.post("/api/teacher/forms/import", dependencies=[Depends(teacher_guard)])
async def forms_import(responses: UploadFile = File(...), config: UploadFile = File(...), agent: bool = False):
    """Upload the Google Form responses CSV + quiz config JSON -> zip of student + teacher reports."""
    import tempfile, zipfile, io, csv as _csv, forms_pipeline as fp
    rows = list(_csv.DictReader(io.StringIO((await responses.read()).decode("utf-8-sig"))))
    cfg = json.loads((await config.read()).decode("utf-8"))
    out = tempfile.mkdtemp(prefix="forms_reports_")
    files, skipped = fp.run(rows, cfg, out, agent)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, os.path.basename(f))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": 'attachment; filename="form_reports.zip"'})

class TrainBody(BaseModel):
    episodes: int = 30000
    seed: int = 1

@app.post("/api/teacher/train", dependencies=[Depends(teacher_guard)])
def teacher_train(body: TrainBody):
    eps = max(5000, min(100000, body.episodes))
    np.save(CAND, train_q(episodes=eps, seed=body.seed))
    OPE_STATE.update(passed=False, cand_mtime=None, source=None)
    return {"ok": True, "episodes": eps}

class OpeBody(BaseModel):
    source: str = "synthetic"      # "real" | "synthetic"
    n_synthetic: int = 400
    bootstrap: int = 150
    min_ess: float = 0.10

def _pct(a): return {k: float(np.percentile(a, p)) for k, p in (("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))}

@app.post("/api/teacher/ope", dependencies=[Depends(teacher_guard)])
def teacher_ope(body: OpeBody):
    if not os.path.exists(CAND): raise HTTPException(400, "Train a candidate policy first.")
    trajs = load_real_trajectories(storage.read_events()) if body.source == "real" else generate_synthetic_logs(max(100, min(1500, body.n_synthetic)))
    if len(trajs) < 5: raise HTTPException(400, "Not enough logged sessions yet. Use synthetic data or run more lessons.")
    r = run_full_ope_analysis(trajs, np.load(CAND), [0.5, 0.3, 0.2, 0.15, 0.1, DEFAULT_TAU],
                              alpha_ess=body.min_ess, bootstrap_samples=max(50, min(500, body.bootstrap)))
    OPE_STATE.update(passed=bool(r["passed_gate"]), cand_mtime=os.path.getmtime(CAND), source=body.source)
    return {"n": r["n_trajectories"], "source": body.source, "baseline": r["behavior_baseline"], "dr": r["best_dr"],
            "ci_dr": list(r["ci_dr"]), "ci_pdwis": list(r["ci_pdwis"]), "pdwis": None if r["best_pdwis"] != r["best_pdwis"] else r["best_pdwis"],
            "best_tau": r["best_tau"], "feasible": r["feasible_tau_found"], "passed": bool(r["passed_gate"]),
            "var_reduction": r["var_reduction"], "sweep": r["sweep_df"].round(4).to_dict("records"),
            "dist": {"pdis": _pct(r["pdwis_boots"]), "dr": _pct(r["dr_boots"])}}

class LinearTrainBody(BaseModel):
    episodes: int = 15000
    seed: int = 1

@app.post("/api/teacher/train_linear", dependencies=[Depends(teacher_guard)])
def teacher_train_linear(body: LinearTrainBody):
    import rl_bandit as rb
    eps = max(3000, min(60000, body.episodes))
    W = rb.train_linear_q(episodes=eps, seed=body.seed)
    rb.save(W)
    bench = rb.evaluate_linear(W, n=1000, seed=body.seed)
    return {"ok": True, "episodes": eps, "benchmark": bench}

@app.get("/api/teacher/linear_report", dependencies=[Depends(teacher_guard)])
def teacher_linear_report():
    import rl_bandit as rb
    rep = json.load(open(rb.REPORT_PATH)) if os.path.exists(rb.REPORT_PATH) else {}
    return {"linear_available": ts.get_linear_w() is not None, "linear_benchmark": rep}

@app.post("/api/teacher/promote", dependencies=[Depends(teacher_guard)])
def teacher_promote():
    if not os.path.exists(CAND): raise HTTPException(400, "No candidate policy.")
    if not (OPE_STATE["passed"] and OPE_STATE["source"] == "real" and OPE_STATE["cand_mtime"] == os.path.getmtime(CAND)):
        raise HTTPException(409, "A real-data safety check has not passed for this candidate. Synthetic evaluation cannot authorize promotion.")
    if os.path.exists(LIVE): shutil.copy(LIVE, LIVE.replace(".npy", "_backup.npy"))
    shutil.copy(CAND, LIVE)
    OPE_STATE.update(passed=False, cand_mtime=None, source=None)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
