# Ms. Nova web front end

Copy these into your existing `ai_teacher_app/` folder (nothing else is overwritten):

    server.py            FastAPI server (API + serves the page)
    tutor_service.py     the tutor logic, framework-free (same behaviour as app.py)
    static/index.html    the dark-theme front end
    requirements.txt     adds fastapi, uvicorn, python-multipart

Run:

    pip install -r requirements.txt
    python server.py            # opens http://127.0.0.1:8000

API key (never sent to the browser), first match wins:
  1. environment variable GEMINI_API_KEY
  2. .streamlit/secrets.toml  ->  GEMINI_API_KEY = "..."
  3. typed into the Connect dialog (kept in server memory only, not written to disk)
Type `demo` in that dialog for offline maths-only demo mode.

Optional: add TEACHER_PIN = "1234" to secrets.toml (or env) to lock the Teacher view.
Optional: PORT=9000 python server.py   ·   NO_BROWSER=1 to skip auto-opening the browser.

The Streamlit app (streamlit run app.py) still works; both share the same models, logs and memory.
