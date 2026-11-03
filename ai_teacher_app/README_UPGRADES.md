> **Newer version:** See `README_HYBRID_AGENTS.md` for the added hybrid semantic RAG and multi-agent tutor. The BM25-only limitations described below applied to the previous version.

# Ms. Nova — Learning Lab upgrade

## New additions

- **Concept maps and practice quizzes**: generate study concepts, prerequisite links, and question-answer-explanation sets using uploaded notes.
- **Flashcards with source validation**: generate 3–12 Gemini flashcards from retrieved uploaded notes only. Reject cards whose source cannot be matched to a retrieved document.
- **Personalized study plan**: allocate 15–180 minutes across topics using tracked BKT mastery. Deterministic; works without an API key.
- **Misconception coach**: derive explanations, targeted practice, and checking questions from actual logged misconceptions.
- **Source explorer**: inspect BM25 excerpts and filenames used for note grounding.
- **Explainable RL**: inspect the logged state, chosen action, and action propensity. The interpretation is descriptive, not a causal attribution to Q-values.
- **Learning analytics**: inspect topic-level mastery and answered counts.
- **Safety fix**: promoting a candidate RL policy now requires **real-data OPE**. Synthetic evaluation remains available for demonstration but cannot authorize deployment.
- **Packaging fix**: added ReportLab to dependencies for PDF exports.

## Run

1. `python -m venv .venv`
2. Activate the virtual environment (Windows: `.venv\\Scripts\\activate`; Unix: `source .venv/bin/activate`).
3. `pip install -r requirements.txt`
4. Set `GEMINI_API_KEY` as an environment variable, or enter a key in the classroom UI.
5. `python server.py`, then open `http://127.0.0.1:8000`.
6. Start a classroom session; upload notes and answer some questions; click **Open AI Learning Lab** in the sidebar.

## Important limitations

- The RAG search remains BM25 (keyword retrieval), not hybrid vector search. This upgrade exposes evidence; it does not pretend to deliver semantic embedding retrieval.
- Gemini-generated explanations/flashcards require review for accuracy and do not provide formal proof of faithfulness. Flashcards validate source identity but not every claim.
- No real student learning gain improvement has been experimentally established. No DQN or PPO model is included in this upgrade.
- Sessions and uploaded notes are kept in server memory. This is a local academic demo, not a production multiuser deployment.
- For sharing, never include `.streamlit/secrets.toml`, real user profiles, or API keys.
