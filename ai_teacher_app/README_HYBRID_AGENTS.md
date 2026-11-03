# Ms. Nova — Hybrid RAG and Multi-agent Teaching Upgrade

This extends the prior Learning Lab without removing the original chatbot, RL
policy, uploaded-material indexing, flashcards, student reports or Streamlit app.

## What changed

1. **BM25 + sentence embeddings + reciprocal rank fusion (RRF)** in `rag.py`.
   Dense search uses local `sentence-transformers/all-MiniLM-L6-v2`; results
   are fused with lexical BM25 candidates. If dense embeddings are unavailable,
   the app still uses BM25. No hosted vector database is required.
2. **Cross-encoder reranking** of candidates with
   `cross-encoder/ms-marco-MiniLM-L-6-v2` when installed/enabled. If the
   cross-encoder fails, fused retrieval remains available.
3. **Source ID provenance** is preserved as `[S<n>]` with exact PDF page or
   uploaded filename. Existing note-grounded features use the new retriever
   without API changes.
4. **Multi-agent teaching flow**, implemented explicitly in `multi_agent.py`:
   Planner (Gemini structured output) → Retriever (local hybrid RAG) → Teacher
   (Gemini structured output with citations) → Critic (separate Gemini call).
   Answers that fail source attribution or the critic's review are withheld.
5. **UI controls:** enable Multi-agent teaching in the main classroom sidebar,
   or use the independent **Multi-agent teacher** and **Hybrid search status**
   cards in the AI Learning Lab. The trace shows which agents ran and the
   retrieval mode, without exposing API keys.

## Install on Windows (PowerShell)

```powershell
cd ai_teacher_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ai.txt
$env:GEMINI_API_KEY = 'YOUR_KEY_HERE'  # or enter your key in the web UI
python server.py
```

Then visit http://127.0.0.1:8000, create a session, upload a `.txt`, `.md`
or selectable-text `.pdf` file, and open the AI Learning Lab. The first time
local models are used they **may download pretrained weights**, which needs an
internet connection. Subsequent local inference works offline if weights are
cached; Gemini calls still need network access.

If you only install `requirements.txt`, keyword BM25 works but **semantic
retrieval and cross-encoder reranking are not active**. Check the status card
or `/api/session/{session_id}/lab/rag-status` to see the actual retrieval mode
used by the most recent search. Do not call the result hybrid if the status
says `bm25`.

## Runtime environment variables

| Variable | Default | Use |
| --- | --- | --- |
| `NOVA_RAG_EMBEDDINGS` | `auto` | `off` disables dense retrieval; `auto` enables it if installed |
| `NOVA_RAG_RERANK` | `auto` | `off` disables the cross-encoder |
| `NOVA_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Change embedding model |
| `NOVA_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Change reranker model |

To explicitly disable optional model loading (e.g. for offline lab tests):

```powershell
$env:NOVA_RAG_EMBEDDINGS = 'off'
$env:NOVA_RAG_RERANK = 'off'
```

## HTTP API

- `GET /api/session/{sid}/lab/rag-status`: feature activation and last search.
- `GET /api/session/{sid}/lab/evidence?query=...`: retrieved texts and scores.
- `POST /api/session/{sid}/lab/agent-teach`: JSON `{"question":"...","mode":"lesson"}`.
- `POST /api/session/{sid}/settings`: JSON `{"agent_mode":true}` enables
  multi-agent lessons and topic questions in the main classroom. Existing
  single-model behavior remains the default; with no uploaded notes the
  classroom automatically uses the old tutor behavior.

The pipeline requires a real Gemini API key, not `demo`. In grounded mode,
no relevant source = no generated answer. The model-based critic is a
quality-control step, **not a guarantee of factual correctness**. Only source
identities are checked deterministically; claims still need instructor review.

## Testing and limitations

Run `python -m pytest tests -q`. Unit tests replace expensive embedding,
reranking and Gemini model calls with mocks so they work offline. Optional
model downloads and full live Gemini requests have **not** been validated by
these tests.

- Dense models use RAM and may take time on first launch.
- Scanned PDFs are not OCRed; provide text-based PDFs or converted text.
- Retrieved notes are **untrusted content**; the prompts instruct each agent
  not to obey commands inside them, but prompt-injection risk remains.
- Data is in server memory/local JSON. This is an academic demonstration,
  not a production student data platform; add authentication, access controls,
  retention policies, monitoring and a thorough privacy review before use.
- Three Gemini calls per grounded explanation are slower and cost more than
  single-agent chat; benchmark both modes for groundedness, errors, latency,
  and learning usefulness.

## Retrieval comparison for your report

Run `python evaluate_retrieval.py --dataset examples/sample_rag_eval.json --k 3`
to compare keyword-only and currently enabled retrieval using Hit@k and MRR@k.
The included dataset is **illustrative**, not evidence of better performance
on real student materials. Replace it with a manually labelled evaluation set
from your curriculum for the project report. When the optional models are not
installed, both comparisons correctly report BM25 mode.
