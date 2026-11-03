# Offline verification report

- Python syntax compilation: passed for all supplied `.py` files.
- Eight automated tests: passed.
- BM25 reference ranking and `[S#]` source IDs: passed.
- Hybrid RRF and cross-encoder order with deterministic *mock* models: passed.
- BM25 fallback after simulated embedding failure: passed.
- Multi-agent pipeline including planner, retrieval, teaching and critic with mock LLM outputs: passed.
- Rejects fabricated source IDs and critic-rejected teaching drafts: passed.
- FastAPI agent toggle and Learning Lab route: passed.
- Illustrative 4-question offline BM25 evaluation: Hit@3 1.0 and MRR@3 1.0.

## What was NOT verified

- Real pretrained embedding or reranking weights were not installed in the test environment. The actual dense retrieval and cross-encoder inference path requires `pip install -r requirements-ai.txt` and possible first-run model downloads.
- Live Gemini API calls, model costs/latency, and end-to-end teaching accuracy were not tested because no key was used.
- The sample evaluation queries are small and designed for demonstration. They are not enough to establish a statistically meaningful performance improvement.
- No deployment security or privacy audit was conducted.
