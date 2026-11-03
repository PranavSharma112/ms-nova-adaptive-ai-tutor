# Ms. Nova — AI Teacher v2 (trained RL policy + Gemini)

A chat tutor. A **trained** RL policy silently decides the next teaching move
(question difficulty 1-5, or a worked example); Gemini turns that into lessons,
questions, grading and feedback. Students never see the machinery.

## Run
```bash
pip install -r requirements.txt
python train_policy.py            # trains models/q_table.npy (~30 s) and prints a benchmark
streamlit run app.py              # paste a Gemini key in the sidebar, or type  demo  (offline, maths only)
python smoke_test.py              # tests (no API key needed)
```
Model names: set `NOVA_MODEL=<name>` to override; otherwise a fallback list in `teacher_engine.py` is tried in order.

## RL (what changed)
| Before | Now |
|---|---|
| Q-table was random noise | Tabular Q-learning trained on a **randomised BKT simulator** (`rl_engine.train_q`) |
| App used ad-hoc tiers; sim used BKT | App runs the **same online BKT + 125-state encoding** as training (`BKTTracker`) |
| argmax(softmax) + hard rules | **Softmax sampling** (τ=0.08) with a **state-based safety mask** (≤1 step change, no harder after a miss) |
| Reward off-by-one | Reward = learning gain + correctness − mismatch to target 1+4·p_L − frustration |
| Actions: difficulty only | + **worked example** action; spaced **review** of weakest earlier topic (rule-based) |
| Speed = hard gate | Hints and slow answers weaken BKT evidence instead |
| OPE unused | Every decision is logged with its **exact propensity**; `pages/2_Policy_Check.py` runs PDIS + DR + bootstrap and only promotes a candidate if the CI lower bound beats observed behaviour |

Simulator benchmark (2000 simulated students, `models/train_report.json`): the deployed policy
reaches slightly higher mastery gain than staircase/fixed/random with far fewer long wrong streaks
than staircase — **but this is inside a simulator whose learning model (students learn fastest when
P(correct)≈0.7) is an assumption.** Treat it as a sanity check; the real test is the OPE gate on real logs.
Caveats: worked examples are capped at 2 per topic in the app (so logged propensities reflect the capped
distribution); reviews are logged with `action=None` and excluded from OPE.

## GenAI (what changed)
- New `google-genai` SDK; **schema-enforced JSON** (pydantic) with validate + retry
- Difficulty **rubric**; questions avoid repeats; 3-level **hint ladder** per question
- **Answer-key verification**: independent re-solve + exact math check (sympy); regenerate on mismatch
- **Injection-safe grading**: student text fenced as untrusted; exact check overrides the LLM on numeric/algebra answers
- **Error tagging** (misconception / slip / misread…) → next question targets it; Socratic follow-up on wrong answers
- **RAG** over uploaded notes/PDFs with `[S1]` citations; **long-term memory** (interests, strengths, misconceptions) in `data/profiles.json`
- **Calibrated question bank** (`question_bank.py`): reuses items and learns real difficulty from responses
- **Handwritten-work photo** grading, **streaming** lessons/chat, LaTeX maths, optional read-aloud, session **recap**
- Optional **prefetch** of likely next questions (uses more API calls)

## Files
`rl_engine.py` sim + BKT + policy + training · `ope_engine.py` PDIS/DR/gate · `teacher_engine.py` Gemini ·
`app.py` UI · `pages/2_Policy_Check.py` teacher view · `storage.py` logs+memory · `rag.py` · `question_bank.py`
