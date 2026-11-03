import json, os, re, time, uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import question_bank as qbank
import storage
from rag import RagIndex, format_context
from rl_engine import (BKTTracker, DEFAULT_TAU, MASTERY_THRESHOLD, NUM_ACTIONS, WORKED_EXAMPLE,
                       action_mask, choose_action, compute_reward, discretize_mastery, policy_probs, train_q)
import teacher_engine as te

st.set_page_config(page_title="Ms. Nova — Your AI Teacher", page_icon="👩‍🏫", layout="centered")

MASTERY_LABELS = {0: "just starting out", 1: "building the basics", 2: "developing solid footing",
                  3: "showing a strong grasp", 4: "very close to mastery"}
TREND_LABELS = {0: "had a rough last few questions", 1: "leaning toward mistakes recently", 2: "a mixed recent run",
                3: "leaning toward correct answers recently", 4: "on a strong roll"}
SLOW_SECONDS = {1: 40, 2: 50, 3: 70, 4: 90, 5: 120}     # slower than this (and correct) => weaker evidence
MIN_QUESTIONS_PER_TOPIC = 3
MAX_WORKED_PER_TOPIC = 2
REVIEW_EVERY = 5                                          # spaced-review cadence (questions)
Q_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "q_table.npy")


# ---------------------------------------------------------------- resources
@st.cache_resource
def load_q_table():
    if os.path.exists(Q_PATH):
        return np.load(Q_PATH), True
    return train_q(episodes=8000, seed=0), False          # quick fallback; run train_policy.py for the real one

@st.cache_resource
def get_executor():
    return ThreadPoolExecutor(max_workers=3)

Q_TABLE, Q_IS_TRAINED = load_q_table()


# ---------------------------------------------------------------- state
def init_state():
    d = {"stage": "greet", "subject": None, "chat": [], "topics": None, "topic_idx": 0,
         "trackers": {}, "topic_stats": {}, "worked_count": {}, "since_review": 0,
         "current_item": None, "current_ctx": None, "hint_level": 0, "awaiting_answer": False,
         "question_started_at": None, "recent_questions": [], "seen_ids": [], "misconceptions": [],
         "focus": "", "just_showed_example": False, "prefetch": {}, "spoken": 0, "saved": False,
         "session_id": uuid.uuid4().hex[:10], "last_decision": None, "student_name": "",
         "profile": storage.empty_profile(), "rag": RagIndex(), "rag_files": []}
    for k, v in d.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()
S = st.session_state

def say(role, content): S.chat.append({"role": role, "content": content})
def topic_name(i=None):
    i = S.topic_idx if i is None else i
    return S.topics[min(i, len(S.topics) - 1)] if S.topics else S.subject
def tracker(i=None) -> BKTTracker:
    i = S.topic_idx if i is None else i
    if i not in S.trackers: S.trackers[i] = BKTTracker()
    return S.trackers[i]
def stats(i):
    return S.topic_stats.setdefault(i, {"n": 0, "correct": 0, "hints": 0})
def mastery_label(tr): return MASTERY_LABELS[discretize_mastery(tr.p_L)]
def trend_label(tr):
    from rl_engine import streak_index
    return TREND_LABELS[streak_index(tr.history)]
def rag_context(topic):
    hits = S.rag.retrieve(f"{S.subject} {topic}", k=3) if len(S.rag) else []
    return format_context(hits), hits


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    with st.expander("⚙️ Quick setup", expanded=not S.get("_key_seen")):
        def _get_key():
            try:
                return st.secrets["GEMINI_API_KEY"]        # .streamlit/secrets.toml
            except Exception:
                return os.environ.get("GEMINI_API_KEY", "")  # or an environment variable

        gemini_api_key = _get_key()
        if not gemini_api_key:                               # only show the box if no key is configured
            gemini_api_key = st.text_input("Gemini API key", type="password",
                                           help="Free at aistudio.google.com/apikey. Type 'demo' for offline demo mode.")
        S.student_name = st.text_input("Your name (optional, for memory)", value=S.student_name)
        verify = st.toggle("Double-check answer keys", value=True, help="Extra AI call per question; catches wrong keys.")
        prefetch = st.toggle("Speed mode (prefetch)", value=False, help="Pre-generates likely next questions in the background. Uses more API calls.")
        read_aloud = st.toggle("🔊 Read Ms. Nova aloud", value=False)
        show_debug = st.toggle("Teacher view (policy details)", value=False)
    if gemini_api_key: S._key_seen = True

    with st.expander("📚 My study material (optional)"):
        ups = st.file_uploader("Upload notes / textbook (txt, md, pdf)", type=["txt", "md", "pdf"], accept_multiple_files=True)
        for up in ups or []:
            if up.name not in S.rag_files:
                try:
                    n = S.rag.add_file(up.name, up.getvalue()); S.rag_files.append(up.name)
                    st.success(f"Added {up.name} ({n} passages)")
                except Exception as e:
                    st.error(str(e))
        if S.rag_files: st.caption("Using: " + ", ".join(S.rag_files))

    if S.subject and S.topics:
        st.divider(); st.header("📈 Progress")
        for i, t in enumerate(S.topics):
            tr = S.trackers.get(i)
            p = tr.p_L if tr else 0.0
            mark = "✅ " if (tr and tr.p_L >= MASTERY_THRESHOLD and stats(i)["n"] >= MIN_QUESTIONS_PER_TOPIC) else ("👉 " if i == S.topic_idx else "")
            st.progress(min(1.0, p), text=f"{mark}{t}")
        n_all = sum(s["n"] for s in S.topic_stats.values()); c_all = sum(s["correct"] for s in S.topic_stats.values())
        st.caption(f"{c_all}/{n_all} correct so far" if n_all else "No questions answered yet")

    if S.subject:
        c1, c2 = st.columns(2)
        if c1.button("📝 Recap"):
            ts = {S.topics[i]: {**s, "mastery": S.trackers[i].p_L} for i, s in S.topic_stats.items() if i in S.trackers}
            if ts:
                with st.spinner("..."): say("teacher", te.generate_recap(ts, S.misconceptions, gemini_api_key))
                st.rerun()
        if c2.button("💾 Save"):
            notes = "\n".join(f"{S.topics[i]}: {s['correct']}/{s['n']} correct, {s['hints']} hints" for i, s in S.topic_stats.items())
            notes += "\nMistakes: " + "; ".join(S.misconceptions[-8:]) + "\nStudent said: " + " | ".join(t["content"] for t in S.chat if t["role"] == "student")[-600:]
            try:
                S.profile = te.summarize_profile(S.profile, notes, gemini_api_key)
                storage.save_profile(S.student_name, S.profile); S.saved = True
                st.success("Saved — I'll remember this next time!")
            except Exception as e:
                st.error(f"Couldn't save memory: {e}")
    if st.button("🔄 Restart"):
        for k in list(S.keys()):
            if k not in ("_key_seen",): S.pop(k, None)
        st.rerun()

    if show_debug:
        st.divider(); st.subheader("🔧 Policy details")
        st.caption(("Trained policy loaded" if Q_IS_TRAINED else "⚠️ models/q_table.npy missing — using a quick 8k-episode fallback. Run train_policy.py.")
                   + f" · τ={DEFAULT_TAU}")
        if S.last_decision: st.json(S.last_decision)

st.title("👩‍🏫 Ms. Nova — Your AI Teacher")
st.caption("Let's learn something together — one bite-sized topic at a time.")

if S.stage == "greet" and not S.chat:
    say("teacher", "Hi there! 😊 I'm Ms. Nova. What subject would you like to study together today?")
    S.stage = "need_subject"

for turn in S.chat:
    with st.chat_message(turn["role"], avatar="👩‍🏫" if turn["role"] == "teacher" else "🙂"):
        st.markdown(turn["content"])

if read_aloud and S.chat and S.chat[-1]["role"] == "teacher" and S.spoken != len(S.chat):
    S.spoken = len(S.chat)
    plain = re.sub(r"[*_`#$\\]|\[S\d+\]", "", S.chat[-1]["content"])
    components.html(f"<script>speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance({json.dumps(plain)}));</script>", height=0)


# ---------------------------------------------------------------- content helpers
def profile_text(): return storage.profile_blurb(S.profile)

def stream_and_say(gen, prefix=""):
    with st.chat_message("teacher", avatar="👩‍🏫"):
        if prefix: st.markdown(prefix)
        text = st.write_stream(gen)
    say("teacher", (prefix + "\n\n" if prefix else "") + text)

def teach_current_topic():
    topic, tr = topic_name(), tracker()
    ctx, hits = rag_context(topic)
    stream_and_say(te.stream_lesson(topic, S.subject, mastery_label(tr), gemini_api_key, profile_text(), ctx),
                   prefix=f"📘 **{topic}**")
    if hits: S.chat[-1]["content"] += "\n\n_Sources: " + "; ".join(f"[S{i}] {src}" for i, src, _ in hits) + "_"

def make_item(topic_idx, difficulty):
    topic, tr = topic_name(topic_idx), tracker(topic_idx)
    ctx, _ = rag_context(topic)
    return dict(difficulty=difficulty, subject=S.subject, topic=topic, mastery_label=mastery_label(tr),
                trend_label=trend_label(tr), api_key=gemini_api_key, recent_questions=list(S.recent_questions),
                profile_text=profile_text(), context=ctx, focus=S.focus, verify=verify)

def start_prefetch(topic_idx, d):
    if not prefetch or te.is_demo(gemini_api_key): return
    S.prefetch = {}
    for dd in {max(1, d - 1), d, min(5, d + 1)}:
        S.prefetch[(topic_idx, dd)] = get_executor().submit(te.generate_question_item, **make_item(topic_idx, dd))

def get_item(topic_idx, d):
    topic = topic_name(topic_idx)
    fut = S.prefetch.pop((topic_idx, d), None)
    if fut is not None:
        try: return fut.result(timeout=90), "prefetch"
        except Exception: pass
    if not S.focus and np.random.rand() < 0.5:                       # reuse calibrated bank item half the time
        it = qbank.pick(S.subject, topic, d, S.seen_ids)
        if it: return it, "bank"
    return te.generate_question_item(**make_item(topic_idx, d)), "generated"

def pick_review_topic():
    if S.topic_idx == 0 or S.since_review < REVIEW_EVERY: return None
    cands = [i for i in range(S.topic_idx) if i in S.trackers]
    if not cands: return None
    # forgetting: prefer the earlier topic with the lowest current mastery
    return min(cands, key=lambda i: S.trackers[i].p_L)

def ask_question():
    """Decide the next teaching move with the trained policy, then act on it."""
    rev = pick_review_topic()
    if rev is not None:
        tr = tracker(rev); d = int(min(5, max(1, tr.last_difficulty + 1)))
        S.since_review = 0
        return serve_question(rev, d, kind="review", decision=None)

    tr, ti = tracker(), S.topic_idx
    state = tr.state
    disallow = []
    if S.just_showed_example or S.worked_count.get(ti, 0) >= MAX_WORKED_PER_TOPIC: disallow.append(WORKED_EXAMPLE)
    a, prob = choose_action(Q_TABLE, state, tau=DEFAULT_TAU, disallow=disallow)
    S.last_decision = {"state": int(state), "action": int(a), "propensity": round(prob, 3), "p_L": round(tr.p_L, 3),
                       "mask": [bool(x) for x in action_mask(state)], "disallowed": disallow}
    S.just_showed_example = False
    decision = {"state": int(state), "action": int(a), "propensity": float(prob), "p_before": tr.p_L}

    if a == WORKED_EXAMPLE:
        weakness = S.misconceptions[-1] if S.misconceptions else ""
        ctx, _ = rag_context(topic_name())
        d = tr.last_difficulty + 1
        stream_and_say(te.stream_worked_example(topic_name(), S.subject, d, weakness, gemini_api_key, profile_text(), ctx),
                       prefix="🧩 **Let me show you one worked out first**")
        hist = list(tr.history); pb, pa = tr.worked_example()
        r = compute_reward(pb, pa, d, None, hist, is_worked_example=True)
        storage.log_event({"session_id": S.session_id, "subject": S.subject, "topic": topic_name(), "kind": "worked_example",
                           **{k: decision[k] for k in ("state", "action", "propensity")}, "tau": DEFAULT_TAU,
                           "reward": r, "next_state": tr.state, "p_before": pb, "p_after": pa})
        S.worked_count[ti] = S.worked_count.get(ti, 0) + 1
        S.just_showed_example = True
        S.stage = "question_pending"; st.rerun()
        return
    serve_question(ti, a + 1, kind="question", decision=decision)

def serve_question(topic_idx, d, kind, decision):
    with st.spinner("Ms. Nova is preparing your next question..."):
        item, source = get_item(topic_idx, d)
    S.focus = ""
    iid = item.get("bank_id") or qbank.add(item, S.subject, topic_name(topic_idx), d)
    S.seen_ids.append(iid); S.recent_questions.append(item["question"])
    S.current_item = item
    S.current_ctx = {"topic_idx": topic_idx, "difficulty": d, "kind": kind, "decision": decision, "iid": iid, "source": source}
    S.hint_level, S.awaiting_answer, S.question_started_at = 0, True, time.time()
    if kind != "review": S.since_review += 1
    lead = "🔁 *Quick review from earlier —* " if kind == "review" else ""
    body = f"{lead}{item.get('intro', '')}\n\n**Question:** {item['question']}"
    if item.get("choices"): body += "\n\n" + "\n".join(f"- {c}" for c in item["choices"])
    if item.get("verified") is False: body += "\n\n_(Heads-up: I couldn't fully double-check this one — tell me if something looks off!)_"
    say("teacher", body)
    if kind != "review": start_prefetch(topic_idx, d)
    S.stage = "quiz"; st.rerun()


# ---------------------------------------------------------------- grading + RL update
def handle_answer(text, image=None):
    item, ctx = S.current_item, S.current_ctx
    elapsed = time.time() - (S.question_started_at or time.time())
    hints = S.hint_level
    with st.spinner("Checking your answer..."):
        g = te.check_answer(item, text, gemini_api_key, image=image, hints_used=hints)
    correct = int(g["correct"])
    say("teacher", f"{'✅' if correct else '❌'} {g['feedback']}" + (f"\n\n🤔 {g['followup_question']}" if g.get("followup_question") else ""))

    ti, d = ctx["topic_idx"], ctx["difficulty"]
    tr = tracker(ti); hist = list(tr.history)
    slow = correct and elapsed > SLOW_SECONDS.get(d, 60)
    pb, pa = tr.observe(d, correct, guess_bonus=0.15 * hints + (0.10 if slow else 0.0))
    reward = compute_reward(pb, pa, d, correct, hist)
    s = stats(ti); s["n"] += 1; s["correct"] += correct; s["hints"] += hints
    qbank.record(ctx["iid"], bool(correct), hints)
    if not correct and g.get("error_type") in ("conceptual_misconception", "misread_question") and g.get("misconception_note"):
        S.misconceptions.append(g["misconception_note"]); S.focus = g["misconception_note"]

    dec = ctx["decision"] or {}
    storage.log_event({"session_id": S.session_id, "subject": S.subject, "topic": topic_name(ti), "topic_idx": ti,
                       "kind": ctx["kind"], "state": dec.get("state"), "action": dec.get("action"),
                       "propensity": dec.get("propensity"), "tau": DEFAULT_TAU, "reward": reward, "next_state": tr.state,
                       "difficulty": d, "correct": correct, "elapsed": round(elapsed, 1), "hints_used": hints,
                       "error_type": g.get("error_type"), "skill_tag": item.get("skill_tag"), "verified": item.get("verified"),
                       "item_id": ctx["iid"], "question_source": ctx["source"], "p_before": pb, "p_after": pa})

    S.current_item, S.awaiting_answer = None, False
    S.stage = "question_pending"
    if ctx["kind"] != "review" and tr.p_L >= MASTERY_THRESHOLD and s["n"] >= MIN_QUESTIONS_PER_TOPIC:
        fin = topic_name(ti)
        if S.topic_idx < len(S.topics) - 1:
            S.topic_idx += 1; S.since_review = 0
            say("teacher", f"🚀 You've really got **{fin}** — let's move on to the next topic!")
            S.stage = "lesson_pending"
        else:
            say("teacher", f"🎓 You've mastered every topic in this {S.subject} course. Amazing work!")
            ts = {S.topics[i]: {**x, "mastery": S.trackers[i].p_L} for i, x in S.topic_stats.items() if i in S.trackers}
            say("teacher", te.generate_recap(ts, S.misconceptions, gemini_api_key))
            S.stage = "done"
    st.rerun()


# ---------------------------------------------------------------- stage machine
if S.stage == "need_subject" and S.subject and gemini_api_key: S.stage = "planning"
if S.stage == "planning" and gemini_api_key:
    if S.student_name: S.profile = storage.load_profile(S.student_name)
    try:
        with st.spinner(f"Ms. Nova is planning your {S.subject} lessons..."):
            ctx, _ = rag_context(S.subject)
            S.topics = te.generate_topics(S.subject, gemini_api_key, ctx)
    except Exception as e:
        st.error(f"Couldn't plan the course: {e}"); st.stop()
    if S.profile.get("sessions"): say("teacher", f"Welcome back{', ' + S.student_name if S.student_name else ''}! 🌟 " + (S.profile.get("summary") or ""))
    S.stage = "lesson_pending"; st.rerun()
if S.stage == "lesson_pending" and gemini_api_key:
    try: teach_current_topic()
    except Exception as e: st.error(f"Couldn't prepare the lesson: {e}"); st.stop()
    S.stage = "question_pending"; st.rerun()
if S.stage == "question_pending" and gemini_api_key and S.topics:
    try: ask_question()
    except Exception as e: st.error(f"Couldn't prepare a question: {e}"); st.stop()


# ---------------------------------------------------------------- quiz controls + input
if S.stage == "quiz" and S.awaiting_answer:
    c1, c2 = st.columns([1, 3])
    lvl = S.hint_level
    if c1.button(f"💡 Hint ({lvl}/3)", disabled=lvl >= 3):
        say("teacher", f"💡 **Hint {lvl + 1}:** {S.current_item['hints'][lvl]}")
        S.hint_level += 1; st.rerun()
    with c2.expander("📷 Submit handwritten work instead"):
        photo = st.file_uploader("Photo of your work", type=["png", "jpg", "jpeg", "webp"], key=f"photo_{len(S.chat)}")
        if photo and st.button("Check my photo"):
            say("student", "📷 *(sent a photo of my work)*")
            handle_answer("", image=(photo.getvalue(), photo.type or "image/jpeg"))

placeholder = ("Which subject would you like to study today?" if S.stage == "need_subject"
               else "Type your answer, or ask me anything...")
user_msg = st.chat_input(placeholder)
if user_msg:
    say("student", user_msg)
    if S.stage == "need_subject":
        S.subject = user_msg.strip()
        say("teacher", f"{S.subject} — great pick! 🚀" + ("" if gemini_api_key else
            " Pop a free Gemini key into the ⚙️ setup box in the sidebar (or type `demo`) and we'll get going."))
        st.rerun()
    elif S.stage == "quiz" and S.awaiting_answer:
        # a question mark + no digits looks like a side question, not an answer
        looks_like_question = user_msg.strip().endswith("?") and len(user_msg.split()) > 3 and S.current_item.get("question_type") != "explain"
        if looks_like_question:
            with st.chat_message("teacher", avatar="👩‍🏫"):
                txt = st.write_stream(te.stream_chat_reply(S.chat[:-1], user_msg, S.subject or "the topic", mastery_label(tracker()),
                                                           S.current_item["question"], gemini_api_key, profile_text(), rag_context(topic_name())[0]))
            say("teacher", txt); st.rerun()
        else:
            handle_answer(user_msg)
    else:
        with st.chat_message("teacher", avatar="👩‍🏫"):
            txt = st.write_stream(te.stream_chat_reply(S.chat[:-1], user_msg, S.subject or "the topic",
                                                       mastery_label(tracker()) if S.topics else "just starting out",
                                                       "", gemini_api_key, profile_text(), rag_context(topic_name() if S.topics else "")[0]))
        say("teacher", txt); st.rerun()
