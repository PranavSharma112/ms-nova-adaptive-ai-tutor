"""
tutor_service.py - framework-free tutor session (same logic as the Streamlit app.py,
but every action is a generator of JSON-able events so any front end can stream them).

Event types yielded:
  message{role,content,kind,meta} | stream_start{id,kind} | delta{id,text} | stream_end{id,content}
  status{text} | state{progress,quiz} | celebrate{topic} | need_key | error{message}
"""
from __future__ import annotations
import threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Generator, List, Optional

import numpy as np

import question_bank as qbank
import storage
import teacher_engine as te
from rag import RagIndex, format_context
from rl_engine import (BKTTracker, DEFAULT_TAU, MASTERY_THRESHOLD, WORKED_EXAMPLE, action_mask,
                       choose_action, compute_reward, discretize_mastery, streak_index, train_q)
import os

MASTERY_LABELS = {0: "just starting out", 1: "building the basics", 2: "developing solid footing",
                  3: "showing a strong grasp", 4: "very close to mastery"}
TREND_LABELS = {0: "had a rough last few questions", 1: "leaning toward mistakes recently", 2: "a mixed recent run",
                3: "leaning toward correct answers recently", 4: "on a strong roll"}
SLOW_SECONDS = {1: 40, 2: 50, 3: 70, 4: 90, 5: 120}
MIN_QUESTIONS_PER_TOPIC = 3
MAX_WORKED_PER_TOPIC = 2
REVIEW_EVERY = 5
Q_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "q_table.npy")
LINEAR_Q_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "linear_q.npy")

_Q = {"mtime": None, "q": None, "trained": False}
_LQ = {"mtime": None, "w": None}
_Q_LOCK = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=3)


def get_linear_w():
    """Live linear-Q weights; hot-reloads when models/linear_q.npy changes."""
    import rl_bandit as rb
    with _Q_LOCK:
        if os.path.exists(LINEAR_Q_PATH):
            m = os.path.getmtime(LINEAR_Q_PATH)
            if _LQ["mtime"] != m:
                _LQ.update(mtime=m, w=np.load(LINEAR_Q_PATH))
        return _LQ["w"]


def get_q():
    """Live Q-table; hot-reloads when models/q_table.npy changes (e.g. after a promotion)."""
    with _Q_LOCK:
        if os.path.exists(Q_PATH):
            m = os.path.getmtime(Q_PATH)
            if _Q["mtime"] != m:
                _Q.update(mtime=m, q=np.load(Q_PATH), trained=True)
        elif _Q["q"] is None:
            _Q.update(q=train_q(episodes=8000, seed=0), trained=False)
        return _Q["q"], _Q["trained"]


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "429" in msg or "quota" in low or "rate" in low and "limit" in low or "resource_exhausted" in low:
        return ("Gemini's rate limit was reached. Wait a minute and press Try again. "
                "Turning off 'Double-check answers' and 'Speed mode' in Settings uses fewer requests.")
    if "api key" in low or "api_key" in low or "permission" in low or "401" in msg or "403" in msg:
        return "Gemini rejected the API key. Check it in Settings > Connection, or type demo to use demo mode."
    if "no gemini api key" in low:
        return "No Gemini API key is connected yet. Add one in Settings > Connection, or use demo mode."
    return msg[:300]


class TutorSession:
    def __init__(self, api_key: str = "", student_name: str = ""):
        self.id = uuid.uuid4().hex[:12]
        self.created = time.time()
        self.api_key = (api_key or "").strip()
        self.student_name = student_name
        self.settings = {"verify": True, "prefetch": False, "policy": "tabular", "agent_mode": False}
        self.lock = threading.Lock()
        self.profile = storage.empty_profile()
        self.rag, self.rag_files = RagIndex(), []
        self.chat: List[Dict] = []
        self.stage = "greet"
        self.subject: Optional[str] = None
        self.topics: Optional[List[str]] = None
        self.topic_idx = 0
        self.trackers: Dict[int, BKTTracker] = {}
        self.topic_stats: Dict[int, Dict] = {}
        self.worked_count: Dict[int, int] = {}
        self.since_review = 0
        self.current_item: Optional[Dict] = None
        self.current_ctx: Optional[Dict] = None
        self.hint_level = 0
        self.socratic_dialogue: List[Dict] = []
        self.socratic_rounds = 0
        self.socratic_pending: Optional[Dict] = None  # {"item","ctx","g","correct":0,"ti","d","elapsed"}
        self.awaiting = False
        self.question_started_at: Optional[float] = None
        self.recent_questions: List[str] = []
        self.seen_ids: List[str] = []
        self.misconceptions: List[str] = []
        self.focus = ""
        self.just_showed_example = False
        self.prefetch: Dict = {}
        self.last_decision: Optional[Dict] = None
        self.log_id = self.id

    # ------------------------------------------------------------- helpers
    def has_key(self) -> bool:
        return bool(self.api_key) or bool(os.environ.get("GEMINI_API_KEY"))

    def say(self, role: str, content: str, kind: str = "text", meta: Optional[Dict] = None) -> Dict:
        m = {"role": role, "content": content, "kind": kind, "meta": meta or {}}
        self.chat.append(m)
        return {"type": "message", **m}

    def topic_name(self, i: Optional[int] = None) -> str:
        i = self.topic_idx if i is None else i
        return self.topics[min(i, len(self.topics) - 1)] if self.topics else (self.subject or "")

    def tracker(self, i: Optional[int] = None) -> BKTTracker:
        i = self.topic_idx if i is None else i
        if i not in self.trackers: self.trackers[i] = BKTTracker()
        return self.trackers[i]

    def stats(self, i: int) -> Dict:
        return self.topic_stats.setdefault(i, {"n": 0, "correct": 0, "hints": 0})

    def mastery_label(self, tr) -> str: return MASTERY_LABELS[discretize_mastery(tr.p_L)]
    def trend_label(self, tr) -> str: return TREND_LABELS[streak_index(tr.history)]
    def profile_text(self) -> str: return storage.profile_blurb(self.profile)

    def rag_context(self, topic: str):
        hits = self.rag.retrieve(f"{self.subject} {topic}", k=3) if len(self.rag) else []
        return format_context(hits), hits

    def is_mastered(self, i: int) -> bool:
        tr = self.trackers.get(i)
        return bool(tr and tr.p_L >= MASTERY_THRESHOLD and self.topic_stats.get(i, {}).get("n", 0) >= MIN_QUESTIONS_PER_TOPIC)

    # ------------------------------------------------------------- snapshots
    def progress(self) -> Dict:
        topics = []
        for i, name in enumerate(self.topics or []):
            tr, s = self.trackers.get(i), self.topic_stats.get(i, {"n": 0, "correct": 0, "hints": 0})
            topics.append({"name": name, "mastery": round(tr.p_L, 3) if tr else 0.0, "n": s["n"], "correct": s["correct"],
                           "hints": s["hints"], "mastered": self.is_mastered(i), "current": i == self.topic_idx})
        _, trained = get_q()
        return {"subject": self.subject, "topics": topics, "topic_idx": self.topic_idx, "stage": self.stage,
                "answered": sum(s["n"] for s in self.topic_stats.values()),
                "correct": sum(s["correct"] for s in self.topic_stats.values()),
                "worked": sum(self.worked_count.values()), "policy_trained": trained, "tau": DEFAULT_TAU,
                "debug": self.last_decision}

    def quiz(self) -> Dict:
        it = self.current_item or {}
        return {"awaiting": bool(self.awaiting), "hint_level": self.hint_level, "hints_max": 3,
                "qtype": it.get("question_type"), "choices": it.get("choices") or []}

    def snapshot(self) -> Dict:
        return {"session_id": self.id, "chat": self.chat, "progress": self.progress(), "quiz": self.quiz(),
                "settings": self.settings, "rag_files": self.rag_files, "student_name": self.student_name,
                "has_key": self.has_key(), "demo": te.is_demo(self.api_key), "stage": self.stage, "subject": self.subject}

    def state_event(self) -> Dict:
        return {"type": "state", "progress": self.progress(), "quiz": self.quiz(), "has_key": self.has_key(),
                "demo": te.is_demo(self.api_key), "rag_files": self.rag_files}

    # ------------------------------------------------------------- streaming helper
    def _stream(self, gen, kind: str, prefix: str = "", meta: Optional[Dict] = None):
        sid = uuid.uuid4().hex[:8]
        yield {"type": "stream_start", "id": sid, "kind": kind, "prefix": prefix}
        buf: List[str] = []
        try:
            for chunk in gen:
                buf.append(chunk)
                yield {"type": "delta", "id": sid, "text": chunk}
        except Exception as e:
            if not buf:
                yield {"type": "stream_end", "id": sid, "content": "", "aborted": True}
                raise
            yield {"type": "error", "message": friendly_error(e)}
        text = "".join(buf)
        content = (prefix + "\n\n" if prefix else "") + text
        self.chat.append({"role": "teacher", "content": content, "kind": kind, "meta": meta or {}})
        yield {"type": "stream_end", "id": sid, "content": content}
        return text

    # ------------------------------------------------------------- pipeline
    def continue_(self):
        if self.stage == "need_subject" and self.subject and self.has_key(): self.stage = "planning"
        yield from self._advance()

    def start(self):
        if not self.chat:
            yield self.say("teacher", "Hi there! 😊 I'm Ms. Nova. What would you like to study together today?", "greeting")
            self.stage = "need_subject"
        yield self.state_event()

    def _advance(self):
        for _ in range(8):
            if self.stage == "planning":
                yield {"type": "status", "text": f"Planning your {self.subject} course…"}
                if self.student_name: self.profile = storage.load_profile(self.student_name)
                ctx, _ = self.rag_context(self.subject)
                self.topics = te.generate_topics(self.subject, self.api_key, ctx)
                if self.profile.get("sessions"):
                    who = f", {self.student_name}" if self.student_name else ""
                    yield self.say("teacher", f"Welcome back{who}! 🌟 " + (self.profile.get("summary") or ""))
                self.stage = "lesson"
                yield self.state_event()
            elif self.stage == "lesson":
                yield {"type": "status", "text": "Ms. Nova is preparing the lesson…"}
                yield from self._teach()
                self.stage = "question_pending"
            elif self.stage == "question_pending":
                yield {"type": "status", "text": "Choosing your next question…"}
                yield from self._ask_question()
                if self.stage == "quiz": break
            else:
                break
        yield self.state_event()

    def _teach(self):
        topic, tr = self.topic_name(), self.tracker()
        ctx, hits = self.rag_context(topic)
        if self.settings.get("agent_mode") and len(self.rag) and not te.is_demo(self.api_key):
            import multi_agent
            result = multi_agent.run(self, topic, mode="lesson")
            text = result['answer'] if result['status'] == 'model_reviewed' else result['message']
            metadata = {"sources": [f"[S{h['id']}] {h['source']}" for h in result['evidence']],
                        "agent_status": result['status'], "agent_trace": result['trace']}
            yield from self._stream(iter([text]), "lesson", prefix=f"📘 **{topic}**", meta=metadata)
            return
        gen = te.stream_lesson(topic, self.subject, self.mastery_label(tr), self.api_key, self.profile_text(), ctx)
        yield from self._stream(gen, "lesson", prefix=f"📘 **{topic}**",
                                meta={"sources": [f"[S{i}] {src}" for i, src, _ in hits]})
        if hits:
            self.chat[-1]["meta"]["sources"] = [f"[S{i}] {src}" for i, src, _ in hits]

    # ------------------------------------------------------------- questions
    def _item_args(self, topic_idx: int, difficulty: int) -> Dict:
        topic, tr = self.topic_name(topic_idx), self.tracker(topic_idx)
        ctx, _ = self.rag_context(topic)
        return dict(difficulty=difficulty, subject=self.subject, topic=topic, mastery_label=self.mastery_label(tr),
                    trend_label=self.trend_label(tr), api_key=self.api_key, recent_questions=list(self.recent_questions),
                    profile_text=self.profile_text(), context=ctx, focus=self.focus, verify=self.settings["verify"])

    def _start_prefetch(self, topic_idx: int, d: int):
        if not self.settings["prefetch"] or te.is_demo(self.api_key): return
        self.prefetch = {}
        for dd in {max(1, d - 1), d, min(5, d + 1)}:
            self.prefetch[(topic_idx, dd)] = _executor.submit(te.generate_question_item, **self._item_args(topic_idx, dd))

    def _get_item(self, topic_idx: int, d: int):
        fut = self.prefetch.pop((topic_idx, d), None)
        if fut is not None:
            try: return fut.result(timeout=90), "prefetch"
            except Exception: pass
        if not self.focus and np.random.rand() < 0.5:
            it = qbank.pick(self.subject, self.topic_name(topic_idx), d, self.seen_ids)
            if it: return it, "bank"
        return te.generate_question_item(**self._item_args(topic_idx, d)), "generated"

    def _pick_review_topic(self) -> Optional[int]:
        if self.topic_idx == 0 or self.since_review < REVIEW_EVERY: return None
        cands = [i for i in range(self.topic_idx) if i in self.trackers]
        return min(cands, key=lambda i: self.trackers[i].p_L) if cands else None

    def _ask_question(self):
        rev = self._pick_review_topic()
        if rev is not None:
            tr = self.tracker(rev); d = int(min(5, max(1, tr.last_difficulty + 1)))
            self.since_review = 0
            yield from self._serve_question(rev, d, "review", None); return

        tr, ti = self.tracker(), self.topic_idx
        state = tr.state
        disallow = []
        if self.just_showed_example or self.worked_count.get(ti, 0) >= MAX_WORKED_PER_TOPIC: disallow.append(WORKED_EXAMPLE)
        if self.settings.get("policy") == "linear" and get_linear_w() is not None:
            import rl_bandit as rb
            a, prob = rb.choose_action(get_linear_w(), tr, tau=DEFAULT_TAU, disallow=disallow)
        else:
            Q, _ = get_q()
            a, prob = choose_action(Q, state, tau=DEFAULT_TAU, disallow=disallow)
        self.last_decision = {"state": int(state), "action": int(a), "propensity": round(prob, 3), "p_L": round(tr.p_L, 3),
                              "label": "Worked example" if a == WORKED_EXAMPLE else f"Question, level {a + 1}",
                              "mask": [bool(x) for x in action_mask(state)], "disallowed": disallow}
        self.just_showed_example = False
        decision = {"state": int(state), "action": int(a), "propensity": float(prob), "p_before": tr.p_L}

        if a == WORKED_EXAMPLE:
            weakness = self.misconceptions[-1] if self.misconceptions else ""
            ctx, _ = self.rag_context(self.topic_name())
            d = tr.last_difficulty + 1
            yield {"type": "status", "text": "Ms. Nova is preparing a worked example…"}
            gen = te.stream_worked_example(self.topic_name(), self.subject, d, weakness, self.api_key, self.profile_text(), ctx)
            yield from self._stream(gen, "worked", prefix="🧩 **Let me show you one worked out first**")
            hist = list(tr.history); pb, pa = tr.worked_example()
            r = compute_reward(pb, pa, d, None, hist, is_worked_example=True)
            storage.log_event({"session_id": self.log_id, "student": self.student_name or "", "subject": self.subject, "topic": self.topic_name(), "kind": "worked_example",
                               "state": decision["state"], "action": decision["action"], "propensity": decision["propensity"],
                               "tau": DEFAULT_TAU, "reward": r, "next_state": tr.state, "p_before": pb, "p_after": pa})
            self.worked_count[ti] = self.worked_count.get(ti, 0) + 1
            self.just_showed_example = True
            self.stage = "question_pending"
            yield self.state_event()
            return
        yield from self._serve_question(ti, a + 1, "question", decision)

    def _serve_question(self, topic_idx: int, d: int, kind: str, decision):
        yield {"type": "status", "text": "Ms. Nova is preparing your next question…"}
        item, source = self._get_item(topic_idx, d)
        self.focus = ""
        iid = item.get("bank_id") or qbank.add(item, self.subject, self.topic_name(topic_idx), d)
        self.seen_ids.append(iid); self.recent_questions.append(item["question"])
        self.current_item = item
        self.current_ctx = {"topic_idx": topic_idx, "difficulty": d, "kind": kind, "decision": decision, "iid": iid, "source": source}
        self.hint_level, self.awaiting, self.question_started_at = 0, True, time.time()
        if kind != "review": self.since_review += 1
        lead = "🔁 *Quick review from earlier —* " if kind == "review" else ""
        body = f"{lead}{item.get('intro', '')}\n\n**Question:** {item['question']}"
        if item.get("verified") is False:
            body += "\n\n_I couldn't fully double-check this one, so tell me if something looks off._"
        self.stage = "quiz"
        yield self.say("teacher", body, "question", {"choices": item.get("choices") or [], "review": kind == "review",
                                                     "skill": item.get("skill_tag"), "qtype": item.get("question_type")})
        if kind != "review": self._start_prefetch(topic_idx, d)
        yield self.state_event()

    # ------------------------------------------------------------- student input
    def handle_message(self, text: str):
        text = (text or "").strip()
        if not text: return
        yield self.say("student", text[:4000])
        if self.stage == "need_subject":
            self.subject = text[:120]
            yield self.say("teacher", f"{self.subject}, great pick! 🚀")
            if not self.has_key():
                yield {"type": "need_key"}; yield self.state_event(); return
            self.stage = "planning"
            yield from self._advance(); return
        if self.stage == "socratic":
            yield from self.continue_socratic(text); return
        if self.stage == "quiz" and self.awaiting:
            looks_like_q = (text.endswith("?") and len(text.split()) > 3 and (self.current_item or {}).get("question_type") != "explain")
            if looks_like_q:
                yield from self._chat_reply(text, (self.current_item or {}).get("question", "")); yield self.state_event()
            else:
                yield from self.grade(text)
            return
        yield from self._chat_reply(text); yield self.state_event()

    def _chat_reply(self, text: str, current_q: str = ""):
        tr = self.tracker() if self.topics else None
        label = self.mastery_label(tr) if tr else "just starting out"
        ctx = self.rag_context(self.topic_name())[0] if self.topics else ""
        hist = [t for t in self.chat[:-1] if t["role"] in ("student", "teacher")]
        if self.settings.get("agent_mode") and len(self.rag) and not te.is_demo(self.api_key):
            import multi_agent
            yield {"type": "status", "text": "Planner → hybrid retrieval → teacher → critic…"}
            result = multi_agent.run(self, text, mode="question", current_question=current_q)
            message = result['answer'] if result['status'] == 'model_reviewed' else result['message']
            yield from self._stream(iter([message]), "text",
                                    meta={"sources": [f"[S{h['id']}] {h['source']}" for h in result['evidence']],
                                          "agent_status": result['status'], "agent_trace": result['trace']})
            return
        gen = te.stream_chat_reply(hist, text, self.subject or "the topic", label, current_q, self.api_key, self.profile_text(), ctx)
        yield {"type": "status", "text": "Ms. Nova is thinking…"}
        yield from self._stream(gen, "text")

    def hint(self):
        if self.stage != "quiz" or not self.awaiting or self.hint_level >= 3:
            yield {"type": "error", "message": "No more hints for this question."}; return
        yield self.say("teacher", f"💡 **Hint {self.hint_level + 1}:** {self.current_item['hints'][self.hint_level]}", "hint")
        self.hint_level += 1
        yield self.state_event()

    def grade(self, text: str, image: Optional[tuple] = None):
        item, ctx = self.current_item, self.current_ctx
        if not item or not ctx: return
        elapsed = time.time() - (self.question_started_at or time.time())
        hints = self.hint_level
        yield {"type": "status", "text": "Checking your answer…"}
        g = te.check_answer(item, text, self.api_key, image=image, hints_used=hints)
        correct = int(g["correct"])
        fb = f"{'✅' if correct else '❌'} {g['feedback']}"
        yield self.say("teacher", fb, "correct" if correct else "wrong", {"error_type": g.get("error_type")})

        ti, d = ctx["topic_idx"], ctx["difficulty"]
        tr = self.tracker(ti); hist = list(tr.history)
        slow = correct and elapsed > SLOW_SECONDS.get(d, 60)
        pb, pa = tr.observe(d, correct, guess_bonus=0.15 * hints + (0.10 if slow else 0.0))
        reward = compute_reward(pb, pa, d, correct, hist)
        s = self.stats(ti); s["n"] += 1; s["correct"] += correct; s["hints"] += hints
        qbank.record(ctx["iid"], bool(correct), hints)
        if not correct and g.get("error_type") in ("conceptual_misconception", "misread_question") and g.get("misconception_note"):
            self.misconceptions.append(g["misconception_note"]); self.focus = g["misconception_note"]

        dec = ctx["decision"] or {}
        storage.log_event({"session_id": self.log_id, "student": self.student_name or "", "subject": self.subject, "topic": self.topic_name(ti), "topic_idx": ti,
                           "kind": ctx["kind"], "state": dec.get("state"), "action": dec.get("action"),
                           "propensity": dec.get("propensity"), "tau": DEFAULT_TAU, "reward": reward, "next_state": tr.state,
                           "difficulty": d, "correct": correct, "elapsed": round(elapsed, 1), "hints_used": hints,
                           "error_type": g.get("error_type"), "skill_tag": item.get("skill_tag"), "verified": item.get("verified"),
                           "item_id": ctx["iid"], "question_source": ctx["source"], "p_before": pb, "p_after": pa})

        if not correct and g.get("followup_question") and item.get("question_type") != "explain":
            self.socratic_dialogue = [{"role": "teacher", "content": g["followup_question"]}]
            self.socratic_rounds = 0
            self.socratic_pending = {"ti": ti, "item": item}
            self.stage = "socratic"
            yield self.say("teacher", f"🤔 {g['followup_question']}", "socratic")
            yield self.state_event(); return

        yield from self._finish_question(ctx, ti)

    def _finish_question(self, ctx: Dict, ti: int):
        self.current_item, self.awaiting = None, False
        self.stage = "question_pending"
        if ctx["kind"] != "review" and self.is_mastered(ti):
            fin = self.topic_name(ti)
            yield {"type": "celebrate", "topic": fin, "index": ti}
            if self.topic_idx < len(self.topics) - 1:
                self.topic_idx += 1; self.since_review = 0
                yield self.say("teacher", f"🚀 You've really got **{fin}**. Let's move on to the next topic!", "milestone")
                self.stage = "lesson"
            else:
                yield self.say("teacher", f"🎓 You've mastered every topic in this {self.subject} course. Amazing work!", "milestone")
                self.stage = "done"
                yield from self.recap()
        yield self.state_event()
        if self.stage != "done":
            yield from self._advance()

    def continue_socratic(self, text: str):
        pend = self.socratic_pending
        item = (pend or {}).get("item") or self.current_item
        ctx = self.current_ctx
        if not pend or not item or not ctx:
            self.stage = "quiz" if self.current_item else "question_pending"; return
        self.socratic_dialogue.append({"role": "student", "content": text})
        self.socratic_rounds += 1
        step = te.socratic_step(item.get("question", ""), item.get("answer", ""), item.get("solution", ""),
                                self.socratic_dialogue, text, self.api_key)
        yield self.say("teacher", step.get("feedback", ""), "socratic")
        resolved = bool(step.get("resolved")) or self.socratic_rounds >= 3
        if resolved:
            note = "Nicely reasoned through it! 🎉" if step.get("resolved") else f"Here's the full solution:\n\n{item.get('solution', '')}"
            yield self.say("teacher", note, "socratic_close")
            self.socratic_pending, self.socratic_dialogue, self.socratic_rounds = None, [], 0
            yield from self._finish_question(ctx, pend["ti"])
        else:
            self.socratic_dialogue.append({"role": "teacher", "content": step.get("next_prompt", "")})
            yield self.say("teacher", f"🤔 {step.get('next_prompt', '')}", "socratic")
            yield self.state_event()

    # ------------------------------------------------------------- extras
    def recap(self):
        ts = {self.topics[i]: {**x, "mastery": self.trackers[i].p_L} for i, x in self.topic_stats.items() if i in self.trackers}
        if not ts:
            yield {"type": "error", "message": "Answer a question or two first, then I can write a recap."}; return
        yield {"type": "status", "text": "Writing your recap…"}
        yield self.say("teacher", te.generate_recap(ts, self.misconceptions, self.api_key), "recap")
        yield self.state_event()

    def save_memory(self) -> Dict:
        notes = "\n".join(f"{self.topics[i]}: {s['correct']}/{s['n']} correct, {s['hints']} hints" for i, s in self.topic_stats.items())
        notes += "\nMistakes: " + "; ".join(self.misconceptions[-8:]) + "\nStudent said: " + \
                 " | ".join(t["content"] for t in self.chat if t["role"] == "student")[-600:]
        self.profile = te.summarize_profile(self.profile, notes, self.api_key)
        storage.save_profile(self.student_name, self.profile)
        return {"ok": True, "summary": self.profile.get("summary", "")}

    def add_material(self, name: str, data: bytes) -> Dict:
        n = self.rag.add_file(name, data)
        if name not in self.rag_files: self.rag_files.append(name)
        return {"added": n, "files": self.rag_files}
