"""Synthetic event logs with KNOWN injected patterns, used to test insight_engine."""
import numpy as np
from rl_engine import BKTTracker

TOPICS = ["Probability basics", "Bayes rule", "Generative models"]

def make(student, pattern, seed):
    rng = np.random.default_rng(seed); ev = []; ts = 1.79e9
    for ti, topic in enumerate(TOPICS):
        tr = BKTTracker(p_init=0.25); d = 2
        n_q = 12 if (pattern == "wheel" and ti == 1) else 6
        for k in range(n_q):
            pc = tr.p_correct(d)
            t = float(rng.normal(45 + 8 * d, 8)); hints = 0; err = "none"
            if pattern == "wheel" and ti == 1: pc = 0.25
            if pattern == "rapid" and k % 2 == 0: t = 2.0; pc = 0.2
            if pattern == "hints" and k % 2 == 0: hints, t = 3, 15.0
            if pattern == "strong": pc = 0.95
            if pattern == "slow" : t = 160.0
            c = int(rng.random() < pc)
            if not c:
                err = "conceptual_misconception" if pattern == "wheel" and ti == 1 else ("arithmetic_slip" if pattern == "strong" else "incomplete_answer")
            pb, pa = tr.observe(d, c, guess_bonus=0.15 * hints)
            ev.append({"ts": ts, "session_id": student + "_s1", "student": student, "topic": topic, "kind": "question",
                       "difficulty": d, "correct": c, "elapsed": round(t, 1), "hints_used": hints, "error_type": err,
                       "skill_tag": topic + " core", "item_id": f"{topic[:3]}{d}{k%3}", "p_before": pb, "p_after": pa})
            ts += t + 5
            d = min(5, d + 1) if c else max(1, d - 1)
    return ev

STUDENTS = {"Asha": "strong", "Ravi": "wheel", "Meera": "rapid", "Kabir": "hints", "Isha": "slow"}
def all_events():
    out = []
    for i, (s, p) in enumerate(STUDENTS.items()): out += make(s, p, i)
    return out
