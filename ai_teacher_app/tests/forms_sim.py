"""Simulated Google Forms CSV export (2 weekly quizzes, 30 students, hidden profiles)."""
import csv, json, numpy as np
from cohort_sim import TOPICS, MISC

PROF = {  # true learn, slip prob when known, overconfident, prereq gap
  "fast_learner": dict(k0=0.55, learn=0.35, slip=0.05, over=False, gap=False),
  "careless":     dict(k0=0.50, learn=0.30, slip=0.30, over=False, gap=False),
  "prereq_gap":   dict(k0=0.35, learn=0.20, slip=0.08, over=False, gap=True),
  "overconfident":dict(k0=0.30, learn=0.15, slip=0.08, over=True,  gap=False),
  "struggling":   dict(k0=0.12, learn=0.06, slip=0.10, over=False, gap=False),
}
GAP_TOPICS = {"Bayes rule", "Likelihood & MLE"}

def config():
    qs = []
    for ti, t in enumerate(TOPICS):
        for j, d in enumerate((1, 2, 3)):
            m = MISC[t]
            qs.append({"id": f"t{ti}q{j}", "topic": t, "difficulty": d, "header": f"Q{len(qs)+1}. {t} (level {d})",
                       "answer": "A", "options": ["A", "B", "C", "D"],
                       "distractors": {"B": m[0], "C": m[1 % len(m)]}, "slip_options": ["D"],
                       "confidence_header": f"Q{len(qs)+1}. How sure are you?"})
    return {"form_title": "GAI weekly quiz", "form_id": "gai-weekly", "id_column": "Email Address", "name_column": "Full name",
            "timestamp_column": "Timestamp", "questions": qs}

def responses(seed=3):
    rng = np.random.default_rng(seed); cfg = config(); rows, truth = [], {}
    names = ["Aarav","Diya","Kabir","Meera","Rohan","Isha","Vihaan","Anaya","Arjun","Sara","Reyansh","Myra","Ayaan","Kiara",
             "Vivaan","Aadhya","Krish","Navya","Shaurya","Pari","Atharv","Riya","Dhruv","Tara","Ishaan","Zoya","Advik","Mira","Yash","Nisha"]
    for i, n in enumerate(names):
        p = list(PROF)[i % 5]; P = PROF[p]; truth[n] = p
        know = {t: P["k0"] * (0.4 if P["gap"] and t in GAP_TOPICS else 1) for t in TOPICS}
        for week, day in ((1, 3), (2, 10)):
            r = {"Timestamp": f"11/{day:02d}/2026 10:{i:02d}:00", "Email Address": f"{n.lower()}@school.in", "Full name": n}
            for q in cfg["questions"]:
                t = q["topic"]; k = rng.random() < know[t] * (1.15 - 0.15 * q["difficulty"])
                if k: a = "D" if rng.random() < P["slip"] else "A"
                else: a = rng.choice(["B", "C"]) if rng.random() < 0.75 else rng.choice(["A", "D"])
                conf = ("Sure" if (k or P["over"]) else rng.choice(["Not sure", "Guessing"]))
                if P["over"] and not k: conf = "Sure" if rng.random() < 0.8 else "Not sure"
                r[q["header"]] = a; r[q["confidence_header"]] = conf
            rows.append(r)
            lr = P["learn"] * (0.35 if P["gap"] else 1)
            know = {t: v + (1 - v) * (lr if t not in GAP_TOPICS or not P["gap"] else lr * 0.5) for t, v in know.items()}
    return cfg, rows, truth

def write(path_csv, path_cfg):
    cfg, rows, truth = responses()
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(cfg, open(path_cfg, "w"), indent=1)
    return truth
