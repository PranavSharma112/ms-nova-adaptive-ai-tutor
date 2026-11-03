"""
Synthetic cohort with HIDDEN learner profiles, used to prove the deep engine
recovers what it cannot see. 30 students x 5 topics x 3 sessions (~40 answers).
Each profile changes the true BKT parameters and behaviour, NOT the labels.
"""
import numpy as np
from rl_engine import BKTTracker, SLIP, GUESS

TOPICS = ["Probability basics", "Conditional probability", "Bayes rule", "Generative models", "Likelihood & MLE"]
PREREQ = {"Bayes rule": "Conditional probability", "Likelihood & MLE": "Probability basics"}
MISC = {
  "Conditional probability": ["confuses P(A|B) with P(B|A)", "treats P(A|B) as P(A and B)", "reverses the condition"],
  "Bayes rule": ["forgets the prior", "ignores base rate", "uses P(B|A) as the answer"],
  "Probability basics": ["adds probabilities of non-exclusive events", "probability above 1"],
  "Generative models": ["mixes up generative and discriminative", "thinks GANs need labels"],
  "Likelihood & MLE": ["treats likelihood as a probability over parameters", "forgets the log"],
}
PROFILES = {   # p_transit, slip_scale, guess_scale, rapid_rate, hint_rate, time_mult, fatigue, prereq_gap
  "fast_learner":    dict(tr=0.30, sl=0.6, gs=1.0, rapid=0.00, hint=0.05, tm=0.7, fat=0.0, gap=False),
  "careless":        dict(tr=0.20, sl=2.6, gs=1.0, rapid=0.03, hint=0.05, tm=0.6, fat=0.0, gap=False),
  "rapid_guesser":   dict(tr=0.10, sl=1.0, gs=1.0, rapid=0.35, hint=0.00, tm=0.9, fat=0.2, gap=False),
  "prereq_gap":      dict(tr=0.16, sl=1.0, gs=1.0, rapid=0.02, hint=0.20, tm=1.1, fat=0.0, gap=True),
  "hint_dependent":  dict(tr=0.12, sl=1.0, gs=1.0, rapid=0.00, hint=0.75, tm=1.0, fat=0.0, gap=False),
  "slow_steady":     dict(tr=0.07, sl=0.8, gs=0.8, rapid=0.00, hint=0.15, tm=1.9, fat=0.0, gap=False),
}

def simulate(student, profile, rng, start_ts=1.79e9):
    P = PROFILES[profile]; ev = []; ts = start_ts
    mean_t = {1: 30, 2: 40, 3: 55, 4: 70, 5: 90}
    for sess in range(3):
        topics = TOPICS[sess * 2: sess * 2 + 2] if sess < 2 else TOPICS[4:] + TOPICS[:1]
        for t in topics:
            tr = BKTTracker(p_init=0.15, p_transit=P["tr"] * (0.35 if P["gap"] and t in PREREQ else 1),
                            slip_scale=P["sl"], guess_scale=P["gs"])
            review = sess == 2 and t == TOPICS[0]
            d, n_q, k = 2, (4 if review else int(rng.integers(7, 11))), 0
            while k < n_q:
                if not review and tr.p_L < 0.6 and d >= 2 and rng.random() < 0.15:     # worked example
                    pb, pa = tr.worked_example()
                    ev.append(dict(ts=ts, session_id=f"{student}_s{sess+1}", student=student, topic=t, kind="worked_example",
                                   p_before=pb, p_after=pa)); ts += 60; continue
                frac = k / max(1, n_q - 1)
                rapid = rng.random() < P["rapid"] * (1 + P["fat"] * 3 * frac)
                hints = int(rng.integers(1, 4)) if rng.random() < P["hint"] else 0
                slip, guess = tr._sg(d)
                truth = rng.random() < tr.p_L                       # hidden: does student know it?
                if rapid: c = int(rng.random() < 0.2)
                else:
                    c = int(rng.random() < (1 - slip)) if truth else int(rng.random() < min(0.9, guess + 0.2 * hints))
                if c: err = "none"
                elif rapid: err = "off_topic"
                elif truth: err = "arithmetic_slip"
                else: err = "conceptual_misconception" if rng.random() < 0.7 else ("incomplete_answer" if rng.random() < .6 else "misread_question")
                el = 2.0 + rng.random() * 3 if rapid else float(max(5, rng.normal(mean_t[d] * P["tm"] * (1 + 0.15 * hints), 8)))
                pb, pa = tr.observe(d, c, guess_bonus=0.15 * hints)
                note = rng.choice(MISC[t]) if err == "conceptual_misconception" else ""
                ev.append(dict(ts=ts, session_id=f"{student}_s{sess+1}", student=student, topic=t,
                               kind="review" if review else "question", difficulty=d, correct=c, elapsed=round(el, 1),
                               hints_used=hints, error_type=err, misconception_note=note, skill_tag=t,
                               item_id=f"{t[:4]}-L{d}-{int(rng.integers(0, 4))}", p_before=pb, p_after=pa))
                ts += el + 10; k += 1
                d = min(5, d + 1) if c and tr.p_L > 0.5 else (max(1, d - 1) if not c else d)
        ts += 86400 * 2
    return ev

def cohort(n_per=5, seed=7):
    rng = np.random.default_rng(seed); ev, truth = [], {}
    names = ["Aarav","Diya","Kabir","Meera","Rohan","Isha","Vihaan","Anaya","Arjun","Sara","Reyansh","Myra","Ayaan","Kiara",
             "Vivaan","Aadhya","Krish","Navya","Shaurya","Pari","Atharv","Riya","Dhruv","Tara","Ishaan","Zoya","Advik","Mira","Yash","Nisha"]
    i = 0
    for prof in PROFILES:
        for _ in range(n_per):
            s = names[i]; i += 1; truth[s] = prof; ev += simulate(s, prof, rng)
    return ev, truth
