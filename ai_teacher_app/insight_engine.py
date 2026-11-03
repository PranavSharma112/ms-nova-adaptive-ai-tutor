"""
insight_engine.py
-----------------
Deterministic learning analytics over storage.interactions.jsonl events.
Turns raw question events into (a) per-student diagnostics and (b) class-level
insights, using detectors grounded in the ITS / assessment literature:

  wheel_spinning     Beck & Gong (2013): >=10 opportunities on a skill without mastery
  rapid_guessing     Wise & Ma (2012): response < 10% of mean item time (cap 10 s)
  hint_abuse         Aleven et al. (2006): clicking through hints instead of thinking
  help_avoidance     Aleven et al. (2006): repeated errors without asking for help
  plus misconception persistence, careless slips, misreading, incomplete answers,
  fluency gap, frustration streaks, fatigue/decline, difficulty ceiling, forgetting.

WHY deterministic: every flag must be explainable and reproducible from the log.
Gemini (optional, in report_builder) only rewrites prose; it never decides a flag.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, List, Optional

SLOW_SECONDS = {1: 40, 2: 50, 3: 70, 4: 90, 5: 120}      # same as tutor_service.py L28
MASTERY = 0.90                                          # rl_engine.MASTERY_THRESHOLD
WHEEL_N = 10                                            # Beck & Gong opportunities
EARLY_WHEEL_N = 5
RAPID_FRAC, RAPID_CAP = 0.10, 10.0                      # Wise & Ma normative threshold

SEV_RANK = {"high": 3, "medium": 2, "low": 1}

# ------------------------------------------------------------------ playbook
# For each pattern: what it means, what the student does now, what the teacher changes.
PLAYBOOK: Dict[str, Dict[str, str]] = {
    "wheel_spinning": {
        "title": "Wheel-spinning (practice without progress)",
        "meaning": "Many attempts on the same topic but mastery is not rising. More of the same practice will not fix it.",
        "student": "Stop drilling this topic. Re-read one worked example, write the key idea in your own words, then try 2 easier questions.",
        "teacher": "Switch the method, not the quantity: re-teach the prerequisite with a worked example, then restart at level 1-2.",
        "material": "Add a prerequisite primer and 2 fully worked examples with faded steps for this topic."},
    "early_wheel_risk": {
        "title": "Early warning: stuck on a topic",
        "meaning": "Five or more attempts with under 50% correct. Likely to wheel-spin if nothing changes.",
        "student": "Ask for one hint early on the next question and compare your method with the hint before answering.",
        "teacher": "Check the prerequisite now; a short 1:1 explanation is cheaper than 10 more failed questions.",
        "material": "Tag this topic's prerequisite in the notes so the tutor can retrieve it."},
    "rapid_guessing": {
        "title": "Rapid guessing (answering faster than reading)",
        "meaning": "Answers came in under 10% of the typical time for that level. These answers measure effort, not knowledge.",
        "student": "Before answering, write one line: what is the question asking? Then answer.",
        "teacher": "Treat these answers as missing, not wrong. Ask about motivation or fatigue; shorten sessions.",
        "material": "Prefer short-answer over multiple choice for this student so guessing does not pay."},
    "hint_abuse": {
        "title": "Hint over-use (clicking through hints)",
        "meaning": "Two or more hints taken quickly, before a real attempt. The last hint gives the answer away.",
        "student": "Make one honest attempt before the first hint, and wait 30 seconds between hints.",
        "teacher": "Re-test the same skill without hints; mastery earned with 2-3 hints is not yet mastery.",
        "material": "Make hint 1 a question back to the student, not a step of the solution."},
    "help_avoidance": {
        "title": "Help avoidance (stuck but not asking)",
        "meaning": "Three or more wrong answers with zero hints. The student is not using available support.",
        "student": "After one wrong answer, take hint 1. Using a hint is a learning move, not a failure.",
        "teacher": "Normalise help: model asking a question aloud; check for anxiety about looking weak.",
        "material": "Show hint 1 automatically after the second miss on a topic."},
    "persistent_misconception": {
        "title": "Persistent misconception",
        "meaning": "The same conceptual error repeats on the same skill. The mental model is wrong, not the arithmetic.",
        "student": "Write the wrong rule you were using next to the correct rule, then solve one question each way.",
        "teacher": "Use a contrasting example (right vs wrong method side by side) and a 'spot the error' item.",
        "material": "Add a 'common mistakes' box for this skill in the notes."},
    "careless_slips": {
        "title": "Careless slips despite knowing the topic",
        "meaning": "Arithmetic or small errors while mastery was already high. Knowledge is there; checking is not.",
        "student": "Add a 20-second check: re-substitute or estimate the answer before submitting.",
        "teacher": "Do not re-teach the concept. Coach a checking routine instead.",
        "material": "Include an 'estimate first' line in worked solutions."},
    "misreading": {
        "title": "Misreading the question",
        "meaning": "Errors came from answering a different question than the one asked.",
        "student": "Underline what is asked and the units before solving.",
        "teacher": "Check question wording for ambiguity; read one question aloud together.",
        "material": "Review flagged items for ambiguous wording."},
    "incomplete_answers": {
        "title": "Incomplete answers",
        "meaning": "Right direction but stops early or skips the justification.",
        "student": "End each answer by checking: did I answer every part, with units and a reason?",
        "teacher": "Share the rubric; show a full-marks answer next to a partial one.",
        "material": "Add an explicit answer checklist to explain-type items."},
    "fluency_gap": {
        "title": "Slow but correct (fluency gap)",
        "meaning": "Correct answers took longer than the expected time for the level. Understanding exists, speed does not.",
        "student": "Do short timed sets of 3 easy questions on this topic daily for a week.",
        "teacher": "Add spaced, low-stakes retrieval practice; no new content needed.",
        "material": "Create a 5-question speed drill for this topic."},
    "frustration_streak": {
        "title": "Frustration streak",
        "meaning": "Three or more wrong answers in a row. Risk of disengagement.",
        "student": "Take a 5-minute break, then restart with one easier question to rebuild momentum.",
        "teacher": "Cap the run: after 2 misses, drop a level or show a worked example.",
        "material": "Ensure each topic has at least 2 level-1 items in the bank."},
    "fatigue_decline": {
        "title": "Decline within the session",
        "meaning": "Accuracy in the last third of the session fell well below the first third.",
        "student": "Keep sessions to about 20 minutes, or split into two shorter sittings.",
        "teacher": "Schedule harder topics early in the session.",
        "material": "-"},
    "difficulty_ceiling": {
        "title": "Difficulty ceiling",
        "meaning": "Accuracy collapses above a specific level. That level is the current frontier.",
        "student": "Practise at the ceiling level with hints allowed until 3 in a row are correct.",
        "teacher": "Target instruction exactly at the ceiling level; skip easier review.",
        "material": "Add bridging items between the last passed level and the ceiling."},
    "forgetting": {
        "title": "Forgetting on review",
        "meaning": "A topic that was strong was answered wrong on a later review.",
        "student": "Review this topic again in 2 days and in 1 week (spaced repetition).",
        "teacher": "Increase review frequency for this topic.",
        "material": "Add this topic to flashcards."},
}


def _f(x, d=0.0):
    try: return float(x)
    except (TypeError, ValueError): return d


def _flag(code, sev, topic, evidence, n_events=0):
    pb = PLAYBOOK[code]
    return {"code": code, "title": pb["title"], "severity": sev, "topic": topic, "evidence": evidence,
            "meaning": pb["meaning"], "student_action": pb["student"], "teacher_action": pb["teacher"],
            "material_change": pb["material"], "n_events": n_events}


# ------------------------------------------------------------------ norms
def time_norms(events: List[Dict]) -> Dict[int, float]:
    """Mean answer time per difficulty across ALL events (normative baseline)."""
    by = defaultdict(list)
    for e in events:
        if e.get("kind") != "worked_example" and e.get("elapsed") is not None:
            by[int(e.get("difficulty") or 0)].append(_f(e["elapsed"]))
    return {d: mean(v) for d, v in by.items() if v}


def rapid_threshold(d: int, norms: Dict[int, float]) -> float:
    base = norms.get(d, SLOW_SECONDS.get(d, 60) * 0.6)
    return min(RAPID_CAP, RAPID_FRAC * base)


def slow_threshold(d: int, norms: Dict[int, float]) -> float:
    return max(SLOW_SECONDS.get(d, 60), 1.5 * norms.get(d, 60))


# ------------------------------------------------------------------ student
def analyze_student(events: List[Dict], norms: Optional[Dict[int, float]] = None,
                    misconceptions: Optional[List[str]] = None) -> Dict:
    """events: one student's events (any number of sessions), any order."""
    ev = sorted(events, key=lambda e: _f(e.get("ts")))
    qs = [e for e in ev if e.get("kind") != "worked_example"]
    norms = norms or time_norms(qs)
    n = len(qs)
    if n == 0:
        return {"n": 0, "flags": [], "topics": {}, "summary": {}, "timeline": []}

    # ---- per-topic metrics
    topics: Dict[str, Dict] = {}
    for e in qs:
        t = e.get("topic", "?")
        T = topics.setdefault(t, {"n": 0, "correct": 0, "hints": 0, "times": [], "p": [], "diff": [],
                                  "errors": Counter(), "skills": Counter(), "worked": 0, "start_p": _f(e.get("p_before"))})
        T["n"] += 1; T["correct"] += int(bool(e.get("correct"))); T["hints"] += int(e.get("hints_used") or 0)
        T["times"].append(_f(e.get("elapsed"))); T["p"].append(_f(e.get("p_after"))); T["diff"].append(int(e.get("difficulty") or 0))
        if not e.get("correct"):
            T["errors"][e.get("error_type") or "unknown"] += 1
            T["skills"][(e.get("skill_tag") or t, e.get("error_type"))] += 1
    for e in ev:
        if e.get("kind") == "worked_example" and e.get("topic") in topics: topics[e["topic"]]["worked"] += 1
    for t, T in topics.items():
        T["accuracy"] = T["correct"] / T["n"]
        T["end_p"] = T["p"][-1]
        T["gain"] = T["end_p"] - T["start_p"]
        T["mean_time"] = mean(T["times"])
        T["mastered"] = T["end_p"] >= MASTERY

    flags: List[Dict] = []

    # ---- 1 wheel-spinning (per topic)
    for t, T in topics.items():
        cor = [int(bool(e.get("correct"))) for e in qs if e.get("topic") == t]
        three = any(cor[i:i + 3] == [1, 1, 1] for i in range(len(cor) - 2))
        if T["n"] >= WHEEL_N and not (T["mastered"] or three):
            flags.append(_flag("wheel_spinning", "high", t,
                               f"{T['n']} attempts, mastery {T['start_p']:.2f} → {T['end_p']:.2f}, never 3 correct in a row", T["n"]))
        elif EARLY_WHEEL_N <= T["n"] < WHEEL_N and T["accuracy"] < 0.5 and not T["mastered"]:
            flags.append(_flag("early_wheel_risk", "medium", t,
                               f"{T['n']} attempts, {T['accuracy']:.0%} correct, mastery {T['end_p']:.2f}", T["n"]))

    # ---- 2 rapid guessing
    rapid = [e for e in qs if _f(e.get("elapsed"), 1e9) < rapid_threshold(int(e.get("difficulty") or 0), norms)]
    if rapid:
        share = len(rapid) / n
        acc = mean(int(bool(e.get("correct"))) for e in rapid)
        if len(rapid) >= 2 or share >= 0.15:
            flags.append(_flag("rapid_guessing", "high" if share >= 0.25 else "medium", "all",
                               f"{len(rapid)} of {n} answers ({share:.0%}) under the time threshold; {acc:.0%} of those correct", len(rapid)))

    # ---- 3 hint abuse vs 4 help avoidance
    abuse = [e for e in qs if int(e.get("hints_used") or 0) >= 2
             and _f(e.get("elapsed"), 1e9) < 0.5 * norms.get(int(e.get("difficulty") or 0), 60)]
    hint_rate = sum(int(e.get("hints_used") or 0) for e in qs) / n
    if len(abuse) >= 2 or (hint_rate >= 1.5 and n >= 4):
        flags.append(_flag("hint_abuse", "medium", "all",
                           f"{len(abuse)} quick multi-hint answers; {hint_rate:.1f} hints per question on average", len(abuse)))
    avoid = [e for e in qs if not e.get("correct") and int(e.get("hints_used") or 0) == 0]
    if len(avoid) >= 3 and len(avoid) / max(1, n - sum(T["correct"] for T in topics.values())) >= 0.7:
        flags.append(_flag("help_avoidance", "medium", "all",
                           f"{len(avoid)} wrong answers with no hint requested", len(avoid)))

    # ---- 5-8 error-type patterns
    for t, T in topics.items():
        for (skill, et), k in T["skills"].items():
            if et == "conceptual_misconception" and k >= 2:
                flags.append(_flag("persistent_misconception", "high", t, f"'{skill}': same conceptual error {k} times", k))
    slips = [e for e in qs if e.get("error_type") == "arithmetic_slip" and _f(e.get("p_before")) >= 0.7]
    if len(slips) >= 2:
        flags.append(_flag("careless_slips", "low", "all", f"{len(slips)} slips while mastery was already ≥ 0.70", len(slips)))
    mis = [e for e in qs if e.get("error_type") == "misread_question"]
    if len(mis) >= 2:
        flags.append(_flag("misreading", "medium", "all", f"{len(mis)} answers to a different question than asked", len(mis)))
    inc = [e for e in qs if e.get("error_type") == "incomplete_answer"]
    if len(inc) >= 2:
        flags.append(_flag("incomplete_answers", "low", "all", f"{len(inc)} partially complete answers", len(inc)))

    # ---- 9 fluency gap
    for t, T in topics.items():
        # WATCH: fixed SLOW_SECONDS alone flags normal variation; require 1.5x the class norm too.
        slow = [e for e in qs if e.get("topic") == t and e.get("correct")
                and _f(e.get("elapsed")) > slow_threshold(int(e.get("difficulty") or 0), norms)]
        if len(slow) >= 2 and len(slow) >= 0.4 * max(1, T["correct"]):
            flags.append(_flag("fluency_gap", "low", t, f"{len(slow)} of {T['correct']} correct answers were slower than expected", len(slow)))

    # ---- 10 frustration streak
    run = best = 0
    for e in qs:
        run = run + 1 if not e.get("correct") else 0; best = max(best, run)
    if best >= 3:
        flags.append(_flag("frustration_streak", "high" if best >= 4 else "medium", "all", f"longest run of wrong answers: {best}", best))

    # ---- 11 fatigue / decline (needs >= 9 questions)
    if n >= 9:
        k = n // 3
        a1 = mean(int(bool(e.get("correct"))) for e in qs[:k]); a3 = mean(int(bool(e.get("correct"))) for e in qs[-k:])
        if a1 - a3 >= 0.25:
            flags.append(_flag("fatigue_decline", "medium", "all", f"accuracy first third {a1:.0%} → last third {a3:.0%}", n))

    # ---- 12 difficulty ceiling
    by_d = defaultdict(list)
    for e in qs: by_d[int(e.get("difficulty") or 0)].append(int(bool(e.get("correct"))))
    acc_d = {d: (mean(v), len(v)) for d, v in sorted(by_d.items())}
    ds = [d for d in acc_d if acc_d[d][1] >= 2]
    for lo, hi in zip(ds, ds[1:]):
        if acc_d[lo][0] >= 0.7 and acc_d[hi][0] <= 0.4:
            flags.append(_flag("difficulty_ceiling", "medium", "all",
                               f"level {lo}: {acc_d[lo][0]:.0%} correct → level {hi}: {acc_d[hi][0]:.0%}", acc_d[hi][1]))
            break

    # ---- 13 forgetting on review
    forg = [e for e in qs if e.get("kind") == "review" and not e.get("correct") and _f(e.get("p_before")) >= 0.7]
    for t in sorted({e.get("topic") for e in forg}):
        flags.append(_flag("forgetting", "medium", t, "missed on review after mastery ≥ 0.70",
                           sum(1 for e in forg if e.get("topic") == t)))

    # WHY: one card per pattern; per-topic hits merge so reports do not repeat themselves.
    merged: Dict[str, Dict] = {}
    for f in flags:
        m = merged.get(f["code"])
        if m is None:
            merged[f["code"]] = {**f, "topics": [f["topic"]], "evidence_list": [f["evidence"]]}
        else:
            m["topics"].append(f["topic"]); m["evidence_list"].append(f["evidence"]); m["n_events"] += f["n_events"]
            if SEV_RANK[f["severity"]] > SEV_RANK[m["severity"]]: m["severity"] = f["severity"]
    if "wheel_spinning" in merged and "early_wheel_risk" in merged:      # same topic can't be both
        ew = merged["early_wheel_risk"]
        keep = [(t, ev) for t, ev in zip(ew["topics"], ew["evidence_list"]) if t not in merged["wheel_spinning"]["topics"]]
        if keep: ew["topics"], ew["evidence_list"] = [k[0] for k in keep], [k[1] for k in keep]
        else: merged.pop("early_wheel_risk")
    flags = sorted(merged.values(), key=lambda f: (-SEV_RANK[f["severity"]], -f["n_events"]))

    # ---- summary + strengths
    correct = sum(T["correct"] for T in topics.values())
    strengths = [t for t, T in topics.items() if T["mastered"] or (T["accuracy"] >= 0.8 and T["n"] >= 3)]
    priorities = [t for t, _ in sorted(topics.items(), key=lambda kv: kv[1]["end_p"]) if not topics[t]["mastered"]][:3]
    errs = Counter()
    for T in topics.values(): errs.update(T["errors"])
    summary = {"questions": n, "accuracy": correct / n, "hints_per_q": hint_rate,
               "mean_time": mean(_f(e.get("elapsed")) for e in qs), "worked_examples": sum(T["worked"] for T in topics.values()),
               "topics_mastered": sum(T["mastered"] for T in topics.values()), "topics": len(topics),
               "strengths": strengths, "priorities": priorities, "error_mix": dict(errs),
               "acc_by_difficulty": {d: v[0] for d, v in acc_d.items()}, "n_by_difficulty": {d: v[1] for d, v in acc_d.items()},
               "rapid_ids": [i for i, e in enumerate(qs) if e in rapid], "longest_wrong_run": best,
               "misconception_notes": (misconceptions or [])[-8:]}
    timeline = [{"i": i + 1, "topic": e.get("topic"), "d": int(e.get("difficulty") or 0), "correct": int(bool(e.get("correct"))),
                 "p": _f(e.get("p_after")), "t": _f(e.get("elapsed")), "hints": int(e.get("hints_used") or 0),
                 "rapid": e in rapid, "err": e.get("error_type")} for i, e in enumerate(qs)]
    for T in topics.values():
        T["errors"] = dict(T["errors"]); T.pop("skills")
    return {"n": n, "flags": flags, "topics": topics, "summary": summary, "timeline": timeline,
            "rapid_thresholds": {d: round(rapid_threshold(d, norms), 1) for d in sorted(by_d)},
            "slow_thresholds": {d: round(slow_threshold(d, norms), 1) for d in sorted(by_d)}}


# ------------------------------------------------------------------ class
def analyze_class(events: List[Dict], student_of=None) -> Dict:
    """events: all events. student_of(event) -> student key (default: event['student'] or session_id)."""
    key = student_of or (lambda e: e.get("student") or e.get("session_id", "?"))
    norms = time_norms(events)
    by_s = defaultdict(list)
    for e in events: by_s[key(e)].append(e)
    students = {s: analyze_student(ev, norms) for s, ev in by_s.items()}
    students = {s: r for s, r in students.items() if r["n"] > 0}
    if not students:
        return {"n_students": 0}

    # pattern prevalence
    prev = Counter(); prev_students = defaultdict(set)
    for s, r in students.items():
        for f in r["flags"]:
            prev[f["code"]] += 1; prev_students[f["code"]].add(s)

    # topic difficulty
    tstat = defaultdict(lambda: {"n": 0, "correct": 0, "gain": [], "students": 0, "mastered": 0, "errors": Counter()})
    for s, r in students.items():
        for t, T in r["topics"].items():
            X = tstat[t]; X["n"] += T["n"]; X["correct"] += T["correct"]; X["gain"].append(T["gain"])
            X["students"] += 1; X["mastered"] += int(T["mastered"]); X["errors"].update(T["errors"])
    topics = {t: {"accuracy": X["correct"] / X["n"], "mean_gain": mean(X["gain"]), "students": X["students"],
                  "mastery_rate": X["mastered"] / X["students"], "n": X["n"], "top_error": (X["errors"].most_common(1) or [("-", 0)])[0]}
              for t, X in tstat.items()}

    # item quality (needs >= 3 unhinted attempts)
    items = defaultdict(lambda: {"n": 0, "correct": 0, "label": 0, "topic": "", "times": []})
    for e in events:
        if e.get("kind") == "worked_example" or not e.get("item_id") or int(e.get("hints_used") or 0) > 0: continue
        I = items[e["item_id"]]; I["n"] += 1; I["correct"] += int(bool(e.get("correct")))
        I["label"] = int(e.get("difficulty") or 0); I["topic"] = e.get("topic", ""); I["times"].append(_f(e.get("elapsed")))
    item_flags = []
    for iid, I in items.items():
        if I["n"] < 3: continue
        p = I["correct"] / I["n"]; emp = 1 + 4 * (1 - p)
        if abs(emp - I["label"]) >= 2:
            item_flags.append({"item_id": iid, "topic": I["topic"], "label": I["label"], "observed": round(emp, 1),
                               "p_correct": p, "n": I["n"],
                               "fix": "re-label harder or rewrite" if emp > I["label"] else "re-label easier or add depth"})
        elif p == 0:
            item_flags.append({"item_id": iid, "topic": I["topic"], "label": I["label"], "observed": 5.0, "p_correct": 0, "n": I["n"],
                               "fix": "check answer key / wording"})

    # risk ranking: severity-weighted flags + low mastery
    risk = []
    for s, r in students.items():
        score = sum(SEV_RANK[f["severity"]] for f in r["flags"]) + 5 * (1 - r["summary"]["accuracy"])
        risk.append({"student": s, "risk": round(score, 1), "accuracy": r["summary"]["accuracy"],
                     "top_flag": r["flags"][0]["title"] if r["flags"] else "none", "n": r["n"]})
    risk.sort(key=lambda x: -x["risk"])

    return {"n_students": len(students), "n_events": sum(r["n"] for r in students.values()), "students": students,
            "prevalence": {c: {"flags": prev[c], "students": len(prev_students[c]), "share": len(prev_students[c]) / len(students),
                               "title": PLAYBOOK[c]["title"], "teacher": PLAYBOOK[c]["teacher"], "material": PLAYBOOK[c]["material"]}
                           for c in prev},
            "topics": topics, "item_flags": item_flags, "risk": risk, "norms": norms}
