"""
deep_engine.py
--------------
Learner MODELLING on the interaction log (insight_engine.py only counts patterns).

Per student
  * personal BKT fit (maximum likelihood, grid search): learn rate, slip, guess, start knowledge
  * per-topic learn rate  -> prerequisite gaps show up as one topic learning far slower
  * effort filter (Wise & Ma rapid guesses removed before fitting)
  * forecast: questions still needed to reach mastery (Monte Carlo with the fitted model)
  * counterfactuals: accuracy if slips were caught / if rapid guesses were real attempts
  * hint efficacy: does a hint transfer to the NEXT unaided answer?
  * behaviour fingerprint (7 dimensions) as class percentiles
  * forensic timeline: each answer with likely cause
Per class
  * k-means clusters on the fingerprint -> instruction groups, named from centroids
  * item analysis: difficulty + discrimination (point-biserial vs rest-score)
  * teaching-move effectiveness: next-answer mastery gain after worked example vs not
  * early warning: forecast questions-to-mastery on weakest topic
  * misconception clustering (token-overlap fallback; LLM agent can name clusters)
"""
from __future__ import annotations
import itertools, math, re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Dict, List, Optional
import numpy as np

import insight_engine as ie
from rl_engine import SLIP, GUESS

MASTERY = 0.90
GRID_INIT = [0.05, 0.15, 0.3, 0.5]
GRID_TR = [0.02, 0.05, 0.08, 0.12, 0.17, 0.23, 0.3, 0.4]
GRID_SL = [0.5, 0.8, 1.2, 1.7, 2.4, 3.2]
GRID_GS = [0.6, 1.0, 1.5]


# ------------------------------------------------------------------ BKT likelihood
def _ll(seq, p0, tr, sls, gss):
    """seq: list of (topic, d, correct, hints). Returns log-likelihood, topics reset to p0."""
    ll, p, last = 0.0, p0, None
    for t, d, c, h in seq:
        if t != last: p, last = p0, t
        slip = min(0.45, SLIP[d] * sls); guess = min(0.6, GUESS[d] * gss + 0.15 * h)
        pc = min(1 - 1e-6, max(1e-6, p * (1 - slip) + (1 - p) * guess))
        ll += math.log(pc if c else 1 - pc)
        post = p * (1 - slip) / pc if c else p * slip / (1 - pc)
        p = post + (1 - post) * tr
    return ll


def fit_bkt(seq) -> Dict:
    best = (-1e18, None)
    for p0, tr, sl, gs in itertools.product(GRID_INIT, GRID_TR, GRID_SL, GRID_GS):
        v = _ll(seq, p0, tr, sl, gs)
        if v > best[0]: best = (v, (p0, tr, sl, gs))
    p0, tr, sl, gs = best[1]
    out = {"p_init": p0, "learn": tr, "slip_scale": sl, "guess_scale": gs, "ll": best[0], "n": len(seq)}
    # per-topic learn rate with the other parameters fixed
    by_t = defaultdict(list)
    for x in seq: by_t[x[0]].append(x)
    out["topic_learn"] = {t: max(GRID_TR, key=lambda r: _ll(s, p0, r, sl, gs)) for t, s in by_t.items() if len(s) >= 4}
    return out


def forecast_to_mastery(p, fit, d=3, sims=300, cap=40, seed=0) -> float:
    """Expected further questions until p >= 0.90 at difficulty d under the fitted model."""
    rng = np.random.default_rng(seed); out = []
    slip = min(0.45, SLIP[d] * fit["slip_scale"]); guess = min(0.6, GUESS[d] * fit["guess_scale"])
    for _ in range(sims):
        q, k = p, 0
        while q < MASTERY and k < cap:
            pc = q * (1 - slip) + (1 - q) * guess; c = rng.random() < pc
            post = q * (1 - slip) / pc if c else q * slip / (1 - pc)
            q = post + (1 - post) * fit["learn"]; k += 1
        out.append(k)
    return float(np.mean(out))


def _growth(qs):
    """Accuracy change from first to last submission/session (None if only one)."""
    order = list(dict.fromkeys(e.get("session_id") for e in qs))
    if len(order) < 2: return None
    a = [int(bool(e["correct"])) for e in qs if e.get("session_id") == order[0]]
    b = [int(bool(e["correct"])) for e in qs if e.get("session_id") == order[-1]]
    return mean(b) - mean(a)


def _topic_gap_acc(qs):
    by = defaultdict(list)
    for e in qs: by[e["topic"]].append(int(bool(e["correct"])))
    accs = [mean(v) for v in by.values() if len(v) >= 3]
    return (min(accs) - mean(accs)) if len(accs) >= 3 else None


# ------------------------------------------------------------------ per student
def _qs(events): return sorted([e for e in events if e.get("kind") != "worked_example"], key=lambda e: e.get("ts", 0))


def deep_student(events: List[Dict], norms: Dict[int, float]) -> Dict:
    qs = _qs(events)
    if not qs: return {"n": 0}
    rapid = [e for e in qs if float(e.get("elapsed") or 1e9) < ie.rapid_threshold(int(e["difficulty"]), norms)]
    rid = {id(e) for e in rapid}
    clean = [e for e in qs if id(e) not in rid]
    seq = [(e["topic"], int(e["difficulty"]), int(bool(e["correct"])), int(e.get("hints_used") or 0)) for e in clean]
    fit = fit_bkt(seq) if len(seq) >= 6 else None

    acc = mean(int(bool(e["correct"])) for e in qs)
    eff_acc = mean(int(bool(e["correct"])) for e in clean) if clean else acc
    slips = sum(1 for e in qs if e.get("error_type") == "arithmetic_slip")
    cf = {"actual": acc, "effort_adjusted": eff_acc, "if_slips_caught": acc + slips / len(qs),
          "slips": slips, "rapid": len(rapid), "rapid_share": len(rapid) / len(qs)}

    # hint efficacy: after a hinted answer, is the NEXT unaided answer on that topic correct?
    after_h, after_n = [], []
    for a, b in zip(qs, qs[1:]):
        if a["topic"] != b["topic"] or int(b.get("hints_used") or 0) > 0: continue
        (after_h if int(a.get("hints_used") or 0) > 0 else after_n).append(int(bool(b["correct"])))
    hint = {"hinted_q": sum(1 for e in qs if int(e.get("hints_used") or 0) > 0),
            "transfer_after_hint": mean(after_h) if after_h else None, "after_no_hint": mean(after_n) if after_n else None,
            "n_after_hint": len(after_h)}

    # topics: last mastery + forecast
    topics = {}
    for t in dict.fromkeys(e["topic"] for e in qs):
        tq = [e for e in qs if e["topic"] == t]; pl = float(tq[-1]["p_after"])
        topics[t] = {"n": len(tq), "acc": mean(int(bool(e["correct"])) for e in tq), "start": float(tq[0]["p_before"]), "end": pl,
                     "learn": (fit or {}).get("topic_learn", {}).get(t),
                     "to_mastery": 0.0 if pl >= MASTERY or not fit else forecast_to_mastery(pl, fit, d=3),
                     "errors": dict(Counter(e["error_type"] for e in tq if not e["correct"])),
                     "notes": [e.get("misconception_note") for e in tq if e.get("misconception_note")]}

    # session decline
    by_s = defaultdict(list)
    for e in qs: by_s[e.get("session_id")].append(int(bool(e["correct"])))
    declines = [mean(v[:len(v)//2]) - mean(v[len(v)//2:]) for v in by_s.values() if len(v) >= 8]
    timed = [e for e in clean if e.get("elapsed") is not None]          # Google Forms: no per-question time
    tratio = mean(float(e["elapsed"]) / norms.get(int(e["difficulty"]), 60) for e in timed) if timed else None
    conf = [(float(e["confidence"]), int(bool(e["correct"]))) for e in qs if e.get("confidence") is not None]
    overconf = (mean(c for c, _ in conf) - mean(k for _, k in conf)) if conf else None
    sure_wrong = sum(1 for c, k in conf if c >= 0.99 and not k)
    lucky = sum(1 for c, k in conf if c <= 0.01 and k)

    feat = {"learn_rate": fit["learn"] if fit else None, "slip": fit["slip_scale"] if fit else None,
            "rapid_share": cf["rapid_share"], "hint_rate": mean(int(e.get("hints_used") or 0) > 0 for e in qs),
            "time_ratio": tratio, "decline": mean(declines) if declines else 0.0,
            "topic_gap": (min(fit["topic_learn"].values()) / median(fit["topic_learn"].values())) if fit and fit["topic_learn"] else 1.0,
            "overconfidence": overconf,
            # observable features (used when the log has no timing, e.g. Google Forms)
            "accuracy": acc,
            "slip_share": slips / len(qs),
            "misconception_share": sum(1 for e in qs if e.get("error_type") == "conceptual_misconception") / len(qs),
            "growth": _growth(qs),
            "topic_gap_acc": _topic_gap_acc(qs)}

    # forensic timeline
    tl, prev = [], None
    for i, e in enumerate(qs, 1):
        c = bool(e["correct"]); h = int(e.get("hints_used") or 0); et = e.get("error_type")
        if id(e) in rid: cause = "Rapid guess: answered before the question could be read"
        elif c and h >= 2: cause = "Correct with heavy help: knowledge not yet independent"
        elif c and prev is not None and not prev: cause = "Recovered after a miss"
        elif c: cause = "Secure answer"
        elif et == "arithmetic_slip": cause = "Slip: knew it (mastery %.2f) but execution failed" % float(e["p_before"])
        elif et == "conceptual_misconception": cause = "Misconception: " + (e.get("misconception_note") or "wrong rule")
        elif et == "misread_question": cause = "Answered a different question"
        elif et == "incomplete_answer": cause = "Stopped early / missing justification"
        else: cause = "Unrelated or empty answer"
        tl.append({"i": i, "topic": e["topic"], "d": int(e["difficulty"]), "c": int(c), "h": h, "t": float(e.get("elapsed") or 0), "conf": e.get("confidence"),
                   "pb": float(e["p_before"]), "pa": float(e["p_after"]), "cause": cause, "rapid": id(e) in rid,
                   "session": e.get("session_id")})
        prev = c
    cf["sure_wrong"], cf["lucky_guess"] = (sure_wrong, lucky) if conf else (None, None)
    return {"n": len(qs), "fit": fit, "cf": cf, "hint": hint, "topics": topics, "features": feat, "timeline": tl}


# ------------------------------------------------------------------ class
ALL_FEATS = ["learn_rate", "slip", "rapid_share", "hint_rate", "time_ratio", "decline", "topic_gap", "overconfidence"]
# WHY: forms_sim recovery test showed BKT parameters are not identifiable from 2 snapshots of 15 items
# (purity fell to ~40%); directly observed rates recover the hidden profiles much better there.
FORM_FEATS = ["accuracy", "growth", "slip_share", "misconception_share", "topic_gap_acc", "overconfidence"]
FEATS = ALL_FEATS[:7]   # rebound per class in deep_class() to the features the data supports
FEAT_LABEL = {"learn_rate": "Learning speed", "slip": "Carelessness", "rapid_share": "Rapid guessing", "hint_rate": "Hint reliance",
              "time_ratio": "Time per answer", "decline": "In-session decline", "topic_gap": "Even across topics",
              "overconfidence": "Overconfidence", "accuracy": "Accuracy", "growth": "Growth week to week",
              "slip_share": "Careless slips", "misconception_share": "Misconceptions", "topic_gap_acc": "Even across topics"}


def _kmeans(X, k, seed=0, iters=100):
    rng = np.random.default_rng(seed); best = None
    for r in range(10):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            C2 = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(C, C2): break
            C = C2
        sse = ((X - C[lab]) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, lab, C)
    return best[1], best[2]


def _silhouette(X, lab):
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); s = []
    for i in range(len(X)):
        same = lab == lab[i]
        if same.sum() <= 1: s.append(0); continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


GROUP_RULES = [  # (feature, sign, name, plan)
    ("rapid_share", +1, "Disengaged guessers", "Short sessions, short-answer items only, effort check-in; treat fast answers as missing."),
    ("hint_rate", +1, "Hint-dependent", "Fade hints: first attempt unaided, hint 1 as a question, re-test the skill without hints next day."),
    ("slip", +1, "Fast but careless", "Do not re-teach; teach a 20-second checking routine and estimate-first habit."),
    ("topic_gap", -1, "Prerequisite gap", "Re-teach the prerequisite topic, then return; give a 5-question diagnostic on it first."),
    ("time_ratio", +1, "Slow and steady", "Same content, more spaced retrieval; timed low-stakes drills to build fluency."),
    ("overconfidence", +1, "Overconfident", "Ask for a confidence rating before feedback; use 'sure but wrong' items as contrasting cases in class."),
    ("slip_share", +1, "Knows it, slips", "Do not re-teach; teach a checking routine; mark method and answer separately."),
    ("misconception_share", +1, "Misconception-heavy", "Contrasting cases: wrong rule vs right rule side by side; 'spot the error' items."),
    ("topic_gap_acc", -1, "Topic gap", "Diagnose and re-teach the weakest topic's prerequisite before moving on."),
    ("growth", +1, "Rapid improvers", "Keep momentum: harder transfer items; ask them to explain their fix to peers."),
    ("accuracy", +1, "Secure performers", "Extend with level 3+ transfer problems; use as peer explainers."),
    ("accuracy", -1, "Struggling overall", "Small-group re-teach with worked examples; daily 10-minute retrieval."),
    ("learn_rate", -1, "Struggling overall", "Small-group re-teach of the unit core with worked examples; daily 10-minute retrieval."),
    ("learn_rate", +1, "Fast learners", "Compress practice; move to level 4-5 and transfer problems; use them as peer explainers."),
]


def _name_groups(C, mu, sd):
    """Assign each centroid a distinct archetype maximising total z-score (exact, k <= 6)."""
    # WHY: greedy naming in cluster order mislabelled groups in the recovery test.
    k = len(C); best = None
    rules = [i for i, r in enumerate(GROUP_RULES) if r[0] in FEATS]
    for perm in itertools.permutations(rules, k):
        if len({GROUP_RULES[r][2] for r in perm}) < k: continue
        sc = sum(C[j, FEATS.index(GROUP_RULES[r][0])] * GROUP_RULES[r][1] for j, r in enumerate(perm))
        if best is None or sc > best[0]: best = (sc, perm)
    out = []
    for j, r in enumerate(best[1]):
        f, sg, n, p = GROUP_RULES[r]
        out.append({"name": n, "plan": p, "driver": FEAT_LABEL[f], "z": float(C[j, FEATS.index(f)] * sg)})
    return out


def deep_class(events: List[Dict], key=lambda e: e.get("student") or e.get("session_id")) -> Dict:
    norms = ie.time_norms(events)
    by = defaultdict(list)
    for e in events: by[key(e)].append(e)
    S = {s: deep_student(ev, norms) for s, ev in by.items()}
    S = {s: r for s, r in S.items() if r["n"] and r["fit"]}
    names = list(S)
    global FEATS
    # WHY: a feature is only used if every student has it and it varies (Forms data has no time/hints).
    timed = any(e.get("elapsed") is not None for e in events)
    FEATS = [f for f in (ALL_FEATS if timed else FORM_FEATS) if all(S[s]["features"].get(f) is not None for s in names)
             and np.std([S[s]["features"][f] for s in names]) > 1e-9]
    X = np.array([[S[s]["features"][f] for f in FEATS] for s in names], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    # WHY: mid-rank percentile; with "<=" a tie at 0 (no rapid guesses) showed as a high percentile.
    pct = {s: {f: float(((X[:, j] < X[i, j]).sum() + 0.5 * ((X[:, j] == X[i, j]).sum() - 1)) / max(1, len(names) - 1))
               for j, f in enumerate(FEATS)} for i, s in enumerate(names)}

    best = None
    for k in range(3, min(7, len(names) - 1)):
        lab, C = _kmeans(Z, k); sil = _silhouette(Z, lab)
        if best is None or sil > best[0]: best = (sil, k, lab, C)
    sil, k, lab, C = best
    gnames = _name_groups(C, mu, sd)
    groups = [{**gnames[j], "members": [names[i] for i in range(len(names)) if lab[i] == j]} for j in range(k)]
    group_of = {names[i]: gnames[lab[i]]["name"] for i in range(len(names))}

    # item analysis (point-biserial vs rest score, unhinted, effort-filtered)
    # WATCH: rest-score correlation is biased under ADAPTIVE delivery (strong students meet
    # easy items only early). Use the model's knowledge estimate at that moment instead.
    items = defaultdict(list)
    for e in events:
        s = key(e)
        if s in S and e.get("kind") != "worked_example" and not int(e.get("hints_used") or 0) and e.get("item_id"):
            if float(e.get("elapsed") or 1e9) >= ie.rapid_threshold(int(e["difficulty"]), norms):
                items[e["item_id"]].append((float(e["p_before"]), int(bool(e["correct"])), e["topic"], int(e["difficulty"])))
    item_rows = []
    for iid, rows in items.items():
        if len(rows) < 8: continue
        y = np.array([r[1] for r in rows]); x = np.array([r[0] for r in rows])
        disc = float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 and x.std() > 0 else 0.0
        p = float(y.mean())
        # WATCH: n~10 per item makes r pure noise (recovery test flagged 19/53 identical items).
        # Only call an item bad when the 95% CI of r (Fisher z) is entirely below 0.1.
        zse = 1.96 / math.sqrt(max(1, len(rows) - 3)); zr = math.atanh(max(-0.999, min(0.999, disc)))
        hi = math.tanh(zr + zse)
        verdict = "good" if disc >= 0.3 else ("revise: does not separate strong from weak" if hi < 0.1 else "not enough data")
        if p > 0.95: verdict = "too easy"
        if p < 0.1: verdict = "check key / wording"
        item_rows.append({"item": iid, "topic": rows[0][2], "level": rows[0][3], "n": len(rows), "p": p, "disc": disc, "verdict": verdict})
    item_rows.sort(key=lambda r: r["disc"])

    # teaching-move effectiveness: mastery gain on the next answer after a worked example vs after a normal miss
    after_we, after_miss = [], []
    for s, ev in by.items():
        ev = sorted(ev, key=lambda e: e.get("ts", 0))
        for a, b in zip(ev, ev[1:]):
            if b.get("kind") == "worked_example" or a.get("topic") != b.get("topic"): continue
            g = float(b["p_after"]) - float(b["p_before"])
            if a.get("kind") == "worked_example": after_we.append((g, int(bool(b["correct"]))))
            elif not a.get("correct"): after_miss.append((g, int(bool(b["correct"]))))
    moves = {"worked_example": {"n": len(after_we), "next_correct": mean(x[1] for x in after_we) if after_we else None},
             "plain_retry": {"n": len(after_miss), "next_correct": mean(x[1] for x in after_miss) if after_miss else None}}

    # misconception clusters (token overlap fallback)
    notes = Counter()
    for e in events:
        if e.get("misconception_note"): notes[(e["topic"], e["misconception_note"].strip())] += 1
    misc = cluster_notes(notes)

    # early warning
    warn = []
    for s in names:
        worst = max(S[s]["topics"].items(), key=lambda kv: kv[1]["to_mastery"])
        if worst[1]["to_mastery"] >= 12:
            warn.append({"student": s, "topic": worst[0], "to_mastery": worst[1]["to_mastery"], "group": group_of[s],
                         "why": FEAT_LABEL[max(FEATS, key=lambda f: abs(Z[names.index(s), FEATS.index(f)]))]})
    warn.sort(key=lambda w: -w["to_mastery"])

    # curriculum: per-topic class learn rate
    tl = defaultdict(list)
    for s in names:
        for t, v in S[s]["fit"]["topic_learn"].items(): tl[t].append(v)
    topic_learn = {t: median(v) for t, v in tl.items()}
    return {"feats": list(FEATS), "students": S, "names": names, "X": X, "Z": Z, "pct": pct, "groups": groups, "group_of": group_of,
            "silhouette": sil, "k": k, "items": item_rows, "moves": moves, "misconceptions": misc, "warnings": warn,
            "topic_learn": topic_learn, "norms": norms}


def cluster_notes(notes: Counter, thr=0.4):
    tok = lambda s: set(re.findall(r"[a-z()|]+", s.lower())) - {"the", "a", "of", "as", "to", "and", "with", "uses", "is"}
    clusters = []
    for (t, n), c in notes.most_common():
        for cl in clusters:
            a, b = tok(n), tok(cl["label"])
            if cl["topic"] == t and len(a & b) / max(1, len(a | b)) >= thr:
                cl["count"] += c; cl["variants"].append(n); break
        else:
            clusters.append({"topic": t, "label": n, "count": c, "variants": [n]})
    return sorted(clusters, key=lambda c: -c["count"])
