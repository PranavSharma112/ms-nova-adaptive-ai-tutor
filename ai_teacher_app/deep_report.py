"""
deep_report.py
--------------
Deep STUDENT report (HTML, inline SVG, offline). Built on deep_engine.deep_class so
every student is described relative to the class model.
"""
from __future__ import annotations
import math, time
from statistics import median
from typing import Dict, Optional

import deep_engine as de
import llm_agent as ag
from report_builder import CSS, C, E, _page, _kpis, svg_hbars, svg_mastery_line

EXTRA_CSS = f"""
.hero{{background:{C['navy']};color:#fff;padding:18px 22px;margin:14px 0}} .hero b{{font-size:22px}} .hero p{{margin:6px 0 0;color:#D6E2F0}}
.why{{font-size:12.5px;color:{C['grey']};margin:4px 0 10px}} .tl td{{font-size:12px;padding:4px 6px}}
.pill{{display:inline-block;padding:1px 7px;border-radius:9px;color:#fff;font-size:11px;font-weight:700}}
"""


# ------------------------------------------------------------------ charts
def svg_radar(student_pct: Dict[str, float], labels: Dict[str, str], w=440, h=360):
    feats = list(labels); n = len(feats); cx, cy, R = w / 2, h / 2 + 5, 125
    pt = lambda i, r: (cx + R * r * math.sin(2 * math.pi * i / n), cy - R * r * math.cos(2 * math.pi * i / n))
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for ring in (.25, .5, .75, 1):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, ring) for i in range(n)))
        out.append(f'<polygon points="{pts}" fill="none" stroke="{C["line"]}"/>')
    med = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, .5) for i in range(n)))
    out.append(f'<polygon points="{med}" fill="none" stroke="{C["grey"]}" stroke-dasharray="4 3"/>')
    sp = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, max(.04, student_pct[f])) for i, f in enumerate(feats)))
    out.append(f'<polygon points="{sp}" fill="{C["blue"]}" fill-opacity="0.25" stroke="{C["blue"]}" stroke-width="2"/>')
    for i, f in enumerate(feats):
        x, y = pt(i, 1.2); v = student_pct[f]
        out.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="11.5" fill="#1B2433">{E(labels[f])}</text>'
                   f'<text x="{x:.1f}" y="{y+13:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="{C["blue"]}">{v:.0%}</text>')
    out.append(f'<text x="8" y="{h-6}" font-size="10" fill="{C["grey"]}">percentile in class · dashed = class median</text>')
    return "".join(out) + "</svg>"


def svg_waterfall(steps, w=760, h=230):
    """steps: [(label, value 0-1, kind)] kind: base | add | total"""
    L, B, T = 50, 40, 15; n = len(steps); bw = (w - L - 20) / n
    Y = lambda v: T + (h - T - B) * (1 - v)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for v in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{L}" x2="{w-10}" y1="{Y(v)}" y2="{Y(v)}" stroke="{C["line"]}" stroke-width=".6"/>'
                   f'<text x="{L-6}" y="{Y(v)+4}" font-size="10" text-anchor="end" fill="{C["grey"]}">{v:.0%}</text>')
    run = 0
    for i, (lab, v, kind) in enumerate(steps):
        x = L + i * bw + bw * .15
        if kind == "add": lo, hi, col = run, run + v, C["teal"]; run += v
        else: lo, hi, col = 0, v, C["navy"] if kind == "base" else C["blue"]; run = v
        out.append(f'<rect x="{x:.1f}" y="{Y(hi):.1f}" width="{bw*.7:.1f}" height="{max(1, Y(lo)-Y(hi)):.1f}" fill="{col}"/>'
                   f'<text x="{x+bw*.35:.1f}" y="{Y(hi)-5:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="{col}">'
                   f'{"+" if kind=="add" else ""}{v:.0%}</text>'
                   f'<text x="{x+bw*.35:.1f}" y="{h-B+16}" text-anchor="middle" font-size="11" fill="#1B2433">{E(lab)}</text>')
    return "".join(out) + "</svg>"


def svg_compare_bars(rows, w=760, label_w=190, maxv=0.45):
    """rows: [(label, student, class)]"""
    h = 34 * len(rows) + 30; bw = w - label_w - 80
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i, (lab, s, c) in enumerate(rows):
        y = 10 + i * 34; col = C["red"] if s < .5 * c else C["amber"] if s < .85 * c else C["teal"]
        out.append(f'<text x="{label_w-8}" y="{y+16}" text-anchor="end" font-size="12">{E(lab)}</text>'
                   f'<rect x="{label_w}" y="{y+2}" width="{bw*min(1,s/maxv):.1f}" height="18" fill="{col}"/>'
                   f'<line x1="{label_w+bw*min(1,c/maxv):.1f}" x2="{label_w+bw*min(1,c/maxv):.1f}" y1="{y-2}" y2="{y+24}" stroke="{C["navy"]}" stroke-width="2.5"/>'
                   f'<text x="{label_w+bw+8}" y="{y+16}" font-size="12" font-weight="700" fill="{col}">{s:.2f}</text>')
    out.append(f'<text x="{label_w}" y="{h-6}" font-size="10" fill="{C["grey"]}">bar = your learning rate per question · navy line = class median</text>')
    return "".join(out) + "</svg>"


# ------------------------------------------------------------------ actions ranked by marks
def ranked_actions(r: Dict, group: Dict, D: Dict):
    cf, n, acts = r["cf"], r["n"], []
    if cf["slips"]:
        acts.append((cf["slips"] / n, "Check before you submit",
                     f"{cf['slips']} answers were wrong although you already knew the topic. A 20-second check "
                     f"(re-substitute, estimate the size of the answer) recovers about {cf['slips']/n:.0%} of all marks.",
                     "Today: on the next 5 questions, write your estimate first, then compare."))
    if cf["rapid"]:
        g = cf["rapid"] * max(0, cf["effort_adjusted"] - 0.2) / n
        acts.append((g, "Read before answering",
                     f"{cf['rapid']} answers came faster than reading time and were mostly wrong. When you take your time you score "
                     f"{cf['effort_adjusted']:.0%}; that gap is worth about {g:.0%} of all marks.",
                     "Today: say or write what the question asks before answering."))
    h = r["hint"]
    if h["transfer_after_hint"] is not None and h["after_no_hint"] is not None and h["n_after_hint"] >= 3 \
            and h["transfer_after_hint"] < h["after_no_hint"] - 0.1:
        g = (h["after_no_hint"] - h["transfer_after_hint"]) * h["n_after_hint"] / n
        acts.append((g, "Make hints stick",
                     f"After a hinted question your next unaided answer is right {h['transfer_after_hint']:.0%} of the time, "
                     f"vs {h['after_no_hint']:.0%} otherwise: the hint solved that question but did not teach the step.",
                     "Today: after using a hint, redo the same type of question without it."))
    if cf.get("sure_wrong"):
        acts.append((cf["sure_wrong"] / n, "Doubt your 'sure' answers",
                     f"{cf['sure_wrong']} answers were marked Sure but were wrong. Confident errors are the ones you never revisit.",
                     "Today: for each 'sure but wrong' item, write why the right answer is right in one line."))
    notes = {}
    for t, T in r["topics"].items():
        for m in T["notes"]: notes[(t, m)] = notes.get((t, m), 0) + 1
    if notes:
        (t, m), k = max(notes.items(), key=lambda kv: kv[1])
        acts.append((k / n, f"Fix one wrong rule: {m}",
                     f"This exact mistake appeared {k} times in {t}. It is a rule problem, so more practice alone repeats it.",
                     f"Today: write the wrong rule and the right rule side by side, then solve one {t} question each way."))
    worst = max(r["topics"].items(), key=lambda kv: kv[1]["to_mastery"])
    if worst[1]["to_mastery"] > 0:
        acts.append((0.0, f"Finish {worst[0]}",
                     f"Mastery is {worst[1]['end']:.2f}; your personal model predicts about {worst[1]['to_mastery']:.0f} more level-3 "
                     f"questions to reach 0.90.", f"This week: {min(10, round(worst[1]['to_mastery']))} questions in 2-3 short sessions."))
    return sorted(acts, key=lambda a: -a[0])


# ------------------------------------------------------------------ page
def student_page(D: Dict, name: str, llm=None) -> str:
    r = D["students"][name]; fit = r["fit"]; cf = r["cf"]; group = next(g for g in D["groups"] if name in g["members"])
    cl_learn = median(D["students"][s]["fit"]["learn"] for s in D["names"])
    cl_slip = median(D["students"][s]["fit"]["slip_scale"] for s in D["names"])
    body = [f'<h1>Deep learning report: {E(name)}</h1><div class="sub">Generated {time.strftime("%d %b %Y")} · {r["n"]} answers · '
            f'{len(r["topics"])} topics · compared with {len(D["names"])} classmates</div>']

    # hero: who you are as a learner
    speed = fit["learn"] / cl_learn; care = fit["slip_scale"] / cl_slip
    form = "time_ratio" not in D["feats"] and "rapid_share" not in D["feats"]
    if form:   # WHY: BKT parameters are not identifiable from 2 form snapshots (forms_sim test); use observed rates
        cl_acc = median(D["students"][x]["cf"]["actual"] for x in D["names"])
        conf = [(x["conf"], x["c"]) for x in r["timeline"] if x["conf"] is not None]
        sure = [c for v, c in conf if v == 1.0]
        body.append(f'<div class="hero"><b>Learner profile: {E(group["name"])}</b><p>Accuracy <b>{cf["actual"]:.0%}</b> (class median {cl_acc:.0%}). '
                    f'When you said <b>Sure</b> you were right <b>{(sum(sure)/len(sure) if sure else 0):.0%}</b> of the time; '
                    f'<b>{cf.get("sure_wrong") or 0}</b> answers were sure but wrong. Main driver of your profile: <b>{E(group["driver"])}</b>.</p></div>')
    else:
      body.append(f'<div class="hero"><b>Learner profile: {E(group["name"])}</b><p>You learn <b>{speed:.1f}×</b> the class median per question '
                  f'and make execution slips <b>{care:.1f}×</b> as often. Effort-adjusted accuracy is <b>{cf["effort_adjusted"]:.0%}</b> '
                  f'(raw {cf["actual"]:.0%}). Main driver of your profile: <b>{E(group["driver"])}</b>.</p></div>')
    body.append(_kpis([(r["n"], "answers", C["navy"]), (f'{cf["actual"]:.0%}', "raw accuracy", C["red"]),
                       (f'{cf["if_slips_caught"]:.0%}', "if slips were caught", C["teal"]),
                       (sum(1 for t in r["topics"].values() if t["end"] >= .9), "topics mastered", C["blue"]),
                       (f'{sum(t["to_mastery"] for t in r["topics"].values()):.0f}', "questions left to master all", C["amber"])]))

    facts = {"name": name, "profile": group["name"], "speed_vs_class": round(speed, 1), "slips_vs_class": round(care, 1),
             "accuracy": round(cf["actual"], 2), "accuracy_if_slips_caught": round(cf["if_slips_caught"], 2),
             "effort_adjusted": round(cf["effort_adjusted"], 2),
             "topics": {t: {"mastery": round(T["end"], 2), "questions_to_mastery": round(T["to_mastery"])} for t, T in r["topics"].items()}}
    note = ag.coach_note(llm, facts)
    if note: body.append(f'<div class="card" style="border-color:{C["purple"]}"><h3>Coach note (AI agent, facts-checked)</h3><p>{E(note)}</p></div>')

    acts = ranked_actions(r, group, D)
    if False: pass
    body.append("<h2>Your top actions, ranked by marks they recover</h2><div class='why'>Each gain is computed from your own answers, not a general tip.</div><div class='todo'>")
    for g, title, why, today in acts[:4]:
        # WATCH: confident errors are not automatically recoverable marks; label them as answers affected.
        lab = f"{g:.0%} of answers affected" if title.startswith("Doubt") else f"+{g:.0%} marks"
        tag = f'<span class="pill" style="background:{C["teal"]}">{lab}</span> ' if g > 0 else ""
        body.append(f'<div class="act">{tag}<b>{E(title)}</b><br>{E(why)}<br><i>{E(today)}</i></div>')
    body.append("</div>")

    body.append(f'<div class="grid2"><div><h2>Learner fingerprint</h2><div class="why">{len(D["feats"])} behaviours as your percentile in the class.</div>')
    body.append(svg_radar(D["pct"][name], {f: de.FEAT_LABEL[f] for f in D["feats"]}))
    body.append('</div><div><h2>Your personal learning model</h2><div class="why">Fitted to your answers by maximum likelihood (BKT). Rapid guesses removed first.'
                + (' <b>Low confidence: quiz data has only a few snapshots; use the observed rates above.</b>' if form else '') + '</div>')
    body.append(f'<table><tr><th>Parameter</th><th>You</th><th>Class</th><th>Meaning</th></tr>'
                f'<tr><td>Learning rate</td><td><b>{fit["learn"]:.2f}</b></td><td>{cl_learn:.2f}</td><td>chance to learn per question</td></tr>'
                f'<tr><td>Slip factor</td><td><b>{fit["slip_scale"]:.1f}×</b></td><td>{cl_slip:.1f}×</td><td>wrong although known</td></tr>'
                f'<tr><td>Guess factor</td><td><b>{fit["guess_scale"]:.1f}×</b></td><td>1.0×</td><td>right although not known</td></tr>'
                f'<tr><td>Start knowledge</td><td><b>{fit["p_init"]:.2f}</b></td><td>-</td><td>before the first question</td></tr></table>')
    body.append("</div></div>")

    body.append("<h2>Where your marks went</h2><div class='why'>Raw accuracy, plus what each fixable habit is worth.</div>")
    rapid_gain = cf["rapid"] * max(0, cf["effort_adjusted"] - .2) / r["n"]; slip_gain = cf["slips"] / r["n"]
    body.append(svg_waterfall([("Raw accuracy", cf["actual"], "base"), ("+ catch slips", slip_gain, "add"),
                               ("+ no rapid guesses", rapid_gain, "add"), ("Reachable now", cf["actual"] + slip_gain + rapid_gain, "total")]))

    body.append("<h2>Topic forecast</h2><div class='why'>Questions still needed to reach mastery 0.90, simulated 300 times with your fitted model.</div>")
    tf = sorted(r["topics"].items(), key=lambda kv: -kv[1]["to_mastery"])
    body.append(svg_hbars([(t, T["to_mastery"], C["teal"] if T["to_mastery"] == 0 else C["amber"] if T["to_mastery"] < 10 else C["red"])
                           for t, T in tf], maxv=max(10, max(T["to_mastery"] for _, T in tf)), fmt="{:.0f} q", w=760, label_w=190))
    body.append("<h2>How fast you learn each topic</h2><div class='why'>Per-topic learning rate (low confidence: about 9 answers per topic). "
                "A topic far below the class line usually means a missing prerequisite.</div>")
    body.append(svg_compare_bars([(t, T["learn"] or 0, D["topic_learn"].get(t, cl_learn)) for t, T in r["topics"].items()]))

    body.append("<h2>Mastery journey</h2>")
    body.append(svg_mastery_line([{"i": x["i"], "topic": x["topic"], "p": x["pa"], "correct": x["c"], "rapid": x["rapid"], "d": x["d"]} for x in r["timeline"]]))

    h = r["hint"]
    if h["hinted_q"] and h["n_after_hint"] < 3:
        body.append(f'<h2>Do hints help you?</h2><p>You used hints on <b>{h["hinted_q"]}</b> questions: too few follow-ups '
                    f'(n = {h["n_after_hint"]}) to judge whether hints transfer. Not scored.</p>')
    elif h["hinted_q"]:
        body.append(f'<h2>Do hints help you?</h2><p>You used hints on <b>{h["hinted_q"]}</b> questions. Next unaided answer correct: '
                    f'<b>{(h["transfer_after_hint"] or 0):.0%}</b> after a hint vs <b>{(h["after_no_hint"] or 0):.0%}</b> after no hint '
                    f'(n = {h["n_after_hint"]}).</p>')

    body.append("<h2>Every answer, explained</h2><table class='tl'><tr><th>#</th><th>Topic</th><th>Lvl</th><th>Result</th><th>Time / confidence</th><th>Hints</th><th>Mastery</th><th>Likely cause</th></tr>")
    for x in r["timeline"]:
        col = C["amber"] if x["rapid"] else C["teal"] if x["c"] else C["red"]
        body.append(f'<tr><td>{x["i"]}</td><td>{E(x["topic"])}</td><td>{x["d"]}</td><td><span class="pill" style="background:{col}">'
                    f'{"right" if x["c"] else "wrong"}</span></td><td>{(f"{x['t']:.0f}s" if x["t"] else ({1.0:"sure",0.5:"unsure",0.0:"guess"}.get(x["conf"], "–")))}</td><td>{x["h"]}</td>'
                    f'<td>{x["pb"]:.2f} → {x["pa"]:.2f}</td><td>{E(x["cause"])}</td></tr>')
    body.append("</table>")
    body.append(f'<div class="card" style="border-color:{C["navy"]}"><h3>What your teacher will do with your group ({E(group["name"])})</h3><p>{E(group["plan"])}</p></div>')
    body.append('<div class="note">Model: per-student Bayesian Knowledge Tracing fitted by maximum likelihood; rapid guesses (Wise &amp; Ma, 2012) removed '
                'before fitting. Forecasts are model estimates. Groups: k-means on 7 behaviour features. AI text, if present, is checked so it '
                'contains no number absent from the computed facts.</div>')
    return _page(f"Deep report {name}", "".join(body)).replace("</style>", EXTRA_CSS + "</style>")


# ================================================================== TEACHER PAGE
def teacher_page(D: Dict, events, title="Class", llm=None) -> str:
    from collections import Counter, defaultdict
    from report_builder import svg_heatmap
    S, names = D["students"], D["names"]
    form = not any(e.get("elapsed") is not None for e in events)
    accs = [S[s]["cf"]["actual"] for s in names]
    body = [f'<h1>Teacher insight report: {E(title)}</h1><div class="sub">Generated {time.strftime("%d %b %Y")} · {len(names)} students · '
            f'{sum(S[s]["n"] for s in names)} answers · source: {"Google Form" if form else "Ms. Nova tutor"}</div>']
    misc = D["misconceptions"]; named = ag.name_misconceptions(llm, misc)
    biggest = max(D["groups"], key=lambda g: len(g["members"]))
    hardest = min(D["topic_learn"] or {"-": 0}, key=lambda t: D["topic_learn"][t]) if D["topic_learn"] else "-"
    tacc = defaultdict(list)
    for e in events:
        if e.get("kind") != "worked_example": tacc[e["topic"]].append(int(bool(e["correct"])))
    weakest = min(tacc, key=lambda t: sum(tacc[t]) / len(tacc[t]))
    body.append(f'<div class="hero"><b>{len(D["groups"])} instruction groups found.</b><p>Largest: <b>{E(biggest["name"])}</b> '
                f'({len(biggest["members"])} students). Weakest topic: <b>{E(weakest)}</b> ({sum(tacc[weakest])/len(tacc[weakest]):.0%} correct). '
                f'Top misconception: <b>{E(misc[0]["label"]) if misc else "none"}</b> ({misc[0]["count"] if misc else 0} times).</p></div>')
    sure_wrong = sum((S[s]["cf"].get("sure_wrong") or 0) for s in names)
    body.append(_kpis([(len(names), "students", C["navy"]), (f"{sum(accs)/len(accs):.0%}", "mean accuracy", C["teal"]),
                       (len(D["warnings"]) if not form else sum(1 for s in names if S[s]["cf"]["actual"] < .4), "students at risk", C["red"]),
                       (sure_wrong if form else len([i for i in D["items"] if i["verdict"].startswith("revise")]),
                        "'sure but wrong' answers" if form else "items to revise", C["amber"]),
                       (len(misc), "distinct misconceptions", C["purple"])]))
    brief = ag.teacher_brief(llm, {"groups": {g["name"]: len(g["members"]) for g in D["groups"]}, "weakest_topic": weakest,
                                   "top_misconceptions": [(m["label"], m["count"]) for m in misc[:3]]})
    if brief: body.append(f'<div class="card" style="border-color:{C["purple"]}"><h3>AI brief (facts-checked)</h3><p>{E(brief)}</p></div>')

    body.append("<h2>Instruction groups: who to teach together, and how</h2><div class='why'>k-means on each student's behaviour "
                f"fingerprint ({', '.join(de.FEAT_LABEL[f] for f in D['feats'])}); silhouette {D['silhouette']:.2f}.</div>")
    for g in sorted(D["groups"], key=lambda g: -len(g["members"])):
        ga = sum(S[m]["cf"]["actual"] for m in g["members"]) / len(g["members"])
        body.append(f'<div class="card" style="border-color:{C["blue"]}"><h3>{E(g["name"])}<span class="tag" style="background:{C["grey"]}">'
                    f'{len(g["members"])} students · {ga:.0%} correct</span></h3><div class="sub"><b>Defining trait:</b> {E(g["driver"])} · '
                    f'<b>Members:</b> {E(", ".join(g["members"]))}</div><div class="act" style="margin-top:6px"><b>Lesson plan:</b> {E(g["plan"])}</div></div>')

    body.append("<h2>Mastery map</h2><div class='why'>Final mastery estimate per student and topic, sorted by group.</div>")
    order = [m for g in D["groups"] for m in g["members"]]; topics = list(tacc)
    body.append(svg_heatmap(order, topics, lambda s, t: S[s]["topics"][t]["end"] if t in S[s]["topics"] else None))

    body.append("<h2>Misconceptions across the class</h2><table><tr><th>Topic</th><th>Misconception</th><th>Times</th><th>Students</th><th>Fix</th></tr>")
    who = defaultdict(set)
    for e in events:
        if e.get("misconception_note"): who[(e["topic"], e["misconception_note"].strip())].add(e.get("student"))
    for m in misc[:12]:
        n = named.get(m["label"], {}); fix = n.get("fix") or "Contrasting example: solve once with this wrong rule and once correctly; ask which passes a sanity check. (Run with --agent for an item-specific fix.)"
        body.append(f'<tr><td>{E(m["topic"])}</td><td><b>{E(n.get("name") or m["label"])}</b></td><td>{m["count"]}</td>'
                    f'<td>{len(who[(m["topic"], m["label"])])}</td><td>{E(fix)}</td></tr>')
    body.append("</table>")

    if form:
        body.append("<h2>Answer-option analysis</h2><div class='why'>What share of the class chose each option. A distractor chosen by more "
                    "students than the key means the class holds that misconception, or the item is ambiguous.</div><table><tr><th>Item</th>"
                    "<th>Topic</th><th>Correct</th><th>Most chosen wrong option</th><th>Share</th><th>Verdict</th></tr>")
        it = defaultdict(list)
        for e in events: it[e["item_id"]].append(e)
        rows = []
        for iid, ev in it.items():
            p = sum(int(bool(x["correct"])) for x in ev) / len(ev)
            wrong = Counter((x.get("answer_given"), x.get("misconception_note")) for x in ev if not x["correct"] and x.get("answer_given"))
            (opt, note), k = wrong.most_common(1)[0] if wrong else ((None, ""), 0)
            share = k / len(ev)
            verdict = "re-teach now" if share >= p else ("watch" if share >= .25 else "ok")
            rows.append((share - p, iid, ev[0]["topic"], p, f"{(opt or '-').upper()}: {note or 'unmapped / slip'}", share, verdict))
        for _, iid, t, p, o, sh, v in sorted(rows, reverse=True)[:15]:
            col = C["red"] if v == "re-teach now" else C["amber"] if v == "watch" else C["teal"]
            body.append(f'<tr><td>{E(iid)}</td><td>{E(t)}</td><td>{p:.0%}</td><td>{E(o)}</td><td>{sh:.0%}</td>'
                        f'<td><span class="pill" style="background:{col}">{v}</span></td></tr>')
        body.append("</table>")
        body.append("<h2>Confidence vs correctness</h2><div class='why'>'Sure but wrong' is the most important cell: the student holds a "
                    "wrong rule confidently and will not seek help.</div>")
        cells = Counter()
        for e in events:
            if e.get("confidence") is not None: cells[({1.0: "Sure", .5: "Not sure", 0.0: "Guessing"}[e["confidence"]], bool(e["correct"]))] += 1
        tot = sum(cells.values()) or 1
        body.append(svg_hbars([(f"{c} · {'right' if k else 'wrong'}", cells[(c, k)] / tot,
                                C["red"] if (c == "Sure" and not k) else C["teal"] if (c == "Sure" and k) else C["amber"])
                               for c in ("Sure", "Not sure", "Guessing") for k in (True, False)], w=760, label_w=190))
        top_sw = sorted(names, key=lambda s: -(S[s]["cf"].get("sure_wrong") or 0))[:6]
        body.append("<p><b>Most 'sure but wrong':</b> " + E(", ".join(f"{s} ({S[s]['cf']['sure_wrong']})" for s in top_sw)) + "</p>")
    else:
        body.append("<h2>Early warning</h2><table><tr><th>Student</th><th>Topic</th><th>Questions to mastery</th><th>Group</th><th>Main driver</th></tr>")
        body.append("".join(f'<tr><td>{E(w["student"])}</td><td>{E(w["topic"])}</td><td><b>{w["to_mastery"]:.0f}</b></td><td>{E(w["group"])}</td>'
                            f'<td>{E(w["why"])}</td></tr>' for w in D["warnings"]) + "</table>")
        mv = D["moves"]
        if mv["worked_example"]["n"]:
            body.append(f'<h2>Which teaching move works here</h2><p>Next answer correct after a <b>worked example</b>: '
                        f'<b>{mv["worked_example"]["next_correct"]:.0%}</b> (n={mv["worked_example"]["n"]}) vs after a plain retry: '
                        f'<b>{mv["plain_retry"]["next_correct"]:.0%}</b> (n={mv["plain_retry"]["n"]}). Observational, not a controlled trial.</p>')

    body.append("<h2>Student roster</h2><table><tr><th>Student</th><th>Group</th><th>Accuracy</th><th>Weakest topic</th><th>First action</th></tr>")
    for s in sorted(names, key=lambda s: S[s]["cf"]["actual"]):
        r = S[s]; wt = min(r["topics"], key=lambda t: r["topics"][t]["end"])
        acts = ranked_actions(r, None, D); a0 = acts[0][1] if acts else "-"
        body.append(f'<tr><td>{E(s)}</td><td>{E(D["group_of"][s])}</td><td>{r["cf"]["actual"]:.0%}</td><td>{E(wt)}</td><td>{E(a0)}</td></tr>')
    body.append("</table>")
    body.append('<div class="note">Groups: k-means on standardised behaviour features, k chosen by silhouette. Mastery: BKT replayed in answer order. '
                'Misconceptions come from mapped wrong options (forms) or the grader (tutor). AI text is optional and number-checked. '
                'Validated on simulated cohorts with hidden profiles; confirm decisions with the student.</div>')
    return _page(f"Teacher report {title}", "".join(body)).replace("</style>", EXTRA_CSS + "</style>")
