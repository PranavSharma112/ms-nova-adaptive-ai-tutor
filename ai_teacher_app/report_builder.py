"""
report_builder.py
-----------------
Renders insight_engine output as two self-contained HTML reports (print to PDF from
the browser): a STUDENT report and a TEACHER (class) report. Charts are inline SVG,
so the files need no internet, no JS and no extra packages.

Optional: `narrative=True` asks Gemini to write a 3-sentence summary from the
computed facts only. Flags, numbers and actions always come from insight_engine.
"""
from __future__ import annotations
import html, time
from typing import Dict, List, Optional

import insight_engine as ie

C = {"navy": "#1F3A5F", "blue": "#1F6FB2", "teal": "#12A089", "amber": "#E08A1E", "red": "#C0392B",
     "purple": "#7B5EA7", "grey": "#5A6B7D", "soft": "#F3F5F8", "line": "#D5DBE3"}
SEV_COLOR = {"high": C["red"], "medium": C["amber"], "low": C["blue"]}
E = html.escape


# ------------------------------------------------------------------ SVG primitives
def svg_hbars(rows: List[tuple], maxv: float = 1.0, fmt="{:.0%}", w=560, color=C["blue"], label_w=170) -> str:
    """rows: [(label, value, optional_color)]"""
    h = 28 * len(rows) + 10; bw = w - label_w - 70
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i, r in enumerate(rows):
        lab, v = r[0], r[1]; col = r[2] if len(r) > 2 else color; y = 8 + i * 28
        out.append(f'<text x="{label_w-8}" y="{y+15}" text-anchor="end" font-size="12" fill="#1B2433">{E(str(lab)[:28])}</text>')
        out.append(f'<rect x="{label_w}" y="{y}" width="{bw}" height="20" fill="{C["soft"]}"/>')
        out.append(f'<rect x="{label_w}" y="{y}" width="{max(1, bw*min(1, v/maxv)):.1f}" height="20" fill="{col}"/>')
        out.append(f'<text x="{label_w+bw+6}" y="{y+15}" font-size="12" font-weight="700" fill="{col}">{fmt.format(v)}</text>')
    return "".join(out) + "</svg>"


def svg_mastery_line(tl: List[Dict], w=760, h=260) -> str:
    if not tl: return ""
    L, R, T, B = 40, 10, 15, 35; n = len(tl)
    X = lambda i: L + (w - L - R) * (i / max(1, n - 1)); Y = lambda p: T + (h - T - B) * (1 - p)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for p in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{L}" x2="{w-R}" y1="{Y(p)}" y2="{Y(p)}" stroke="{C["line"]}" stroke-width="0.6"/>'
                   f'<text x="{L-6}" y="{Y(p)+4}" font-size="10" text-anchor="end" fill="{C["grey"]}">{p:.2f}</text>')
    out.append(f'<line x1="{L}" x2="{w-R}" y1="{Y(.9)}" y2="{Y(.9)}" stroke="{C["teal"]}" stroke-dasharray="5 4"/>'
               f'<text x="{w-R}" y="{Y(.9)-4}" font-size="10" text-anchor="end" fill="{C["teal"]}">mastery 0.90</text>')
    prev_topic, seg = None, []
    for i, e in enumerate(tl):                     # topic boundaries
        if e["topic"] != prev_topic:
            out.append(f'<line x1="{X(i)}" x2="{X(i)}" y1="{T}" y2="{h-B}" stroke="{C["line"]}"/>'
                       f'<text x="{X(i)+3}" y="{h-B+14}" font-size="10" fill="{C["grey"]}">{E(str(e["topic"])[:22])}</text>')
            if len(seg) > 1: out.append(f'<polyline fill="none" stroke="{C["navy"]}" stroke-width="2" points="{" ".join(seg)}"/>')
            seg, prev_topic = [], e["topic"]
        seg.append(f"{X(i):.1f},{Y(e['p']):.1f}")
    if len(seg) > 1: out.append(f'<polyline fill="none" stroke="{C["navy"]}" stroke-width="2" points="{" ".join(seg)}"/>')
    for i, e in enumerate(tl):
        col = C["teal"] if e["correct"] else C["red"]
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(e["p"]):.1f}" r="4.5" fill="{col}"><title>Q{e["i"]} L{e["d"]} {"✔" if e["correct"] else "✘"} p={e["p"]:.2f}</title></circle>')
        if e["rapid"]: out.append(f'<circle cx="{X(i):.1f}" cy="{Y(e["p"]):.1f}" r="8" fill="none" stroke="{C["amber"]}" stroke-width="2"/>')
    out.append(f'<text x="{L}" y="{h-4}" font-size="10" fill="{C["grey"]}">question number →   ● correct  ● wrong  ◯ rapid guess</text>')
    return "".join(out) + "</svg>"


def svg_time_scatter(tl: List[Dict], thr: Dict, slow_thr: Dict, w=760, h=230) -> str:
    if not tl: return ""
    L, R, T, B = 45, 10, 12, 30; n = len(tl); tmax = max(30, max(e["t"] for e in tl) * 1.1)
    X = lambda i: L + (w - L - R) * (i / max(1, n - 1)); Y = lambda t: T + (h - T - B) * (1 - t / tmax)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for t in range(0, int(tmax) + 1, max(10, int(tmax / 5) // 10 * 10)):
        out.append(f'<line x1="{L}" x2="{w-R}" y1="{Y(t)}" y2="{Y(t)}" stroke="{C["line"]}" stroke-width="0.6"/>'
                   f'<text x="{L-6}" y="{Y(t)+4}" font-size="10" text-anchor="end" fill="{C["grey"]}">{t}s</text>')
    for i, e in enumerate(tl):
        slow = e["correct"] and e["t"] > slow_thr.get(e["d"], 1e9)
        col = C["amber"] if e["rapid"] else (C["purple"] if slow else (C["teal"] if e["correct"] else C["red"]))
        out.append(f'<rect x="{X(i)-4:.1f}" y="{Y(e["t"]):.1f}" width="8" height="{max(1, Y(0)-Y(e["t"])):.1f}" fill="{col}" opacity="0.85"/>')
        x0 = X(i) - 6; x1 = X(i) + 6
        out.append(f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{Y(thr.get(e["d"], 0)):.1f}" y2="{Y(thr.get(e["d"], 0)):.1f}" stroke="{C["amber"]}" stroke-width="1.5"/>')
    out.append(f'<text x="{L}" y="{h-6}" font-size="10" fill="{C["grey"]}">bars = seconds per answer · amber tick = rapid-guess threshold · green correct · red wrong · amber rapid · purple correct but slow</text>')
    return "".join(out) + "</svg>"


def svg_heatmap(rows: List[str], cols: List[str], val, w=760) -> str:
    cw = (w - 150) / max(1, len(cols)); rh = 30; h = 45 + rh * len(rows)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for j, c in enumerate(cols):
        out.append(f'<text x="{150+cw*j+cw/2:.1f}" y="30" text-anchor="middle" font-size="11" fill="{C["grey"]}">{E(c[:18])}</text>')
    for i, r in enumerate(rows):
        y = 40 + i * rh
        out.append(f'<text x="142" y="{y+19}" text-anchor="end" font-size="12" fill="#1B2433">{E(r[:20])}</text>')
        for j, c in enumerate(cols):
            v = val(r, c)
            if v is None: fill, txt = "#FFFFFF", "–"
            else:
                fill = C["red"] if v < .45 else C["amber"] if v < .7 else C["blue"] if v < .9 else C["teal"]; txt = f"{v:.2f}"
            out.append(f'<rect x="{150+cw*j+2:.1f}" y="{y}" width="{cw-4:.1f}" height="{rh-4}" rx="4" fill="{fill}" stroke="{C["line"]}"/>'
                       f'<text x="{150+cw*j+cw/2:.1f}" y="{y+18}" text-anchor="middle" font-size="12" font-weight="700" fill="{"#1B2433" if v is None else "#fff"}">{txt}</text>')
    return "".join(out) + "</svg>"


# ------------------------------------------------------------------ page shell
CSS = f"""
*{{box-sizing:border-box}} body{{font-family:Calibri,'Segoe UI',Arial,sans-serif;color:#1B2433;margin:0;background:#fff}}
.wrap{{max-width:980px;margin:0 auto;padding:28px 34px}} h1{{color:{C['navy']};margin:0 0 4px;font-size:28px}}
h2{{color:{C['navy']};border-bottom:3px solid {C['navy']};padding-bottom:4px;margin:30px 0 12px;font-size:20px}}
.sub{{color:{C['grey']};font-size:13px}} .kpis{{margin:18px 0;font-size:0}}
.kpi{{display:inline-block;vertical-align:top;width:19%;margin-right:1%;background:{C['soft']};border-left:5px solid {C['blue']};padding:10px 12px;font-size:14px}} .kpi b{{font-size:24px;display:block}}
.kpi span{{font-size:12px;color:{C['grey']}}} .card{{background:{C['soft']};border-left:6px solid {C['blue']};padding:12px 16px;margin:10px 0;page-break-inside:avoid}}
.card h3{{margin:0 0 6px;font-size:16px}} .tag{{display:inline-block;color:#fff;border-radius:3px;padding:1px 7px;font-size:11px;font-weight:700;margin-left:6px}}
.grid2{{font-size:0}} .grid2>div{{display:inline-block;vertical-align:top;width:49%;margin-right:1%;font-size:14px}} .act{{background:#fff;border:1px solid {C['line']};padding:8px 10px;font-size:13.5px}}
.act b{{color:{C['navy']}}} table{{border-collapse:collapse;width:100%;font-size:13px}} th{{background:{C['navy']};color:#fff;text-align:left;padding:6px}}
td{{padding:6px;border-bottom:1px solid {C['line']}}} tr:nth-child(even) td{{background:{C['soft']}}}
.todo{{counter-reset:n}} .todo .act{{margin:8px 0;padding-left:44px;position:relative;font-size:14.5px}}
.todo .act:before{{counter-increment:n;content:counter(n);position:absolute;left:10px;top:7px;background:{C['teal']};color:#fff;width:24px;height:24px;border-radius:50%;text-align:center;font-weight:700;line-height:24px}}
.note{{font-size:11.5px;color:{C['grey']};margin-top:30px;border-top:1px solid {C['line']};padding-top:8px}}
@media print{{.wrap{{padding:10px}} h2{{page-break-after:avoid}}}}
"""


def _page(title, body):
    return f'<!doctype html><html><head><meta charset="utf-8"><title>{E(title)}</title><style>{CSS}</style></head><body><div class="wrap">{body}</div></body></html>'


def _kpis(items):
    return '<div class="kpis">' + "".join(f'<div class="kpi" style="border-color:{c}"><b>{v}</b><span>{E(l)}</span></div>' for v, l, c in items) + "</div>"


def _flag_card(f, who="student"):
    col = SEV_COLOR[f["severity"]]
    topics = ", ".join(t for t in f.get("topics", [f["topic"]]) if t != "all") or "whole session"
    ev = "; ".join(f.get("evidence_list", [f["evidence"]]))
    a1 = ("What you do now", f["student_action"]) if who == "student" else ("Teacher move", f["teacher_action"])
    a2 = ("Ask your teacher for", f["teacher_action"]) if who == "student" else ("Change in material", f["material_change"])
    return (f'<div class="card" style="border-color:{col}"><h3>{E(f["title"])}<span class="tag" style="background:{col}">{f["severity"].upper()}</span></h3>'
            f'<div class="sub"><b>Where:</b> {E(topics)} &nbsp;·&nbsp; <b>Evidence:</b> {E(ev)}</div>'
            f'<p style="margin:6px 0">{E(f["meaning"])}</p><div class="grid2"><div class="act"><b>{a1[0]}:</b> {E(a1[1])}</div>'
            f'<div class="act"><b>{a2[0]}:</b> {E(a2[1])}</div></div></div>')


def _narrative(facts: str, audience: str, api_key: Optional[str]) -> str:
    """Optional Gemini summary; facts only, never new numbers. Fails silently to ''."""
    if not api_key: return ""
    try:
        import teacher_engine as te
        if te.is_demo(api_key): return ""
        txt = te._generate_text(f"Write a warm, precise 3-sentence summary for the {audience}. Use ONLY these facts, add no new numbers, "
                                f"no diagnosis words, no emojis.\nFACTS:\n{facts}", api_key)
        return f'<div class="card" style="border-color:{C["purple"]}"><h3>Summary</h3><p>{E(txt)}</p></div>'
    except Exception:
        return ""


# ------------------------------------------------------------------ STUDENT REPORT
def study_plan(a: Dict) -> List[tuple]:
    """7-day plan: weakest topics first, spaced review of strengths (days 2 and 6)."""
    pr = a["summary"]["priorities"] or list(a["topics"])[:1]
    st = a["summary"]["strengths"]
    codes = {f["code"] for f in a["flags"]}
    how = ("Worked example first, then 2 easy questions" if codes & {"wheel_spinning", "early_wheel_risk", "persistent_misconception"}
           else "3 questions at your ceiling level, hints allowed")
    plan = []
    for day in range(1, 8):
        t = pr[(day - 1) % len(pr)]
        rev = st[(day // 2) % len(st)] if st and day in (2, 4, 6) else None
        task = f"{t}: {how} (20 min)"
        if "fluency_gap" in codes and day in (3, 5, 7): task = f"{t}: timed set of 3 easy questions (10 min)"
        plan.append((f"Day {day}", task, f"Review {rev}: 3 flashcards (5 min)" if rev else "–"))
    return plan


def student_report(events: List[Dict], name: str = "Student", norms=None, misconceptions=None,
                   api_key: Optional[str] = None) -> str:
    a = ie.analyze_student(events, norms, misconceptions)
    if a["n"] == 0:
        return _page("Student report", f"<h1>{E(name)}</h1><p>No answered questions yet.</p>")
    s = a["summary"]
    body = [f'<h1>Learning report: {E(name)}</h1><div class="sub">Generated {time.strftime("%d %b %Y %H:%M")} · {s["questions"]} questions · {s["topics"]} topics</div>']
    body.append(_kpis([(s["questions"], "questions answered", C["navy"]), (f'{s["accuracy"]:.0%}', "correct", C["teal"]),
                       (f'{s["topics_mastered"]}/{s["topics"]}', "topics mastered (≥ 0.90)", C["blue"]),
                       (f'{s["hints_per_q"]:.1f}', "hints per question", C["amber"]), (f'{s["mean_time"]:.0f}s', "average time", C["purple"])]))
    body.append(_narrative(f"accuracy {s['accuracy']:.0%}; strengths {s['strengths']}; priorities {s['priorities']}; "
                           f"patterns {[f['title'] for f in a['flags']]}", "student", api_key))

    body.append("<h2>Do these 3 things next</h2><div class='todo'>")
    acts = [f'<b>{E(f["title"])}:</b> {E(f["student_action"])}' for f in a["flags"][:3]]
    if len(acts) < 3 and s["priorities"]: acts.append(f"<b>Next topic:</b> spend your next session on <b>{E(s['priorities'][0])}</b>.")
    if len(acts) < 3: acts.append("<b>Keep it fresh:</b> review your strongest topic again in 3 days.")
    body.append("".join(f'<div class="act">{x}</div>' for x in acts[:3]) + "</div>")

    body.append("<h2>Your mastery journey</h2><div class='sub'>Estimated chance you know each topic (BKT), after every answer.</div>")
    body.append(svg_mastery_line(a["timeline"]))

    body.append("<h2>Topic by topic</h2><table><tr><th>Topic</th><th>Questions</th><th>Correct</th><th>Hints</th><th>Mastery start → end</th><th>Status</th></tr>")
    for t, T in a["topics"].items():
        st = "✔ mastered" if T["mastered"] else ("in progress" if T["end_p"] >= .45 else "needs work")
        body.append(f"<tr><td>{E(t)}</td><td>{T['n']}</td><td>{T['accuracy']:.0%}</td><td>{T['hints']}</td>"
                    f"<td>{T['start_p']:.2f} → <b>{T['end_p']:.2f}</b></td><td>{st}</td></tr>")
    body.append("</table>")

    body.append('<div class="grid2"><div><h2>Accuracy by level</h2>')
    body.append(svg_hbars([(f"Level {d} (n={s['n_by_difficulty'][d]})", v, C["red"] if v < .45 else C["amber"] if v < .7 else C["teal"])
                           for d, v in s["acc_by_difficulty"].items()], w=420, label_w=120))
    body.append("</div><div><h2>Why answers were wrong</h2>")
    em = s["error_mix"]; tot = sum(em.values()) or 1
    body.append(svg_hbars([(k.replace("_", " "), v / tot, C["red"]) for k, v in sorted(em.items(), key=lambda kv: -kv[1])], w=420, label_w=185)
                if em else "<p>No wrong answers.</p>")
    body.append("</div></div>")

    body.append("<h2>Time per answer</h2>" + svg_time_scatter(a["timeline"], a["rapid_thresholds"], a["slow_thresholds"]))

    body.append(f"<h2>Patterns found ({len(a['flags'])})</h2>")
    body.append("".join(_flag_card(f, "student") for f in a["flags"]) or "<p>No problem patterns detected. Keep going.</p>")

    body.append("<h2>Your 7-day plan</h2><table><tr><th>Day</th><th>Main task</th><th>Spaced review</th></tr>")
    body.append("".join(f"<tr><td>{d}</td><td>{E(t)}</td><td>{E(r)}</td></tr>" for d, t, r in study_plan(a)) + "</table>")

    tips = ["Test yourself before re-reading: closing the notes and recalling beats re-reading.",
            "Space it out: 3 short sessions beat 1 long one.",
            "Mix topics in review so you practise choosing the method, not only applying it."]
    if s["strengths"]: body.append(f"<h2>Strengths</h2><p>{E(', '.join(s['strengths']))}. Use these as anchors when learning related topics.</p>")
    body.append("<h2>Study habits that work</h2>" + "".join(f'<div class="act" style="margin:6px 0">{E(t)}</div>' for t in tips))
    if s["misconception_notes"]:
        body.append("<h2>Mistakes noted by the tutor</h2><ul>" + "".join(f"<li>{E(m)}</li>" for m in s["misconception_notes"]) + "</ul>")
    body.append('<div class="note">Mastery = Bayesian Knowledge Tracing estimate, not a grade. Patterns are rule-based on your answer log; '
                'rapid guess = faster than 10% of the typical time for that level (Wise &amp; Ma, 2012); wheel-spinning = 10+ attempts '
                'without mastery (Beck &amp; Gong, 2013).</div>')
    return _page(f"Learning report {name}", "".join(body))


# ------------------------------------------------------------------ TEACHER REPORT
def teacher_report(events: List[Dict], title: str = "Class", student_of=None, api_key: Optional[str] = None) -> str:
    c = ie.analyze_class(events, student_of)
    if not c.get("n_students"):
        return _page("Teacher report", "<h1>Teacher report</h1><p>No events logged yet.</p>")
    S = c["students"]; names = sorted(S, key=lambda s: -next(r["risk"] for r in c["risk"] if r["student"] == s))
    topics = list(c["topics"])
    mean_acc = sum(r["summary"]["accuracy"] for r in S.values()) / len(S)
    body = [f'<h1>Teacher insight report: {E(title)}</h1><div class="sub">Generated {time.strftime("%d %b %Y %H:%M")} · '
            f'{c["n_students"]} students · {c["n_events"]} answers · {len(topics)} topics</div>']
    at_risk = sum(1 for r in c["risk"] if any(f["severity"] == "high" for f in S[r["student"]]["flags"]))
    body.append(_kpis([(c["n_students"], "students", C["navy"]), (f"{mean_acc:.0%}", "mean accuracy", C["teal"]),
                       (at_risk, "students with a HIGH flag", C["red"]), (len(c["item_flags"]), "questions to fix", C["amber"]),
                       (f"{sum(T['mastery_rate'] for T in c['topics'].values())/len(topics):.0%}", "topic mastery rate", C["blue"])]))
    prev = sorted(c["prevalence"].items(), key=lambda kv: -kv[1]["students"])
    body.append(_narrative(f"{c['n_students']} students; mean accuracy {mean_acc:.0%}; most common patterns "
                           f"{[(v['title'], v['students']) for _, v in prev[:3]]}; hardest topic "
                           f"{min(c['topics'].items(), key=lambda kv: kv[1]['accuracy'])[0]}", "teacher", api_key))

    body.append("<h2>Do this in your next class</h2><div class='todo'>")
    acts = []
    if c["risk"] and c["risk"][0]["risk"] > 0:
        r0 = c["risk"][0]; f0 = S[r0["student"]]["flags"]
        if f0: acts.append(f"<b>Meet {E(r0['student'])} 1:1 (5 min):</b> {E(f0[0]['teacher_action'])}")
    hard = min(c["topics"].items(), key=lambda kv: kv[1]["mastery_rate"] + kv[1]["accuracy"])
    acts.append(f"<b>Re-teach {E(hard[0])}:</b> {hard[1]['accuracy']:.0%} accuracy, {hard[1]['mastery_rate']:.0%} mastered; "
                f"top error: {E(hard[1]['top_error'][0].replace('_', ' '))}. Use a worked example plus a contrasting wrong example.")
    if prev: acts.append(f"<b>Whole-class habit ({E(prev[0][1]['title'])}, {prev[0][1]['students']} students):</b> {E(prev[0][1]['teacher'])}")
    if c["item_flags"]: acts.append(f"<b>Fix {len(c['item_flags'])} question(s)</b> whose real difficulty does not match the label (table below).")
    body.append("".join(f'<div class="act">{x}</div>' for x in acts[:4]) + "</div>")

    body.append("<h2>Who needs attention first</h2>")
    mx = max(r["risk"] for r in c["risk"]) or 1
    body.append(svg_hbars([(r["student"], r["risk"], C["red"] if r["risk"] >= .6 * mx else C["amber"] if r["risk"] >= .3 * mx else C["teal"])
                           for r in c["risk"]], maxv=mx, fmt="{:.1f}", w=760, label_w=160))
    body.append("<table><tr><th>Student</th><th>Answers</th><th>Accuracy</th><th>Main pattern</th><th>Your move</th></tr>")
    for r in c["risk"]:
        f = S[r["student"]]["flags"]
        body.append(f"<tr><td>{E(r['student'])}</td><td>{r['n']}</td><td>{r['accuracy']:.0%}</td><td>{E(r['top_flag'])}</td>"
                    f"<td>{E(f[0]['teacher_action'] if f else 'Extend: give level 4-5 challenge items.')}</td></tr>")
    body.append("</table>")

    body.append("<h2>Mastery map (student × topic)</h2><div class='sub'>Final BKT mastery. Red &lt; 0.45 · amber &lt; 0.70 · blue &lt; 0.90 · green mastered.</div>")
    body.append(svg_heatmap(names, topics, lambda s, t: S[s]["topics"][t]["end_p"] if t in S[s]["topics"] else None))

    body.append('<div class="grid2"><div><h2>How common is each pattern</h2>')
    body.append(svg_hbars([(v["title"].split(" (")[0], v["share"], C["red"]) for _, v in prev], w=440, label_w=190) if prev else "<p>None.</p>")
    body.append("</div><div><h2>Topic difficulty</h2>")
    body.append(svg_hbars([(t, T["accuracy"], C["blue"]) for t, T in sorted(c["topics"].items(), key=lambda kv: kv[1]["accuracy"])], w=440, label_w=150))
    body.append("</div></div>")

    body.append("<h2>Teaching playbook for the patterns in this class</h2>")
    for code, v in prev:
        body.append(f'<div class="card" style="border-color:{C["amber"]}"><h3>{E(v["title"])}<span class="tag" style="background:{C["grey"]}">'
                    f'{v["students"]} of {c["n_students"]} students</span></h3><div class="grid2"><div class="act"><b>Teaching move:</b> {E(v["teacher"])}</div>'
                    f'<div class="act"><b>Change in material:</b> {E(v["material"])}</div></div>'
                    f'<div class="sub" style="margin-top:6px">Students: {E(", ".join(s for s in names if any(f["code"] == code for f in S[s]["flags"])))}</div></div>')

    body.append("<h2>Question quality</h2>")
    if c["item_flags"]:
        body.append("<table><tr><th>Item</th><th>Topic</th><th>Label</th><th>Observed level</th><th>Correct</th><th>n</th><th>Fix</th></tr>")
        body.append("".join(f"<tr><td>{E(i['item_id'])}</td><td>{E(i['topic'])}</td><td>{i['label']}</td><td><b>{i['observed']}</b></td>"
                            f"<td>{i['p_correct']:.0%}</td><td>{i['n']}</td><td>{E(i['fix'])}</td></tr>" for i in c["item_flags"]) + "</table>")
    else:
        body.append("<p>No mislabelled items (needs ≥ 3 unhinted attempts per item).</p>")
    body.append('<div class="note">Method: rule-based detectors on the interaction log (insight_engine.py). Wheel-spinning: Beck &amp; Gong (2013). '
                'Rapid guessing: Wise &amp; Ma (2012) normative 10% threshold. Help abuse / avoidance: Aleven et al. (2006). '
                'Observed item level = 1 + 4·(1 − share correct, unhinted). Mastery is a model estimate; confirm high-stakes decisions with the student.</div>')
    return _page(f"Teacher report {title}", "".join(body))
