"""
llm_agent.py
------------
Optional LLM agent for the deep reports. Provider order: OpenAI (ChatGPT) if
OPENAI_API_KEY is set, else Gemini via teacher_engine, else None (reports still work).

The agent only WRITES and NAMES. It never computes or decides:
  * coach_note(facts)        -> personal coaching paragraph for the student
  * teacher_brief(facts)     -> 4-sentence class brief
  * name_misconceptions(...) -> short names + one-line fix per misconception cluster
Guard: any number in the LLM output that is not in the facts is rejected (output dropped).
"""
from __future__ import annotations
import json, os, re
from typing import Callable, Dict, List, Optional

SYSTEM = ("You are an expert teacher and learning scientist writing for a report. Use ONLY the JSON facts given. "
          "Never invent numbers, names or diagnoses. Plain, warm, direct English. No emojis, no em dashes.")


def _openai(prompt: str) -> str:
    from openai import OpenAI                       # pip install openai
    cli = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = cli.chat.completions.create(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3,
                                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""


def _gemini(api_key: str) -> Callable[[str], str]:
    import teacher_engine as te
    return lambda prompt: te._generate_text(SYSTEM + "\n\n" + prompt, api_key)


def provider(gemini_key: Optional[str] = None) -> Optional[Callable[[str], str]]:
    if os.environ.get("OPENAI_API_KEY"): return _openai
    key = gemini_key or os.environ.get("GEMINI_API_KEY")
    if key:
        try:
            import teacher_engine as te
            if not te.is_demo(key): return _gemini(key)
        except Exception: pass
    return None


_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers_ok(text: str, facts: Dict) -> bool:
    """Every number in text must appear in the facts (as given, as %, or rounded)."""
    allowed = set()
    def walk(x):
        if isinstance(x, dict): [walk(v) for v in x.values()]
        elif isinstance(x, (list, tuple)): [walk(v) for v in x]
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            for v in (x, x * 100):
                allowed.update({f"{v:.0f}", f"{v:.1f}", f"{v:.2f}", str(v)})
        elif isinstance(x, str): allowed.update(_NUM.findall(x))
    walk(facts)
    allowed.update({str(i) for i in range(1, 11)})   # list numbering / "3 things"
    return all(n in allowed or n.rstrip("0").rstrip(".") in allowed for n in _NUM.findall(text))


def _ask(llm, task: str, facts: Dict) -> str:
    if llm is None: return ""
    try:
        out = llm(f"{task}\n\nFACTS (JSON):\n{json.dumps(facts, default=str)[:6000]}").strip()
        return out if numbers_ok(out, facts) else ""
    except Exception:
        return ""


def coach_note(llm, facts: Dict) -> str:
    return _ask(llm, "Write a 120-word personal coaching note to this student: what their pattern is, why it costs marks, "
                     "and the single most valuable habit to change this week.", facts)


def teacher_brief(llm, facts: Dict) -> str:
    return _ask(llm, "Write a 4-sentence brief for the teacher: the main finding, the group needing action first, "
                     "the teaching move with the best evidence, and one material change.", facts)


def name_misconceptions(llm, clusters: List[Dict]) -> Dict[str, Dict]:
    """Returns {label: {"name":..., "fix":...}}; empty if no LLM."""
    if llm is None or not clusters: return {}
    try:
        raw = llm("For each misconception give a short name (max 5 words) and a one-line teaching fix (a contrasting example). "
                  "Return ONLY JSON list of {label, name, fix}.\n" + json.dumps([{"topic": c["topic"], "label": c["label"]} for c in clusters[:12]]))
        data = json.loads(re.sub(r"```(json)?", "", raw).strip())
        return {d["label"]: {"name": d["name"], "fix": d["fix"]} for d in data if "label" in d}
    except Exception:
        return {}
