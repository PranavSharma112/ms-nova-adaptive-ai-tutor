"""storage.py - interaction logging (feeds OPE) + long-term student memory (profile)."""
import json, os, re, time
from typing import Dict, List

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
LOG_PATH = os.path.join(DATA_DIR, "interactions.jsonl")
PROFILE_PATH = os.path.join(DATA_DIR, "profiles.json")


def log_event(event: Dict) -> None:
    event = {"ts": time.time(), **event}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=float) + "\n")


def read_events() -> List[Dict]:
    if not os.path.exists(LOG_PATH): return []
    out = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try: out.append(json.loads(line))
            except json.JSONDecodeError: continue
    return out


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "guest").strip().lower())[:40] or "guest"


def empty_profile() -> Dict:
    return {"summary": "", "interests": [], "strengths": [], "weaknesses": [], "misconceptions": [], "sessions": 0}


def load_profile(name: str) -> Dict:
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return {**empty_profile(), **json.load(f).get(_key(name), {})}
    except (FileNotFoundError, json.JSONDecodeError):
        return empty_profile()


def save_profile(name: str, profile: Dict) -> None:
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f: allp = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        allp = {}
    allp[_key(name)] = profile
    with open(PROFILE_PATH, "w", encoding="utf-8") as f: json.dump(allp, f, indent=2)


def profile_blurb(profile: Dict) -> str:
    """Short text injected into prompts so examples feel personal."""
    bits = []
    if profile.get("interests"):     bits.append("Interests: " + ", ".join(profile["interests"][:5]))
    if profile.get("strengths"):     bits.append("Strengths: " + ", ".join(profile["strengths"][:4]))
    if profile.get("weaknesses"):    bits.append("Struggles with: " + ", ".join(profile["weaknesses"][:4]))
    if profile.get("misconceptions"):bits.append("Past misconceptions: " + "; ".join(profile["misconceptions"][:3]))
    return "\n".join(bits)
