"""question_bank.py - cache generated questions and learn their REAL difficulty from
student responses (a light IRT-style calibration). Lets the RL policy pick from
calibrated items instead of trusting the LLM's difficulty label."""
import hashlib, json, os, random
from typing import Dict, List, Optional
from storage import DATA_DIR

BANK_PATH = os.path.join(DATA_DIR, "question_bank.json")


def _load() -> Dict:
    try:
        with open(BANK_PATH, encoding="utf-8") as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save(b: Dict):
    with open(BANK_PATH, "w", encoding="utf-8") as f: json.dump(b, f)

def item_id(subject: str, topic: str, question: str) -> str:
    return hashlib.sha1(f"{subject}|{topic}|{question}".encode()).hexdigest()[:12]

def add(item: Dict, subject: str, topic: str, difficulty: int) -> str:
    if item.get("verified") is False:            # never bank answers that failed verification
        return item_id(subject, topic, item.get("question", ""))
    b = _load(); iid = item_id(subject, topic, item["question"])
    if iid not in b:
        b[iid] = {"subject": subject, "topic": topic, "labeled": difficulty, "n": 0, "n_correct": 0, "item": item}
        _save(b)
    return iid

def record(iid: str, correct: bool, hints_used: int = 0):
    b = _load()
    if iid in b and hints_used == 0:             # hinted answers are biased evidence
        b[iid]["n"] += 1; b[iid]["n_correct"] += int(correct); _save(b)

def calibrated_difficulty(rec: Dict) -> float:
    n, k = rec["n"], rec["n_correct"]
    p = (k + 1) / (n + 2)                        # Laplace smoothing
    emp = 1 + 4 * (1 - p)                        # low success -> hard
    w = n / (n + 10)                             # trust data more as n grows
    return (1 - w) * rec["labeled"] + w * emp

def pick(subject: str, topic: str, difficulty: int, exclude: List[str]) -> Optional[Dict]:
    cands = [(iid, r) for iid, r in _load().items()
             if r["subject"] == subject and r["topic"] == topic and iid not in exclude
             and round(calibrated_difficulty(r)) == difficulty]
    if not cands: return None
    iid, r = random.choice(cands)
    return {**r["item"], "bank_id": iid}
