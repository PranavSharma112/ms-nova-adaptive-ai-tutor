import sys, os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mock_events import all_events
import insight_engine as ie

def test_injected_patterns_detected():
    C = ie.analyze_class(all_events())
    codes = {s: {f["code"] for f in r["flags"]} for s, r in C["students"].items()}
    assert "wheel_spinning" in codes["Ravi"] and "persistent_misconception" in codes["Ravi"]
    assert "rapid_guessing" in codes["Meera"]
    assert "hint_abuse" in codes["Kabir"]
    assert "fluency_gap" in codes["Isha"]
    assert not any(f["severity"] == "high" for f in C["students"]["Asha"]["flags"])
    for s in ("Asha", "Ravi", "Isha"): assert "rapid_guessing" not in codes[s]
    assert C["risk"][-1]["student"] == "Asha"

def test_invariants():
    ev = all_events(); C = ie.analyze_class(ev)
    assert C["n_events"] == len([e for e in ev if e["kind"] != "worked_example"])
    for r in C["students"].values():
        assert sum(T["n"] for T in r["topics"].values()) == r["n"]
        assert 0 <= r["summary"]["accuracy"] <= 1
        assert len({f["code"] for f in r["flags"]}) == len(r["flags"])     # merged, no duplicates

def test_empty():
    assert ie.analyze_student([])["n"] == 0 and ie.analyze_class([])["n_students"] == 0

if __name__ == "__main__":
    test_injected_patterns_detected(); test_invariants(); test_empty(); print("ALL PASS")
