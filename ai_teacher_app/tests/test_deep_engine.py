"""Recovery test: the engine must find what the simulator hid."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from cohort_sim import cohort
import deep_engine as de, llm_agent as ag

def main():
    ev, truth = cohort(); D = de.deep_class(ev)
    m = lambda p, f: np.mean([D["students"][s]["features"][f] for s in D["names"] if truth[s] == p])
    assert m("careless", "slip") > 2 * m("fast_learner", "slip")                 # carelessness recovered
    assert m("rapid_guesser", "rapid_share") > 0.15 > m("careless", "rapid_share")
    assert m("hint_dependent", "hint_rate") > 0.5
    assert m("slow_steady", "time_ratio") > 1.5
    assert m("fast_learner", "learn_rate") > m("slow_steady", "learn_rate")
    purity = sum(max(collections.Counter(truth[x] for x in g["members"]).values()) for g in D["groups"]) / len(D["names"])
    assert purity >= 0.7, purity
    named = {g["name"]: collections.Counter(truth[x] for x in g["members"]).most_common(1)[0][0] for g in D["groups"]}
    assert named.get("Hint-dependent") == "hint_dependent" and named.get("Disengaged guessers") == "rapid_guesser"
    bad = sum(1 for i in D["items"] if i["verdict"].startswith("revise"))
    assert bad <= 0.05 * len(D["items"]) + 1, bad                                 # all sim items equal: few false alarms
    assert ag.numbers_ok("45% of 46", {"a": 0.45, "n": 46}) and not ag.numbers_ok("72%", {"a": 0.45})
    print(f"ALL PASS  purity={purity:.0%}  k={D['k']}  silhouette={D['silhouette']:.2f}  item_false_alarms={bad}/{len(D['items'])}")

if __name__ == "__main__": main()
