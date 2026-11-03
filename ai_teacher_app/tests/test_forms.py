"""End-to-end: simulated Google Form CSV -> events -> groups -> 31 reports. Hidden profiles must be recovered."""
import sys, os, json, tempfile, collections
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import forms_sim, forms_adapter as fa, deep_engine as de, forms_pipeline as fp

def main():
    d = tempfile.mkdtemp(); truth = forms_sim.write(f"{d}/r.csv", f"{d}/c.json")
    cfg = json.load(open(f"{d}/c.json")); rows = fa.read_csv(f"{d}/r.csv")
    ev = fa.to_events(rows, cfg)
    assert len(ev) == len(rows) * len(cfg["questions"])                       # 1 event per answer
    assert all(e["elapsed"] is None and e["confidence"] in (0.0, .5, 1.0) for e in ev)
    D = de.deep_class(ev)
    assert "time_ratio" not in D["feats"] and "overconfidence" in D["feats"]   # forms mode
    purity = sum(max(collections.Counter(truth[x] for x in g["members"]).values()) for g in D["groups"]) / len(D["names"])
    named = {g["name"]: collections.Counter(truth[x] for x in g["members"]).most_common(1)[0][0] for g in D["groups"]}
    assert named.get("Overconfident") == "overconfident" and named.get("Knows it, slips") == "careless", named
    assert purity >= 0.65, purity
    files, skipped = fp.run(rows, cfg, f"{d}/out")
    assert len(files) == 31 and not skipped
    print(f"ALL PASS  purity={purity:.0%}  groups={sorted(named)}")

if __name__ == "__main__": main()
