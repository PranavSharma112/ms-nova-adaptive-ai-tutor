"""
forms_pipeline.py : Google Form responses -> one report per student + one teacher report.

  python forms_pipeline.py --csv responses.csv --config forms/quiz_config.json --out reports/
  python forms_pipeline.py --sheet <sheet_url> --creds service_account.json --config ... --out reports/
Add --agent to let ChatGPT (OPENAI_API_KEY) or Gemini (GEMINI_API_KEY) write coach notes.
"""
import argparse, json, os, re
import forms_adapter as fa, deep_engine as de, deep_report as dr, llm_agent as ag

def run(rows, cfg, out, agent=False):
    os.makedirs(out, exist_ok=True)
    ev = fa.to_events(rows, cfg); D = de.deep_class(ev); llm = ag.provider() if agent else None
    files = []
    for s in D["names"]:
        p = os.path.join(out, f"student_{re.sub(r'[^A-Za-z0-9]+', '_', s)}.html")
        open(p, "w", encoding="utf-8").write(dr.student_page(D, s, llm)); files.append(p)
    p = os.path.join(out, "teacher_report.html")
    open(p, "w", encoding="utf-8").write(dr.teacher_page(D, ev, cfg.get("form_title", "Class"), llm)); files.append(p)
    skipped = sorted({e["student"] for e in ev} - set(D["names"]))
    return files, skipped

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--csv"); a.add_argument("--sheet"); a.add_argument("--creds")
    a.add_argument("--config", required=True); a.add_argument("--out", default="reports"); a.add_argument("--agent", action="store_true")
    x = a.parse_args(); cfg = json.load(open(x.config, encoding="utf-8"))
    rows = fa.read_csv(x.csv) if x.csv else fa.read_sheet(x.sheet, x.creds)
    files, skipped = run(rows, cfg, x.out, x.agent)
    print(f"{len(files)} reports written to {x.out}/" + (f"; skipped (too few answers): {skipped}" if skipped else ""))
