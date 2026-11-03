"""Teacher/admin page: evaluate a CANDIDATE tutoring policy offline before it goes live."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

import storage
from ope_engine import load_real_trajectories, run_full_ope_analysis
from rl_engine import generate_synthetic_logs, train_q, DEFAULT_TAU

st.set_page_config(page_title="Policy check", page_icon="🧪", layout="wide")
st.title("🧪 Policy check (teacher view)")
st.caption("Before a new tutoring policy replaces the live one, estimate how it would have performed on logged sessions, "
           "with uncertainty. It is only promoted if the lower confidence bound beats what actually happened.")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE, CAND = os.path.join(ROOT, "models", "q_table.npy"), os.path.join(ROOT, "models", "q_table_candidate.npy")
REPORT = os.path.join(ROOT, "models", "train_report.json")

if os.path.exists(REPORT):
    st.subheader("Simulator benchmark of the live policy")
    st.dataframe(pd.DataFrame(json.load(open(REPORT))).T.round(3), width="stretch")
    st.caption("Simulated students (randomised BKT; learning fastest when P(correct)≈0.7 — an assumption). "
               "Simulation is a sanity check, not proof of real-world benefit.")

events = storage.read_events()
real = load_real_trajectories(events)
st.subheader("Data")
src = st.radio("Logs to evaluate on", [f"Real sessions ({len(real)} trajectories)", "Synthetic logs (demo)"],
               index=0 if len(real) >= 30 else 1, horizontal=True)
n_syn = st.slider("Synthetic trajectories", 100, 1500, 400, 100) if src.startswith("Synthetic") else 0
if src.startswith("Real") and len(real) < 30:
    st.warning("Fewer than 30 real trajectories — estimates will be very noisy and the gate will (rightly) refuse to approve.")

st.subheader("Candidate policy")
c1, c2, c3 = st.columns(3)
episodes = c1.number_input("Training episodes", 5000, 100000, 30000, 5000)
seed = c2.number_input("Seed", 0, 999, 1)
if c3.button("Train new candidate"):
    with st.spinner("Training..."):
        np.save(CAND, train_q(episodes=int(episodes), seed=int(seed)))
    st.success("Candidate saved to models/q_table_candidate.npy")

if not os.path.exists(CAND):
    st.info("No candidate yet — train one above."); st.stop()

boots = st.slider("Bootstrap samples", 50, 500, 150, 50)
alpha = st.slider("Min ESS fraction", 0.02, 0.5, 0.10, 0.02)
if st.button("Run off-policy evaluation", type="primary"):
    trajs = real if src.startswith("Real") else generate_synthetic_logs(n_syn)
    if len(trajs) < 5: st.error("Not enough trajectories."); st.stop()
    with st.spinner("Running PDIS + Doubly Robust + bootstrap..."):
        res = run_full_ope_analysis(trajs, np.load(CAND), [0.5, 0.3, 0.2, 0.15, 0.1, DEFAULT_TAU], alpha_ess=alpha, bootstrap_samples=boots)
    st.session_state.ope = res

res = st.session_state.get("ope")
if res:
    a, b, c, d = st.columns(4)
    a.metric("Behaviour (what happened)", f"{res['behavior_baseline']:.3f}")
    b.metric("Candidate (DR)", f"{res['best_dr']:.3f}")
    c.metric("DR 95% CI", f"{res['ci_dr'][0]:.2f} – {res['ci_dr'][1]:.2f}")
    d.metric("Variance cut vs PDIS", f"{res['var_reduction']:.0f}%")
    if res["passed_gate"]:
        st.success("✅ Safety gate PASSED — lower confidence bound beats the behaviour baseline.")
        if st.button("Promote candidate to live"):
            if os.path.exists(LIVE): shutil.copy(LIVE, LIVE.replace(".npy", "_backup.npy"))
            shutil.copy(CAND, LIVE); st.cache_resource.clear()
            st.success("Promoted. Previous policy backed up as q_table_backup.npy.")
    else:
        st.error("⛔ Safety gate NOT passed — keep the current policy" + ("" if res["feasible_tau_found"]
                 else " (no temperature met the effective-sample-size / max-weight limits: too little overlap with logged behaviour)."))
    st.dataframe(res["sweep_df"].round(3), width="stretch")
    st.bar_chart(pd.DataFrame({"PDIS": res["pdwis_boots"], "DR": res["dr_boots"]}).describe().loc[["mean", "25%", "75%"]].T)
