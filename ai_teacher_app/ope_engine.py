"""
ope_engine.py  -  Off-Policy Evaluation (PDIS + Doubly Robust + bootstrap safety gate)

Now wired into the product: real sessions are logged by storage.py, converted to
trajectories by `load_real_trajectories`, and a *candidate* Q-table is compared
against what actually happened before it is allowed to replace the live one
(see pages/2_Policy_Check.py).

Fixes vs v1
  * target-policy probabilities use the SAME masked softmax as the live app
  * baseline is the DISCOUNTED return (v1 compared undiscounted vs discounted)
  * action count inferred from the Q-table (6 actions incl. worked example)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from rl_engine import (NUM_STATES, NUM_ACTIONS, GAMMA, softmax, policy_probs,
                       generate_synthetic_logs)  # softmax re-exported for old imports


def _pi(q_table: np.ndarray, s: int, tau: float) -> np.ndarray:
    return policy_probs(q_table[s], s, tau)


def fit_tabular_outcome_model(trajectories, num_states=NUM_STATES, num_actions=NUM_ACTIONS,
                              gamma=GAMMA, epochs=25) -> np.ndarray:
    """Q_hat(s,a) by Fitted Q-Iteration; terminal transitions bootstrap to 0."""
    q = np.zeros((num_states, num_actions))
    tot = np.zeros_like(q); cnt = np.zeros_like(q)
    for tr in trajectories:
        for s, a, r in zip(tr["states"], tr["actions"], tr["rewards"]):
            tot[s, a] += r; cnt[s, a] += 1
    q = np.divide(tot, cnt, out=np.zeros_like(q), where=cnt > 0)
    for _ in range(epochs):
        acc = np.zeros_like(q); c2 = np.zeros_like(q)
        for tr in trajectories:
            T = len(tr["actions"])
            for t in range(T):
                s, a, r = tr["states"][t], tr["actions"][t], tr["rewards"][t]
                v_next = np.max(q[tr["states"][t + 1]]) if t + 1 < T else 0.0
                acc[s, a] += r + gamma * v_next; c2[s, a] += 1
        m = c2 > 0
        q[m] = acc[m] / c2[m]
    return q


def compute_pdwis(trajectories, q_table, tau, gamma=GAMMA) -> Tuple[float, float, float]:
    """Per-decision weighted IS. Returns (value, ESS, max cumulative weight)."""
    N = len(trajectories)
    if N == 0: return 0.0, 0.0, 0.0
    cum, final, max_w = [], [], 0.0
    for tr in trajectories:
        w, ws = 1.0, []
        for s, a, mu in zip(tr["states"], tr["actions"], tr["behavior_probs"]):
            w *= _pi(q_table, s, tau)[a] / mu
            ws.append(w); max_w = max(max_w, w)
        cum.append(ws); final.append(w)
    fw = np.array(final)
    ess = (fw.sum() ** 2) / (fw ** 2).sum() if (fw ** 2).sum() > 0 else 0.0
    H = max(len(t["actions"]) for t in trajectories)
    v = 0.0
    for t in range(H):
        num = den = 0.0
        for i, tr in enumerate(trajectories):
            if t < len(tr["rewards"]):
                num += cum[i][t] * tr["rewards"][t]; den += cum[i][t]
        if den > 0: v += gamma ** t * num / den
    return float(v), float(ess), float(max_w)


def compute_doubly_robust(trajectories, target_q, outcome_q, tau, gamma=GAMMA) -> float:
    if not trajectories: return 0.0
    rets = []
    for tr in trajectories:
        w_prev, val = 1.0, 0.0
        for t, (s, a, r, mu) in enumerate(zip(tr["states"], tr["actions"], tr["rewards"], tr["behavior_probs"])):
            pi = _pi(target_q, s, tau)
            v_hat = float(np.sum(pi * outcome_q[s]))
            w = w_prev * pi[a] / mu
            val += gamma ** t * (w * r - w * outcome_q[s, a] + w_prev * v_hat)
            w_prev = w
        rets.append(val)
    return float(np.mean(rets))


def behavior_value(trajectories, gamma=GAMMA) -> float:
    return float(np.mean([sum(gamma ** t * r for t, r in enumerate(tr["rewards"])) for tr in trajectories]))


def run_full_ope_analysis(trajectories, candidate_q, tau_grid, alpha_ess=0.10, max_w_bound=50.0,
                          bootstrap_samples=200, gamma=GAMMA, seed=0) -> Dict:
    rng = np.random.default_rng(seed)
    N = len(trajectories)
    min_ess = alpha_ess * N
    outcome_q = fit_tabular_outcome_model(trajectories, gamma=gamma)

    rows, best_tau, best_val = [], None, -np.inf
    for tau in sorted(tau_grid, reverse=True):
        v, ess, mw = compute_pdwis(trajectories, candidate_q, tau, gamma)
        ok = (ess >= min_ess) and (mw <= max_w_bound)
        rows.append({"tau": tau, "v_pdwis": v, "ess": ess, "max_w": mw, "feasible": ok})
        if ok and v > best_val: best_val, best_tau = v, tau
    feasible_found = best_tau is not None
    if not feasible_found:
        best_tau = max(tau_grid)                    # most conservative (closest to behaviour)

    best_dr = compute_doubly_robust(trajectories, candidate_q, outcome_q, best_tau, gamma)
    p_b, d_b = [], []
    for _ in range(bootstrap_samples):
        boot = [trajectories[i] for i in rng.integers(0, N, N)]
        p_b.append(compute_pdwis(boot, candidate_q, best_tau, gamma)[0])
        d_b.append(compute_doubly_robust(boot, candidate_q, outcome_q, best_tau, gamma))
    base = behavior_value(trajectories, gamma)
    ci_dr = (float(np.percentile(d_b, 2.5)), float(np.percentile(d_b, 97.5)))
    vp, vd = np.var(p_b), np.var(d_b)
    return {"best_tau": best_tau, "feasible_tau_found": feasible_found, "behavior_baseline": base,
            "best_pdwis": best_val if feasible_found else float("nan"), "best_dr": best_dr,
            "ci_dr": ci_dr, "ci_pdwis": (float(np.percentile(p_b, 2.5)), float(np.percentile(p_b, 97.5))),
            "pdwis_boots": p_b, "dr_boots": d_b, "sweep_df": pd.DataFrame(rows),
            "passed_gate": bool(feasible_found and ci_dr[0] > base),
            "var_pdwis": float(vp), "var_dr": float(vd),
            "var_reduction": float((1 - vd / vp) * 100) if vp > 0 else 0.0, "n_trajectories": N}


def load_real_trajectories(events: List[Dict], min_len: int = 2) -> List[Dict]:
    """Group logged decision events (storage.log_event) into per-(session,topic) trajectories."""
    groups: Dict[Tuple, Dict] = {}
    for e in events:
        if e.get("action") is None or e.get("propensity") is None: continue
        g = groups.setdefault((e["session_id"], e["topic"]),
                              {"states": [], "actions": [], "rewards": [], "behavior_probs": []})
        g["states"].append(int(e["state"])); g["actions"].append(int(e["action"]))
        g["rewards"].append(float(e["reward"])); g["behavior_probs"].append(max(1e-6, float(e["propensity"])))
    return [g for g in groups.values() if len(g["actions"]) >= min_len]
