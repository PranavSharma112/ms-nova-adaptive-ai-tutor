"""
rl_bandit.py
------------
Contextual bandit / linear function-approximation policy, as an alternative
to the tabular Q-table in rl_engine.py. Same reward, same BKTTracker/
StudentSimulator, same action_mask/safety rules -- only the Q-function
changes from a 125x6 table to a continuous 9-feature -> 6-action linear model,
so mastery/difficulty/streak are no longer binned.

Trained with the same simulator via semi-gradient SARSA-ish TD(0) updates.
Exposes: featurize_tracker, choose_action, train_linear_q, evaluate (reuses
rl_engine._run_policy), save/load.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
import rl_engine as rl

N_FEAT = 9
MODEL_PATH = "models/linear_q.npy"
REPORT_PATH = "models/linear_q_report.json"


def featurize_tracker(bkt: rl.BKTTracker) -> np.ndarray:
    """Continuous features -- no discretization/binning."""
    p_L = bkt.p_L
    d_norm = bkt.last_difficulty / 4.0
    h = bkt.history[-4:]
    streak = (sum(2 * o - 1 for o in h) / 4.0) if h else 0.0
    n_recent = len(h)
    recent_acc = (sum(h) / n_recent) if n_recent else 0.5
    return np.array([
        1.0, p_L, p_L ** 2, d_norm, streak,
        recent_acc, p_L * d_norm, float(n_recent == 0), abs(p_L - 0.7),
    ])


def q_values(W: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return W @ phi  # (6,9) @ (9,) -> (6,)


def choose_action(W: np.ndarray, bkt: rl.BKTTracker, tau: float = rl.DEFAULT_TAU,
                   rng: Optional[np.random.Generator] = None, greedy: bool = False,
                   disallow=None) -> Tuple[int, float]:
    phi = featurize_tracker(bkt)
    q = q_values(W, phi)
    probs = rl.policy_probs(q, bkt.state, tau, disallow)
    rng = rng or np.random.default_rng()
    a = int(np.argmax(probs)) if greedy else int(rng.choice(rl.NUM_ACTIONS, p=probs))
    return a, float(probs[a])


def train_linear_q(episodes: int = 20000, alpha: float = 0.01, gamma: float = rl.GAMMA,
                    eps_start: float = 1.0, eps_end: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.zeros((rl.NUM_ACTIONS, N_FEAT))
    sim = rl.StudentSimulator(rng)
    for ep in range(episodes):
        eps = max(eps_end, eps_start * (1 - ep / (0.8 * episodes)))
        sim.reset()
        phi = featurize_tracker(sim.bkt)
        done = False
        while not done:
            mask = rl.action_mask(sim.bkt.state)
            allowed = np.flatnonzero(mask)
            q = q_values(W, phi)
            if rng.random() < eps:
                a = int(rng.choice(allowed))
            else:
                a = int(allowed[np.argmax(q[allowed])])
            _, r, done = sim.step(a)
            phi2 = featurize_tracker(sim.bkt)
            if done:
                target = r
            else:
                m2 = rl.action_mask(sim.bkt.state)
                target = r + gamma * np.max(q_values(W, phi2)[m2])
            td_err = target - q[a]
            W[a] += alpha * td_err * phi
            phi = phi2
    return W


def evaluate_linear(W: np.ndarray, n: int = 2000, seed: int = 123):
    def linear_sample(sim, s, rng):
        return choose_action(W, sim.bkt, rng=rng)[0]

    def linear_greedy(sim, s, rng):
        return choose_action(W, sim.bkt, rng=rng, greedy=True)[0]
    return {
        "linear Q (contextual, sampled)": rl._run_policy(linear_sample, n, seed),
        "linear Q (greedy)": rl._run_policy(linear_greedy, n, seed),
    }


def save(W: np.ndarray, path: str = MODEL_PATH):
    np.save(path, W)


def load(path: str = MODEL_PATH) -> Optional[np.ndarray]:
    try:
        return np.load(path)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    import json, os
    os.makedirs("models", exist_ok=True)
    Q = rl.train_q(episodes=15000)
    W = train_linear_q(episodes=15000)
    save(W)
    report = {
        "fixed_difficulty_3": rl._run_policy(lambda sim, s, rng: 2, 2000, 1),
        "staircase": rl._run_policy(lambda sim, s, rng: (min(4, sim.bkt.last_difficulty + 1) if sim.bkt.history and sim.bkt.history[-1] else max(0, sim.bkt.last_difficulty - 1)) if sim.bkt.history else sim.bkt.last_difficulty, 2000, 1),
        "tabular Q-table (deployed)": rl._run_policy(lambda sim, s, rng: rl.choose_action(Q, s, rng=rng)[0], 2000, 1),
        **evaluate_linear(W, 2000, 1),
    }
    json.dump(report, open(REPORT_PATH, "w"), indent=2)
    for k, v in report.items():
        print(f"{k:32s} reward={v['mean_reward']:.3f} gain={v['mean_mastery_gain']:.3f} frustr%={v['pct_frustrated(3+ wrong in a row)']:.1f}")
