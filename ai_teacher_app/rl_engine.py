"""
rl_engine.py
------------
Single source of truth for the RL side, shared by the simulator, the trainer,
the OPE code and the live app (so train-time and serve-time states MATCH).

State  (125 = 5 x 5 x 5): mastery bin M (from BKT p_L), last difficulty D, streak C
Action (6): 0..4 = ask a question at difficulty 1..5, 5 = show a worked example

Key fixes vs. v1
  * The policy is actually TRAINED (tabular Q-learning on a randomized BKT simulator).
  * The app uses the same online BKT + state encoding as the simulator.
  * Reward mismatch penalty targets difficulty ~ 1 + 4*p_L (no off-by-one).
  * Policy = softmax over Q with a *safety mask* (no big jumps, no going harder
    after a miss). Because the mask is a pure function of the state, propensities
    are exactly reproducible for off-policy evaluation.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

NUM_STATES = 125
NUM_ACTIONS = 6            # 0-4 difficulty, 5 worked example
WORKED_EXAMPLE = 5
MASTERY_THRESHOLD = 0.90   # topic considered mastered
GAMMA = 0.95
DEFAULT_TAU = 0.08        # deployment softmax temperature (kept >0 so we log real exploration)

SLIP = {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25}
GUESS = {1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15, 5: 0.10}
WORKED_EXAMPLE_BOOST = 0.12
WORKED_EXAMPLE_COST = 0.06


# ---------------------------------------------------------------- state encoding
def discretize_mastery(p_L: float) -> int:
    if p_L < 0.20: return 0
    if p_L < 0.45: return 1
    if p_L < 0.70: return 2
    if p_L < 0.90: return 3
    return 4

def streak_index(history: List[int]) -> int:
    if not history:
        return 2
    net = sum(2 * o - 1 for o in history[-4:])
    return max(-2, min(2, net)) + 2

def encode_state(M: int, D: int, C: int) -> int:
    return M * 25 + D * 5 + C

def decode_state(s: int) -> Tuple[int, int, int]:
    return s // 25, (s % 25) // 5, s % 5


# ---------------------------------------------------------------- reward (shared)
def compute_reward(p_before: float, p_after: float, difficulty: int, correct: Optional[int],
                   history: List[int], is_worked_example: bool = False) -> float:
    """nlg (normalised learning gain) + correctness bonus - difficulty mismatch
    - frustration penalty. `difficulty` is 1..5. `history` is BEFORE this step."""
    nlg = (p_after - p_before) / (1.0 - p_before) if p_before < 1.0 else 0.0
    if is_worked_example:
        return nlg - WORKED_EXAMPLE_COST
    target = 1.0 + 4.0 * p_before                     # ZPD-ish target difficulty
    mismatch = 0.05 * (difficulty - target) ** 2
    frustration = 0.15 if (len(history) >= 2 and history[-1] == 0 and history[-2] == 0 and correct == 0) else 0.0
    return nlg + 0.1 * (correct or 0) - mismatch - frustration


# ---------------------------------------------------------------- BKT tracker
class BKTTracker:
    """Online BKT for one topic. Used identically by simulator and live app."""
    def __init__(self, p_init=0.20, p_transit=0.15, slip_scale=1.0, guess_scale=1.0, zpd_learning=False):
        # zpd_learning=True (simulator only): students learn fastest when P(correct)~0.7.
        self.p_init, self.p_transit, self.zpd_learning = p_init, p_transit, zpd_learning
        self.slip_scale, self.guess_scale = slip_scale, guess_scale
        self.reset()

    def reset(self):
        self.p_L = self.p_init
        self.last_difficulty = 2      # index (0-4); start in the middle-low
        self.history: List[int] = []
        return self.state

    @property
    def state(self) -> int:
        return encode_state(discretize_mastery(self.p_L), self.last_difficulty, streak_index(self.history))

    def _sg(self, d: int, guess_bonus: float = 0.0):
        slip = min(0.45, SLIP[d] * self.slip_scale)
        guess = min(0.60, GUESS[d] * self.guess_scale + guess_bonus)
        return slip, guess

    def p_correct(self, d: int) -> float:
        slip, guess = self._sg(d)
        return self.p_L * (1 - slip) + (1 - self.p_L) * guess

    def observe(self, d: int, correct: int, guess_bonus: float = 0.0) -> Tuple[float, float]:
        """Update after a question. guess_bonus>0 => weaker evidence of mastery
        (e.g. hint used / very slow answer). Returns (p_before, p_after)."""
        p_before = self.p_L
        slip, guess = self._sg(d, guess_bonus)
        pc = p_before * (1 - slip) + (1 - p_before) * guess
        if correct:
            post = p_before * (1 - slip) / pc
        else:
            post = p_before * slip / (1 - pc)
        transit = self.p_transit
        if self.zpd_learning:
            transit *= 0.4 + 1.2 * float(np.exp(-((pc - 0.7) / 0.22) ** 2))
        self.p_L = post + (1 - post) * min(0.6, transit)
        self.history.append(int(correct))
        self.last_difficulty = d - 1
        return p_before, self.p_L

    def worked_example(self) -> Tuple[float, float]:
        p_before = self.p_L
        self.p_L = p_before + (1 - p_before) * WORKED_EXAMPLE_BOOST
        return p_before, self.p_L


# ---------------------------------------------------------------- policy
def action_mask(state: int) -> np.ndarray:
    """Safety mask, a pure function of state (so propensities are reproducible).
    - at most one difficulty step from the last question
    - never harder right after a miss (streak leaning wrong)
    - worked example only when the student is struggling"""
    _, D, C = decode_state(state)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for a in range(5):
        if abs(a - D) <= 1:
            mask[a] = True
    if C <= 1:                       # recent misses -> no harder than last
        for a in range(D + 1, 5):
            mask[a] = False
    if C <= 1 and D >= 1:            # struggling -> teaching move allowed
        mask[WORKED_EXAMPLE] = True
    if not mask.any():
        mask[D] = True
    return mask

def softmax(q_values: np.ndarray, tau: float) -> np.ndarray:
    z = np.asarray(q_values, dtype=float) / tau
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def policy_probs(q_row: np.ndarray, state: int, tau: float, disallow: Optional[List[int]] = None) -> np.ndarray:
    mask = action_mask(state).copy()
    for a in (disallow or []):
        if mask.sum() > 1: mask[a] = False        # app-level caps (e.g. max worked examples)
    z = np.where(mask, np.asarray(q_row, dtype=float) / tau, -np.inf)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def choose_action(q_table: np.ndarray, state: int, tau: float = DEFAULT_TAU,
                  rng: Optional[np.random.Generator] = None, greedy: bool = False,
                  disallow: Optional[List[int]] = None) -> Tuple[int, float]:
    """Returns (action, propensity). The propensity is the exact probability used to sample,
    which is what gets logged for off-policy evaluation."""
    probs = policy_probs(q_table[state], state, tau, disallow)
    rng = rng or np.random.default_rng()
    a = int(np.argmax(probs)) if greedy else int(rng.choice(NUM_ACTIONS, p=probs))
    return a, float(probs[a])


# ---------------------------------------------------------------- simulator
class StudentSimulator:
    """Randomised BKT students (domain randomisation) so the policy doesn't
    overfit one fake student."""
    def __init__(self, rng: Optional[np.random.Generator] = None, randomize: bool = True, max_steps: int = 12):
        self.rng = rng or np.random.default_rng()
        self.randomize, self.max_steps = randomize, max_steps
        self.reset()

    def reset(self) -> int:
        r = self.rng
        if self.randomize:
            self.bkt = BKTTracker(p_init=float(r.uniform(0.05, 0.40)), p_transit=float(r.uniform(0.08, 0.25)),
                                  slip_scale=float(r.uniform(0.6, 1.6)), guess_scale=float(r.uniform(0.6, 1.4)),
                                  zpd_learning=True)
        else:
            self.bkt = BKTTracker(zpd_learning=True)
        self.steps = 0
        return self.bkt.state

    @property
    def p_L(self): return self.bkt.p_L

    def step(self, action: int) -> Tuple[int, float, bool]:
        b = self.bkt
        if action == WORKED_EXAMPLE:
            pb, pa = b.worked_example()
            reward = compute_reward(pb, pa, 0, None, b.history, is_worked_example=True)
        else:
            d = action + 1
            correct = int(self.rng.random() < b.p_correct(d))
            hist_before = list(b.history)
            pb, pa = b.observe(d, correct)
            reward = compute_reward(pb, pa, d, correct, hist_before)
        self.steps += 1
        done = bool(b.p_L >= 0.95 or self.steps >= self.max_steps)
        return b.state, reward, done


# ---------------------------------------------------------------- training
def train_q(episodes: int = 30000, alpha: float = 0.08, gamma: float = GAMMA,
            eps_start: float = 1.0, eps_end: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))
    sim = StudentSimulator(rng)
    for ep in range(episodes):
        eps = max(eps_end, eps_start * (1 - ep / (0.8 * episodes)))
        s, done = sim.reset(), False
        while not done:
            mask = action_mask(s)
            allowed = np.flatnonzero(mask)
            if rng.random() < eps:
                a = int(rng.choice(allowed))
            else:
                a = int(allowed[np.argmax(Q[s, allowed])])
            s2, r, done = sim.step(a)
            if done:
                target = r
            else:
                m2 = action_mask(s2)
                target = r + gamma * np.max(Q[s2][m2])
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
    return Q


# ---------------------------------------------------------------- evaluation
def _run_policy(pick, n: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    gains, steps_to_mastery, rewards, frustr, worked = [], [], [], [], 0
    for _ in range(n):
        sim = StudentSimulator(rng)
        s, done, R, t, p0 = sim.reset(), False, 0.0, 0, sim.p_L
        streak_wrong, maxwrong = 0, 0
        while not done:
            a = pick(sim, s, rng)
            worked += (a == WORKED_EXAMPLE)
            hb = len(sim.bkt.history)
            s, r, done = sim.step(a)
            R += r; t += 1
            if a != WORKED_EXAMPLE:
                streak_wrong = streak_wrong + 1 if sim.bkt.history[-1] == 0 else 0
                maxwrong = max(maxwrong, streak_wrong)
        gains.append(sim.p_L - p0)
        steps_to_mastery.append(t if sim.p_L >= MASTERY_THRESHOLD else 12)
        rewards.append(R); frustr.append(maxwrong >= 3)
    return {"mean_reward": float(np.mean(rewards)), "mean_mastery_gain": float(np.mean(gains)),
            "mean_steps_to_mastery": float(np.mean(steps_to_mastery)),
            "pct_frustrated(3+ wrong in a row)": float(np.mean(frustr)) * 100,
            "worked_examples_per_student": worked / n}

def evaluate_policies(Q: np.ndarray, n: int = 2000, seed: int = 123) -> Dict[str, Dict[str, float]]:
    def fixed(d):  return lambda sim, s, rng: d - 1
    def rand(sim, s, rng): return int(rng.integers(0, 5))
    def staircase(sim, s, rng):
        h = sim.bkt.history; D = sim.bkt.last_difficulty
        if not h: return D
        return min(4, D + 1) if h[-1] else max(0, D - 1)
    def zpd(sim, s, rng):   # oracle-ish heuristic: difficulty with P(correct) closest to 0.7
        return int(np.argmin([abs(sim.bkt.p_correct(d) - 0.7) for d in range(1, 6)]))
    def q_greedy(sim, s, rng): return choose_action(Q, s, rng=rng, greedy=True)[0]
    def q_sample(sim, s, rng): return choose_action(Q, s, rng=rng)[0]
    pols = {"fixed_difficulty_3": fixed(3), "random": rand, "staircase": staircase,
            "zpd_heuristic (sees true p_L)": zpd, "Q-policy (greedy)": q_greedy, "Q-policy (sampled, deployed)": q_sample}
    return {k: _run_policy(v, n, seed) for k, v in pols.items()}


# ---------------------------------------------------------------- logs for OPE
def generate_synthetic_logs(num_students: int = 300, seed: int = 7) -> List[Dict]:
    """Behaviour policy = 60% staircase rule + 40% uniform (guarantees support)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_students):
        sim = StudentSimulator(rng)
        s, done = sim.reset(), False
        traj = {"states": [], "actions": [], "rewards": [], "behavior_probs": []}
        while not done:
            mask = action_mask(s)
            D = decode_state(s)[1]; h = sim.bkt.history
            rule = D if not h else (min(4, D + 1) if h[-1] else max(0, D - 1))
            probs = np.where(mask, 0.4 / mask.sum(), 0.0)
            probs[rule if mask[rule] else int(np.flatnonzero(mask)[0])] += 0.6
            probs = probs / probs.sum()
            a = int(rng.choice(NUM_ACTIONS, p=probs))
            s2, r, done = sim.step(a)
            traj["states"].append(s); traj["actions"].append(a)
            traj["rewards"].append(r); traj["behavior_probs"].append(float(probs[a]))
            s = s2
        out.append(traj)
    return out
