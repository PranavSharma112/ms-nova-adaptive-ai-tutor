"""Train the tutoring policy and save models/q_table.npy + models/train_report.json.
Usage:  python train_policy.py [--episodes 30000] [--seed 0]"""
import argparse, json, os
import numpy as np
from rl_engine import train_q, evaluate_policies, DEFAULT_TAU

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs("models", exist_ok=True)
    Q = train_q(episodes=a.episodes, seed=a.seed)
    np.save("models/q_table.npy", Q)
    rep = evaluate_policies(Q, n=a.eval)
    json.dump(rep, open("models/train_report.json", "w"), indent=2)
    print(f"{'policy':34s} {'reward':>7s} {'gain':>6s} {'steps':>6s} {'frustr%':>8s} {'worked':>7s}")
    for k, v in rep.items():
        print(f"{k:34s} {v['mean_reward']:7.3f} {v['mean_mastery_gain']:6.3f} "
              f"{v['mean_steps_to_mastery']:6.2f} {v['pct_frustrated(3+ wrong in a row)']:8.1f} {v['worked_examples_per_student']:7.2f}")
