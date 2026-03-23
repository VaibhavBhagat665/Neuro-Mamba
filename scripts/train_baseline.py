#!/usr/bin/env python3
"""
Train PPO-MLP and PPO-LSTM Baselines on SafetyPointGoal1-v0.

These baselines are needed for the Pareto frontier comparison.
Results are saved as .npy files in ``results/``.

Usage:
    python scripts/train_baseline.py                    # both baselines
    python scripts/train_baseline.py --variant mlp      # MLP only
    python scripts/train_baseline.py --variant lstm     # LSTM only
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


# ── Callback ─────────────────────────────────────────────────


class SafetyMetricCallback(BaseCallback):
    """Track per-episode returns and cumulative safety violations."""

    def __init__(self):
        super().__init__()
        self.episode_violations: list[float] = []
        self.episode_returns: list[float] = []
        self._cur_viol = 0.0
        self._cur_ret = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            self._cur_viol += info.get("cost", 0.0)
            self._cur_ret += self.locals["rewards"][0]
            if info.get("terminated", False) or info.get("truncated", False):
                self.episode_violations.append(self._cur_viol)
                self.episode_returns.append(self._cur_ret)
                self._cur_viol = 0.0
                self._cur_ret = 0.0
        return True


# ── Training Functions ───────────────────────────────────────


def train_ppo_mlp(
    total_timesteps: int = 1_000_000,
    save_dir: str = "results",
) -> None:
    """Train a standard PPO-MLP baseline (Variant A)."""
    print("=" * 60)
    print("Training PPO-MLP Baseline (Variant A)")
    print("=" * 60)

    env = DummyVecEnv([lambda: safety_gymnasium.make("SafetyPointGoal1-v0")])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
    )

    callback = SafetyMetricCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "baseline_mlp_returns.npy"), callback.episode_returns)
    np.save(os.path.join(save_dir, "baseline_mlp_violations.npy"), callback.episode_violations)
    model.save(os.path.join(save_dir, "ppo_mlp_model"))
    print(f"PPO-MLP results saved to {save_dir}/")


def train_ppo_lstm(
    total_timesteps: int = 1_000_000,
    save_dir: str = "results",
) -> None:
    """Train a PPO-LSTM baseline (Variant B)."""
    print("=" * 60)
    print("Training PPO-LSTM Baseline (Variant B)")
    print("=" * 60)

    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        print("ERROR: sb3-contrib not installed.  pip install sb3-contrib")
        return

    env = DummyVecEnv([lambda: safety_gymnasium.make("SafetyPointGoal1-v0")])

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        policy_kwargs=dict(lstm_hidden_size=64),
        verbose=1,
    )

    callback = SafetyMetricCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "baseline_lstm_returns.npy"), callback.episode_returns)
    np.save(os.path.join(save_dir, "baseline_lstm_violations.npy"), callback.episode_violations)
    model.save(os.path.join(save_dir, "ppo_lstm_model"))
    print(f"PPO-LSTM results saved to {save_dir}/")


# ── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO baselines")
    parser.add_argument(
        "--variant",
        choices=["mlp", "lstm", "both"],
        default="both",
        help="Which baseline to train (default: both)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1M)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Output directory for models and metrics",
    )
    args = parser.parse_args()

    if args.variant in ("mlp", "both"):
        train_ppo_mlp(args.timesteps, args.save_dir)
    if args.variant in ("lstm", "both"):
        train_ppo_lstm(args.timesteps, args.save_dir)
