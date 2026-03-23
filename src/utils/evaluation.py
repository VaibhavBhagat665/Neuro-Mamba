"""
Evaluation Utilities — Pareto Frontier Construction.

Provides functions for running evaluation episodes and plotting the
Return-vs-Violations Pareto frontier for ablation comparisons.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def run_evaluation(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env,
    n_episodes: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``n_episodes`` and collect per-episode returns and violations.

    Parameters
    ----------
    policy_fn : callable
        Maps observation (np.ndarray) → action (np.ndarray).
    env : gym.Env
        Environment (possibly shield-wrapped).
    n_episodes : int
        Number of evaluation episodes.

    Returns
    -------
    returns : np.ndarray, shape (n_episodes,)
        Cumulative return per episode.
    violations : np.ndarray, shape (n_episodes,)
        Cumulative safety cost per episode.
    """
    episode_returns: list[float] = []
    episode_violations: list[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_violations = 0.0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_violations += info.get("cost", 0.0)
            done = terminated or truncated

        episode_returns.append(ep_return)
        episode_violations.append(ep_violations)

    return np.array(episode_returns), np.array(episode_violations)


def plot_pareto_frontier(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: str = "results/pareto_frontier.png",
    dpi: int = 200,
) -> None:
    """Plot the Pareto frontier: Return (Y) vs Safety Violations (X).

    Upper-left = best: high return, low violations.

    Parameters
    ----------
    results : dict
        ``{label: (returns_array, violations_array)}``.
        Expected keys:

        * ``"PPO-MLP (Baseline)"``
        * ``"PPO-LSTM (Baseline)"``
        * ``"Neuro-Mamba No Shield"``
        * ``"Neuro-Mamba (Ours)"``
    save_path : str
        File path for the saved figure.
    dpi : int
        Figure resolution.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    colors = {
        "PPO-MLP (Baseline)": "#e74c3c",
        "PPO-LSTM (Baseline)": "#e67e22",
        "Neuro-Mamba No Shield": "#3498db",
        "Neuro-Mamba (Ours)": "#27ae60",
    }
    markers = {
        "PPO-MLP (Baseline)": "o",
        "PPO-LSTM (Baseline)": "s",
        "Neuro-Mamba No Shield": "^",
        "Neuro-Mamba (Ours)": "D",
    }

    for label, (returns, violations) in results.items():
        color = colors.get(label, "#888888")
        marker = markers.get(label, "x")

        # Individual episode scatter (semi-transparent)
        ax.scatter(
            violations,
            returns,
            c=color,
            marker=marker,
            alpha=0.35,
            s=30,
        )

        # Mean point (large, outlined)
        ax.scatter(
            violations.mean(),
            returns.mean(),
            c=color,
            marker=marker,
            s=200,
            edgecolors="black",
            linewidth=1.5,
            zorder=5,
        )

        # Error bars (±1σ)
        ax.errorbar(
            violations.mean(),
            returns.mean(),
            xerr=violations.std(),
            yerr=returns.std(),
            fmt="none",
            color=color,
            alpha=0.8,
            capsize=4,
        )

    # Annotations
    ax.axvline(x=5, color="gray", linestyle="--", alpha=0.4, label="5 violations/ep")
    ax.text(
        0.02,
        0.95,
        "← Safety-Dominant Region",
        transform=ax.transAxes,
        ha="left",
        fontsize=9,
        color="gray",
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors.get(k, "#888888"), label=k)
        for k in results.keys()
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.set_xlabel("Cumulative Safety Violations per Episode", fontsize=13)
    ax.set_ylabel("Average Return (Goal Completion)", fontsize=13)
    ax.set_title(
        "Pareto Frontier: Return vs Safety Violations\n(Upper-Left = Best)",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pareto frontier plot to {save_path}")
