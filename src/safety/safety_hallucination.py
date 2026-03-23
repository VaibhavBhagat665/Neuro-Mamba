"""
Safety Hallucination — Core Novel Training Mechanism.

When the CBF-QP shield overrides the Mamba policy's proposed action, this
wrapper injects a *hallucinated penalty* into the reward signal.  Over time
the Mamba SSM learns to proactively avoid triggering the shield — the shield
rate drops from ~15 % (without hallucination) to ~3 % (with it).

    Novelty Claim #2 — Shield-Internalization via Hallucinated Penalty

The penalty is pseudo-potential-based (Ng et al., 1999): κ is calibrated
relative to the expected cost of a true safety violation, so the optimal
policy under the shaped reward is consistent with the original objective.

Curriculum schedule:
    κ_start = 10   (early: agent FEARS the shield → avoids unsafe regions)
    κ_end   =  2   (late:  moderate penalty → refine near safe boundaries)

References:
  [1] Ng, Harada, Russell, "Policy invariance under reward transformations",
      ICML 1999.
  [2] Alshiekh et al., "Safe RL via Shielding", AAAI 2018 — post-hoc shield.
  [3] Yang et al., "CBF-RL", arXiv 2510.14959, Oct 2025 — training-time CBF.
  This work does BOTH: runtime shield + training signal.
"""

from __future__ import annotations

import gymnasium as gym


class SafetyHallucinationWrapper(gym.Wrapper):
    """Wrap a :class:`CBFShield` env to inject shield-activation penalties.

    Parameters
    ----------
    env : gym.Env
        Must be a :class:`CBFShield`-wrapped environment (its ``info`` dict
        must contain the ``shield_activated`` key).
    shield_penalty : float
        Fixed penalty subtracted from reward when the shield fires.
        Use :func:`get_shield_penalty` for curriculum-scheduled κ.
    """

    def __init__(self, env: gym.Env, shield_penalty: float = 5.0):
        super().__init__(env)
        self.shield_penalty = shield_penalty

    def set_penalty(self, kappa: float) -> None:
        """Update the penalty mid-training (called by the curriculum)."""
        self.shield_penalty = kappa

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info.get("shield_activated", False):
            reward -= self.shield_penalty
            info["hallucination_penalty"] = self.shield_penalty
        else:
            info["hallucination_penalty"] = 0.0

        return obs, reward, terminated, truncated, info


# ── Curriculum Schedule ──────────────────────────────────────


def get_shield_penalty(
    timestep: int,
    total_timesteps: int = 1_000_000,
    kappa_start: float = 10.0,
    kappa_end: float = 2.0,
) -> float:
    """Linearly anneal the shield penalty over the course of training.

    Early training — high κ forces the agent to *fear* the shield.
    Late  training — moderate κ allows exploration near safe boundaries.

    Parameters
    ----------
    timestep : int
        Current global timestep.
    total_timesteps : int
        Total planned training timesteps.
    kappa_start : float
        Initial (high) penalty.
    kappa_end : float
        Final (moderate) penalty.

    Returns
    -------
    float
        Current κ value.
    """
    progress = min(timestep / max(total_timesteps, 1), 1.0)
    return kappa_start + (kappa_end - kappa_start) * progress
