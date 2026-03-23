"""
Sequence-Aware Rollout Buffer for Mamba PPO.

Standard PPO rollout buffers treat each transition (s, a, r) as independent.
With a Mamba backbone the SSM hidden state is meaningful only across *sequences*:
we must chunk rollouts into (B, L, dim) tensors so the parallel scan can learn
from trajectory context.

This module also provides GAE (Generalized Advantage Estimation) computation.
"""

from __future__ import annotations

import numpy as np


class SequenceRolloutBuffer:
    """Stores a single rollout and provides sequence-chunked batches.

    Usage::

        buf = SequenceRolloutBuffer(n_steps=2048, obs_dim=60, action_dim=2, seq_len=32)
        for t in range(n_steps):
            buf.add(obs, action, reward, log_prob, value, done, shield_flag)
        buf.compute_gae(last_value, gamma, gae_lambda)
        batches = buf.get_sequence_batches(batch_size=64, device="cuda")
        buf.reset()

    Parameters
    ----------
    n_steps : int
        Number of environment steps per rollout.
    obs_dim : int
        Observation dimensionality.
    action_dim : int
        Action dimensionality.
    seq_len : int
        Mamba training sequence length L (must divide ``n_steps``).
    """

    def __init__(
        self,
        n_steps: int = 2048,
        obs_dim: int = 60,
        action_dim: int = 2,
        seq_len: int = 32,
    ):
        assert (
            n_steps % seq_len == 0
        ), f"n_steps ({n_steps}) must be divisible by seq_len ({seq_len})"

        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.n_seqs = n_steps // seq_len  # number of sequences per rollout

        self._ptr = 0
        self._allocate()

    def _allocate(self) -> None:
        self.obs = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros(self.n_steps, dtype=np.float32)
        self.shields = np.zeros(self.n_steps, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns = np.zeros(self.n_steps, dtype=np.float32)

    # ── Data Collection ──────────────────────────────────────

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
        done: float,
        shield: float = 0.0,
    ) -> None:
        """Append a single transition."""
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.log_probs[self._ptr] = log_prob
        self.values[self._ptr] = value
        self.dones[self._ptr] = done
        self.shields[self._ptr] = shield
        self._ptr += 1

    def reset(self) -> None:
        """Clear buffer for next rollout."""
        self._ptr = 0
        self._allocate()

    # ── GAE ──────────────────────────────────────────────────

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE (Schulman et al., 2015) over the full rollout.

        Must be called *after* all ``n_steps`` transitions are added and
        *before* ``get_sequence_batches``.
        """
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    # ── Sequence Batching ────────────────────────────────────

    def get_sequence_batches(self) -> dict[str, np.ndarray]:
        """Chunk the flat rollout into ``(n_seqs, seq_len, dim)`` tensors.

        Returns a dict ready for :meth:`MambaPPOTrainer.ppo_update`.
        """
        def _reshape(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 1:
                return arr.reshape(self.n_seqs, self.seq_len)
            return arr.reshape(self.n_seqs, self.seq_len, -1)

        return {
            "obs": _reshape(self.obs),           # (S, L, obs_dim)
            "actions": _reshape(self.actions),   # (S, L, act_dim)
            "advantages": _reshape(self.advantages),  # (S, L)
            "returns": _reshape(self.returns),        # (S, L)
            "log_probs": _reshape(self.log_probs),    # (S, L)
        }

    # ── Diagnostics ──────────────────────────────────────────

    @property
    def shield_rate(self) -> float:
        """Fraction of steps where the shield activated."""
        if self._ptr == 0:
            return 0.0
        return float(self.shields[: self._ptr].mean())

    @property
    def mean_reward(self) -> float:
        if self._ptr == 0:
            return 0.0
        return float(self.rewards[: self._ptr].mean())
