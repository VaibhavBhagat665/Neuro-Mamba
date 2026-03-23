"""
Mamba Actor-Critic for Safe Reinforcement Learning.

Implements the Mamba SSM (Gu & Dao, ICLR 2024) as a shared policy–value backbone
for on-policy PPO in continuous-control safety-gymnasium environments.

Architecture
────────────
  obs ──► InputProj (Linear + SiLU) ──► MambaEncoder (N × MambaBlock) ──┬─► PolicyHead → μ, σ
                                                                        └─► ValueHead  → V(s)

Key design choices:
  • Shared trunk reduces parameter count and improves feature reuse (Schulman et al., 2017).
  • Log-std is a *learned parameter*, not a network output — standard for continuous PPO.
  • The Mamba hidden state serves as implicit "safety memory": it can remember hazard
    positions observed many steps ago, unlike an MLP.  This is Novelty Claim #3.

References:
  [1] Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
      arXiv 2312.00752, ICLR 2024.
  [2] Dao & Gu, "Transformers are SSMs", ICML 2024 (Mamba-2 / SSD).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from mamba_ssm import Mamba


# ────────────────────────────────────────────────────────────
#  Building Blocks
# ────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Single Mamba layer with pre-norm residual connection.

    Follows the architecture in Gu & Dao (2023) Figure 3:
        x ──► LayerNorm ──► Mamba ──► (+x) ──► out
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,   # State expansion factor N — 16 is optimal per ablations
            d_conv=d_conv,     # Local convolution width — 4 is standard
            expand=expand,     # Inner dimension multiplier — 2× is standard
        )

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            inference_params: Mamba recurrent state cache (inference only).
        Returns:
            (B, L, d_model) with residual.
        """
        return x + self.mamba(self.norm(x), inference_params=inference_params)


class MambaEncoder(nn.Module):
    """Stack of ``n_layers`` MambaBlocks.

    During *training*  (L > 1): uses CUDA parallel scan — O(L) work, O(log L) depth.
    During *inference* (L = 1): uses recurrent mode via ``inference_params`` — O(1)/step.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, inference_params=inference_params)
        return x


# ────────────────────────────────────────────────────────────
#  Actor-Critic
# ────────────────────────────────────────────────────────────

class MambaActorCritic(nn.Module):
    """Full Actor-Critic with Mamba backbone for continuous Safe RL.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality (60 for SafetyPointGoal1-v0).
    action_dim : int
        Action dimensionality (2 for PointRobot: forward velocity, turn velocity).
    d_model : int
        Width of the Mamba trunk.
    n_layers : int
        Depth of the Mamba encoder.
    d_state : int
        SSM state expansion factor N.
    d_conv : int
        Mamba local convolution width.
    expand : int
        Mamba inner dimension multiplier.
    """

    def __init__(
        self,
        obs_dim: int = 60,
        action_dim: int = 2,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Input projection: obs_dim → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.SiLU(),  # Mamba internally uses SiLU (Swish); stay consistent
        )

        # Shared Mamba backbone
        self.backbone = MambaEncoder(
            d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Policy head (Actor) — Gaussian with learned log-std
        self.policy_mean = nn.Linear(d_model, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))

        # Value head (Critic)
        self.value_head = nn.Linear(d_model, 1)

        # Proper initialization (critical for stable PPO training)
        self._init_weights()

    # ── Weight Initialization ────────────────────────────────

    def _init_weights(self) -> None:
        """Orthogonal init following PPO best practices."""
        for name, param in self.named_parameters():
            if "policy_mean" in name and "weight" in name:
                nn.init.orthogonal_(param, gain=0.01)      # small init for policy
            elif "value_head" in name and "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "input_proj" in name and "weight" in name:
                nn.init.orthogonal_(param, gain=np.sqrt(2))

    # ── Forward ──────────────────────────────────────────────

    def forward(
        self,
        obs_seq: torch.Tensor,
        inference_params=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_seq: (B, L, obs_dim)
                Training:  L = rollout_length (e.g. 32).
                Inference: L = 1 (autoregressive).
            inference_params: Mamba recurrent cache for L = 1 mode.

        Returns:
            mu:    (B, L, action_dim)  — policy mean
            std:   (B, L, action_dim)  — policy standard deviation
            value: (B, L, 1)           — state value estimate
        """
        x = self.input_proj(obs_seq)                          # (B, L, d_model)
        x = self.backbone(x, inference_params=inference_params)  # (B, L, d_model)

        mu = self.policy_mean(x)                              # (B, L, action_dim)
        std = torch.exp(self.policy_log_std).expand_as(mu)    # (B, L, action_dim)
        value = self.value_head(x)                            # (B, L, 1)
        return mu, std, value

    # ── Single-step (inference) ──────────────────────────────

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        inference_params=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action for a *single* timestep.

        Args:
            obs: (B, obs_dim)

        Returns:
            action:   (B, action_dim)
            log_prob: (B,)
            value:    (B,)
            entropy:  (B,)
        """
        obs_seq = obs.unsqueeze(1)                           # (B, 1, obs_dim)
        mu, std, value = self.forward(obs_seq, inference_params)
        mu = mu.squeeze(1)                                   # (B, action_dim)
        std = std.squeeze(1)
        value = value.squeeze(1).squeeze(-1)                 # (B,)

        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, value, entropy

    # ── Batch evaluation (training) ──────────────────────────

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs / values for a *batch of sequences* (PPO update).

        Args:
            obs_seq:     (B, L, obs_dim)
            actions_seq: (B, L, action_dim)

        Returns:
            log_prob: (B, L)
            value:    (B, L)
            entropy:  (B, L)
        """
        mu, std, value = self.forward(obs_seq)               # training — no cache
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(actions_seq).sum(-1)         # sum over action dims
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(-1), entropy
