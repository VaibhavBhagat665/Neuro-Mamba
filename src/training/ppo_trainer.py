"""
Mamba PPO Trainer — PPO training loop for sequential Mamba policies.

Key difference from standard PPO:
    Standard PPO treats each observation independently (MLP).
    Mamba PPO must process **sequences** to leverage the SSM memory.
    The rollout buffer stores sequences of length L, not independent tuples.

References:
    [1] Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.
    [2] Engstrom et al., "Implementation Matters in Deep Policy Gradients",
        ICLR 2020 — PPO implementation best practices.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from src.models.mamba_actor_critic import MambaActorCritic
from src.safety.safety_hallucination import get_shield_penalty
from src.training.rollout_buffer import SequenceRolloutBuffer


class MambaPPOTrainer:
    """Full PPO training loop designed for the Mamba actor-critic.

    Parameters
    ----------
    env : gym.Env
        Should be wrapped with ``CBFShield`` ➜ ``SafetyHallucinationWrapper``.
    model : MambaActorCritic
        The policy / value network.
    device : str
        ``"cuda"`` or ``"cpu"``.
    n_steps : int
        Steps collected per rollout.
    n_epochs : int
        SGD epochs over each rollout.
    batch_size : int
        Number of sequences per minibatch.
    gamma, gae_lambda : float
        Discount and GAE parameters.
    clip_range : float
        PPO clipping epsilon.
    ent_coef : float
        Entropy bonus coefficient.
    vf_coef : float
        Value-loss coefficient.
    max_grad_norm : float
        Gradient clipping.
    lr : float
        Adam learning rate.
    seq_len : int
        Sequence length L for Mamba parallel scan.
    total_timesteps : int
        Total training budget.
    checkpoint_dir : str
        Where to save model checkpoints.
    checkpoint_interval : int
        Save a checkpoint every N timesteps.
    use_hallucination_curriculum : bool
        Whether to anneal the shield penalty over training.
    kappa_start, kappa_end : float
        Curriculum bounds for the hallucination penalty.
    use_wandb : bool
        Whether to log to Weights & Biases.
    wandb_project : str
        WandB project name.
    """

    def __init__(
        self,
        env,
        model: MambaActorCritic,
        *,
        device: str = "cuda",
        # PPO
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        lr: float = 3e-4,
        # Sequence
        seq_len: int = 32,
        # Training
        total_timesteps: int = 1_000_000,
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 100_000,
        # Hallucination curriculum
        use_hallucination_curriculum: bool = True,
        kappa_start: float = 10.0,
        kappa_end: float = 2.0,
        # Logging
        use_wandb: bool = True,
        wandb_project: str = "neuro-mamba-shield",
    ):
        self.env = env
        self.model = model.to(device)
        self.device = device

        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.seq_len = seq_len
        self.total_timesteps = total_timesteps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.use_curriculum = use_hallucination_curriculum
        self.kappa_start = kappa_start
        self.kappa_end = kappa_end
        self.use_wandb = use_wandb and _WANDB_AVAILABLE

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

        self.buffer = SequenceRolloutBuffer(
            n_steps=n_steps,
            obs_dim=model.obs_dim,
            action_dim=model.action_dim,
            seq_len=seq_len,
        )

        os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Rollout Collection ───────────────────────────────────

    def collect_rollout(self) -> dict[str, Any]:
        """Collect ``n_steps`` of experience and fill the buffer."""
        self.buffer.reset()
        obs, _ = self.env.reset()
        episode_returns: list[float] = []
        episode_violations: list[float] = []
        ep_ret, ep_viol = 0.0, 0.0

        for _ in range(self.n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value, _ = self.model.get_action_and_value(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            action_np = np.clip(
                action_np,
                self.env.action_space.low,
                self.env.action_space.high,
            )

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.add(
                obs=obs,
                action=action_np,
                reward=reward,
                log_prob=log_prob.item(),
                value=value.item(),
                done=float(done),
                shield=float(info.get("shield_activated", 0)),
            )

            ep_ret += reward
            ep_viol += info.get("cost", 0.0)

            if done:
                episode_returns.append(ep_ret)
                episode_violations.append(ep_viol)
                ep_ret, ep_viol = 0.0, 0.0
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        # Compute bootstrap value for GAE
        last_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, last_value, _ = self.model.get_action_and_value(last_obs_t)

        self.buffer.compute_gae(last_value.item(), self.gamma, self.gae_lambda)

        return {
            "episode_returns": episode_returns,
            "episode_violations": episode_violations,
            "shield_rate": self.buffer.shield_rate,
            "mean_reward": self.buffer.mean_reward,
        }

    # ── PPO Update ───────────────────────────────────────────

    def ppo_update(self) -> dict[str, float]:
        """Run PPO update epochs over sequence batches."""
        batches = self.buffer.get_sequence_batches()
        n_seqs = len(batches["obs"])

        total_pl, total_vl, total_ent, n_updates = 0.0, 0.0, 0.0, 0

        for _ in range(self.n_epochs):
            idxs = np.random.permutation(n_seqs)

            for start in range(0, n_seqs, self.batch_size):
                mb_idx = idxs[start : start + self.batch_size]
                if len(mb_idx) == 0:
                    continue

                obs_b = torch.FloatTensor(batches["obs"][mb_idx]).to(self.device)
                act_b = torch.FloatTensor(batches["actions"][mb_idx]).to(self.device)
                adv_b = torch.FloatTensor(batches["advantages"][mb_idx]).to(self.device)
                ret_b = torch.FloatTensor(batches["returns"][mb_idx]).to(self.device)
                old_lp = torch.FloatTensor(batches["log_probs"][mb_idx]).to(self.device)

                # Normalize advantages
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                # Forward
                new_lp, value, entropy = self.model.evaluate_actions(obs_b, act_b)

                # Clipped policy loss
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                ) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(value, ret_b)

                # Combined loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            "policy_loss": total_pl / denom,
            "value_loss": total_vl / denom,
            "entropy": total_ent / denom,
        }

    # ── Checkpointing ────────────────────────────────────────

    def save_checkpoint(self, timestep: int) -> str:
        path = os.path.join(self.checkpoint_dir, f"model_{timestep}.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "timestep": timestep,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt.get("timestep", 0)

    # ── Main Training Loop ───────────────────────────────────

    def train(self) -> dict[str, list]:
        """Run the full training loop.

        Returns a dict of logged metrics for post-hoc analysis.
        """
        if self.use_wandb:
            wandb.init(
                project="neuro-mamba-shield",
                config={
                    "d_model": self.model.d_model,
                    "n_steps": self.n_steps,
                    "seq_len": self.seq_len,
                    "total_timesteps": self.total_timesteps,
                    "clip_range": self.clip_range,
                    "ent_coef": self.ent_coef,
                },
            )

        history: dict[str, list] = {
            "timestep": [],
            "mean_reward": [],
            "shield_rate": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "episode_returns": [],
            "episode_violations": [],
        }

        timestep = 0
        iteration = 0

        while timestep < self.total_timesteps:
            # ── Curriculum: update κ ─────────────────────────
            if self.use_curriculum:
                kappa = get_shield_penalty(
                    timestep,
                    self.total_timesteps,
                    self.kappa_start,
                    self.kappa_end,
                )
                # Update the hallucination wrapper (if wrapped)
                if hasattr(self.env, "set_penalty"):
                    self.env.set_penalty(kappa)

            # ── Collect rollout ──────────────────────────────
            rollout_info = self.collect_rollout()
            timestep += self.n_steps
            iteration += 1

            # ── PPO update ───────────────────────────────────
            update_info = self.ppo_update()

            # ── Logging ──────────────────────────────────────
            history["timestep"].append(timestep)
            history["mean_reward"].append(rollout_info["mean_reward"])
            history["shield_rate"].append(rollout_info["shield_rate"])
            history["policy_loss"].append(update_info["policy_loss"])
            history["value_loss"].append(update_info["value_loss"])
            history["entropy"].append(update_info["entropy"])
            history["episode_returns"].extend(rollout_info["episode_returns"])
            history["episode_violations"].extend(rollout_info["episode_violations"])

            if self.use_wandb:
                log_dict = {
                    "timestep": timestep,
                    "mean_reward": rollout_info["mean_reward"],
                    "shield_rate": rollout_info["shield_rate"],
                    **update_info,
                }
                if rollout_info["episode_returns"]:
                    log_dict["ep_return_mean"] = np.mean(
                        rollout_info["episode_returns"]
                    )
                    log_dict["ep_violation_mean"] = np.mean(
                        rollout_info["episode_violations"]
                    )
                wandb.log(log_dict)

            # ── Console ──────────────────────────────────────
            if timestep % self.checkpoint_interval < self.n_steps:
                self.save_checkpoint(timestep)
                ep_rets = rollout_info["episode_returns"]
                avg_ret = np.mean(ep_rets) if ep_rets else 0.0
                print(
                    f"[{timestep:>9,d}]  "
                    f"Return: {avg_ret:7.2f} | "
                    f"Shield: {rollout_info['shield_rate']:.3f} | "
                    f"PL: {update_info['policy_loss']:.4f} | "
                    f"VL: {update_info['value_loss']:.4f} | "
                    f"Ent: {update_info['entropy']:.4f}"
                )

        # Final save
        self.save_checkpoint(timestep)

        if self.use_wandb:
            wandb.finish()

        return history
