#!/usr/bin/env python3
"""
Train Neuro-Mamba Agent — Ablation Variants A–E.

Variants:
    A — PPO-MLP          (run via train_baseline.py instead)
    B — PPO-LSTM         (run via train_baseline.py instead)
    C — Mamba-Only       (no shield, no hallucination)
    D — Mamba + Shield   (CBF-QP shield, no hallucination)
    E — Ours             (CBF-QP shield + hallucination curriculum)  ← FULL SYSTEM

Usage:
    python scripts/train_mamba.py --variant E                 # recommended
    python scripts/train_mamba.py --variant C                 # ablation: no shield
    python scripts/train_mamba.py --variant D                 # ablation: no hallucination
    python scripts/train_mamba.py --config config/custom.yaml # custom config
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml
import numpy as np
import torch
import safety_gymnasium

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mamba_actor_critic import MambaActorCritic
from src.safety.cbf_shield import CBFShield
from src.safety.safety_hallucination import SafetyHallucinationWrapper
from src.training.ppo_trainer import MambaPPOTrainer


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, variant: str):
    """Construct the environment stack based on ablation variant.

    C — raw env (Mamba-Only)
    D — env → CBFShield
    E — env → CBFShield → SafetyHallucinationWrapper
    """
    env = safety_gymnasium.make(
        cfg["environment"]["env_id"],
        render_mode=None,
    )

    if variant in ("D", "E"):
        shield_cfg = cfg["shield"]
        env = CBFShield(
            env,
            d_safe=shield_cfg["d_safe"],
            gamma=shield_cfg["gamma_cbf"],
            dt=shield_cfg["dt"],
            max_speed=shield_cfg["max_speed"],
            lidar_hazard_idx_start=shield_cfg["lidar_hazard_start"],
            lidar_hazard_idx_end=shield_cfg["lidar_hazard_end"],
            n_lidar_bins=shield_cfg["n_lidar_bins"],
            lidar_max_dist=shield_cfg["lidar_max_dist"],
        )

    if variant == "E":
        hall_cfg = cfg["hallucination"]
        env = SafetyHallucinationWrapper(
            env,
            shield_penalty=hall_cfg["kappa_start"],
        )

    return env


def build_model(cfg: dict, device: str) -> MambaActorCritic:
    """Instantiate the MambaActorCritic from config."""
    model_cfg = cfg["model"]
    env_cfg = cfg["environment"]
    return MambaActorCritic(
        obs_dim=env_cfg["obs_dim"],
        action_dim=env_cfg["action_dim"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        d_state=model_cfg["d_state"],
        d_conv=model_cfg["d_conv"],
        expand=model_cfg["expand"],
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Neuro-Mamba agent (ablation variants C/D/E)"
    )
    parser.add_argument(
        "--variant",
        choices=["C", "D", "E"],
        default="E",
        help="Ablation variant (default: E = full system)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda / cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory override",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # ── Configuration ────────────────────────────────────────
    cfg = load_config(args.config)
    device = args.device or cfg["training"]["device"]
    seed = args.seed or cfg["training"]["seed"]
    total_timesteps = args.timesteps or cfg["training"]["total_timesteps"]
    ckpt_dir = args.checkpoint_dir or os.path.join(
        "checkpoints", f"variant_{args.variant}"
    )

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print(f"  Neuro-Mamba — Variant {args.variant}")
    print(f"  Device: {device}  |  Seed: {seed}  |  Steps: {total_timesteps:,}")
    print("=" * 60)

    # ── Build Environment & Model ────────────────────────────
    env = build_env(cfg, args.variant)
    model = build_model(cfg, device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,}")

    # ── Build Trainer ────────────────────────────────────────
    ppo_cfg = cfg["ppo"]
    hall_cfg = cfg["hallucination"]

    trainer = MambaPPOTrainer(
        env=env,
        model=model,
        device=device,
        n_steps=ppo_cfg["n_steps"],
        n_epochs=ppo_cfg["n_epochs"],
        batch_size=ppo_cfg["batch_size"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        lr=ppo_cfg["learning_rate"],
        seq_len=ppo_cfg["seq_len"],
        total_timesteps=total_timesteps,
        checkpoint_dir=ckpt_dir,
        checkpoint_interval=cfg["training"]["checkpoint_interval"],
        use_hallucination_curriculum=(args.variant == "E"),
        kappa_start=hall_cfg["kappa_start"],
        kappa_end=hall_cfg["kappa_end"],
        use_wandb=True,
        wandb_project=cfg["wandb"]["project"],
    )

    # ── Resume from checkpoint if requested ──────────────────
    if args.resume:
        start_ts = trainer.load_checkpoint(args.resume)
        print(f"  Resumed from checkpoint at timestep {start_ts:,}")

    # ── Train ────────────────────────────────────────────────
    history = trainer.train()

    # ── Save final metrics ───────────────────────────────────
    os.makedirs("results", exist_ok=True)
    tag = f"variant_{args.variant}"
    np.save(f"results/{tag}_returns.npy", history["episode_returns"])
    np.save(f"results/{tag}_violations.npy", history["episode_violations"])
    print(f"\nTraining complete. Results saved to results/{tag}_*.npy")


if __name__ == "__main__":
    main()
