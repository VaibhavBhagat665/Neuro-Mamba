"""
Side-by-Side Comparison Video Renderer.

Renders baseline (left) vs Neuro-Mamba (right) with a flashing red border
on the right panel when the CBF-QP shield activates.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import imageio

    _IMAGEIO_AVAILABLE = True
except ImportError:
    _IMAGEIO_AVAILABLE = False


# ── HUD Helpers ──────────────────────────────────────────────


def _add_label(
    frame: np.ndarray, text: str, color: tuple[int, int, int]
) -> np.ndarray:
    """Overlay a text label onto the top-left of the frame."""
    if not _CV2_AVAILABLE:
        return frame
    frame = frame.copy()
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def _add_shield_flash(frame: np.ndarray) -> np.ndarray:
    """Flash a red border and add a 'SHIELD ACTIVE' label."""
    if not _CV2_AVAILABLE:
        return frame
    frame = frame.copy()
    t = 8  # border thickness
    red = [255, 30, 30]
    frame[:t, :] = red
    frame[-t:, :] = red
    frame[:, :t] = red
    frame[:, -t:] = red
    cv2.putText(
        frame,
        "SHIELD ACTIVE",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 30, 30),
        2,
    )
    return frame


# ── Main Renderer ────────────────────────────────────────────


def render_side_by_side_comparison(
    baseline_policy: Callable[[np.ndarray], np.ndarray],
    mamba_policy: Callable[[np.ndarray], np.ndarray],
    env_fn: Callable,
    output_path: str = "results/comparison.mp4",
    n_frames: int = 1000,
    fps: int = 30,
) -> None:
    """Render a side-by-side MP4: Baseline (left) | Neuro-Mamba (right).

    A red border flashes on the right panel whenever the shield activates.

    Parameters
    ----------
    baseline_policy : callable
        obs → action for the baseline agent.
    mamba_policy : callable
        obs → action for the Mamba agent (env must include CBFShield).
    env_fn : callable
        Zero-argument factory that returns a fresh environment instance.
        The Mamba env should be shield-wrapped.
    output_path : str
        Where to save the MP4 video.
    n_frames : int
        Total frames to render.
    fps : int
        Frames per second.
    """
    if not _IMAGEIO_AVAILABLE:
        raise ImportError("render requires `imageio` and `imageio-ffmpeg`.")

    env_baseline = env_fn()
    env_mamba = env_fn()

    obs_b, _ = env_baseline.reset()
    obs_m, _ = env_mamba.reset()

    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=8
    )

    for _ in range(n_frames):
        # --- Baseline step ---
        act_b = baseline_policy(obs_b)
        obs_b, _, term_b, trunc_b, _ = env_baseline.step(act_b)
        frame_b = env_baseline.render()
        if term_b or trunc_b:
            obs_b, _ = env_baseline.reset()

        # --- Mamba step ---
        act_m = mamba_policy(obs_m)
        obs_m, _, term_m, trunc_m, info_m = env_mamba.step(act_m)
        frame_m = env_mamba.render()
        shield_on = info_m.get("shield_activated", False)
        if term_m or trunc_m:
            obs_m, _ = env_mamba.reset()

        # --- HUD overlays ---
        if frame_b is not None and frame_m is not None:
            frame_b = _add_label(frame_b, "PPO-MLP BASELINE", (220, 50, 50))
            frame_m = _add_label(frame_m, "NEURO-MAMBA (OURS)", (50, 200, 100))
            if shield_on:
                frame_m = _add_shield_flash(frame_m)
            combined = np.concatenate([frame_b, frame_m], axis=1)
            writer.append_data(combined)

    writer.close()
    print(f"Video saved to {output_path}")
