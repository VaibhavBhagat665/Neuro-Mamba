"""
Deterministic CBF-QP Safety Shield for SafetyPointGoal1-v0.

Implements a Control Barrier Function (CBF) with a Quadratic Program (QP) solver
that intercepts unsafe actions at every timestep, providing provable safety guarantees
under known dynamics.

Mathematical Foundation
───────────────────────
A Control Barrier Function h(x): ℝⁿ → ℝ defines a safe set C = {x : h(x) ≥ 0}.
A controller u is safe iff:

    ḣ(x, u) + α(h(x)) ≥ 0          (CBF condition)

For the Point robot:
    h(x) = d_obstacle − d_safe       (barrier function)
    ḣ(x, u) = −v·cos(θ_obs)·u_fwd   (time derivative under control u)

The shield solves:
    u* = argmin_u ‖u − u_proposed‖²
         s.t.  ḣ(x, u) + γ·h(x) ≥ 0
               u_min ≤ u ≤ u_max

References:
  [1] Alshiekh et al., "Safe RL via Shielding", AAAI 2018.
  [2] Hsu et al., "Shields for Safe RL", CACM 2025.
  [3] Yang et al., "CBF-RL", arXiv 2510.14959, Oct 2025.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

try:
    import osqp
    import scipy.sparse as sp

    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False


class CBFShield(gym.Wrapper):
    """Gymnasium wrapper that applies a CBF-QP shield to every action.

    If the proposed action satisfies the CBF condition, it passes through
    unchanged.  Otherwise, the shield solves a QP for the minimum-norm
    correction that restores safety.

    Parameters
    ----------
    env : gym.Env
        Unwrapped (or lightly wrapped) SafetyGymnasium environment.
    d_safe : float
        Minimum safe distance to hazard (meters).
    gamma : float
        CBF class-K gain.  Higher → more conservative.
    dt : float
        Environment timestep (seconds).
    max_speed : float
        Maximum forward velocity of the PointRobot.
    lidar_hazard_idx_start / _end : int
        Slice indices for hazard LiDAR in the observation vector.
    n_lidar_bins : int
        Number of angular LiDAR bins.
    lidar_max_dist : float
        Maximum range of the LiDAR sensor.
    """

    def __init__(
        self,
        env: gym.Env,
        d_safe: float = 0.25,
        gamma: float = 1.0,
        dt: float = 0.02,
        max_speed: float = 0.5,
        lidar_hazard_idx_start: int = 16,
        lidar_hazard_idx_end: int = 32,
        n_lidar_bins: int = 16,
        lidar_max_dist: float = 3.0,
    ):
        super().__init__(env)
        if not _OSQP_AVAILABLE:
            raise ImportError(
                "CBFShield requires `osqp` and `scipy`.  "
                "Install with:  pip install osqp scipy"
            )

        self.d_safe = d_safe
        self.gamma = gamma
        self.dt = dt
        self.max_speed = max_speed
        self.lidar_start = lidar_hazard_idx_start
        self.lidar_end = lidar_hazard_idx_end
        self.n_bins = n_lidar_bins
        self.lidar_max_dist = lidar_max_dist

        # Tracking
        self.shield_activated_count = 0
        self.total_steps = 0

        # OSQP solver (lazily initialised on first QP call)
        self._solver: osqp.OSQP | None = None
        self._solver_initialized = False

        # Cached observation for use in step()
        self._last_obs: np.ndarray | None = None

    # ── LiDAR Decoding ───────────────────────────────────────

    def _get_hazard_distance_and_angle(
        self, obs: np.ndarray
    ) -> tuple[float, float]:
        """Decode minimum hazard distance and its bearing from LiDAR.

        SafetyGymnasium encodes LiDAR as:
            obs_lidar[i] = exp(−d_i / lidar_max_dist)
        so:
            d_i = −ln(obs_lidar[i]) × lidar_max_dist

        Returns:
            d_min:          closest hazard distance (metres)
            theta_obstacle: bearing of closest hazard (rad, ∈ [−π, π])
        """
        lidar_raw = obs[self.lidar_start : self.lidar_end]
        lidar_clipped = np.clip(lidar_raw, 1e-8, 1.0)
        distances = -np.log(lidar_clipped) * self.lidar_max_dist

        min_idx = int(np.argmin(distances))
        d_min = float(distances[min_idx])
        theta_obstacle = (2.0 * np.pi * min_idx / self.n_bins) - np.pi
        return d_min, theta_obstacle

    # ── CBF Evaluation ───────────────────────────────────────

    def _compute_cbf_value(self, d_obstacle: float) -> float:
        """h(x) = d_obstacle − d_safe."""
        return d_obstacle - self.d_safe

    @staticmethod
    def _compute_cbf_dot(u_forward: float, theta_obstacle: float) -> float:
        """ḣ(x, u) ≈ −u_forward · cos(θ_obstacle).

        Forward motion decreases d_obstacle when the hazard is in front
        (|θ_obstacle| < π/2).
        """
        return -u_forward * np.cos(theta_obstacle)

    # ── Shield Decision ──────────────────────────────────────

    def shield(
        self, obs: np.ndarray, proposed_action: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Apply CBF-QP shield.

        Returns:
            safe_action:      np.ndarray of shape (2,)
            shield_activated: True if the QP was needed.
        """
        d_obstacle, theta_obstacle = self._get_hazard_distance_and_angle(obs)
        h = self._compute_cbf_value(d_obstacle)
        u_fw = float(proposed_action[0])
        cbf_dot = self._compute_cbf_dot(u_fw, theta_obstacle)

        # CBF condition: ḣ + γ·h ≥ 0
        if cbf_dot + self.gamma * h >= 0.0:
            return proposed_action.copy(), False

        # Shield activated — solve QP
        safe_action = self._solve_qp(proposed_action, theta_obstacle, h)
        return safe_action, True

    # ── QP Solver ────────────────────────────────────────────

    def _solve_qp(
        self,
        u_proposed: np.ndarray,
        theta_obstacle: float,
        h: float,
    ) -> np.ndarray:
        """Solve the minimum-norm safe-action QP via OSQP.

        min_{u}  ‖u − u_proposed‖²
        s.t.     −cos(θ)·u_fwd + γ·h ≥ 0   (CBF)
                 −max_speed ≤ u_fwd ≤ max_speed
                 −π ≤ u_turn ≤ π
        """
        # Objective: min 0.5 uᵀPu + qᵀu  (P = 2I)
        P = sp.eye(2, format="csc") * 2.0
        q = -2.0 * u_proposed

        # Constraint matrix: [CBF row; I₂ (bounds)]
        a_cbf = np.array([-np.cos(theta_obstacle), 0.0])
        A = sp.vstack(
            [sp.csc_matrix(a_cbf.reshape(1, 2)), sp.eye(2, format="csc")],
            format="csc",
        )
        lower = np.array([-self.gamma * h, -self.max_speed, -np.pi])
        upper = np.array([np.inf, self.max_speed, np.pi])

        if not self._solver_initialized:
            self._solver = osqp.OSQP()
            self._solver.setup(
                P,
                q,
                A,
                lower,
                upper,
                warm_starting=True,
                verbose=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=1000,
                polish=False,  # skip polishing for real-time speed
            )
            self._solver_initialized = True
        else:
            self._solver.update(q=q, l=lower, u=upper, Px=P.data, Ax=A.data)

        result = self._solver.solve()

        if result.info.status == "solved":
            return np.array(result.x, dtype=np.float64)

        # Fallback: zero forward velocity, keep turning
        safe = u_proposed.copy()
        safe[0] = min(safe[0], 0.0)
        return safe

    # ── Gym Interface ────────────────────────────────────────

    def step(self, proposed_action: np.ndarray):
        """Intercept step with shield."""
        self.total_steps += 1
        safe_action, activated = self.shield(self._last_obs, proposed_action)
        if activated:
            self.shield_activated_count += 1

        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self._last_obs = obs.copy()

        info["shield_activated"] = activated
        info["shield_rate"] = (
            self.shield_activated_count / self.total_steps
            if self.total_steps > 0
            else 0.0
        )
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs.copy()
        return obs, info
