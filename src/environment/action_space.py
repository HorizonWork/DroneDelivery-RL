"""
Action Space Implementation
Implements 4D continuous action space for body-frame velocity commands.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ActionConfig:
    """Configuration for action space."""

    # Action dimensions (4D continuous as per report)
    action_dim: int = 4

    # Action bounds (body-frame velocities + yaw rate)
    max_translational_velocity: float = 5.0  # m/s (from report: 5 m/s max)
    max_yaw_rate: float = 1.0  # rad/s

    # --- DEBATE AI MODIFICATION ---
    # action_low and action_high are now properties to ensure they are always
    # consistent with max_translational_velocity and max_yaw_rate.
    @property
    def action_low(self) -> Tuple[float, ...]:
        return (
            -self.max_translational_velocity,
            -self.max_translational_velocity,
            -self.max_translational_velocity,
            -self.max_yaw_rate,
        )

    @property
    def action_high(self) -> Tuple[float, ...]:
        return (
            self.max_translational_velocity,
            self.max_translational_velocity,
            self.max_translational_velocity,
            self.max_yaw_rate,
        )

    # --- END OF MODIFICATION ---

    # Safety constraints
    enable_safety_limits: bool = True
    emergency_stop_threshold: float = 10.0  # m/s (absolute velocity limit)


class ActionSpace:
    """
    4D continuous action space implementation.
    Actions: [vx, vy, vz, ω] - body-frame velocity commands + yaw rate.

    CRITICAL: PPO outputs unbounded Gaussian actions - must clip before use!
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}

        self.config = ActionConfig(**config.get("action", {}))
        self.logger = logging.getLogger(__name__)

        # Validate configuration
        assert self.config.action_dim == 4, (
            f"Action dimension must be 4, got {self.config.action_dim}"
        )
        assert len(self.config.action_low) == 4, "action_low must have 4 elements"
        assert len(self.config.action_high) == 4, "action_high must have 4 elements"

        # Action bounds as numpy arrays
        self.action_low = np.array(self.config.action_low, dtype=np.float32)
        self.action_high = np.array(self.config.action_high, dtype=np.float32)

        # Action component names
        self.action_names = ["vx", "vy", "vz", "yaw_rate"]

        # Action mapping for discrete to continuous conversion
        self.action_map = {
            "forward": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "backward": np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "left": np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            "right": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "up": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "down": np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            "rotate_left": np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            "rotate_right": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "hover": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

        # Action statistics for monitoring
        self.action_history = []
        self.max_history_length = 1000

        # Clipping statistics
        self.clip_count = 0
        self.total_actions = 0

        self.logger.info(
            f"Action space initialized: {self.config.action_dim}D continuous"
        )
        self.logger.info(
            f"Bounds: vx=[{self.action_low[0]:.1f}, {self.action_high[0]:.1f}], "
            f"vy=[{self.action_low[1]:.1f}, {self.action_high[1]:.1f}], "
            f"vz=[{self.action_low[2]:.1f}, {self.action_high[2]:.1f}], "
            f"yaw=[{self.action_low[3]:.1f}, {self.action_high[3]:.1f}]"
        )

    def sample(self) -> np.ndarray:
        """Sample random action from action space."""
        action = np.random.uniform(self.action_low, self.action_high).astype(np.float32)
        return action

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        Clip action to valid bounds.

        Args:
            action: Raw action vector

        Returns:
            Clipped action vector
        """
        clipped_action = np.clip(action, self.action_low, self.action_high).astype(
            np.float32
        )

        # Track clipping statistics
        if not np.allclose(action, clipped_action, atol=1e-6):
            self.clip_count += 1
            if self.clip_count % 100 == 0:
                clip_rate = (self.clip_count / max(1, self.total_actions)) * 100
                self.logger.debug(
                    f"Action clipping rate: {clip_rate:.1f}% ({self.clip_count}/{self.total_actions})"
                )

        self.total_actions += 1

        return clipped_action

    def apply_safety_constraints(
        self, action: np.ndarray, current_state: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Apply safety constraints to action.

        CRITICAL: This is called from airsim_env.step() and must clip PPO actions!

        Args:
            action: Raw action vector [vx, vy, vz, ω] (potentially unbounded from PPO)
            current_state: Current drone state (for context-aware constraints)

        Returns:
            Safety-constrained action vector (guaranteed within bounds)
        """
        # ALWAYS clip first, even if safety_limits disabled
        safe_action = self.clip_action(action)

        if not self.config.enable_safety_limits:
            return safe_action

        # Emergency velocity limit
        translational_velocity = np.linalg.norm(safe_action[:3])
        if translational_velocity > self.config.emergency_stop_threshold:
            # Scale down translational velocities
            scale_factor = self.config.emergency_stop_threshold / translational_velocity
            safe_action[:3] *= scale_factor
            self.logger.warning(
                f"Emergency velocity limit applied: {translational_velocity:.2f} -> "
                f"{np.linalg.norm(safe_action[:3]):.2f} m/s"
            )

        # Context-aware constraints (if state provided)
        if current_state is not None:
            safe_action = self._apply_contextual_constraints(safe_action, current_state)

        return safe_action

    def _apply_contextual_constraints(
        self, action: np.ndarray, current_state: Dict[str, Any]
    ) -> np.ndarray:
        """Apply context-aware safety constraints based on current state."""
        constrained_action = action.copy()

        # 1. Altitude constraints (prevent ground collision)
        # --- DEBATE AI MODIFICATION ---
        # Corrected the logic to prevent ground collision in NED coordinates.
        # If too low and commanded to descend (vz > 0 in NED), force an upward velocity.
        current_altitude = abs(current_state.get("position", (0, 0, 0))[2])
        if (
            current_altitude < 1.0 and constrained_action[2] > 0
        ):  # Too low and descending
            self.logger.warning(
                f"Altitude safety triggered! Current alt: {current_altitude:.2f}m. Forcing upward velocity."
            )
            constrained_action[2] = -0.5  # Force a moderate upward velocity
        # --- END OF MODIFICATION ---

        # 2. Battery-aware constraints
        battery_level = current_state.get("battery_level", 1.0)
        if battery_level < 0.2:  # Low battery
            energy_scale = max(0.5, battery_level * 2)
            constrained_action[:3] *= energy_scale

        # 3. Obstacle proximity constraints
        if "nearby_obstacles" in current_state:
            obstacles = current_state["nearby_obstacles"]
            if obstacles:
                min_distance = min(obstacles)
                if min_distance < 2.0:  # Within 2m of an obstacle
                    safety_scale = max(0.3, min_distance / 2.0)
                    constrained_action[:3] *= safety_scale

        return constrained_action

    def convert_to_body_frame(
        self, action: np.ndarray, current_orientation: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Convert action to body frame (actions are already in body frame)."""
        return action.copy()

    def validate_action(self, action: np.ndarray) -> bool:
        """
        Validate action vector.

        Args:
            action: Action vector to validate

        Returns:
            True if valid, False otherwise
        """
        # Check dimension
        if len(action) != self.config.action_dim:
            self.logger.error(
                f"Invalid action dimension: {len(action)} != {self.config.action_dim}"
            )
            return False

        # Check for NaN or infinite values
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            self.logger.error("Action contains NaN or infinite values")
            return False

        # Warn but don't reject out-of-bounds (will be clipped later)
        tolerance = 0.1
        if np.any(action < self.action_low - tolerance) or np.any(
            action > self.action_high + tolerance
        ):
            self.logger.warning(f"Action outside bounds: {action}")
            # Don't return False - let clipping handle it

        return True

    def get_action_info(self) -> Dict[str, Any]:
        """Get information about action space structure."""
        return {
            "dimension": self.config.action_dim,
            "type": "continuous",
            "components": {
                "vx": {
                    "index": 0,
                    "bounds": [self.action_low[0], self.action_high[0]],
                    "description": "Forward velocity (body frame) [m/s]",
                },
                "vy": {
                    "index": 1,
                    "bounds": [self.action_low[1], self.action_high[1]],
                    "description": "Right velocity (body frame) [m/s]",
                },
                "vz": {
                    "index": 2,
                    "bounds": [self.action_low[2], self.action_high[2]],
                    "description": "Down velocity (body frame) [m/s]",
                },
                "yaw_rate": {
                    "index": 3,
                    "bounds": [self.action_low[3], self.action_high[3]],
                    "description": "Yaw rate [rad/s]",
                },
            },
            "safety_features": {
                "emergency_stop_threshold": self.config.emergency_stop_threshold,
                "contextual_constraints": self.config.enable_safety_limits,
                "altitude_protection": True,
                "battery_awareness": True,
                "obstacle_avoidance": True,
            },
        }

    def record_action(self, action: np.ndarray):
        """Record action for statistics and monitoring."""
        self.action_history.append(action.copy())

        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)

    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent actions."""
        if not self.action_history:
            return {"message": "No actions recorded yet"}

        actions = np.array(self.action_history)

        return {
            "num_actions": len(self.action_history),
            "mean_action": np.mean(actions, axis=0).tolist(),
            "std_action": np.std(actions, axis=0).tolist(),
            "min_action": np.min(actions, axis=0).tolist(),
            "max_action": np.max(actions, axis=0).tolist(),
            "component_names": self.action_names,
            "velocity_statistics": {
                "mean_translational_speed": float(
                    np.mean(np.linalg.norm(actions[:, :3], axis=1))
                ),
                "max_translational_speed": float(
                    np.max(np.linalg.norm(actions[:, :3], axis=1))
                ),
                "mean_yaw_rate": float(np.mean(np.abs(actions[:, 3]))),
            },
            "clipping_rate": (self.clip_count / max(1, self.total_actions)) * 100,
        }

    def reset_statistics(self):
        """Reset action statistics."""
        self.action_history.clear()
        self.clip_count = 0
        self.total_actions = 0
        self.logger.info("Action statistics reset")

    def is_zero_action(self, action: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if action is effectively zero (hover)."""
        return np.allclose(action, 0.0, atol=tolerance)

    def create_hover_action(self) -> np.ndarray:
        """Create hover action (all zeros)."""
        return np.zeros(self.config.action_dim, dtype=np.float32)

    def create_emergency_stop_action(self) -> np.ndarray:
        """Create emergency stop action."""
        return self.create_hover_action()
