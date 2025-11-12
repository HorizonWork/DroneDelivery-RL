"""
Random Baseline Agent
Performs random exploration as lower bound baseline.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import random


class RandomAgent:
    """
    Random exploration agent for baseline comparison.
    Provides lower bound performance for comparison with intelligent methods.
    """

    def __init__(self, config: Dict[str, Any]):
        # Action space bounds
        self.max_velocity = config.get("max_velocity", 2.0)  # m/s (conservative)
        self.max_yaw_rate = config.get("max_yaw_rate", 0.5)  # rad/s (conservative)

        # Random walk parameters
        self.action_persistence = config.get(
            "action_persistence", 0.7
        )  # Keep same action
        self.exploration_bias = config.get("exploration_bias", 0.3)  # Towards goal

        # Current action (for persistence)
        self.current_action = np.array([0.0, 0.0, 0.0, 0.0])

        # Goal tracking
        self.current_goal: Optional[Tuple[float, float, float]] = None

        # Random seed
        random.seed(config.get("seed", 42))
        np.random.seed(config.get("seed", 42))

        # Safety parameters
        self.collision_avoidance = config.get("collision_avoidance", True)
        self.safety_radius = config.get("safety_radius", 1.0)  # meters

    def reset(self):
        """Reset agent state for new episode."""
        self.current_action = np.array([0.0, 0.0, 0.0, 0.0])
        self.current_goal = None

    def set_goal(self, goal_pos: Tuple[float, float, float]):
        """Set current goal position."""
        self.current_goal = goal_pos

    def get_action(
        self,
        current_pos: Tuple[float, float, float],
        current_yaw: float,
        obstacles: List[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Generate random action with optional goal bias.

        Args:
            current_pos: Current (x, y, z) position
            current_yaw: Current yaw angle
            obstacles: List of obstacle positions (for basic avoidance)

        Returns:
            (vx, vy, vz, yaw_rate): Random velocity commands
        """
        # Decide whether to keep current action or generate new one
        if (
            random.random() < self.action_persistence
            and np.linalg.norm(self.current_action) > 0
        ):
            # Keep current action with small noise
            noise_scale = 0.1
            noise = np.random.normal(0, noise_scale, 4)
            action = self.current_action + noise
        else:
            # Generate new random action
            if self.current_goal and random.random() < self.exploration_bias:
                # Biased towards goal
                action = self._generate_goal_biased_action(current_pos)
            else:
                # Purely random
                action = self._generate_random_action()

        # Apply safety constraints
        if self.collision_avoidance and obstacles:
            action = self._apply_collision_avoidance(action, current_pos, obstacles)

        # Clip to action bounds
        action[0] = np.clip(action[0], -self.max_velocity, self.max_velocity)  # vx
        action[1] = np.clip(action[1], -self.max_velocity, self.max_velocity)  # vy
        action[2] = np.clip(action[2], -self.max_velocity, self.max_velocity)  # vz
        action[3] = np.clip(
            action[3], -self.max_yaw_rate, self.max_yaw_rate
        )  # yaw_rate

        self.current_action = action.copy()

        return tuple(action)

    def _generate_random_action(self) -> np.ndarray:
        """Generate purely random action."""
        vx = random.uniform(-self.max_velocity, self.max_velocity)
        vy = random.uniform(-self.max_velocity, self.max_velocity)
        vz = random.uniform(-self.max_velocity, self.max_velocity)
        yaw_rate = random.uniform(-self.max_yaw_rate, self.max_yaw_rate)

        return np.array([vx, vy, vz, yaw_rate])

    def _generate_goal_biased_action(
        self, current_pos: Tuple[float, float, float]
    ) -> np.ndarray:
        """Generate action biased towards goal with random noise."""
        if not self.current_goal:
            return self._generate_random_action()

        # Direction to goal
        goal_direction = np.array(self.current_goal) - np.array(current_pos)
        goal_distance = np.linalg.norm(goal_direction)

        if goal_distance < 0.1:  # Very close to goal
            return np.array([0.0, 0.0, 0.0, 0.0])

        # Normalize direction
        unit_direction = goal_direction / goal_distance

        # Scale by random factor (0.3 to 1.0 of max velocity)
        speed_factor = random.uniform(0.3, 1.0)
        goal_velocity = unit_direction * self.max_velocity * speed_factor

        # Add random noise
        noise_scale = 0.5
        noise = np.random.normal(0, noise_scale, 3)
        noisy_velocity = goal_velocity + noise

        # Random yaw rate
        yaw_rate = random.uniform(-self.max_yaw_rate, self.max_yaw_rate)

        action = np.array(
            [noisy_velocity[0], noisy_velocity[1], noisy_velocity[2], yaw_rate]
        )

        return action

    def _apply_collision_avoidance(
        self,
        action: np.ndarray,
        current_pos: Tuple[float, float, float],
        obstacles: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        """Apply basic collision avoidance to action."""
        if not obstacles:
            return action

        # Check if any obstacles are within safety radius
        repulsion_force = np.zeros(3)

        for obs_pos in obstacles:
            direction_to_obs = np.array(obs_pos) - np.array(current_pos)
            distance = np.linalg.norm(direction_to_obs)

            if distance < self.safety_radius and distance > 0.1:
                # Add repulsion force (inverse quadratic)
                repulsion_strength = 1.0 / (distance**2)
                unit_direction = direction_to_obs / distance
                repulsion_force -= unit_direction * repulsion_strength

        # Apply repulsion to action
        if np.linalg.norm(repulsion_force) > 0:
            # Normalize repulsion force
            repulsion_force = repulsion_force / np.linalg.norm(repulsion_force)

            # Mix with original action (50% original, 50% avoidance)
            avoidance_velocity = repulsion_force * self.max_velocity * 0.5
            action[:3] = 0.5 * action[:3] + 0.5 * avoidance_velocity

        return action

    def get_status(self) -> Dict[str, Any]:
        """Get agent status for debugging."""
        return {
            "current_action": self.current_action.tolist(),
            "current_goal": self.current_goal,
            "max_velocity": self.max_velocity,
            "max_yaw_rate": self.max_yaw_rate,
        }
