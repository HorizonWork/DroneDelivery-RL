import numpy as np
import logging
import time
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

dataclass
class RewardConfig:

    goal_bonus: float = 500.0
    distance_penalty: float = 5.0
    time_penalty: float = 0.1
    thrust_penalty: float = 0.01
    jerk_penalty: float = 10.0
    collision_penalty: float = 1000.0

    control_dt: float = 0.05
    goal_tolerance: float = 0.5
    num_rotors: int = 4

    drone_mass: float = 1.5
    gravity: float = 9.81
    hover_thrust_per_motor: float = 3.675

dataclass
class RewardComponents:

    goal_bonus: float = 0.0
    distance_penalty: float = 0.0
    time_penalty: float = 0.0
    thrust_penalty: float = 0.0
    jerk_penalty: float = 0.0
    collision_penalty: float = 0.0
    total_reward: float = 0.0

class RewardFunction:

    def __init__(self, config: Dict[str, Any]):
        self.config = RewardConfig(config.get("reward", {}))
        self.logger = logging.getLogger(__name__)

        self.previous_velocity_command: Optional[np.ndarray] = None
        self.previous_action_time: Optional[float] = None

        self.episode_rewards: List[RewardComponents] = []
        self.cumulative_energy: float = 0.0
        self.episode_start_time: float = 0.0
        self.last_components: Optional[RewardComponents] = None

        self.max_reasonable_distance = 100.0
        self.max_reasonable_thrust = 20.0

        self.logger.info("Reward function initialized with Equation (2) coefficients")
        self.logger.info(
            f"Goal bonus: {self.config.goal_bonus}, "
            f"Distance penalty: {self.config.distance_penalty}, "
            f"Time penalty: {self.config.time_penalty}, "
            f"Thrust penalty: {self.config.thrust_penalty}, "
            f"Jerk penalty: {self.config.jerk_penalty}, "
            f"Collision penalty: {self.config.collision_penalty}"
        )

    def compute_reward(
        self,
        state: Dict[str, Any],
        action: np.ndarray,
        next_state: Dict[str, Any],
        info: Dict[str, Any],
    ) - float:

        current_time = time.time()
        components = RewardComponents()

        components.goal_bonus = self._compute_goal_bonus(next_state, info)

        components.distance_penalty = self._compute_distance_penalty(next_state)

        components.time_penalty = self._compute_time_penalty()

        components.thrust_penalty = self._compute_thrust_penalty(action, state)

        components.jerk_penalty = self._compute_jerk_penalty(action, current_time)

        components.collision_penalty = self._compute_collision_penalty(info)

        components.total_reward = (
            components.goal_bonus
            + components.distance_penalty
            + components.time_penalty
            + components.thrust_penalty
            + components.jerk_penalty
            + components.collision_penalty
        )

        self.episode_rewards.append(components)
        self.last_components = components

        self.cumulative_energy += abs(
            components.thrust_penalty / self.config.thrust_penalty
        )

        return float(components.total_reward)

    def _compute_goal_bonus(
        self, next_state: Dict[str, Any], info: Dict[str, Any]
    ) - float:

        goal_reached = info.get("goal_reached", False)

        if goal_reached:
            self.logger.info("Goal reached! Applying goal bonus.")
            return self.config.goal_bonus

        if "goal_position" in next_state and "position" in next_state:
            current_pos = np.array(next_state["position"])
            goal_pos = np.array(next_state["goal_position"])
            distance_to_goal = np.linalg.norm(goal_pos - current_pos)

            if distance_to_goal = self.config.goal_tolerance:
                self.logger.info(
                    f"Goal reached by distance: {distance_to_goal:.3f}m = {self.config.goal_tolerance}m"
                )
                return self.config.goal_bonus

        return 0.0

    def _compute_distance_penalty(self, next_state: Dict[str, Any]) - float:

        if "goal_position" not in next_state or "position" not in next_state:
            return 0.0

        current_pos = np.array(next_state["position"])
        goal_pos = np.array(next_state["goal_position"])
        distance_to_goal = np.linalg.norm(goal_pos - current_pos)

        distance_to_goal = min(distance_to_goal, self.max_reasonable_distance)

        penalty = -self.config.distance_penalty  distance_to_goal
        return float(penalty)

    def _compute_time_penalty(self) - float:

        penalty = -self.config.time_penalty  self.config.control_dt
        return float(penalty)

    def _compute_thrust_penalty(
        self, action: np.ndarray, state: Dict[str, Any]
    ) - float:

        motor_thrusts = self._velocity_to_thrust(action, state)

        thrust_squared_sum = np.sum(motor_thrusts2)

        penalty = -self.config.thrust_penalty  thrust_squared_sum
        return float(penalty)

    def _velocity_to_thrust(
        self, velocity_command: np.ndarray, state: Dict[str, Any]
    ) - np.ndarray:

        vx, vy, vz, yaw_rate = velocity_command

        hover_thrust = self.config.hover_thrust_per_motor

        accel_x = vx
        accel_y = vy
        accel_z = vz

        thrust_x = self.config.drone_mass  accel_x / self.config.num_rotors
        thrust_y = self.config.drone_mass  accel_y / self.config.num_rotors
        thrust_z = (
            self.config.drone_mass
             (accel_z + self.config.gravity)
            / self.config.num_rotors
        )

        motor_thrusts = np.array(
            [
                hover_thrust + thrust_z + thrust_x - thrust_y,
                hover_thrust + thrust_z + thrust_x + thrust_y,
                hover_thrust + thrust_z - thrust_x - thrust_y,
                hover_thrust + thrust_z - thrust_x + thrust_y,
            ]
        )

        motor_thrusts = np.clip(motor_thrusts, 0.0, self.max_reasonable_thrust)

        return motor_thrusts

    def _compute_jerk_penalty(self, action: np.ndarray, current_time: float) - float:

        if self.previous_velocity_command is None or self.previous_action_time is None:
            self.previous_velocity_command = action.copy()
            self.previous_action_time = current_time
            return 0.0

        dt = current_time - self.previous_action_time
        if dt = 0:
            dt = self.config.control_dt

        velocity_derivative = (action - self.previous_velocity_command) / dt

        jerk_squared = np.sum(velocity_derivative2)

        penalty = -self.config.jerk_penalty  jerk_squared

        self.previous_velocity_command = action.copy()
        self.previous_action_time = current_time

        return float(penalty)

    def _compute_collision_penalty(self, info: Dict[str, Any]) - float:

        collision_occurred = info.get("collision", False)

        if collision_occurred:
            self.logger.warning("Collision detected! Applying collision penalty.")
            return -self.config.collision_penalty

        return 0.0

    def reset_episode(self):

        self.previous_velocity_command = None
        self.previous_action_time = None
        self.episode_rewards.clear()
        self.cumulative_energy = 0.0
        self.episode_start_time = time.time()
        self.last_components = None

        self.logger.debug("Reward function reset for new episode")

    def get_last_components_dict(self) - Optional[Dict[str, float]]:

        component = self.last_components
        if component is None:
            return None
        return {
            "goal_bonus": float(component.goal_bonus),
            "distance_penalty": float(component.distance_penalty),
            "time_penalty": float(component.time_penalty),
            "thrust_penalty": float(component.thrust_penalty),
            "jerk_penalty": float(component.jerk_penalty),
            "collision_penalty": float(component.collision_penalty),
            "total_reward": float(component.total_reward),
        }

    def get_episode_statistics(self) - Dict[str, Any]:

        if not self.episode_rewards:
            return {"message": "No rewards recorded yet"}

        components = {
            "goal_bonus": [r.goal_bonus for r in self.episode_rewards],
            "distance_penalty": [r.distance_penalty for r in self.episode_rewards],
            "time_penalty": [r.time_penalty for r in self.episode_rewards],
            "thrust_penalty": [r.thrust_penalty for r in self.episode_rewards],
            "jerk_penalty": [r.jerk_penalty for r in self.episode_rewards],
            "collision_penalty": [r.collision_penalty for r in self.episode_rewards],
            "total_reward": [r.total_reward for r in self.episode_rewards],
        }

        stats = {}
        for name, values in components.items():
            if values:
                stats[name] = {
                    "sum": float(np.sum(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        stats["episode_info"] = {
            "num_steps": len(self.episode_rewards),
            "total_reward": sum(r.total_reward for r in self.episode_rewards),
            "cumulative_energy": self.cumulative_energy,
            "episode_duration": time.time() - self.episode_start_time,
            "goal_reached": any(r.goal_bonus  0 for r in self.episode_rewards),
            "collision_occurred": any(
                r.collision_penalty  0 for r in self.episode_rewards
            ),
        }

        return stats

    def get_episode_totals(self) - Optional[Dict[str, float]]:

        if not self.episode_rewards:
            return None

        totals = {
            "goal_bonus": sum(r.goal_bonus for r in self.episode_rewards),
            "distance_penalty": sum(r.distance_penalty for r in self.episode_rewards),
            "time_penalty": sum(r.time_penalty for r in self.episode_rewards),
            "thrust_penalty": sum(r.thrust_penalty for r in self.episode_rewards),
            "jerk_penalty": sum(r.jerk_penalty for r in self.episode_rewards),
            "collision_penalty": sum(
                r.collision_penalty for r in self.episode_rewards
            ),
        }
        totals["total_reward"] = sum(r.total_reward for r in self.episode_rewards)
        totals["steps"] = len(self.episode_rewards)
        return {k: float(v) for k, v in totals.items()}

    def validate_reward_config(self) - bool:

        if self.config.goal_bonus = 0:
            self.logger.error("Goal bonus must be positive")
            return False

        if self.config.distance_penalty = 0:
            self.logger.error("Distance penalty coefficient must be positive")
            return False

        if self.config.collision_penalty = 0:
            self.logger.error("Collision penalty coefficient must be positive")
            return False

        if self.config.control_dt = 0 or self.config.control_dt  1.0:
            self.logger.error(f"Invalid control timestep: {self.config.control_dt}")
            return False

        if self.config.drone_mass = 0:
            self.logger.error("Drone mass must be positive")
            return False

        if self.config.num_rotors != 4:
            self.logger.warning("Reward function designed for quadrotor (4 rotors)")

        return True

    def get_reward_info(self) - Dict[str, Any]:

        return {
            "equation": "R(s_t, a_t) = 5001{goal} - 5d_t - 0.1Δt - 0.01Σu_i² - 10j_t - 1000c_t",
            "coefficients": {
                "goal_bonus": self.config.goal_bonus,
                "distance_penalty": self.config.distance_penalty,
                "time_penalty": self.config.time_penalty,
                "thrust_penalty": self.config.thrust_penalty,
                "jerk_penalty": self.config.jerk_penalty,
                "collision_penalty": self.config.collision_penalty,
            },
            "parameters": {
                "control_dt": self.config.control_dt,
                "goal_tolerance": self.config.goal_tolerance,
                "drone_mass": self.config.drone_mass,
                "num_rotors": self.config.num_rotors,
            },
            "components": {
                "goal_bonus": "Large positive reward for reaching target",
                "distance_penalty": "Penalizes current distance to goal",
                "time_penalty": "Penalizes elapsed time per step",
                "thrust_penalty": "Penalizes energy consumption (thrust squared)",
                "jerk_penalty": "Penalizes sudden velocity changes",
                "collision_penalty": "Large negative reward for collisions",
            },
        }
