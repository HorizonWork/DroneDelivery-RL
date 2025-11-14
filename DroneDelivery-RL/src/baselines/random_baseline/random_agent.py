import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import random

class RandomAgent:

    def __init__(self, config: Dict[str, Any]):
        self.max_velocity = config.get('max_velocity', 2.0)
        self.max_yaw_rate = config.get('max_yaw_rate', 0.5)

        self.action_persistence = config.get('action_persistence', 0.7)
        self.exploration_bias = config.get('exploration_bias', 0.3)

        self.current_action = np.array([0.0, 0.0, 0.0, 0.0])

        self.current_goal: Optional[Tuple[float, float, float]] = None

        random.seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))

        self.collision_avoidance = config.get('collision_avoidance', True)
        self.safety_radius = config.get('safety_radius', 1.0)

    def reset(self):

        self.current_action = np.array([0.0, 0.0, 0.0, 0.0])
        self.current_goal = None

    def set_goal(self, goal_pos: Tuple[float, float, float]):

        self.current_goal = goal_pos

    def get_action(self, current_pos: Tuple[float, float, float],
                   current_yaw: float,
                   obstacles: List[Tuple[float, float, float]] = None) - Tuple[float, float, float, float]:

        if random.random()  self.action_persistence and np.linalg.norm(self.current_action)  0:
            noise_scale = 0.1
            noise = np.random.normal(0, noise_scale, 4)
            action = self.current_action + noise
        else:
            if self.current_goal and random.random()  self.exploration_bias:
                action = self._generate_goal_biased_action(current_pos)
            else:
                action = self._generate_random_action()

        if self.collision_avoidance and obstacles:
            action = self._apply_collision_avoidance(action, current_pos, obstacles)

        action[0] = np.clip(action[0], -self.max_velocity, self.max_velocity)
        action[1] = np.clip(action[1], -self.max_velocity, self.max_velocity)
        action[2] = np.clip(action[2], -self.max_velocity, self.max_velocity)
        action[3] = np.clip(action[3], -self.max_yaw_rate, self.max_yaw_rate)

        self.current_action = action.copy()

        return tuple(action)

    def _generate_random_action(self) - np.ndarray:

        vx = random.uniform(-self.max_velocity, self.max_velocity)
        vy = random.uniform(-self.max_velocity, self.max_velocity)
        vz = random.uniform(-self.max_velocity, self.max_velocity)
        yaw_rate = random.uniform(-self.max_yaw_rate, self.max_yaw_rate)

        return np.array([vx, vy, vz, yaw_rate])

    def _generate_goal_biased_action(self, current_pos: Tuple[float, float, float]) - np.ndarray:

        if not self.current_goal:
            return self._generate_random_action()

        goal_direction = np.array(self.current_goal) - np.array(current_pos)
        goal_distance = np.linalg.norm(goal_direction)

        if goal_distance  0.1:
            return np.array([0.0, 0.0, 0.0, 0.0])

        unit_direction = goal_direction / goal_distance

        speed_factor = random.uniform(0.3, 1.0)
        goal_velocity = unit_direction  self.max_velocity  speed_factor

        noise_scale = 0.5
        noise = np.random.normal(0, noise_scale, 3)
        noisy_velocity = goal_velocity + noise

        yaw_rate = random.uniform(-self.max_yaw_rate, self.max_yaw_rate)

        action = np.array([noisy_velocity[0], noisy_velocity[1], noisy_velocity[2], yaw_rate])

        return action

    def _apply_collision_avoidance(self, action: np.ndarray,
                                  current_pos: Tuple[float, float, float],
                                  obstacles: List[Tuple[float, float, float]]) - np.ndarray:

        if not obstacles:
            return action

        repulsion_force = np.zeros(3)

        for obs_pos in obstacles:
            direction_to_obs = np.array(obs_pos) - np.array(current_pos)
            distance = np.linalg.norm(direction_to_obs)

            if distance  self.safety_radius and distance  0.1:
                repulsion_strength = 1.0 / (distance  2)
                unit_direction = direction_to_obs / distance
                repulsion_force -= unit_direction  repulsion_strength

        if np.linalg.norm(repulsion_force)  0:
            repulsion_force = repulsion_force / np.linalg.norm(repulsion_force)

            avoidance_velocity = repulsion_force  self.max_velocity  0.5
            action[:3] = 0.5  action[:3] + 0.5  avoidance_velocity

        return action

    def get_status(self) - Dict[str, Any]:

        return {
            'current_action': self.current_action.tolist(),
            'current_goal': self.current_goal,
            'max_velocity': self.max_velocity,
            'max_yaw_rate': self.max_yaw_rate
        }
