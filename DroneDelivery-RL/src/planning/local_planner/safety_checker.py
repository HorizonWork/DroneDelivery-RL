import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

dataclass
class SafetyResult:

    is_safe: bool
    min_clearance: float
    collision_risk: float
    critical_points: List[Tuple[float, float, float]]

class SafetyChecker:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.min_clearance = config.get('min_clearance', 0.5)
        self.drone_radius = config.get('drone_radius', 0.2)
        self.safety_margin = config.get('safety_margin', 0.3)
        self.default_obstacle_radius = config.get('obstacle_radius', 0.3)

        self.prediction_horizon = config.get('prediction_horizon', 2.0)
        self.prediction_resolution = config.get('prediction_resolution', 0.1)
        self.collision_probability_threshold = config.get('collision_prob_threshold', 0.1)

        self.path_sample_resolution = config.get('path_sample_resolution', 0.2)

        self.logger.info("Safety Checker initialized")
        self.logger.info(f"Min clearance: {self.min_clearance}m, Drone radius: {self.drone_radius}m")
        self.logger.info(f"Prediction horizon: {self.prediction_horizon}s")

    def check_path_safety(self, start_pos: np.ndarray, end_pos: np.ndarray,
                         obstacles: List[Tuple[float, float, float]],
                         prediction_time: float = 0.0) - bool:

        if not obstacles:
            return True

        path_samples = self._sample_path(start_pos, end_pos)

        for sample_point in path_samples:
            if not self._check_point_safety(sample_point, obstacles, prediction_time):
                return False

        return True

    def check_full_path_safety(self, path: List[Tuple[float, float, float]],
                              obstacles: List[Tuple[float, float, float]],
                              execution_time_estimate: float = 0.0) - SafetyResult:

        if not path:
            return SafetyResult(is_safe=False, min_clearance=0.0, collision_risk=1.0, critical_points=[])

        min_clearance = float('inf')
        collision_risk = 0.0
        critical_points = []

        time_per_segment = execution_time_estimate / max(1, len(path) - 1)

        for i in range(len(path) - 1):
            segment_time = i  time_per_segment

            start_pos = np.array(path[i])
            end_pos = np.array(path[i + 1])
            samples = self._sample_path(start_pos, end_pos)

            for j, sample_point in enumerate(samples):
                sample_time = segment_time + j  time_per_segment / len(samples)

                safety_result = self._analyze_point_safety(sample_point, obstacles, sample_time)

                min_clearance = min(min_clearance, safety_result.min_clearance)
                collision_risk = max(collision_risk, safety_result.collision_risk)

                if safety_result.collision_risk  self.collision_probability_threshold:
                    critical_points.append(tuple(sample_point))

        is_safe = (min_clearance = self.min_clearance and
                  collision_risk  self.collision_probability_threshold)

        return SafetyResult(
            is_safe=is_safe,
            min_clearance=min_clearance,
            collision_risk=collision_risk,
            critical_points=critical_points
        )

    def _sample_path(self, start_pos: np.ndarray, end_pos: np.ndarray) - List[np.ndarray]:

        segment_length = np.linalg.norm(end_pos - start_pos)
        num_samples = max(2, int(np.ceil(segment_length / self.path_sample_resolution)))

        samples = []
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples  1 else 0
            sample_point = start_pos + t  (end_pos - start_pos)
            samples.append(sample_point)

        return samples

    def _check_point_safety(self, point: np.ndarray, obstacles: List[Tuple[float, float, float]],
                           prediction_time: float) - bool:

        safety_result = self._analyze_point_safety(point, obstacles, prediction_time)

        return (safety_result.min_clearance = self.min_clearance and
                safety_result.collision_risk  self.collision_probability_threshold)

    def _analyze_point_safety(self, point: np.ndarray, obstacles: List[Tuple[float, float, float]],
                            prediction_time: float) - SafetyResult:

        if not obstacles:
            return SafetyResult(is_safe=True, min_clearance=float('inf'),
                              collision_risk=0.0, critical_points=[])

        min_clearance = float('inf')
        max_collision_risk = 0.0

        for obstacle_pos in obstacles:
            obstacle_array = np.array(obstacle_pos)

            predicted_obstacle_pos = obstacle_array

            distance = np.linalg.norm(point - predicted_obstacle_pos)
            clearance = distance - self.default_obstacle_radius - self.drone_radius
            min_clearance = min(min_clearance, clearance)

            if clearance  self.safety_margin:
                risk = max(0.0, 1.0 - clearance / self.safety_margin)
                max_collision_risk = max(max_collision_risk, risk)

        return SafetyResult(
            is_safe=min_clearance = self.min_clearance and max_collision_risk  self.collision_probability_threshold,
            min_clearance=min_clearance,
            collision_risk=max_collision_risk,
            critical_points=[tuple(point)] if max_collision_risk  self.collision_probability_threshold else []
        )

    def get_safety_margins(self) - Dict[str, float]:

        return {
            'min_clearance': self.min_clearance,
            'drone_radius': self.drone_radius,
            'safety_margin': self.safety_margin,
            'total_safety_radius': self.min_clearance + self.drone_radius + self.safety_margin
        }
