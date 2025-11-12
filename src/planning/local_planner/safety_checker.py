"""
Safety Checker
Validates path safety with dynamic obstacle prediction and clearance analysis.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SafetyResult:
    """Safety check result."""

    is_safe: bool
    min_clearance: float
    collision_risk: float
    critical_points: List[Tuple[float, float, float]]


class SafetyChecker:
    """
    Safety validation for S-RRT paths.
    Checks dynamic obstacle collision risk and maintains safety clearances.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Safety parameters
        self.min_clearance = config.get("min_clearance", 0.5)  # meters
        self.drone_radius = config.get("drone_radius", 0.2)  # meters
        self.safety_margin = config.get("safety_margin", 0.3)  # additional margin
        self.default_obstacle_radius = config.get("obstacle_radius", 0.3)

        # Collision prediction
        self.prediction_horizon = config.get("prediction_horizon", 2.0)  # seconds
        self.prediction_resolution = config.get("prediction_resolution", 0.1)  # seconds
        self.collision_probability_threshold = config.get(
            "collision_prob_threshold", 0.1
        )

        # Path sampling for safety checking
        self.path_sample_resolution = config.get(
            "path_sample_resolution", 0.2
        )  # meters

        self.logger.info("Safety Checker initialized")
        self.logger.info(
            f"Min clearance: {self.min_clearance}m, Drone radius: {self.drone_radius}m"
        )
        self.logger.info(f"Prediction horizon: {self.prediction_horizon}s")

    def check_path_safety(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
        prediction_time: float = 0.0,
    ) -> bool:
        """
        Check if path segment is safe.

        Args:
            start_pos: Start position
            end_pos: End position
            obstacles: Current obstacle positions
            prediction_time: Time offset for obstacle prediction

        Returns:
            True if path is safe
        """
        if not obstacles:
            return True

        # Sample points along path
        path_samples = self._sample_path(start_pos, end_pos)

        # Check each sample point
        for sample_point in path_samples:
            if not self._check_point_safety(sample_point, obstacles, prediction_time):
                return False

        return True

    def check_full_path_safety(
        self,
        path: List[Tuple[float, float, float]],
        obstacles: List[Tuple[float, float, float]],
        execution_time_estimate: float = 0.0,
    ) -> SafetyResult:
        """
        Check safety of complete path with dynamic obstacle prediction.

        Args:
            path: Complete path to check
            obstacles: Current obstacle positions
            execution_time_estimate: Estimated execution time for path

        Returns:
            Comprehensive safety result
        """
        if not path:
            return SafetyResult(
                is_safe=False, min_clearance=0.0, collision_risk=1.0, critical_points=[]
            )

        min_clearance = float("inf")
        collision_risk = 0.0
        critical_points = []

        # Time per path segment (estimated)
        time_per_segment = execution_time_estimate / max(1, len(path) - 1)

        # Check each path segment
        for i in range(len(path) - 1):
            segment_time = i * time_per_segment

            # Sample points along segment
            start_pos = np.array(path[i])
            end_pos = np.array(path[i + 1])
            samples = self._sample_path(start_pos, end_pos)

            # Check safety at each sample
            for j, sample_point in enumerate(samples):
                sample_time = segment_time + j * time_per_segment / len(samples)

                safety_result = self._analyze_point_safety(
                    sample_point, obstacles, sample_time
                )

                min_clearance = min(min_clearance, safety_result.min_clearance)
                collision_risk = max(collision_risk, safety_result.collision_risk)

                if safety_result.collision_risk > self.collision_probability_threshold:
                    critical_points.append(tuple(sample_point))

        # Overall safety assessment
        is_safe = (
            min_clearance >= self.min_clearance
            and collision_risk < self.collision_probability_threshold
        )

        return SafetyResult(
            is_safe=is_safe,
            min_clearance=min_clearance,
            collision_risk=collision_risk,
            critical_points=critical_points,
        )

    def _sample_path(
        self, start_pos: np.ndarray, end_pos: np.ndarray
    ) -> List[np.ndarray]:
        """
        Sample points along path segment.

        Args:
            start_pos: Segment start
            end_pos: Segment end

        Returns:
            List of sample points
        """
        segment_length = np.linalg.norm(end_pos - start_pos)
        num_samples = max(2, int(np.ceil(segment_length / self.path_sample_resolution)))

        samples = []
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            sample_point = start_pos + t * (end_pos - start_pos)
            samples.append(sample_point)

        return samples

    def _check_point_safety(
        self,
        point: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
        prediction_time: float,
    ) -> bool:
        """
        Check safety of single point.

        Args:
            point: Point to check
            obstacles: Obstacle positions
            prediction_time: Time offset for prediction

        Returns:
            True if point is safe
        """
        safety_result = self._analyze_point_safety(point, obstacles, prediction_time)

        return (
            safety_result.min_clearance >= self.min_clearance
            and safety_result.collision_risk < self.collision_probability_threshold
        )

    def _analyze_point_safety(
        self,
        point: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
        prediction_time: float,
    ) -> SafetyResult:
        """
        Analyze safety metrics for a point.

        Args:
            point: Point to analyze
            obstacles: Obstacle positions
            prediction_time: Time for obstacle prediction

        Returns:
            Detailed safety analysis
        """
        if not obstacles:
            return SafetyResult(
                is_safe=True,
                min_clearance=float("inf"),
                collision_risk=0.0,
                critical_points=[],
            )

        min_clearance = float("inf")
        max_collision_risk = 0.0

        for obstacle_pos in obstacles:
            obstacle_array = np.array(obstacle_pos)

            # Predict obstacle position at prediction_time
            # Simplified: assume constant velocity (would use obstacle tracker in full implementation)
            predicted_obstacle_pos = obstacle_array  # Placeholder

            # Calculate clearance
            distance = np.linalg.norm(point - predicted_obstacle_pos)
            clearance = distance - self.default_obstacle_radius - self.drone_radius
            min_clearance = min(min_clearance, clearance)

            # Calculate collision risk
            if clearance < self.safety_margin:
                # Risk increases as clearance decreases
                risk = max(0.0, 1.0 - clearance / self.safety_margin)
                max_collision_risk = max(max_collision_risk, risk)

        return SafetyResult(
            is_safe=min_clearance >= self.min_clearance
            and max_collision_risk < self.collision_probability_threshold,
            min_clearance=min_clearance,
            collision_risk=max_collision_risk,
            critical_points=(
                [tuple(point)]
                if max_collision_risk > self.collision_probability_threshold
                else []
            ),
        )

    def get_safety_margins(self) -> Dict[str, float]:
        """Get current safety margin parameters."""
        return {
            "min_clearance": self.min_clearance,
            "drone_radius": self.drone_radius,
            "safety_margin": self.safety_margin,
            "total_safety_radius": self.min_clearance
            + self.drone_radius
            + self.safety_margin,
        }
