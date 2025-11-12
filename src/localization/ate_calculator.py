"""
ATE (Absolute Trajectory Error) Calculator
Calculates centimeter-scale trajectory accuracy for evaluation.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time


class ATECalculator:
    """
    Calculates Absolute Trajectory Error for trajectory evaluation.
    Implements standard ATE calculation used in SLAM benchmarking.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cfg = self.config
        self.logger = logging.getLogger(__name__)

        # ATE calculation parameters
        self.alignment_method = cfg.get(
            "alignment_method", "sim3"
        )  # 'sim3', 'se3', 'sim3_scale'
        self.max_time_difference = cfg.get(
            "max_time_difference", 0.02
        )  # 20ms tolerance
        self.min_trajectory_length = cfg.get("min_trajectory_length", 10)

        # Ground truth storage
        self.ground_truth: List[Tuple[Tuple[float, float, float], float]] = []

        # ATE history for running calculations
        self.ate_history: List[float] = []
        self.max_history_length = cfg.get("max_history_length", 1000)

        # Performance targets (from reports)
        self.target_ate_euroc = 0.036  # 3.6cm
        self.target_ate_tumvi = 0.009  # 9mm
        self.target_rpe = 0.050  # 5cm/s

        self.logger.info("ATE Calculator initialized")
        self.logger.info(f"Alignment method: {self.alignment_method}")
        self.logger.info(
            f"Target ATE: EuRoC={self.target_ate_euroc*100:.1f}cm, TUM-VI={self.target_ate_tumvi*10:.0f}mm"
        )

    def set_ground_truth(
        self, ground_truth: List[Tuple[Tuple[float, float, float], float]]
    ):
        """
        Set ground truth trajectory.

        Args:
            ground_truth: List of (position, timestamp) tuples
        """
        self.ground_truth = ground_truth.copy()
        self.logger.info(f"Ground truth set: {len(ground_truth)} poses")

    def calculate_ate(
        self,
        estimated_trajectory: List[Tuple[Tuple[float, float, float], float]],
        ground_truth: Optional[List[Tuple[Tuple[float, float, float], float]]] = None,
    ) -> float:
        """
        Calculate Absolute Trajectory Error.

        Args:
            estimated_trajectory: List of (position, timestamp) tuples from SLAM
            ground_truth: Optional ground truth (uses stored if None)

        Returns:
            ATE in meters
        """
        if ground_truth is None:
            ground_truth = self.ground_truth

        if not ground_truth or not estimated_trajectory:
            return 0.0

        if len(estimated_trajectory) < self.min_trajectory_length:
            return 0.0

        try:
            # Associate trajectories by timestamp
            associated_pairs = self._associate_trajectories(
                estimated_trajectory, ground_truth
            )

            if len(associated_pairs) < self.min_trajectory_length:
                self.logger.warning(
                    f"Insufficient associated poses: {len(associated_pairs)}"
                )
                return 0.0

            # Extract positions
            estimated_positions = np.array([pair[0] for pair in associated_pairs])
            ground_truth_positions = np.array([pair[1] for pair in associated_pairs])

            # Align trajectories
            aligned_estimated = self._align_trajectories(
                estimated_positions, ground_truth_positions
            )

            # Calculate ATE
            position_errors = np.linalg.norm(
                aligned_estimated - ground_truth_positions, axis=1
            )
            ate = float(np.sqrt(np.mean(position_errors**2)))  # RMSE

            # Store in history
            self.ate_history.append(ate)
            if len(self.ate_history) > self.max_history_length:
                self.ate_history.pop(0)

            return ate

        except Exception as e:
            self.logger.error(f"ATE calculation error: {e}")
            return 0.0

    def _associate_trajectories(
        self,
        estimated: List[Tuple[Tuple[float, float, float], float]],
        ground_truth: List[Tuple[Tuple[float, float, float], float]],
    ) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Associate estimated and ground truth poses by timestamp.

        Args:
            estimated: Estimated trajectory
            ground_truth: Ground truth trajectory

        Returns:
            List of associated pose pairs
        """
        associated = []

        # Sort both trajectories by timestamp
        estimated_sorted = sorted(estimated, key=lambda x: x[1])
        ground_truth_sorted = sorted(ground_truth, key=lambda x: x[1])

        # Associate by closest timestamp
        for est_pos, est_time in estimated_sorted:
            best_match = None
            min_time_diff = float("inf")

            for gt_pos, gt_time in ground_truth_sorted:
                time_diff = abs(est_time - gt_time)

                if time_diff < min_time_diff and time_diff <= self.max_time_difference:
                    min_time_diff = time_diff
                    best_match = gt_pos

            if best_match is not None:
                associated.append((est_pos, best_match))

        return associated

    def _align_trajectories(
        self, estimated: np.ndarray, ground_truth: np.ndarray
    ) -> np.ndarray:
        """
        Align estimated trajectory to ground truth using similarity transformation.

        Args:
            estimated: Estimated positions [N, 3]
            ground_truth: Ground truth positions [N, 3]

        Returns:
            Aligned estimated trajectory [N, 3]
        """
        if self.alignment_method == "none":
            return estimated

        # Center both trajectories
        est_center = np.mean(estimated, axis=0)
        gt_center = np.mean(ground_truth, axis=0)

        est_centered = estimated - est_center
        gt_centered = ground_truth - gt_center

        if self.alignment_method == "translation_only":
            # Only translation alignment
            return estimated - est_center + gt_center

        elif self.alignment_method == "sim3" or self.alignment_method == "se3":
            # Similarity or rigid transformation

            # Calculate optimal rotation using Procrustes analysis
            H = est_centered.T @ gt_centered
            U, _, Vt = np.linalg.svd(H)
            R_opt = Vt.T @ U.T

            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R_opt) < 0:
                Vt[-1, :] *= -1
                R_opt = Vt.T @ U.T

            # Calculate optimal scale (for similarity transform)
            if self.alignment_method == "sim3":
                numerator = np.sum(gt_centered * (est_centered @ R_opt))
                denominator = np.sum(est_centered * est_centered)
                scale = numerator / denominator if denominator > 0 else 1.0
            else:
                scale = 1.0  # SE(3) has no scale

            # Apply transformation
            aligned = scale * (est_centered @ R_opt) + gt_center

            return aligned

        else:
            self.logger.warning(f"Unknown alignment method: {self.alignment_method}")
            return estimated

    def calculate_rpe(
        self,
        estimated_trajectory: List[Tuple[Tuple[float, float, float], float]],
        delta_time: float = 1.0,
    ) -> float:
        """
        Calculate Relative Pose Error (RPE).

        Args:
            estimated_trajectory: Estimated trajectory
            delta_time: Time interval for RPE calculation

        Returns:
            RPE in meters/second
        """
        if len(estimated_trajectory) < 2:
            return 0.0

        try:
            rpe_errors = []

            # Calculate relative motions
            for i in range(1, len(estimated_trajectory)):
                pos1, time1 = estimated_trajectory[i - 1]
                pos2, time2 = estimated_trajectory[i]

                dt = time2 - time1
                if dt <= 0 or dt > delta_time * 2:  # Skip invalid intervals
                    continue

                # Estimated relative motion
                est_motion = np.array(pos2) - np.array(pos1)
                est_distance = np.linalg.norm(est_motion)

                # Find corresponding ground truth motion
                gt_motion = self._get_ground_truth_motion(time1, time2)

                if gt_motion is not None:
                    gt_distance = np.linalg.norm(gt_motion)

                    # RPE as difference in motion magnitudes
                    rpe_error = abs(est_distance - gt_distance) / dt
                    rpe_errors.append(rpe_error)

            if rpe_errors:
                return float(np.sqrt(np.mean(np.array(rpe_errors) ** 2)))
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"RPE calculation error: {e}")
            return 0.0

    def _get_ground_truth_motion(
        self, time1: float, time2: float
    ) -> Optional[np.ndarray]:
        """
        Get ground truth motion between two timestamps.

        Args:
            time1, time2: Start and end timestamps

        Returns:
            Ground truth motion vector or None
        """
        # Find ground truth poses closest to timestamps
        gt1 = None
        gt2 = None
        min_diff1 = float("inf")
        min_diff2 = float("inf")

        for gt_pos, gt_time in self.ground_truth:
            diff1 = abs(gt_time - time1)
            diff2 = abs(gt_time - time2)

            if diff1 < min_diff1 and diff1 <= self.max_time_difference:
                gt1 = gt_pos
                min_diff1 = diff1

            if diff2 < min_diff2 and diff2 <= self.max_time_difference:
                gt2 = gt_pos
                min_diff2 = diff2

        if gt1 is not None and gt2 is not None:
            return np.array(gt2) - np.array(gt1)

        return None

    def get_ate_statistics(self) -> Dict[str, Any]:
        """
        Get ATE statistics.

        Returns:
            Statistics dictionary
        """
        if not self.ate_history:
            return {"message": "No ATE calculations performed yet"}

        ate_array = np.array(self.ate_history)

        return {
            "current_ate": self.ate_history[-1] if self.ate_history else 0.0,
            "mean_ate": float(np.mean(ate_array)),
            "std_ate": float(np.std(ate_array)),
            "min_ate": float(np.min(ate_array)),
            "max_ate": float(np.max(ate_array)),
            "median_ate": float(np.median(ate_array)),
            "calculations_performed": len(self.ate_history),
            "target_comparison": {
                "euroc_target": self.target_ate_euroc,
                "tumvi_target": self.target_ate_tumvi,
                "meets_euroc_target": (
                    self.ate_history[-1] <= self.target_ate_euroc
                    if self.ate_history
                    else False
                ),
                "meets_tumvi_target": (
                    self.ate_history[-1] <= self.target_ate_tumvi
                    if self.ate_history
                    else False
                ),
            },
        }

    def validate_accuracy(self) -> Dict[str, bool]:
        """
        Validate trajectory accuracy against targets.

        Returns:
            Validation results
        """
        if not self.ate_history:
            return {"no_data": True}

        current_ate = self.ate_history[-1]

        return {
            "centimeter_scale": current_ate <= 0.05,  # < 5cm
            "euroc_accuracy": current_ate <= self.target_ate_euroc,
            "tumvi_accuracy": current_ate <= self.target_ate_tumvi,
            "consistent_accuracy": (
                np.std(self.ate_history[-50:]) <= 0.01
                if len(self.ate_history) >= 50
                else True
            ),
        }

    def reset(self):
        """Reset ATE calculator."""
        self.ate_history.clear()
        self.logger.debug("ATE Calculator reset")


# Backwards compatibility alias for legacy imports
ATCalculator = ATECalculator
