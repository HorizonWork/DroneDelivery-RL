import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time

class ATECalculator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.alignment_method = config.get('alignment_method', 'sim3')
        self.max_time_difference = config.get('max_time_difference', 0.02)
        self.min_trajectory_length = config.get('min_trajectory_length', 10)

        self.ground_truth: List[Tuple[Tuple[float, float, float], float]] = []

        self.ate_history: List[float] = []
        self.max_history_length = config.get('max_history_length', 1000)

        self.target_ate_euroc = 0.036
        self.target_ate_tumvi = 0.009
        self.target_rpe = 0.050

        self.logger.info("ATE Calculator initialized")
        self.logger.info(f"Alignment method: {self.alignment_method}")
        self.logger.info(f"Target ATE: EuRoC={self.target_ate_euroc100:.1f}cm, TUM-VI={self.target_ate_tumvi10:.0f}mm")

    def set_ground_truth(self, ground_truth: List[Tuple[Tuple[float, float, float], float]]):

        self.ground_truth = ground_truth.copy()
        self.logger.info(f"Ground truth set: {len(ground_truth)} poses")

    def calculate_ate(self, estimated_trajectory: List[Tuple[Tuple[float, float, float], float]],
                     ground_truth: Optional[List[Tuple[Tuple[float, float, float], float]]] = None) - float:

        if ground_truth is None:
            ground_truth = self.ground_truth

        if not ground_truth or not estimated_trajectory:
            return 0.0

        if len(estimated_trajectory)  self.min_trajectory_length:
            return 0.0

        try:
            associated_pairs = self._associate_trajectories(estimated_trajectory, ground_truth)

            if len(associated_pairs)  self.min_trajectory_length:
                self.logger.warning(f"Insufficient associated poses: {len(associated_pairs)}")
                return 0.0

            estimated_positions = np.array([pair[0] for pair in associated_pairs])
            ground_truth_positions = np.array([pair[1] for pair in associated_pairs])

            aligned_estimated = self._align_trajectories(estimated_positions, ground_truth_positions)

            position_errors = np.linalg.norm(aligned_estimated - ground_truth_positions, axis=1)
            ate = float(np.sqrt(np.mean(position_errors  2)))

            self.ate_history.append(ate)
            if len(self.ate_history)  self.max_history_length:
                self.ate_history.pop(0)

            return ate

        except Exception as e:
            self.logger.error(f"ATE calculation error: {e}")
            return 0.0

    def _associate_trajectories(self, estimated: List[Tuple[Tuple[float, float, float], float]],
                              ground_truth: List[Tuple[Tuple[float, float, float], float]]) - List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:

        associated = []

        estimated_sorted = sorted(estimated, key=lambda x: x[1])
        ground_truth_sorted = sorted(ground_truth, key=lambda x: x[1])

        for est_pos, est_time in estimated_sorted:
            best_match = None
            min_time_diff = float('inf')

            for gt_pos, gt_time in ground_truth_sorted:
                time_diff = abs(est_time - gt_time)

                if time_diff  min_time_diff and time_diff = self.max_time_difference:
                    min_time_diff = time_diff
                    best_match = gt_pos

            if best_match is not None:
                associated.append((est_pos, best_match))

        return associated

    def _align_trajectories(self, estimated: np.ndarray, ground_truth: np.ndarray) - np.ndarray:

        if self.alignment_method == 'none':
            return estimated

        est_center = np.mean(estimated, axis=0)
        gt_center = np.mean(ground_truth, axis=0)

        est_centered = estimated - est_center
        gt_centered = ground_truth - gt_center

        if self.alignment_method == 'translation_only':
            return estimated - est_center + gt_center

        elif self.alignment_method == 'sim3' or self.alignment_method == 'se3':

            H = est_centered.T  gt_centered
            U, _, Vt = np.linalg.svd(H)
            R_opt = Vt.T  U.T

            if np.linalg.det(R_opt)  0:
                Vt[-1, :] = -1
                R_opt = Vt.T  U.T

            if self.alignment_method == 'sim3':
                numerator = np.sum(gt_centered  (est_centered  R_opt))
                denominator = np.sum(est_centered  est_centered)
                scale = numerator / denominator if denominator  0 else 1.0
            else:
                scale = 1.0

            aligned = scale  (est_centered  R_opt) + gt_center

            return aligned

        else:
            self.logger.warning(f"Unknown alignment method: {self.alignment_method}")
            return estimated

    def calculate_rpe(self, estimated_trajectory: List[Tuple[Tuple[float, float, float], float]],
                     delta_time: float = 1.0) - float:

        if len(estimated_trajectory)  2:
            return 0.0

        try:
            rpe_errors = []

            for i in range(1, len(estimated_trajectory)):
                pos1, time1 = estimated_trajectory[i-1]
                pos2, time2 = estimated_trajectory[i]

                dt = time2 - time1
                if dt = 0 or dt  delta_time  2:
                    continue

                est_motion = np.array(pos2) - np.array(pos1)
                est_distance = np.linalg.norm(est_motion)

                gt_motion = self._get_ground_truth_motion(time1, time2)

                if gt_motion is not None:
                    gt_distance = np.linalg.norm(gt_motion)

                    rpe_error = abs(est_distance - gt_distance) / dt
                    rpe_errors.append(rpe_error)

            if rpe_errors:
                return float(np.sqrt(np.mean(np.array(rpe_errors)  2)))
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"RPE calculation error: {e}")
            return 0.0

    def _get_ground_truth_motion(self, time1: float, time2: float) - Optional[np.ndarray]:

        gt1 = None
        gt2 = None
        min_diff1 = float('inf')
        min_diff2 = float('inf')

        for gt_pos, gt_time in self.ground_truth:
            diff1 = abs(gt_time - time1)
            diff2 = abs(gt_time - time2)

            if diff1  min_diff1 and diff1 = self.max_time_difference:
                gt1 = gt_pos
                min_diff1 = diff1

            if diff2  min_diff2 and diff2 = self.max_time_difference:
                gt2 = gt_pos
                min_diff2 = diff2

        if gt1 is not None and gt2 is not None:
            return np.array(gt2) - np.array(gt1)

        return None

    def get_ate_statistics(self) - Dict[str, Any]:

        if not self.ate_history:
            return {'message': 'No ATE calculations performed yet'}

        ate_array = np.array(self.ate_history)

        return {
            'current_ate': self.ate_history[-1] if self.ate_history else 0.0,
            'mean_ate': float(np.mean(ate_array)),
            'std_ate': float(np.std(ate_array)),
            'min_ate': float(np.min(ate_array)),
            'max_ate': float(np.max(ate_array)),
            'median_ate': float(np.median(ate_array)),
            'calculations_performed': len(self.ate_history),
            'target_comparison': {
                'euroc_target': self.target_ate_euroc,
                'tumvi_target': self.target_ate_tumvi,
                'meets_euroc_target': self.ate_history[-1] = self.target_ate_euroc if self.ate_history else False,
                'meets_tumvi_target': self.ate_history[-1] = self.target_ate_tumvi if self.ate_history else False
            }
        }

    def validate_accuracy(self) - Dict[str, bool]:

        if not self.ate_history:
            return {'no_data': True}

        current_ate = self.ate_history[-1]

        return {
            'centimeter_scale': current_ate = 0.05,
            'euroc_accuracy': current_ate = self.target_ate_euroc,
            'tumvi_accuracy': current_ate = self.target_ate_tumvi,
            'consistent_accuracy': np.std(self.ate_history[-50:]) = 0.01 if len(self.ate_history) = 50 else True
        }

    def reset(self):

        self.ate_history.clear()
        self.logger.debug("ATE Calculator reset")
