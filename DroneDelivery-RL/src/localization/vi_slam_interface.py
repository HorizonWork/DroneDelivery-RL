import numpy as np
import logging
import time
import threading
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque

from src.localization.orb_slam3_wrapper import ORBSLAM3Wrapper
from src.localization.coordinate_transforms import CoordinateTransforms
from src.localization.ate_calculator import ATECalculator

from src.utils.imu_preintegration import IMUPreintegrator

dataclass
class SLAMOutput:

    pose: Optional[Tuple[float, float, float]]
    orientation: Optional[Tuple[float, float, float, float]]
    covariance: Optional[np.ndarray]
    ate_error: float
    tracking_quality: str
    num_features: int
    timestamp: float

class VISLAMInterface:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.orb_slam = ORBSLAM3Wrapper(config.get('orb_slam3', {}))
        self.coord_transforms = CoordinateTransforms(config.get('transforms', {}))
        self.imu_preintegrator = IMUPreintegrator(config.get('imu', {}))
        self.ate_calculator = ATECalculator(config.get('ate', {}))

        self.stereo_queue: deque = deque(maxlen=30)
        self.imu_queue: deque = deque(maxlen=200)

        self.is_running = False
        self.current_output: Optional[SLAMOutput] = None
        self.pose_history: List[SLAMOutput] = []

        self.processing_thread: Optional[threading.Thread] = None
        self.data_lock = threading.Lock()

        self.pose_callbacks: List[Callable[[SLAMOutput], None]] = []

        self.processing_fps = 0.0
        self.last_process_time = 0.0

        self.logger.info("VI-SLAM Interface initialized")

    def start(self) - bool:

        try:
            if not self.orb_slam.initialize():
                return False

            self.imu_preintegrator.start()

            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            self.logger.info("VI-SLAM interface started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start VI-SLAM: {e}")
            return False

    def stop(self):

        self.is_running = False

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)

        self.orb_slam.shutdown()
        self.imu_preintegrator.stop()

        self.logger.info("VI-SLAM interface stopped")

    def process_stereo_frame(self, left_image: np.ndarray, right_image: np.ndarray,
                           timestamp: float):

        if not self.is_running:
            return

        with self.data_lock:
            self.stereo_queue.append({
                'left': left_image,
                'right': right_image,
                'timestamp': timestamp
            })

    def process_imu_measurement(self, imu_data: Dict[str, Any], timestamp: float):

        if not self.is_running:
            return

        with self.data_lock:
            self.imu_queue.append({
                'data': imu_data,
                'timestamp': timestamp
            })

    def _processing_loop(self):

        self.logger.info("VI-SLAM processing loop started")

        while self.is_running:
            try:
                process_start = time.time()

                self._process_imu_queue()

                pose_updated = self._process_stereo_queue()

                if pose_updated:
                    self._update_ate()

                    process_time = time.time() - process_start
                    if process_time  0:
                        self.processing_fps = 0.9  self.processing_fps + 0.1 / process_time

                time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"VI-SLAM processing error: {e}")
                time.sleep(0.1)

    def _process_imu_queue(self):

        with self.data_lock:
            imu_batch = list(self.imu_queue)
            self.imu_queue.clear()

        for imu_item in imu_batch:
            try:
                self.imu_preintegrator.add_measurement(
                    imu_item['data'], imu_item['timestamp']
                )

                accel = np.array(imu_item['data']['linear_acceleration'])
                gyro = np.array(imu_item['data']['angular_velocity'])

                self.orb_slam.process_imu_measurement(accel, gyro, imu_item['timestamp'])

            except Exception as e:
                self.logger.error(f"IMU processing error: {e}")

    def _process_stereo_queue(self) - bool:

        pose_updated = False

        with self.data_lock:
            stereo_batch = list(self.stereo_queue)
            self.stereo_queue.clear()

        for stereo_item in stereo_batch:
            try:
                pose_estimate = self.orb_slam.process_stereo_frame(
                    stereo_item['left'],
                    stereo_item['right'],
                    stereo_item['timestamp']
                )

                if pose_estimate is not None:
                    world_pos, world_quat = self.coord_transforms.transform_pose(
                        tuple(pose_estimate.position),
                        self._rotation_matrix_to_quaternion(pose_estimate.orientation),
                        'camera', 'world'
                    )

                    slam_output = SLAMOutput(
                        pose=world_pos,
                        orientation=world_quat,
                        covariance=None,
                        ate_error=0.0,
                        tracking_quality="GOOD" if pose_estimate.confidence  0.7 else "OK" if pose_estimate.confidence  0.5 else "POOR",
                        num_features=pose_estimate.num_inliers,
                        timestamp=stereo_item['timestamp']
                    )

                    self.current_output = slam_output
                    self.pose_history.append(slam_output)

                    if len(self.pose_history)  10000:
                        self.pose_history.pop(0)

                    for callback in self.pose_callbacks:
                        try:
                            callback(slam_output)
                        except Exception as e:
                            self.logger.error(f"Pose callback error: {e}")

                    pose_updated = True

            except Exception as e:
                self.logger.error(f"Stereo processing error: {e}")

        return pose_updated

    def _rotation_matrix_to_quaternion(self, rotation_matrix: np.ndarray) - Tuple[float, float, float, float]:

        rotation = R.from_matrix(rotation_matrix)
        quat = rotation.as_quat()
        return (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))

    def _update_ate(self):

        if len(self.pose_history)  2:
            return

        trajectory = [(output.pose, output.timestamp) for output in self.pose_history[-100:]]

        ate = self.ate_calculator.calculate_ate(trajectory)

        if self.current_output:
            self.current_output.ate_error = ate

    def get_current_pose(self) - Optional[SLAMOutput]:

        return self.current_output

    def get_ate_error(self) - float:

        return self.current_output.ate_error if self.current_output else 0.0

    def set_ground_truth(self, ground_truth: List[Tuple[Tuple[float, float, float], float]]):

        self.ate_calculator.set_ground_truth(ground_truth)

    def add_pose_callback(self, callback: Callable[[SLAMOutput], None]):

        self.pose_callbacks.append(callback)

    def get_statistics(self) - Dict[str, Any]:

        orb_stats = {
            'tracking_state': self.orb_slam.get_tracking_state(),
            'keyframes': self.orb_slam.keyframe_count,
            'map_points': len(self.orb_slam.get_map_points())
        }

        ate_stats = self.ate_calculator.get_ate_statistics()
        imu_stats = self.imu_preintegrator.get_statistics()

        return {
            'orb_slam3': orb_stats,
            'ate_calculator': ate_stats,
            'imu_preintegrator': imu_stats,
            'interface': {
                'processing_fps': self.processing_fps,
                'poses_tracked': len(self.pose_history),
                'stereo_queue_size': len(self.stereo_queue),
                'imu_queue_size': len(self.imu_queue)
            }
        }

    def reset(self):

        with self.data_lock:
            self.stereo_queue.clear()
            self.imu_queue.clear()

        self.orb_slam.reset()
        self.imu_preintegrator.reset()
        self.ate_calculator.reset()

        self.current_output = None
        self.pose_history.clear()
        self.processing_fps = 0.0

        self.logger.info("VI-SLAM interface reset")
