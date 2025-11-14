import numpy as np
import logging
import time
import threading
import subprocess
import os
import yaml
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R

dataclass
class SLAMState:

    is_initialized: bool = False
    is_tracking: bool = False
    num_map_points: int = 0
    num_keyframes: int = 0
    last_pose_timestamp: float = 0.0
    tracking_quality: str = "UNKNOWN"

dataclass
class TrajectoryPoint:

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    timestamp: float
    covariance: Optional[np.ndarray] = None

class SLAMBridge:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.slam_config_path = config.get('slam_config_path', 'config/slam/orb_slam3_config.yaml')
        self.vocabulary_path = config.get('vocabulary_path', 'data/slam/ORBvoc.txt')
        self.camera_config_path = config.get('camera_config_path', 'config/slam/camera_calibration.yaml')
        self.imu_config_path = config.get('imu_config_path', 'config/slam/imu_calibration.yaml')

        self.trajectory_output = config.get('trajectory_output', 'data/slam/trajectories/trajectory.tum')
        self.map_output = config.get('map_output', 'data/slam/maps/map.bin')

        self.slam_process: Optional[subprocess.Popen] = None
        self.slam_thread: Optional[threading.Thread] = None
        self.is_running = False

        self.slam_state = SLAMState()
        self.trajectory: List[TrajectoryPoint] = []
        self.current_pose: Optional[TrajectoryPoint] = None
        self.map_points: List[Tuple[float, float, float]] = []

        self.ground_truth_trajectory: List[TrajectoryPoint] = []
        self.current_ate: float = 0.0

        self.pose_callbacks: List[Callable[[TrajectoryPoint], None]] = []
        self.state_callbacks: List[Callable[[SLAMState], None]] = []

        self.camera_params = self._load_camera_parameters()
        self.imu_params = self._load_imu_parameters()

        self.image_queue: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.imu_queue: List[Tuple[Dict[str, Any], float]] = []
        self.max_queue_size = config.get('max_queue_size', 100)

        self.data_lock = threading.Lock()

        self.last_processed_image_time = 0.0
        self.last_processed_imu_time = 0.0

        self.logger.info("SLAM Bridge initialized")

    def _load_camera_parameters(self) - Dict[str, Any]:

        try:
            with open(self.camera_config_path, 'r') as f:
                params = yaml.safe_load(f)

            required_keys = ['left', 'right', 'stereo']
            for key in required_keys:
                if key not in params:
                    raise ValueError(f"Missing camera parameter: {key}")

            self.logger.info("Camera parameters loaded successfully")
            return params

        except Exception as e:
            self.logger.error(f"Failed to load camera parameters: {e}")
            return {
                'left': {
                    'width': 640, 'height': 480,
                    'fx': 460.0, 'fy': 460.0, 'cx': 320.0, 'cy': 240.0,
                    'distortion': [0.0, 0.0, 0.0, 0.0]
                },
                'right': {
                    'width': 640, 'height': 480,
                    'fx': 460.0, 'fy': 460.0, 'cx': 320.0, 'cy': 240.0,
                    'distortion': [0.0, 0.0, 0.0, 0.0]
                },
                'stereo': {'baseline_m': 0.10, 'rectify': True}
            }

    def _load_imu_parameters(self) - Dict[str, Any]:

        try:
            with open(self.imu_config_path, 'r') as f:
                params = yaml.safe_load(f)

            self.logger.info("IMU parameters loaded successfully")
            return params

        except Exception as e:
            self.logger.error(f"Failed to load IMU parameters: {e}")
            return {
                'rate_hz': 200,
                'accelerometer': {
                    'noise_density': 0.02,
                    'random_walk': 0.0002,
                    'bias_init': [0.0, 0.0, 0.0]
                },
                'gyroscope': {
                    'noise_density': 0.0015,
                    'random_walk': 0.0001,
                    'bias_init': [0.0, 0.0, 0.0]
                },
                'gravity_mps2': 9.81
            }

    def start_slam(self) - bool:

        if self.is_running:
            self.logger.warning("SLAM already running")
            return True

        try:
            os.makedirs(os.path.dirname(self.trajectory_output), exist_ok=True)
            os.makedirs(os.path.dirname(self.map_output), exist_ok=True)

            self.is_running = True
            self.slam_thread = threading.Thread(target=self._slam_processing_loop)
            self.slam_thread.daemon = True
            self.slam_thread.start()

            self.logger.info("SLAM system started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start SLAM: {e}")
            self.is_running = False
            return False

    def stop_slam(self):

        if not self.is_running:
            return

        self.is_running = False

        if self.slam_thread and self.slam_thread.is_alive():
            self.slam_thread.join(timeout=5.0)

        if self.slam_process and self.slam_process.poll() is None:
            self.slam_process.terminate()
            self.slam_process.wait(timeout=5.0)

        self.logger.info("SLAM system stopped")

    def process_stereo_frame(self, left_image: np.ndarray, right_image: np.ndarray,
                           timestamp: float):

        if not self.is_running:
            return

        with self.data_lock:
            self.image_queue.append((left_image.copy(), right_image.copy(), timestamp))

            if len(self.image_queue)  self.max_queue_size:
                self.image_queue.pop(0)

    def process_imu_data(self, imu_data: Dict[str, Any], timestamp: float):

        if not self.is_running:
            return

        with self.data_lock:
            self.imu_queue.append((imu_data.copy(), timestamp))

            if len(self.imu_queue)  self.max_queue_size:
                self.imu_queue.pop(0)

    def _slam_processing_loop(self):

        self.logger.info("SLAM processing loop started")

        while self.is_running:
            try:
                self._process_queued_images()
                self._process_queued_imu()

                self._update_slam_state()

                self._calculate_ate()

                time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"SLAM processing error: {e}")
                time.sleep(0.1)

        self.logger.info("SLAM processing loop stopped")

    def _process_queued_images(self):

        with self.data_lock:
            if not self.image_queue:
                return

            left_img, right_img, timestamp = self.image_queue.pop(0)

        if timestamp = self.last_processed_image_time:
            return

        self.last_processed_image_time = timestamp

        pose = self._process_visual_frame(left_img, right_img, timestamp)

        if pose:
            self.current_pose = pose
            self.trajectory.append(pose)

            for callback in self.pose_callbacks:
                try:
                    callback(pose)
                except Exception as e:
                    self.logger.error(f"Pose callback error: {e}")

    def _process_queued_imu(self):

        with self.data_lock:
            if not self.imu_queue:
                return

            imu_batch = self.imu_queue.copy()
            self.imu_queue.clear()

        for imu_data, timestamp in imu_batch:
            if timestamp = self.last_processed_imu_time:
                continue

            self.last_processed_imu_time = timestamp

            self._process_imu_measurement(imu_data, timestamp)

    def _process_visual_frame(self, left_img: np.ndarray, right_img: np.ndarray,
                            timestamp: float) - Optional[TrajectoryPoint]:

        try:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)

            orb = cv2.ORB_create(nfeatures=1000)

            kp1, desc1 = orb.detectAndCompute(left_gray, None)
            kp2, desc2 = orb.detectAndCompute(right_gray, None)

            if desc1 is None or desc2 is None or len(kp1)  50:
                self.slam_state.tracking_quality = "POOR"
                return None

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches)  20:
                self.slam_state.tracking_quality = "POOR"
                return None

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]])

            K = np.array([
                [self.camera_params['left']['fx'], 0, self.camera_params['left']['cx']],
                [0, self.camera_params['left']['fy'], self.camera_params['left']['cy']],
                [0, 0, 1]
            ])

            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                         prob=0.999, threshold=1.0)

            if E is None:
                self.slam_state.tracking_quality = "LOST"
                return None

            _, R_mat, t_vec, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

            position = (float(t_vec[0, 0]), float(t_vec[1, 0]), float(t_vec[2, 0]))

            rotation = R.from_matrix(R_mat)
            quaternion = rotation.as_quat()
            orientation = (float(quaternion[3]), float(quaternion[0]),
                          float(quaternion[1]), float(quaternion[2]))

            pose = TrajectoryPoint(
                position=position,
                orientation=orientation,
                timestamp=timestamp
            )

            self.slam_state.tracking_quality = "GOOD"
            self.slam_state.is_tracking = True
            self.slam_state.num_map_points = len(matches)

            return pose

        except Exception as e:
            self.logger.error(f"Visual processing error: {e}")
            self.slam_state.tracking_quality = "LOST"
            return None

    def _process_imu_measurement(self, imu_data: Dict[str, Any], timestamp: float):

        accel = np.array(imu_data['linear_acceleration'])
        gyro = np.array(imu_data['angular_velocity'])

        accel_bias = np.array(self.imu_params['accelerometer']['bias_init'])
        gyro_bias = np.array(self.imu_params['gyroscope']['bias_init'])

        corrected_accel = accel - accel_bias
        corrected_gyro = gyro - gyro_bias

        pass

    def _update_slam_state(self):

        current_time = time.time()

        if (current_time - self.last_processed_image_time  1.0 and
            self.slam_state.is_tracking):
            self.slam_state.is_tracking = False
            self.slam_state.tracking_quality = "LOST"

        if len(self.trajectory)  10 and not self.slam_state.is_initialized:
            self.slam_state.is_initialized = True
            self.logger.info("SLAM system initialized")

        self.slam_state.num_keyframes = len(self.trajectory)

        for callback in self.state_callbacks:
            try:
                callback(self.slam_state)
            except Exception as e:
                self.logger.error(f"State callback error: {e}")

    def _calculate_ate(self):

        if not self.ground_truth_trajectory or not self.trajectory:
            return

        if len(self.trajectory) == 0:
            return

        latest_pose = self.trajectory[-1]

        closest_gt = None
        min_time_diff = float('inf')

        for gt_pose in self.ground_truth_trajectory:
            time_diff = abs(gt_pose.timestamp - latest_pose.timestamp)
            if time_diff  min_time_diff:
                min_time_diff = time_diff
                closest_gt = gt_pose

        if closest_gt and min_time_diff  0.1:
            pos_error = np.linalg.norm(
                np.array(latest_pose.position) - np.array(closest_gt.position)
            )

            self.current_ate = pos_error

    def get_current_pose(self) - Optional[TrajectoryPoint]:

        return self.current_pose

    def get_slam_state(self) - SLAMState:

        return self.slam_state

    def get_trajectory(self) - List[TrajectoryPoint]:

        return self.trajectory.copy()

    def get_ate(self) - float:

        return self.current_ate

    def set_ground_truth(self, ground_truth: List[TrajectoryPoint]):

        self.ground_truth_trajectory = ground_truth.copy()
        self.logger.info(f"Ground truth set with {len(ground_truth)} poses")

    def add_pose_callback(self, callback: Callable[[TrajectoryPoint], None]):

        self.pose_callbacks.append(callback)

    def add_state_callback(self, callback: Callable[[SLAMState], None]):

        self.state_callbacks.append(callback)

    def save_trajectory(self, filepath: Optional[str] = None):

        if filepath is None:
            filepath = self.trajectory_output

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                f.write("

                for point in self.trajectory:
                    f.write(f"{point.timestamp:.6f} "
                           f"{point.position[0]:.6f} {point.position[1]:.6f} {point.position[2]:.6f} "
                           f"{point.orientation[1]:.6f} {point.orientation[2]:.6f} "
                           f"{point.orientation[3]:.6f} {point.orientation[0]:.6f}\n")

            self.logger.info(f"Trajectory saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save trajectory: {e}")

    def reset(self):

        with self.data_lock:
            self.trajectory.clear()
            self.current_pose = None
            self.map_points.clear()
            self.image_queue.clear()
            self.imu_queue.clear()

        self.slam_state = SLAMState()
        self.current_ate = 0.0
        self.last_processed_image_time = 0.0
        self.last_processed_imu_time = 0.0

        self.logger.info("SLAM system reset")

    def __enter__(self):

        self.start_slam()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.stop_slam()
