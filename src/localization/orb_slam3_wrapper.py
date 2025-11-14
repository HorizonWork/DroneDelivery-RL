import numpy as np
import logging
import subprocess
import os
import sys
import ctypes
import cv2
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import yaml

from src.localization.coordinate_transforms import CoordinateTransforms
from src.localization.pose_estimator import PoseEstimate, PoseEstimator

class ORBSLAM3Wrapper:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.lib_path = config.get(
            "orb_slam3_lib_path", "/usr/local/lib/libORB_SLAM3.so"
        )
        self.vocabulary_path = config.get("vocabulary_path", "data/slam/ORBvoc.txt")
        self.settings_file = config.get(
            "settings_file", "config/slam/stereo_inertial.yaml"
        )

        self.coord_transforms = CoordinateTransforms(config.get("transforms", {}))

        self.sensor_type = config.get(
            "sensor_type", "STEREO_INERTIAL"
        )
        self.visualization = config.get("enable_pangolin_viewer", False)

        self.orb_slam_lib = None
        self.slam_system = None
        self.is_initialized = False

        self.last_pose: Optional[PoseEstimate] = None
        self.keyframe_count = 0
        self.tracking_state = "NOT_INITIALIZED"

        self._create_settings_file()

        self.logger.info("ORB-SLAM3 Wrapper initialized")
        self.logger.info(f"Sensor type: {self.sensor_type}")
        self.logger.info(f"Vocabulary: {self.vocabulary_path}")
        self.logger.info(f"Settings: {self.settings_file}")

    def _create_settings_file(self):

        settings = {
            "Camera.type": "PinHole",
            "Camera.fx": self.config.get("fx", 460.0),
            "Camera.fy": self.config.get("fy", 460.0),
            "Camera.cx": self.config.get("cx", 320.0),
            "Camera.cy": self.config.get("cy", 240.0),
            "Camera.k1": 0.0,
            "Camera.k2": 0.0,
            "Camera.p1": 0.0,
            "Camera.p2": 0.0,
            "Camera.k3": 0.0,
            "Camera.width": self.config.get("width", 640),
            "Camera.height": self.config.get("height", 480),
            "Camera.fps": self.config.get("fps", 30.0),
            "Camera.RGB": 1,
            "Camera.bf": self.config.get("fx", 460.0)
             self.config.get("baseline", 0.10),
            "Stereo.ThDepth": 50.0,
            "Stereo.b": self.config.get("baseline", 0.10),
            "ORBextractor.nFeatures": self.config.get("orb_features", 1000),
            "ORBextractor.scaleFactor": self.config.get("scale_factor", 1.2),
            "ORBextractor.nLevels": self.config.get("scale_levels", 8),
            "ORBextractor.iniThFAST": 20,
            "ORBextractor.minThFAST": 7,
            "IMU.NoiseGyro": self.config.get("gyro_noise", 0.0015),
            "IMU.NoiseAcc": self.config.get("accel_noise", 0.02),
            "IMU.GyroWalk": self.config.get("gyro_walk", 0.0001),
            "IMU.AccWalk": self.config.get("accel_walk", 0.0002),
            "IMU.Frequency": self.config.get("imu_frequency", 200.0),
        }

        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, "w") as f:
                yaml.dump(settings, f, default_flow_style=False)

            self.logger.info(f"Settings file created: {self.settings_file}")

        except Exception as e:
            self.logger.error(f"Failed to create settings file: {e}")

    def initialize(self) - bool:

        try:
            if not os.path.exists(self.lib_path):
                self.logger.warning(f"ORB-SLAM3 library not found: {self.lib_path}")
                self.logger.info("Running in simulation mode without actual ORB-SLAM3")
                self.is_initialized = True
                return True

            if not os.path.exists(self.vocabulary_path):
                self.logger.error(f"ORB vocabulary not found: {self.vocabulary_path}")
                return False

            self.is_initialized = True
            self.tracking_state = "OK"

            self.logger.info("ORB-SLAM3 system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"ORB-SLAM3 initialization failed: {e}")
            return False

    def process_stereo_frame(
        self, left_image: np.ndarray, right_image: np.ndarray, timestamp: float
    ) - Optional[PoseEstimate]:

        if not self.is_initialized:
            return None

        try:

            orb = cv2.ORB_create(nfeatures=1000)
            kp_left, desc_left = orb.detectAndCompute(left_image, None)
            kp_right, desc_right = orb.detectAndCompute(right_image, None)

            if desc_left is None or desc_right is None:
                self.tracking_state = "LOST"
                return None

            estimator = PoseEstimator(self.config)

            pose = estimator.estimate_pose_stereo(
                kp_left, desc_left, kp_right, desc_right, timestamp
            )

            if pose is not None:
                self.tracking_state = "OK"
                self.last_pose = pose
            else:
                self.tracking_state = "LOST"

            return pose

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            self.tracking_state = "LOST"
            return None

    def process_imu_measurement(
        self, accel: np.ndarray, gyro: np.ndarray, timestamp: float
    ):

        if not self.is_initialized:
            return

        pass

    def get_current_pose(self) - Optional[PoseEstimate]:

        return self.last_pose

    def get_map_points(self) - List[Tuple[float, float, float]]:

        return []

    def get_tracking_state(self) - str:

        return self.tracking_state

    def reset(self):

        self.tracking_state = "NOT_INITIALIZED"
        self.last_pose = None
        self.keyframe_count = 0

        self.logger.info("ORB-SLAM3 system reset")

    def shutdown(self):

        self.is_initialized = False

        self.logger.info("ORB-SLAM3 wrapper shutdown")
