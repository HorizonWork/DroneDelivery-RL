import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

dataclass
class Pose:

    position: np.ndarray
    orientation: np.ndarray

    def to_matrix(self) - np.ndarray:

        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(self.orientation).as_matrix()
        T[:3, 3] = self.position
        return T

    classmethod
    def from_matrix(cls, T: np.ndarray):

        position = T[:3, 3]
        rotation = Rotation.from_matrix(T[:3, :3])
        orientation = rotation.as_quat()
        return cls(position, orientation)

class CoordinateTransformer:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.frames = {
            "world": "ENU",
            "body": "FRD",
            "camera": "FRD",
            "slam": "arbitrary",
        }

        self.body_to_camera_offset = np.array(
            config.get("body_to_camera_offset", [0.1, 0.0, -0.05])
        )
        self.slam_to_world_transform = np.eye(4)

        self.grid_resolution = config.get("grid_resolution", 0.5)
        self.grid_origin = np.array(
            config.get("grid_origin", [0, 0, 0])
        )

        self.building_bounds = {
            "x_min": config.get("x_min", 0.0),
            "x_max": config.get("x_max", 20.0),
            "y_min": config.get("y_min", 0.0),
            "y_max": config.get("y_max", 40.0),
            "z_min": config.get("z_min", 0.0),
            "z_max": config.get("z_max", 15.0),
        }

        self.logger.info("Coordinate Transformer initialized")
        self.logger.info(f"Grid resolution: {self.grid_resolution}m")
        self.logger.info(f"Building bounds: {self.building_bounds}")

    def world_to_grid(self, world_position: np.ndarray) - Tuple[int, int, int]:

        relative_pos = world_position - self.grid_origin
        grid_indices = (relative_pos / self.grid_resolution).astype(int)

        max_indices = np.array(
            [
                int(
                    (self.building_bounds["x_max"] - self.building_bounds["x_min"])
                    / self.grid_resolution
                ),
                int(
                    (self.building_bounds["y_max"] - self.building_bounds["y_min"])
                    / self.grid_resolution
                ),
                int(
                    (self.building_bounds["z_max"] - self.building_bounds["z_min"])
                    / self.grid_resolution
                ),
            ]
        )

        grid_indices = np.clip(grid_indices, 0, max_indices - 1)

        return tuple(grid_indices)

    def grid_to_world(self, grid_indices: Tuple[int, int, int]) - np.ndarray:

        grid_pos = np.array(grid_indices)  self.grid_resolution
        world_position = grid_pos + self.grid_origin

        return world_position

    def world_to_grid(
        world_position: Tuple[float, float, float],
        cell_size: float = 0.5,
        floor_height: float = 3.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) - Tuple[int, int, int]:

        x, y, z = world_position
        ox, oy, oz = origin
        gx = int(np.floor((x - ox) / cell_size))
        gy = int(np.floor((y - oy) / cell_size))
        gz = int(np.floor((z - oz) / floor_height))
        return (gx, gy, gz)

    def grid_to_world(
        grid_indices: Tuple[int, int, int],
        cell_size: float = 0.5,
        floor_height: float = 3.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) - Tuple[float, float, float]:

        gx, gy, gz = grid_indices
        ox, oy, oz = origin
        x = ox + (gx + 0.5)  cell_size
        y = oy + (gy + 0.5)  cell_size
        z = oz + gz  floor_height + 0.5  floor_height
        return (float(x), float(y), float(z))

    def body_to_world(self, body_position: np.ndarray, drone_pose: Pose) - np.ndarray:

        T_world_body = drone_pose.to_matrix()

        body_homogeneous = np.append(body_position, 1.0)
        world_homogeneous = T_world_body  body_homogeneous

        return world_homogeneous[:3]

    def world_to_body(self, world_position: np.ndarray, drone_pose: Pose) - np.ndarray:

        T_world_body = drone_pose.to_matrix()
        T_body_world = np.linalg.inv(T_world_body)

        world_homogeneous = np.append(world_position, 1.0)
        body_homogeneous = T_body_world  world_homogeneous

        return body_homogeneous[:3]

    def slam_to_world(self, slam_pose: Pose) - Pose:

        slam_matrix = slam_pose.to_matrix()
        world_matrix = self.slam_to_world_transform  slam_matrix

        return Pose.from_matrix(world_matrix)

    def calibrate_slam_to_world(
        self, slam_positions: List[np.ndarray], world_positions: List[np.ndarray]
    ):

        if len(slam_positions)  3:
            self.logger.warning("Need at least 3 points for calibration")
            return

        slam_points = np.array(slam_positions)
        world_points = np.array(world_positions)

        slam_centroid = np.mean(slam_points, axis=0)
        world_centroid = np.mean(world_points, axis=0)

        translation = world_centroid - slam_centroid

        slam_scale = np.linalg.norm(slam_points - slam_centroid, axis=1)
        world_scale = np.linalg.norm(world_points - world_centroid, axis=1)
        scale = np.mean(world_scale / (slam_scale + 1e-6))

        self.slam_to_world_transform[:3, :3] = scale
        self.slam_to_world_transform[:3, 3] = translation

        self.logger.info(
            f"SLAM calibration: scale={scale:.3f}, translation={translation}"
        )

    def get_floor_number(self, world_position: np.ndarray) - int:

        floor = max(1, min(5, int(world_position[2]
        return floor

    def get_floor_bounds(self, floor_number: int) - Dict[str, float]:

        floor_height = 3.0

        return {
            "x_min": self.building_bounds["x_min"],
            "x_max": self.building_bounds["x_max"],
            "y_min": self.building_bounds["y_min"],
            "y_max": self.building_bounds["y_max"],
            "z_min": (floor_number - 1)  floor_height,
            "z_max": floor_number  floor_height,
        }

    def is_position_valid(self, world_position: np.ndarray) - bool:

        x, y, z = world_position

        return (
            self.building_bounds["x_min"] = x = self.building_bounds["x_max"]
            and self.building_bounds["y_min"] = y = self.building_bounds["y_max"]
            and self.building_bounds["z_min"] = z = self.building_bounds["z_max"]
        )

class FrameConverter:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.use_enu = config.get(
            "use_enu_convention", True
        )

        self.camera_offset = np.array(config.get("camera_offset", [0.1, 0.0, -0.05]))
        self.imu_offset = np.array(config.get("imu_offset", [0.0, 0.0, 0.0]))

        if self.use_enu:
            self.enu_to_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            self.ned_to_enu = self.enu_to_ned.T

        self.logger.info("Frame Converter initialized")
        self.logger.info(f"Convention: {'ENU' if self.use_enu else 'NED'}")

    def ned_to_enu(self, ned_vector: np.ndarray) - np.ndarray:

        if not self.use_enu:
            return ned_vector
        return self.ned_to_enu  ned_vector

    def enu_to_ned(self, enu_vector: np.ndarray) - np.ndarray:

        if not self.use_enu:
            return enu_vector
        return self.enu_to_ned  enu_vector

    def body_velocity_to_world(
        self, body_velocity: np.ndarray, orientation_quat: np.ndarray
    ) - np.ndarray:

        rotation = Rotation.from_quat(orientation_quat)
        rotation_matrix = rotation.as_matrix()

        world_velocity = rotation_matrix  body_velocity

        return world_velocity

    def world_velocity_to_body(
        self, world_velocity: np.ndarray, orientation_quat: np.ndarray
    ) - np.ndarray:

        rotation = Rotation.from_quat(orientation_quat)
        rotation_matrix = rotation.as_matrix().T

        body_velocity = rotation_matrix  world_velocity

        return body_velocity

    def get_relative_position(
        self, position_a: np.ndarray, position_b: np.ndarray
    ) - Dict[str, float]:

        diff = position_b - position_a

        return {
            "distance": float(np.linalg.norm(diff)),
            "horizontal_distance": float(np.linalg.norm(diff[:2])),
            "vertical_distance": float(abs(diff[2])),
            "bearing": float(np.arctan2(diff[1], diff[0])),
            "elevation": float(np.arctan2(diff[2], np.linalg.norm(diff[:2]))),
            "direction_vector": diff / (np.linalg.norm(diff) + 1e-8),
        }
