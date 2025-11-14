import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from scipy.spatial.transform import Rotation as R

dataclass
class ObservationConfig:

    pose_dim: int = 3
    velocity_dim: int = 3
    goal_vector_dim: int = 3
    battery_dim: int = 1
    occupancy_dim: int = 24
    localization_error_dim: int = 1

    pose_update_hz: float = 200.0
    velocity_update_hz: float = 200.0
    occupancy_update_hz: float = 20.0

    position_bounds: Tuple[float, float] = (-50.0, 50.0)
    velocity_bounds: Tuple[float, float] = (-10.0, 10.0)
    goal_distance_bounds: Tuple[float, float] = (0.0, 100.0)
    occupancy_range: float = 5.0
    max_localization_error: float = 1.0

class ObservationSpace:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'observation': {
                    'pose_dim': 3,
                    'velocity_dim': 3,
                    'goal_vector_dim': 3,
                    'battery_dim': 1,
                    'occupancy_dim': 24,
                    'localization_error_dim': 1,
                    'position_bounds': (-50.0, 50.0),
                    'velocity_bounds': (-10.0, 10.0),
                    'goal_distance_bounds': (0.0, 100.0),
                    'occupancy_range': 5.0,
                    'max_localization_error': 1.0,
                    'pose_update_hz': 200.0,
                    'velocity_update_hz': 200.0,
                    'occupancy_update_hz': 20.0
                }
            }
        self.config = ObservationConfig(config.get('observation', {}))
        self.logger = logging.getLogger(__name__)

        self.total_dim = (
            self.config.pose_dim +
            self.config.velocity_dim +
            self.config.goal_vector_dim +
            self.config.battery_dim +
            self.config.occupancy_dim +
            self.config.localization_error_dim
        )

        assert self.total_dim == 35, f"Observation dim must be 35, got {self.total_dim}"

        self.obs_mean = np.zeros(self.total_dim)
        self.obs_var = np.ones(self.total_dim)
        self.obs_count = 0

        self.indices = self._compute_indices()

        self.last_occupancy_update = 0.0
        self.cached_occupancy = np.zeros(self.config.occupancy_dim)

        self.logger.info(f"Observation space initialized: {self.total_dim}D")
        self.logger.info(f"Components: pose={self.config.pose_dim}, "
                        f"velocity={self.config.velocity_dim}, "
                        f"goal={self.config.goal_vector_dim}, "
                        f"battery={self.config.battery_dim}, "
                        f"occupancy={self.config.occupancy_dim}, "
                        f"localization={self.config.localization_error_dim}")

    def _compute_indices(self) - Dict[str, Tuple[int, int]]:

        indices = {}
        start = 0

        indices['pose'] = (start, start + self.config.pose_dim)
        start += self.config.pose_dim

        indices['velocity'] = (start, start + self.config.velocity_dim)
        start += self.config.velocity_dim

        indices['goal_vector'] = (start, start + self.config.goal_vector_dim)
        start += self.config.goal_vector_dim

        indices['battery'] = (start, start + self.config.battery_dim)
        start += self.config.battery_dim

        indices['occupancy'] = (start, start + self.config.occupancy_dim)
        start += self.config.occupancy_dim

        indices['localization_error'] = (start, start + self.config.localization_error_dim)

        return indices

    def build_observation(self, drone_state: Dict[str, Any],
                         goal_position: Tuple[float, float, float],
                         sensor_data: Dict[str, Any],
                         slam_data: Dict[str, Any]) - np.ndarray:

        current_time = time.time()
        observation = np.zeros(self.total_dim, dtype=np.float32)

        pose_obs = self._extract_pose_observation(drone_state, slam_data)
        start, end = self.indices['pose']
        observation[start:end] = pose_obs

        velocity_obs = self._extract_velocity_observation(drone_state)
        start, end = self.indices['velocity']
        observation[start:end] = velocity_obs

        goal_obs = self._extract_goal_observation(drone_state, goal_position)
        start, end = self.indices['goal_vector']
        observation[start:end] = goal_obs

        battery_obs = self._extract_battery_observation(drone_state)
        start, end = self.indices['battery']
        observation[start:end] = battery_obs

        occupancy_obs = self._extract_occupancy_observation(sensor_data, current_time)
        start, end = self.indices['occupancy']
        observation[start:end] = occupancy_obs

        localization_obs = self._extract_localization_observation(slam_data)
        start, end = self.indices['localization_error']
        observation[start:end] = localization_obs

        observation = self._normalize_observation(observation)

        return observation

    def _extract_pose_observation(self, drone_state: Dict[str, Any],
                                 slam_data: Dict[str, Any]) - np.ndarray:

        pose_obs = np.zeros(7, dtype=np.float32)

        if 'slam_pose' in slam_data and slam_data['slam_pose'] is not None:
            position = slam_data['slam_pose']['position']
            pose_obs[0:3] = [position[0], position[1], position[2]]
        else:
            position = drone_state.get('position', (0.0, 0.0, 0.0))
            pose_obs[0:3] = [position[0], position[1], position[2]]

        if 'slam_pose' in slam_data and slam_data['slam_pose'] is not None:
            orientation = slam_data['slam_pose']['orientation']
            pose_obs[3:7] = [orientation[0], orientation[1], orientation[2], orientation[3]]
        else:
            orientation = drone_state.get('orientation', (1.0, 0.0, 0.0, 0.0))
            pose_obs[3:7] = [orientation[0], orientation[1], orientation[2], orientation[3]]

        pose_obs[0:3] = np.clip(pose_obs[0:3],
                               self.config.position_bounds[0],
                               self.config.position_bounds[1])

        quat_norm = np.linalg.norm(pose_obs[3:7])
        if quat_norm  0:
            pose_obs[3:7] /= quat_norm
        else:
            pose_obs[3:7] = [1.0, 0.0, 0.0, 0.0]

        return pose_obs

    def _extract_velocity_observation(self, drone_state: Dict[str, Any]) - np.ndarray:

        velocity_obs = np.zeros(4, dtype=np.float32)

        linear_velocity = drone_state.get('linear_velocity', (0.0, 0.0, 0.0))
        velocity_obs[0:3] = [linear_velocity[0], linear_velocity[1], linear_velocity[2]]

        angular_velocity = drone_state.get('angular_velocity', (0.0, 0.0, 0.0))
        velocity_obs[3] = angular_velocity[2]

        velocity_obs = np.clip(velocity_obs,
                              self.config.velocity_bounds[0],
                              self.config.velocity_bounds[1])

        return velocity_obs

    def _extract_goal_observation(self, drone_state: Dict[str, Any],
                                 goal_position: Tuple[float, float, float]) - np.ndarray:

        goal_obs = np.zeros(3, dtype=np.float32)

        current_pos = drone_state.get('position', (0.0, 0.0, 0.0))

        goal_obs[0] = goal_position[0] - current_pos[0]
        goal_obs[1] = goal_position[1] - current_pos[1]
        goal_obs[2] = goal_position[2] - current_pos[2]

        goal_distance = np.linalg.norm(goal_obs)
        if goal_distance  self.config.goal_distance_bounds[1]:
            goal_obs = goal_obs / goal_distance  self.config.goal_distance_bounds[1]

        return goal_obs

    def _extract_battery_observation(self, drone_state: Dict[str, Any]) - np.ndarray:

        battery_obs = np.zeros(1, dtype=np.float32)

        battery_level = drone_state.get('battery_level', 1.0)
        battery_obs[0] = float(np.clip(battery_level, 0.0, 1.0))

        return battery_obs

    def _extract_occupancy_observation(self, sensor_data: Dict[str, Any],
                                     current_time: float) - np.ndarray:

        update_interval = 1.0 / self.config.occupancy_update_hz

        if current_time - self.last_occupancy_update = update_interval:
            if 'occupancy_histogram' in sensor_data:
                occupancy_data = sensor_data['occupancy_histogram']
                if occupancy_data is not None and len(occupancy_data) == self.config.occupancy_dim:
                    self.cached_occupancy = np.array(occupancy_data, dtype=np.float32)
                    self.cached_occupancy = np.clip(self.cached_occupancy, 0.0, 1.0)
                else:
                    self.cached_occupancy = self._generate_synthetic_occupancy(sensor_data)
            else:
                self.cached_occupancy = np.zeros(self.config.occupancy_dim, dtype=np.float32)

            self.last_occupancy_update = current_time

        return self.cached_occupancy.copy()

    def _generate_synthetic_occupancy(self, sensor_data: Dict[str, Any]) - np.ndarray:

        occupancy = np.zeros(self.config.occupancy_dim, dtype=np.float32)

        if 'depth_image' in sensor_data and sensor_data['depth_image'] is not None:
            depth_image = sensor_data['depth_image']

            height, width = depth_image.shape
            center_x, center_y = width

            for sector in range(self.config.occupancy_dim):
                angle = 2  np.pi  sector / self.config.occupancy_dim

                for radius in range(1, min(center_x, center_y), 5):
                    sample_x = int(center_x + radius  np.cos(angle))
                    sample_y = int(center_y + radius  np.sin(angle))

                    if (0 = sample_x  width and 0 = sample_y  height):
                        depth = depth_image[sample_y, sample_x]

                        if 0  depth = self.config.occupancy_range:
                            occupancy[sector] = max(occupancy[sector],
                                                  1.0 - depth / self.config.occupancy_range)

        return occupancy

    def _extract_localization_observation(self, slam_data: Dict[str, Any]) - np.ndarray:

        localization_obs = np.zeros(1, dtype=np.float32)

        ate = slam_data.get('ate', 0.0)

        normalized_ate = min(ate / self.config.max_localization_error, 1.0)
        localization_obs[0] = float(normalized_ate)

        return localization_obs

    def _normalize_observation(self, observation: np.ndarray) - np.ndarray:

        if self.obs_count == 0:
            self.obs_mean = observation.copy()
            self.obs_var = np.ones_like(observation)
        else:
            delta = observation - self.obs_mean
            self.obs_mean += delta / (self.obs_count + 1)
            self.obs_var = (self.obs_count  self.obs_var + delta  (observation - self.obs_mean)) / (self.obs_count + 1)

        self.obs_count += 1

        epsilon = 1e-8
        normalized_obs = (observation - self.obs_mean) / (np.sqrt(self.obs_var) + epsilon)

        normalized_obs = np.clip(normalized_obs, -5.0, 5.0)

        return normalized_obs

    def get_observation_info(self) - Dict[str, Any]:

        return {
            'total_dimension': self.total_dim,
            'components': {
                'pose': {
                    'indices': self.indices['pose'],
                    'dimension': self.config.pose_dim,
                    'description': '3D position (x, y, z) from VI-SLAM'
                },
                'velocity': {
                    'indices': self.indices['velocity'],
                    'dimension': self.config.velocity_dim,
                    'description': 'Body-frame velocities [vx,vy,vz] + yaw rate ω'
                },
                'goal_vector': {
                    'indices': self.indices['goal_vector'],
                    'dimension': self.config.goal_vector_dim,
                    'description': '3D vector from current position to goal'
                },
                'battery': {
                    'indices': self.indices['battery'],
                    'dimension': self.config.battery_dim,
                    'description': 'Remaining battery fraction [0,1]'
                },
                'occupancy': {
                    'indices': self.indices['occupancy'],
                    'dimension': self.config.occupancy_dim,
                    'description': '24-sector occupancy histogram (updated at 20Hz)'
                },
                'localization_error': {
                    'indices': self.indices['localization_error'],
                    'dimension': self.config.localization_error_dim,
                    'description': 'ATE estimate from SLAM system'
                }
            },
            'update_frequencies': {
                'pose': f"{self.config.pose_update_hz} Hz",
                'velocity': f"{self.config.velocity_update_hz} Hz",
                'occupancy': f"{self.config.occupancy_update_hz} Hz"
            },
            'normalization': {
                'method': 'running_mean_std',
                'clip_range': [-5.0, 5.0]
            }
        }

    def reset_normalization(self):

        self.obs_mean = np.zeros(self.total_dim)
        self.obs_var = np.ones(self.total_dim)
        self.obs_count = 0
        self.logger.info("Observation normalization reset")

    def get_component(self, observation: np.ndarray, component: str) - np.ndarray:

        if component not in self.indices:
            raise ValueError(f"Unknown component: {component}")

        start, end = self.indices[component]
        return observation[start:end]

    def validate_observation(self, observation: np.ndarray) - bool:

        if len(observation) != self.total_dim:
            self.logger.error(f"Invalid observation dimension: {len(observation)} != {self.total_dim}")
            return False

        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            self.logger.error("Observation contains NaN or infinite values")
            return False

        pose_component = self.get_component(observation, 'pose')
        quat = pose_component[3:7]
        quat_norm = np.linalg.norm(quat)
        if not (0.9 = quat_norm = 1.1):
            self.logger.warning(f"Quaternion not normalized: norm={quat_norm}")

        return True
