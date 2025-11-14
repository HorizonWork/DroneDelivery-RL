import numpy as np
import cv2
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass
from collections import deque
import math

try:
    import airsim
except ImportError:
    airsim = None

if TYPE_CHECKING:
    from airsim import MultirotorClient

dataclass
class SensorReading:

    timestamp: float

    stereo_left: Optional[np.ndarray] = None
    stereo_right: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None

    imu_data: Optional[Dict[str, Any]] = None

    occupancy_histogram: Optional[np.ndarray] = None
    depth_features: Optional[Dict[str, float]] = None

dataclass
class OccupancyGrid:

    grid: np.ndarray
    resolution: float
    origin: Tuple[float, float, float]
    dimensions: Tuple[int, int, int]

class SensorBridge:

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        if config is None:
            config = {
                "occupancy_sectors": 24,
                "depth_range_max": 100.0,
                "occupancy_range": 5.0,
                "camera_frequency": 30.0,
                "camera_resolution": (640, 480),
                "imu_frequency": 200.0,
                "grid_resolution": 0.5,
                "building_dims": {
                    "length": 20.0,
                    "width": 40.0,
                    "height": 15.0,
                },
                "vehicle_name": "Drone1",
                "camera_names": {
                    "stereo_left": "front_left",
                    "stereo_right": "front_right",
                    "depth": "front_center",
                },
                "distance_sensor_name": "Distance",
            }
        self.config = dict(config)
        self.logger = logging.getLogger(__name__)

        self.occupancy_sectors = self.config.get("occupancy_sectors", 24)
        self.depth_range_max = self.config.get("depth_range_max", 100.0)
        self.occupancy_range = self.config.get("occupancy_range", 5.0)

        self.camera_frequency = self.config.get("camera_frequency", 30.0)
        self.camera_resolution = self.config.get("camera_resolution", [640, 480])
        self.camera_names = self.config.get(
            "camera_names",
            {
                "stereo_left": "front_left",
                "stereo_right": "front_right",
                "depth": "front_center",
            },
        )

        self.imu_frequency = self.config.get("imu_frequency", 200.0)

        self.grid_resolution = self.config.get("grid_resolution", 0.5)
        self.building_dims = self.config.get(
            "building_dims",
            {
                "length": 20.0,
                "width": 40.0,
                "height": 15.0,
            },
        )
        self.vehicle_name = self.config.get("vehicle_name", "Drone1")
        self.distance_sensor_name = self.config.get("distance_sensor_name", "Distance")

        self.occupancy_grid = self._create_occupancy_grid()

        self.latest_reading: Optional[SensorReading] = None
        self.sensor_history: deque = deque(maxlen=100)

        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False

        self.data_lock = threading.Lock()

        self.sensor_callbacks: List[Callable[[SensorReading], None]] = []
        self.occupancy_callbacks: List[Callable[[np.ndarray], None]] = []

        self._setup_feature_extractors()

        self.client: Optional["MultirotorClient"] = None
        self._initialize_airsim_client()

        self.logger.info("Sensor Bridge initialized")

    def _initialize_airsim_client(self):

        if airsim is None:
            self.logger.warning(
                "AirSim Python package not available. Live sensor capture disabled."
            )
            self.client = None
            return

        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            self.client = client
            self.logger.info("AirSim client ready for sensor bridge")
        except Exception as exc:
            self.client = None
            self.logger.error(f"Failed to initialize AirSim client: {exc}")

    def _ensure_client(self) - bool:

        if self.client is not None:
            return True
        self._initialize_airsim_client()
        return self.client is not None

    def _create_occupancy_grid(self) - OccupancyGrid:

        x_cells = int(self.building_dims["length"] / self.grid_resolution)
        y_cells = int(self.building_dims["width"] / self.grid_resolution)
        z_cells = int(self.building_dims["height"] / self.grid_resolution)

        grid = np.full((x_cells, y_cells, z_cells), -1, dtype=np.int8)

        occupancy_grid = OccupancyGrid(
            grid=grid,
            resolution=self.grid_resolution,
            origin=(0.0, 0.0, 0.0),
            dimensions=(x_cells, y_cells, z_cells),
        )

        self.logger.info(f"Occupancy grid created: {x_cells}{y_cells}{z_cells} cells")
        return occupancy_grid

    def _setup_feature_extractors(self):

        self.orb_detector = cv2.ORB_create(nfeatures=500)

        self.depth_processor = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        self.sector_angles = np.linspace(
            0, 2  np.pi, self.occupancy_sectors, endpoint=False
        )

        self.logger.info("Feature extractors initialized")

    def _decode_scene_image(self, response: Any) - Optional[np.ndarray]:

        if response is None or getattr(response, "width", 0) == 0:
            return None
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        if img1d.size == 0:
            return None
        try:
            return img1d.reshape(response.height, response.width, 3)
        except ValueError:
            return None

    def _decode_depth_image(self, response: Any) - Optional[np.ndarray]:

        if response is None or getattr(response, "width", 0) == 0:
            return None
        try:
            depth = airsim.list_to_2d_float_array(
                response.image_data_float, response.width, response.height
            )
            return depth.astype(np.float32)
        except Exception:
            return None

    def _fetch_camera_frames(self) - Dict[str, Optional[np.ndarray]]:

        if not self._ensure_client():
            return {"left": None, "right": None, "depth": None}

        try:
            requests = [
                airsim.ImageRequest(
                    self.camera_names.get("stereo_left", "front_left"),
                    airsim.ImageType.Scene,
                    False,
                    False,
                ),
                airsim.ImageRequest(
                    self.camera_names.get("stereo_right", "front_right"),
                    airsim.ImageType.Scene,
                    False,
                    False,
                ),
                airsim.ImageRequest(
                    self.camera_names.get("depth", "front_center"),
                    airsim.ImageType.DepthPlanar,
                    True,
                ),
            ]
            responses = self.client.simGetImages(
                requests, vehicle_name=self.vehicle_name
            )
        except Exception as exc:
            self.logger.error(f"Failed to fetch camera data: {exc}")
            return {"left": None, "right": None, "depth": None}

        left_resp = responses[0] if len(responses)  0 else None
        right_resp = responses[1] if len(responses)  1 else None
        depth_resp = responses[2] if len(responses)  2 else None

        left = self._decode_scene_image(left_resp) if left_resp is not None else None
        right = self._decode_scene_image(right_resp) if right_resp is not None else None
        depth = self._decode_depth_image(depth_resp) if depth_resp is not None else None

        return {
            "left": left if left is not None else left_resp,
            "right": right if right is not None else right_resp,
            "depth": depth if depth is not None else depth_resp,
        }

    def _fetch_imu_sample(self) - Optional[Any]:

        if not self._ensure_client():
            return None
        try:
            imu = self.client.getImuData(imu_name="Imu", vehicle_name=self.vehicle_name)
            return imu
        except Exception as exc:
            self.logger.error(f"Failed to fetch IMU data: {exc}")
            return None

    staticmethod
    def _serialize_imu_data(imu: Any) - Dict[str, Any]:

        return {
            "orientation": {
                "w": imu.orientation.w_val,
                "x": imu.orientation.x_val,
                "y": imu.orientation.y_val,
                "z": imu.orientation.z_val,
            },
            "angular_velocity": {
                "x": imu.angular_velocity.x_val,
                "y": imu.angular_velocity.y_val,
                "z": imu.angular_velocity.z_val,
            },
            "linear_acceleration": {
                "x": imu.linear_acceleration.x_val,
                "y": imu.linear_acceleration.y_val,
                "z": imu.linear_acceleration.z_val,
            },
        }

    def _fetch_gps_sample(self) - Optional[Any]:
        if not self._ensure_client():
            return None
        try:
            return self.client.getGpsData(vehicle_name=self.vehicle_name)
        except Exception as exc:
            self.logger.error(f"Failed to fetch GPS data: {exc}")
            return None

    def _fetch_barometer_sample(self) - Optional[Any]:
        if not self._ensure_client():
            return None
        try:
            return self.client.getBarometerData(vehicle_name=self.vehicle_name)
        except Exception as exc:
            self.logger.error(f"Failed to fetch barometer data: {exc}")
            return None

    def _fetch_magnetometer_sample(self) - Optional[Any]:
        if not self._ensure_client():
            return None
        try:
            return self.client.getMagnetometerData(vehicle_name=self.vehicle_name)
        except Exception as exc:
            self.logger.error(f"Failed to fetch magnetometer data: {exc}")
            return None

    def _fetch_distance_sample(self) - Optional[Any]:
        if not self._ensure_client():
            return None
        try:
            return self.client.getDistanceSensorData(
                distance_sensor_name=self.distance_sensor_name,
                vehicle_name=self.vehicle_name,
            )
        except Exception as exc:
            self.logger.error(f"Failed to fetch distance sensor data: {exc}")
            return None

    def _fetch_vehicle_state(self) - Optional[Any]:
        if not self._ensure_client():
            return None
        try:
            return self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        except Exception as exc:
            self.logger.error(f"Failed to fetch vehicle state: {exc}")
            return None

    def start_processing(self):

        if self.is_processing:
            return

        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.logger.info("Sensor processing started")

    def stop_processing(self):

        self.is_processing = False

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        self.logger.info("Sensor processing stopped")

    def process_sensor_data(
        self,
        stereo_left: np.ndarray,
        stereo_right: np.ndarray,
        depth_image: np.ndarray,
        imu_data: Dict[str, Any],
        timestamp: float,
    ) - SensorReading:

        reading = SensorReading(timestamp=timestamp)
        reading.stereo_left = stereo_left.copy() if stereo_left is not None else None
        reading.stereo_right = stereo_right.copy() if stereo_right is not None else None
        reading.depth_image = depth_image.copy() if depth_image is not None else None
        reading.imu_data = imu_data.copy() if imu_data else None

        if depth_image is not None:
            reading.occupancy_histogram = self._compute_occupancy_histogram(depth_image)
            reading.depth_features = self._extract_depth_features(depth_image)

        with self.data_lock:
            self.latest_reading = reading
            self.sensor_history.append(reading)

        for callback in self.sensor_callbacks:
            try:
                callback(reading)
            except Exception as e:
                self.logger.error(f"Sensor callback error: {e}")

        return reading

    def capture_sensor_snapshot(self) - Optional[SensorReading]:

        if airsim is None:
            self.logger.warning("AirSim SDK not available; cannot capture snapshot.")
            return None

        frames = self._fetch_camera_frames()
        imu_sample = self._fetch_imu_sample()
        imu_data = self._serialize_imu_data(imu_sample) if imu_sample else {}

        if (
            frames["depth"] is None
            and frames["left"] is None
            and frames["right"] is None
        ):
            self.logger.warning("No camera data available from AirSim.")
            return None

        return self.process_sensor_data(
            stereo_left=frames["left"],
            stereo_right=frames["right"],
            depth_image=frames["depth"],
            imu_data=imu_data,
            timestamp=time.time(),
        )

    def _compute_occupancy_histogram(self, depth_image: np.ndarray) - np.ndarray:

        if depth_image is None or depth_image.size == 0:
            return np.zeros(self.occupancy_sectors)

        height, width = depth_image.shape
        center_x, center_y = width

        occupancy_hist = np.zeros(self.occupancy_sectors)
        sector_counts = np.zeros(self.occupancy_sectors)

        for y in range(height):
            for x in range(width):
                depth = depth_image[y, x]

                if np.isnan(depth) or np.isinf(depth) or depth = 0:
                    continue

                if depth  self.occupancy_range:
                    continue

                dx = x - center_x
                dy = y - center_y
                angle = math.atan2(dy, dx)

                if angle  0:
                    angle += 2  math.pi

                sector_idx = int(angle / (2  math.pi)  self.occupancy_sectors)
                sector_idx = min(sector_idx, self.occupancy_sectors - 1)

                occupancy_value = 1.0 / (depth + 0.1)
                occupancy_hist[sector_idx] += occupancy_value
                sector_counts[sector_idx] += 1

        for i in range(self.occupancy_sectors):
            if sector_counts[i]  0:
                occupancy_hist[i] /= sector_counts[i]

        max_val = np.max(occupancy_hist)
        if max_val  0:
            occupancy_hist = occupancy_hist / max_val

        return occupancy_hist

    def _extract_depth_features(self, depth_image: np.ndarray) - Dict[str, float]:

        if depth_image is None or depth_image.size == 0:
            return {
                "mean_depth": 0.0,
                "min_depth": 0.0,
                "std_depth": 0.0,
                "obstacle_density": 0.0,
            }

        valid_depths = depth_image[
            (depth_image  0)
             (np.isnan(depth_image))
             (np.isinf(depth_image))
             (depth_image  self.depth_range_max)
        ]

        if len(valid_depths) == 0:
            return {
                "mean_depth": 0.0,
                "min_depth": 0.0,
                "std_depth": 0.0,
                "obstacle_density": 0.0,
            }

        features = {
            "mean_depth": float(np.mean(valid_depths)),
            "min_depth": float(np.min(valid_depths)),
            "std_depth": float(np.std(valid_depths)),
            "obstacle_density": float(
                len(valid_depths[valid_depths  2.0]) / len(valid_depths)
            ),
        }

        return features

    def update_occupancy_grid(
        self,
        position: Tuple[float, float, float],
        depth_image: np.ndarray,
        camera_orientation: np.ndarray,
    ):

        if depth_image is None:
            return

        try:
            world_points = self._project_depth_to_world(
                depth_image, position, camera_orientation
            )

            for point in world_points:
                self._mark_occupancy_cell(point, occupied=True)

            for callback in self.occupancy_callbacks:
                try:
                    callback(self.occupancy_grid.grid)
                except Exception as e:
                    self.logger.error(f"Occupancy callback error: {e}")

        except Exception as e:
            self.logger.error(f"Occupancy grid update error: {e}")

    def _project_depth_to_world(
        self,
        depth_image: np.ndarray,
        position: Tuple[float, float, float],
        orientation: np.ndarray,
    ) - List[Tuple[float, float, float]]:

        world_points = []
        height, width = depth_image.shape

        fx = fy = 460.0
        cx, cy = width

        step = 10
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_image[y, x]

                if depth = 0 or np.isnan(depth) or depth  10.0:
                    continue

                cam_x = (x - cx)  depth / fx
                cam_y = (y - cy)  depth / fy
                cam_z = depth

                cam_point = np.array([cam_x, cam_y, cam_z])
                world_point = orientation  cam_point + np.array(position)

                world_points.append(tuple(world_point))

        return world_points

    def _mark_occupancy_cell(
        self, world_point: Tuple[float, float, float], occupied: bool = True
    ):

        x_idx = int(
            (world_point[0] - self.occupancy_grid.origin[0]) / self.grid_resolution
        )
        y_idx = int(
            (world_point[1] - self.occupancy_grid.origin[1]) / self.grid_resolution
        )
        z_idx = int(
            (world_point[2] - self.occupancy_grid.origin[2]) / self.grid_resolution
        )

        if (
            0 = x_idx  self.occupancy_grid.dimensions[0]
            and 0 = y_idx  self.occupancy_grid.dimensions[1]
            and 0 = z_idx  self.occupancy_grid.dimensions[2]
        ):

            self.occupancy_grid.grid[x_idx, y_idx, z_idx] = 1 if occupied else 0

    def get_occupancy_histogram(self) - Optional[np.ndarray]:

        with self.data_lock:
            if (
                self.latest_reading
                and self.latest_reading.occupancy_histogram is not None
            ):
                return self.latest_reading.occupancy_histogram.copy()
        return None

    def get_depth_features(self) - Optional[Dict[str, float]]:

        with self.data_lock:
            if self.latest_reading and self.latest_reading.depth_features is not None:
                return self.latest_reading.depth_features.copy()
        return None

    def get_occupancy_grid(self) - OccupancyGrid:

        return self.occupancy_grid

    def add_sensor_callback(self, callback: Callable[[SensorReading], None]):

        self.sensor_callbacks.append(callback)

    def add_occupancy_callback(self, callback: Callable[[np.ndarray], None]):

        self.occupancy_callbacks.append(callback)

    def reset(self):

        with self.data_lock:
            self.latest_reading = None
            self.sensor_history.clear()

        self.occupancy_grid.grid.fill(-1)

        self.logger.info("Sensor bridge reset")

    def _processing_loop(self):

        self.logger.info("Sensor processing loop started")

        while self.is_processing:
            try:
                self._update_occupancy_grid_decay()

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Sensor processing error: {e}")
                time.sleep(0.1)

        self.logger.info("Sensor processing loop stopped")

    def _update_occupancy_grid_decay(self):

        decay_factor = 0.999

        occupied_cells = self.occupancy_grid.grid == 1
        if np.any(occupied_cells):
            pass

    def get_imu_data(self) - Optional[Dict[str, Any]]:

        live = self._fetch_imu_sample()
        if live is not None:
            with self.data_lock:
                if self.latest_reading:
                    self.latest_reading.imu_data = self._serialize_imu_data(live)
            return live

        with self.data_lock:
            if self.latest_reading and self.latest_reading.imu_data is not None:
                return self.latest_reading.imu_data.copy()
        return None

    def get_camera_data(self) - Optional[Dict[str, np.ndarray]]:

        frames = self._fetch_camera_frames()
        if any(value is not None for value in frames.values()):
            with self.data_lock:
                if self.latest_reading:
                    self.latest_reading.stereo_left = frames["left"]
                    self.latest_reading.stereo_right = frames["right"]
                    self.latest_reading.depth_image = frames["depth"]
            return frames

        with self.data_lock:
            if self.latest_reading:
                return {
                    "left": (
                        self.latest_reading.stereo_left.copy()
                        if self.latest_reading.stereo_left is not None
                        else None
                    ),
                    "right": (
                        self.latest_reading.stereo_right.copy()
                        if self.latest_reading.stereo_right is not None
                        else None
                    ),
                    "depth": (
                        self.latest_reading.depth_image.copy()
                        if self.latest_reading.depth_image is not None
                        else None
                    ),
                }
        return None

    def get_gps_data(self) - Optional[Dict[str, float]]:

        live = self._fetch_gps_sample()
        if live is not None:
            return live
        return None

    def get_barometer_data(self) - Optional[Dict[str, float]]:

        live = self._fetch_barometer_sample()
        if live is not None:
            return live
        return None

    def get_magnetometer_data(self) - Optional[Dict[str, float]]:

        live = self._fetch_magnetometer_sample()
        if live is not None:
            return live
        return None

    def get_distance_data(self) - Optional[Dict[str, float]]:

        live = self._fetch_distance_sample()
        if live is not None:
            return live

        with self.data_lock:
            if self.latest_reading and self.latest_reading.depth_features:
                return {
                    "front": self.latest_reading.depth_features.get("mean_depth", 0.0),
                    "down": 0.0,
                    "min_distance": self.latest_reading.depth_features.get(
                        "min_depth", 0.0
                    ),
                }
        return None

    def get_fused_sensor_data(self) - Optional[SensorReading]:

        return self.capture_sensor_snapshot()

    def get_position(self) - Optional[Any]:

        state = self._fetch_vehicle_state()
        if state:
            return state.kinematics_estimated.position
        return None

    def get_orientation(self) - Optional[Any]:

        state = self._fetch_vehicle_state()
        if state:
            return state.kinematics_estimated.orientation
        return None

    def __enter__(self):

        self.start_processing()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.stop_processing()
