"""
Sensor Bridge
Manages sensor data integration and processing for 35-dimensional observation space.
Implements exact sensor specifications from Table 1 in report.
"""

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
    import airsim  # type: ignore
except ImportError:  # pragma: no cover
    airsim = None

if TYPE_CHECKING:  # pragma: no cover
    from airsim import MultirotorClient  # type: ignore


@dataclass
class SensorReading:
    """Complete sensor reading bundle."""

    timestamp: float

    # Visual sensors (stereo cameras)
    stereo_left: Optional[np.ndarray] = None
    stereo_right: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None

    # Inertial sensors
    imu_data: Optional[Dict[str, Any]] = None

    # Processed features
    occupancy_histogram: Optional[np.ndarray] = None  # 24 sectors
    depth_features: Optional[Dict[str, float]] = None


@dataclass
class OccupancyGrid:
    """3D occupancy grid representation."""

    grid: np.ndarray
    resolution: float  # meters per cell
    origin: Tuple[float, float, float]
    dimensions: Tuple[int, int, int]  # cells in x, y, z


class SensorBridge:
    """
    Sensor data integration bridge.
    Processes raw sensor data into features for 35D observation space.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the bridge and connect to AirSim if available."""
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

        # Sensor configuration (exact match with Table 1)
        self.occupancy_sectors = self.config.get("occupancy_sectors", 24)
        self.depth_range_max = self.config.get("depth_range_max", 100.0)
        self.occupancy_range = self.config.get("occupancy_range", 5.0)

        # Camera parameters
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

        # IMU parameters
        self.imu_frequency = self.config.get("imu_frequency", 200.0)

        # Occupancy grid parameters (5-floor building)
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

        # Initialize occupancy grid
        # Initialize occupancy grid
        self.occupancy_grid = self._create_occupancy_grid()

        # Data storage
        self.latest_reading: Optional[SensorReading] = None
        self.sensor_history: deque = deque(maxlen=100)

        # Processing threads
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False

        # Thread synchronization
        self.data_lock = threading.Lock()

        # Callbacks
        self.sensor_callbacks: List[Callable[[SensorReading], None]] = []
        self.occupancy_callbacks: List[Callable[[np.ndarray], None]] = []

        # Feature extraction parameters
        self._setup_feature_extractors()

        # AirSim client
        self.client: Optional["MultirotorClient"] = None
        self._initialize_airsim_client()

        self.logger.info("Sensor Bridge initialized")

    def _initialize_airsim_client(self):
        """Attempt to create an AirSim client for live sensor queries."""
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
        except Exception as exc:  # pragma: no cover - requires AirSim runtime
            self.client = None
            self.logger.error(f"Failed to initialize AirSim client: {exc}")

    def _ensure_client(self) -> bool:
        """Return True if an AirSim client is ready."""
        if self.client is not None:
            return True
        self._initialize_airsim_client()
        return self.client is not None

    def _create_occupancy_grid(self) -> OccupancyGrid:
        """Create 3D occupancy grid for building."""
        # Grid dimensions in cells
        x_cells = int(self.building_dims["length"] / self.grid_resolution)
        y_cells = int(self.building_dims["width"] / self.grid_resolution)
        z_cells = int(self.building_dims["height"] / self.grid_resolution)

        # Initialize empty grid (0 = free, 1 = occupied, -1 = unknown)
        grid = np.full((x_cells, y_cells, z_cells), -1, dtype=np.int8)

        occupancy_grid = OccupancyGrid(
            grid=grid,
            resolution=self.grid_resolution,
            origin=(0.0, 0.0, 0.0),
            dimensions=(x_cells, y_cells, z_cells),
        )

        self.logger.info(f"Occupancy grid created: {x_cells}×{y_cells}×{z_cells} cells")
        return occupancy_grid

    def _setup_feature_extractors(self):
        """Setup feature extraction algorithms."""
        # ORB feature detector for visual features
        self.orb_detector = cv2.ORB_create(nfeatures=500)

        # Depth image processor
        self.depth_processor = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        # Occupancy histogram sectors (24-sector division as per Table 1)
        self.sector_angles = np.linspace(
            0, 2 * np.pi, self.occupancy_sectors, endpoint=False
        )

        self.logger.info("Feature extractors initialized")

    def _decode_scene_image(self, response: Any) -> Optional[np.ndarray]:
        """Convert an AirSim scene image response into a numpy array."""
        if response is None or getattr(response, "width", 0) == 0:
            return None
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        if img1d.size == 0:
            return None
        try:
            return img1d.reshape(response.height, response.width, 3)
        except ValueError:
            return None

    def _decode_depth_image(self, response: Any) -> Optional[np.ndarray]:
        """Convert an AirSim depth response into a 2D float array."""
        if response is None or getattr(response, "width", 0) == 0:
            return None
        try:
            depth = airsim.list_to_2d_float_array(
                response.image_data_float, response.width, response.height
            )
            return depth.astype(np.float32)
        except Exception:
            return None

    def _fetch_camera_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """Retrieve stereo + depth frames from AirSim."""
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
        except Exception as exc:  # pragma: no cover - requires AirSim runtime
            self.logger.error(f"Failed to fetch camera data: {exc}")
            return {"left": None, "right": None, "depth": None}

        left_resp = responses[0] if len(responses) > 0 else None
        right_resp = responses[1] if len(responses) > 1 else None
        depth_resp = responses[2] if len(responses) > 2 else None

        left = self._decode_scene_image(left_resp) if left_resp is not None else None
        right = self._decode_scene_image(right_resp) if right_resp is not None else None
        depth = self._decode_depth_image(depth_resp) if depth_resp is not None else None

        return {
            "left": left if left is not None else left_resp,
            "right": right if right is not None else right_resp,
            "depth": depth if depth is not None else depth_resp,
        }

    def _fetch_imu_sample(self) -> Optional[Any]:
        """Fetch raw IMU data from AirSim."""
        if not self._ensure_client():
            return None
        try:
            imu = self.client.getImuData(imu_name="Imu", vehicle_name=self.vehicle_name)
            return imu
        except Exception as exc:  # pragma: no cover
            self.logger.error(f"Failed to fetch IMU data: {exc}")
            return None

    @staticmethod
    def _serialize_imu_data(imu: Any) -> Dict[str, Any]:
        """Convert an AirSim IMU sample to a lightweight dictionary."""
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

    def _fetch_gps_sample(self) -> Optional[Any]:
        if not self._ensure_client():
            return None
        try:
            return self.client.getGpsData(vehicle_name=self.vehicle_name)
        except Exception as exc:  # pragma: no cover
            self.logger.error(f"Failed to fetch GPS data: {exc}")
            return None

    def _fetch_barometer_sample(self) -> Optional[Any]:
        if not self._ensure_client():
            return None
        try:
            return self.client.getBarometerData(vehicle_name=self.vehicle_name)
        except Exception as exc:  # pragma: no cover
            self.logger.error(f"Failed to fetch barometer data: {exc}")
            return None

    def _fetch_magnetometer_sample(self) -> Optional[Any]:
        if not self._ensure_client():
            return None
        try:
            return self.client.getMagnetometerData(vehicle_name=self.vehicle_name)
        except Exception as exc:  # pragma: no cover
            self.logger.error(f"Failed to fetch magnetometer data: {exc}")
            return None

    def _fetch_distance_sample(self) -> Optional[Any]:
        if not self._ensure_client():
            return None
        try:
            return self.client.getDistanceSensorData(
                distance_sensor_name=self.distance_sensor_name,
                vehicle_name=self.vehicle_name,
            )
        except Exception as exc:  # pragma: no cover
            self.logger.error(f"Failed to fetch distance sensor data: {exc}")
            return None

    def _fetch_vehicle_state(self) -> Optional[Any]:
        if not self._ensure_client():
            return None
        try:
            return self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        except Exception as exc:  # pragma: no cover
            self.logger.error(f"Failed to fetch vehicle state: {exc}")
            return None

    def start_processing(self):
        """Start sensor data processing thread."""
        if self.is_processing:
            return

        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.logger.info("Sensor processing started")

    def stop_processing(self):
        """Stop sensor data processing thread."""
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
    ) -> SensorReading:
        """
        Process complete sensor data bundle.

        Args:
            stereo_left: Left stereo image
            stereo_right: Right stereo image
            depth_image: Depth image
            imu_data: IMU measurements
            timestamp: Data timestamp

        Returns:
            Processed sensor reading
        """
        # Create sensor reading
        reading = SensorReading(timestamp=timestamp)
        reading.stereo_left = stereo_left.copy() if stereo_left is not None else None
        reading.stereo_right = stereo_right.copy() if stereo_right is not None else None
        reading.depth_image = depth_image.copy() if depth_image is not None else None
        reading.imu_data = imu_data.copy() if imu_data else None

        # Process depth image for occupancy
        if depth_image is not None:
            reading.occupancy_histogram = self._compute_occupancy_histogram(depth_image)
            reading.depth_features = self._extract_depth_features(depth_image)

        # Store reading
        with self.data_lock:
            self.latest_reading = reading
            self.sensor_history.append(reading)

        # Notify callbacks
        for callback in self.sensor_callbacks:
            try:
                callback(reading)
            except Exception as e:
                self.logger.error(f"Sensor callback error: {e}")

        return reading

    def capture_sensor_snapshot(self) -> Optional[SensorReading]:
        """
        Pull fresh sensor data from AirSim, process it, and cache the reading.
        """
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

    def _compute_occupancy_histogram(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Compute 24-sector occupancy histogram from depth image.
        Implements exact Table 1 specification: 24-dimensional occupancy vector.

        Args:
            depth_image: Depth image array

        Returns:
            24-dimensional occupancy histogram
        """
        if depth_image is None or depth_image.size == 0:
            return np.zeros(self.occupancy_sectors)

        height, width = depth_image.shape
        center_x, center_y = width // 2, height // 2

        # Initialize histogram
        occupancy_hist = np.zeros(self.occupancy_sectors)
        sector_counts = np.zeros(self.occupancy_sectors)

        # Process each pixel
        for y in range(height):
            for x in range(width):
                depth = depth_image[y, x]

                # Skip invalid depths
                if np.isnan(depth) or np.isinf(depth) or depth <= 0:
                    continue

                # Skip depths beyond occupancy range
                if depth > self.occupancy_range:
                    continue

                # Calculate angle from center
                dx = x - center_x
                dy = y - center_y
                angle = math.atan2(dy, dx)

                # Normalize angle to [0, 2π]
                if angle < 0:
                    angle += 2 * math.pi

                # Determine sector
                sector_idx = int(angle / (2 * math.pi) * self.occupancy_sectors)
                sector_idx = min(sector_idx, self.occupancy_sectors - 1)

                # Accumulate inverse depth (closer objects have higher influence)
                occupancy_value = 1.0 / (depth + 0.1)  # Avoid division by zero
                occupancy_hist[sector_idx] += occupancy_value
                sector_counts[sector_idx] += 1

        # Normalize by number of pixels per sector
        for i in range(self.occupancy_sectors):
            if sector_counts[i] > 0:
                occupancy_hist[i] /= sector_counts[i]

        # Normalize to [0, 1] range
        max_val = np.max(occupancy_hist)
        if max_val > 0:
            occupancy_hist = occupancy_hist / max_val

        return occupancy_hist

    def _extract_depth_features(self, depth_image: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from depth image.

        Args:
            depth_image: Depth image array

        Returns:
            Dictionary of depth features
        """
        if depth_image is None or depth_image.size == 0:
            return {
                "mean_depth": 0.0,
                "min_depth": 0.0,
                "std_depth": 0.0,
                "obstacle_density": 0.0,
            }

        # Filter valid depths
        valid_depths = depth_image[
            (depth_image > 0)
            & (~np.isnan(depth_image))
            & (~np.isinf(depth_image))
            & (depth_image < self.depth_range_max)
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
                len(valid_depths[valid_depths < 2.0]) / len(valid_depths)
            ),
        }

        return features

    def update_occupancy_grid(
        self,
        position: Tuple[float, float, float],
        depth_image: np.ndarray,
        camera_orientation: np.ndarray,
    ):
        """
        Update global occupancy grid from depth observation.

        Args:
            position: Current drone position
            depth_image: Depth image
            camera_orientation: Camera orientation matrix
        """
        if depth_image is None:
            return

        try:
            # Project depth image to world coordinates
            world_points = self._project_depth_to_world(
                depth_image, position, camera_orientation
            )

            # Update occupancy grid
            for point in world_points:
                self._mark_occupancy_cell(point, occupied=True)

            # Notify callbacks
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
    ) -> List[Tuple[float, float, float]]:
        """
        Project depth image pixels to world coordinates.

        Args:
            depth_image: Depth image
            position: Camera position in world
            orientation: Camera orientation matrix

        Returns:
            List of world coordinate points
        """
        world_points = []
        height, width = depth_image.shape

        # Camera intrinsics (simplified)
        fx = fy = 460.0  # From camera calibration
        cx, cy = width // 2, height // 2

        # Sample pixels (not every pixel for performance)
        step = 10
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_image[y, x]

                if depth <= 0 or np.isnan(depth) or depth > 10.0:
                    continue

                # Convert to camera coordinates
                cam_x = (x - cx) * depth / fx
                cam_y = (y - cy) * depth / fy
                cam_z = depth

                # Transform to world coordinates
                cam_point = np.array([cam_x, cam_y, cam_z])
                world_point = orientation @ cam_point + np.array(position)

                world_points.append(tuple(world_point))

        return world_points

    def _mark_occupancy_cell(
        self, world_point: Tuple[float, float, float], occupied: bool = True
    ):
        """
        Mark cell in occupancy grid as occupied or free.

        Args:
            world_point: Point in world coordinates
            occupied: True if occupied, False if free
        """
        # Convert world coordinates to grid coordinates
        x_idx = int(
            (world_point[0] - self.occupancy_grid.origin[0]) / self.grid_resolution
        )
        y_idx = int(
            (world_point[1] - self.occupancy_grid.origin[1]) / self.grid_resolution
        )
        z_idx = int(
            (world_point[2] - self.occupancy_grid.origin[2]) / self.grid_resolution
        )

        # Check bounds
        if (
            0 <= x_idx < self.occupancy_grid.dimensions[0]
            and 0 <= y_idx < self.occupancy_grid.dimensions[1]
            and 0 <= z_idx < self.occupancy_grid.dimensions[2]
        ):

            self.occupancy_grid.grid[x_idx, y_idx, z_idx] = 1 if occupied else 0

    def get_occupancy_histogram(self) -> Optional[np.ndarray]:
        """
        Get latest 24-sector occupancy histogram.

        Returns:
            24-dimensional occupancy vector or None
        """
        with self.data_lock:
            if (
                self.latest_reading
                and self.latest_reading.occupancy_histogram is not None
            ):
                return self.latest_reading.occupancy_histogram.copy()
        return None

    def get_depth_features(self) -> Optional[Dict[str, float]]:
        """
        Get latest depth features.

        Returns:
            Depth features dictionary or None
        """
        with self.data_lock:
            if self.latest_reading and self.latest_reading.depth_features is not None:
                return self.latest_reading.depth_features.copy()
        return None

    def get_occupancy_grid(self) -> OccupancyGrid:
        """
        Get current occupancy grid.

        Returns:
            Current occupancy grid
        """
        return self.occupancy_grid

    def add_sensor_callback(self, callback: Callable[[SensorReading], None]):
        """Add callback for sensor data updates."""
        self.sensor_callbacks.append(callback)

    def add_occupancy_callback(self, callback: Callable[[np.ndarray], None]):
        """Add callback for occupancy grid updates."""
        self.occupancy_callbacks.append(callback)

    def reset(self):
        """Reset sensor data and occupancy grid."""
        with self.data_lock:
            self.latest_reading = None
            self.sensor_history.clear()

        # Reset occupancy grid
        self.occupancy_grid.grid.fill(-1)  # Unknown

        self.logger.info("Sensor bridge reset")

    def _processing_loop(self):
        """Main sensor processing loop."""
        self.logger.info("Sensor processing loop started")

        while self.is_processing:
            try:
                # Perform periodic processing tasks
                self._update_occupancy_grid_decay()

                time.sleep(0.1)  # 10Hz processing

            except Exception as e:
                self.logger.error(f"Sensor processing error: {e}")
                time.sleep(0.1)

        self.logger.info("Sensor processing loop stopped")

    def _update_occupancy_grid_decay(self):
        """Apply temporal decay to occupancy grid for dynamic environments."""
        # Implement occupancy decay for dynamic obstacles
        # This helps handle moving obstacles that might leave "ghost" occupancy

        # Simple implementation: slightly reduce occupancy values over time
        decay_factor = 0.999  # Very slow decay

        occupied_cells = self.occupancy_grid.grid == 1
        if np.any(occupied_cells):
            # Apply probabilistic decay (simplified)
            pass  # Real implementation would apply proper Bayesian updates

    def get_imu_data(self) -> Optional[Dict[str, Any]]:
        """
        Get latest IMU data.

        Returns:
            IMU data dictionary or None
        """
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

    def get_camera_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get latest camera data (stereo images + depth).

        Returns:
            Dictionary with 'left', 'right', 'depth' images or None
        """
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

    def get_gps_data(self) -> Optional[Dict[str, float]]:
        """
        Get GPS data (simulated from ground truth).

        Returns:
            Dictionary with lat, lon, alt or None
        """
        live = self._fetch_gps_sample()
        if live is not None:
            return live
        return None

    def get_barometer_data(self) -> Optional[Dict[str, float]]:
        """
        Get barometer data (altitude + pressure).

        Returns:
            Dictionary with altitude and pressure or None
        """
        live = self._fetch_barometer_sample()
        if live is not None:
            return live
        return None

    def get_magnetometer_data(self) -> Optional[Dict[str, float]]:
        """
        Get magnetometer data (magnetic field).

        Returns:
            Dictionary with magnetic field components or None
        """
        live = self._fetch_magnetometer_sample()
        if live is not None:
            return live
        return None

    def get_distance_data(self) -> Optional[Dict[str, float]]:
        """
        Get distance sensor data (ultrasonic/lidar).

        Returns:
            Dictionary with distance measurements or None
        """
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

    def get_fused_sensor_data(self) -> Optional[SensorReading]:
        """
        Capture and return a fresh fused sensor reading from AirSim.
        """
        return self.capture_sensor_snapshot()

    def get_position(self) -> Optional[Any]:
        """Return current multirotor position from AirSim."""
        state = self._fetch_vehicle_state()
        if state:
            return state.kinematics_estimated.position
        return None

    def get_orientation(self) -> Optional[Any]:
        """Return current multirotor orientation from AirSim."""
        state = self._fetch_vehicle_state()
        if state:
            return state.kinematics_estimated.orientation
        return None

    def __enter__(self):
        """Context manager entry."""
        self.start_processing()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_processing()
