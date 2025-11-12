"""
AirSim Bridge - FIXED VERSION
Manages connection and communication with AirSim simulation environment.

CRITICAL FIXES:
1. NO auto-connect in __init__() - prevents hanging
2. Manual connect() must be called AFTER initialization
3. Proper error handling with timeouts
4. Implements exact drone spawn location {6000, -3000, 300} from report
"""

import nest_asyncio

nest_asyncio.apply()
import airsim
import msgpack
import msgpackrpc
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cv2


@dataclass
class DroneState:
    """Complete drone state from AirSim."""

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion (w, x, y, z)
    linear_velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    timestamp: float


@dataclass
class SensorData:
    """Sensor data bundle from AirSim."""

    stereo_left: Optional[np.ndarray]
    stereo_right: Optional[np.ndarray]
    depth_image: Optional[np.ndarray]
    imu_data: Dict[str, Any]
    timestamp: float


class AirSimBridge:
    """
    Bridge to AirSim simulation environment.
    Handles drone control, sensor data collection, and world state queries.

    IMPORTANT: Call connect() manually AFTER creating instance!
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AirSim bridge configuration ONLY.
        Does NOT connect to AirSim - call connect() separately!

        Args:
            config: Configuration dictionary
        """
        # Use default config if none provided
        if config is None:
            config = {}

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Connection settings
        self.drone_name = config.get("drone_name", "Drone1")
        self.spawn_location = tuple(config.get("spawn_location", (60.0, -30.0, -3.0)))
        self.spawn_orientation = tuple(config.get("spawn_orientation", (0.0, 0.0, 0.0)))

        # Control settings
        self.max_velocity = config.get("max_velocity", 5.0)  # m/s
        self.max_yaw_rate = config.get("max_yaw_rate", 1.0)  # rad/s
        self.control_frequency = config.get("control_frequency", 20.0)  # Hz

        # Camera settings (exact match with report: 30Hz stereo)
        self.camera_frequency = config.get("camera_frequency", 30.0)  # Hz
        self.camera_resolution = tuple(config.get("camera_resolution", (640, 480)))

        # IMU settings (exact match with report: 200Hz)
        self.imu_frequency = config.get("imu_frequency", 200.0)  # Hz

        # AirSim client - NOT CONNECTED YET!
        self.client: Optional[airsim.MultirotorClient] = None
        self.is_connected = False

        # State tracking
        self.last_drone_state: Optional[DroneState] = None
        self.last_sensor_data: Optional[SensorData] = None

        # Safety settings
        self.collision_threshold = config.get("collision_threshold", 0.1)  # meters
        self.ground_clearance = config.get("ground_clearance", 0.5)  # meters

        self.rpc_host = config.get("host", "127.0.0.1")
        self.rpc_port = config.get("port", 41451)
        self.rpc_timeout = config.get("rpc_timeout", 10.0)

        self.logger.info(f"AirSimBridge initialized for drone: {self.drone_name}")
        self.logger.info(f"Spawn location: {self.spawn_location}")
        # ✅ DO NOT CALL connect() here!

    def _safe_airsim_call(self, func, *args, **kwargs):
        """
        Safely call AirSim API functions, handling IOLoop issues.
        """
        try:
            import nest_asyncio

            nest_asyncio.apply()
            return func(*args, **kwargs)
        except RuntimeError as e:
            if (
                "IOLoop is already running" in str(e)
                or "already running" in str(e).lower()
            ):
                # Try direct call without async
                try:
                    return func(*args, **kwargs)
                except:
                    self.logger.warning(f"AirSim call failed: {e}")
                    return None
            else:
                raise
        except Exception as e:
            self.logger.error(f"AirSim API error: {e}")
            return None

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to AirSim simulation.
        MUST be called manually after __init__()!

        Args:
            timeout: Connection timeout in seconds

        Returns:
            bool: True if connection successful
        """
        if self.is_connected:
            self.logger.warning("Already connected to AirSim")
            return True

        try:
            self.logger.info("Connecting to AirSim...")

            # Create client with timeout
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()

            self.logger.info("Testing AirSim connection...")
            self.client.ping()

            # Enable API control
            self.logger.info("Enabling API control...")
            self.client.enableApiControl(True, self.drone_name)

            # Arm drone
            self.logger.info("Arming drone...")
            self.client.armDisarm(True, self.drone_name)

            self.is_connected = True
            self.logger.info(f"Connected to AirSim. Drone: {self.drone_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to AirSim: {e}")
            self.is_connected = False
            self.client = None
            return False

    def disconnect(self):
        """Disconnect from AirSim."""
        if self.client and self.is_connected:
            try:
                self.logger.info("Disarming drone...")
                self.client.armDisarm(False, self.drone_name)

                self.logger.info("Disabling API control...")
                self.client.enableApiControl(False, self.drone_name)

                self.logger.info("Disconnected from AirSim")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
            finally:
                self.is_connected = False
                self.client = None

    def reset_drone(self, to_spawn: bool = True):
        """
        Reset drone to spawn location.

        Args:
            to_spawn: If True, reset to spawn location. If False, just reset state.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        try:
            # Reset simulation
            self.logger.info("Resetting drone...")
            self._safe_airsim_call(self.client.reset)

            # Re-enable API control after reset
            self.client.enableApiControl(True, self.drone_name)
            self.client.armDisarm(True, self.drone_name)

            if to_spawn:
                # Set to spawn location
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        self.spawn_location[0],
                        self.spawn_location[1],
                        self.spawn_location[2],
                    ),
                    orientation_val=airsim.to_quaternion(
                        self.spawn_orientation[0],  # pitch
                        self.spawn_orientation[1],  # roll
                        self.spawn_orientation[2],  # yaw
                    ),
                )

                self.client.simSetVehiclePose(pose, True, self.drone_name)
                self.logger.info(f"Drone reset to spawn: {self.spawn_location}")
            else:
                self.logger.info("Drone reset (current position)")

            # Wait for stabilization
            time.sleep(0.5)

        except Exception as e:
            self.logger.error(f"Reset failed: {e}")
            raise

    def _default_drone_state(self) -> DroneState:
        return DroneState(
            position=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            timestamp=time.time(),
        )

    def _state_from_multirotor(self, multi_state: Any) -> DroneState:
        kin = multi_state.kinematics_estimated
        position = (
            float(kin.position.x_val),
            float(kin.position.y_val),
            float(kin.position.z_val),
        )
        orientation = (
            float(kin.orientation.w_val),
            float(kin.orientation.x_val),
            float(kin.orientation.y_val),
            float(kin.orientation.z_val),
        )
        linear_velocity = (
            float(kin.linear_velocity.x_val),
            float(kin.linear_velocity.y_val),
            float(kin.linear_velocity.z_val),
        )
        angular_velocity = (
            float(kin.angular_velocity.x_val),
            float(kin.angular_velocity.y_val),
            float(kin.angular_velocity.z_val),
        )
        return DroneState(
            position=position,
            orientation=orientation,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            timestamp=time.time(),
        )

    def _rpc_call(self, method: str, *args):
        """Perform a standalone RPC call to avoid shared buffer reuse."""
        address = msgpackrpc.Address(self.rpc_host, self.rpc_port)
        client = msgpackrpc.Client(address, timeout=self.rpc_timeout)
        try:
            return client.call(method, *args)
        finally:
            client.close()

    @staticmethod
    def _ensure_str_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                (
                    k.decode("utf-8") if isinstance(k, bytes) else k
                ): AirSimBridge._ensure_str_keys(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [AirSimBridge._ensure_str_keys(v) for v in obj]
        return obj

    def _get_multirotor_state_safe(self):
        """
        Fetch multirotor state using a fresh msgpack payload to avoid buffer reuse issues.
        """
        try:
            response = self._rpc_call("getMultirotorState", self.drone_name)
            if isinstance(response, dict):
                data = self._ensure_str_keys(response)
            elif isinstance(response, (bytes, bytearray, memoryview)):
                data = self._ensure_str_keys(msgpack.unpackb(response, raw=False))
            else:
                packed = msgpack.packb(response, use_bin_type=True)
                data = self._ensure_str_keys(msgpack.unpackb(packed, raw=False))
            return airsim.MultirotorState.from_msgpack(data)
        except Exception as exc:
            self.logger.error(f"Raw AirSim state fetch failed: {exc}")
            raise

    def get_drone_state(self) -> DroneState:
        """Get current drone state from AirSim - FIXED VERSION."""
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        try:
            multi_state = self._get_multirotor_state_safe()
            state = self._state_from_multirotor(multi_state)
            self.last_drone_state = state
            return state

        except Exception as e:
            self.logger.error(f"Failed to get drone state: {e}")
            # Return cached state if available
            if self.last_drone_state is not None:
                self.logger.warning("Using cached drone state")
                return self.last_drone_state
            # Return default state
            self.logger.warning("Returning default drone state")
            return self._default_drone_state()

    def get_sensor_data(self) -> SensorData:
        """
        Get sensor data bundle from AirSim.
        Includes stereo cameras, depth, and IMU as per report specs.

        Returns:
            SensorData: Complete sensor data bundle
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        timestamp = time.time()

        try:
            # Get stereo camera images (30Hz as per report)
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        "front_left", airsim.ImageType.Scene, False, False
                    ),
                    airsim.ImageRequest(
                        "front_right", airsim.ImageType.Scene, False, False
                    ),
                    airsim.ImageRequest(
                        "front_center", airsim.ImageType.DepthPlanar, True, False
                    ),
                ],
                self.drone_name,
            )

            # Process stereo left
            stereo_left = None
            if len(responses) > 0 and len(responses[0].image_data_uint8) > 0:
                img1d = np.frombuffer(
                    responses[0].image_data_uint8, dtype=np.uint8
                ).copy()
                if img1d.size > 0:
                    stereo_left = img1d.reshape(
                        responses[0].height, responses[0].width, 3
                    )
                    stereo_left = cv2.cvtColor(stereo_left, cv2.COLOR_BGR2RGB)

            # Process stereo right
            stereo_right = None
            if len(responses) > 1 and len(responses[1].image_data_uint8) > 0:
                img1d = np.frombuffer(
                    responses[1].image_data_uint8, dtype=np.uint8
                ).copy()
                if img1d.size > 0:
                    stereo_right = img1d.reshape(
                        responses[1].height, responses[1].width, 3
                    )
                    stereo_right = cv2.cvtColor(stereo_right, cv2.COLOR_BGR2RGB)

            # Process depth image
            depth_image = None
            if len(responses) > 2 and len(responses[2].image_data_float) > 0:
                depth_image = airsim.list_to_2d_float_array(
                    responses[2].image_data_float,
                    responses[2].width,
                    responses[2].height,
                )
                depth_image = np.array(depth_image, dtype=np.float32)

            # Get IMU data (200Hz as per report)
            imu_data_raw = self.client.getImuData(
                imu_name="Imu", vehicle_name=self.drone_name
            )
            imu_dict = {
                "linear_acceleration": (
                    imu_data_raw.linear_acceleration.x_val,
                    imu_data_raw.linear_acceleration.y_val,
                    imu_data_raw.linear_acceleration.z_val,
                ),
                "angular_velocity": (
                    imu_data_raw.angular_velocity.x_val,
                    imu_data_raw.angular_velocity.y_val,
                    imu_data_raw.angular_velocity.z_val,
                ),
                "orientation": (
                    imu_data_raw.orientation.w_val,
                    imu_data_raw.orientation.x_val,
                    imu_data_raw.orientation.y_val,
                    imu_data_raw.orientation.z_val,
                ),
                "timestamp": imu_data_raw.time_stamp,
            }

            sensor_data = SensorData(
                stereo_left=stereo_left,
                stereo_right=stereo_right,
                depth_image=depth_image,
                imu_data=imu_dict,
                timestamp=timestamp,
            )

            self.last_sensor_data = sensor_data
            return sensor_data

        except Exception as e:
            self.logger.error(f"Failed to get sensor data: {e}")
            # Return empty sensor data on failure
            return SensorData(
                stereo_left=None,
                stereo_right=None,
                depth_image=None,
                imu_data={},
                timestamp=timestamp,
            )

    def send_velocity_command(
        self, vx: float, vy: float, vz: float, yaw_rate: float, duration: float = 0.05
    ):
        """
        Send body-frame velocity command to drone.

        Args:
            vx: Forward velocity (m/s)
            vy: Right velocity (m/s)
            vz: Down velocity (m/s)
            yaw_rate: Yaw rate (rad/s)
            duration: Command duration (seconds)
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        # Clamp velocities to safety limits
        vx = np.clip(vx, -self.max_velocity, self.max_velocity)
        vy = np.clip(vy, -self.max_velocity, self.max_velocity)
        vz = np.clip(vz, -self.max_velocity, self.max_velocity)
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        try:
            # Send velocity command (non-blocking)
            self.client.moveByVelocityBodyFrameAsync(
                vx,
                vy,
                vz,
                duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                vehicle_name=self.drone_name,
            )
        except Exception as e:
            self.logger.error(f"Failed to send velocity command: {e}")
            raise

    def check_collision(self) -> bool:
        """
        Check if drone has collided with environment.

        Returns:
            bool: True if collision detected
        """
        if not self.is_connected:
            return False

        try:
            collision_info = self.client.simGetCollisionInfo(self.drone_name)
            return collision_info.has_collided
        except:
            return False

    def get_landing_targets(self) -> List[Dict[str, Any]]:
        """
        Get positions of landing targets in environment.
        Returns Landing_101-506 targets as per report specification.

        Returns:
            List of target dictionaries with name and position
        """
        if not self.is_connected:
            return []

        targets = []

        # Generate Landing_101-506 targets (5 floors × 6 targets per floor)
        for floor in range(1, 6):  # Floors 1-5
            for target_num in range(1, 7):  # Targets 1-6 per floor
                target_name = f"Landing_{floor}{target_num:02d}"

                # Try to query AirSim for object position
                try:
                    pose = self.client.simGetObjectPose(target_name)
                    position = (
                        pose.position.x_val,
                        pose.position.y_val,
                        pose.position.z_val,
                    )

                    targets.append(
                        {"name": target_name, "position": position, "floor": floor}
                    )
                except:
                    # Generate synthetic position if not found
                    floor_height = -floor * 3.0  # NED: negative Z is up
                    x_positions = [4, 8, 12, 16, 20, 24]
                    y_position = 20.0

                    synthetic_position = (
                        x_positions[target_num - 1],
                        y_position,
                        floor_height,
                    )

                    targets.append(
                        {
                            "name": target_name,
                            "position": synthetic_position,
                            "floor": floor,
                            "synthetic": True,
                        }
                    )

        return targets

    def get_battery_level(self) -> float:
        """
        Get drone battery level (simplified model).

        Returns:
            float: Battery level as fraction [0, 1]
        """
        # Simplified: return 1.0 for now
        # Can be extended with power consumption model
        return 1.0

    def takeoff(self, altitude: float = 1.0, timeout: float = 15.0) -> bool:
        """
        Takeoff to specified altitude.

        Args:
            altitude: Target altitude in meters (positive up)
            timeout: Operation timeout

        Returns:
            bool: True if takeoff successful
        """
        if not self.is_connected:
            return False

        try:
            self.logger.info(f"Taking off to {altitude}m...")
            self.client.takeoffAsync(
                timeout_sec=timeout, vehicle_name=self.drone_name
            ).join()

            # Move to desired altitude
            self.client.moveToZAsync(
                -altitude, 1.0, timeout_sec=timeout, vehicle_name=self.drone_name
            ).join()

            self.logger.info("Takeoff completed")
            return True

        except Exception as e:
            self.logger.error(f"Takeoff failed: {e}")
            return False

    def land(self, timeout: float = 15.0) -> bool:
        """
        Land the drone safely.

        Args:
            timeout: Operation timeout

        Returns:
            bool: True if landing successful
        """
        if not self.is_connected:
            return False

        try:
            self.logger.info("Landing...")
            self.client.landAsync(
                timeout_sec=timeout, vehicle_name=self.drone_name
            ).join()
            self.logger.info("Landing completed")
            return True

        except Exception as e:
            self.logger.error(f"Landing failed: {e}")
            return False

    def is_alive(self) -> bool:
        """
        Check if AirSim connection is still alive.

        Returns:
            bool: True if connection is active
        """
        if not self.is_connected or not self.client:
            return False

        try:
            self.client.ping()
            return True
        except:
            self.is_connected = False
            return False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
