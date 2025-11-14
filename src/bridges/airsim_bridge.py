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

dataclass
class DroneState:

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    linear_velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    timestamp: float

dataclass
class SensorData:

    stereo_left: Optional[np.ndarray]
    stereo_right: Optional[np.ndarray]
    depth_image: Optional[np.ndarray]
    imu_data: Dict[str, Any]
    timestamp: float

class AirSimBridge:

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        if config is None:
            config = {}

        self.config = config
        self.logger = logging.getLogger(__name__)

        self.drone_name = config.get("drone_name", "Drone1")

        if "spawn_location" not in config:
            raise ValueError("spawn_location must be set in AirSim config")

        spawn_location = config["spawn_location"]
        if len(spawn_location) != 3:
            raise ValueError("spawn_location must contain exactly 3 values (x, y, z)")

        self.spawn_location = tuple(float(value) for value in spawn_location)

        spawn_orientation = config.get("spawn_orientation", (0.0, 0.0, 0.0))
        if len(spawn_orientation) != 3:
            raise ValueError("spawn_orientation must contain 3 values (pitch, roll, yaw)")
        self.spawn_orientation = tuple(float(value) for value in spawn_orientation)

        self._sanitize_spawn_location()

        self.max_velocity = config.get("max_velocity", 5.0)
        self.max_yaw_rate = config.get("max_yaw_rate", 1.0)
        self.control_frequency = config.get("control_frequency", 20.0)

        self.camera_frequency = config.get("camera_frequency", 30.0)
        self.camera_resolution = tuple(config.get("camera_resolution", (640, 480)))

        self.imu_frequency = config.get("imu_frequency", 200.0)

        self.client: Optional[airsim.MultirotorClient] = None
        self.is_connected = False

        self.last_drone_state: Optional[DroneState] = None
        self.last_sensor_data: Optional[SensorData] = None

        self.collision_threshold = config.get("collision_threshold", 0.1)
        self.ground_clearance = config.get("ground_clearance", 0.5)

        self.rpc_host = config.get("host", "127.0.0.1")
        self.rpc_port = config.get("port", 41451)
        self.rpc_timeout = config.get("rpc_timeout", 10.0)

        self.logger.info(f"AirSimBridge initialized for drone: {self.drone_name}")
        self.logger.info(f"Spawn location: {self.spawn_location}")

    def _safe_airsim_call(self, func, args, kwargs):

        try:
            return func(args, kwargs)
        except RuntimeError as e:
            if "IOLoop is already running" in str(e) or "already running" in str(e).lower():
                self.logger.warning(f"IOLoop error, retrying: {e}")
                try:
                    return func(args, kwargs)
                except:
                    return None
            else:
                raise
        except Exception as e:
            self.logger.error(f"AirSim API error: {e}")
            return None

    def _sanitize_spawn_location(self) - Tuple[float, float, float]:

        x, y, z = self.spawn_location
        if z  0:
            self.logger.warning("Spawn z=.2f  0 (NED). Forcing z=-3.0 for safety.", z)
            z = -3.0
            self.spawn_location = (x, y, z)
        return self.spawn_location

    def connect(self, timeout: float = 10.0, max_retries: int = 3) - bool:

        if self.is_connected:
            self.logger.warning("Already connected to AirSim")
            return True

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"Connecting to AirSim (attempt {attempt}/{max_retries})...")

                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()

                self.logger.info("Testing AirSim connection...")
                self.client.ping()

                available = self.client.listVehicles()
                if available:
                    self.logger.info(f"Vehicles reported by AirSim: {available}")
                    if self.drone_name not in available:
                        self.logger.warning(
                            "Vehicle 's' not found. Falling back to 's'.",
                            self.drone_name,
                            available[0],
                        )
                        self.drone_name = available[0]
                else:
                    self.logger.warning(
                        "AirSim returned no vehicles; continuing with 's'.",
                        self.drone_name,
                    )

                self.logger.info("Enabling API control...")
                self.client.enableApiControl(True, self.drone_name)

                self.logger.info("Arming drone...")
                self.client.armDisarm(True, self.drone_name)

                self.is_connected = True
                self.logger.info(f"Connected to AirSim. Drone: {self.drone_name}")
                return True

            except Exception as e:
                self.logger.error(f"Connection attempt {attempt} failed: {e}")
                if attempt  max_retries:
                    time.sleep(2.0  attempt)
                    continue

        self.logger.error(f"Failed to connect after {max_retries} attempts")
        self.is_connected = False
        self.client = None
        return False

    def disconnect(self):

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

        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        try:
            self.logger.info("Resetting drone...")
            try:
                self.client.reset()
            except RuntimeError as err:
                if "IOLoop is already running" in str(err):
                    self.logger.warning("reset() hit IOLoop issue; falling back to direct RPC call.")
                    self._rpc_call("reset")
                else:
                    raise

            try:
                self.client.enableApiControl(True, self.drone_name)
                self.client.armDisarm(True, self.drone_name)
            except RuntimeError as err:
                if "IOLoop is already running" in str(err):
                    self.logger.warning("armDisarm/enableApiControl hit IOLoop issue; retrying via RPC.")
                    self._rpc_call("enableApiControl", True, self.drone_name)
                    self._rpc_call("armDisarm", True, self.drone_name)
                else:
                    raise

            if to_spawn:
                spawn_location = self._sanitize_spawn_location()
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        spawn_location[0],
                        spawn_location[1],
                        spawn_location[2],
                    ),
                    orientation_val=airsim.to_quaternion(
                        self.spawn_orientation[0],
                        self.spawn_orientation[1],
                        self.spawn_orientation[2],
                    ),
                )

                try:
                    self.client.simSetVehiclePose(pose, True, self.drone_name)
                except RuntimeError as err:
                    if "IOLoop is already running" in str(err):
                        self.logger.warning("simSetVehiclePose hit IOLoop issue; retrying via RPC call.")
                        self._rpc_call("simSetVehiclePose", pose, True, self.drone_name)
                    else:
                        raise

                self.logger.info(f"Drone reset to spawn: {self.spawn_location}")
            else:
                self.logger.info("Drone reset (current position)")

            time.sleep(0.5)

        except Exception as e:
            self.logger.error(f"Reset failed: {e}")
            raise

    def _default_drone_state(self) - DroneState:
        return DroneState(
            position=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            timestamp=time.time(),
        )

    def _state_from_multirotor(self, multi_state: Any) - DroneState:
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

    def _rpc_call(self, method: str, args):

        address = msgpackrpc.Address(self.rpc_host, self.rpc_port)
        client = msgpackrpc.Client(address, timeout=self.rpc_timeout)
        try:
            return client.call(method, args)
        finally:
            client.close()

    staticmethod
    def _ensure_str_keys(obj: Any) - Any:
        if isinstance(obj, dict):
            return {
                (k.decode("utf-8") if isinstance(k, bytes) else k): AirSimBridge._ensure_str_keys(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [AirSimBridge._ensure_str_keys(v) for v in obj]
        return obj

    def _get_multirotor_state_safe(self):

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

    def get_drone_state(self) - DroneState:

        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        try:
            multi_state = self._get_multirotor_state_safe()
            state = self._state_from_multirotor(multi_state)
            self.last_drone_state = state
            return state

        except Exception as e:
            self.logger.error(f"Failed to get drone state: {e}")
            if self.last_drone_state is not None:
                self.logger.warning("Using cached drone state")
                return self.last_drone_state
            self.logger.warning("Returning default drone state")
            return self._default_drone_state()

    def get_sensor_data(self) - SensorData:

        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        timestamp = time.time()

        try:
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("front_right", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False),
                ],
                self.drone_name,
            )

            stereo_left = None
            if len(responses)  0 and len(responses[0].image_data_uint8)  0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).copy()
                if img1d.size  0:
                    stereo_left = img1d.reshape(responses[0].height, responses[0].width, 3)
                    stereo_left = cv2.cvtColor(stereo_left, cv2.COLOR_BGR2RGB)

            stereo_right = None
            if len(responses)  1 and len(responses[1].image_data_uint8)  0:
                img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8).copy()
                if img1d.size  0:
                    stereo_right = img1d.reshape(responses[1].height, responses[1].width, 3)
                    stereo_right = cv2.cvtColor(stereo_right, cv2.COLOR_BGR2RGB)

            depth_image = None
            if len(responses)  2 and len(responses[2].image_data_float)  0:
                depth_image = airsim.list_to_2d_float_array(
                    responses[2].image_data_float,
                    responses[2].width,
                    responses[2].height,
                )
                depth_image = np.array(depth_image, dtype=np.float32)

            imu_data_raw = self.client.getImuData(imu_name="Imu", vehicle_name=self.drone_name)
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

        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        vx = np.clip(vx, -self.max_velocity, self.max_velocity)
        vy = np.clip(vy, -self.max_velocity, self.max_velocity)
        vz = np.clip(vz, -self.max_velocity, self.max_velocity)
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        try:
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

    def check_collision(self) - bool:

        if not self.is_connected:
            return False

        try:
            collision_info = self.client.simGetCollisionInfo(self.drone_name)
            return collision_info.has_collided
        except:
            return False

    def emergency_stop(self):

        if not self.is_connected:
            return

        try:
            self.logger.error("EMERGENCY STOP - Hovering in place")
            self.client.moveByVelocityBodyFrameAsync(
                0, 0, 0, 1.0,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                vehicle_name=self.drone_name
            ).join()
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")

    def get_landing_targets(self) - List[Dict[str, Any]]:

        if not self.is_connected:
            return []

        targets = []

        for floor in range(1, 6):
            for target_num in range(1, 7):
                target_name = f"Landing_{floor}{target_num:02d}"

                try:
                    pose = self.client.simGetObjectPose(target_name)
                    position = (
                        pose.position.x_val,
                        pose.position.y_val,
                        pose.position.z_val,
                    )

                    targets.append({"name": target_name, "position": position, "floor": floor})
                except:
                    floor_height = -floor  3.0
                    x_positions = [4, 8, 12, 16, 20, 24]
                    y_position = 20.0

                    synthetic_position = (x_positions[target_num - 1], y_position, floor_height)

                    targets.append({
                        "name": target_name,
                        "position": synthetic_position,
                        "floor": floor,
                        "synthetic": True,
                    })

        return targets

    def get_battery_level(self) - float:

        return 1.0

    def takeoff(self, altitude: float = 1.0, timeout: float = 15.0) - bool:

        if not self.is_connected:
            return False

        try:
            self.logger.info(f"Taking off to {altitude}m...")
            self.client.takeoffAsync(timeout_sec=timeout, vehicle_name=self.drone_name).join()

            self.client.moveToZAsync(
                -altitude, 1.0, timeout_sec=timeout, vehicle_name=self.drone_name
            ).join()

            max_verification_time = 5.0
            start_time = time.time()
            while time.time() - start_time  max_verification_time:
                state = self.get_drone_state()
                current_altitude = abs(state.position[2])
                if abs(current_altitude - altitude)  0.5:
                    self.logger.info(f"Takeoff completed - altitude: {current_altitude:.2f}m")
                    return True
                time.sleep(0.5)

            self.logger.warning(f"Takeoff altitude verification timeout")
            return True

        except Exception as e:
            self.logger.error(f"Takeoff failed: {e}")
            return False

    def land(self, timeout: float = 15.0) - bool:

        if not self.is_connected:
            return False

        try:
            self.logger.info("Landing...")
            self.client.landAsync(timeout_sec=timeout, vehicle_name=self.drone_name).join()
            self.logger.info("Landing completed")
            return True

        except Exception as e:
            self.logger.error(f"Landing failed: {e}")
            return False

    def is_alive(self) - bool:

        if not self.is_connected or not self.client:
            return False

        try:
            self.client.ping()
            return True
        except:
            self.is_connected = False
            return False

    def __enter__(self):

        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.disconnect()
