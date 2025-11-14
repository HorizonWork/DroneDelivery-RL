import airsim
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cv2
from scipy.spatial.transform import Rotation as R

dataclass
class DroneState:

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    linear_velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    timestamp: float

dataclass
class SensorData:

    stereo_left: np.ndarray
    stereo_right: np.ndarray
    depth_image: np.ndarray
    imu_data: Dict[str, Any]
    timestamp: float

class AirSimBridge:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'drone_name': 'Drone0',
                'spawn_location': (6000.0, -3000.0, 300.0),
                'spawn_orientation': (0.0, 0.0, 0.0),
                'max_velocity': 5.0,
                'max_yaw_rate': 1.0,
                'control_frequency': 20.0,
                'camera_frequency': 30.0,
                'camera_resolution': (640, 480),
                'imu_frequency': 200.0,
                'collision_threshold': 0.1,
                'ground_clearance': 0.5
            }
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.drone_name = config.get('drone_name', 'Drone0')
        self.spawn_location = config.get('spawn_location', [6000.0, -3000.0, 300.0])
        self.spawn_orientation = config.get('spawn_orientation', [0.0, 0.0, 0.0])

        self.max_velocity = config.get('max_velocity', 5.0)
        self.max_yaw_rate = config.get('max_yaw_rate', 1.0)
        self.control_frequency = config.get('control_frequency', 20.0)

        self.camera_frequency = config.get('camera_frequency', 30.0)
        self.camera_resolution = config.get('camera_resolution', [640, 480])

        self.imu_frequency = config.get('imu_frequency', 200.0)

        self.client: Optional[airsim.MultirotorClient] = None
        self.is_connected = False

        self.last_drone_state: Optional[DroneState] = None
        self.last_sensor_data: Optional[SensorData] = None

        self.collision_threshold = config.get('collision_threshold', 0.1)
        self.ground_clearance = config.get('ground_clearance', 0.5)

    def connect(self) - bool:

        try:
            self.logger.info("Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()

            self.client.enableApiControl(True, self.drone_name)
            self.client.armDisarm(True, self.drone_name)

            self.reset_drone()

            self.is_connected = True
            self.logger.info(f"Connected to AirSim. Drone: {self.drone_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to AirSim: {e}")
            self.is_connected = False
            return False

    def disconnect(self):

        if self.client and self.is_connected:
            try:
                self.client.armDisarm(False, self.drone_name)
                self.client.enableApiControl(False, self.drone_name)
                self.logger.info("Disconnected from AirSim")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
            finally:
                self.is_connected = False
                self.client = None

    def reset_drone(self):

        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        pose = airsim.Pose(
            position_val=airsim.Vector3r(
                self.spawn_location[0],
                self.spawn_location[1],
                self.spawn_location[2]
            ),
            orientation_val=airsim.to_quaternion(
                self.spawn_orientation[0],
                self.spawn_orientation[1],
                self.spawn_orientation[2]
            )
        )

        self.client.reset()
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)

        self.client.simSetVehiclePose(pose, True, self.drone_name)

        time.sleep(1.0)

        self.logger.info(f"Drone reset to spawn location: {self.spawn_location}")

    def get_drone_state(self) - DroneState:

        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        kinematics = self.client.simGetGroundTruthKinematics(self.drone_name)

        position = (
            kinematics.position.x_val,
            kinematics.position.y_val,
            kinematics.position.z_val
        )

        orientation = (
            kinematics.orientation.w_val,
            kinematics.orientation.x_val,
            kinematics.orientation.y_val,
            kinematics.orientation.z_val
        )

        linear_velocity = (
            kinematics.linear_velocity.x_val,
            kinematics.linear_velocity.y_val,
            kinematics.linear_velocity.z_val
        )

        angular_velocity = (
            kinematics.angular_velocity.x_val,
            kinematics.angular_velocity.y_val,
            kinematics.angular_velocity.z_val
        )

        state = DroneState(
            position=position,
            orientation=orientation,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            timestamp=time.time()
        )

        self.last_drone_state = state
        return state

    def get_sensor_data(self) - SensorData:

        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        timestamp = time.time()

        responses = self.client.simGetImages([
            airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_right", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("depth", airsim.ImageType.DepthPerspective, True, False)
        ], self.drone_name)

        stereo_left = None
        if len(responses)  0 and responses[0].pixels_as_float:
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            stereo_left = img1d.reshape(responses[0].height, responses[0].width, 3)
            stereo_left = cv2.cvtColor(stereo_left, cv2.COLOR_BGR2RGB)

        stereo_right = None
        if len(responses)  1 and responses[1].pixels_as_float:
            img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
            stereo_right = img1d.reshape(responses[1].height, responses[1].width, 3)
            stereo_right = cv2.cvtColor(stereo_right, cv2.COLOR_BGR2RGB)

        depth_image = None
        if len(responses)  2:
            depth_image = airsim.list_to_2d_float_array(
                responses[2].image_data_float,
                responses[2].width,
                responses[2].height
            )
            depth_image = np.array(depth_image, dtype=np.float32)

        imu_data = self.client.getImuData(imu_name="Imu", vehicle_name=self.drone_name)
        imu_dict = {
            'linear_acceleration': (
                imu_data.linear_acceleration.x_val,
                imu_data.linear_acceleration.y_val,
                imu_data.linear_acceleration.z_val
            ),
            'angular_velocity': (
                imu_data.angular_velocity.x_val,
                imu_data.angular_velocity.y_val,
                imu_data.angular_velocity.z_val
            ),
            'orientation': (
                imu_data.orientation.w_val,
                imu_data.orientation.x_val,
                imu_data.orientation.y_val,
                imu_data.orientation.z_val
            ),
            'timestamp': imu_data.time_stamp
        }

        sensor_data = SensorData(
            stereo_left=stereo_left,
            stereo_right=stereo_right,
            depth_image=depth_image,
            imu_data=imu_dict,
            timestamp=timestamp
        )

        self.last_sensor_data = sensor_data
        return sensor_data

    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float,
                             duration: float = 0.05):

        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")

        vx = np.clip(vx, -self.max_velocity, self.max_velocity)
        vy = np.clip(vy, -self.max_velocity, self.max_velocity)
        vz = np.clip(vz, -self.max_velocity, self.max_velocity)
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        self.client.moveByVelocityBodyFrameAsync(
            vx, vy, vz, duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=self.drone_name
        )

    def check_collision(self) - bool:

        if not self.is_connected:
            return False

        collision_info = self.client.simGetCollisionInfo(self.drone_name)
        return collision_info.has_collided

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
                        pose.position.z_val
                    )

                    targets.append({
                        'name': target_name,
                        'position': position,
                        'floor': floor
                    })
                except:
                    floor_height = floor  3.0

                    x_positions = [4, 8, 12, 16, 20, 24]
                    y_position = 20.0

                    synthetic_position = (
                        x_positions[target_num - 1],
                        y_position,
                        floor_height
                    )

                    targets.append({
                        'name': target_name,
                        'position': synthetic_position,
                        'floor': floor,
                        'synthetic': True
                    })

        return targets

    def get_dynamic_obstacles(self) - List[Tuple[float, float, float]]:

        if not self.is_connected:
            return []

        obstacles = []

        for i in range(1, 13):
            try:
                human_name = f"Human_{i}"
                pose = self.client.simGetObjectPose(human_name)
                position = (
                    pose.position.x_val,
                    pose.position.y_val,
                    pose.position.z_val
                )
                obstacles.append(position)

            except:
                continue

        return obstacles

    def get_battery_level(self) - float:

        if not self.is_connected:
            return 0.0

        try:

            state = self.get_drone_state()
            velocity_magnitude = np.linalg.norm(state.linear_velocity)

            base_consumption = 0.001
            velocity_consumption = velocity_magnitude  0.0005

            total_consumption = base_consumption + velocity_consumption

            return max(0.0, 1.0 - total_consumption  time.time() / 3600)

        except:
            return 1.0

    def takeoff(self, altitude: float = 1.0) - bool:

        if not self.is_connected:
            return False

        try:
            self.client.takeoffAsync(vehicle_name=self.drone_name).join()

            self.client.moveToZAsync(-altitude, 1.0, vehicle_name=self.drone_name).join()

            self.logger.info(f"Takeoff completed to {altitude}m altitude")
            return True

        except Exception as e:
            self.logger.error(f"Takeoff failed: {e}")
            return False

    def land(self) - bool:

        if not self.is_connected:
            return False

        try:
            self.client.landAsync(vehicle_name=self.drone_name).join()
            self.logger.info("Landing completed")
            return True

        except Exception as e:
            self.logger.error(f"Landing failed: {e}")
            return False

    def emergency_stop(self):

        if self.is_connected:
            try:
                self.client.armDisarm(False, self.drone_name)
                self.logger.warning("Emergency stop activated")
            except Exception as e:
                self.logger.error(f"Emergency stop failed: {e}")

    def get_world_bounds(self) - Dict[str, float]:

        return {
            'x_min': 0.0, 'x_max': 20.0,
            'y_min': 0.0, 'y_max': 40.0,
            'z_min': 0.0, 'z_max': 15.0
        }

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
