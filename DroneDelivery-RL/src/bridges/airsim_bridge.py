"""
AirSim Bridge
Manages connection and communication with AirSim simulation environment.
Implements exact drone spawn location {6000, -3000, 300} from report.
"""

import airsim
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cv2
from scipy.spatial.transform import Rotation as R

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
    stereo_left: np.ndarray
    stereo_right: np.ndarray  
    depth_image: np.ndarray
    imu_data: Dict[str, Any]
    timestamp: float

class AirSimBridge:
    """
    Bridge to AirSim simulation environment.
    Handles drone control, sensor data collection, and world state queries.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Use default config if none provided
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
        
        # Connection settings
        self.drone_name = config.get('drone_name', 'Drone0')
        self.spawn_location = config.get('spawn_location', [6000.0, -3000.0, 300.0])
        self.spawn_orientation = config.get('spawn_orientation', [0.0, 0.0, 0.0])
        
        # Control settings
        self.max_velocity = config.get('max_velocity', 5.0)        # m/s
        self.max_yaw_rate = config.get('max_yaw_rate', 1.0)        # rad/s
        self.control_frequency = config.get('control_frequency', 20.0)  # Hz
        
        # Camera settings (exact match with old report: 30Hz stereo)
        self.camera_frequency = config.get('camera_frequency', 30.0)  # Hz
        self.camera_resolution = config.get('camera_resolution', [640, 480])
        
        # IMU settings (exact match with old report: 200Hz)
        self.imu_frequency = config.get('imu_frequency', 200.0)    # Hz
        
        # AirSim client
        self.client: Optional[airsim.MultirotorClient] = None
        self.is_connected = False
        
        # State tracking
        self.last_drone_state: Optional[DroneState] = None
        self.last_sensor_data: Optional[SensorData] = None
        
        # Safety settings
        self.collision_threshold = config.get('collision_threshold', 0.1)  # meters
        self.ground_clearance = config.get('ground_clearance', 0.5)        # meters
        
    def connect(self) -> bool:
        """
        Connect to AirSim simulation.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.logger.info("Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            
            # Enable API control
            self.client.enableApiControl(True, self.drone_name)
            self.client.armDisarm(True, self.drone_name)
            
            # Reset to spawn location
            self.reset_drone()
            
            self.is_connected = True
            self.logger.info(f"Connected to AirSim. Drone: {self.drone_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to AirSim: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from AirSim."""
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
        """Reset drone to spawn location with exact coordinates from report."""
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")
        
        # Convert spawn location to AirSim coordinates
        pose = airsim.Pose(
            position_val=airsim.Vector3r(
                self.spawn_location[0], 
                self.spawn_location[1], 
                self.spawn_location[2]
            ),
            orientation_val=airsim.to_quaternion(
                self.spawn_orientation[0],  # pitch
                self.spawn_orientation[1],  # roll  
                self.spawn_orientation[2]   # yaw
            )
        )
        
        # Reset drone
        self.client.reset()
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        
        # Set initial pose
        self.client.simSetVehiclePose(pose, True, self.drone_name)
        
        # Wait for stabilization
        time.sleep(1.0)
        
        self.logger.info(f"Drone reset to spawn location: {self.spawn_location}")
    
    def get_drone_state(self) -> DroneState:
        """
        Get current drone state from AirSim.
        
        Returns:
            DroneState: Complete drone state information
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to AirSim")
        
        # Get kinematics state
        kinematics = self.client.simGetGroundTruthKinematics(self.drone_name)
        
        # Extract position (NED coordinates)
        position = (
            kinematics.position.x_val,
            kinematics.position.y_val, 
            kinematics.position.z_val
        )
        
        # Extract orientation (quaternion)
        orientation = (
            kinematics.orientation.w_val,
            kinematics.orientation.x_val,
            kinematics.orientation.y_val,
            kinematics.orientation.z_val
        )
        
        # Extract velocities
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
        
        # Get stereo camera images (30Hz as per old report)
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_left", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_right", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("depth", airsim.ImageType.DepthPerspective, True, False)
        ], self.drone_name)
        
        # Process stereo left
        stereo_left = None
        if len(responses) > 0 and responses[0].pixels_as_float:
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            stereo_left = img1d.reshape(responses[0].height, responses[0].width, 3)
            stereo_left = cv2.cvtColor(stereo_left, cv2.COLOR_BGR2RGB)
        
        # Process stereo right  
        stereo_right = None
        if len(responses) > 1 and responses[1].pixels_as_float:
            img1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
            stereo_right = img1d.reshape(responses[1].height, responses[1].width, 3)
            stereo_right = cv2.cvtColor(stereo_right, cv2.COLOR_BGR2RGB)
        
        # Process depth image
        depth_image = None
        if len(responses) > 2:
            depth_image = airsim.list_to_2d_float_array(
                responses[2].image_data_float, 
                responses[2].width, 
                responses[2].height
            )
            depth_image = np.array(depth_image, dtype=np.float32)
        
        # Get IMU data (200Hz as per old report)
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
        
        # Send velocity command
        self.client.moveByVelocityBodyFrameAsync(
            vx, vy, vz, duration, 
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=self.drone_name
        )
    
    def check_collision(self) -> bool:
        """
        Check if drone has collided with environment.
        
        Returns:
            bool: True if collision detected
        """
        if not self.is_connected:
            return False
        
        collision_info = self.client.simGetCollisionInfo(self.drone_name)
        return collision_info.has_collided
    
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
                
                # Query AirSim for object position (if placed in environment)
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
                    # If target not found in simulation, generate synthetic position
                    # Based on floor layout from report (20×40×3m per floor)
                    floor_height = floor * 3.0  # 3m per floor
                    
                    # Distribute targets across floor area
                    x_positions = [4, 8, 12, 16, 20, 24]  # 6 positions along length
                    y_position = 20.0  # Center of width
                    
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
    
    def get_dynamic_obstacles(self) -> List[Tuple[float, float, float]]:
        """
        Get positions of dynamic obstacles (human agents) from AirSim.
        
        Returns:
            List of obstacle positions as (x, y, z) tuples
        """
        if not self.is_connected:
            return []
        
        obstacles = []
        
        # Query for human agents (typically named Human_1, Human_2, etc.)
        for i in range(1, 13):  # Up to 12 human agents as per report
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
                # Human agent not found or not implemented
                continue
        
        return obstacles
    
    def get_battery_level(self) -> float:
        """
        Get drone battery level.
        
        Returns:
            float: Battery level as fraction [0, 1]
        """
        if not self.is_connected:
            return 0.0
        
        try:
            # AirSim doesn't have built-in battery simulation
            # Use energy consumption model based on flight time
            # This is a simplified implementation
            
            state = self.get_drone_state()
            velocity_magnitude = np.linalg.norm(state.linear_velocity)
            
            # Simple battery model: higher velocities consume more battery
            # In practice, this would integrate thrust commands over time
            base_consumption = 0.001  # Base consumption per second
            velocity_consumption = velocity_magnitude * 0.0005
            
            total_consumption = base_consumption + velocity_consumption
            
            # For now, return constant value (can be extended)
            return max(0.0, 1.0 - total_consumption * time.time() / 3600)  # 1 hour flight time
            
        except:
            return 1.0  # Full battery if calculation fails
    
    def takeoff(self, altitude: float = 1.0) -> bool:
        """
        Takeoff to specified altitude.
        
        Args:
            altitude: Target altitude (positive up)
            
        Returns:
            bool: True if takeoff successful
        """
        if not self.is_connected:
            return False
        
        try:
            self.client.takeoffAsync(vehicle_name=self.drone_name).join()
            
            # Move to desired altitude
            self.client.moveToZAsync(-altitude, 1.0, vehicle_name=self.drone_name).join()
            
            self.logger.info(f"Takeoff completed to {altitude}m altitude")
            return True
            
        except Exception as e:
            self.logger.error(f"Takeoff failed: {e}")
            return False
    
    def land(self) -> bool:
        """
        Land the drone safely.
        
        Returns:
            bool: True if landing successful
        """
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
        """Emergency stop - immediately disable motors."""
        if self.is_connected:
            try:
                self.client.armDisarm(False, self.drone_name)
                self.logger.warning("Emergency stop activated")
            except Exception as e:
                self.logger.error(f"Emergency stop failed: {e}")
    
    def get_world_bounds(self) -> Dict[str, float]:
        """
        Get world coordinate bounds for 5-floor building.
        
        Returns:
            Dictionary with min/max bounds for each axis
        """
        return {
            'x_min': 0.0, 'x_max': 20.0,      # 20m length per report
            'y_min': 0.0, 'y_max': 40.0,      # 40m width per report  
            'z_min': 0.0, 'z_max': 15.0       # 5 floors × 3m = 15m height
        }
    
    def is_alive(self) -> bool:
        """
        Check if AirSim connection is still alive.
        
        Returns:
            bool: True if connection is active
        """
        if not self.is_connected or not self.client:
            return False
        
        try:
            # Try a simple API call to test connection
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
