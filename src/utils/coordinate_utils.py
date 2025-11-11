"""
Coordinate Utilities
Coordinate system transformations for VI-SLAM, planning, and control.
Handles conversions between world, body, camera, and grid frames.
"""

import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

@dataclass
class Pose:
    """6DOF pose representation."""
    position: np.ndarray      # [x, y, z]
    orientation: np.ndarray   # [qx, qy, qz, qw] quaternion
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(self.orientation).as_matrix()
        T[:3, 3] = self.position
        return T
    
    @classmethod
    def from_matrix(cls, T: np.ndarray):
        """Create pose from 4x4 transformation matrix."""
        position = T[:3, 3]
        rotation = Rotation.from_matrix(T[:3, :3])
        orientation = rotation.as_quat()  # [x, y, z, w]
        return cls(position, orientation)

class CoordinateTransformer:
    """
    Coordinate system transformer for multi-frame operations.
    Handles all coordinate conversions in the drone delivery system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Coordinate frame definitions
        self.frames = {
            'world': 'ENU',          # East-North-Up world frame
            'body': 'FRD',           # Forward-Right-Down body frame  
            'camera': 'FRD',         # Forward-Right-Down camera frame
            'slam': 'arbitrary'      # VI-SLAM arbitrary origin
        }
        
        # Transformation parameters
        self.body_to_camera_offset = np.array(config.get('body_to_camera_offset', [0.1, 0.0, -0.05]))  # meters
        self.slam_to_world_transform = np.eye(4)  # Will be calibrated
        
        # Grid parameters for planning
        self.grid_resolution = config.get('grid_resolution', 0.5)     # meters per cell
        self.grid_origin = np.array(config.get('grid_origin', [0, 0, 0]))  # world coordinates
        
        # Building bounds
        self.building_bounds = {
            'x_min': config.get('x_min', 0.0),
            'x_max': config.get('x_max', 20.0),
            'y_min': config.get('y_min', 0.0),
            'y_max': config.get('y_max', 40.0),
            'z_min': config.get('z_min', 0.0),
            'z_max': config.get('z_max', 15.0)    # 5 floors × 3m
        }
        
        self.logger.info("Coordinate Transformer initialized")
        self.logger.info(f"Grid resolution: {self.grid_resolution}m")
        self.logger.info(f"Building bounds: {self.building_bounds}")
    
    def world_to_grid(self, world_position: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert world coordinates to grid indices.
        
        Args:
            world_position: [x, y, z] in world frame
            
        Returns:
            (grid_x, grid_y, grid_z) indices
        """
        relative_pos = world_position - self.grid_origin
        grid_indices = (relative_pos / self.grid_resolution).astype(int)
        
        # Clamp to grid bounds
        max_indices = np.array([
            int((self.building_bounds['x_max'] - self.building_bounds['x_min']) / self.grid_resolution),
            int((self.building_bounds['y_max'] - self.building_bounds['y_min']) / self.grid_resolution),
            int((self.building_bounds['z_max'] - self.building_bounds['z_min']) / self.grid_resolution)
        ])
        
        grid_indices = np.clip(grid_indices, 0, max_indices - 1)
        
        return tuple(grid_indices)
    
    def grid_to_world(self, grid_indices: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert grid indices to world coordinates.
        
        Args:
            grid_indices: (grid_x, grid_y, grid_z)
            
        Returns:
            [x, y, z] world coordinates
        """
        grid_pos = np.array(grid_indices) * self.grid_resolution
        world_position = grid_pos + self.grid_origin
        
        return world_position


def world_to_grid(world_position: Tuple[float, float, float],
                  cell_size: float = 0.5,
                  floor_height: float = 3.0,
                  origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
                  ) -> Tuple[int, int, int]:
    """
    Lightweight helper used by tests: convert world coordinates to grid indices.
    """
    x, y, z = world_position
    ox, oy, oz = origin
    gx = int(np.floor((x - ox) / cell_size))
    gy = int(np.floor((y - oy) / cell_size))
    gz = int(np.floor((z - oz) / floor_height))
    return (gx, gy, gz)


def grid_to_world(grid_indices: Tuple[int, int, int],
                  cell_size: float = 0.5,
                  floor_height: float = 3.0,
                  origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
                  ) -> Tuple[float, float, float]:
    """
    Inverse of world_to_grid; returns the center of the target cell in world coordinates.
    """
    gx, gy, gz = grid_indices
    ox, oy, oz = origin
    x = ox + (gx + 0.5) * cell_size
    y = oy + (gy + 0.5) * cell_size
    z = oz + gz * floor_height + 0.5 * floor_height
    return (float(x), float(y), float(z))
    
    def body_to_world(self, body_position: np.ndarray, drone_pose: Pose) -> np.ndarray:
        """
        Transform position from body frame to world frame.
        
        Args:
            body_position: Position in body frame
            drone_pose: Current drone pose
            
        Returns:
            Position in world frame
        """
        # Create transformation matrix
        T_world_body = drone_pose.to_matrix()
        
        # Transform position (homogeneous coordinates)
        body_homogeneous = np.append(body_position, 1.0)
        world_homogeneous = T_world_body @ body_homogeneous
        
        return world_homogeneous[:3]
    
    def world_to_body(self, world_position: np.ndarray, drone_pose: Pose) -> np.ndarray:
        """
        Transform position from world frame to body frame.
        
        Args:
            world_position: Position in world frame
            drone_pose: Current drone pose
            
        Returns:
            Position in body frame
        """
        # Inverse transformation
        T_world_body = drone_pose.to_matrix()
        T_body_world = np.linalg.inv(T_world_body)
        
        # Transform position
        world_homogeneous = np.append(world_position, 1.0)
        body_homogeneous = T_body_world @ world_homogeneous
        
        return body_homogeneous[:3]
    
    def slam_to_world(self, slam_pose: Pose) -> Pose:
        """
        Convert VI-SLAM pose to world frame.
        
        Args:
            slam_pose: Pose from VI-SLAM (arbitrary origin)
            
        Returns:
            Pose in world frame
        """
        # Apply SLAM to world transformation
        slam_matrix = slam_pose.to_matrix()
        world_matrix = self.slam_to_world_transform @ slam_matrix
        
        return Pose.from_matrix(world_matrix)
    
    def calibrate_slam_to_world(self, slam_positions: List[np.ndarray], 
                               world_positions: List[np.ndarray]):
        """
        Calibrate SLAM to world coordinate transformation.
        
        Args:
            slam_positions: Known positions in SLAM frame
            world_positions: Corresponding positions in world frame
        """
        if len(slam_positions) < 3:
            self.logger.warning("Need at least 3 points for calibration")
            return
        
        slam_points = np.array(slam_positions)
        world_points = np.array(world_positions)
        
        # Solve for transformation (simplified - assumes similarity transform)
        # In practice, would use more robust methods like ICP
        slam_centroid = np.mean(slam_points, axis=0)
        world_centroid = np.mean(world_points, axis=0)
        
        # Translation
        translation = world_centroid - slam_centroid
        
        # Scale (simplified)
        slam_scale = np.linalg.norm(slam_points - slam_centroid, axis=1)
        world_scale = np.linalg.norm(world_points - world_centroid, axis=1)
        scale = np.mean(world_scale / (slam_scale + 1e-6))
        
        # Update transformation matrix
        self.slam_to_world_transform[:3, :3] *= scale
        self.slam_to_world_transform[:3, 3] = translation
        
        self.logger.info(f"SLAM calibration: scale={scale:.3f}, translation={translation}")
    
    def get_floor_number(self, world_position: np.ndarray) -> int:
        """
        Get floor number from world position.
        
        Args:
            world_position: [x, y, z] world coordinates
            
        Returns:
            Floor number (1-5)
        """
        floor = max(1, min(5, int(world_position[2] // 3.0) + 1))
        return floor
    
    def get_floor_bounds(self, floor_number: int) -> Dict[str, float]:
        """
        Get coordinate bounds for specific floor.
        
        Args:
            floor_number: Floor number (1-5)
            
        Returns:
            Floor coordinate bounds
        """
        floor_height = 3.0  # meters per floor
        
        return {
            'x_min': self.building_bounds['x_min'],
            'x_max': self.building_bounds['x_max'],
            'y_min': self.building_bounds['y_min'], 
            'y_max': self.building_bounds['y_max'],
            'z_min': (floor_number - 1) * floor_height,
            'z_max': floor_number * floor_height
        }
    
    def is_position_valid(self, world_position: np.ndarray) -> bool:
        """Check if world position is within building bounds."""
        x, y, z = world_position
        
        return (self.building_bounds['x_min'] <= x <= self.building_bounds['x_max'] and
                self.building_bounds['y_min'] <= y <= self.building_bounds['y_max'] and
                self.building_bounds['z_min'] <= z <= self.building_bounds['z_max'])

class FrameConverter:
    """
    Specialized frame converter for drone-specific transformations.
    Handles body frame, NED/ENU conversions, and sensor frame alignment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Frame conventions
        self.use_enu = config.get('use_enu_convention', True)  # East-North-Up vs North-East-Down
        
        # Sensor mounting offsets
        self.camera_offset = np.array(config.get('camera_offset', [0.1, 0.0, -0.05]))
        self.imu_offset = np.array(config.get('imu_offset', [0.0, 0.0, 0.0]))
        
        # Rotation matrices for frame conversions
        if self.use_enu:
            # ENU to NED conversion
            self.enu_to_ned = np.array([
                [0, 1, 0],
                [1, 0, 0], 
                [0, 0, -1]
            ])
            self.ned_to_enu = self.enu_to_ned.T
        
        self.logger.info("Frame Converter initialized")
        self.logger.info(f"Convention: {'ENU' if self.use_enu else 'NED'}")
    
    def ned_to_enu(self, ned_vector: np.ndarray) -> np.ndarray:
        """Convert North-East-Down to East-North-Up."""
        if not self.use_enu:
            return ned_vector
        return self.ned_to_enu @ ned_vector
    
    def enu_to_ned(self, enu_vector: np.ndarray) -> np.ndarray:
        """Convert East-North-Up to North-East-Down.""" 
        if not self.use_enu:
            return enu_vector
        return self.enu_to_ned @ enu_vector
    
    def body_velocity_to_world(self, body_velocity: np.ndarray, orientation_quat: np.ndarray) -> np.ndarray:
        """
        Transform body frame velocity to world frame.
        
        Args:
            body_velocity: [vx, vy, vz] in body frame
            orientation_quat: [qx, qy, qz, qw] body orientation
            
        Returns:
            [vx, vy, vz] in world frame
        """
        # Convert quaternion to rotation matrix
        rotation = Rotation.from_quat(orientation_quat)
        rotation_matrix = rotation.as_matrix()
        
        # Transform velocity
        world_velocity = rotation_matrix @ body_velocity
        
        return world_velocity
    
    def world_velocity_to_body(self, world_velocity: np.ndarray, orientation_quat: np.ndarray) -> np.ndarray:
        """Transform world frame velocity to body frame."""
        rotation = Rotation.from_quat(orientation_quat)
        rotation_matrix = rotation.as_matrix().T  # Transpose for inverse
        
        body_velocity = rotation_matrix @ world_velocity
        
        return body_velocity
    
    def get_relative_position(self, position_a: np.ndarray, position_b: np.ndarray) -> Dict[str, float]:
        """
        Get relative position information between two points.
        
        Args:
            position_a: First position
            position_b: Second position
            
        Returns:
            Relative position information
        """
        diff = position_b - position_a
        
        return {
            'distance': float(np.linalg.norm(diff)),
            'horizontal_distance': float(np.linalg.norm(diff[:2])),
            'vertical_distance': float(abs(diff[2])),
            'bearing': float(np.arctan2(diff[1], diff[0])),  # radians
            'elevation': float(np.arctan2(diff[2], np.linalg.norm(diff[:2]))),
            'direction_vector': diff / (np.linalg.norm(diff) + 1e-8)
        }
