"""
Coordinate Transforms
Handles all coordinate system transformations for VI-SLAM.
AirSim NED ↔ SLAM coordinate systems.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
from scipy.spatial.transform import Rotation as R
import logging

class CoordinateTransforms:
    """
    Coordinate system transformation utilities.
    Handles conversions between AirSim NED, camera, body, and world frames.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # AirSim spawn location (NED coordinates)
        self.airsim_spawn = np.array(config.get('airsim_spawn', [6000.0, -3000.0, 300.0]))
        
        # Camera extrinsics (camera to body frame)
        self.camera_to_body_translation = np.array(config.get('camera_offset', [0.1, 0.0, -0.05]))
        self.camera_to_body_rotation = R.from_euler('xyz', config.get('camera_rotation', [0, 0, 0]))
        
        # World frame origin (SLAM coordinate system)
        self.world_origin = np.array(config.get('world_origin', [0.0, 0.0, 0.0]))
        
        # Transformation matrices
        self.T_body_camera = self._build_transform_matrix(
            self.camera_to_body_rotation.as_matrix(),
            self.camera_to_body_translation
        )
        self.T_camera_body = np.linalg.inv(self.T_body_camera)
        
        self.logger.info("Coordinate transforms initialized")
        self.logger.info(f"AirSim spawn: {self.airsim_spawn}")
        self.logger.info(f"Camera offset: {self.camera_to_body_translation}")
    
    def _build_transform_matrix(self, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Build 4x4 homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T
    
    def airsim_to_world(self, airsim_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert AirSim NED coordinates to SLAM world coordinates.
        
        Args:
            airsim_pos: Position in AirSim NED frame
            
        Returns:
            Position in SLAM world frame
        """
        # Convert AirSim position relative to spawn
        relative_pos = np.array(airsim_pos) - self.airsim_spawn
        
        # AirSim uses NED (North-East-Down), convert to ENU (East-North-Up) for SLAM
        # NED to ENU: [N,E,D] -> [E,N,-D]
        world_pos = np.array([relative_pos[1], relative_pos[0], -relative_pos[2]])
        
        # Add world origin offset
        world_pos += self.world_origin
        
        return tuple(world_pos.astype(float))
    
    def world_to_airsim(self, world_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert SLAM world coordinates to AirSim NED coordinates.
        
        Args:
            world_pos: Position in SLAM world frame
            
        Returns:
            Position in AirSim NED frame
        """
        # Remove world origin offset
        relative_pos = np.array(world_pos) - self.world_origin
        
        # ENU to NED conversion: [E,N,U] -> [N,E,-U]
        ned_pos = np.array([relative_pos[1], relative_pos[0], -relative_pos[2]])
        
        # Add AirSim spawn offset
        airsim_pos = ned_pos + self.airsim_spawn
        
        return tuple(airsim_pos.astype(float))
    
    def camera_to_body(self, camera_pos: np.ndarray) -> np.ndarray:
        """
        Transform position from camera frame to body frame.
        
        Args:
            camera_pos: Position in camera frame
            
        Returns:
            Position in body frame
        """
        camera_pos_homo = np.append(camera_pos, 1.0)
        body_pos_homo = self.T_body_camera @ camera_pos_homo
        return body_pos_homo[:3]
    
    def body_to_camera(self, body_pos: np.ndarray) -> np.ndarray:
        """
        Transform position from body frame to camera frame.
        
        Args:
            body_pos: Position in body frame
            
        Returns:
            Position in camera frame
        """
        body_pos_homo = np.append(body_pos, 1.0)
        camera_pos_homo = self.T_camera_body @ body_pos_homo
        return camera_pos_homo[:3]
    
    def quaternion_to_euler(self, quat: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles.
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x,y,z,w]
        return tuple(rotation.as_euler('xyz').astype(float))
    
    def euler_to_quaternion(self, euler: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """
        Convert Euler angles to quaternion.
        
        Args:
            euler: Euler angles [roll, pitch, yaw] in radians
            
        Returns:
            Quaternion [w, x, y, z]
        """
        rotation = R.from_euler('xyz', euler)
        quat_scipy = rotation.as_quat()  # [x, y, z, w]
        return (float(quat_scipy[3]), float(quat_scipy[0]), float(quat_scipy[1]), float(quat_scipy[2]))
    
    def transform_pose(self, position: Tuple[float, float, float],
                      orientation: Tuple[float, float, float, float],
                      from_frame: str, to_frame: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """
        Transform pose between coordinate frames.
        
        Args:
            position: Position tuple
            orientation: Quaternion [w, x, y, z]
            from_frame: Source frame ('airsim', 'world', 'camera', 'body')
            to_frame: Target frame
            
        Returns:
            (transformed_position, transformed_orientation)
        """
        if from_frame == to_frame:
            return position, orientation
        
        # Convert position
        if from_frame == 'airsim' and to_frame == 'world':
            new_pos = self.airsim_to_world(position)
        elif from_frame == 'world' and to_frame == 'airsim':
            new_pos = self.world_to_airsim(position)
        else:
            new_pos = position  # Identity for now
        
        # For simplicity, keep orientation unchanged in this implementation
        # Full implementation would handle all frame transformations
        new_orientation = orientation
        
        return new_pos, new_orientation


def transform_pose(position: Tuple[float, float, float],
                   orientation: Tuple[float, float, float, float],
                   from_frame: str = 'airsim',
                   to_frame: str = 'world',
                   config: Optional[Dict[str, Any]] = None
                   ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Convenience wrapper so callers/tests can transform a pose without
    creating a CoordinateTransforms instance explicitly.
    """
    transformer = CoordinateTransforms(config or {})
    return transformer.transform_pose(position, orientation, from_frame, to_frame)
