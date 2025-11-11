"""
Dynamic Obstacles Tracker
Tracks and predicts dynamic obstacle movements for S-RRT planning.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque

@dataclass
class DynamicObstacle:
    """Dynamic obstacle representation."""
    id: str
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    radius: float
    last_update: float
    confidence: float = 1.0
    trajectory_history: deque = field(default_factory=lambda: deque(maxlen=10))

@dataclass
class PredictedPosition:
    """Predicted obstacle position at future time."""
    position: np.ndarray
    time: float
    confidence: float

class DynamicObstacleTracker:
    """
    Tracks dynamic obstacles and predicts future positions.
    Supports obstacle trajectory prediction for safe path planning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Tracking parameters
        self.max_prediction_time = config.get('max_prediction_time', 3.0)    # seconds
        self.velocity_smoothing = config.get('velocity_smoothing', 0.7)     # exponential smoothing
        self.position_noise_std = config.get('position_noise_std', 0.1)     # meters
        
        # Obstacle parameters
        self.default_obstacle_radius = config.get('default_obstacle_radius', 0.3)  # meters
        self.max_velocity = config.get('max_velocity', 2.0)              # m/s (human walking speed)
        self.max_acceleration = config.get('max_acceleration', 1.0)       # m/s²
        
        # Association parameters
        self.max_association_distance = config.get('max_association_distance', 1.0)  # meters
        self.obstacle_timeout = config.get('obstacle_timeout', 2.0)      # seconds
        
        # Storage
        self.obstacles: Dict[str, DynamicObstacle] = {}
        self.next_obstacle_id = 0
        
        # Prediction models
        self.prediction_method = config.get('prediction_method', 'constant_velocity')
        
        self.logger.info("Dynamic Obstacle Tracker initialized")
        self.logger.info(f"Prediction time: {self.max_prediction_time}s")
        self.logger.info(f"Prediction method: {self.prediction_method}")
    
    def update_obstacles(self, detected_positions: List[Tuple[float, float, float]]):
        """
        Update obstacle positions and associate with existing tracks.
        
        Args:
            detected_positions: List of detected obstacle positions
        """
        current_time = time.time()
        
        # Convert to numpy arrays
        detections = [np.array(pos) for pos in detected_positions]
        
        # Associate detections with existing obstacles
        associations = self._associate_detections(detections, current_time)
        
        # Update existing obstacles
        for obstacle_id, detection_idx in associations.items():
            if obstacle_id in self.obstacles and detection_idx < len(detections):
                self._update_obstacle(obstacle_id, detections[detection_idx], current_time)
        
        # Create new obstacles for unassociated detections
        unassociated_detections = set(range(len(detections))) - set(associations.values())
        for detection_idx in unassociated_detections:
            self._create_new_obstacle(detections[detection_idx], current_time)
        
        # Remove timed-out obstacles
        self._remove_stale_obstacles(current_time)
        
        self.logger.debug(f"Updated {len(self.obstacles)} dynamic obstacles")
    
    def _associate_detections(self, detections: List[np.ndarray], 
                            current_time: float) -> Dict[str, int]:
        """
        Associate detections with existing obstacle tracks.
        
        Args:
            detections: List of detection positions
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping obstacle_id to detection_index
        """
        associations = {}
        
        if not detections or not self.obstacles:
            return associations
        
        # Calculate distance matrix
        obstacle_ids = list(self.obstacles.keys())
        distance_matrix = np.full((len(obstacle_ids), len(detections)), float('inf'))
        
        for i, obstacle_id in enumerate(obstacle_ids):
            obstacle = self.obstacles[obstacle_id]
            
            # Predict current position
            dt = current_time - obstacle.last_update
            predicted_pos = self._predict_position(obstacle, dt)
            
            for j, detection in enumerate(detections):
                distance = np.linalg.norm(predicted_pos - detection)
                distance_matrix[i, j] = distance
        
        # Greedy assignment (could use Hungarian algorithm for optimal assignment)
        for _ in range(min(len(obstacle_ids), len(detections))):
            # Find minimum distance
            min_i, min_j = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
            min_distance = distance_matrix[min_i, min_j]
            
            if min_distance <= self.max_association_distance:
                obstacle_id = obstacle_ids[min_i]
                associations[obstacle_id] = min_j
                
                # Remove from consideration
                distance_matrix[min_i, :] = float('inf')
                distance_matrix[:, min_j] = float('inf')
            else:
                break  # No more valid associations
        
        return associations
    
    def _update_obstacle(self, obstacle_id: str, new_position: np.ndarray, 
                        current_time: float):
        """
        Update existing obstacle with new detection.
        
        Args:
            obstacle_id: Obstacle identifier
            new_position: New detected position
            current_time: Current timestamp
        """
        obstacle = self.obstacles[obstacle_id]
        dt = current_time - obstacle.last_update
        
        if dt <= 0:
            return  # Invalid time update
        
        # Calculate new velocity
        new_velocity = (new_position - obstacle.position) / dt
        
        # Apply velocity smoothing and bounds
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)
        obstacle.velocity = (self.velocity_smoothing * obstacle.velocity + 
                           (1 - self.velocity_smoothing) * new_velocity)
        
        # Calculate acceleration
        acceleration = (obstacle.velocity - obstacle.trajectory_history[-1][1] if obstacle.trajectory_history else np.zeros(3)) / dt
        obstacle.acceleration = np.clip(acceleration, -self.max_acceleration, self.max_acceleration)
        
        # Update position
        obstacle.position = new_position.copy()
        obstacle.last_update = current_time
        obstacle.confidence = min(1.0, obstacle.confidence + 0.1)  # Increase confidence
        
        # Add to trajectory history
        obstacle.trajectory_history.append((new_position.copy(), obstacle.velocity.copy(), current_time))
    
    def _create_new_obstacle(self, position: np.ndarray, current_time: float):
        """
        Create new obstacle track.
        
        Args:
            position: Initial position
            current_time: Current timestamp
        """
        obstacle_id = f"DynObs_{self.next_obstacle_id}"
        self.next_obstacle_id += 1
        
        obstacle = DynamicObstacle(
            id=obstacle_id,
            position=position.copy(),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            radius=self.default_obstacle_radius,
            last_update=current_time,
            confidence=0.5  # Lower confidence for new obstacles
        )
        
        obstacle.trajectory_history.append((position.copy(), np.zeros(3), current_time))
        self.obstacles[obstacle_id] = obstacle
        
        self.logger.debug(f"Created new obstacle: {obstacle_id} at {position}")
    
    def _remove_stale_obstacles(self, current_time: float):
        """Remove obstacles that haven't been updated recently."""
        stale_obstacles = []
        
        for obstacle_id, obstacle in self.obstacles.items():
            if current_time - obstacle.last_update > self.obstacle_timeout:
                stale_obstacles.append(obstacle_id)
        
        for obstacle_id in stale_obstacles:
            del self.obstacles[obstacle_id]
            self.logger.debug(f"Removed stale obstacle: {obstacle_id}")
    
    def predict_obstacle_positions(self, prediction_time: float) -> Dict[str, PredictedPosition]:
        """
        Predict obstacle positions at future time.
        
        Args:
            prediction_time: Time in the future (seconds)
            
        Returns:
            Dictionary of predicted positions
        """
        predictions = {}
        
        for obstacle_id, obstacle in self.obstacles.items():
            predicted_pos = self._predict_position(obstacle, prediction_time)
            
            # Confidence decreases with prediction time
            confidence = obstacle.confidence * np.exp(-prediction_time / 2.0)
            
            predictions[obstacle_id] = PredictedPosition(
                position=predicted_pos,
                time=prediction_time,
                confidence=confidence
            )
        
        return predictions
    
    def _predict_position(self, obstacle: DynamicObstacle, dt: float) -> np.ndarray:
        """
        Predict obstacle position after time dt.
        
        Args:
            obstacle: Obstacle to predict
            dt: Time delta for prediction
            
        Returns:
            Predicted position
        """
        if self.prediction_method == 'constant_velocity':
            # Simple constant velocity model
            predicted_pos = obstacle.position + obstacle.velocity * dt
            
        elif self.prediction_method == 'constant_acceleration':
            # Constant acceleration model
            predicted_pos = (obstacle.position + 
                           obstacle.velocity * dt + 
                           0.5 * obstacle.acceleration * dt**2)
            
        elif self.prediction_method == 'trajectory_fitting':
            # Fit trajectory from history (more complex)
            predicted_pos = self._predict_with_trajectory_fitting(obstacle, dt)
            
        else:
            # Fallback to constant position
            predicted_pos = obstacle.position.copy()
        
        return predicted_pos
    
    def _predict_with_trajectory_fitting(self, obstacle: DynamicObstacle, dt: float) -> np.ndarray:
        """
        Predict position using trajectory history fitting.
        
        Args:
            obstacle: Obstacle with trajectory history
            dt: Prediction time
            
        Returns:
            Predicted position
        """
        if len(obstacle.trajectory_history) < 3:
            # Fallback to constant velocity
            return obstacle.position + obstacle.velocity * dt
        
        # Extract recent trajectory
        positions = [entry[0] for entry in obstacle.trajectory_history]
        times = [entry[2] for entry in obstacle.trajectory_history]
        
        # Simple linear extrapolation
        if len(positions) >= 2:
            recent_velocity = (positions[-1] - positions[-2]) / (times[-1] - times[-2])
            predicted_pos = obstacle.position + recent_velocity * dt
        else:
            predicted_pos = obstacle.position + obstacle.velocity * dt
        
        return predicted_pos
    
    def get_obstacles_in_region(self, center: Tuple[float, float, float],
                               radius: float) -> List[DynamicObstacle]:
        """
        Get obstacles within specified region.
        
        Args:
            center: Region center
            radius: Search radius
            
        Returns:
            List of obstacles in region
        """
        center_array = np.array(center)
        obstacles_in_region = []
        
        for obstacle in self.obstacles.values():
            distance = np.linalg.norm(obstacle.position - center_array)
            if distance <= radius:
                obstacles_in_region.append(obstacle)
        
        return obstacles_in_region
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get obstacle tracking statistics."""
        if not self.obstacles:
            return {'num_obstacles': 0}
        
        # Calculate statistics
        positions = [obs.position for obs in self.obstacles.values()]
        velocities = [np.linalg.norm(obs.velocity) for obs in self.obstacles.values()]
        confidences = [obs.confidence for obs in self.obstacles.values()]
        
        return {
            'num_obstacles': len(self.obstacles),
            'average_velocity': float(np.mean(velocities)) if velocities else 0.0,
            'max_velocity': float(np.max(velocities)) if velocities else 0.0,
            'average_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'prediction_method': self.prediction_method,
            'next_id': self.next_obstacle_id
        }
    
    def reset(self):
        """Reset obstacle tracker."""
        self.obstacles.clear()
        self.next_obstacle_id = 0
        self.logger.debug("Dynamic obstacle tracker reset")
