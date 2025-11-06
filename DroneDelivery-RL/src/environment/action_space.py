"""
Action Space Implementation
Implements 4D continuous action space for body-frame velocity commands.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class ActionConfig:
    """Configuration for action space."""
    # Action dimensions (4D continuous as per report)
    action_dim: int = 4
    
    # Action bounds (body-frame velocities + yaw rate)
    max_translational_velocity: float = 5.0    # m/s (from report: 5 m/s max)
    max_yaw_rate: float = 1.0                  # rad/s
    
    # Action space bounds
    action_low: Tuple[float, ...] = (-5.0, -5.0, -5.0, -1.0)   # [vx_min, vy_min, vz_min, ω_min]
    action_high: Tuple[float, ...] = (5.0, 5.0, 5.0, 1.0)      # [vx_max, vy_max, vz_max, ω_max]
    
    # Safety constraints
    enable_safety_limits: bool = True
    emergency_stop_threshold: float = 10.0     # m/s (absolute velocity limit)

class ActionSpace:
    """
    4D continuous action space implementation.
    Actions: [vx, vy, vz, ω] - body-frame velocity commands + yaw rate.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = ActionConfig(**config.get('action', {}))
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        assert self.config.action_dim == 4, f"Action dimension must be 4, got {self.config.action_dim}"
        assert len(self.config.action_low) == 4, "action_low must have 4 elements"
        assert len(self.config.action_high) == 4, "action_high must have 4 elements"
        
        # Action bounds as numpy arrays
        self.action_low = np.array(self.config.action_low, dtype=np.float32)
        self.action_high = np.array(self.config.action_high, dtype=np.float32)
        
        # Action component names
        self.action_names = ['vx', 'vy', 'vz', 'yaw_rate']
        
        # Action statistics for monitoring
        self.action_history = []
        self.max_history_length = 1000
        
        self.logger.info(f"Action space initialized: {self.config.action_dim}D continuous")
        self.logger.info(f"Bounds: vx=[{self.action_low[0]:.1f}, {self.action_high[0]:.1f}], "
                        f"vy=[{self.action_low[1]:.1f}, {self.action_high[1]:.1f}], "
                        f"vz=[{self.action_low[2]:.1f}, {self.action_high[2]:.1f}], "
                        f"ω=[{self.action_low[3]:.1f}, {self.action_high[3]:.1f}]")
    
    def sample(self) -> np.ndarray:
        """
        Sample random action from action space.
        
        Returns:
            Random 4D action vector
        """
        action = np.random.uniform(self.action_low, self.action_high).astype(np.float32)
        return action
    
    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        Clip action to valid bounds.
        
        Args:
            action: Raw action vector
            
        Returns:
            Clipped action vector
        """
        clipped_action = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        return clipped_action
    
    def apply_safety_constraints(self, action: np.ndarray, 
                                current_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply safety constraints to action.
        
        Args:
            action: Raw action vector [vx, vy, vz, ω]
            current_state: Current drone state (for context-aware constraints)
            
        Returns:
            Safety-constrained action vector
        """
        if not self.config.enable_safety_limits:
            return self.clip_action(action)
        
        safe_action = action.copy()
        
        # 1. Clip to basic bounds
        safe_action = self.clip_action(safe_action)
        
        # 2. Emergency velocity limit
        translational_velocity = np.linalg.norm(safe_action[:3])
        if translational_velocity > self.config.emergency_stop_threshold:
            # Scale down translational velocities
            scale_factor = self.config.emergency_stop_threshold / translational_velocity
            safe_action[:3] *= scale_factor
            self.logger.warning(f"Emergency velocity limit applied: {translational_velocity:.2f} -> "
                              f"{np.linalg.norm(safe_action[:3]):.2f} m/s")
        
        # 3. Context-aware constraints (if state provided)
        if current_state is not None:
            safe_action = self._apply_contextual_constraints(safe_action, current_state)
        
        return safe_action
    
    def _apply_contextual_constraints(self, action: np.ndarray, 
                                    current_state: Dict[str, Any]) -> np.ndarray:
        """
        Apply context-aware safety constraints based on current state.
        
        Args:
            action: Action vector
            current_state: Current drone state
            
        Returns:
            Contextually constrained action
        """
        constrained_action = action.copy()
        
        # 1. Altitude constraints (prevent ground collision)
        current_altitude = current_state.get('position', (0, 0, 0))[2]
        if current_altitude < 1.0 and constrained_action[2] < 0:  # Too low and descending
            constrained_action[2] = max(constrained_action[2], -0.5)  # Limit descent rate
            
        # 2. Battery-aware constraints
        battery_level = current_state.get('battery_level', 1.0)
        if battery_level < 0.2:  # Low battery
            # Reduce maximum velocities to conserve energy
            energy_scale = max(0.5, battery_level * 2)  # Scale from 0.5x to 1.0x
            constrained_action[:3] *= energy_scale
            
        # 3. Obstacle proximity constraints
        if 'nearby_obstacles' in current_state:
            obstacles = current_state['nearby_obstacles']
            if obstacles:
                # Find closest obstacle
                min_distance = min(obstacles)
                if min_distance < 2.0:  # Within 2m of obstacle
                    # Reduce velocity based on proximity
                    safety_scale = max(0.3, min_distance / 2.0)
                    constrained_action[:3] *= safety_scale
        
        return constrained_action
    
    def convert_to_body_frame(self, action: np.ndarray, 
                             current_orientation: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Convert action to body frame (if needed).
        Note: Actions are already in body frame, but this can be used for transformations.
        
        Args:
            action: Action vector [vx, vy, vz, ω]
            current_orientation: Current orientation quaternion [w, x, y, z]
            
        Returns:
            Body-frame action vector
        """
        # Actions are already in body frame as specified in report
        # This method is provided for potential future use or coordinate transformations
        return action.copy()
    
    def validate_action(self, action: np.ndarray) -> bool:
        """
        Validate action vector.
        
        Args:
            action: Action vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check dimension
        if len(action) != self.config.action_dim:
            self.logger.error(f"Invalid action dimension: {len(action)} != {self.config.action_dim}")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            self.logger.error("Action contains NaN or infinite values")
            return False
        
        # Check bounds (with small tolerance)
        tolerance = 1e-6
        if np.any(action < self.action_low - tolerance) or np.any(action > self.action_high + tolerance):
            self.logger.warning(f"Action outside bounds: {action}")
            return False
        
        return True
    
    def get_action_info(self) -> Dict[str, Any]:
        """
        Get information about action space structure.
        
        Returns:
            Dictionary with action space details
        """
        return {
            'dimension': self.config.action_dim,
            'type': 'continuous',
            'components': {
                'vx': {
                    'index': 0,
                    'bounds': [self.action_low[0], self.action_high[0]],
                    'description': 'Forward velocity (body frame) [m/s]'
                },
                'vy': {
                    'index': 1,
                    'bounds': [self.action_low[1], self.action_high[1]],
                    'description': 'Right velocity (body frame) [m/s]'
                },
                'vz': {
                    'index': 2,
                    'bounds': [self.action_low[2], self.action_high[2]],
                    'description': 'Down velocity (body frame) [m/s]'
                },
                'yaw_rate': {
                    'index': 3,
                    'bounds': [self.action_low[3], self.action_high[3]],
                    'description': 'Yaw rate [rad/s]'
                }
            },
            'safety_features': {
                'emergency_stop_threshold': self.config.emergency_stop_threshold,
                'contextual_constraints': self.config.enable_safety_limits,
                'altitude_protection': True,
                'battery_awareness': True,
                'obstacle_avoidance': True
            }
        }
    
    def record_action(self, action: np.ndarray):
        """
        Record action for statistics and monitoring.
        
        Args:
            action: Action vector to record
        """
        self.action_history.append(action.copy())
        
        # Limit history size
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about recent actions.
        
        Returns:
            Dictionary with action statistics
        """
        if not self.action_history:
            return {'message': 'No actions recorded yet'}
        
        actions = np.array(self.action_history)
        
        return {
            'num_actions': len(self.action_history),
            'mean_action': np.mean(actions, axis=0).tolist(),
            'std_action': np.std(actions, axis=0).tolist(),
            'min_action': np.min(actions, axis=0).tolist(),
            'max_action': np.max(actions, axis=0).tolist(),
            'component_names': self.action_names,
            'velocity_statistics': {
                'mean_translational_speed': float(np.mean(np.linalg.norm(actions[:, :3], axis=1))),
                'max_translational_speed': float(np.max(np.linalg.norm(actions[:, :3], axis=1))),
                'mean_yaw_rate': float(np.mean(np.abs(actions[:, 3])))
            }
        }
    
    def reset_statistics(self):
        """Reset action statistics."""
        self.action_history.clear()
        self.logger.info("Action statistics reset")
    
    def is_zero_action(self, action: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Check if action is effectively zero (hover).
        
        Args:
            action: Action vector
            tolerance: Tolerance for zero check
            
        Returns:
            True if action is zero within tolerance
        """
        return np.allclose(action, 0.0, atol=tolerance)
    
    def create_hover_action(self) -> np.ndarray:
        """
        Create hover action (all zeros).
        
        Returns:
            Zero action vector for hovering
        """
        return np.zeros(self.config.action_dim, dtype=np.float32)
    
    def create_emergency_stop_action(self) -> np.ndarray:
        """
        Create emergency stop action.
        
        Returns:
            Emergency stop action vector
        """
        # Emergency stop: zero all velocities
        return self.create_hover_action()
