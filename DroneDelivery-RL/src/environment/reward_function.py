"""
Reward Function Implementation
Implements exact Equation (2) from report with all coefficients.
"""

import numpy as np
import logging
import time
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

@dataclass
class RewardConfig:
    """Configuration for reward function (Equation 2)."""
    # Reward coefficients (exact match with Equation 2)
    goal_bonus: float = 500.0          # 500·1{goal}
    distance_penalty: float = 5.0      # -5·d_t
    time_penalty: float = 0.1          # -0.1·Δt
    thrust_penalty: float = 0.01       # -0.01·Σu_i²
    jerk_penalty: float = 10.0         # -10·j_t
    collision_penalty: float = 1000.0  # -1000·c_t
    
    # Physical parameters
    control_dt: float = 0.05           # Δt = 0.05s as specified
    goal_tolerance: float = 0.5        # Goal reached within 0.5m
    num_rotors: int = 4                # Quadrotor configuration
    
    # Energy calculation parameters
    drone_mass: float = 1.5            # kg
    gravity: float = 9.81              # m/s²
    hover_thrust_per_motor: float = 3.675  # N (mass*g/4)

@dataclass
class RewardComponents:
    """Breakdown of reward components for analysis."""
    goal_bonus: float = 0.0
    distance_penalty: float = 0.0
    time_penalty: float = 0.0
    thrust_penalty: float = 0.0
    jerk_penalty: float = 0.0
    collision_penalty: float = 0.0
    total_reward: float = 0.0

class RewardFunction:
    """
    Energy-aware reward function implementation.
    Exactly matches Equation (2): R(s_t, a_t) = 500·1{goal} - 5·d_t - 0.1·Δt - 0.01·Σu_i² - 10·j_t - 1000·c_t
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = RewardConfig(**config.get('reward', {}))
        self.logger = logging.getLogger(__name__)
        
        # State tracking for jerk calculation
        self.previous_velocity_command: Optional[np.ndarray] = None
        self.previous_action_time: Optional[float] = None
        
        # Episode statistics
        self.episode_rewards: List[RewardComponents] = []
        self.cumulative_energy: float = 0.0
        self.episode_start_time: float = 0.0
        
        # Validation bounds
        self.max_reasonable_distance = 100.0  # meters
        self.max_reasonable_thrust = 20.0     # N per motor
        
        self.logger.info("Reward function initialized with Equation (2) coefficients")
        self.logger.info(f"Goal bonus: {self.config.goal_bonus}, "
                        f"Distance penalty: {self.config.distance_penalty}, "
                        f"Time penalty: {self.config.time_penalty}, "
                        f"Thrust penalty: {self.config.thrust_penalty}, "
                        f"Jerk penalty: {self.config.jerk_penalty}, "
                        f"Collision penalty: {self.config.collision_penalty}")
    
    def compute_reward(self, state: Dict[str, Any], action: np.ndarray,
                      next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """
        Compute reward according to Equation (2).
        
        Args:
            state: Current state dictionary
            action: Action taken [vx, vy, vz, yaw_rate]
            next_state: Resulting state
            info: Additional information (collision, goal_reached, etc.)
            
        Returns:
            Scalar reward value
        """
        current_time = time.time()
        components = RewardComponents()
        
        # 1. Goal bonus: 500·1{goal}
        components.goal_bonus = self._compute_goal_bonus(next_state, info)
        
        # 2. Distance penalty: -5·d_t
        components.distance_penalty = self._compute_distance_penalty(next_state)
        
        # 3. Time penalty: -0.1·Δt
        components.time_penalty = self._compute_time_penalty()
        
        # 4. Thrust penalty: -0.01·Σu_i²
        components.thrust_penalty = self._compute_thrust_penalty(action, state)
        
        # 5. Jerk penalty: -10·j_t
        components.jerk_penalty = self._compute_jerk_penalty(action, current_time)
        
        # 6. Collision penalty: -1000·c_t
        components.collision_penalty = self._compute_collision_penalty(info)
        
        # Total reward (Equation 2)
        components.total_reward = (
            components.goal_bonus +
            components.distance_penalty +
            components.time_penalty +
            components.thrust_penalty +
            components.jerk_penalty +
            components.collision_penalty
        )
        
        # Store for analysis
        self.episode_rewards.append(components)
        
        # Update energy tracking
        self.cumulative_energy += abs(components.thrust_penalty / self.config.thrust_penalty)
        
        return float(components.total_reward)
    
    def _compute_goal_bonus(self, next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """
        Compute goal reaching bonus: 500·1{goal}
        
        Args:
            next_state: Next state after action
            info: Episode information
            
        Returns:
            Goal bonus (500.0 if goal reached, 0.0 otherwise)
        """
        # Check if goal reached
        goal_reached = info.get('goal_reached', False)
        
        if goal_reached:
            self.logger.info("Goal reached! Applying goal bonus.")
            return self.config.goal_bonus
        
        # Alternative: check distance to goal
        if 'goal_position' in next_state and 'position' in next_state:
            current_pos = np.array(next_state['position'])
            goal_pos = np.array(next_state['goal_position'])
            distance_to_goal = np.linalg.norm(goal_pos - current_pos)
            
            if distance_to_goal <= self.config.goal_tolerance:
                self.logger.info(f"Goal reached by distance: {distance_to_goal:.3f}m <= {self.config.goal_tolerance}m")
                return self.config.goal_bonus
        
        return 0.0
    
    def _compute_distance_penalty(self, next_state: Dict[str, Any]) -> float:
        """
        Compute distance penalty: -5·d_t
        
        Args:
            next_state: Next state after action
            
        Returns:
            Distance penalty (negative value)
        """
        if 'goal_position' not in next_state or 'position' not in next_state:
            return 0.0
        
        current_pos = np.array(next_state['position'])
        goal_pos = np.array(next_state['goal_position'])
        distance_to_goal = np.linalg.norm(goal_pos - current_pos)
        
        # Clamp distance to reasonable bounds
        distance_to_goal = min(distance_to_goal, self.max_reasonable_distance)
        
        penalty = -self.config.distance_penalty * distance_to_goal
        return float(penalty)
    
    def _compute_time_penalty(self) -> float:
        """
        Compute time penalty: -0.1·Δt
        
        Returns:
            Time step penalty (negative value)
        """
        penalty = -self.config.time_penalty * self.config.control_dt
        return float(penalty)
    
    def _compute_thrust_penalty(self, action: np.ndarray, state: Dict[str, Any]) -> float:
        """
        Compute thrust penalty: -0.01·Σu_i²
        
        Args:
            action: Velocity command [vx, vy, vz, yaw_rate]
            state: Current state for thrust calculation
            
        Returns:
            Thrust penalty (negative value)
        """
        # Convert velocity commands to motor thrusts
        motor_thrusts = self._velocity_to_thrust(action, state)
        
        # Calculate thrust squared sum
        thrust_squared_sum = np.sum(motor_thrusts ** 2)
        
        # Apply penalty coefficient
        penalty = -self.config.thrust_penalty * thrust_squared_sum
        return float(penalty)
    
    def _velocity_to_thrust(self, velocity_command: np.ndarray, 
                          state: Dict[str, Any]) -> np.ndarray:
        """
        Convert body-frame velocity commands to motor thrust estimates.
        
        Args:
            velocity_command: [vx, vy, vz, yaw_rate]
            state: Current state for calculation
            
        Returns:
            Estimated motor thrusts [N] for 4 rotors
        """
        vx, vy, vz, yaw_rate = velocity_command
        
        # Simplified thrust model (real implementation would use full dynamics)
        # Assume thrust proportional to desired acceleration + gravity compensation
        
        # Gravity compensation (hover thrust)
        hover_thrust = self.config.hover_thrust_per_motor
        
        # Acceleration-based thrust estimation
        # This is simplified - real model would consider full quadrotor dynamics
        # For now, we'll use the velocity as a proxy for required acceleration
        # In a real implementation, we would need previous velocity to compute acceleration
        accel_x = vx  # Using velocity as a proxy for acceleration demand
        accel_y = vy
        accel_z = vz  # Keep z velocity separate for gravity compensation
        
        # Convert accelerations to thrust contributions
        thrust_x = self.config.drone_mass * accel_x / self.config.num_rotors
        thrust_y = self.config.drone_mass * accel_y / self.config.num_rotors
        thrust_z = self.config.drone_mass * (accel_z + self.config.gravity) / self.config.num_rotors  # Add gravity compensation to z-thrust
        
        # Distribute thrust among 4 motors (simplified quadrotor model)
        # Front-left, Front-right, Rear-left, Rear-right
        motor_thrusts = np.array([
            hover_thrust + thrust_z + thrust_x - thrust_y,  # Front-left
            hover_thrust + thrust_z + thrust_x + thrust_y,  # Front-right  
            hover_thrust + thrust_z - thrust_x - thrust_y,  # Rear-left
            hover_thrust + thrust_z - thrust_x + thrust_y   # Rear-right
        ])
        
        # Clamp to reasonable bounds
        motor_thrusts = np.clip(motor_thrusts, 0.0, self.max_reasonable_thrust)
        
        return motor_thrusts
    
    def _compute_jerk_penalty(self, action: np.ndarray, current_time: float) -> float:
        """
        Compute jerk penalty: -10·j_t
        
        Args:
            action: Current velocity command [vx, vy, vz, yaw_rate]
            current_time: Current timestamp
            
        Returns:
            Jerk penalty (negative value)
        """
        if self.previous_velocity_command is None or self.previous_action_time is None:
            # First action - no jerk
            self.previous_velocity_command = action.copy()
            self.previous_action_time = current_time
            return 0.0
        
        # Calculate time difference
        dt = current_time - self.previous_action_time
        if dt <= 0:
            dt = self.config.control_dt  # Fallback to expected dt
        
        # Calculate velocity derivatives (jerk components)
        velocity_derivative = (action - self.previous_velocity_command) / dt
        
        # Jerk magnitude squared
        jerk_squared = np.sum(velocity_derivative ** 2)
        
        # Apply penalty coefficient
        penalty = -self.config.jerk_penalty * jerk_squared
        
        # Update previous values
        self.previous_velocity_command = action.copy()
        self.previous_action_time = current_time
        
        return float(penalty)
    
    def _compute_collision_penalty(self, info: Dict[str, Any]) -> float:
        """
        Compute collision penalty: -1000·c_t
        
        Args:
            info: Episode information
            
        Returns:
            Collision penalty (negative value if collision occurred)
        """
        collision_occurred = info.get('collision', False)
        
        if collision_occurred:
            self.logger.warning("Collision detected! Applying collision penalty.")
            return -self.config.collision_penalty
        
        return 0.0
    
    def reset_episode(self):
        """Reset episode-specific tracking variables."""
        self.previous_velocity_command = None
        self.previous_action_time = None
        self.episode_rewards.clear()
        self.cumulative_energy = 0.0
        self.episode_start_time = time.time()
        
        self.logger.debug("Reward function reset for new episode")
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for current episode.
        
        Returns:
            Dictionary with reward component statistics
        """
        if not self.episode_rewards:
            return {'message': 'No rewards recorded yet'}
        
        # Convert to numpy arrays for easier analysis
        components = {
            'goal_bonus': [r.goal_bonus for r in self.episode_rewards],
            'distance_penalty': [r.distance_penalty for r in self.episode_rewards],
            'time_penalty': [r.time_penalty for r in self.episode_rewards],
            'thrust_penalty': [r.thrust_penalty for r in self.episode_rewards],
            'jerk_penalty': [r.jerk_penalty for r in self.episode_rewards],
            'collision_penalty': [r.collision_penalty for r in self.episode_rewards],
            'total_reward': [r.total_reward for r in self.episode_rewards]
        }
        
        stats = {}
        for name, values in components.items():
            if values:
                stats[name] = {
                    'sum': float(np.sum(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Additional episode statistics
        stats['episode_info'] = {
            'num_steps': len(self.episode_rewards),
            'total_reward': sum(r.total_reward for r in self.episode_rewards),
            'cumulative_energy': self.cumulative_energy,
            'episode_duration': time.time() - self.episode_start_time,
            'goal_reached': any(r.goal_bonus > 0 for r in self.episode_rewards),
            'collision_occurred': any(r.collision_penalty < 0 for r in self.episode_rewards)
        }
        
        return stats
    
    def validate_reward_config(self) -> bool:
        """
        Validate reward function configuration.
        
        Returns:
            True if configuration is valid
        """
        # Check positive coefficients where expected
        if self.config.goal_bonus <= 0:
            self.logger.error("Goal bonus must be positive")
            return False
        
        if self.config.distance_penalty <= 0:
            self.logger.error("Distance penalty coefficient must be positive")
            return False
        
        if self.config.collision_penalty <= 0:
            self.logger.error("Collision penalty coefficient must be positive")
            return False
        
        # Check control timestep
        if self.config.control_dt <= 0 or self.config.control_dt > 1.0:
            self.logger.error(f"Invalid control timestep: {self.config.control_dt}")
            return False
        
        # Check physical parameters
        if self.config.drone_mass <= 0:
            self.logger.error("Drone mass must be positive")
            return False
        
        if self.config.num_rotors != 4:
            self.logger.warning("Reward function designed for quadrotor (4 rotors)")
        
        return True
    
    def get_reward_info(self) -> Dict[str, Any]:
        """
        Get information about reward function configuration.
        
        Returns:
            Dictionary with reward function details
        """
        return {
            'equation': "R(s_t, a_t) = 500·1{goal} - 5·d_t - 0.1·Δt - 0.01·Σu_i² - 10·j_t - 1000·c_t",
            'coefficients': {
                'goal_bonus': self.config.goal_bonus,
                'distance_penalty': self.config.distance_penalty,
                'time_penalty': self.config.time_penalty,
                'thrust_penalty': self.config.thrust_penalty,
                'jerk_penalty': self.config.jerk_penalty,
                'collision_penalty': self.config.collision_penalty
            },
            'parameters': {
                'control_dt': self.config.control_dt,
                'goal_tolerance': self.config.goal_tolerance,
                'drone_mass': self.config.drone_mass,
                'num_rotors': self.config.num_rotors
            },
            'components': {
                'goal_bonus': 'Large positive reward for reaching target',
                'distance_penalty': 'Penalizes current distance to goal',
                'time_penalty': 'Penalizes elapsed time per step',
                'thrust_penalty': 'Penalizes energy consumption (thrust squared)',
                'jerk_penalty': 'Penalizes sudden velocity changes',
                'collision_penalty': 'Large negative reward for collisions'
            }
        }
