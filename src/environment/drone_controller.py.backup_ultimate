"""
Drone Controller
Low-level drone control interface for AirSim integration.
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

@dataclass
class ControllerState:
    """Internal controller state."""
    last_command_time: float = 0.0
    command_history: List[np.ndarray] = None
    is_active: bool = False
    emergency_stop: bool = False
    
    def __post_init__(self):
        if self.command_history is None:
            self.command_history = []

@dataclass
class DroneStatus:
    """Drone status information."""
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    is_armed: bool
    is_flying: bool
    battery_level: float
    connection_status: bool

class DroneController:
    """
    Low-level drone control interface.
    Converts high-level actions to AirSim commands with safety features.
    """
    
    def __init__(self, config: Dict[str, Any] = None, 
             airsim_bridge: Any = None):
        """Initialize drone controller."""
        self.config = config or {}  # ← THÊM config!
        self.airsim_bridge = airsim_bridge
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize controller_state to fix __del__ error
        from dataclasses import dataclass
        
        @dataclass
        class ControllerState:
            is_active: bool = False
        
        self.controller_state = ControllerState()
        self.control_frequency = config.get('control_frequency', 20.0)  # Hz
        self.max_velocity = config.get('max_velocity', 5.0)             # m/s
        self.max_yaw_rate = config.get('max_yaw_rate', 1.0)             # rad/s
        
        # Safety parameters
        self.min_altitude = config.get('min_altitude', 0.5)             # meters
        self.max_altitude = config.get('max_altitude', 20.0)            # meters
        self.collision_avoidance = config.get('collision_avoidance', True)
        self.emergency_land_altitude = config.get('emergency_land_altitude', 0.3)
        
        # Control smoothing
        self.velocity_smoothing = config.get('velocity_smoothing', 0.8)
        self.command_timeout = config.get('command_timeout', 0.5)       # seconds
        
        # Internal state
        self.controller_state = ControllerState()
        self.last_drone_status: Optional[DroneStatus] = None
        self.smoothed_velocity = np.zeros(4)  # [vx, vy, vz, yaw_rate]
        
        # Command validation
        self.max_acceleration = config.get('max_acceleration', 10.0)    # m/s²
        self.max_jerk = config.get('max_jerk', 20.0)                   # m/s³
        
        # Threading for real-time control
        self.control_thread: Optional[threading.Thread] = None
        self.is_controlling = False
        self.control_lock = threading.Lock()
        
        # AirSim bridge reference (set by environment)
        self.airsim_bridge = None
        
        self.logger.info("Drone Controller initialized")
        self.logger.info(f"Control frequency: {self.control_frequency} Hz")
        self.logger.info(f"Max velocity: {self.max_velocity} m/s")
        self.logger.info(f"Safety features: collision_avoidance={self.collision_avoidance}")
    
    def set_airsim_bridge(self, bridge):
        """Set AirSim bridge reference."""
        self.airsim_bridge = bridge
        self.logger.info("AirSim bridge connected to drone controller")
    
    def start_control_loop(self):
        """Start real-time control loop."""
        if self.is_controlling:
            return
        
        self.is_controlling = True
        self.controller_state.is_active = True
        
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        self.logger.info("Control loop started")
    
    def stop_control_loop(self):
        """Stop real-time control loop."""
        self.is_controlling = False
        self.controller_state.is_active = False
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        
        self.logger.info("Control loop stopped")
    
    def _control_loop(self):
        """Main control loop running at specified frequency."""
        control_period = 1.0 / self.control_frequency
        
        while self.is_controlling:
            loop_start = time.time()
            
            try:
                with self.control_lock:
                    self._update_drone_status()
                    self._safety_checks()
                    self._execute_pending_commands()
                
                # Maintain control frequency
                elapsed = time.time() - loop_start
                sleep_time = max(0, control_period - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                time.sleep(control_period)
    
    def _update_drone_status(self):
        """Update internal drone status."""
        if not self.airsim_bridge or not self.airsim_bridge.is_connected:
            return
        
        try:
            # Get drone state from AirSim
            drone_state = self.airsim_bridge.get_drone_state()
            battery_level = self.airsim_bridge.get_battery_level()
            
            self.last_drone_status = DroneStatus(
                position=drone_state.position,
                orientation=drone_state.orientation,
                velocity=drone_state.linear_velocity,
                angular_velocity=drone_state.angular_velocity,
                is_armed=True,  # Assume armed if connected
                is_flying=abs(drone_state.position[2]) > 0.1,  # Z > 0.1m
                battery_level=battery_level,
                connection_status=self.airsim_bridge.is_alive()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update drone status: {e}")
    
    def _safety_checks(self):
        """Perform safety checks and emergency actions."""
        if not self.last_drone_status:
            return
        
        # Check altitude limits
        current_altitude = abs(self.last_drone_status.position[2])
        
        if current_altitude < self.min_altitude and self.last_drone_status.is_flying:
            self.logger.warning(f"Altitude too low: {current_altitude:.2f}m")
            # Could implement emergency altitude correction here
        
        if current_altitude > self.max_altitude:
            self.logger.warning(f"Altitude too high: {current_altitude:.2f}m")
            self.emergency_stop()
        
        # Check battery level
        if self.last_drone_status.battery_level < 0.1:
            self.logger.error("Critical battery level - initiating emergency landing")
            self.emergency_land()
        
        # Check connection
        if not self.last_drone_status.connection_status:
            self.logger.error("Lost connection to drone")
            self.emergency_stop()
        
        # Check collision
        if self.airsim_bridge and self.airsim_bridge.check_collision():
            self.logger.error("Collision detected - emergency stop")
            self.emergency_stop()
    
    def execute_action(self, action: np.ndarray):
        """
        Execute high-level action command.
        
        Args:
            action: 4D action vector [vx, vy, vz, yaw_rate]
        """
        if not self._validate_action(action):
            self.logger.warning("Invalid action received - using safe fallback")
            action = np.zeros(4)  # Hover command
        
        with self.control_lock:
            # Apply safety constraints
            safe_action = self._apply_safety_constraints(action)
            
            # Apply smoothing
            smoothed_action = self._apply_velocity_smoothing(safe_action)
            
            # Convert to AirSim command
            self._send_velocity_command(smoothed_action)
            
            # Update state
            self.controller_state.last_command_time = time.time()
            self.controller_state.command_history.append(action.copy())
            
            # Limit history size
            if len(self.controller_state.command_history) > 100:
                self.controller_state.command_history.pop(0)
    
    def _validate_action(self, action: np.ndarray) -> bool:
        """
        Validate action vector.
        
        Args:
            action: Action to validate
            
        Returns:
            True if valid
        """
        # Check dimensions
        if len(action) != 4:
            return False
        
        # Check for NaN/inf
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            return False
        
        # Check bounds
        if (np.any(np.abs(action[:3]) > self.max_velocity) or 
            np.abs(action[3]) > self.max_yaw_rate):
            return False
        
        # Check acceleration limits (if we have previous command)
        if self.controller_state.command_history:
            previous_action = self.controller_state.command_history[-1]
            dt = 1.0 / self.control_frequency
            
            acceleration = (action[:3] - previous_action[:3]) / dt
            if np.any(np.abs(acceleration) > self.max_acceleration):
                self.logger.warning(f"Action exceeds acceleration limits: {acceleration}")
                return False
        
        return True
    
    def _apply_safety_constraints(self, action: np.ndarray) -> np.ndarray:
        """
        Apply safety constraints to action.
        
        Args:
            action: Raw action
            
        Returns:
            Safety-constrained action
        """
        safe_action = action.copy()
        
        # Basic velocity limits
        safe_action[:3] = np.clip(safe_action[:3], -self.max_velocity, self.max_velocity)
        safe_action[3] = np.clip(safe_action[3], -self.max_yaw_rate, self.max_yaw_rate)
        
        # Altitude-based constraints
        if self.last_drone_status:
            current_altitude = abs(self.last_drone_status.position[2])
            
            # Prevent descent if too low
            if current_altitude < self.min_altitude and safe_action[2] > 0:  # Descending (positive Z in NED)
                safe_action[2] = min(safe_action[2], -0.1)  # Force slight ascent
            
            # Prevent ascent if too high
            if current_altitude > self.max_altitude and safe_action[2] < 0:  # Ascending (negative Z in NED)
                safe_action[2] = max(safe_action[2], 0.1)   # Force slight descent
        
        # Emergency stop override
        if self.controller_state.emergency_stop:
            safe_action = np.zeros(4)  # Full stop
        
        return safe_action
    
    def _apply_velocity_smoothing(self, action: np.ndarray) -> np.ndarray:
        """
        Apply exponential smoothing to velocity commands.
        
        Args:
            action: Raw action
            
        Returns:
            Smoothed action
        """
        alpha = self.velocity_smoothing
        
        # Exponential moving average
        self.smoothed_velocity = alpha * self.smoothed_velocity + (1 - alpha) * action
        
        return self.smoothed_velocity.copy()
    
    def _send_velocity_command(self, action: np.ndarray):
        """
        Send velocity command to AirSim.
        
        Args:
            action: Smoothed action [vx, vy, vz, yaw_rate]
        """
        if not self.airsim_bridge or not self.airsim_bridge.is_connected:
            return
        
        try:
            vx, vy, vz, yaw_rate = action
            duration = 1.0 / self.control_frequency
            
            self.airsim_bridge.send_velocity_command(
                float(vx), float(vy), float(vz), float(yaw_rate), duration
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send velocity command: {e}")
    
    def _execute_pending_commands(self):
        """Execute any pending special commands."""
        # Check for command timeout
        if self.controller_state.last_command_time > 0:
            time_since_command = time.time() - self.controller_state.last_command_time
            
            if time_since_command > self.command_timeout:
                # Send hover command if no recent action
                self._send_velocity_command(np.zeros(4))
    
    def emergency_stop(self):
        """Immediate emergency stop."""
        self.controller_state.emergency_stop = True
        
        if self.airsim_bridge:
            try:
                self.airsim_bridge.emergency_stop()
                self.logger.error("Emergency stop executed")
            except Exception as e:
                self.logger.error(f"Emergency stop failed: {e}")
    
    def emergency_land(self):
        """Emergency landing sequence."""
        self.logger.warning("Initiating emergency landing")
        
        if self.airsim_bridge:
            try:
                # Gradual descent
                descent_rate = 0.5  # m/s
                self._send_velocity_command(np.array([0, 0, descent_rate, 0]))
                
                # Monitor altitude and land when low enough
                if self.last_drone_status:
                    current_altitude = abs(self.last_drone_status.position[2])
                    if current_altitude < self.emergency_land_altitude:
                        self.airsim_bridge.land()
                        
            except Exception as e:
                self.logger.error(f"Emergency landing failed: {e}")
    
    def takeoff(self, altitude: float = 2.0) -> bool:
        """
        Takeoff to specified altitude.
        
        Args:
            altitude: Target altitude
            
        Returns:
            Success status
        """
        if not self.airsim_bridge:
            return False
        
        self.logger.info(f"Taking off to {altitude}m altitude")
        
        try:
            success = self.airsim_bridge.takeoff(altitude)
            if success:
                self.controller_state.emergency_stop = False
                self.start_control_loop()
            return success
            
        except Exception as e:
            self.logger.error(f"Takeoff failed: {e}")
            return False
    
    def land(self) -> bool:
        """
        Land the drone safely.
        
        Returns:
            Success status
        """
        if not self.airsim_bridge:
            return False
        
        self.logger.info("Landing drone")
        
        try:
            self.stop_control_loop()
            success = self.airsim_bridge.land()
            return success
            
        except Exception as e:
            self.logger.error(f"Landing failed: {e}")
            return False
    
    def get_status(self) -> Optional[DroneStatus]:
        """
        Get current drone status.
        
        Returns:
            Current drone status
        """
        return self.last_drone_status
    
    def get_control_statistics(self) -> Dict[str, Any]:
        """
        Get control statistics.
        
        Returns:
            Control statistics dictionary
        """
        if not self.controller_state.command_history:
            return {'message': 'No commands executed yet'}
        
        commands = np.array(self.controller_state.command_history)
        
        return {
            'total_commands': len(self.controller_state.command_history),
            'mean_velocities': np.mean(commands, axis=0).tolist(),
            'max_velocities': np.max(np.abs(commands), axis=0).tolist(),
            'std_velocities': np.std(commands, axis=0).tolist(),
            'emergency_stops': int(self.controller_state.emergency_stop),
            'control_active': self.controller_state.is_active,
            'last_command_age': time.time() - self.controller_state.last_command_time
        }
    
    def reset(self):
        """Reset controller state for new episode."""
        with self.control_lock:
            self.controller_state = ControllerState()
            self.smoothed_velocity = np.zeros(4)
            self.last_drone_status = None
        
        self.logger.debug("Drone controller reset")
    
    def is_ready(self) -> bool:
        """
        Check if controller is ready for commands.
        
        Returns:
            True if ready
        """
        return (self.airsim_bridge is not None and 
                self.airsim_bridge.is_connected and
                self.controller_state.is_active and
                not self.controller_state.emergency_stop)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, "controller_state"):  # ← CHECK TRƯỚC KHI DÙNG!
                self.stop_control_loop()
        except Exception as e:
            pass  # Ignore errors during cleanup