import numpy as np
import logging
import time
import threading
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

dataclass
class ControllerState:

    last_command_time: float = 0.0
    command_history: List[np.ndarray] = None
    is_active: bool = False
    emergency_stop: bool = False

    def __post_init__(self):
        if self.command_history is None:
            self.command_history = []

dataclass
class DroneStatus:

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    is_armed: bool
    is_flying: bool
    battery_level: float
    connection_status: bool

class DroneController:

    def __init__(self, config: Dict[str, Any] = None, airsim_bridge: Any = None):

        self.config = config or {}
        self.airsim_bridge = airsim_bridge
        self.logger = logging.getLogger(__name__)

        self.controller_state = ControllerState()

        self.control_frequency = self.config.get("control_frequency", 20.0)
        self.max_velocity = self.config.get("max_velocity", 5.0)
        self.max_yaw_rate = self.config.get("max_yaw_rate", 1.0)

        self.action_scale = self.config.get("action_scale", 5.0)
        self.expect_normalized_actions = self.config.get("expect_normalized_actions", True)

        self.min_altitude = self.config.get("min_altitude", 0.5)
        self.max_altitude = self.config.get("max_altitude", 20.0)
        self.collision_avoidance = self.config.get("collision_avoidance", True)
        self.emergency_land_altitude = self.config.get("emergency_land_altitude", 0.3)

        self.velocity_smoothing = self.config.get("velocity_smoothing", 0.8)
        self.command_timeout = self.config.get("command_timeout", 0.5)

        self.last_drone_status: Optional[DroneStatus] = None
        self.smoothed_velocity = np.zeros(4)

        self.max_acceleration = self.config.get("max_acceleration", 20.0)
        self.max_jerk = self.config.get("max_jerk", 50.0)

        self.strict_acceleration_check = False
        self.logger.warning("Strict acceleration check DISABLED - actions will be clipped only")

        self.control_thread: Optional[threading.Thread] = None
        self.is_controlling = False
        self.control_lock = threading.Lock()

        self.logger.info("Drone Controller initialized")
        self.logger.info(f"Control frequency: {self.control_frequency} Hz")
        self.logger.info(f"Max velocity: {self.max_velocity} m/s")
        self.logger.info(f"Action scale: {self.action_scale}")
        self.logger.info(f"Max acceleration: {self.max_acceleration} m/s²")
        self.logger.info(f"Strict acceleration check: {self.strict_acceleration_check}")

    def set_airsim_bridge(self, bridge):

        self.airsim_bridge = bridge
        self.logger.info("AirSim bridge connected to drone controller")

    def start_control_loop(self):

        if self.is_controlling:
            return

        self.is_controlling = True
        self.controller_state.is_active = True

        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

        self.logger.info("Control loop started")

    def stop_control_loop(self):

        self.is_controlling = False
        self.controller_state.is_active = False

        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)

        self.logger.info("Control loop stopped")

    def _control_loop(self):

        control_period = 1.0 / self.control_frequency

        while self.is_controlling:
            loop_start = time.time()

            try:
                with self.control_lock:
                    self._update_drone_status()
                    self._safety_checks()
                    self._execute_pending_commands()

                elapsed = time.time() - loop_start
                sleep_time = max(0, control_period - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                time.sleep(control_period)

    def _update_drone_status(self):

        if not self.airsim_bridge or not self.airsim_bridge.is_connected:
            return

        try:
            drone_state = self.airsim_bridge.get_drone_state()
            battery_level = self.airsim_bridge.get_battery_level()

            self.last_drone_status = DroneStatus(
                position=drone_state.position,
                orientation=drone_state.orientation,
                velocity=drone_state.linear_velocity,
                angular_velocity=drone_state.angular_velocity,
                is_armed=True,
                is_flying=abs(drone_state.position[2])  0.1,
                battery_level=battery_level,
                connection_status=self.airsim_bridge.is_alive(),
            )

        except Exception as e:
            self.logger.error(f"Failed to update drone status: {e}")

    def _safety_checks(self):

        if not self.last_drone_status:
            return

        current_altitude = abs(self.last_drone_status.position[2])

        if current_altitude  self.min_altitude and self.last_drone_status.is_flying:
            self.logger.warning(f"Altitude too low: {current_altitude:.2f}m")

        if current_altitude  self.max_altitude:
            self.logger.warning(f"Altitude too high: {current_altitude:.2f}m")
            self.emergency_stop()

        if self.last_drone_status.battery_level  0.1:
            self.logger.error("Critical battery level - initiating emergency landing")
            self.emergency_land()

        if not self.last_drone_status.connection_status:
            self.logger.error("Lost connection to drone")
            self.emergency_stop()

        if self.airsim_bridge and self.airsim_bridge.check_collision():
            self.logger.error("Collision detected - emergency stop")
            self.emergency_stop()

    def execute_action(self, action: np.ndarray):

        if self.expect_normalized_actions:
            scaled_action = action.copy()
            scaled_action[:3] = action[:3]  self.action_scale
            scaled_action[3] = action[3]  self.max_yaw_rate
        else:
            scaled_action = action.copy()

        if not self._validate_action(scaled_action):
            self.logger.warning("Invalid action received - using safe fallback")
            scaled_action = np.zeros(4)

        with self.control_lock:
            safe_action = self._apply_safety_constraints(scaled_action)

            smoothed_action = self._apply_velocity_smoothing(safe_action)

            self._send_velocity_command(smoothed_action)

            self.controller_state.last_command_time = time.time()
            self.controller_state.command_history.append(scaled_action.copy())

            if len(self.controller_state.command_history)  100:
                self.controller_state.command_history.pop(0)

    def _validate_action(self, action: np.ndarray) - bool:

        if len(action) != 4:
            self.logger.warning(f"Invalid action dimensions: {len(action)}")
            return False

        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            self.logger.warning("Action contains NaN or Inf")
            return False

        if np.any(np.abs(action[:3])  self.max_velocity  2.0):
            self.logger.warning(f"Action velocities too high: {action[:3]}")
            return False

        if np.abs(action[3])  self.max_yaw_rate  2.0:
            self.logger.warning(f"Action yaw rate too high: {action[3]}")
            return False

        if self.controller_state.command_history:
            previous_action = self.controller_state.command_history[-1]
            dt = 1.0 / self.control_frequency

            acceleration = (action[:3] - previous_action[:3]) / dt
            accel_magnitude = np.linalg.norm(acceleration)

            if accel_magnitude  self.max_acceleration:
                if self.strict_acceleration_check:
                    self.logger.warning(
                        f"Action exceeds acceleration limits: {acceleration} "
                        f"(magnitude: {accel_magnitude:.2f} m/s²)"
                    )
                    return False
                else:
                    self.logger.debug(
                        f"High acceleration: {accel_magnitude:.2f} m/s² "
                        f"(limit: {self.max_acceleration} m/s²)"
                    )

        return True

    def _apply_safety_constraints(self, action: np.ndarray) - np.ndarray:

        safe_action = action.copy()

        safe_action[:3] = np.clip(
            safe_action[:3], -self.max_velocity, self.max_velocity
        )
        safe_action[3] = np.clip(safe_action[3], -self.max_yaw_rate, self.max_yaw_rate)

        if self.last_drone_status:
            current_altitude = abs(self.last_drone_status.position[2])

            if current_altitude  self.min_altitude and safe_action[2]  0:
                safe_action[2] = min(safe_action[2], -0.1)

            if current_altitude  self.max_altitude and safe_action[2]  0:
                safe_action[2] = max(safe_action[2], 0.1)

        if self.controller_state.emergency_stop:
            safe_action = np.zeros(4)

        return safe_action

    def _apply_velocity_smoothing(self, action: np.ndarray) - np.ndarray:

        alpha = self.velocity_smoothing

        self.smoothed_velocity = alpha  self.smoothed_velocity + (1 - alpha)  action

        return self.smoothed_velocity.copy()

    def _send_velocity_command(self, action: np.ndarray):

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

        if self.controller_state.last_command_time  0:
            time_since_command = time.time() - self.controller_state.last_command_time

            if time_since_command  self.command_timeout:
                self._send_velocity_command(np.zeros(4))

    def emergency_stop(self):

        self.controller_state.emergency_stop = True

        if self.airsim_bridge:
            try:
                self.airsim_bridge.emergency_stop()
                self.logger.error("Emergency stop executed")
            except Exception as e:
                self.logger.error(f"Emergency stop failed: {e}")

    def emergency_land(self):

        self.logger.warning("Initiating emergency landing")

        if self.airsim_bridge:
            try:
                descent_rate = 0.5
                self._send_velocity_command(np.array([0, 0, descent_rate, 0]))

                if self.last_drone_status:
                    current_altitude = abs(self.last_drone_status.position[2])
                    if current_altitude  self.emergency_land_altitude:
                        self.airsim_bridge.land()

            except Exception as e:
                self.logger.error(f"Emergency landing failed: {e}")

    def takeoff(self, altitude: float = 2.0) - bool:

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

    def land(self) - bool:

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

    def get_status(self) - Optional[DroneStatus]:

        return self.last_drone_status

    def get_control_statistics(self) - Dict[str, Any]:

        if not self.controller_state.command_history:
            return {"message": "No commands executed yet"}

        commands = np.array(self.controller_state.command_history)

        return {
            "total_commands": len(self.controller_state.command_history),
            "mean_velocities": np.mean(commands, axis=0).tolist(),
            "max_velocities": np.max(np.abs(commands), axis=0).tolist(),
            "std_velocities": np.std(commands, axis=0).tolist(),
            "emergency_stops": int(self.controller_state.emergency_stop),
            "control_active": self.controller_state.is_active,
            "last_command_age": time.time() - self.controller_state.last_command_time,
        }

    def reset(self):

        with self.control_lock:
            self.controller_state = ControllerState()
            self.smoothed_velocity = np.zeros(4)
            self.last_drone_status = None

        self.logger.debug("Drone controller reset")

    def is_ready(self) - bool:

        return (
            self.airsim_bridge is not None
            and self.airsim_bridge.is_connected
            and self.controller_state.is_active
            and not self.controller_state.emergency_stop
        )

    def __del__(self):

        try:
            if hasattr(self, "controller_state"):
                self.stop_control_loop()
        except Exception:
            pass
