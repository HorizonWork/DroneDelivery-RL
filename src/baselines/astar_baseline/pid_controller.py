"""
PID Controller for following A* waypoints.
Implements classic PID control for baseline comparison.
"""

import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class PIDGains:
    """PID controller gains."""

    kp: float
    ki: float
    kd: float


class PIDController:
    """
    3D PID controller for drone waypoint following.
    Outputs velocity commands to track A* waypoints.
    """

    def __init__(self, config: Dict[str, Any]):
        # PID gains for position control
        self.position_gains = PIDGains(
            kp=config.get("position_kp", 2.0),
            ki=config.get("position_ki", 0.1),
            kd=config.get("position_kd", 0.5),
        )

        # PID gains for yaw control
        self.yaw_gains = PIDGains(
            kp=config.get("yaw_kp", 1.5),
            ki=config.get("yaw_ki", 0.05),
            kd=config.get("yaw_kd", 0.3),
        )

        # Control limits
        self.max_velocity = config.get("max_velocity", 5.0)  # m/s
        self.max_yaw_rate = config.get("max_yaw_rate", 1.0)  # rad/s

        # PID state for position control (x, y, z)
        self.position_errors = np.zeros(3)
        self.position_integral = np.zeros(3)
        self.position_derivative = np.zeros(3)
        self.previous_position_error = np.zeros(3)

        # PID state for yaw control
        self.yaw_error = 0.0
        self.yaw_integral = 0.0
        self.yaw_derivative = 0.0
        self.previous_yaw_error = 0.0

        # Timing
        self.dt = config.get("control_dt", 0.05)  # 20Hz control

        # Integral windup limits
        self.integral_limit = config.get("integral_limit", 10.0)

    def reset(self):
        """Reset PID controller state."""
        self.position_errors.fill(0.0)
        self.position_integral.fill(0.0)
        self.position_derivative.fill(0.0)
        self.previous_position_error.fill(0.0)

        self.yaw_error = 0.0
        self.yaw_integral = 0.0
        self.yaw_derivative = 0.0
        self.previous_yaw_error = 0.0

    def compute_control(
        self,
        current_pos: Tuple[float, float, float],
        current_yaw: float,
        target_pos: Tuple[float, float, float],
        target_yaw: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """
        Compute PID control commands.

        Args:
            current_pos: Current (x, y, z) position
            current_yaw: Current yaw angle (radians)
            target_pos: Target (x, y, z) position
            target_yaw: Target yaw angle (radians)

        Returns:
            (vx, vy, vz, yaw_rate): Body-frame velocity commands + yaw rate
        """
        # Position errors
        current_pos_array = np.array(current_pos)
        target_pos_array = np.array(target_pos)
        self.position_errors = target_pos_array - current_pos_array

        # Position PID computation
        self.position_integral += self.position_errors * self.dt

        # Anti-windup: clamp integral terms
        self.position_integral = np.clip(
            self.position_integral, -self.integral_limit, self.integral_limit
        )

        self.position_derivative = (
            self.position_errors - self.previous_position_error
        ) / self.dt

        # Position PID output (world frame)
        position_output = (
            self.position_gains.kp * self.position_errors
            + self.position_gains.ki * self.position_integral
            + self.position_gains.kd * self.position_derivative
        )

        # Convert to body frame velocities
        # For simplicity, assume body frame aligned with world frame
        # In practice, would need rotation matrix transformation
        vx, vy, vz = position_output

        # Clamp velocities
        velocity_magnitude = np.linalg.norm([vx, vy, vz])
        if velocity_magnitude > self.max_velocity:
            scale = self.max_velocity / velocity_magnitude
            vx *= scale
            vy *= scale
            vz *= scale

        # Yaw control
        self.yaw_error = self._normalize_angle(target_yaw - current_yaw)
        self.yaw_integral += self.yaw_error * self.dt
        self.yaw_integral = np.clip(
            self.yaw_integral, -self.integral_limit, self.integral_limit
        )

        self.yaw_derivative = (self.yaw_error - self.previous_yaw_error) / self.dt

        yaw_rate = (
            self.yaw_gains.kp * self.yaw_error
            + self.yaw_gains.ki * self.yaw_integral
            + self.yaw_gains.kd * self.yaw_derivative
        )

        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # Update previous errors
        self.previous_position_error = self.position_errors.copy()
        self.previous_yaw_error = self.yaw_error

        return (float(vx), float(vy), float(vz), float(yaw_rate))

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def get_status(self) -> Dict[str, Any]:
        """Get controller status for debugging."""
        return {
            "position_error": self.position_errors.tolist(),
            "position_integral": self.position_integral.tolist(),
            "position_derivative": self.position_derivative.tolist(),
            "yaw_error": self.yaw_error,
            "yaw_integral": self.yaw_integral,
            "yaw_derivative": self.yaw_derivative,
        }
