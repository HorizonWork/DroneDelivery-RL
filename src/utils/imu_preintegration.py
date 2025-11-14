import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from scipy.spatial.transform import Rotation as R

dataclass
class IMUBias:

    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_bias_cov: np.ndarray = field(default_factory=lambda: np.eye(3)  1e-4)
    gyro_bias_cov: np.ndarray = field(default_factory=lambda: np.eye(3)  1e-6)

dataclass
class PreintegrationResult:

    delta_position: np.ndarray
    delta_velocity: np.ndarray
    delta_rotation: np.ndarray
    covariance: np.ndarray
    dt: float
    num_measurements: int

class IMUPreintegrator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.imu_frequency = config.get("imu_frequency", 200.0)
        self.gravity = config.get("gravity", np.array([0, 0, 9.81]))

        self.accel_noise_std = config.get("accel_noise_std", 0.02)
        self.gyro_noise_std = config.get("gyro_noise_std", 0.0015)
        self.accel_walk_std = config.get("accel_walk_std", 0.0002)
        self.gyro_walk_std = config.get("gyro_walk_std", 0.0001)

        self.bias = IMUBias()
        self.estimate_bias = config.get("estimate_bias", True)

        self.measurements: deque = deque(maxlen=1000)
        self.preintegration_cache: Dict[float, PreintegrationResult] = {}
        self.cache_max_size = config.get("cache_max_size", 100)

        self.is_running = False
        self.last_timestamp = 0.0

        dt = 1.0 / self.imu_frequency
        self.accel_cov = np.eye(3)  (self.accel_noise_std2 / dt)
        self.gyro_cov = np.eye(3)  (self.gyro_noise_std2 / dt)

        self.logger.info("IMU Preintegrator initialized")
        self.logger.info(f"Frequency: {self.imu_frequency}Hz")
        self.logger.info(
            f"Noise - Accel: {self.accel_noise_std}, Gyro: {self.gyro_noise_std}"
        )

    def start(self):

        self.is_running = True
        self.last_timestamp = 0.0
        self.logger.info("IMU preintegration started")

    def stop(self):

        self.is_running = False
        self.logger.info("IMU preintegration stopped")

    def add_measurement(self, imu_data: Dict[str, Any], timestamp: float):

        if not self.is_running:
            return

        if timestamp = self.last_timestamp:
            return

        accel = np.array(imu_data["linear_acceleration"])
        gyro = np.array(imu_data["angular_velocity"])

        accel_corrected = accel - self.bias.accel_bias
        gyro_corrected = gyro - self.bias.gyro_bias

        measurement = {
            "accel": accel_corrected,
            "gyro": gyro_corrected,
            "timestamp": timestamp,
            "dt": timestamp - self.last_timestamp if self.last_timestamp  0 else 0.0,
        }

        self.measurements.append(measurement)
        self.last_timestamp = timestamp

        if len(self.measurements)  100 and len(self.measurements)  50 == 0:
            self._update_bias_estimates()

    def get_preintegration(
        self, end_timestamp: float, start_timestamp: Optional[float] = None
    ) - Optional[PreintegrationResult]:

        cache_key = end_timestamp
        if cache_key in self.preintegration_cache:
            return self.preintegration_cache[cache_key]

        relevant_measurements = []

        for measurement in self.measurements:
            meas_time = measurement["timestamp"]

            if start_timestamp is None:
                if meas_time = end_timestamp:
                    relevant_measurements.append(measurement)
            else:
                if start_timestamp = meas_time = end_timestamp:
                    relevant_measurements.append(measurement)

        if len(relevant_measurements)  2:
            return None

        result = self._preintegrate_measurements(relevant_measurements)

        if len(self.preintegration_cache) = self.cache_max_size:
            oldest_key = min(self.preintegration_cache.keys())
            del self.preintegration_cache[oldest_key]

        self.preintegration_cache[cache_key] = result

        return result

    def _preintegrate_measurements(
        self, measurements: List[Dict[str, Any]]
    ) - PreintegrationResult:

        delta_position = np.zeros(3)
        delta_velocity = np.zeros(3)
        delta_rotation = np.eye(3)

        covariance = np.zeros((9, 9))

        Q_accel = self.accel_cov
        Q_gyro = self.gyro_cov

        total_dt = 0.0

        for i, measurement in enumerate(measurements[:-1]):
            dt = measurements[i + 1]["timestamp"] - measurement["timestamp"]
            if dt = 0:
                continue

            accel = measurement["accel"]
            gyro = measurement["gyro"]

            gyro_magnitude = np.linalg.norm(gyro)
            if gyro_magnitude  1e-8:
                axis = gyro / gyro_magnitude
                angle = gyro_magnitude  dt

                K = np.array(
                    [
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0],
                    ]
                )

                dR = np.eye(3) + np.sin(angle)  K + (1 - np.cos(angle))  K  K
                delta_rotation = delta_rotation  dR

            accel_world = delta_rotation  accel
            delta_velocity += accel_world  dt

            delta_position += delta_velocity  dt + 0.5  accel_world  dt2

            accel_var = np.trace(Q_accel)  dt2
            gyro_var = np.trace(Q_gyro)  dt2

            covariance[0:3, 0:3] += np.eye(3)  accel_var
            covariance[3:6, 3:6] += np.eye(3)  accel_var
            covariance[6:9, 6:9] += np.eye(3)  gyro_var

            total_dt += dt

        result = PreintegrationResult(
            delta_position=delta_position,
            delta_velocity=delta_velocity,
            delta_rotation=delta_rotation,
            covariance=covariance,
            dt=total_dt,
            num_measurements=len(measurements),
        )

        return result

    def _update_bias_estimates(self):

        if not self.estimate_bias or len(self.measurements)  100:
            return

        try:
            recent_measurements = list(self.measurements)[-100:]

            accels = np.array([m["accel"] for m in recent_measurements])
            gyros = np.array([m["gyro"] for m in recent_measurements])

            expected_gravity_magnitude = np.linalg.norm(self.gravity)
            actual_gravity = np.mean(accels, axis=0)
            gravity_error = np.linalg.norm(actual_gravity) - expected_gravity_magnitude

            if abs(gravity_error)  0.5:
                bias_update = actual_gravity  0.01
                self.bias.accel_bias = 0.9  self.bias.accel_bias + 0.1  bias_update

            mean_gyro = np.mean(gyros, axis=0)
            if np.linalg.norm(mean_gyro)  0.01:
                bias_update = mean_gyro  0.01
                self.bias.gyro_bias = 0.9  self.bias.gyro_bias + 0.1  bias_update

            self.logger.debug(
                f"Bias updated - Accel: {self.bias.accel_bias}, Gyro: {self.bias.gyro_bias}"
            )

        except Exception as e:
            self.logger.error(f"Bias update error: {e}")

    def get_current_bias(self) - IMUBias:

        return self.bias

    def get_statistics(self) - Dict[str, Any]:

        return {
            "is_running": self.is_running,
            "measurements_buffered": len(self.measurements),
            "preintegrations_cached": len(self.preintegration_cache),
            "current_bias": {
                "accel": self.bias.accel_bias.tolist(),
                "gyro": self.bias.gyro_bias.tolist(),
            },
            "noise_parameters": {
                "accel_noise": self.accel_noise_std,
                "gyro_noise": self.gyro_noise_std,
                "accel_walk": self.accel_walk_std,
                "gyro_walk": self.gyro_walk_std,
            },
        }

    def reset(self):

        self.measurements.clear()
        self.preintegration_cache.clear()
        self.bias = IMUBias()
        self.last_timestamp = 0.0

        self.logger.debug("IMU Preintegrator reset")
