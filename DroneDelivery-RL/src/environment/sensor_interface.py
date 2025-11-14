import numpy as np
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque

dataclass
class SensorData:

    timestamp: float

    stereo_left: Optional[np.ndarray] = None
    stereo_right: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None

    imu_data: Optional[Dict[str, Any]] = None

    occupancy_histogram: Optional[np.ndarray] = None
    depth_features: Optional[Dict[str, float]] = None

    data_quality: float = 1.0
    processing_latency: float = 0.0

class SensorInterface:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.camera_frequency = config.get('camera_frequency', 30.0)
        self.imu_frequency = config.get('imu_frequency', 200.0)
        self.occupancy_frequency = config.get('occupancy_frequency', 20.0)

        self.latest_data: Optional[SensorData] = None
        self.data_history: deque = deque(maxlen=100)

        self.frame_drops = 0
        self.processing_times: List[float] = []
        self.max_processing_history = 1000

        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.data_lock = threading.Lock()

        self.data_callbacks: List[Callable[[SensorData], None]] = []

        self.airsim_bridge = None
        self.sensor_bridge = None

        self.logger.info("Sensor Interface initialized")
        self.logger.info(f"Frequencies - Camera: {self.camera_frequency}Hz, "
                        f"IMU: {self.imu_frequency}Hz, "
                        f"Occupancy: {self.occupancy_frequency}Hz")

    def set_bridges(self, airsim_bridge, sensor_bridge):

        self.airsim_bridge = airsim_bridge
        self.sensor_bridge = sensor_bridge
        self.logger.info("Sensor bridges connected")

    def start(self):

        if self.is_running:
            return

        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.logger.info("Sensor interface started")

    def stop(self):

        self.is_running = False

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        self.logger.info("Sensor interface stopped")

    def _processing_loop(self):

        self.logger.info("Sensor processing loop started")

        while self.is_running:
            try:
                self._collect_and_process_data()

                time.sleep(1.0 / self.camera_frequency)

            except Exception as e:
                self.logger.error(f"Sensor processing error: {e}")
                time.sleep(0.1)

        self.logger.info("Sensor processing loop stopped")

    def _collect_and_process_data(self):

        processing_start = time.time()

        raw_data = self._collect_raw_data()
        if not raw_data:
            self.frame_drops += 1
            return

        processed_data = self._process_sensor_data(raw_data)

        processing_time = time.time() - processing_start
        processed_data.processing_latency = processing_time

        with self.data_lock:
            self.latest_data = processed_data
            self.data_history.append(processed_data)

        self.processing_times.append(processing_time)
        if len(self.processing_times)  self.max_processing_history:
            self.processing_times.pop(0)

        for callback in self.data_callbacks:
            try:
                callback(processed_data)
            except Exception as e:
                self.logger.error(f"Sensor callback error: {e}")

    def _collect_raw_data(self) - Optional[Dict[str, Any]]:

        if not self.airsim_bridge or not self.airsim_bridge.is_connected:
            return None

        try:
            sensor_data = self.airsim_bridge.get_sensor_data()

            return {
                'stereo_left': sensor_data.stereo_left,
                'stereo_right': sensor_data.stereo_right,
                'depth_image': sensor_data.depth_image,
                'imu_data': sensor_data.imu_data,
                'timestamp': sensor_data.timestamp
            }

        except Exception as e:
            self.logger.error(f"Failed to collect sensor data: {e}")
            return None

    def _process_sensor_data(self, raw_data: Dict[str, Any]) - SensorData:

        processed = SensorData(timestamp=raw_data['timestamp'])

        processed.stereo_left = raw_data['stereo_left']
        processed.stereo_right = raw_data['stereo_right']
        processed.depth_image = raw_data['depth_image']
        processed.imu_data = raw_data['imu_data']

        if self.sensor_bridge:
            try:
                if processed.depth_image is not None:
                    occupancy = self.sensor_bridge.get_occupancy_histogram()
                    if occupancy is not None:
                        processed.occupancy_histogram = occupancy

                    depth_features = self.sensor_bridge.get_depth_features()
                    if depth_features is not None:
                        processed.depth_features = depth_features

            except Exception as e:
                self.logger.error(f"Sensor bridge processing error: {e}")

        processed.data_quality = self._assess_data_quality(processed)

        return processed

    def _assess_data_quality(self, data: SensorData) - float:

        quality_score = 1.0

        if data.stereo_left is None or data.stereo_right is None:
            quality_score -= 0.3
        elif data.stereo_left.size == 0 or data.stereo_right.size == 0:
            quality_score -= 0.2

        if data.depth_image is None:
            quality_score -= 0.2
        elif np.all(np.isnan(data.depth_image)) or np.all(data.depth_image == 0):
            quality_score -= 0.1

        if data.imu_data is None:
            quality_score -= 0.2

        if data.processing_latency  0.1:
            quality_score -= 0.1

        if data.occupancy_histogram is None:
            quality_score -= 0.1

        return max(0.0, quality_score)

    def get_latest_data(self) - Optional[SensorData]:

        with self.data_lock:
            return self.latest_data

    def get_data_history(self, max_count: int = 10) - List[SensorData]:

        with self.data_lock:
            return list(self.data_history)[-max_count:]

    def update_sensor_data(self, processed_reading):

        if not processed_reading:
            return

        sensor_data = SensorData(
            timestamp=processed_reading.timestamp,
            stereo_left=processed_reading.stereo_left,
            stereo_right=processed_reading.stereo_right,
            depth_image=processed_reading.depth_image,
            occupancy_histogram=processed_reading.occupancy_histogram,
            depth_features=processed_reading.depth_features,
            data_quality=1.0,
            processing_latency=0.0
        )

        with self.data_lock:
            self.latest_data = sensor_data
            self.data_history.append(sensor_data)

    def add_data_callback(self, callback: Callable[[SensorData], None]):

        self.data_callbacks.append(callback)

    def get_statistics(self) - Dict[str, Any]:

        stats = {
            'data_points_collected': len(self.data_history),
            'frame_drops': self.frame_drops,
            'is_running': self.is_running,
            'callbacks_registered': len(self.data_callbacks)
        }

        if self.processing_times:
            stats['processing_performance'] = {
                'mean_processing_time': np.mean(self.processing_times),
                'max_processing_time': np.max(self.processing_times),
                'processing_frequency': 1.0 / np.mean(self.processing_times) if np.mean(self.processing_times)  0 else 0
            }

        if self.data_history:
            recent_quality = [data.data_quality for data in list(self.data_history)[-50:]]
            stats['data_quality'] = {
                'mean_quality': np.mean(recent_quality),
                'min_quality': np.min(recent_quality),
                'quality_trend': 'stable'
            }

        return stats

    def get_sensor_health(self) - Dict[str, str]:

        health = {}

        if self.latest_data:
            if self.latest_data.stereo_left is not None and self.latest_data.stereo_right is not None:
                health['stereo_camera'] = 'healthy'
            else:
                health['stereo_camera'] = 'degraded'

            if self.latest_data.depth_image is not None:
                health['depth_sensor'] = 'healthy'
            else:
                health['depth_sensor'] = 'degraded'

            if self.latest_data.imu_data is not None:
                health['imu'] = 'healthy'
            else:
                health['imu'] = 'degraded'

            if self.latest_data.processing_latency  0.05:
                health['processing'] = 'healthy'
            elif self.latest_data.processing_latency  0.1:
                health['processing'] = 'degraded'
            else:
                health['processing'] = 'critical'
        else:
            health = {
                'stereo_camera': 'offline',
                'depth_sensor': 'offline',
                'imu': 'offline',
                'processing': 'offline'
            }

        return health

    def reset(self):

        with self.data_lock:
            self.latest_data = None
            self.data_history.clear()

        self.frame_drops = 0
        self.processing_times.clear()

        self.logger.debug("Sensor interface reset")

    def is_healthy(self) - bool:

        health = self.get_sensor_health()

        critical_sensors = ['stereo_camera', 'imu', 'processing']

        for sensor in critical_sensors:
            if health.get(sensor) in ['offline', 'critical']:
                return False

        return True

    def __del__(self):

        try:
            if hasattr(self, 'is_running'):
                self.stop()
        except:
            pass
