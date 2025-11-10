"""
Sensor Interface
High-level interface for sensor data integration and processing.
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque

@dataclass
class SensorData:
    """Comprehensive sensor data bundle."""
    timestamp: float
    
    # Visual data
    stereo_left: Optional[np.ndarray] = None
    stereo_right: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    
    # Inertial data
    imu_data: Optional[Dict[str, Any]] = None
    
    # Processed features
    occupancy_histogram: Optional[np.ndarray] = None
    depth_features: Optional[Dict[str, float]] = None
    
    # Quality metrics
    data_quality: float = 1.0
    processing_latency: float = 0.0

class SensorInterface:
    """
    High-level sensor data interface.
    Coordinates sensor data collection, processing, and distribution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sensor configuration
        self.camera_frequency = config.get('camera_frequency', 30.0)      # Hz
        self.imu_frequency = config.get('imu_frequency', 200.0)           # Hz
        self.occupancy_frequency = config.get('occupancy_frequency', 20.0) # Hz
        
        # Data storage
        self.latest_data: Optional[SensorData] = None
        self.data_history: deque = deque(maxlen=100)
        
        # Quality monitoring
        self.frame_drops = 0
        self.processing_times: List[float] = []
        self.max_processing_history = 1000
        
        # Threading
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.data_lock = threading.Lock()
        
        # Callbacks for data updates
        self.data_callbacks: List[Callable[[SensorData], None]] = []
        
        # Bridge references (set by environment)
        self.airsim_bridge = None
        self.sensor_bridge = None
        
        self.logger.info("Sensor Interface initialized")
        self.logger.info(f"Frequencies - Camera: {self.camera_frequency}Hz, "
                        f"IMU: {self.imu_frequency}Hz, "
                        f"Occupancy: {self.occupancy_frequency}Hz")
    
    def set_bridges(self, airsim_bridge, sensor_bridge):
        """Set bridge references."""
        self.airsim_bridge = airsim_bridge
        self.sensor_bridge = sensor_bridge
        self.logger.info("Sensor bridges connected")
    
    def start(self):
        """Start sensor data processing."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Sensor interface started")
    
    def stop(self):
        """Stop sensor data processing."""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        self.logger.info("Sensor interface stopped")
    
    def _processing_loop(self):
        """Main sensor processing loop."""
        self.logger.info("Sensor processing loop started")
        
        while self.is_running:
            try:
                # Process sensor data at camera frequency
                self._collect_and_process_data()
                
                # Sleep to maintain frequency
                time.sleep(1.0 / self.camera_frequency)
                
            except Exception as e:
                self.logger.error(f"Sensor processing error: {e}")
                time.sleep(0.1)
        
        self.logger.info("Sensor processing loop stopped")
    
    def _collect_and_process_data(self):
        """Collect and process sensor data from all sources."""
        processing_start = time.time()
        
        # Collect raw sensor data
        raw_data = self._collect_raw_data()
        if not raw_data:
            self.frame_drops += 1
            return
        
        # Process data
        processed_data = self._process_sensor_data(raw_data)
        
        # Calculate processing latency
        processing_time = time.time() - processing_start
        processed_data.processing_latency = processing_time
        
        # Store processed data
        with self.data_lock:
            self.latest_data = processed_data
            self.data_history.append(processed_data)
        
        # Track processing performance
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_processing_history:
            self.processing_times.pop(0)
        
        # Notify callbacks
        for callback in self.data_callbacks:
            try:
                callback(processed_data)
            except Exception as e:
                self.logger.error(f"Sensor callback error: {e}")
    
    def _collect_raw_data(self) -> Optional[Dict[str, Any]]:
        """Collect raw sensor data from AirSim."""
        if not self.airsim_bridge or not self.airsim_bridge.is_connected:
            return None
        
        try:
            # Get sensor data from AirSim
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
    
    def _process_sensor_data(self, raw_data: Dict[str, Any]) -> SensorData:
        """
        Process raw sensor data into structured format.
        
        Args:
            raw_data: Raw sensor data dictionary
            
        Returns:
            Processed sensor data
        """
        processed = SensorData(timestamp=raw_data['timestamp'])
        
        # Copy image data
        processed.stereo_left = raw_data['stereo_left']
        processed.stereo_right = raw_data['stereo_right']
        processed.depth_image = raw_data['depth_image']
        processed.imu_data = raw_data['imu_data']
        
        # Process occupancy histogram if sensor bridge available
        if self.sensor_bridge:
            try:
                if processed.depth_image is not None:
                    # Get occupancy histogram from sensor bridge
                    occupancy = self.sensor_bridge.get_occupancy_histogram()
                    if occupancy is not None:
                        processed.occupancy_histogram = occupancy
                    
                    # Get depth features
                    depth_features = self.sensor_bridge.get_depth_features()
                    if depth_features is not None:
                        processed.depth_features = depth_features
                
            except Exception as e:
                self.logger.error(f"Sensor bridge processing error: {e}")
        
        # Calculate data quality metrics
        processed.data_quality = self._assess_data_quality(processed)
        
        return processed
    
    def _assess_data_quality(self, data: SensorData) -> float:
        """
        Assess overall data quality.
        
        Args:
            data: Sensor data to assess
            
        Returns:
            Quality score [0, 1]
        """
        quality_score = 1.0
        
        # Check image availability and quality
        if data.stereo_left is None or data.stereo_right is None:
            quality_score -= 0.3
        elif data.stereo_left.size == 0 or data.stereo_right.size == 0:
            quality_score -= 0.2
        
        # Check depth data
        if data.depth_image is None:
            quality_score -= 0.2
        elif np.all(np.isnan(data.depth_image)) or np.all(data.depth_image == 0):
            quality_score -= 0.1
        
        # Check IMU data
        if data.imu_data is None:
            quality_score -= 0.2
        
        # Check processing latency
        if data.processing_latency > 0.1:  # > 100ms is concerning
            quality_score -= 0.1
        
        # Check occupancy data
        if data.occupancy_histogram is None:
            quality_score -= 0.1
        
        return max(0.0, quality_score)
    
    def get_latest_data(self) -> Optional[SensorData]:
        """
        Get latest sensor data.
        
        Returns:
            Latest sensor data or None
        """
        with self.data_lock:
            return self.latest_data
    
    def get_data_history(self, max_count: int = 10) -> List[SensorData]:
        """
        Get recent sensor data history.
        
        Args:
            max_count: Maximum number of historical readings
            
        Returns:
            List of recent sensor data
        """
        with self.data_lock:
            return list(self.data_history)[-max_count:]
    
    def update_sensor_data(self, processed_reading):
        """
        Update sensor data from external processor.
        
        Args:
            processed_reading: Processed sensor reading from sensor bridge
        """
        if not processed_reading:
            return
        
        # Convert sensor bridge reading to our format
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
        """
        Add callback for sensor data updates.
        
        Args:
            callback: Callback function
        """
        self.data_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get sensor interface statistics.
        
        Returns:
            Statistics dictionary
        """
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
                'processing_frequency': 1.0 / np.mean(self.processing_times) if np.mean(self.processing_times) > 0 else 0
            }
        
        # Data quality statistics
        if self.data_history:
            recent_quality = [data.data_quality for data in list(self.data_history)[-50:]]
            stats['data_quality'] = {
                'mean_quality': np.mean(recent_quality),
                'min_quality': np.min(recent_quality),
                'quality_trend': 'stable'  # Could be calculated from trend
            }
        
        return stats
    
    def get_sensor_health(self) -> Dict[str, str]:
        """
        Get sensor health status.
        
        Returns:
            Health status for each sensor
        """
        health = {}
        
        if self.latest_data:
            # Camera health
            if self.latest_data.stereo_left is not None and self.latest_data.stereo_right is not None:
                health['stereo_camera'] = 'healthy'
            else:
                health['stereo_camera'] = 'degraded'
            
            # Depth sensor health
            if self.latest_data.depth_image is not None:
                health['depth_sensor'] = 'healthy'
            else:
                health['depth_sensor'] = 'degraded'
            
            # IMU health
            if self.latest_data.imu_data is not None:
                health['imu'] = 'healthy'
            else:
                health['imu'] = 'degraded'
            
            # Processing health
            if self.latest_data.processing_latency < 0.05:  # < 50ms
                health['processing'] = 'healthy'
            elif self.latest_data.processing_latency < 0.1:  # < 100ms
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
        """Reset sensor interface for new episode."""
        with self.data_lock:
            self.latest_data = None
            self.data_history.clear()
        
        self.frame_drops = 0
        self.processing_times.clear()
        
        self.logger.debug("Sensor interface reset")
    
    def is_healthy(self) -> bool:
        """
        Check if sensor interface is healthy.
        
        Returns:
            True if all sensors are functioning properly
        """
        health = self.get_sensor_health()
        
        # All sensors should be healthy or at least degraded (not offline/critical)
        critical_sensors = ['stereo_camera', 'imu', 'processing']
        
        for sensor in critical_sensors:
            if health.get(sensor) in ['offline', 'critical']:
                return False
        
        return True
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            if hasattr(self, 'is_running'):
                self.stop()
        except:
            pass
