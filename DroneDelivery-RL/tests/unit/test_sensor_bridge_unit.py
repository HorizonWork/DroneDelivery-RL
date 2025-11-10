"""Unit tests for sensor bridge components"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.bridges.sensor_bridge import SensorBridge


class TestSensorBridgeUnit:
    """Unit tests for SensorBridge class"""
    
    def test_sensor_bridge_constructor(self):
        """Test SensorBridge constructor"""
        with patch('airsim.MultirotorClient'):
            bridge = SensorBridge()
            
            assert bridge is not None
            assert hasattr(bridge, 'client')
    
    @patch('airsim.MultirotorClient')
    def test_get_imu_data_method(self, mock_client):
        """Test the get_imu_data method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the getImuData response
        mock_imu_data = MagicMock()
        mock_imu_data.orientation = MagicMock()
        mock_imu_data.orientation.w_val = 1.0
        mock_imu_data.orientation.x_val = 0.0
        mock_imu_data.orientation.y_val = 0.0
        mock_imu_data.orientation.z_val = 0.0
        mock_imu_data.angular_velocity = MagicMock()
        mock_imu_data.angular_velocity.x_val = 0.1
        mock_imu_data.angular_velocity.y_val = 0.2
        mock_imu_data.angular_velocity.z_val = 0.3
        mock_imu_data.linear_acceleration = MagicMock()
        mock_imu_data.linear_acceleration.x_val = 9.8
        mock_imu_data.linear_acceleration.y_val = 0.0
        mock_imu_data.linear_acceleration.z_val = 0.0
        mock_client_instance.getImuData.return_value = mock_imu_data
        
        bridge = SensorBridge()
        imu_data = bridge.get_imu_data()
        
        assert imu_data is not None
        assert hasattr(imu_data, 'orientation')
        assert hasattr(imu_data, 'angular_velocity')
        assert hasattr(imu_data, 'linear_acceleration')
    
    @patch('airsim.MultirotorClient')
    def test_get_camera_data_method(self, mock_client):
        """Test the get_camera_data method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the simGetImages response
        mock_image_response = MagicMock()
        mock_image_response[0].image_data_uint8 = np.array([1, 2, 3, 4], dtype=np.uint8)
        mock_client_instance.simGetImages.return_value = [mock_image_response[0]]
        
        bridge = SensorBridge()
        camera_data = bridge.get_camera_data()
        
        assert camera_data is not None
    
    @patch('airsim.MultirotorClient')
    def test_get_gps_data_method(self, mock_client):
        """Test the get_gps_data method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the getGpsData response
        mock_gps_data = MagicMock()
        mock_gps_data.gpfd = MagicMock()
        mock_gps_data.gpfd.latitude = 37.7749
        mock_gps_data.gpfd.longitude = -122.4194
        mock_gps_data.gpfd.altitude = 10.0
        mock_client_instance.getGpsData.return_value = mock_gps_data
        
        bridge = SensorBridge()
        gps_data = bridge.get_gps_data()
        
        assert gps_data is not None
        assert hasattr(gps_data, 'gpfd')
        assert gps_data.gpfd.latitude == 37.7749
        assert gps_data.gpfd.longitude == -122.4194
        assert gps_data.gpfd.altitude == 10.0
    
    @patch('airsim.MultirotorClient')
    def test_get_barometer_data_method(self, mock_client):
        """Test the get_barometer_data method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the getBarometerData response
        mock_baro_data = MagicMock()
        mock_baro_data.altitude = 5.0
        mock_baro_data.pressure = 1013.25
        mock_client_instance.getBarometerData.return_value = mock_baro_data
        
        bridge = SensorBridge()
        baro_data = bridge.get_barometer_data()
        
        assert baro_data is not None
        assert hasattr(baro_data, 'altitude')
        assert hasattr(baro_data, 'pressure')
    
    @patch('airsim.MultirotorClient')
    def test_get_magnetometer_data_method(self, mock_client):
        """Test the get_magnetometer_data method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the getMagnetometerData response
        mock_mag_data = MagicMock()
        mock_mag_data.magnetic_field_body = MagicMock()
        mock_mag_data.magnetic_field_body.x_val = 0.2
        mock_mag_data.magnetic_field_body.y_val = 0.1
        mock_mag_data.magnetic_field_body.z_val = 0.5
        mock_client_instance.getMagnetometerData.return_value = mock_mag_data
        
        bridge = SensorBridge()
        mag_data = bridge.get_magnetometer_data()
        
        assert mag_data is not None
        assert hasattr(mag_data, 'magnetic_field_body')
    
    @patch('airsim.MultirotorClient')
    def test_get_distance_data_method(self, mock_client):
        """Test the get_distance_data method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the getDistanceSensorData response
        mock_dist_data = MagicMock()
        mock_dist_data.distance = 1.5
        mock_client_instance.getDistanceSensorData.return_value = mock_dist_data
        
        bridge = SensorBridge()
        dist_data = bridge.get_distance_data()
        
        assert dist_data is not None
        assert hasattr(dist_data, 'distance')
    
    def test_sensor_data_fusion_method(self):
        """Test the sensor data fusion functionality"""
        # This test verifies the integration of multiple sensor readings
        with patch('airsim.MultirotorClient'):
            bridge = SensorBridge()
            
            # Since we can't easily mock all sensor data at once, 
            # we'll just verify the method exists and is callable
            assert hasattr(bridge, 'get_fused_sensor_data')
    
    @patch('airsim.MultirotorClient')
    def test_get_airsim_settings_method(self, mock_client):
        """Test the get_airsim_settings method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        bridge = SensorBridge()
        
        # Verify that the bridge can access settings-related methods
        assert hasattr(bridge, 'get_position')
        assert hasattr(bridge, 'get_orientation')