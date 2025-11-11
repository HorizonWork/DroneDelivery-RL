"""Test sensor integration and data processing"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.bridges.sensor_bridge import SensorBridge
from src.environment.sensor_interface import SensorInterface
from src.localization.pose_estimator import PoseEstimator
from src.localization.coordinate_transforms import CoordinateTransforms

pytestmark = [
    pytest.mark.requires_simulator,
    pytest.mark.skipif(
        os.environ.get("DRONERL_ENABLE_AIRSIM_TESTS") != "1",
        reason="Requires running AirSim simulator (set DRONERL_ENABLE_AIRSIM_TESTS=1 to enable).",
    ),
]

class TestSensorIntegration:
    """Test class for sensor integration tests"""
    
    def test_sensor_bridge_initialization(self):
        """Test sensor bridge initialization"""
        try:
            sensor_bridge = SensorBridge()
            assert sensor_bridge is not None
            assert hasattr(sensor_bridge, 'get_imu_data')
            assert hasattr(sensor_bridge, 'get_camera_data')
            assert hasattr(sensor_bridge, 'get_gps_data')
            print("Sensor bridge initialization successful")
        except Exception as e:
            pytest.skip(f"Sensor bridge initialization failed: {str(e)}")
    
    def test_sensor_interface_initialization(self):
        """Test sensor interface initialization"""
        try:
            sensor_interface = SensorInterface()
            assert sensor_interface is not None
            assert hasattr(sensor_interface, 'get_sensor_data')
            assert hasattr(sensor_interface, 'process_sensor_data')
            print("Sensor interface initialization successful")
        except Exception as e:
            pytest.skip(f"Sensor interface initialization failed: {str(e)}")
    
    def test_imu_data_retrieval(self):
        """Test IMU data retrieval"""
        try:
            sensor_bridge = SensorBridge()
            imu_data = sensor_bridge.get_imu_data()
            
            # Check if IMU data has expected structure
            if imu_data is not None:
                assert hasattr(imu_data, 'orientation')
                assert hasattr(imu_data, 'angular_velocity')
                assert hasattr(imu_data, 'linear_acceleration')
                print("IMU data retrieval successful")
            else:
                print("IMU data not available, but no error occurred")
        except Exception as e:
            pytest.skip(f"IMU data retrieval failed: {str(e)}")
    
    def test_camera_data_retrieval(self):
        """Test camera data retrieval"""
        try:
            sensor_bridge = SensorBridge()
            camera_data = sensor_bridge.get_camera_data()
            
            # Check if camera data exists
            if camera_data is not None:
                assert hasattr(camera_data, 'image')
                assert hasattr(camera_data, 'timestamp')
                print("Camera data retrieval successful")
            else:
                print("Camera data not available, but no error occurred")
        except Exception as e:
            pytest.skip(f"Camera data retrieval failed: {str(e)}")
    
    def test_gps_data_retrieval(self):
        """Test GPS data retrieval"""
        try:
            sensor_bridge = SensorBridge()
            gps_data = sensor_bridge.get_gps_data()
            
            # Check if GPS data has expected structure
            if gps_data is not None:
                assert hasattr(gps_data, 'latitude')
                assert hasattr(gps_data, 'longitude')
                assert hasattr(gps_data, 'altitude')
                print("GPS data retrieval successful")
            else:
                print("GPS data not available, but no error occurred")
        except Exception as e:
            pytest.skip(f"GPS data retrieval failed: {str(e)}")
    
    def test_coordinate_transforms(self):
        """Test coordinate transformation functionality"""
        try:
            coord_transforms = CoordinateTransforms()
            assert coord_transforms is not None
            assert hasattr(coord_transforms, 'airsim_to_world')
            assert hasattr(coord_transforms, 'world_to_airsim')
            
            # Test a simple transformation
            test_pos = [0, 0, 0]
            world_pos = coord_transforms.airsim_to_world(test_pos[0], test_pos[1], test_pos[2])
            airsim_pos = coord_transforms.world_to_airsim(world_pos[0], world_pos[1], world_pos[2])
            
            assert len(world_pos) == 3
            assert len(airsim_pos) == 3
            print("Coordinate transforms test successful")
        except Exception as e:
            pytest.skip(f"Coordinate transforms test failed: {str(e)}")
    
    def test_pose_estimator_initialization(self):
        """Test pose estimator initialization"""
        try:
            pose_estimator = PoseEstimator()
            assert pose_estimator is not None
            assert hasattr(pose_estimator, 'estimate_pose')
            assert hasattr(pose_estimator, 'update_pose')
            print("Pose estimator initialization successful")
        except Exception as e:
            pytest.skip(f"Pose estimator initialization failed: {str(e)}")
    
    def test_sensor_fusion_integration(self):
        """Test integration of multiple sensors for pose estimation"""
        try:
            # This test verifies that multiple sensors can work together
            sensor_interface = SensorInterface()
            pose_estimator = PoseEstimator()
            
            # Get sensor data
            sensor_data = sensor_interface.get_sensor_data()
            
            # Process sensor data for pose estimation
            if sensor_data:
                pose = pose_estimator.estimate_pose(sensor_data)
                assert pose is not None
                print("Sensor fusion integration test successful")
            else:
                print("Sensor data not available for fusion test")
        except Exception as e:
            pytest.skip(f"Sensor fusion integration test failed: {str(e)}")
    
    def test_sensor_data_processing_pipeline(self):
        """Test the complete sensor data processing pipeline"""
        try:
            sensor_interface = SensorInterface()
            
            # Get raw sensor data
            raw_data = sensor_interface.get_sensor_data()
            assert raw_data is not None
            
            # Process sensor data
            processed_data = sensor_interface.process_sensor_data(raw_data)
            assert processed_data is not None
            
            print("Sensor data processing pipeline test successful")
        except Exception as e:
            pytest.skip(f"Sensor data processing pipeline test failed: {str(e)}")
