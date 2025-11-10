"""Unit tests for AirSim bridge components"""

import pytest
from unittest.mock import patch, MagicMock
import airsim
from src.bridges.airsim_bridge import AirSimBridge


class TestAirSimBridgeUnit:
    """Unit tests for AirSimBridge class"""
    
    @patch('airsim.MultirotorClient')
    def test_airsim_bridge_constructor(self, mock_client):
        """Test AirSimBridge constructor"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        bridge = AirSimBridge()
        
        assert bridge is not None
        assert hasattr(bridge, 'client')
        mock_client.assert_called_once()
    
    @patch('airsim.MultirotorClient')
    def test_connect_method(self, mock_client):
        """Test the connect method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        bridge = AirSimBridge()
        bridge.connect()
        
        mock_client_instance.confirmConnection.assert_called_once()
    
    @patch('airsim.MultirotorClient')
    def test_get_position_method(self, mock_client):
        """Test the get_position method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the getMultirotorState response
        mock_state = MagicMock()
        mock_kinematics = MagicMock()
        mock_kinematics.position = MagicMock()
        mock_kinematics.position.x_val = 1.0
        mock_kinematics.position.y_val = 2.0
        mock_kinematics.position.z_val = 3.0
        mock_state.kinematics_estimated = mock_kinematics
        mock_client_instance.getMultirotorState.return_value = mock_state
        
        bridge = AirSimBridge()
        position = bridge.get_position()
        
        assert position is not None
        assert len(position) == 3
        assert position[0] == 1.0
        assert position[1] == 2.0
        assert position[2] == 3.0
    
    @patch('airsim.MultirotorClient')
    def test_get_orientation_method(self, mock_client):
        """Test the get_orientation method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock the getMultirotorState response
        mock_state = MagicMock()
        mock_kinematics = MagicMock()
        mock_kinematics.orientation = MagicMock()
        mock_kinematics.orientation.w_val = 1.0
        mock_kinematics.orientation.x_val = 0.0
        mock_kinematics.orientation.y_val = 0.0
        mock_kinematics.orientation.z_val = 0.0
        mock_state.kinematics_estimated = mock_kinematics
        mock_client_instance.getMultirotorState.return_value = mock_state
        
        bridge = AirSimBridge()
        orientation = bridge.get_orientation()
        
        assert orientation is not None
        assert len(orientation) == 4
        assert orientation[0] == 1.0  # w
        assert orientation[1] == 0.0  # x
        assert orientation[2] == 0.0  # y
        assert orientation[3] == 0.0  # z
    
    @patch('airsim.MultirotorClient')
    def test_takeoff_method(self, mock_client):
        """Test the takeoff method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        bridge = AirSimBridge()
        bridge.takeoff()
        
        mock_client_instance.takeoffAsync.assert_called_once()
    
    @patch('airsim.MultirotorClient')
    def test_land_method(self, mock_client):
        """Test the land method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        bridge = AirSimBridge()
        bridge.land()
        
        mock_client_instance.landAsync.assert_called_once()
    
    @patch('airsim.MultirotorClient')
    def test_move_to_position_method(self, mock_client):
        """Test the move_to_position method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        bridge = AirSimBridge()
        bridge.move_to_position(10, 10, -10)
        
        mock_client_instance.moveToPositionAsync.assert_called_once_with(10, 10, -10, 3)
    
    @patch('airsim.MultirotorClient')
    def test_arm_disarm_method(self, mock_client):
        """Test the arm_disarm method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        bridge = AirSimBridge()
        bridge.arm_disarm(True)
        
        mock_client_instance.armDisarm.assert_called_once_with(True)
    
    @patch('airsim.MultirotorClient')
    def test_enable_api_control_method(self, mock_client):
        """Test the enable_api_control method"""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        bridge = AirSimBridge()
        bridge.enable_api_control(True)
        
        mock_client_instance.enableApiControl.assert_called_once_with(True)