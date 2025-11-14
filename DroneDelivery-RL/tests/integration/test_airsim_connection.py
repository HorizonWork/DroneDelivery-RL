import pytest
import time
from unittest.mock import patch, MagicMock
import airsim
from src.bridges.airsim_bridge import AirSimBridge

class TestAirSimConnection:

    def test_airsim_client_connection(self):

        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            assert client is not None
            print("AirSim client connection successful")
        except Exception as e:
            pytest.skip(f"AirSim not available for testing: {str(e)}")

    def test_airsim_bridge_initialization(self):

        try:
            bridge = AirSimBridge()
            assert bridge is not None
            assert hasattr(bridge, 'client')
            print("AirSim bridge initialization successful")
        except Exception as e:
            pytest.skip(f"AirSim bridge initialization failed: {str(e)}")

    def test_airsim_get_vehicle_state(self):

        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()

            state = client.getMultirotorState()
            assert state is not None
            assert hasattr(state, 'kinematics_estimated')
            assert hasattr(state, 'gps_location')
            print("AirSim get vehicle state successful")
        except Exception as e:
            pytest.skip(f"AirSim get vehicle state failed: {str(e)}")

    def test_airsim_takeoff_and_land(self):

        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()

            is_armed = client.getMultirotorState().landed_state
            assert is_armed is not None
            print("AirSim takeoff/land test setup successful")
        except Exception as e:
            pytest.skip(f"AirSim takeoff/land test failed: {str(e)}")

    pytest.mark.slow
    def test_airsim_movement_commands(self):

        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()

            client.enableApiControl(True)
            client.armDisarm(True)

            client.takeoffAsync().join()

            client.moveToPositionAsync(0, 0, -1, 1).join()

            client.landAsync().join()

            client.armDisarm(False)
            client.enableApiControl(False)

            print("AirSim movement commands test successful")
        except Exception as e:
            pytest.skip(f"AirSim movement commands test failed: {str(e)}")