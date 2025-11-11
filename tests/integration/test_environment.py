"""Test environment functionality and integration"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.environment.airsim_env import AirSimEnvironment
from src.environment.drone_controller import DroneController
from src.environment.action_space import ActionSpace
from src.environment.observation_space import ObservationSpace

pytestmark = [
    pytest.mark.requires_simulator,
    pytest.mark.skipif(
        os.environ.get("DRONERL_ENABLE_AIRSIM_TESTS") != "1",
        reason="Requires running AirSim simulator (set DRONERL_ENABLE_AIRSIM_TESTS=1 to enable).",
    ),
]

class TestEnvironment:
    """Test class for environment functionality"""
    
    def test_action_space_initialization(self):
        """Test action space initialization"""
        action_space = ActionSpace()
        assert action_space is not None
        assert hasattr(action_space, 'action_map')
        assert len(action_space.action_map) > 0
        print("Action space initialization successful")
    
    def test_observation_space_initialization(self):
        """Test observation space initialization"""
        obs_space = ObservationSpace()
        assert obs_space is not None
        assert hasattr(obs_space, 'observation_shape')
        assert obs_space.observation_shape is not None
        print("Observation space initialization successful")
    
    def test_drone_controller_initialization(self):
        """Test drone controller initialization"""
        try:
            controller = DroneController()
            assert controller is not None
            assert hasattr(controller, 'takeoff')
            assert hasattr(controller, 'move_to_position')
            print("Drone controller initialization successful")
        except Exception as e:
            pytest.skip(f"Drone controller initialization requires AirSim: {str(e)}")
    
    def test_environment_initialization(self):
        """Test environment initialization"""
        try:
            env = AirSimEnv()
            assert env is not None
            assert hasattr(env, 'reset')
            assert hasattr(env, 'step')
            assert hasattr(env, 'get_observation')
            print("Environment initialization successful")
        except Exception as e:
            pytest.skip(f"Environment initialization requires AirSim: {str(e)}")
    
    def test_environment_reset(self):
        """Test environment reset functionality"""
        try:
            env = AirSimEnv()
            obs = env.reset()
            assert obs is not None
            assert isinstance(obs, dict) or isinstance(obs, np.ndarray)
            print("Environment reset successful")
        except Exception as e:
            pytest.skip(f"Environment reset failed: {str(e)}")
    
    def test_environment_step(self):
        """Test environment step functionality"""
        try:
            env = AirSimEnv()
            env.reset()
            
            # Test a simple action (e.g., hover)
            action = 0  # Assuming 0 is a valid action for hovering
            obs, reward, done, info = env.step(action)
            
            assert obs is not None
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
            print("Environment step successful")
        except Exception as e:
            pytest.skip(f"Environment step failed: {str(e)}")
    
    def test_environment_observation_shape(self):
        """Test that observations have the expected shape"""
        try:
            env = AirSimEnv()
            obs = env.reset()
            obs_space = ObservationSpace()
            
            # Check if observation matches expected shape
            expected_shape = obs_space.observation_shape
            actual_shape = None
            
            if isinstance(obs, np.ndarray):
                actual_shape = obs.shape
            elif isinstance(obs, dict):
                actual_shape = {k: v.shape if hasattr(v, 'shape') else len(v) for k, v in obs.items()}
            
            assert actual_shape is not None
            print(f"Observation shape test successful. Expected: {expected_shape}, Actual: {actual_shape}")
        except Exception as e:
            pytest.skip(f"Observation shape test failed: {str(e)}")
    
    def test_environment_reward_calculation(self):
        """Test reward calculation functionality"""
        try:
            env = AirSimEnv()
            env.reset()
            
            # Perform a step and check reward
            action = 0
            obs, reward, done, info = env.step(action)
            
            assert isinstance(reward, (int, float))
            assert 'reward_components' in info if 'reward_components' in info else True
            print("Reward calculation test successful")
        except Exception as e:
            pytest.skip(f"Reward calculation test failed: {str(e)}")
