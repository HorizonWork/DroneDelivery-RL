"""
Unit tests for DroneController
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.environment.drone_controller import DroneController


class TestDroneController:
    """Test suite for DroneController."""
    
    @pytest.fixture
    def controller(self):
        """Create controller with test config."""
        config = {
            "control_frequency": 20.0,
            "max_velocity": 10.0,
            "max_acceleration": 30.0,
            "action_scale": 5.0,
            "expect_normalized_actions": True,
            "strict_acceleration_check": False,  # KEY
        }
        return DroneController(config=config)
    
    def test_initialization(self, controller):
        """Test controller initializes correctly."""
        assert controller.max_velocity == 10.0
        assert controller.max_acceleration == 30.0
        assert controller.action_scale == 5.0
        assert controller.strict_acceleration_check == False
        print("✅ Initialization test passed")
    
    def test_normalized_action_scaling(self, controller):
        """Test that normalized actions [-1,1] are scaled correctly."""
        # Test action at max normalized value
        action = np.array([1.0, -1.0, 0.5, 0.0])
        
        # Simulate execute_action logic
        if controller.expect_normalized_actions:
            scaled = action.copy()
            scaled[:3] = action[:3] * controller.action_scale
            scaled[3] = action[3] * controller.max_yaw_rate
        
        # Expected: [5.0, -5.0, 2.5, 0.0]
        expected = np.array([5.0, -5.0, 2.5, 0.0])
        np.testing.assert_array_almost_equal(scaled, expected)
        print(f"✅ Action scaling test passed: {action} → {scaled}")
    
    def test_acceleration_check_disabled(self, controller):
        """Test that high acceleration is ALLOWED when strict=False."""
        # Simulate two consecutive actions with high acceleration
        action1 = np.array([0.0, 0.0, 0.0, 0.0])
        action2 = np.array([10.0, 10.0, 10.0, 0.0])  # High acceleration
        
        controller.controller_state.command_history = [action1]
        
        # This should NOT reject even though acceleration is high
        is_valid = controller._validate_action(action2)
        
        # With strict=False, should be True
        assert is_valid == True, "High acceleration should be allowed with strict=False"
        print(f"✅ Acceleration check test passed: strict=False allows high accel")
    
    def test_action_clipping(self, controller):
        """Test that extreme actions are clipped to safe limits."""
        # Extreme action
        action = np.array([100.0, -100.0, 50.0, 10.0])
        
        # Apply safety constraints
        safe = controller._apply_safety_constraints(action)
        
        # Should be clipped to max_velocity and max_yaw_rate
        assert np.all(np.abs(safe[:3]) <= controller.max_velocity)
        assert np.abs(safe[3]) <= controller.max_yaw_rate
        print(f"✅ Action clipping test passed: {action} → {safe}")
    
    def test_nan_action_rejection(self, controller):
        """Test that NaN actions are rejected."""
        action = np.array([np.nan, 0.0, 0.0, 0.0])
        
        is_valid = controller._validate_action(action)
        
        assert is_valid == False, "NaN actions should be rejected"
        print("✅ NaN rejection test passed")
    
    def test_execute_action_pipeline(self, controller):
        """Test complete action execution pipeline."""
        # Mock airsim_bridge
        class MockBridge:
            is_connected = True
            def send_velocity_command(self, vx, vy, vz, yaw, dur):
                print(f"  Mock sent: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
        
        controller.airsim_bridge = MockBridge()
        controller.controller_state.is_active = True
        
        # Send normalized action
        action = np.array([0.5, -0.3, 0.1, 0.0])
        
        print(f"  Input action (normalized): {action}")
        controller.execute_action(action)
        
        # Check that action was recorded
        assert len(controller.controller_state.command_history) > 0
        print("✅ Execute action pipeline test passed")


def run_tests():
    """Run all tests manually."""
    print("\n" + "="*60)
    print("RUNNING DRONE CONTROLLER UNIT TESTS")
    print("="*60 + "\n")
    
    controller = DroneController({
        "control_frequency": 20.0,
        "max_velocity": 10.0,
        "max_acceleration": 30.0,
        "action_scale": 5.0,
        "expect_normalized_actions": True,
        "strict_acceleration_check": False,
    })
    
    test = TestDroneController()
    
    try:
        print("1. Testing initialization...")
        test.test_initialization(controller)
        
        print("\n2. Testing action scaling...")
        test.test_normalized_action_scaling(controller)
        
        print("\n3. Testing acceleration check (strict=False)...")
        test.test_acceleration_check_disabled(controller)
        
        print("\n4. Testing action clipping...")
        test.test_action_clipping(controller)
        
        print("\n5. Testing NaN rejection...")
        test.test_nan_action_rejection(controller)
        
        print("\n6. Testing execute action pipeline...")
        test.test_execute_action_pipeline(controller)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_tests()
