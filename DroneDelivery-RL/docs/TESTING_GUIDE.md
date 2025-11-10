# Testing Guide for DroneDelivery-RL

This guide provides detailed information about the test suite for the DroneDelivery-RL project and how to run various types of tests to verify system functionality.

## Test Structure

The test suite is organized into several categories:

- **Unit Tests** (`tests/unit/`): Test individual components in isolation
- **Integration Tests** (`tests/integration/`): Test how components work together
- **System Tests** (`tests/system/`): End-to-end testing of the complete system

## Available Tests

### 1. AirSim Connection Tests

**File**: `tests/integration/test_airsim_connection.py`

These tests verify the connection to the AirSim simulator and basic drone operations:

- `test_airsim_client_connection`: Tests basic AirSim client connection
- `test_airsim_bridge_initialization`: Tests AirSim bridge initialization
- `test_airsim_get_vehicle_state`: Tests retrieving vehicle state from AirSim
- `test_airsim_takeoff_and_land`: Tests basic takeoff and land commands
- `test_airsim_movement_commands`: Tests movement commands (requires AirSim simulation)

### 2. Environment Tests

**File**: `tests/integration/test_environment.py`

These tests verify the environment functionality:

- `test_action_space_initialization`: Tests action space initialization
- `test_observation_space_initialization`: Tests observation space initialization
- `test_drone_controller_initialization`: Tests drone controller initialization
- `test_environment_initialization`: Tests environment initialization
- `test_environment_reset`: Tests environment reset functionality
- `test_environment_step`: Tests environment step functionality
- `test_environment_observation_shape`: Tests observation shape
- `test_environment_reward_calculation`: Tests reward calculation

### 3. Sensor Integration Tests

**File**: `tests/integration/test_sensors_integration.py`

These tests verify sensor data processing and fusion:

- `test_sensor_bridge_initialization`: Tests sensor bridge initialization
- `test_sensor_interface_initialization`: Tests sensor interface initialization
- `test_imu_data_retrieval`: Tests IMU data retrieval
- `test_camera_data_retrieval`: Tests camera data retrieval
- `test_gps_data_retrieval`: Tests GPS data retrieval
- `test_coordinate_transforms`: Tests coordinate transformation functionality
- `test_pose_estimator_initialization`: Tests pose estimator initialization
- `test_sensor_fusion_integration`: Tests integration of multiple sensors
- `test_sensor_data_processing_pipeline`: Tests complete sensor data processing

### 4. ROS Integration Tests

**File**: `tests/integration/test_ros_integration.py`

These tests verify ROS communication and integration:

- `test_ros_bridge_initialization`: Tests ROS bridge initialization
- `test_ros_airsim_bridge_integration`: Tests integration between ROS and AirSim bridges
- `test_ros_message_publishing`: Tests ROS message publishing
- `test_ros_message_subscription`: Tests ROS message subscription
- `test_ros_parameter_server_access`: Tests access to ROS parameter server
- `test_ros_service_call`: Tests ROS service calling
- `test_complete_ros_airsim_workflow`: Tests complete ROS-AirSim workflow
- `test_ros_node_communication`: Tests ROS node communication patterns

### 5. SLAM Integration Tests

**File**: `tests/integration/test_slam_integration.py`

These tests verify SLAM functionality:

- `test_slam_bridge_initialization`: Tests SLAM bridge initialization
- `test_orb_slam3_wrapper_initialization`: Tests ORB-SLAM3 wrapper initialization
- `test_ate_calculator_initialization`: Tests ATE calculator initialization
- `test_slam_pose_estimation`: Tests SLAM pose estimation
- `test_slam_trajectory_tracking`: Tests SLAM trajectory tracking
- `test_ate_calculation`: Tests ATE calculation
- `test_slam_image_processing`: Tests SLAM image processing
- `test_slam_integration_with_sensors`: Tests SLAM-sensor integration
- `test_slam_trajectory_evaluation`: Tests SLAM trajectory evaluation

### 6. Unit Tests

#### 6.1 AirSim Bridge Unit Tests

**File**: `tests/unit/test_airsim_bridge_unit.py`

These tests verify individual AirSim bridge methods:

- `test_airsim_bridge_constructor`: Tests AirSimBridge constructor
- `test_connect_method`: Tests the connect method
- `test_get_position_method`: Tests the get_position method
- `test_get_orientation_method`: Tests the get_orientation method
- `test_takeoff_method`: Tests the takeoff method
- `test_land_method`: Tests the land method
- `test_move_to_position_method`: Tests the move_to_position method
- `test_arm_disarm_method`: Tests the arm_disarm method
- `test_enable_api_control_method`: Tests the enable_api_control method

#### 6.2 Sensor Bridge Unit Tests

**File**: `tests/unit/test_sensor_bridge_unit.py`

These tests verify individual sensor bridge methods:

- `test_sensor_bridge_constructor`: Tests SensorBridge constructor
- `test_get_imu_data_method`: Tests the get_imu_data method
- `test_get_camera_data_method`: Tests the get_camera_data method
- `test_get_gps_data_method`: Tests the get_gps_data method
- `test_get_barometer_data_method`: Tests the get_barometer_data method
- `test_get_magnetometer_data_method`: Tests the get_magnetometer_data method
- `test_get_distance_data_method`: Tests the get_distance_data method
- `test_sensor_data_fusion_method`: Tests sensor data fusion functionality

## How to Run Tests

### Running All Tests

```bash
# Run all tests using the project's test runner
python run_tests.py

# Or run all tests using pytest directly
python -m pytest tests/ -v
```

### Running Specific Test Categories

```bash
# Run unit tests only
python -m pytest tests/unit/ -v

# Run integration tests only
python -m pytest tests/integration/ -v

# Run AirSim-specific tests only
python -m pytest tests/ -k "airsim" -v

# Run environment tests only
python -m pytest tests/ -k "environment" -v

# Run sensor integration tests only
python -m pytest tests/ -k "sensor" -v

# Run SLAM integration tests only
python -m pytest tests/ -k "slam" -v

# Run ROS integration tests only
python -m pytest tests/ -k "ros" -v
```

### Running Specific Test Files

```bash
# Run a specific test file
python -m pytest tests/integration/test_airsim_connection.py -v

# Run a specific test class
python -m pytest tests/integration/test_airsim_connection.py::TestAirSimConnection -v

# Run a specific test method
python -m pytest tests/integration/test_airsim_connection.py::TestAirSimConnection::test_airsim_client_connection -v
```

### Advanced Test Options

```bash
# Run tests in verbose mode with detailed output
python -m pytest tests/ -v -s

# Run tests and stop on first failure
python -m pytest tests/ -x

# Run tests and show local variables in tracebacks
python -m pytest tests/ -l

# Run tests with coverage report
python -m pytest tests/ --cov=src/

# Run tests in parallel (if pytest-xdist is installed)
python -m pytest tests/ -n auto
```

## Test Results Interpretation

- **✓ (Green)**: Test passed successfully
- **✗ (Red)**: Test failed due to assertion error or exception
- **Skip**: Test was skipped due to missing dependencies or conditions
- **Slow tests**: Tests marked with `@pytest.mark.slow` that require actual simulation

## Troubleshooting Common Issues

### AirSim Connection Issues

If AirSim tests are failing due to connection issues:

1. Make sure AirSim is installed and running
2. Verify that the AirSim settings.json file is configured correctly
3. Check that the AirSim executable is running before executing tests

### Missing Dependencies

Some tests might be skipped due to missing dependencies. Install them using:

```bash
pip install -r requirements.txt
```

### ROS Integration Issues

For ROS integration tests:

1. Make sure ROS is properly installed
2. Source the ROS environment: `source /opt/ros/noetic/setup.bash`
3. Ensure the ROS workspace is built: `catkin_make` in ros_ws

## Adding New Tests

When adding new functionality to the project, follow these guidelines for testing:

1. Add unit tests for individual functions/methods in `tests/unit/`
2. Add integration tests for component interactions in `tests/integration/`
3. Follow the existing test structure and naming conventions
4. Use pytest fixtures for common test setup
5. Include proper assertions and error handling in tests