---

This guide explains how to integrate your existing AirSim map với DroneDelivery-RL system, including configuration setup, bridge connection, và testing procedures.

---

- Microsoft AirSim: 1.8.1+ installed
- Unreal Engine: 4.27+ with your custom map
- Existing Map: Your 5-floor building map with landing points
- Network: AirSim API accessible on localhost:41451

- GPU: GTX 1060+ for realistic graphics
- RAM: 16GB+ (8GB for AirSim + 8GB for DroneDelivery-RL)
- CPU: 8+ cores recommended
- OS: Windows 10/11 or Ubuntu 20.04+

---

Create/update your AirSim settings file:

Location:
- Windows: USERPROFILE/Documents/AirSim/settings.json
- Linux: /Documents/AirSim/settings.json

{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ApiServerPort": 41451,

  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "EnableApiControl": true,
      "AllowAPIAlways": true,

      "Cameras": {
        "front_left": {
          "CameraName": "front_left",
          "ImageType": 0,
          "X": 0.30, "Y": -0.20, "Z": 0.0,
          "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
        },
        "front_right": {
          "CameraName": "front_right",
          "ImageType": 0,
          "X": 0.30, "Y": 0.20, "Z": 0.0,
          "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
        }
      },

      "Sensors": {
        "Imu": {
          "SensorType": 2,
          "Enabled": true
        },
        "Magnetometer": {
          "SensorType": 3,
          "Enabled": true
        },
        "Gps": {
          "SensorType": 4,
          "Enabled": true
        },
        "Lidar1": {
          "SensorType": 6,
          "Enabled": true,
          "NumberOfChannels": 16,
          "Range": 50.0,
          "PointsPerSecond": 10000,
          "X": 0.0, "Y": 0.0, "Z": -0.1,
          "DataFrame": "SensorLocalFrame"
        }
      }
    }
  },

 "PhysicsEngineName": "FastPhysicsEngine",
  "SpeedUnitFactor": 1.0,
  "ClockSpeed": 1.0,

  "Recording": {
    "RecordOnMove": false,
    "RecordInterval": 0.05
  }
}

File: config/airsim_config.yaml

simulation:
  mode: "airsim"

environment:
  use_existing_map: true

  building:
    floors: 5
    x_max: 20.0
    y_max: 40.0
    z_max: 15.0
    floor_height: 3.0

  landing_points:
    floor_1:
      - [2.1, 3.5, 0.5]
      - [18.7, 36.2, 0.5]
    floor_2:
      - [2.1, 3.5, 3.5]
      - [18.7, 36.2, 3.5]
    floor_3:
      - [2.1, 3.5, 6.5]
      - [18.7, 36.2, 6.5]
    floor_4:
      - [2.1, 3.5, 9.5]
      - [18.7, 36.2, 9.5]
    floor_5:
      - [2.1, 3.5, 12.5]
      - [18.7, 36.2, 12.5]

  obstacles:
    detect_from_airsim: true
    use_lidar: true
    use_depth_camera: true
    safety_margin: 0.5

airsim:
  connection:
    host: "localhost"
    port: 41451
    timeout: 10.0
    vehicle_name: "Drone1"

  control:
    max_velocity: 5.0
    max_angular_velocity: 2.0
    control_frequency: 20
    command_timeout: 1.0

  sensors:
    stereo_cameras:
      enabled: true
      resolution:
      capture_rate: 30

    imu:
      enabled: true
      sampling_rate: 200

    lidar:
      enabled: true
      channels: 16
      range: 50.0
      points_per_second: 1000

    gps:
      enabled: false

training:
  use_airsim: true
  airsim_reset_on_episode: true
  airsim_async_mode: false

---

python -c "
import sys
sys.path.append('src')
from bridges.airsim_bridge import AirSimDroneBridge

config = {
    'host': 'localhost',
    'port': 41451,
    'vehicle_name': 'Drone1'
}

bridge = AirSimDroneBridge(config)
success = bridge.connect()

if success:
    print(' AirSim connection successful!')

    position = bridge.get_current_position()
    print(f'Current position: {position}')

    observation = bridge.get_observation()
    print(f'Observation shape: {observation.shape}')

    bridge.disconnect()
else:
    print(' AirSim connection failed!')
"

python -c "
import sys
sys.path.append('src')
from environment import DroneEnvironment

config = {
    'mode': 'airsim',
    'airsim': {
        'connection': {
            'host': 'localhost',
            'port': 41451,
            'vehicle_name': 'Drone1'
        }
    }
}

env = DroneEnvironment(config)
observation = env.reset()

print(f' AirSim environment initialized')
print(f'Observation shape: {observation.shape}')
print(f'Building dimensions: {env.building_x_max}x{env.building_y_max}x{env.building_z_max}')
"

python scripts/training/train_phase.py \
    --phase single_floor \
    --config config/airsim_config.yaml \
    --timesteps 1000

---

Step 1: Measure your AirSim building dimensions

building:
  x_max: [YOUR_X_DIMENSION]
  y_max: [YOUR_Y_DIMENSION]
  z_max: [YOUR_Z_HEIGHT]

Step 2: Map your landing points

landing_points:
  floor_1:
    - [YOUR_X1, YOUR_Y1, YOUR_Z1]
    - [YOUR_X2, YOUR_Y2, YOUR_Z2]

Step 3: Configure obstacles detection

obstacles:
  detect_from_airsim: true
  use_lidar: true
 use_depth_camera: true
  static_obstacles_file: null

---

python scripts/training/train_phase.py \
    --phase single_floor \
    --config config/airsim_config.yaml

python scripts/training/train_full_curriculum.py \
    --config config/airsim_config.yaml

python scripts/training/train_ppo.py \
    --config config/airsim_config.yaml \
    --name airsim_training

python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/ppo_final.pt \
    --config config/airsim_config.yaml \
    --episodes 50

python scripts/evaluation/benchmark_baselines.py \
    --config config/airsim_config.yaml

python scripts/utilities/collect_data.py \
    --model models/checkpoints/ppo_final.pt \
    --config config/airsim_config.yaml \
    --episodes 100 \
    --scenarios airsim_realistic

---

{
  "ViewMode": "NoDisplay",
  "ClockSpeed": 5.0,
  "PhysicsEngineName": "FastPhysicsEngine",

  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 320,
        "Height": 240,
        "FOV_Degrees": 90
      }
    ]
  }
}

training:
  airsim_async_mode: true
  airsim_step_size: 0.1

sensors:
  stereo_cameras:
    resolution:
    capture_rate: 10

---

netstat -an  grep 41451

python -c "
import airsim
client = airsim.MultirotorClient()
client.confirmConnection()
print(' Manual connection works')
"

htop
nvidia-smi

bridge = AirSimDroneBridge(config)
obs = bridge.get_observation()
print(f"Observation shape: {obs.shape}")

print(f"Position: {obs[0:3]}")
print(f"Orientation: {obs[3:7]}")
print(f"Velocity: {obs[7:11]}")

python tests/debug/test_airsim_sensors.py

python tests/debug/visualize_airsim_data.py

python tests/debug/profile_airsim_performance.py

---

 AirSim Environment Features:
-  Realistic graphics and physics
- Accurate sensor simulation
-  Your existing building layout
-  Proper landing point mapping
-  Obstacle detection from environment

 Performance Metrics:
-  Training: 50-100 episodes/hour (vs 200+ in PyBullet)
-  Inference: 25-50ms per step
-  Success Rate: Similar to simulation results
-  Visual Quality: Photorealistic rendering

 Metric  PyBullet  AirSim  Real Hardware
------------------------------------------
 Speed  200 ep/h  80 ep/h  20 ep/h
 Realism  Low  High  Highest
 Setup  Easy  Medium  Hard
 Debugging  Easy  Medium  Hard

---

- [ ] AirSim running với your custom map
- [ ] API accessible on localhost:41451
- [ ] Landing points measured and documented
- [ ] Building dimensions confirmed
- [ ] Sensors configured properly

- [ ] Connection test passes
- [ ] Observation shape correct (35D)
- [ ] Action execution works
- [ ] Sensor data streaming properly
- [ ] Environment reset functional

- [ ] Training test completes successfully
- [ ] Evaluation produces reasonable results
- [ ] Performance acceptable for your use case
- [ ] Stability maintained over long runs
- [ ] Results consistency với simulation baseline

---

After successful AirSim integration:

python scripts/training/train_phase.py \
    --phase single_floor \
    --config config/airsim_config.yaml \
    --timesteps 10000

python scripts/evaluation/compare_simulators.py \
    --model-airsim models/airsim_trained.pt \
    --model-pybullet models/pybullet_trained.pt

python scripts/setup/setup_ros_integration.py

 Your AirSim integration is now ready for realistic drone delivery simulation!