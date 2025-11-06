# AIRSIM INTEGRATION GUIDE
## Connecting DroneDelivery-RL with Microsoft AirSim

---

## 🎯 **OVERVIEW**

This guide explains how to integrate your existing AirSim map với DroneDelivery-RL system, including configuration setup, bridge connection, và testing procedures.

---

## 🔧 **PREREQUISITES**

### AirSim Setup:
- **Microsoft AirSim**: 1.8.1+ installed
- **Unreal Engine**: 4.27+ with your custom map
- **Existing Map**: Your 5-floor building map with landing points
- **Network**: AirSim API accessible on localhost:41451

### System Requirements:
- **GPU**: GTX 1060+ for realistic graphics
- **RAM**: 16GB+ (8GB for AirSim + 8GB for DroneDelivery-RL)
- **CPU**: 8+ cores recommended
- **OS**: Windows 10/11 or Ubuntu 20.04+

---

## 📁 **CONFIGURATION SETUP**

### 1. AirSim Settings (settings.json)
Create/update your AirSim settings file:

**Location**: 
- Windows: `%USERPROFILE%/Documents/AirSim/settings.json`
- Linux: `~/Documents/AirSim/settings.json`

```
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
```

### 2. DroneDelivery-RL Configuration

**File**: `config/airsim_config.yaml`
```
# AirSim Integration Configuration
simulation:
  mode: "airsim" # Use AirSim instead of PyBullet
  
environment:
  # Use your existing AirSim map configuration
  use_existing_map: true
  
  # Your existing building dimensions (update these!)
  building:
    floors: 5
    x_max: 20.0      # YOUR actual building X dimension
    y_max: 40.0      # YOUR actual building Y dimension  
    z_max: 15.0      # YOUR actual building height
    floor_height: 3.0
  
  # Your existing landing points (update coordinates!)
  landing_points:
    floor_1:
      - [2.1, 3.5, 0.5]    # Update with YOUR coordinates
      - [18.7, 36.2, 0.5]  # Update with YOUR coordinates
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
  
  # Obstacle detection from AirSim
  obstacles:
    detect_from_airsim: true
    use_lidar: true
    use_depth_camera: true
    safety_margin: 0.5  # meters

# AirSim Connection Settings
airsim:
  connection:
    host: "localhost"  # AirSim running locally
    port: 41451        # Default AirSim API port
    timeout: 10.0      # Connection timeout
    vehicle_name: "Drone1"
  
  # Control Parameters
  control:
    max_velocity: 5.0        # m/s
    max_angular_velocity: 2.0 # rad/s
    control_frequency: 20     # Hz
    command_timeout: 1.0      # seconds
  
  # Sensor Configuration
  sensors:
    stereo_cameras:
      enabled: true
      resolution: 
      capture_rate: 30  # Hz
      
    imu:
      enabled: true
      sampling_rate: 200  # Hz
      
    lidar:
      enabled: true
      channels: 16
      range: 50.0  # meters
      points_per_second: 1000
      
    gps:
      enabled: false  # Indoor environment

# Training with AirSim
training:
  use_airsim: true
  airsim_reset_on_episode: true
  airsim_async_mode: false  # Synchronous for training
```

---

## 🚀 **TESTING AIRSIM CONNECTION**

### 1. Basic Connection Test
```
# Start your AirSim environment first!
# Then test connection:

python -c "
import sys
sys.path.append('src')
from bridges.airsim_bridge import AirSimDroneBridge

# Test connection
config = {
    'host': 'localhost',
    'port': 41451,
    'vehicle_name': 'Drone1'
}

bridge = AirSimDroneBridge(config)  
success = bridge.connect()

if success:
    print('✅ AirSim connection successful!')
    
    # Test basic movement
    position = bridge.get_current_position()
    print(f'Current position: {position}')
    
    # Test sensor data
    observation = bridge.get_observation()
    print(f'Observation shape: {observation.shape}')
    
    bridge.disconnect()
else:
    print('❌ AirSim connection failed!')
"
```

### 2. Environment Integration Test
```
# Test DroneEnvironment with AirSim
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

print(f'✅ AirSim environment initialized')
print(f'Observation shape: {observation.shape}')
print(f'Building dimensions: {env.building_x_max}x{env.building_y_max}x{env.building_z_max}')
"
```

### 3. Complete Pipeline Test
```
# Test training với AirSim (short run)
python scripts/training/train_phase.py \
    --phase single_floor \
    --config config/airsim_config.yaml \
    --timesteps 1000

# Expected: Training runs successfully với AirSim
```

---

## 🔧 **CONFIGURATION MAPPING**

### Your Existing Map → DroneDelivery Config

**Step 1**: Measure your AirSim building dimensions
```
# In AirSim, check your building bounds
# Update config/airsim_config.yaml:

building:
  x_max: [YOUR_X_DIMENSION]  # e.g., 25.0
  y_max: [YOUR_Y_DIMENSION]  # e.g., 35.0
  z_max: [YOUR_Z_HEIGHT]     # e.g., 18.0
```

**Step 2**: Map your landing points
```
# For each floor, update coordinates:
landing_points:
  floor_1:
    - [YOUR_X1, YOUR_Y1, YOUR_Z1]
    - [YOUR_X2, YOUR_Y2, YOUR_Z2]
  # ... repeat for all floors
```

**Step 3**: Configure obstacles detection
```
obstacles:
  detect_from_airsim: true
  use_lidar: true           # If you have LiDAR sensor
 use_depth_camera: true    # Use stereo cameras for depth
  static_obstacles_file: null  # Auto-detect from AirSim
```

---

## 🚁 **USAGE WORKFLOWS**

### 1. Training với AirSim
```
# Method 1: Phase-by-phase training
python scripts/training/train_phase.py \
    --phase single_floor \
    --config config/airsim_config.yaml

# Method 2: Full curriculum training
python scripts/training/train_full_curriculum.py \
    --config config/airsim_config.yaml

# Method 3: Main PPO training  
python scripts/training/train_ppo.py \
    --config config/airsim_config.yaml \
    --name airsim_training
```

### 2. Evaluation với AirSim
```
# Evaluate trained model in AirSim
python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/ppo_final.pt \
    --config config/airsim_config.yaml \
    --episodes 50

# Benchmark against baselines
python scripts/evaluation/benchmark_baselines.py \
    --config config/airsim_config.yaml
```

### 3. Data Collection từ AirSim
```
# Collect data from AirSim environment
python scripts/utilities/collect_data.py \
    --model models/checkpoints/ppo_final.pt \
    --config config/airsim_config.yaml \
    --episodes 100 \
    --scenarios airsim_realistic
```

---

## 📊 **PERFORMANCE OPTIMIZATION**

### AirSim Performance Settings:
```
// In settings.json for better performance
{
  "ViewMode": "NoDisplay",        // Disable rendering for training
  "ClockSpeed": 5.0,             // Speed up simulation
  "PhysicsEngineName": "FastPhysicsEngine",
  
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 320,        // Lower resolution for speed
        "Height": 240,
        "FOV_Degrees": 90
      }
    ]
  }
}
```

### DroneDelivery Optimization:
```
# In config/airsim_config.yaml
training:
  airsim_async_mode: true    # Async for better performance
  airsim_step_size: 0.1      # Larger steps for speed
  
sensors:
  stereo_cameras:
    resolution:    # Lower resolution
    capture_rate: 10         # Lower frequency
```

---

## 🔍 **DEBUGGING & TROUBLESHOOTING**

### Common Issues:

#### **1. Connection Failed**
```
# Check AirSim is running
netstat -an | grep 41451

# Test manual connection
python -c "
import airsim
client = airsim.MultirotorClient()
client.confirmConnection()
print('✅ Manual connection works')
"
```

#### **2. Slow Performance**
```
# Check system resources
htop # CPU usage
nvidia-smi  # GPU usage

# Optimize AirSim settings
# Disable unnecessary sensors
# Lower camera resolution
# Increase ClockSpeed
```

#### **3. Observation Shape Mismatch**
```
# Debug observation dimensions
bridge = AirSimDroneBridge(config)
obs = bridge.get_observation()
print(f"Observation shape: {obs.shape}")  # Should be (35,)

# Check individual components
print(f"Position: {obs[0:3]}")      # x, y, z
print(f"Orientation: {obs[3:7]}")   # quaternion
print(f"Velocity: {obs[7:11]}")     # vx, vy, vz, yaw_rate
# ... etc
```

### Debug Scripts:
```
# Test AirSim sensor data
python tests/debug/test_airsim_sensors.py

# Visualize AirSim observations
python tests/debug/visualize_airsim_data.py

# Performance profiling
python tests/debug/profile_airsim_performance.py
```

---

## 📈 **EXPECTED RESULTS**

### Successful Integration:
```
✅ AirSim Environment Features:
-  Realistic graphics and physics
- Accurate sensor simulation
-  Your existing building layout
-  Proper landing point mapping
-  Obstacle detection from environment

✅ Performance Metrics:
-  Training: 50-100 episodes/hour (vs 200+ in PyBullet)
-  Inference: 25-50ms per step
-  Success Rate: Similar to simulation results
-  Visual Quality: Photorealistic rendering
```

### Performance Comparison:
| Metric | PyBullet | AirSim | Real Hardware |
|--------|----------|---------|---------------|
| **Speed** | 200 ep/h | 80 ep/h | 20 ep/h |
| **Realism** | Low | High | Highest |
| **Setup** | Easy | Medium | Hard |
| **Debugging** | Easy | Medium | Hard |

---

## 🎯 **INTEGRATION CHECKLIST**

### Pre-Integration:
- [ ] **AirSim running** với your custom map
- [ ] **API accessible** on localhost:41451
- [ ] **Landing points measured** and documented
- [ ] **Building dimensions** confirmed
- [ ] **Sensors configured** properly

### During Integration:
- [ ] **Connection test** passes
- [ ] **Observation shape** correct (35D)
- [ ] **Action execution** works
- [ ] **Sensor data** streaming properly
- [ ] **Environment reset** functional

### Post-Integration:
- [ ] **Training test** completes successfully
- [ ] **Evaluation** produces reasonable results
- [ ] **Performance** acceptable for your use case
- [ ] **Stability** maintained over long runs
- [ ] **Results consistency** với simulation baseline

---

## 🚀 **NEXT STEPS**

After successful AirSim integration:

### 1. Validate Training:
```
# Quick training validation
python scripts/training/train_phase.py \
    --phase single_floor \
    --config config/airsim_config.yaml \
    --timesteps 10000
```

### 2. Performance Comparison:
```
# Compare AirSim vs PyBullet results
python scripts/evaluation/compare_simulators.py \
    --model-airsim models/airsim_trained.pt \
    --model-pybullet models/pybullet_trained.pt
```

### 3. Real Hardware Preparation:
```
# Test ROS integration preparation
python scripts/setup/setup_ros_integration.py
```

**🚁 Your AirSim integration is now ready for realistic drone delivery simulation! ✨**