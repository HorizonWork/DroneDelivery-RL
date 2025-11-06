# Installation Guide - DroneDelivery-RL

## System Requirements

### Hardware Requirements
- **CPU**: Intel i7 or AMD Ryzen 7 (minimum 8 cores recommended)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space minimum
- **OS**: Ubuntu 20.04/22.04 LTS or Windows 10/11

### Software Prerequisites
- **Python**: 3.8-3.10
- **CUDA**: 11.3 or later (for GPU acceleration)
- **Unreal Engine**: 4.26+ (for AirSim)
- **ROS2**: Humble Hawksbill (optional, for SLAM integration)

## Installation Steps

### Step 1: Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd DroneDelivery-RL

# Create conda environment
conda env create -f environment.yml
conda activate drone-delivery-rl

# Install package in development mode
pip install -e .
```

#### Option B: Using Virtual Environment
```bash
# Clone the repository
git clone <repository-url>
cd DroneDelivery-RL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Step 2: AirSim Setup

#### Download and Install Unreal Engine
```bash
# Register Epic Games account and download Unreal Engine 4.26+
# Follow official Unreal Engine installation guide
```

#### Build AirSim
```bash
# Clone AirSim repository
git clone https://github.com/Microsoft/AirSim.git
cd AirSim

# Linux build
./setup.sh
./build.sh

# Windows build (use Developer Command Prompt)
setup.cmd
build.cmd
```

#### Configure AirSim Environment
```bash
# Copy AirSim settings
cp config/airsim/settings.json ~/Documents/AirSim/settings.json  # Linux/Mac
# OR
copy config\airsim\settings.json %USERPROFILE%\Documents\AirSim\settings.json  # Windows

# Download pre-built 5-floor environment (if available)
# OR build custom environment using Unreal Engine
```

### Step 3: Optional - ROS2 Integration (For SLAM)

#### Install ROS2 Humble (Ubuntu only)
```bash
# Set locale
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Setup sources
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Install ROS2
sudo apt update
sudo apt install ros-humble-desktop python3-argcomplete
sudo apt install ros-dev-tools

# Setup environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### Build ROS2 Workspace
```bash
cd ros_ws
colcon build --packages-select orb_slam3_ros airsim_ros drone_interfaces
source install/setup.bash
```

### Step 4: ORB-SLAM3 Setup (Optional)

#### Install Dependencies
```bash
# Install OpenCV (if not already installed)
sudo apt install libopencv-dev python3-opencv

# Install Eigen3
sudo apt install libeigen3-dev

# Install Pangolin
cd /tmp
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

#### Build ORB-SLAM3
```bash
cd /tmp
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
chmod +x build.sh
./build.sh
```

### Step 5: Verification

#### Verify Python Environment
```bash
python scripts/setup/verify_installation.py
```

#### Test AirSim Connection
```bash
# Start AirSim environment in Unreal Engine
# Then run:
python scripts/setup/test_airsim_connection.py
```

#### Test Environment Setup
```bash
python scripts/setup/test_environment.py
```

## Configuration

### AirSim Configuration
The main AirSim configuration is in `config/airsim/settings.json`:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone0": {
      "VehicleType": "SimpleFlight",
      "X": 6000.0,
      "Y": -3000.0,
      "Z": 300.0,
      "Pitch": 0.0,
      "Roll": 0.0,
      "Yaw": 0.0
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 640,
        "Height": 480,
        "FOV_Degrees": 90
      },
      {
        "ImageType": 3,
        "Width": 640,
        "Height": 480,
        "FOV_Degrees": 90
      }
    ]
  },
  "Recording": {
    "RecordOnMove": false,
    "RecordInterval": 0.05
  }
}
```

### Training Configuration
Main training parameters in `config/training/ppo_hyperparameters.yaml`:

```yaml
# PPO Hyperparameters (Table 2 from report)
learning_rate: 3e-4
rollout_length: 2048
batch_size: 64
epochs_per_update: 10
clip_range: 0.2
discount_factor: 0.99
gae_parameter: 0.95
entropy_coefficient: 0.01
hidden_layers: [256, 128, 64]
activation_function: tanh
total_timesteps: 5000000

# Training settings
save_interval: 10000
log_interval: 100
eval_interval: 50000
max_grad_norm: 0.5
```

### Environment Configuration
Environment setup in `config/training/environment_config.yaml`:

```yaml
# Building specification
building:
  floors: 5
  floor_dimensions:
    length: 20  # meters
    width: 40   # meters
    height: 3   # meters
  cell_size: 0.5
  total_cells: 4000

# Drone spawn configuration
spawn:
  name: "DroneSpawn"
  location: [6000, -3000, 300]

# Landing targets
targets:
  floor_1: ["Landing_101", "Landing_102", "Landing_103", "Landing_104", "Landing_105", "Landing_106"]
  floor_2: ["Landing_201", "Landing_202", "Landing_203", "Landing_204", "Landing_205", "Landing_206"]
  floor_3: ["Landing_301", "Landing_302", "Landing_303", "Landing_304", "Landing_305", "Landing_306"]
  floor_4: ["Landing_401", "Landing_402", "Landing_403", "Landing_404", "Landing_405", "Landing_406"]
  floor_5: ["Landing_501", "Landing_502", "Landing_503", "Landing_504", "Landing_505", "Landing_506"]

# Drone specifications
drone_specs:
  mass: 1.5  # kg
  max_thrust_per_motor: 15.0  # N
  max_translational_speed: 5.0  # m/s
  num_rotors: 4

# Sensor configuration
sensors:
  camera_frequency: 30  # Hz
  imu_frequency: 200    # Hz
  occupancy_sectors: 24
```

## Docker Setup (Alternative)

### Build Docker Images
```bash
# Build base image
docker build -f docker/Dockerfile.base -t drone-delivery-rl:base .

# Build training image
docker build -f docker/Dockerfile.training -t drone-delivery-rl:training .

# Build evaluation image
docker build -f docker/Dockerfile.evaluation -t drone-delivery-rl:evaluation .
```

### Run with Docker Compose
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up

# Run training only
docker-compose -f docker/docker-compose.yml up training

# Run evaluation only
docker-compose -f docker/docker-compose.yml up evaluation
```

## Troubleshooting

### Common Issues

#### 1. AirSim Connection Failed
```bash
# Check if AirSim is running
# Verify settings.json is in correct location
# Check firewall settings
```

#### 2. CUDA Out of Memory
```bash
# Reduce batch size in config/training/ppo_hyperparameters.yaml
# Use gradient accumulation
# Enable mixed precision training
```

#### 3. ROS2 Build Errors
```bash
# Update ROS2 packages
sudo apt update && sudo apt upgrade

# Clean and rebuild workspace
cd ros_ws
rm -rf build install log
colcon build
```

#### 4. Python Environment Issues
```bash
# Recreate conda environment
conda env remove -n drone-delivery-rl
conda env create -f environment.yml

# Or reinstall packages
pip install --force-reinstall -r requirements.txt
```

### Performance Optimization

#### GPU Memory Optimization
```yaml
# In config/training/ppo_hyperparameters.yaml
batch_size: 32  # Reduce from 64 if memory issues
rollout_length: 1024  # Reduce from 2048 if needed
```

#### Training Speed Optimization
```yaml
# Enable mixed precision
use_mixed_precision: true

# Increase number of workers
num_workers: 8

# Use vectorized environments
num_envs: 4
```

## Verification Checklist

### Environment Setup ✓
- [ ] Python environment created and activated
- [ ] All dependencies installed
- [ ] Package installed in development mode

### AirSim Setup ✓
- [ ] Unreal Engine installed
- [ ] AirSim built successfully
- [ ] Settings.json configured correctly
- [ ] 5-floor environment available

### Optional Components ✓
- [ ] ROS2 installed (if using SLAM)
- [ ] ORB-SLAM3 built (if using visual SLAM)
- [ ] ROS workspace built successfully

### Configuration ✓
- [ ] Training parameters match Table 2
- [ ] Environment parameters match report specs
- [ ] Reward function coefficients match Equation (2)
- [ ] Landing targets configured (Landing_101-506)

### Testing ✓
- [ ] Installation verification script passes
- [ ] AirSim connection test successful
- [ ] Environment test runs without errors
- [ ] Training pipeline starts successfully

## Next Steps

After successful installation:

1. **Verify Setup**: Run `python scripts/setup/verify_installation.py`
2. **Test Environment**: Run `python scripts/setup/test_environment.py`
3. **Start Training**: Run `python scripts/training/train_full_curriculum.py`
4. **Monitor Progress**: Use TensorBoard or W&B for monitoring
5. **Evaluate Results**: Run evaluation scripts after training

For detailed usage instructions, see the main README.md and other documentation files in the `docs/` directory.