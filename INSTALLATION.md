- CPU: Intel i7 or AMD Ryzen 7 (minimum 8 cores recommended)
- GPU: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- RAM: 16GB minimum, 32GB recommended
- Storage: 50GB free space minimum
- OS: Ubuntu 20.04/22.04 LTS or Windows 10/11

- Python: 3.8-3.10
- CUDA: 11.3 or later (for GPU acceleration)
- Unreal Engine: 4.26+ (for AirSim)
- ROS2: Humble Hawksbill (optional, for SLAM integration)

bash
git clone https://github.com/HorizonWork/DroneDelivery-RL
cd DroneDelivery-RL

conda env create -f environment.yml
conda activate drone-delivery-rl

pip install -e .

bash
git clone https://github.com/HorizonWork/DroneDelivery-RL
cd DroneDelivery-RL

python -m venv venv
source venv/bin/activate
venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

bash

bash
git clone https:
cd AirSim

./setup.sh
./build.sh

setup.cmd
build.cmd

bash
cp config/airsim/settings.json /Documents/AirSim/settings.json
copy config\airsim\settings.json USERPROFILE\Documents\AirSim\settings.json

bash
locale
sudo apt update  sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update  sudo apt install curl -y
sudo curl -sSL https:
sudo sh -c 'echo "deb http:

sudo apt update
sudo apt install ros-humble-desktop python3-argcomplete
sudo apt install ros-dev-tools

echo "source /opt/ros/humble/setup.bash"  /.bashrc
source /.bashrc

bash
cd ros_ws
colcon build --packages-select orb_slam3_ros airsim_ros drone_interfaces
source install/setup.bash

bash
sudo apt install libopencv-dev python3-opencv

sudo apt install libeigen3-dev

cd /tmp
git clone https:
cd Pangolin
mkdir build  cd build
cmake ..
make -j4
sudo make install

bash
cd /tmp
git clone https:
cd ORB_SLAM3
chmod +x build.sh
./build.sh

bash
python scripts/setup/verify_installation.py

bash
python scripts/setup/test_airsim_connection.py

bash
python scripts/setup/test_environment.py

The main AirSim configuration is in config/airsim/settings.json:

json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone0": {
      "VehicleType": "SimpleFlight",
      "X": 60.0,
      "Y": -30.0,
      "Z": 3.0,
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

Main training parameters in config/training/ppo_hyperparameters.yaml:

yaml
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

save_interval: 10000
log_interval: 100
eval_interval: 50000
max_grad_norm: 0.5

Environment setup in config/training/environment_config.yaml:

yaml
building:
  floors: 5
  floor_dimensions:
    length: 20
    width: 40
    height: 3
  cell_size: 0.5
  total_cells: 4000

spawn:
  name: "DroneSpawn"
  location: [60, -30, 3]

targets:
  floor_1: ["Landing_101", "Landing_102", "Landing_103", "Landing_104", "Landing_105", "Landing_106"]
  floor_2: ["Landing_201", "Landing_202", "Landing_203", "Landing_204", "Landing_205", "Landing_206"]
  floor_3: ["Landing_301", "Landing_302", "Landing_303", "Landing_304", "Landing_305", "Landing_306"]
  floor_4: ["Landing_401", "Landing_402", "Landing_403", "Landing_404", "Landing_405", "Landing_406"]
  floor_5: ["Landing_501", "Landing_502", "Landing_503", "Landing_504", "Landing_505", "Landing_506"]

drone_specs:
  mass: 1.5
  max_thrust_per_motor: 15.0
  max_translational_speed: 5.0
  num_rotors: 4

sensors:
  camera_frequency: 30
  imu_frequency: 200
  occupancy_sectors: 24

bash
docker build -f docker/Dockerfile.base -t drone-delivery-rl:base .

docker build -f docker/Dockerfile.training -t drone-delivery-rl:training .

docker build -f docker/Dockerfile.evaluation -t drone-delivery-rl:evaluation .

bash
docker-compose -f docker/docker-compose.yml up

docker-compose -f docker/docker-compose.yml up training

docker-compose -f docker/docker-compose.yml up evaluation

bash

bash

bash
sudo apt update  sudo apt upgrade

cd ros_ws
rm -rf build install log
colcon build

bash
conda env remove -n drone-delivery-rl
conda env create -f environment.yml

pip install --force-reinstall -r requirements.txt

yaml
batch_size: 32
rollout_length: 1024

yaml
use_mixed_precision: true

num_workers: 8

num_envs: 4

- [ ] Python environment created and activated
- [ ] All dependencies installed
- [ ] Package installed in development mode

- [ ] Unreal Engine installed
- [ ] AirSim built successfully
- [ ] Settings.json configured correctly
- [ ] 5-floor environment available

- [ ] ROS2 installed (if using SLAM)
- [ ] ORB-SLAM3 built (if using visual SLAM)
- [ ] ROS workspace built successfully

- [ ] Training parameters 
- [ ] Environment parameters
- [ ] Reward function coefficients
- [ ] Landing targets configured (Landing_101-506)

- [ ] Installation verification script passes
- [ ] AirSim connection test successful
- [ ] Environment test runs without errors
- [ ] Training pipeline starts successfully

After successful installation:

1. Verify Setup: Run python scripts/setup/verify_installation.py
2. Test Environment: Run python scripts/setup/test_environment.py
3. Start Training: Run python scripts/training/train_full_curriculum.py
4. Monitor Progress: Use TensorBoard or WB for monitoring
5. Evaluate Results: Run evaluation scripts after training

For detailed usage instructions, see the main README.md and other documentation files in the docs/ directory.