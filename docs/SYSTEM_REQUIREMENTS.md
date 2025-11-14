---

 Component  Minimum  Recommended  Optimal
------------------------------------------
 CPU  4 cores, 2.5GHz  8 cores, 3.0GHz  16 cores, 3.5GHz+
 RAM  8GB  16GB  32GB+
 GPU  None (CPU only)  GTX 1060 6GB  RTX 3070+
 Storage  20GB free  50GB SSD  100GB NVMe SSD
 Network  10Mbps  1Gbps  1Gbps+

- CPU: Intel i7-8700K / AMD Ryzen 7 3700X
- RAM: 16GB DDR4
- GPU: NVIDIA GTX 160 Ti (optional)
- Storage: 50GB SSD
- OS: Ubuntu 20.04 LTS

- CPU: Intel i9-10900K / AMD Ryzen 9 5900X
- RAM: 32GB DDR4
- GPU: NVIDIA RTX 3070 (8GB VRAM)
- Storage: 100GB NVMe SSD
- Training Time: 8-12 hours

- CPU: Intel i7-11700K / AMD Ryzen 7 5800X
- RAM: 16GB DDR4
- GPU: NVIDIA GTX 1660 Super
- Storage: 50GB SSD
- Real-time Performance: 50ms inference

---

 OS  Version  Status  Notes
----------------------------
 Ubuntu  18.04+   Fully Supported  Recommended for development
 Ubuntu  20.04 LTS   Fully Supported  Best compatibility
 Ubuntu  22.04 LTS   Supported  Latest features
 Windows  10/11   Supported  Requires WSL2 for best performance
 macOS  10.15+   Limited Support  CPU training only

- Python: 3.8, 3.9, 3.10 (Recommended: 3.9)
- pip: 21.0+
- conda: 4.12+ (Recommended: Miniconda)

pytorch: "=2.0.0"
torchvision: "=0.15.0"
gymnasium: "=0.29.0"
stable-baselines3: "=2.0.0"

opencv-python: "=4.8.0"
numpy: "=1.24.0"
scipy: "=1.1.0"

pybullet: "=3.2.5"
airsim: "=1.8.1"

matplotlib: "=3.7.0"
seaborn: "=0.12.0"
plotly: "=5.15.0"

pandas: "=2.0.0"
pyyaml: "=6.0"
h5py: "=3.9.0"

tensorboard: "=2.13.0"
wandb: "=0.15.0"

---

- CPU: 4+ cores recommended
- RAM: 4GB+ available during simulation
- Graphics: OpenGL 3.3+ support
- Performance: 60+ FPS at 1024x768

- Unreal Engine: 4.27+
- DirectX: 11/12 support
- GPU: GTX 1060 6GB minimum for realistic graphics
- VRAM: 4GB+ for high-quality environments
- Network: Local network for API communication

 Metric  Target  Hardware
--------------------------
 Simulation FPS  60+  GTX 1660+
 Training Speed  10 ep/hour  RTX 3070+
 Inference Time  50ms  Any modern CPU
 Memory Usage  8GB  16GB RAM

---

- ROS Version: Noetic (Ubuntu 20.04) or Humble (Ubuntu 22.04)
- ROS Packages:

 sudo apt install ros-noetic-desktop-full
  sudo apt install ros-noetic-mavros
  sudo apt install ros-noetic-realsense2-camera

- Flight Controller: PX4 compatible (Pixhawk 4+)
- Onboard Computer: NVIDIA Jetson Xavier NX / Intel NUC
- Sensors:
  - Intel RealSense D435i (Stereo + IMU)
  - Velodyne VLP-16 LiDAR (optional)
  - GPS module (outdoor fallback)

---

- Docker: 20.10+
- Docker Compose: 2.0+
- NVIDIA Docker: 2.0+ (for GPU support)

base_image: "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
python_version: "3.9"
cuda_version: "11.7"

memory: "16GB"
cpu_cores: "8"
gpu_memory: "8GB"

- Host RAM: 16GB+ (8GB for container + 8GB for host)
- Host Storage: 50GB+ available
- Host GPU: NVIDIA with CUDA 1.7+ support

---

 System Configuration  Time (5M timesteps)  Episodes/hour
---------------------------------------------------------
 CPU Only (i7-8700K)  24 hours  50 ep/h
 GPU (GTX 1660 Ti)  12 hours  120 ep/h
 GPU (RTX 3070)  8 hours  200 ep/h
 GPU (RTX 4090)  6 hours  30 ep/h

 System  Inference Time  Episodes (100)  Memory Usage
-----------------------------------------------------
 CPU (i5-8400)  80ms  15 minutes  4GB
 CPU (i7-10700K)  45ms  8 minutes  4GB
 GPU (GTX 1660)  25ms  5 minutes  6GB
 GPU (RTX 3070)  15ms  3 minutes  8GB

CPU_Usage: "60-80"
RAM_Usage: "8-12GB"
GPU_Usage: "85-95"
VRAM_Usage: "4-6GB"
Storage_IO: "50-100 MB/s"

CPU_Usage: "40-60"
RAM_Usage: "4-6GB"
GPU_Usage: "60-80"
VRAM_Usage: "2-4GB"
Storage_IO: "10-20 MB/s"

---

- VS Code: Python extension + Jupyter support
- PyCharm Professional: Full Python IDE
- Vim/Neovim: With Python LSP
- Jupyter Lab: For analysis notebooks

black=23.0.0
flake8=6.0.0
isort=5.12.0
mypy=1.0.0

pytest=7.2.0
pytest-cov=4.0
pytest-mock=3.10.0

sphinx=6.0.0
sphinx-rtd-theme=1.2.0

---

 Provider  Instance Type  vCPUs  RAM  GPU  Cost/hour
-----------------------------------------------------
 AWS  g4dn.xlarge  4  16GB  T4 16GB  0.526
 AWS  g4dn.2xlarge  8  32GB  T4 16GB  0.752
 GCP  n1-highmem-4 + T4  4  26GB  T4 16GB  0.65
 Azure  NC6s_v3  6  112GB  V100 16GB  3.06

 Device  Use Case  Performance  Power
--------------------------------------
 NVIDIA Jetson Orin  Real drone  15-25ms  15-60W
 NVIDIA Jetson Xavier NX  Compact drone  25-40ms  10-25W
 Intel NUC 11  Ground station  30-50ms  28W
 Raspberry Pi 4  Basic control  100-200ms  5-8W

---

python -m pytest tests/ -v

python -m pytest tests/ --cov=src --cov-report=html

python -m pytest tests/performance/ --benchmark-only

python tests/integration/test_simulation.py

python tests/integration/test_airsim.py

python tests/integration/test_full_pipeline.py

- AirSim Simulator: Running in separate process
- Network Connection: Stable local network
- API Endpoints: AirSim API accessible on localhost:41451

---

python --version

free -h

nvidia-smi

df -h

python scripts/setup/verify_installation.py

python tests/benchmarks/system_benchmark.py

python scripts/training/train_phase.py --phase single_floor --timesteps 1000

---

rl:
  ppo:
    batch_size: 128

export CUDA_VISIBLE_DEVICES=""

rl:
  model:
    hidden_size: 256

pip install torch torchvision --index-url https:

sudo apt install htop iotop

---

 Component  Ubuntu 20.04  Ubuntu 22.04  Windows 11  macOS 12+
----------------------------------------------------------------
 PyTorch 2.0
 AirSim
 ROS Noetic
 ROS Humble
 CUDA 1.7
 Docker

Recommended Development Environment: Ubuntu 20.04 LTS với NVIDIA GPU support!