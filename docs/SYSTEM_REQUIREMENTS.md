# SYSTEM REQUIREMENTS
## DroneDelivery-RL Hardware & Software Specifications

---

## 💻 **HARDWARE REQUIREMENTS**

### Minimum System Requirements:
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU** | 4 cores, 2.5GHz | 8 cores, 3.0GHz | 16 cores, 3.5GHz+ |
| **RAM** | 8GB | 16GB | 32GB+ |
| **GPU** | None (CPU only) | GTX 1060 6GB | RTX 3070+ |
| **Storage** | 20GB free | 50GB SSD | 100GB NVMe SSD |
| **Network** | 10Mbps | 1Gbps | 1Gbps+ |

### For Different Use Cases:

#### 🧪 **Research/Development**:
- **CPU**: Intel i7-8700K / AMD Ryzen 7 3700X
- **RAM**: 16GB DDR4
- **GPU**: NVIDIA GTX 160 Ti (optional)
- **Storage**: 50GB SSD
- **OS**: Ubuntu 20.04 LTS

#### 🚁 **Training (5M timesteps)**:
- **CPU**: Intel i9-10900K / AMD Ryzen 9 5900X  
- **RAM**: 32GB DDR4
- **GPU**: NVIDIA RTX 3070 (8GB VRAM)
- **Storage**: 100GB NVMe SSD
- **Training Time**: 8-12 hours

#### 🎯 **Production Deployment**:
- **CPU**: Intel i7-11700K / AMD Ryzen 7 5800X
- **RAM**: 16GB DDR4
- **GPU**: NVIDIA GTX 1660 Super
- **Storage**: 50GB SSD
- **Real-time Performance**: <50ms inference

---

## 🖥️ **SOFTWARE REQUIREMENTS**

### Operating System Support:
| OS | Version | Status | Notes |
|----|---------|--------|-------|
| **Ubuntu** | 18.04+ | ✅ Fully Supported | Recommended for development |
| **Ubuntu** | 20.04 LTS | ✅ Fully Supported | Best compatibility |
| **Ubuntu** | 22.04 LTS | ✅ Supported | Latest features |
| **Windows** | 10/11 | ✅ Supported | Requires WSL2 for best performance |
| **macOS** | 10.15+ | ⚠️ Limited Support | CPU training only |

### Python Environment:
- **Python**: 3.8, 3.9, 3.10 (Recommended: 3.9)
- **pip**: 21.0+
- **conda**: 4.12+ (Recommended: Miniconda)

### Core Dependencies:
```
# Core ML/RL Stack
pytorch: ">=2.0.0"
torchvision: ">=0.15.0"
gymnasium: ">=0.29.0"
stable-baselines3: ">=2.0.0"

# Computer Vision & SLAM
opencv-python: ">=4.8.0"
numpy: ">=1.24.0"
scipy: ">=1.1.0"

# Simulation & Physics
pybullet: ">=3.2.5"
airsim: ">=1.8.1" # Optional

# Visualization & Analysis
matplotlib: ">=3.7.0"
seaborn: ">=0.12.0"
plotly: ">=5.15.0"

# Data & Config
pandas: ">=2.0.0"
pyyaml: ">=6.0"
h5py: ">=3.9.0"

# Logging & Monitoring
tensorboard: ">=2.13.0"
wandb: ">=0.15.0"  # Optional
```

---

## 🚁 **SIMULATION REQUIREMENTS**

### PyBullet Simulation:
- **CPU**: 4+ cores recommended
- **RAM**: 4GB+ available during simulation
- **Graphics**: OpenGL 3.3+ support
- **Performance**: 60+ FPS at 1024x768

### AirSim Integration:
- **Unreal Engine**: 4.27+  
- **DirectX**: 11/12 support
- **GPU**: GTX 1060 6GB minimum for realistic graphics
- **VRAM**: 4GB+ for high-quality environments
- **Network**: Local network for API communication

### Performance Targets:
| Metric | Target | Hardware |
|--------|--------|----------|
| **Simulation FPS** | 60+ | GTX 1660+ |
| **Training Speed** | 10 ep/hour | RTX 3070+ |
| **Inference Time** | <50ms | Any modern CPU |
| **Memory Usage** | <8GB | 16GB RAM |

---

## 🤖 **ROS REQUIREMENTS** (Optional)

### For Real Hardware Deployment:
- **ROS Version**: Noetic (Ubuntu 20.04) or Humble (Ubuntu 22.04)
- **ROS Packages**:
  ```
 sudo apt install ros-noetic-desktop-full
  sudo apt install ros-noetic-mavros*
  sudo apt install ros-noetic-realsense2-camera
  ```

### Hardware Integration:
- **Flight Controller**: PX4 compatible (Pixhawk 4+)
- **Onboard Computer**: NVIDIA Jetson Xavier NX / Intel NUC
- **Sensors**:
  - Intel RealSense D435i (Stereo + IMU)
  - Velodyne VLP-16 LiDAR (optional)
  - GPS module (outdoor fallback)

---

## 📦 **DOCKER REQUIREMENTS**

### Container Runtime:
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Docker**: 2.0+ (for GPU support)

### Container Specifications:
```
# Base container
base_image: "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
python_version: "3.9"
cuda_version: "11.7"

# Resource limits
memory: "16GB"
cpu_cores: "8"
gpu_memory: "8GB"  # If available
```

### Docker System Requirements:
- **Host RAM**: 16GB+ (8GB for container + 8GB for host)
- **Host Storage**: 50GB+ available
- **Host GPU**: NVIDIA with CUDA 1.7+ support

---

## ⚡ **PERFORMANCE BENCHMARKS**

### Training Performance:
| System Configuration | Time (5M timesteps) | Episodes/hour |
|---------------------|---------------------|---------------|
| **CPU Only** (i7-8700K) | ~24 hours | 50 ep/h |
| **GPU** (GTX 1660 Ti) | ~12 hours | 120 ep/h |
| **GPU** (RTX 3070) | ~8 hours | 200 ep/h |
| **GPU** (RTX 4090) | ~6 hours | 30 ep/h |

### Evaluation Performance:
| System | Inference Time | Episodes (100) | Memory Usage |
|--------|---------------|----------------|--------------|
| **CPU** (i5-8400) | 80ms | 15 minutes | 4GB |
| **CPU** (i7-10700K) | 45ms | 8 minutes | 4GB |
| **GPU** (GTX 1660) | 25ms | 5 minutes | 6GB |
| **GPU** (RTX 3070) | 15ms | 3 minutes | 8GB |

### System Resource Usage:
```
# During Training
CPU_Usage: "60-80%"
RAM_Usage: "8-12GB"
GPU_Usage: "85-95%"  # If available
VRAM_Usage: "4-6GB"
Storage_IO: "50-100 MB/s"

# During Evaluation  
CPU_Usage: "40-60%"
RAM_Usage: "4-6GB"
GPU_Usage: "60-80%"
VRAM_Usage: "2-4GB"
Storage_IO: "10-20 MB/s"
```

---

## 🔧 **DEVELOPMENT TOOLS**

### Recommended IDE/Editors:
- **VS Code**: Python extension + Jupyter support
- **PyCharm Professional**: Full Python IDE
- **Vim/Neovim**: With Python LSP
- **Jupyter Lab**: For analysis notebooks

### Development Dependencies:
```
# Code Quality
black>=23.0.0        # Code formatting
flake8>=6.0.0        # Linting
isort>=5.12.0        # Import sorting
mypy>=1.0.0          # Type checking

# Testing
pytest>=7.2.0       # Unit testing
pytest-cov>=4.0   # Coverage reporting
pytest-mock>=3.10.0 # Mocking

# Documentation
sphinx>=6.0.0       # Documentation generation
sphinx-rtd-theme>=1.2.0  # ReadTheDocs theme
```

---

## 🚀 **DEPLOYMENT REQUIREMENTS**

### Cloud Deployment:
| Provider | Instance Type | vCPUs | RAM | GPU | Cost/hour |
|----------|---------------|-------|-----|-----|-----------|
| **AWS** | g4dn.xlarge | 4 | 16GB | T4 16GB | $0.526 |
| **AWS** | g4dn.2xlarge | 8 | 32GB | T4 16GB | $0.752 |
| **GCP** | n1-highmem-4 + T4 | 4 | 26GB | T4 16GB | $0.65 |
| **Azure** | NC6s_v3 | 6 | 112GB | V100 16GB | $3.06 |

### Edge Deployment:
| Device | Use Case | Performance | Power |
|--------|----------|-------------|-------|
| **NVIDIA Jetson Orin** | Real drone | 15-25ms | 15-60W |
| **NVIDIA Jetson Xavier NX** | Compact drone | 25-40ms | 10-25W |
| **Intel NUC 11** | Ground station | 30-50ms | 28W |
| **Raspberry Pi 4** | Basic control | 100-200ms | 5-8W |

---

## 🧪 **TESTING REQUIREMENTS**

### Unit Testing:
```
# Minimum for development
python -m pytest tests/ -v

# With coverage reporting
python -m pytest tests/ --cov=src --cov-report=html

# Performance testing
python -m pytest tests/performance/ --benchmark-only
```

### Integration Testing:
```
# Simulation integration
python tests/integration/test_simulation.py

# AirSim integration (requires AirSim running)
python tests/integration/test_airsim.py

# End-to-end pipeline
python tests/integration/test_full_pipeline.py
```

### Hardware-in-the-Loop Testing:
- **AirSim Simulator**: Running in separate process
- **Network Connection**: Stable local network
- **API Endpoints**: AirSim API accessible on localhost:41451

---

## ✅ **SYSTEM VERIFICATION**

### Pre-installation Check:
```
# Check Python version
python --version  # Should be 3.8+

# Check available RAM
free -h  # Should have 8GB+ available

# Check GPU (if available)
nvidia-smi  # Should show CUDA-capable GPU

# Check disk space
df -h  # Should have 20GB+ free
```

### Post-installation Verification:
```
# Verify installation
python scripts/setup/verify_installation.py

# Performance benchmark
python tests/benchmarks/system_benchmark.py

# Quick functionality test
python scripts/training/train_phase.py --phase single_floor --timesteps 1000
```

---

## 🆘 **TROUBLESHOOTING**

### Common Issues:

#### **Insufficient Memory**:
```
# Reduce batch size in config
rl:
  ppo:
    batch_size: 128 # Instead of 256
```

#### **CUDA Out of Memory**:
```
# Use CPU training
export CUDA_VISIBLE_DEVICES=""

# Or reduce model size
rl:
  model:
    hidden_size: 256 # Instead of 512
```

#### **Slow Training**:
```
# Enable GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# Optimize system
sudo apt install htop iotop
```

---

## 📊 **COMPATIBILITY MATRIX**

| Component | Ubuntu 20.04 | Ubuntu 22.04 | Windows 11 | macOS 12+ |
|-----------|---------------|---------------|------------|-----------|
| **PyTorch 2.0** | ✅ | ✅ | ✅ |
| **AirSim** | ✅ | ⚠️ | ✅ | ❌ |
| **ROS Noetic** | ✅ | ❌ |
| **ROS Humble** | ❌ | ✅ | ❌ | ❌ |
| **CUDA 1.7** | ✅ | ✅ | ❌ |
| **Docker** | ✅ | ✅ | ✅ |

**Recommended Development Environment**: Ubuntu 20.04 LTS với NVIDIA GPU support! 🖥️🚁✨