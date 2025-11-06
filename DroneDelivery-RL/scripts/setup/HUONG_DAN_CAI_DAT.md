# HƯỚNG DẪN CÀI ĐẶT HỆ THỐNG
## DroneDelivery-RL Installation Guide

---

## 🎯 **MỤC TIÊU**

Hướng dẫn cài đặt hoàn chỉnh hệ thống DroneDelivery-RL với tất cả các thành phần cần thiết:
- Môi trường Python và các thư viện phụ trợ
- Mô phỏng AirSim (tùy chọn)
- ROS integration (tùy chọn)
- Docker containerization (tùy chọn)

---

## 📋 **YÊU CẦU HỆ THỐNG**

### Phần cứng tối thiểu:
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 8GB
- **Storage**: 20GB free
- **OS**: Ubuntu 20.04 LTS hoặc Windows 10/11

### Phần mềm cần thiết:
- **Python**: 3.8+
- **Git**: 2.0+
- **CMake**: 3.10+
- **CUDA** (nếu dùng GPU): 11.7+

---

## 🚀 **CÀI ĐẶT TỰ ĐỘNG (Khuyến nghị)**

### 1. Clone repository
```bash
# Clone dự án
git clone <repository-url> DroneDelivery-RL
cd DroneDelivery-RL
```

### 2. Cài đặt tự động (Linux/Ubuntu)
```bash
# Cài đặt môi trường và dependencies
python scripts/setup/build_environment.py

# Hoặc chạy script trực tiếp
chmod +x scripts/setup/install_dependencies.sh
./scripts/setup/install_dependencies.sh
```

### 3. Cài đặt tự động (Windows)
```powershell
# Sử dụng PowerShell
python scripts/setup/build_environment.py
```

---

## 🔧 **CÀI ĐẶT THỦ CÔNG**

### 1. Tạo môi trường ảo
```bash
# Tạo môi trường conda
conda create -n drone-delivery-rl python=3.9
conda activate drone-delivery-rl

# Hoặc tạo môi trường ảo Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 2. Cài đặt dependencies
```bash
# Cài đặt từ requirements.txt
pip install -r requirements.txt

# Hoặc cài đặt từ environment.yml (nếu dùng conda)
conda env update -f environment.yml
```

### 3. Cài đặt package
```bash
# Cài đặt package trong chế độ phát triển
pip install -e .
```

---

## 📦 **CẤU HÌNH MÔI TRƯỜNG**

### 1. Cấu hình hệ thống
```bash
# Kiểm tra cài đặt
python scripts/setup/verify_installation.py

# Output mong đợi:
# ✅ Python version: 3.9.x
# ✅ PyTorch: 2.0+ available
# ✅ CUDA: Available (nếu có GPU)
# ✅ Dependencies: All installed
```

### 2. Thiết lập cấu hình mặc định
```bash
# Copy cấu hình mặc định
cp config/training/ppo_hyperparameters.yaml config/training/default_config.yaml
cp config/evaluation/baseline_config.yaml config/evaluation/default_config.yaml
```

---

## 🎮 **TÍCH HỢP AIRSIM (Tùy chọn)**

### 1. Cài đặt AirSim
```bash
# Cài đặt AirSim Python API
pip install airsim

# Download AirSim Unreal Environment từ GitHub
# https://github.com/microsoft/AirSim
```

### 2. Cấu hình AirSim
```bash
# Tạo file cấu hình AirSim
mkdir -p ~/Documents/AirSim
cp config/airsim/settings.json ~/Documents/AirSim/
```

### 3. Kiểm tra kết nối AirSim
```bash
# Chạy AirSim environment trước
# Sau đó kiểm tra kết nối:
python -c "
import airsim
client = airsim.MultirotorClient()
client.confirmConnection()
print('✅ AirSim connection successful!')
"
```

---

## 🤖 **TÍCH HỢP ROS (Tùy chọn)**

### 1. Cài đặt ROS
```bash
# Ubuntu 20.04 - ROS Noetic
sudo apt update
sudo apt install ros-noetic-desktop-full
source /opt/ros/noetic/setup.bash

# Cài đặt ROS dependencies
pip install roslibpy
pip install rospy
```

### 2. Cấu hình ROS workspace
```bash
# ROS workspace đã được tạo trong dự án
cd ros_ws
catkin_make
source devel/setup.bash
```

---

## 🐳 **DOCKER DEPLOYMENT (Tùy chọn)**

### 1. Cài đặt Docker
```bash
# Ubuntu
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER

# Restart để áp dụng thay đổi
sudo systemctl restart docker
```

### 2. Build Docker image
```bash
# Build image chính
cd docker
docker build -f Dockerfile.base -t drone-delivery-rl:base .

# Build image training
docker build -f Dockerfile.training -t drone-delivery-rl:training .
```

### 3. Chạy container
```bash
# Chạy container với GPU (nếu có)
docker run --gpus all -it --name drone-training drone-delivery-rl:training

# Hoặc chạy container CPU
docker run -it --name drone-training drone-delivery-rl:training
```

---

## 🧪 **KIỂM TRA CÀI ĐẶT**

### 1. Kiểm tra cơ bản
```bash
# Chạy script kiểm tra cài đặt
python scripts/setup/verify_installation.py

# Kiểm tra version tất cả packages
python -c "
import torch
import numpy as np
import gymnasium
import cv2
import airsim

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'NumPy version: {np.__version__}')
print(f'Gymnasium version: {gymnasium.__version__}')
print(f'OpenCV version: {cv2.__version__}')
print(f'AirSim available: {\'airsim\' in globals()}')
"
```

### 2. Kiểm tra môi trường
```bash
# Chạy thử môi trường đơn giản
python -c "
from src.environment import DroneEnvironment
import yaml

# Load config mặc định
with open('config/training/environment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Tạo môi trường
env = DroneEnvironment(config['environment'])
obs = env.reset()
print(f'✅ Environment created successfully')
print(f'Observation shape: {obs.shape}')
"
```

---

## 🚨 **GẶP SỰ CỐ & GIẢI PHÁP**

### 1. CUDA không hoạt động
```bash
# Kiểm tra CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Nếu CUDA không hoạt động, dùng CPU
export CUDA_VISIBLE_DEVICES=""
```

### 2. Memory không đủ
```bash
# Giảm batch size trong config
# config/training/ppo_hyperparameters.yaml:
ppo:
  batch_size: 64  # Giảm từ 128 xuống
  rollout_length: 1024  # Giảm từ 2048 xuống
```

### 3. Dependencies conflict
```bash
# Tạo môi trường mới sạch
conda create -n drone-delivery-rl-clean python=3.9
conda activate drone-delivery-rl-clean
pip install -r requirements.txt
```

### 4. Permission errors
```bash
# Fix permission cho scripts
chmod +x scripts/**/*.sh
chmod +x docker/*.sh
```

---

## 📊 **HIỆU SUẤT CÀI ĐẶT**

### Thời gian cài đặt ước lượng:
| Thành phần | Thời gian | Mô tả |
|------------|-----------|-------|
| **Python packages** | 5-10 phút | pip install requirements |
| **PyTorch** | 5-15 phút | Tùy cấu hình mạng |
| **AirSim** | 10-30 phút | Download và setup |
| **ROS** | 15-45 phút | Full desktop install |
| **Docker** | 10-20 phút | Build base images |

### Dung lượng ổ đĩa:
- **Minimal install**: 5GB
- **Full install**: 15GB
- **With AirSim**: 25GB
- **With Docker**: 30GB

---

## 🔄 **CẬP NHẬT HỆ THỐNG**

### 1. Cập nhật từ repository
```bash
# Pull code mới nhất
git pull origin main

# Cập nhật dependencies
pip install -r requirements.txt --upgrade

# Cập nhật package
pip install -e . --upgrade
```

### 2. Cập nhật môi trường
```bash
# Nếu dùng conda
conda env update -f environment.yml

# Nếu dùng venv, tạo lại môi trường
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 📞 **HỖ TRỢ & LIÊN HỆ**

### Các kênh hỗ trợ:
- **GitHub Issues**: https://github.com/[repo]/issues
- **Documentation**: docs/ folder
- **Email**: [contact@university.edu]

### Troubleshooting:
- **Common issues**: docs/TROUBLESHOOTING.md
- **FAQ**: docs/FAQ.md
- **Community**: [link to community]

---

## ✅ **HOÀN TẤT CÀI ĐẶT**

Sau khi hoàn tất cài đặt, bạn có thể kiểm tra bằng lệnh:

```bash
# Kiểm tra toàn bộ hệ thống
python scripts/setup/verify_installation.py --full

# Output mong đợi:
# ✅ Python environment: OK
# ✅ Dependencies: All satisfied
# ✅ GPU support: Available (nếu có)
# ✅ AirSim integration: Configured (nếu có)
# ✅ ROS integration: Available (nếu có)
# ✅ Ready for training: YES
```

**🎉 Hệ thống DroneDelivery-RL đã sẵn sàng cho quá trình huấn luyện và đánh giá!**