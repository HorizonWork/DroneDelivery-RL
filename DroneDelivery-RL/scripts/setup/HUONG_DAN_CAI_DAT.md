---

Hướng dẫn cài đặt hoàn chỉnh hệ thống DroneDelivery-RL với tất cả các thành phần cần thiết:
- Môi trường Python và các thư viện phụ trợ
- Mô phỏng AirSim (tùy chọn)
- ROS integration (tùy chọn)
- Docker containerization (tùy chọn)

---

- CPU: 4 cores, 2.5GHz
- RAM: 8GB
- Storage: 20GB free
- OS: Ubuntu 20.04 LTS hoặc Windows 10/11

- Python: 3.8+
- Git: 2.0+
- CMake: 3.10+
- CUDA (nếu dùng GPU): 11.7+

---

bash
git clone repository-url DroneDelivery-RL
cd DroneDelivery-RL

bash
python scripts/setup/build_environment.py

chmod +x scripts/setup/install_dependencies.sh
./scripts/setup/install_dependencies.sh

powershell
python scripts/setup/build_environment.py

---

bash
conda create -n drone-delivery-rl python=3.9
conda activate drone-delivery-rl

python -m venv venv
source venv/bin/activate
venv\Scripts\activate

bash
pip install -r requirements.txt

conda env update -f environment.yml

bash
pip install -e .

---

bash
python scripts/setup/verify_installation.py

bash
cp config/training/ppo_hyperparameters.yaml config/training/default_config.yaml
cp config/evaluation/baseline_config.yaml config/evaluation/default_config.yaml

---

bash
pip install airsim

bash
mkdir -p /Documents/AirSim
cp config/airsim/settings.json /Documents/AirSim/

bash
python -c "
import airsim
client = airsim.MultirotorClient()
client.confirmConnection()
print(' AirSim connection successful!')
"

---

bash
sudo apt update
sudo apt install ros-noetic-desktop-full
source /opt/ros/noetic/setup.bash

pip install roslibpy
pip install rospy

bash
cd ros_ws
catkin_make
source devel/setup.bash

---

bash
sudo apt install docker.io docker-compose
sudo usermod -aG docker USER

sudo systemctl restart docker

bash
cd docker
docker build -f Dockerfile.base -t drone-delivery-rl:base .

docker build -f Dockerfile.training -t drone-delivery-rl:training .

bash
docker run --gpus all -it --name drone-training drone-delivery-rl:training

docker run -it --name drone-training drone-delivery-rl:training

---

bash
python scripts/setup/verify_installation.py

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

bash
python -c "
from src.environment import DroneEnvironment
import yaml

with open('config/training/environment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

env = DroneEnvironment(config['environment'])
obs = env.reset()
print(f' Environment created successfully')
print(f'Observation shape: {obs.shape}')
"

---

bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

export CUDA_VISIBLE_DEVICES=""

bash
ppo:
  batch_size: 64
  rollout_length: 1024

bash
conda create -n drone-delivery-rl-clean python=3.9
conda activate drone-delivery-rl-clean
pip install -r requirements.txt

bash
chmod +x scripts.sh
chmod +x docker/.sh

---

 Thành phần  Thời gian  Mô tả
------------------------------
 Python packages  5-10 phút  pip install requirements
 PyTorch  5-15 phút  Tùy cấu hình mạng
 AirSim  10-30 phút  Download và setup
 ROS  15-45 phút  Full desktop install
 Docker  10-20 phút  Build base images

- Minimal install: 5GB
- Full install: 15GB
- With AirSim: 25GB
- With Docker: 30GB

---

bash
git pull origin main

pip install -r requirements.txt --upgrade

pip install -e . --upgrade

bash
conda env update -f environment.yml

deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

---

- GitHub Issues: https:
- Documentation: docs/ folder
- Email: [contactuniversity.edu]

- Common issues: docs/TROUBLESHOOTING.md
- FAQ: docs/FAQ.md
- Community: [link to community]

---

Sau khi hoàn tất cài đặt, bạn có thể kiểm tra bằng lệnh:

bash
python scripts/setup/verify_installation.py --full

 Hệ thống DroneDelivery-RL đã sẵn sàng cho quá trình huấn luyện và đánh giá!