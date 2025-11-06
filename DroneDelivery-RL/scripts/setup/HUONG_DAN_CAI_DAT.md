# HƯỚNG DẪN CÀI ĐẶT DRONEDELIVERY-RL
## Indoor Multi-Floor UAV Delivery - Energy-Aware Navigation System

---

## 🎯 TỔNG QUAN

Hướng dẫn này sẽ giúp bạn cài đặt hoàn chỉnh hệ thống DroneDelivery-RL từ đầu. Sau khi hoàn thành, bạn sẽ có một môi trường đầy đủ để:
- 🚁 **Huấn luyện** PPO agent cho drone navigation  
- 🏢 **Simulation** 5-floor building environment
- 📊 **Đánh giá** performance với Table 3 results
- ⚡ **Optimization** energy-aware navigation

---

## 📋 YÊU CẦU HỆ THỐNG

### Yêu cầu tối thiểu:
- **Python**: 3.8 hoặc mới hơn
- **RAM**: 8GB (khuyến nghị 16GB cho training)  
- **Disk**: 10GB trống (cho data, models, results)
- **OS**: Ubuntu 18.04+, macOS 10.14+, Windows 10+
- **Internet**: Để download packages và dependencies

### Yêu cầu khuyến nghị:
- **CPU**: Multi-core processor (8+ cores tối ưu)
- **GPU**: NVIDIA GPU với CUDA support (không bắt buộc)
- **RAM**: 16GB+ cho training lớn
- **SSD**: Để tăng tốc I/O operations

---

## 🚀 PHƯƠNG PHÁP CÀI ĐẶT

### Phương pháp 1: Cài đặt tự động (KHUYẾN NGHỊ) ⭐

**Bước duy nhất - Chạy script tự động:**
Clone project (nếu chưa có)
```bash
git clone <repository-url> DroneDelivery-RL
cd DroneDelivery-RL
```

Chạy script cài đặt tự động
```bash
python scripts/setup/build_environment.py
```

Windows: Download và chạy installer từ conda.io

**Thời gian**: ~15-20 phút  
**Ưu điểm**: Hoàn toàn tự động, detect hệ thống, xử lý lỗi  
**Nhược điểm**: Ít control, cần internet tốt

---

### Phương pháp 2: Cài đặt từng bước

#### Bước 1: Cài đặt Conda/Mamba
Ubuntu/Debian
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

macOS
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

Windows: Download và chạy installer từ conda.io
text

#### Bước 2: Cài đặt system packages
Ubuntu/Debian
```bash
sudo apt update && sudo apt install -y \
build-essential cmake git wget curl unzip \
libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
libxrender-dev libgomp1 python3-dev
```

macOS (with Homebrew)
```bash
brew install cmake git wget curl
```

Windows: Không cần (Python packages sẽ handle)

#### Bước 3: Setup Conda environment
Chạy script setup conda
```bash
bash scripts/setup/setup_conda_env.sh
```

Hoặc manual:
```bash
conda create -n drone_delivery_rl python=3.9 -y
conda activate drone_delivery_rl
```

text

#### Bước 4: Cài đặt dependencies
Chạy script install dependencies
```bash
bash scripts/setup/install_dependencies.sh
```

Hoặc manual install packages (xem requirements trong script)
text

#### Bước 5: Verify installation
Kiểm tra cài đặt
```bash
python scripts/setup/verify_installation.py
```

Windows: Không cần (Python packages sẽ handle)

**Thời gian**: ~20-30 phút  
**Ưu điểm**: Control từng bước, debug dễ dàng  
**Nhược điểm**: Phức tạp hơn, dễ miss steps

---

## 📁 CẤU TRÚC SAU KHI CÀI ĐẶT

DroneDelivery-RL/
├── 📁 src/ # Source code chính
│ ├── bridges/ # Hardware interfaces
│ ├── environment/ # Drone simulation
│ ├── localization/ # VI-SLAM system
│ ├── planning/ # A* và S-RRT planners
│ ├── rl/ # PPO reinforcement learning
│ └── utils/ # Utilities và tools
├── 📁 scripts/ # Execution scripts
│ ├── evaluation/ # Đánh giá performance
│ ├── setup/ # Cài đặt và setup
│ ├── training/ # Huấn luyện models
│ └── utilities/ # Helper scripts
├── 📁 config/ # Configuration files
│ ├── main_config.yaml # Cấu hình chính
│ └── evaluation_config.yaml # Cấu hình đánh giá
├── 📁 data/ # Data storage
│ ├── trajectories/ # Flight paths
│ └── maps/ # Building maps
├── 📁 models/ # Trained models
│ └── checkpoints/ # Model checkpoints
├── 📁 results/ # Kết quả và reports
│ ├── evaluations/ # Evaluation results
│ └── visualizations/ # Plots và charts
└── 📁 logs/ # System logs

text

---

## 🔧 TÙY CHỌN CÀI ĐẶT

### Cài đặt với tên environment khác:
```bash
python scripts/setup/build_environment.py --env-name my_custom_env
```
Windows: Không cần (Python packages sẽ handle)


### Cài đặt với GPU support:
Sau khi setup xong, install CUDA PyTorch
```bash
conda activate drone_delivery_rl
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Windows: Không cần (Python packages sẽ handle)


### Cài đặt development tools:
```bash
conda activate drone_delivery_rl
pip install jupyter notebook ipython black flake8 pytest
```
Windows: Không cần (Python packages sẽ handle)


### Cài đặt optional packages:
Cho advanced visualization
```bash
pip install plotly dash streamlit
```

Cho distributed training
```bash
pip install ray[rllib]
```

Cho experiment tracking
```bash
pip install mlflow neptune-client
```
Windows: Không cần (Python packages sẽ handle)


---

## ✅ KIỂM TRA CÀI ĐẶT

### Test cơ bản:
Activate environment
```bash
conda activate drone_delivery_rl
```

Chạy verification script
```bash
python scripts/setup/verify_installation.py
```

Windows: Không cần (Python packages sẽ handle)

### Test advanced:
Test environment creation
python -c "
from src.environment import DroneEnvironment
config = {'building': {'floors': 5}}
env = DroneEnvironment(config)
print('✅ Environment creation successful')
"

Test RL agent
python -c "
from src.rl.agents import PPOAgent
config = {'observation_dim': 35, 'action_dim': 4}
agent = PPOAgent(config)
print('✅ RL agent creation successful')
"

Test complete system
python -c "
from src import DroneDeliverySystem
system = DroneDeliverySystem()
print('✅ Complete system integration successful')
print(f'System status: {system.get_system_status()}')
"

Windows: Không cần (Python packages sẽ handle)

### Kết quả mong đợi:
🔍 VERIFYING DRONEDELIVERY-RL INSTALLATION
🧪 Testing Python Environment...
✅ Python 3.9.18
✅ Virtual environment active: drone_delivery_rl

🧪 Testing Core Dependencies...
✅ numpy: 1.24.3
✅ scipy: 1.11.4
✅ matplotlib: 3.7.2
✅ pyyaml: 6.0.1
✅ tqdm: 4.66.1
✅ psutil: 5.9.6

🧪 Testing ML/RL Packages...
✅ torch: 2.1.0+cpu
CUDA available: False
✅ gymnasium: 0.29.1
✅ Environment creation test passed
✅ pybullet: 3.2.5
✅ tensorboard: 2.14.1
✅ wandb: 0.16.0

🧪 Testing Computer Vision...
✅ opencv-python: 4.8.1.78
✅ pillow: 10.0.1

🧪 Testing Project Structure...
✅ src/
✅ src/rl/
✅ src/environment/
✅ scripts/
✅ config/
✅ data/
✅ models/
✅ results/

🧪 Testing Project Imports...
✅ src.utils
✅ src.environment
✅ src.rl.agents
✅ src.planning
✅ src.localization

==================================================
🎉 INSTALLATION VERIFICATION: ALL TESTS PASSED

Your DroneDelivery-RL installation is ready!

Windows: Không cần (Python packages sẽ handle)

---

## 🐛 XỬ LÝ LỖI THƯỜNG GẶP

### Lỗi 1: "conda command not found"
Cài đặt Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Reload shell
```bash
source ~/.bashrc
```

Hoặc
exec bash

Windows: Không cần (Python packages sẽ handle)

### Lỗi 2: "Permission denied" khi install system packages
Ubuntu: Cần sudo
```bash
sudo apt update && sudo apt install -y build-essential
```

macOS: Install Homebrew trước
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Windows: Không cần (Python packages sẽ handle)

### Lỗi 3: "PyTorch installation failed"
Cài riêng PyTorch
```bash
conda activate drone_delivery_rl
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Hoặc với GPU
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Windows: Không cần (Python packages will handle)

### Lỗi 4: "Package conflicts" 
Clean install environment
```bash
conda env remove -n drone_delivery_rl -y
conda clean --all
```

Chạy lại setup
```bash
python scripts/setup/build_environment.py
```

Windows: Không cần (Python packages will handle)

### Lỗi 5: "Import errors" cho project modules
Kiểm tra PYTHONPATH
```bash
export PYTHONPATH="${PWD}/src}:${PYTHONPATH}"
```

Hoặc install project
```bash
pip install -e .
```

Hoặc thêm vào .bashrc
echo 'export PYTHONPATH="/path/to/DroneDelivery-RL/src:${PYTHONPATH}"' >> ~/.bashrc

Windows: Không cần (Python packages will handle)

### Lỗi 6: "Insufficient memory" khi training
Giảm batch size trong config
config/main_config.yaml:
rl:
ppo:
batch_size: 128 # Từ 256 xuống 128
rollout_length: 1024 # Từ 2048 xuống 1024

Windows: Không cần (Python packages will handle)

### Lỗi 7: "Display/GUI errors" cho visualization
Ubuntu: Cài thêm GUI packages
```bash
sudo apt install -y python3-tk
```

SSH remote: Setup X11 forwarding
```bash
ssh -X username@hostname
```

Hoặc dùng headless mode
export MPLBACKEND=Agg # Matplotlib không cần display

Windows: Không cần (Python packages will handle)

---

## 🖥️ HỖ TRỢ THEO HỆ ĐIỀU HÀNH

### 🐧 Ubuntu/Debian Linux
Full setup command
```bash
sudo apt update && \
python scripts/setup/build_environment.py && \
conda activate drone_delivery_rl && \
python scripts/setup/verify_installation.py
```

Windows: Không cần (Python packages will handle)

**Đặc biệt lưu ý**: Cần `sudo` cho system packages

### 🍎 macOS
Cài Homebrew trước (nếu chưa có)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Setup project
```bash
python scripts/setup/build_environment.py
conda activate drone_delivery_rl
python scripts/setup/verify_installation.py
```

Windows: Không cần (Python packages will handle)

**Đặc biệt lưu ý**: Có thể cần install Xcode Command Line Tools

### 🪟 Windows
```batch
:: Mở Anaconda Prompt hoặc PowerShell
:: Navigate to project directory
cd DroneDelivery-RL
```

```batch
:: Chạy setup
python scripts\setup\build_environment.py
```

```batch
:: Activate environment
conda activate drone_delivery_rl
```

```batch
:: Verify
python scripts\setup\verify_installation.py
```

Windows: Không cần (Python packages will handle)

**Đặc biệt lưu ý**: Sử dụng `\` thay vì `/` trong paths

---

## ⏱️ TIMELINE CÀI ĐẶT

### Automatic Setup (build_environment.py):
| Bước | Mô tả | Thời gian | 
|------|-------|-----------|
| 1 | System prerequisites check | 1 phút |
| 2 | System packages install | 3-5 phút |
| 3 | Conda environment setup | 2-3 phút |
| 4 | Python packages install | 8-12 phút |
| 5 | Project structure creation | 1 phút |
| 6 | Data download | 2-3 phút |
| 7 | Installation verification | 1-2 phút |
| **TOTAL** | **Complete setup** | **~20 phút** |

### Manual Setup (step-by-step):
| Bước | Script | Thời gian |
|------|--------|-----------|
| 1 | `setup_conda_env.sh` | 5-8 phút |
| 2 | `install_dependencies.sh` | 10-15 phút |
| 3 | `verify_installation.py` | 2-3 phút |
| **TOTAL** | **Manual setup** | **~25 phút** |

---

## 📋 CHECKLIST HOÀN THÀNH

### Trước khi bắt đầu:
- [ ] **Python 3.8+** đã cài đặt
- [ ] **Conda/Mamba** đã cài đặt  
- [ ] **Git** đã cài đặt (khuyến nghị)
- [ ] **10GB+ disk space** trống
- [ ] **Internet connection** ổn định

### Sau khi setup:
- [ ] **Conda environment** `drone_delivery_rl` active
- [ ] **All packages** import thành công  
- [ ] **Project structure** complete với tất cả folders
- [ ] **Configuration files** tạo thành công
- [ ] **Verification script** pass tất cả tests
- [ ] **Example imports** work correctly

### Bước tiếp theo:
- [ ] **Training**: `python scripts/training/train_ppo.py`
- [ ] **Evaluation**: `python scripts/evaluation/evaluate_model.py`
- [ ] **Visualization**: Check `results/visualizations/`

---

## 🔧 CẤU HÌNH TỪY CHỈNH

### Thay đổi Python version:
Tạo với Python 3.10
```bash
conda create -n drone_delivery_rl python=3.10 -y
```

Hoặc modify trong build_environment.py:
PYTHON_VERSION = "3.10"
Windows: Không cần (Python packages will handle)

### Cài đặt GPU support:
Sau khi setup xong, thay thế PyTorch
```bash
conda activate drone_delivery_rl
conda uninstall pytorch torchvision torchaudio -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Verify GPU
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Windows: Không cần (Python packages will handle)

### Development mode setup:
Thêm development tools
conda activate drone_delivery_rl
pip install
jupyter
notebook
ipython
black
flake8
pytest
pytest-cov

Windows: Không cần (Python packages will handle)

### Minimal installation (chỉ evaluation):
Tạo environment minimal cho chỉ evaluation
```bash
conda create -n drone_eval python=3.9 -y
conda activate drone_eval
pip install torch numpy matplotlib pyyaml tqdm
```

Windows: Không cần (Python packages will handle)

---

## 📊 KIỂM TRA HIỆU SUẤT

### Test system performance:
CPU benchmark
```bash
conda activate drone_delivery_rl
python -c "
import time
import numpy as np
start = time.time()
np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)
print(f'Matrix multiplication time: {time.time()-start:.3f}s')
"
```

Memory usage test
```bash
python -c "
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB')
print(f'CPU cores: {psutil.cpu_count()}')
"
```

PyTorch performance test
```bash
python -c "
import torch
x = torch.randn(1000, 1000)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
if torch.cuda.is_available():
x = x.cuda()
start.record()
torch.mm(x, x)
end.record()
torch.cuda.synchronize()
print(f'GPU matrix mult: {start.elapsed_time(end):.2f}ms')
else:
import time
t = time.time()
torch.mm(x, x)
print(f'CPU matrix mult: {(time.time()-t)*1000:.2f}ms')
"
```

text

---

## 🆘 HỖ TRỢ VÀ TROUBLESHOOTING

### Khi setup không thành công:

1. **Xem logs chi tiết**:
```bash
cat setup.log
```

Hoặc
```bash
tail -f setup.log # Real-time log
```

Windows: Không cần (Python packages will handle)

2. **Force clean reinstall**:
Xóa environment cũ
```bash
conda env remove -n drone_delivery_rl -y
conda clean --all
```

Setup lại
```bash
python scripts/setup/build_environment.py --force
```

Windows: Không cần (Python packages will handle)

3. **Manual dependency install**:
Activate environment
```bash
conda activate drone_delivery_rl
```
Install từng package
```bash
pip install torch
pip install gymnasium
pip install pybullet
... # continue với other packages
```

pip install gymnasium
pip install pybullet

... # continue với other packages

4. **Check disk space**:
df -h # Linux/Mac

Đảm bảo có ít nhất 10GB trống
Windows: Không cần (Python packages will handle)

5. **Check internet connection**:
```bash
ping google.com
```

Hoặc test package download
```bash
pip install --dry-run torch
```

text

### Contact và Support:

- **Project Issues**: Check GitHub issues
- **Setup Problems**: Review `setup.log` và `verification_results.json`
- **Package Conflicts**: Try clean install với fresh conda environment
- **System Specific**: Check OS-specific requirements

---

## 🏁 HOÀN THÀNH CÀI ĐẶT

Khi verification script hiển thị:
🎉 INSTALLATION VERIFICATION: ALL TESTS PASSED

Your DroneDelivery-RL installation is ready!

Windows: Không cần (Python packages will handle)

Bạn đã sẵn sàng để:

1. **🚀 Bắt đầu training**:
```bash
conda activate drone_delivery_rl
python scripts/training/train_ppo.py --config config/main_config.yaml
```

Windows: Không cần (Python packages will handle)

2. **📊 Chạy evaluation** (nếu đã có model):
```bash
python scripts/evaluation/evaluate_model.py \
--model models/checkpoints/ppo_final.pt \
--episodes 100
```

Windows: Không cần (Python packages will handle)

3. **📈 Generate Table 3**:
Chạy complete evaluation pipeline
```bash
bash scripts/evaluation/run_full_evaluation.sh
```

Windows: Không cần (Python packages will handle)

---

## 🎯 SUCCESS CRITERIA

✅ **Installation hoàn thành khi**:
- Tất cả verification tests PASS
- Project imports work correctly  
- Environment creation successful
- Example config files generated
- Ready để start training/evaluation

**Estimated setup time**: 15-30 phút depending on internet speed

**🎉 Chúc mừng! Bạn đã sẵn sàng để develop energy-aware indoor drone delivery system! 🚁✨**