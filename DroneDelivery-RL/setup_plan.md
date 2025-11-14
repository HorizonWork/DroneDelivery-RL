Tên dự án: DroneDelivery-RL - Energy-Aware Indoor Multi-Floor UAV Delivery System
Mục tiêu: Hệ thống điều hướng drone giao hàng nội bộ sử dụng Reinforcement Learning (PPO) với tối ưu hóa năng lượng
Kết quả nghiên cứu: 96.2 success rate, 78 energy savings so với các phương pháp baseline

- CPU: 4 cores, 2.5GHz (8 cores, 3.0GHz khuyến nghị)
- RAM: 8GB (16GB khuyến nghị)
- GPU: Không bắt buộc (NVIDIA RTX 3070+ khuyến nghị cho training)
- Storage: 20GB trống (50GB SSD khuyến nghị)
- Hệ điều hành: Ubuntu 20.04 LTS (hỗ trợ tốt nhất)

- Python: 3.8, 3.9, hoặc 3.10 (khuyến nghị 3.9)
- Conda/Miniconda: 4.12+ (khuyến nghị)
- CUDA: 11.7+ (nếu sử dụng GPU)
- Git: 2.0+
- CMake: 3.10+

DroneDelivery-RL/
 src/
    bridges/
    environment/
    localization/
    planning/
    rl/
    utils/
 scripts/
    evaluation/
    setup/
    training/
    utilities/
 config/
 data/
 models/
 results/
 docker/
 ros_ws/
 docs/

- PPO Agent: Chính sách điều khiển drone
- Curriculum Learning: 3 giai đoạn training (1 floor  2 floors  5 floors)
- Energy Reward: Hàm thưởng tối ưu hóa năng lượng

- A Global Planner: Tìm đường tối ưu
- S-RRT Local Planner: Tránh vật cản động
- Multi-floor Planning: Điều hướng giữa các tầng

- VI-SLAM: Định vị thị giác-inertial
- Stereo Vision: Nhận diện độ sâu
- IMU Integration: Theo dõi vị trí chính xác

- 5-Floor Building: Mô phỏng nội bộ 5 tầng
- Dynamic Obstacles: Vật cản di chuyển
- Energy Modeling: Mô hình tiêu thụ năng lượng

bash
wget https:
bash Miniconda3-latest-Linux-x86_64.sh

source /.bashrc

bash
git clone https://github.com/HorizonWork/DroneDelivery-RL DroneDelivery-RL
cd DroneDelivery-RL

bash
conda env create -f environment.yml
conda activate drone-delivery-rl

conda create -n drone-delivery-rl python=3.9
conda activate drone-delivery-rl

bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -r requirements.txt

pip install -e .

bash
python scripts/setup/verify_installation.py

bash
python scripts/setup/build_environment.py

- Cài đặt Unreal Engine 4.27+
- Tải bản đồ AirSim phù hợp
- Cấu hình settings.json trong config/airsim/

bash
python -m pytest tests/test_syntax_only.py

python -m pytest tests/test_basic_imports.py

python -m pytest tests/

bash
python -c "from src.rl.agents.ppo_agent import PPOAgent; print('PPO Agent imported successfully')"
python -c "from src.environment.airsim_env import AirSimEnvironment; print('Environment imported successfully')"

bash
python scripts/training/train_phase.py --phase single_floor --timesteps 1000

bash
python scripts/training/train_full_curriculum.py --config config/main_config.yaml

bash
python scripts/evaluation/evaluate_model.py --model models/checkpoints/ppo_final.pt

bash
python scripts/evaluation/benchmark_baselines.py

bash
python scripts/utilities/analyze_energy.py --evaluation-results results/model_evaluation.json

- IDE: VS Code với extension Python
- Git: Quản lý phiên bản
- Docker: Container hóa (nếu cần)
- TensorBoard: Theo dõi training

- CUDA-capable GPU: Tăng tốc training (khuyến nghị RTX 3070+)
- Bộ nhớ RAM: Ít nhất 16GB cho training hiệu quả
- SSD: Tăng tốc đọc ghi dữ liệu

- AirSim: Mô phỏng thực tế (tùy chọn)
- PyBullet: Mô phỏng vật lý (mặc định)

bash
python scripts/setup/verify_installation.py

python tests/benchmarks/system_benchmark.py

- [ ] Môi trường Python hoạt động đúng
- [ ] Tất cả dependencies được cài đặt
- [ ] Có thể import tất cả modules chính
- [ ] Training có thể bắt đầu
- [ ] Đánh giá mô hình hoạt động
- [ ] Đạt được 96 success rate trong evaluation

- CUDA không hoạt động: Kiểm tra driver GPU và CUDA toolkit
- Memory overflow: Giảm batch size trong cấu hình
- Import lỗi: Kiểm tra môi trường conda và dependencies

bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

python -c "import torch, numpy, gymnasium; print(f'PyTorch: {torch.__version__}')"

- Sử dụng GPU nếu có
- Điều chỉnh batch size phù hợp với RAM
- Sử dụng multi-processing cho môi trường

- Sử dụng chính sách deterministic
- Tối ưu hóa observation space
- Giảm tần suất cập nhật

---

Lưu ý: Dự án cần 8-12 giờ để training hoàn chỉnh 5 triệu timesteps. Đảm bảo hệ thống có đủ tài nguyên và thời gian để hoàn thành quá trình training.