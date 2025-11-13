# 🚁 DRONE DELIVERY RL - PROJECT ROADMAP

**Project:** Energy-Aware UAV Navigation with Reinforcement Learning  
**Repository:** DroneDelivery-RL  
**Branch:** dev_minhky  
**Last Updated:** November 13, 2025

---

## ✅ TỔNG QUAN CHIẾN LƯỢC

### Mục tiêu
Tái hiện pipeline từ paper với các thành phần:
- **A*** - Global path planning
- **S-RRT*** - Local replanning với dynamic obstacles
- **PPO** - Energy-aware control với RL

### Đặc điểm kỹ thuật
- ✅ **Không dùng SLAM/ROS**: sử dụng ground-truth pose từ simulator
- ✅ **Môi trường**: AirSim/Custom 3D simulator (5 tầng)
- ✅ **Đầu ra**: Bảng kết quả tương đương Table 3 trong paper

---

## 🧩 PHA 1 — CHUẨN BỊ & CẤU HÌNH (4 giờ)

### 🎯 Mục tiêu
Có môi trường mô phỏng hoạt động và mô hình RL khởi động được.

### ✅ Checklist

#### 1.1 Cài đặt môi trường
- [ ] Cài Python 3.8+, PyTorch, Gym/AirSim
- [ ] Cài matplotlib, numpy, pandas, scipy, pyyaml
- [ ] Test GPU availability (CUDA)

**Lệnh thực thi:**
```powershell
# Tạo virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Hoặc dùng conda
conda activate drone-delivery-rl

# Cài đặt dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium airsim-python matplotlib numpy pandas scipy pyyaml tqdm tensorboard opencv-python
```

#### 1.2 Tạo cấu trúc thư mục
- [ ] Tạo `configs/`, `data/`, `models/`, `results/`, `scripts/`
- [ ] Cập nhật `.gitignore`

**Cấu trúc:**
```
drone_rl/
├── data/               # Maps, trajectories
├── models/             # Trained models, checkpoints
├── results/            # Evaluation results, plots
├── runs/               # TensorBoard logs
├── src/
│   ├── environment/    # UAV env, map generator
│   ├── planning/       # A*, S-RRT*
│   ├── rl/             # PPO, curriculum
│   └── baselines/      # A*+PID, RRT*+PID
├── scripts/
│   ├── train_curriculum.py
│   ├── evaluate.py
│   └── generate_report.py
└── requirements.txt
```

#### 1.3 Khởi tạo Occupancy Grid Map cho Path Planning
**🎯 Mục đích:** Tạo discrete 3D grid map để A*/RRT* planning (KHÔNG phải Unreal map)

**2 cách tiếp cận:**

##### **Cách 1: Extract từ Unreal Environment (Recommended)**
- [ ] Tạo `scripts/setup/extract_map_from_unreal.py`
- [ ] Dùng Lidar/Raycast để scan Unreal world
- [ ] Convert continuous 3D → discrete grid (0.5m resolution)
- [ ] Align tọa độ Unreal ↔ Grid coordinates
- [ ] Save → `data/map_5floor.pkl`

**Lệnh:**
```powershell
# Cần AirSim running trong Unreal
python scripts/setup/extract_map_from_unreal.py --resolution 0.5 --output data/map_5floor.pkl
```

##### **Cách 2: Generate Synthetic Map (Faster for initial testing)**
- [ ] Tạo `src/environment/map_generator.py`
- [ ] Generate 3D occupancy grid: 20×40×5 ô, cell = 0.5m
- [ ] Thêm tường, chướng ngại, cầu thang (synthetic)
- [ ] Save map → `data/map_5floor.pkl`
- [ ] **Lưu ý:** Sau này cần sync với Unreal layout!

**Lệnh:**
```powershell
python src/environment/map_generator.py
```

**Kiểm tra:**
```python
from src.environment.map_generator import MultiFloorMapGenerator
data = MultiFloorMapGenerator.load()
print(f"Map shape: {data['grid'].shape}")  # Expected: (5, 20, 40)
print(f"Resolution: {data['cell_size']}m")  # 0.5m
print(f"Free space: {(data['grid']==0).mean()*100:.1f}%")  # Should be > 70%
```

**📌 Lưu ý quan trọng:**
- Occupancy grid ≠ Unreal visual map
- Grid dùng cho **planning algorithms** (A*, RRT*)
- UAV vẫn **bay trong Unreal** (continuous physics)
- Cần đảm bảo tọa độ grid ↔ Unreal khớp nhau!

#### 1.4 Thiết lập mô hình UAV
- [ ] Trọng lượng: ~1.5 kg
- [ ] Max speed: 5 m/s
- [ ] Max thrust: 15 N/motor
- [ ] Battery: 5000 mAh

**Specs (theo paper Table 1):**
| Parameter | Value |
|-----------|-------|
| Mass | 1.5 kg |
| Max Speed | 5 m/s |
| Max Thrust | 15 N/motor |
| Battery | 5000 mAh |
| Motors | 4 |

#### 1.5 Tạo danh sách start-goal ngẫu nhiên
- [ ] Random start/goal giữa các tầng
- [ ] Validate free space (không có obstacle)
- [ ] Store test scenarios

#### 1.6 Kiểm tra module environment
- [ ] UAV spawn đúng vị trí
- [ ] UAV di chuyển được
- [ ] Nhận được pose, velocity, battery
- [ ] Lidar/sensor khoảng cách hoạt động

**File:** `src/environment/uav_env.py`
**Test:**
```powershell
python -c "from src.environment.uav_env import UAVDeliveryEnv; env = UAVDeliveryEnv(); obs, info = env.reset(); print('✓ Environment OK')"
```

#### 1.7 Chạy thử 1 episode (không RL)
- [ ] UAV đi theo A* path với PID controller
- [ ] Không crash
- [ ] Log metrics (time, distance, collisions)

### 📤 Output Pha 1
- ✅ Simulator hoạt động
- ✅ Có thể nhận pose, goal vector, battery
- ✅ Không crash khi test
- ✅ Bản đồ 5 tầng được tạo

---

## ⚙️ PHA 2 — LẬP KẾ HOẠCH ĐƯỜNG ĐI (5 giờ)

### 🎯 Mục tiêu
Tạo hệ thống tìm đường an toàn và linh hoạt.

### ✅ Checklist

#### 2.1 Viết module A*
- [ ] Input: start, goal, occupancy grid
- [ ] Output: danh sách waypoint 3D
- [ ] Heuristic: Euclidean + penalty khi đổi tầng
- [ ] 6-connectivity (x, y, z directions)
- [ ] Collision checking với static obstacles

**File:** `src/planning/astar.py`

**Features:**
```python
class AStarPlanner:
    - plan(start, goal) → List[waypoint]
    - _heuristic(pos, goal) → float
    - _is_collision_free(pos) → bool
    - _reconstruct_path() → List[waypoint]
```

**Test:**
```powershell
python -c "from src.planning.astar import AStarPlanner; from src.environment.map_generator import MultiFloorMapGenerator; data = MultiFloorMapGenerator.load(); planner = AStarPlanner(data['grid']); path = planner.plan((5,5,0), (10,10,-6)); print(f'✓ A* OK: {len(path)} waypoints')"
```

#### 2.2 Viết module S-RRT*
- [ ] Kích hoạt khi có chướng ngại động gần UAV
- [ ] Cost function: C = ℓ + λc/dmin + λκ·κ²
  - ℓ: path length
  - dmin: minimum distance to obstacles
  - κ: curvature
- [ ] Bảo đảm tránh va chạm
- [ ] Quỹ đạo mượt (smooth trajectory)

**File:** `src/planning/srrt_star.py`

**Features:**
```python
class SRRTStar:
    - plan(start, goal, obstacles) → List[waypoint]
    - _random_position() → np.ndarray
    - _nearest_node(pos) → Node
    - _steer(from, to) → np.ndarray
    - _is_collision_free(from, to) → bool
    - _choose_parent(node, near_nodes) → Node
    - _rewire(node, near_nodes)
    - _path_cost(from, to) → float
```

**Cost function parameters:**
- λc = 1.0 (collision weight)
- λκ = 0.5 (curvature weight)

**Test:**
```powershell
python -c "from src.planning.srrt_star import SRRTStar; import numpy as np; planner = SRRTStar(np.zeros((5,20,40))); path = planner.plan(np.array([5,5,0]), np.array([10,10,-3])); print(f'✓ S-RRT* OK: {len(path)} waypoints')"
```

#### 2.3 Kiểm tra bằng mô phỏng obstacle động
- [ ] Spawn người/xe di chuyển
- [ ] Trigger replanning khi obstacle gần
- [ ] Verify collision avoidance

#### 2.4 Ghi log thời gian, độ dài đường, số lần replanning
- [ ] Planning time (ms)
- [ ] Path length (m)
- [ ] Number of replanning events
- [ ] Success rate

### 📤 Output Pha 2
- ✅ Global path (A*) ổn định
- ✅ Local path (S-RRT*) ổn định
- ✅ UAV có thể tái lập đường khi obstacle xuất hiện
- ✅ Log metrics đầy đủ

---

## 🤖 PHA 3 — HUẤN LUYỆN REINFORCEMENT LEARNING (12 giờ)

### 🎯 Mục tiêu
PPO học điều khiển tiết kiệm năng lượng, mượt và an toàn.

### ✅ Checklist

#### 3.1 Thiết lập observation space (ground-truth)
- [ ] Pose: [x, y, z, yaw] (4D)
- [ ] Velocity: [vx, vy, vz] (3D)
- [ ] Goal vector: [gx, gy, gz] (3D)
- [ ] Battery: [battery_normalized] (1D)
- [ ] Obstacle distances: 8 directions (8D)
- **Total:** 19D observation

**Observation vector:**
```
obs = [x, y, z, yaw, vx, vy, vz, goal_x, goal_y, goal_z, battery, d1, d2, ..., d8]
```

#### 3.2 Thiết lập action space
- [ ] Action: [vx, vy, vz, ω] normalized to [-1, 1]
- [ ] Denormalize: vx,vy,vz → [-5, 5] m/s, ω → [-π, π] rad/s

**Action space:**
```
action = [vx, vy, vz, omega]  # 4D continuous
```

#### 3.3 Reward function (Eq.2 trong paper)
- [ ] Goal reached: +500 × 1goal
- [ ] Distance penalty: -5 × dt
- [ ] Time penalty: -0.1 × Δt
- [ ] Control effort: -0.01 × Σui²
- [ ] Jerk penalty: -10 × jt
- [ ] Collision penalty: -1000 × ct

**Reward equation:**
```
R = 500·1goal - 5·dt - 0.1·Δt - 0.01·Σui² - 10·jt - 1000·ct
```

**Components:**
| Term | Weight | Description |
|------|--------|-------------|
| Goal reached | +500 | Terminal reward |
| Distance | -5 | Closer to goal = better |
| Time | -0.1 | Faster = better |
| Control | -0.01 | Smooth control |
| Jerk | -10 | Smooth trajectory |
| Collision | -1000 | Safety critical |

#### 3.4 Cấu hình PPO hyperparameters
- [ ] Learning rate = 3e-4
- [ ] Clip epsilon = 0.2
- [ ] Gamma (γ) = 0.99
- [ ] Lambda (λ) = 0.95 (GAE)
- [ ] Hidden layers: [256, 128, 64]
- [ ] Batch size = 64
- [ ] Rollout buffer = 2048 steps
- [ ] Entropy coefficient = 0.01
- [ ] Value loss coefficient = 0.5
- [ ] Max grad norm = 0.5

**Config file:** `config/training/ppo_hyperparameters.yaml`

#### 3.5 Huấn luyện 3 giai đoạn (curriculum)
- [ ] **Stage 1**: 1 tầng, obstacle tĩnh
  - Success threshold: 85%
  - Episodes: 1000
- [ ] **Stage 2**: 2 tầng, obstacle động
  - Success threshold: 90%
  - Episodes: 2000
- [ ] **Stage 3**: 5 tầng, full dynamic
  - Success threshold: 95%
  - Episodes: 3000

**File:** `src/rl/curriculum.py`

**Curriculum stages:**
```python
stages = [
    {'name': 'Stage1', 'floors': 1, 'dynamic': False, 'threshold': 0.85},
    {'name': 'Stage2', 'floors': 2, 'dynamic': True, 'threshold': 0.90},
    {'name': 'Stage3', 'floors': 5, 'dynamic': True, 'threshold': 0.95}
]
```

#### 3.6 Training script
- [ ] Tạo `scripts/train_curriculum.py`
- [ ] Implement curriculum manager
- [ ] Save checkpoints mỗi stage
- [ ] Log to TensorBoard
- [ ] Periodic model saving (every 100 episodes)

**Lệnh training:**
```powershell
# Train với curriculum (chạy qua đêm ~12h)
python scripts/train_curriculum.py

# Monitor training
tensorboard --logdir=runs --port=6006
```

**Trong browser:**
```
http://localhost:6006
```

#### 3.7 Theo dõi metrics
- [ ] Average reward ↑
- [ ] Energy consumption ↓
- [ ] Collisions → 0
- [ ] Success rate ≥ 95%
- [ ] Policy loss, value loss
- [ ] Entropy (exploration)

**Key metrics to track:**
- Episode reward (target: > 400)
- Success rate (target: > 95%)
- Energy per episode (target: < 650J)
- Collision rate (target: < 1%)
- Training time per stage

### 📤 Output Pha 3
- ✅ Model PPO đã huấn luyện (~5M timesteps)
- ✅ Checkpoints: `checkpoint_stage0.pt`, `checkpoint_stage1.pt`, `checkpoint_stage2.pt`
- ✅ Final model: `final_TIMESTAMP.pt`
- ✅ TensorBoard logs đầy đủ
- ✅ Success rate ≥ 95% ở stage 3

**Expected training time:**
- Stage 1: ~3-4 hours
- Stage 2: ~4-5 hours
- Stage 3: ~3-4 hours
- **Total: ~12 hours**

---

## 📊 PHA 4 — ĐÁNH GIÁ & SO SÁNH (6 giờ)

### 🎯 Mục tiêu
Sinh bảng kết quả và biểu đồ hiệu năng.

### ✅ Checklist

#### 4.1 Chạy Evaluation với PPO
- [ ] Load trained model
- [ ] Run 200 episodes với random start-goal
- [ ] Log metrics: success, energy, time, collisions, ATE
- [ ] Save trajectories

**Metrics to collect:**
| Metric | Unit | Description |
|--------|------|-------------|
| Success Rate | % | Reached goal within threshold |
| Energy | J (Joules) | Total energy consumed |
| Flight Time | s | Episode duration |
| Collision Rate | % | Percentage of collisions |
| ATE | m | Average Trajectory Error |
| Final Distance | m | Distance to goal at end |

#### 4.2 Chạy Baselines

##### 4.2.1 A* + PID
- [ ] Implement `src/baselines/astar_pid.py`
- [ ] A* global planning
- [ ] PID controller (kp=1.0, ki=0.1, kd=0.5)
- [ ] Waypoint following
- [ ] Run 200 episodes

##### 4.2.2 RRT* + PID
- [ ] Implement `src/baselines/rrt_pid.py`
- [ ] RRT* global planning
- [ ] PID controller
- [ ] Run 200 episodes

##### 4.2.3 Random Policy
- [ ] Random actions (baseline)
- [ ] Run 100 episodes

**Lệnh chạy baselines:**
```powershell
# Evaluate all methods
python scripts/evaluate.py
```

#### 4.3 Ghi dữ liệu → results/metrics.csv
- [ ] Tạo pandas DataFrame
- [ ] Columns: Method, Success(%), Energy(J), Time(s), Collision(%)
- [ ] Save to CSV

**Expected format:**
```csv
Method,Success Rate (%),Energy (J),Time (s),Collision Rate (%),Final Distance (m)
PPO,95.5,610 ± 45,31.2 ± 3.4,0.8,0.42 ± 0.15
A*+PID,92.3,820 ± 67,32.1 ± 4.2,1.2,0.68 ± 0.23
RRT*+PID,94.1,720 ± 58,35.3 ± 5.1,2.0,0.55 ± 0.19
Random,12.5,1450 ± 230,28.6 ± 8.9,45.3,8.34 ± 3.67
```

#### 4.4 Phân tích năng lượng (mean ± std)
- [ ] Energy distribution per method
- [ ] Statistical significance (t-test)
- [ ] Energy savings percentage

**Analysis:**
```
Energy savings (PPO vs A*+PID):
  Reduction: (820 - 610) / 820 = 25.6%
```

#### 4.5 Vẽ biểu đồ
- [ ] Reward curve (training)
- [ ] Energy vs Time scatter
- [ ] Success rate bar chart
- [ ] Collision rate comparison
- [ ] Box plots for distributions

**Plots to generate:**
1. Training curves (reward, loss)
2. Success rate comparison (bar chart)
3. Energy consumption (box plot)
4. Flight time distribution (box plot)
5. Collision rate (bar chart)
6. Trajectory visualization (3D plot)

**Save:** `results/comparison_plots.png`

#### 4.6 So sánh kết quả PPO vs baselines
- [ ] Create comparison table
- [ ] Statistical analysis
- [ ] Generate summary report

### 📤 Output Pha 4
✅ **Bảng tương tự Table 3 trong paper:**

| Phương pháp | Success (%) | Energy (J) | Time (s) | Collisions (%) |
|-------------|-------------|------------|----------|----------------|
| A* + PID | ~92 | ~820 ± 67 | ~32 ± 4 | ~1.2 |
| RRT* + PID | ~94 | ~720 ± 58 | ~35 ± 5 | ~2.0 |
| **PPO** | **≥95** | **≤610 ± 45** | **~31 ± 3** | **≤1.0** |
| Random | ~12 | ~1450 | ~29 | ~45 |

✅ **Files generated:**
- `results/comparison_table.csv`
- `results/comparison_plots.png`
- `results/evaluation_metrics.json`
- `results/trajectories/` (trajectory data)

---

## 🧠 PHA 5 — BÁO CÁO & TỐI ƯU HÓA (4 giờ)

### 🎯 Mục tiêu
Xuất báo cáo cuối và chuẩn hóa repo.

### ✅ Checklist

#### 5.1 Xuất file báo cáo
- [ ] Tạo `reproduction_report.md`
- [ ] Sections:
  - Executive Summary
  - Methodology
  - Results & Analysis
  - Comparison với paper
  - Limitations
  - Future Work
- [ ] Embed biểu đồ, bảng kết quả
- [ ] Phân tích energy saving
- [ ] Bàn luận về performance

**File:** `results/reproduction_report.md`

**Structure:**
```markdown
# Energy-Aware UAV Navigation - Reproduction Report

## 1. Executive Summary
## 2. Methodology
   2.1 Environment Setup
   2.2 Planning Algorithms
   2.3 RL Training
## 3. Results
   3.1 Performance Comparison
   3.2 Energy Analysis
   3.3 Statistical Significance
## 4. Discussion
## 5. Limitations
## 6. Future Work
```

#### 5.2 Gộp log, biểu đồ, kết quả vào results/
- [ ] Move all plots to `results/visualizations/`
- [ ] Move logs to `results/logs/`
- [ ] Move metrics to `results/metrics/`
- [ ] Organize by experiment date

**Organization:**
```
results/
├── metrics/
│   ├── comparison_table.csv
│   ├── ppo_metrics.json
│   └── baseline_metrics.json
├── visualizations/
│   ├── training_curves.png
│   ├── comparison_plots.png
│   └── trajectory_3d.png
├── logs/
│   └── evaluation_log.txt
└── reproduction_report.md
```

#### 5.3 Dọn repo: giữ src/, scripts/, configs/, results/, models/
- [ ] Xóa temp files, cache
- [ ] Xóa unused notebooks
- [ ] Update `.gitignore`
- [ ] Clean `__pycache__`
- [ ] Remove large binary files (if not needed)

**Clean commands:**
```powershell
# Remove cache
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force

# Remove temp files
Remove-Item -Path temp/* -Recurse -Force

# Clean pytest cache
Remove-Item -Path .pytest_cache -Recurse -Force
```

#### 5.4 Backup mô hình PPO huấn luyện
- [ ] Copy final model to safe location
- [ ] Compress checkpoints
- [ ] Create model metadata (hyperparams, metrics)

**Backup:**
```powershell
# Backup models
Compress-Archive -Path models/* -DestinationPath backups/models_20251113.zip

# Create metadata
python scripts/generate_model_metadata.py
```

#### 5.5 Commit và push bản final
- [ ] Git add all changes
- [ ] Write comprehensive commit message
- [ ] Tag release version
- [ ] Push to remote

**Git commands:**
```powershell
# Stage changes
git add .

# Commit
git commit -m "feat: Complete PPO training and evaluation pipeline

- Implement 5-floor 3D environment with ground-truth pose
- Add A* and S-RRT* planning modules
- Train PPO with 3-stage curriculum learning
- Evaluate vs baselines (A*+PID, RRT*+PID)
- Achieve 95%+ success rate, 25% energy savings
- Generate comprehensive evaluation report

Results:
- Success: 95.5%
- Energy: 610±45 J (25% reduction vs A*+PID)
- Collision: <1%
"

# Tag release
git tag -a v1.0.0 -m "First complete reproduction"

# Push
git push origin dev_minhky
git push origin v1.0.0
```

#### 5.6 Write README.md
- [ ] Project overview
- [ ] Installation instructions
- [ ] Usage examples
- [ ] Results summary
- [ ] Citation

**README sections:**
```markdown
# 🚁 Energy-Aware UAV Delivery with RL

## Overview
## Features
## Installation
## Quick Start
## Training
## Evaluation
## Results
## Citation
## License
```

#### 5.7 Create CHANGELOG.md
- [ ] Document major changes
- [ ] Version history
- [ ] Breaking changes

### 📤 Output Pha 5
- ✅ Báo cáo hoàn chỉnh: `results/reproduction_report.md`
- ✅ Repo sạch, có thể chia sẻ/submit
- ✅ Models được backup
- ✅ Code documented đầy đủ
- ✅ README.md comprehensive
- ✅ Git history clean
- ✅ Release tagged (v1.0.0)

---

## 🚀 LỆNH CHẠY TOÀN BỘ PIPELINE

### Setup Environment
```powershell
# Activate environment
conda activate drone-delivery-rl
# hoặc
.\venv\Scripts\Activate.ps1

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Phase 1: Preparation (4h)
```powershell
# Generate 5-floor map
python src/environment/map_generator.py

# Test environment
python -c "from src.environment.uav_env import UAVDeliveryEnv; env = UAVDeliveryEnv(); obs, info = env.reset(); print('✓ Environment OK'); env.close()"
```

### Phase 2: Planning (5h)
```powershell
# Test A*
python -c "from src.planning.astar import AStarPlanner; from src.environment.map_generator import MultiFloorMapGenerator; data = MultiFloorMapGenerator.load(); planner = AStarPlanner(data['grid']); path = planner.plan((5,5,0), (10,10,-6)); print(f'✓ A* OK: {len(path)} waypoints')"

# Test S-RRT*
python -c "from src.planning.srrt_star import SRRTStar; import numpy as np; planner = SRRTStar(np.zeros((5,20,40))); path = planner.plan(np.array([5,5,0]), np.array([10,10,-3])); print(f'✓ S-RRT* OK: {len(path)} waypoints')"
```

### Phase 3: Training (12h - overnight)
```powershell
# Start training with curriculum
python scripts/train_curriculum.py

# Monitor in another terminal
tensorboard --logdir=runs --port=6006
# Open browser: http://localhost:6006
```

### Phase 4: Evaluation (6h)
```powershell
# Run comprehensive evaluation
python scripts/evaluate.py

# Check results
cat results/comparison_table.csv
```

### Phase 5: Report & Cleanup (4h)
```powershell
# Generate final report
python scripts/generate_report.py

# View report
cat results/reproduction_report.md

# Cleanup
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force

# Commit
git add .
git commit -m "feat: Complete reproduction with evaluation"
git push origin dev_minhky
```

---

## ⏱️ TIMELINE DỰ KIẾN

| Giai đoạn | Thời gian | Có thể chạy song song | Status |
|-----------|-----------|----------------------|--------|
| **PHA 1** | 4h | Không | ⬜ Pending |
| **PHA 2** | 5h | Không (cần Pha 1) | ⬜ Pending |
| **PHA 3** | 12h | Có (overnight) | ⬜ Pending |
| **PHA 4** | 6h | Không (cần Pha 3) | ⬜ Pending |
| **PHA 5** | 4h | Một phần | ⬜ Pending |

**Tổng thời gian:** ~31 giờ (có thể ~20-24 giờ nếu tối ưu)

**Schedule đề xuất:**
```
Day 1 (8h):
  09:00-13:00  Pha 1 + Pha 2 (setup + planning)
  14:00-18:00  Bắt đầu Pha 3 (training setup)
  18:00-06:00  Training qua đêm (Stage 1-3)

Day 2 (8h):
  09:00-15:00  Pha 4 (evaluation)
  15:00-18:00  Pha 5 (report + cleanup)
```

---

## 📊 EXPECTED RESULTS

### Target Metrics (theo paper)

| Metric | Target | Threshold |
|--------|--------|-----------|
| Success Rate | ≥ 95% | Pass if ≥ 90% |
| Energy (PPO) | ≤ 650 J | Pass if < 750 J |
| Energy Savings | ≥ 20% vs baseline | Pass if ≥ 15% |
| Collision Rate | ≤ 1% | Pass if ≤ 2% |
| Flight Time | ~30-35s | - |

### Key Comparisons

**PPO vs A*+PID:**
- ✅ Energy: 25-30% reduction
- ✅ Success: +3-5%
- ✅ Collision: -30-40%
- ✅ Time: similar or -5%

**PPO vs RRT*+PID:**
- ✅ Energy: 15-20% reduction
- ✅ Success: +1-2%
- ✅ Collision: -50%
- ✅ Time: -10-15%

---

## 🐛 COMMON ISSUES & SOLUTIONS

### Issue 1: AirSim connection failed
**Solution:**
```powershell
# Check AirSim is running
Get-Process | Where-Object {$_.Name -like "*airsim*"}

# Restart AirSim
# Verify settings.json in Documents/AirSim/
```

### Issue 2: CUDA out of memory
**Solution:**
```python
# Reduce batch size in config
batch_size = 32  # instead of 64

# Or reduce network size
hidden_dims = [128, 64, 32]  # instead of [256, 128, 64]
```

### Issue 3: Training not converging
**Solution:**
```python
# Adjust learning rate
lr = 1e-4  # instead of 3e-4

# Increase entropy coefficient
entropy_coef = 0.05  # instead of 0.01
```

### Issue 4: Low success rate in Stage 1
**Solution:**
- Check reward function weights
- Verify environment reset
- Increase training episodes (1500 instead of 1000)
- Reduce difficulty (smaller map, fewer obstacles)

### Issue 5: Path planning fails
**Solution:**
```python
# Check map validity
print(f"Free space: {(grid == 0).sum() / grid.size * 100:.1f}%")

# Should be > 70%

# Increase corridor widths
# Reduce obstacle density
```

---

## 📚 KEY FILES REFERENCE

### Core Implementation Files

| File | Purpose | Lines | Priority |
|------|---------|-------|----------|
| `src/environment/uav_env.py` | Main environment | ~400 | ⭐⭐⭐ |
| `src/environment/map_generator.py` | 3D map generation | ~150 | ⭐⭐⭐ |
| `src/planning/astar.py` | A* planner | ~200 | ⭐⭐⭐ |
| `src/planning/srrt_star.py` | S-RRT* planner | ~300 | ⭐⭐ |
| `src/rl/ppo.py` | PPO agent | ~400 | ⭐⭐⭐ |
| `src/rl/curriculum.py` | Curriculum manager | ~100 | ⭐⭐⭐ |
| `src/baselines/astar_pid.py` | A*+PID baseline | ~150 | ⭐⭐ |
| `scripts/train_curriculum.py` | Training script | ~300 | ⭐⭐⭐ |
| `scripts/evaluate.py` | Evaluation script | ~250 | ⭐⭐⭐ |
| `scripts/generate_report.py` | Report generator | ~150 | ⭐⭐ |

### Configuration Files

| File | Purpose |
|------|---------|
| `config/training/ppo_hyperparameters.yaml` | PPO hyperparams |
| `config/training/curriculum_config.yaml` | Curriculum stages |
| `config/training/reward_weights.yaml` | Reward function weights |
| `config/evaluation/test_scenarios.yaml` | Evaluation scenarios |
| `requirements.txt` | Python dependencies |

---

## 🎓 LEARNING RESOURCES

### Paper References
- Original paper: "Energy-Aware UAV Navigation..." (cite)
- PPO paper: Schulman et al. 2017
- RRT* paper: Karaman & Frazzoli 2011

### Code References
- Stable-Baselines3 PPO: https://github.com/DLR-RM/stable-baselines3
- AirSim Python API: https://microsoft.github.io/AirSim/
- PyTorch RL examples: https://github.com/pytorch/examples/tree/main/reinforcement_learning

### Useful Commands
```powershell
# Check GPU usage
nvidia-smi

# Monitor training progress
Get-Content runs/curriculum_*/events.out.tfevents.* -Wait

# Check model size
Get-ChildItem models/*.pt | Select-Object Name, @{Name="Size(MB)";Expression={$_.Length/1MB}}

# Plot training curves
python scripts/plot_training.py --logdir runs/
```

---

## ✅ FINAL CHECKLIST

### Before Starting
- [ ] AirSim installed and running
- [ ] Python environment configured
- [ ] GPU drivers updated (if using CUDA)
- [ ] Sufficient disk space (>10GB)
- [ ] Repository cloned and on correct branch

### After Each Phase
**Phase 1:**
- [ ] Map generated successfully
- [ ] Environment tested without errors
- [ ] Can spawn UAV and read sensors

**Phase 2:**
- [ ] A* finds valid paths
- [ ] S-RRT* avoids obstacles
- [ ] Planning time < 1s

**Phase 3:**
- [ ] Training started without crashes
- [ ] TensorBoard shows learning progress
- [ ] Checkpoints saved for each stage
- [ ] Success rate increases over time

**Phase 4:**
- [ ] All baselines evaluated
- [ ] Metrics CSV generated
- [ ] Plots created
- [ ] Results match paper trends

**Phase 5:**
- [ ] Report completed
- [ ] Code cleaned and documented
- [ ] Repository pushed to remote
- [ ] Results backed up

### Final Verification
- [ ] Can reproduce results from scratch
- [ ] All tests pass
- [ ] Documentation is complete
- [ ] Code follows style guidelines
- [ ] No sensitive data in repo
- [ ] License file included

---

## 📞 CONTACT & SUPPORT

**Repository:** https://github.com/HorizonWork/DroneDelivery-RL  
**Branch:** dev_minhky  
**Date:** November 13, 2025

For issues, create a GitHub issue or contact the team.

---

## 📝 NOTES & OBSERVATIONS

### Optimization Tips
1. Use mixed precision training (`torch.cuda.amp`) for faster training
2. Parallelize environment rollouts with `SubprocVecEnv`
3. Cache map data to avoid repeated loading
4. Use compiled models (`torch.compile`) in PyTorch 2.0+

### Known Limitations
- Simulation only (no real-world validation)
- Ground-truth pose (no SLAM uncertainty)
- Fixed battery model (no degradation)
- Limited to 5-floor scenarios

### Future Enhancements
- [ ] Add wind disturbances
- [ ] Implement battery degradation model
- [ ] Multi-agent coordination
- [ ] Real-world deployment pipeline
- [ ] Domain randomization for sim-to-real
- [ ] Hierarchical RL for larger spaces

---

## 🏆 SUCCESS CRITERIA

Project is considered **COMPLETE** when:

1. ✅ All 5 phases finished
2. ✅ Success rate ≥ 90%
3. ✅ Energy savings ≥ 15% vs baseline
4. ✅ Collision rate ≤ 2%
5. ✅ Results documented in report
6. ✅ Code pushed to repository
7. ✅ Reproducible from README

Project is considered **EXCELLENT** when:

1. ✅ Success rate ≥ 95%
2. ✅ Energy savings ≥ 25%
3. ✅ Collision rate ≤ 1%
4. ✅ Training converges in < 5M steps
5. ✅ Comprehensive ablation study
6. ✅ Publication-ready report

---

**END OF ROADMAP**

*Last updated: November 13, 2025*  
*Status: Ready for execution*  
*Estimated completion: 2-3 days with overnight training*
