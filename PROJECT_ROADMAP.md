Project: Energy-Aware UAV Navigation with Reinforcement Learning
Repository: DroneDelivery-RL
Branch: dev_minhky
Last Updated: November 13, 2025

---

Tái hiện pipeline từ paper với các thành phần:
- A - Global path planning
- S-RRT - Local replanning với dynamic obstacles
- PPO - Energy-aware control với RL

-  Không dùng SLAM/ROS: sử dụng ground-truth pose từ simulator
-  Môi trường: AirSim/Custom 3D simulator (5 tầng)
-  Đầu ra: Bảng kết quả tương đương Table 3 trong paper

---

Có môi trường mô phỏng hoạt động và mô hình RL khởi động được.

- [ ] Cài Python 3.8+, PyTorch, Gym/AirSim
- [ ] Cài matplotlib, numpy, pandas, scipy, pyyaml
- [ ] Test GPU availability (CUDA)

Lệnh thực thi:
powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

conda activate drone-delivery-rl

pip install torch torchvision --index-url https:
pip install gymnasium airsim-python matplotlib numpy pandas scipy pyyaml tqdm tensorboard opencv-python

- [ ] Tạo configs/, data/, models/, results/, scripts/
- [ ] Cập nhật .gitignore

Cấu trúc:

drone_rl/
 data/
 models/
 results/
 runs/
 src/
    environment/
    planning/
    rl/
    baselines/
 scripts/
    train_curriculum.py
    evaluate.py
    generate_report.py
 requirements.txt

 Mục đích: Tạo discrete 3D grid map để A/RRT planning (KHÔNG phải Unreal map)

2 cách tiếp cận:

- [ ] Tạo scripts/setup/extract_map_from_unreal.py
- [ ] Dùng Lidar/Raycast để scan Unreal world
- [ ] Convert continuous 3D  discrete grid (0.5m resolution)
- [ ] Align tọa độ Unreal  Grid coordinates
- [ ] Save  data/map_5floor.pkl

Lệnh:
powershell
python scripts/setup/extract_map_from_unreal.py --resolution 0.5 --output data/map_5floor.pkl

- [ ] Tạo src/environment/map_generator.py
- [ ] Generate 3D occupancy grid: 20405 ô, cell = 0.5m
- [ ] Thêm tường, chướng ngại, cầu thang (synthetic)
- [ ] Save map  data/map_5floor.pkl
- [ ] Lưu ý: Sau này cần sync với Unreal layout!

Lệnh:
powershell
python src/environment/map_generator.py

Kiểm tra:
python
from src.environment.map_generator import MultiFloorMapGenerator
data = MultiFloorMapGenerator.load()
print(f"Map shape: {data['grid'].shape}")
print(f"Resolution: {data['cell_size']}m")
print(f"Free space: {(data['grid']==0).mean()100:.1f}")

 Lưu ý quan trọng:
- Occupancy grid  Unreal visual map
- Grid dùng cho planning algorithms (A, RRT)
- UAV vẫn bay trong Unreal (continuous physics)
- Cần đảm bảo tọa độ grid  Unreal khớp nhau!

- [ ] Trọng lượng: 1.5 kg
- [ ] Max speed: 5 m/s
- [ ] Max thrust: 15 N/motor
- [ ] Battery: 5000 mAh

Specs (theo paper Table 1):
 Parameter  Value
------------------
 Mass  1.5 kg
 Max Speed  5 m/s
 Max Thrust  15 N/motor
 Battery  5000 mAh
 Motors  4

- [ ] Random start/goal giữa các tầng
- [ ] Validate free space (không có obstacle)
- [ ] Store test scenarios

- [ ] UAV spawn đúng vị trí
- [ ] UAV di chuyển được
- [ ] Nhận được pose, velocity, battery
- [ ] Lidar/sensor khoảng cách hoạt động

File: src/environment/uav_env.py
Test:
powershell
python -c "from src.environment.uav_env import UAVDeliveryEnv; env = UAVDeliveryEnv(); obs, info = env.reset(); print(' Environment OK')"

- [ ] UAV đi theo A path với PID controller
- [ ] Không crash
- [ ] Log metrics (time, distance, collisions)

-  Simulator hoạt động
-  Có thể nhận pose, goal vector, battery
-  Không crash khi test
-  Bản đồ 5 tầng được tạo

---

Tạo hệ thống tìm đường an toàn và linh hoạt.

- [ ] Input: start, goal, occupancy grid
- [ ] Output: danh sách waypoint 3D
- [ ] Heuristic: Euclidean + penalty khi đổi tầng
- [ ] 6-connectivity (x, y, z directions)
- [ ] Collision checking với static obstacles

File: src/planning/astar.py

Features:
python
class AStarPlanner:
    - plan(start, goal)  List[waypoint]
    - _heuristic(pos, goal)  float
    - _is_collision_free(pos)  bool
    - _reconstruct_path()  List[waypoint]

Test:
powershell
python -c "from src.planning.astar import AStarPlanner; from src.environment.map_generator import MultiFloorMapGenerator; data = MultiFloorMapGenerator.load(); planner = AStarPlanner(data['grid']); path = planner.plan((5,5,0), (10,10,-6)); print(f' A OK: {len(path)} waypoints')"

- [ ] Kích hoạt khi có chướng ngại động gần UAV
- [ ] Cost function: C = ℓ + λc/dmin + λκκ²
  - ℓ: path length
  - dmin: minimum distance to obstacles
  - κ: curvature
- [ ] Bảo đảm tránh va chạm
- [ ] Quỹ đạo mượt (smooth trajectory)

File: src/planning/srrt_star.py

Features:
python
class SRRTStar:
    - plan(start, goal, obstacles)  List[waypoint]
    - _random_position()  np.ndarray
    - _nearest_node(pos)  Node
    - _steer(from, to)  np.ndarray
    - _is_collision_free(from, to)  bool
    - _choose_parent(node, near_nodes)  Node
    - _rewire(node, near_nodes)
    - _path_cost(from, to)  float

Cost function parameters:
- λc = 1.0 (collision weight)
- λκ = 0.5 (curvature weight)

Test:
powershell
python -c "from src.planning.srrt_star import SRRTStar; import numpy as np; planner = SRRTStar(np.zeros((5,20,40))); path = planner.plan(np.array([5,5,0]), np.array([10,10,-3])); print(f' S-RRT OK: {len(path)} waypoints')"

- [ ] Spawn người/xe di chuyển
- [ ] Trigger replanning khi obstacle gần
- [ ] Verify collision avoidance

- [ ] Planning time (ms)
- [ ] Path length (m)
- [ ] Number of replanning events
- [ ] Success rate

-  Global path (A) ổn định
-  Local path (S-RRT) ổn định
-  UAV có thể tái lập đường khi obstacle xuất hiện
-  Log metrics đầy đủ

---

PPO học điều khiển tiết kiệm năng lượng, mượt và an toàn.

- [ ] Pose: [x, y, z, yaw] (4D)
- [ ] Velocity: [vx, vy, vz] (3D)
- [ ] Goal vector: [gx, gy, gz] (3D)
- [ ] Battery: [battery_normalized] (1D)
- [ ] Obstacle distances: 8 directions (8D)
- Total: 19D observation

Observation vector:

obs = [x, y, z, yaw, vx, vy, vz, goal_x, goal_y, goal_z, battery, d1, d2, ..., d8]

- [ ] Action: [vx, vy, vz, ω] normalized to [-1, 1]
- [ ] Denormalize: vx,vy,vz  [-5, 5] m/s, ω  [-π, π] rad/s

Action space:

action = [vx, vy, vz, omega]

- [ ] Goal reached: +500  1goal
- [ ] Distance penalty: -5  dt
- [ ] Time penalty: -0.1  Δt
- [ ] Control effort: -0.01  Σui²
- [ ] Jerk penalty: -10  jt
- [ ] Collision penalty: -1000  ct

Reward equation:

R = 5001goal - 5dt - 0.1Δt - 0.01Σui² - 10jt - 1000ct

Components:
 Term  Weight  Description
---------------------------
 Goal reached  +500  Terminal reward
 Distance  -5  Closer to goal = better
 Time  -0.1  Faster = better
 Control  -0.01  Smooth control
 Jerk  -10  Smooth trajectory
 Collision  -1000  Safety critical

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

Config file: config/training/ppo_hyperparameters.yaml

- [ ] Stage 1: 1 tầng, obstacle tĩnh
  - Success threshold: 85
  - Episodes: 1000
- [ ] Stage 2: 2 tầng, obstacle động
  - Success threshold: 90
  - Episodes: 2000
- [ ] Stage 3: 5 tầng, full dynamic
  - Success threshold: 95
  - Episodes: 3000

File: src/rl/curriculum.py

Curriculum stages:
python
stages = [
    {'name': 'Stage1', 'floors': 1, 'dynamic': False, 'threshold': 0.85},
    {'name': 'Stage2', 'floors': 2, 'dynamic': True, 'threshold': 0.90},
    {'name': 'Stage3', 'floors': 5, 'dynamic': True, 'threshold': 0.95}
]

- [ ] Tạo scripts/train_curriculum.py
- [ ] Implement curriculum manager
- [ ] Save checkpoints mỗi stage
- [ ] Log to TensorBoard
- [ ] Periodic model saving (every 100 episodes)

Lệnh training:
powershell
python scripts/train_curriculum.py

tensorboard --logdir=runs --port=6006

Trong browser:

http:

- [ ] Average reward
- [ ] Energy consumption
- [ ] Collisions  0
- [ ] Success rate  95
- [ ] Policy loss, value loss
- [ ] Entropy (exploration)

Key metrics to track:
- Episode reward (target:  400)
- Success rate (target:  95)
- Energy per episode (target:  650J)
- Collision rate (target:  1)
- Training time per stage

-  Model PPO đã huấn luyện (5M timesteps)
-  Checkpoints: checkpoint_stage0.pt, checkpoint_stage1.pt, checkpoint_stage2.pt
-  Final model: final_TIMESTAMP.pt
-  TensorBoard logs đầy đủ
-  Success rate  95 ở stage 3

Expected training time:
- Stage 1: 3-4 hours
- Stage 2: 4-5 hours
- Stage 3: 3-4 hours
- Total: 12 hours

---

Sinh bảng kết quả và biểu đồ hiệu năng.

- [ ] Load trained model
- [ ] Run 200 episodes với random start-goal
- [ ] Log metrics: success, energy, time, collisions, ATE
- [ ] Save trajectories

Metrics to collect:
 Metric  Unit  Description
---------------------------
 Success Rate    Reached goal within threshold
 Energy  J (Joules)  Total energy consumed
 Flight Time  s  Episode duration
 Collision Rate    Percentage of collisions
 ATE  m  Average Trajectory Error
 Final Distance  m  Distance to goal at end

- [ ] Implement src/baselines/astar_pid.py
- [ ] A global planning
- [ ] PID controller (kp=1.0, ki=0.1, kd=0.5)
- [ ] Waypoint following
- [ ] Run 200 episodes

- [ ] Implement src/baselines/rrt_pid.py
- [ ] RRT global planning
- [ ] PID controller
- [ ] Run 200 episodes

- [ ] Random actions (baseline)
- [ ] Run 100 episodes

Lệnh chạy baselines:
powershell
python scripts/evaluate.py

- [ ] Tạo pandas DataFrame
- [ ] Columns: Method, Success(), Energy(J), Time(s), Collision()
- [ ] Save to CSV

Expected format:
csv
Method,Success Rate (),Energy (J),Time (s),Collision Rate (),Final Distance (m)
PPO,95.5,610  45,31.2  3.4,0.8,0.42  0.15
A+PID,92.3,820  67,32.1  4.2,1.2,0.68  0.23
RRT+PID,94.1,720  58,35.3  5.1,2.0,0.55  0.19
Random,12.5,1450  230,28.6  8.9,45.3,8.34  3.67

- [ ] Energy distribution per method
- [ ] Statistical significance (t-test)
- [ ] Energy savings percentage

Analysis:

Energy savings (PPO vs A+PID):
  Reduction: (820 - 610) / 820 = 25.6

- [ ] Reward curve (training)
- [ ] Energy vs Time scatter
- [ ] Success rate bar chart
- [ ] Collision rate comparison
- [ ] Box plots for distributions

Plots to generate:
1. Training curves (reward, loss)
2. Success rate comparison (bar chart)
3. Energy consumption (box plot)
4. Flight time distribution (box plot)
5. Collision rate (bar chart)
6. Trajectory visualization (3D plot)

Save: results/comparison_plots.png

- [ ] Create comparison table
- [ ] Statistical analysis
- [ ] Generate summary report

 Bảng tương tự Table 3 trong paper:

 Phương pháp  Success ()  Energy (J)  Time (s)  Collisions ()
----------------------------------------------------------------
 A + PID  92  820  67  32  4  1.2
 RRT + PID  94  720  58  35  5  2.0
 PPO  95  610  45  31  3  1.0
 Random  12  1450  29  45

 Files generated:
- results/comparison_table.csv
- results/comparison_plots.png
- results/evaluation_metrics.json
- results/trajectories/ (trajectory data)

---

Xuất báo cáo cuối và chuẩn hóa repo.

- [ ] Tạo reproduction_report.md
- [ ] Sections:
  - Executive Summary
  - Methodology
  - Results  Analysis
  - Comparison với paper
  - Limitations
  - Future Work
- [ ] Embed biểu đồ, bảng kết quả
- [ ] Phân tích energy saving
- [ ] Bàn luận về performance

File: results/reproduction_report.md

Structure:
markdown

   2.1 Environment Setup
   2.2 Planning Algorithms
   2.3 RL Training
   3.1 Performance Comparison
   3.2 Energy Analysis
   3.3 Statistical Significance

- [ ] Move all plots to results/visualizations/
- [ ] Move logs to results/logs/
- [ ] Move metrics to results/metrics/
- [ ] Organize by experiment date

Organization:

results/
 metrics/
    comparison_table.csv
    ppo_metrics.json
    baseline_metrics.json
 visualizations/
    training_curves.png
    comparison_plots.png
    trajectory_3d.png
 logs/
    evaluation_log.txt
 reproduction_report.md

- [ ] Xóa temp files, cache
- [ ] Xóa unused notebooks
- [ ] Update .gitignore
- [ ] Clean __pycache__
- [ ] Remove large binary files (if not needed)

Clean commands:
powershell
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force  Remove-Item -Recurse -Force

Remove-Item -Path tempevents.out.tfevents. -Wait

Get-ChildItem models/.pt  Select-Object Name, {Name="Size(MB)";Expression={_.Length/1MB}}

python scripts/plot_training.py --logdir runs/

---

- [ ] AirSim installed and running
- [ ] Python environment configured
- [ ] GPU drivers updated (if using CUDA)
- [ ] Sufficient disk space (10GB)
- [ ] Repository cloned and on correct branch

Phase 1:
- [ ] Map generated successfully
- [ ] Environment tested without errors
- [ ] Can spawn UAV and read sensors

Phase 2:
- [ ] A finds valid paths
- [ ] S-RRT avoids obstacles
- [ ] Planning time  1s

Phase 3:
- [ ] Training started without crashes
- [ ] TensorBoard shows learning progress
- [ ] Checkpoints saved for each stage
- [ ] Success rate increases over time

Phase 4:
- [ ] All baselines evaluated
- [ ] Metrics CSV generated
- [ ] Plots created
- [ ] Results match paper trends

Phase 5:
- [ ] Report completed
- [ ] Code cleaned and documented
- [ ] Repository pushed to remote
- [ ] Results backed up

- [ ] Can reproduce results from scratch
- [ ] All tests pass
- [ ] Documentation is complete
- [ ] Code follows style guidelines
- [ ] No sensitive data in repo
- [ ] License file included

---

Repository: https:
Branch: dev_minhky
Date: November 13, 2025

For issues, create a GitHub issue or contact the team.

---

1. Use mixed precision training (torch.cuda.amp) for faster training
2. Parallelize environment rollouts with SubprocVecEnv
3. Cache map data to avoid repeated loading
4. Use compiled models (torch.compile) in PyTorch 2.0+

- Simulation only (no real-world validation)
- Ground-truth pose (no SLAM uncertainty)
- Fixed battery model (no degradation)
- Limited to 5-floor scenarios

- [ ] Add wind disturbances
- [ ] Implement battery degradation model
- [ ] Multi-agent coordination
- [ ] Real-world deployment pipeline
- [ ] Domain randomization for sim-to-real
- [ ] Hierarchical RL for larger spaces

---

Project is considered COMPLETE when:

1.  All 5 phases finished
2.  Success rate  90
3.  Energy savings  15 vs baseline
4.  Collision rate  2
5.  Results documented in report
6.  Code pushed to repository
7.  Reproducible from README

Project is considered EXCELLENT when:

1.  Success rate  95
2.  Energy savings  25
3.  Collision rate  1
4.  Training converges in  5M steps
5.  Comprehensive ablation study
6.  Publication-ready report

---

END OF ROADMAP

Last updated: November 13, 2025
Status: Ready for execution
Estimated completion: 2-3 days with overnight training
