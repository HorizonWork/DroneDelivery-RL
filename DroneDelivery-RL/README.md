[![Python 3.8+](https:
[![PyTorch](https:
[![License: MIT](https:

Indoor Multi-Floor UAV Delivery with Energy-Aware Navigation through A, S-RRT and Reinforcement Learning

  Research Achievement: 96.2 success rate, 78 energy savings vs baselines

  Table 3 Results: Outperforms A Only, RRT+PID, and Random methods

  Energy Efficiency: 610J average consumption (vs 2800J baseline)

---

This system integrates Visual-Inertial SLAM, A global planning, S-RRT local replanning, and PPO reinforcement learning for energy-efficient indoor drone delivery in 5-floor buildings.

-  PPO-based Navigation: Curriculum learning with 5M timesteps
-  Multi-Floor Environment: 5-floor building simulation
-  Energy Optimization: 25+ energy savings
-  High Success Rate: 96+ navigation success
-  AirSim Integration: Realistic simulation support
-  Research Ready: Table 3 reproduction capabilities

---

git clone https://github.com/HorizonWork/DroneDelivery-RL DroneDelivery-RL
cd DroneDelivery-RL

python scripts/setup/build_environment.py

python scripts/training/train_ppo.py

python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/ppo_final.pt

python scripts/evaluation/benchmark_baselines.py

python scripts/utilities/analyze_energy.py \
    --evaluation-results results/model_evaluation.json

python scripts/utilities/visualize_results.py \
    --evaluation-results results/model_evaluation.json

---

 Method  Success Rate  Energy (J)  Time (s)  Collisions  ATE (cm)
-------------------------------------------
 A Only  75.0  2800450  95.0  8.0  4.5
 RRT+PID  88.0  2400380  78.0  4.0  3.8
 Random  12.0  350800  120.0  35.0  8.0
 PPO (Ours)  96.2  61030  31.5  0.7  0.8

-  Success Rate: 96.2 (Target: 96)
-  Energy Savings: 78 vs A Only (Target: 25)
-  Safety: 0.7 collision rate (Target: 2)
-  Precision: 0.8cm ATE (Target: 5cm)

---

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
  docs/
  ros_ws/

---

-  [Setup Guide](scripts/setup/HUONG_DAN_CAI_DAT.md) - Complete installation instructions
-  [System Requirements](docs/SYSTEM_REQUIREMENTS.md) - Hardware  software requirements

-  [Training Guide](scripts/training/HUONG_DAN_TRAINING.md) - PPO training workflow
-  [Evaluation Guide](scripts/evaluation/HUONG_DAN_EVALUATION.md) - Model evaluation process

-  [Utilities Guide](scripts/utilities/HUONG_DAN_UTILITIES.md) - Analysis tools usage
-  [Results Analysis](docs/RESULTS_ANALYSIS.md) - Understanding evaluation metrics

-  [AirSim Integration](docs/AIRSIM_INTEGRATION.md) - Connect with AirSim simulator
-  [ROS Integration](docs/ROS_INTEGRATION.md) - Real hardware deployment
-  [Docker Deployment](docker/README.md) - Containerized deployment

---

python scripts/setup/build_environment.py

python scripts/training/hyperparameter_search.py --trials 50

python scripts/training/train_full_curriculum.py

python scripts/evaluation/evaluate_model.py --model models/checkpoints/ppo_final.pt
python scripts/evaluation/benchmark_baselines.py

python scripts/utilities/visualize_results.py \
    --evaluation-results results/model_evaluation.json \
    --baseline-results results/baseline_benchmark.json

python scripts/utilities/analyze_energy.py \
    --evaluation-results results/model_evaluation.json

-  Table 3 comparison: Performance vs baselines
-  Training curves: Learning progression plots
-  Energy analysis: Consumption patterns  savings
-  Target validation: Research objectives achievement

---

- PO Agent: Policy gradient optimization
- Curriculum Learning: 3-phase progressive training
- Energy Reward: Custom reward function for efficiency

- A Global Planner: Optimal path finding
- S-RRT Local Planner: Dynamic obstacle avoidance
- Multi-floor Planning: Staircase  elevator navigation

- VI-SLAM: Visual-inertial pose estimation
- Stereo Vision: Depth perception for obstacles
- IMU Integration: Robust pose tracking

- 5-Floor Building: Realistic indoor simulation
- Dynamic Obstacles: Moving obstacles  humans
- Energy Modeling: Realistic power consumption

---

- Success Rate: 96.2 (vs 75 baseline)
- Energy Consumption: 610J (vs 2800J baseline)
- Flight Time: 31.5s average
- Collision Rate: 0.7 (safety critical)

- ATE Error: 0.8cm (centimeter-scale precision)
- RPE Error: 0.12 (drift minimal)
- SLAM Consistency: 99.8 successful tracking

- Energy Savings: 78 vs A Only
- Battery Life: 59 missions per charge
- Power Distribution: 70 thrust, 20 avionics, 10 other

---

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- AirSim (optional)
- ROS Noetic (for real deployment)

pip install -e .[dev]

python run_tests.py

python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/ -k "airsim" -v
python -m pytest tests/ -k "environment" -v
python -m pytest tests/ -k "sensor" -v
python -m pytest tests/ -k "slam" -v
python -m pytest tests/ -k "ros" -v

black src/ scripts/
flake8 src/ scripts/

1. Fork the repository
2. Create feature branch (git checkout -b feature/amazing-feature)
3. Commit changes (git commit -m 'Add amazing feature')
4. Push to branch (git push origin feature/amazing-feature)
5. Open Pull Request

---

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

- FPT University - Research support
- Microsoft AirSim - Simulation platform
- PyTorch Community - Deep learning framework
- Research Papers - VI-SLAM, PPO, and planning algorithms

---

- Authors: Huynh Nhut Huy, Nguyen Ly Minh Ky, Luong Danh Doanh, Nguyen Huy Hoang
- Institution: FPT University Ho Chi Minh City
- Email: [contact information]
- Project: Indoor Multi-Floor UAV Delivery Research

---

If you use this work in your research, please cite:

article{huy2025dronedelivery,
  title={Indoor Multi-Floor UAV Delivery: Energy-Aware Navigation through A, S-RRT and Reinforcement Learning},
  author={Huy, Huynh Nhut and Ky, Nguyen Ly Minh and Doanh, Luong Danh and Hoang, Nguyen Huy},
  journal={FPT University Research},
  year={2025},
  publisher={FPT University Ho Chi Minh City}
}

---

 Ready for indoor drone delivery research and deployment!
