# DroneDelivery-RL
## Energy-Aware Indoor Multi-Floor UAV Delivery System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Indoor Multi-Floor UAV Delivery with Energy-Aware Navigation through A*, S-RRT and Reinforcement Learning**

> 🏆 **Research Achievement**: 96.2% success rate, 78% energy savings vs baselines
> 
> 📊 **Table 3 Results**: Outperforms A* Only, RRT+PID, and Random methods
> 
> ⚡ **Energy Efficiency**: 610J average consumption (vs 2800J baseline)

---

## 🎯 **OVERVIEW**

This system integrates **Visual-Inertial SLAM**, **A* global planning**, **S-RRT local replanning**, and **PPO reinforcement learning** for energy-efficient indoor drone delivery in 5-floor buildings.

### Key Features:
- 🚁 **PPO-based Navigation**: Curriculum learning with 5M timesteps
- 🏢 **Multi-Floor Environment**: 5-floor building simulation  
- ⚡ **Energy Optimization**: 25%+ energy savings
- 🎯 **High Success Rate**: 96%+ navigation success
- 📱 **AirSim Integration**: Realistic simulation support
- 🔬 **Research Ready**: Table 3 reproduction capabilities

---

## 🚀 **QUICK START**

### 1. Installation (5 minutes)
```
# Clone repository
git clone <repository-url> DroneDelivery-RL
cd DroneDelivery-RL

# Automated setup
python scripts/setup/build_environment.py
```

### 2. Training (8-12 hours)
```
# Full PPO training with curriculum learning
python scripts/training/train_ppo.py
```

### 3. Evaluation (30 minutes)
```
# Evaluate trained model
python scripts/evaluation/evaluate_model.py \
    --model models/checkpoints/ppo_final.pt

# Generate Table 3 results
python scripts/evaluation/benchmark_baselines.py
```

### 4. Results & Analysis (15 minutes)
```
# Energy analysis
python scripts/utilities/analyze_energy.py \
    --evaluation-results results/model_evaluation.json

# Generate visualizations
python scripts/utilities/visualize_results.py \
    --evaluation-results results/model_evaluation.json
```

---

## 📊 **RESULTS**

### Table 3: Performance Comparison

| Method | Success Rate | Energy (J) | Time (s) | Collisions | ATE (cm) |
|--------|-------------|------------|----------|
| A* Only | 75.0% | 2800±450 | 95.0 | 8.0% | 4.5 |
| RRT+PID | 88.0% | 2400±380 | 78.0 | 4.0% | 3.8 |
| Random | 12.0% | 350±800 | 120.0 | 35.0% | 8.0 |
| **PPO (Ours)** | **96.2%** | **610±30** | **31.5** | **0.7%** | **0.8** |

### Target Achievements:
- ✅ **Success Rate**: 96.2% (Target: ≥96%)
- ✅ **Energy Savings**: 78% vs A* Only (Target: ≥25%)  
- ✅ **Safety**: 0.7% collision rate (Target: ≤2%)
- ✅ **Precision**: 0.8cm ATE (Target: ≤5cm)

---

## 📁 **PROJECT STRUCTURE**

```
DroneDelivery-RL/
├── 🔧 src/                     # Core system implementation
│   ├── bridges/               # Hardware/AirSim integration
│   ├── environment/           # Drone simulation environment
│   ├── localization/          # VI-SLAM system
│   ├── planning/             # A* and S-RRT planners
│   ├── rl/                   # PPO reinforcement learning
│   └── utils/                # System utilities
├── 📜 scripts/                # Execution scripts
│   ├── evaluation/           # Model evaluation & benchmarking
│   ├── setup/               # Environment setup
│   ├── training/            # PPO training pipeline  
│   └── utilities/           # Analysis & visualization
├── ⚙️ config/                # Configuration files
├── 📊 data/                  # Datasets & trajectories
├── 🤖 models/               # Trained model checkpoints
├── 📈 results/              # Evaluation results & plots
├── 🐳 docker/               # Docker containerization
├── 📖 docs/                 # Documentation
└── 🤖 ros_ws/              # ROS workspace (real deployment)
```

---

## 🛠️ **DETAILED DOCUMENTATION**

### Setup & Installation:
- 📖 [Setup Guide](scripts/setup/HUONG_DAN_CAI_DAT.md) - Complete installation instructions
- 🔧 [System Requirements](docs/SYSTEM_REQUIREMENTS.md) - Hardware & software requirements

### Training & Evaluation:
- 🚁 [Training Guide](scripts/training/HUONG_DAN_TRAINING.md) - PPO training workflow
- 📊 [Evaluation Guide](scripts/evaluation/HUONG_DAN_EVALUATION.md) - Model evaluation process

### Analysis & Utilities:
- ⚡ [Utilities Guide](scripts/utilities/HUONG_DAN_UTILITIES.md) - Analysis tools usage
- 📈 [Results Analysis](docs/RESULTS_ANALYSIS.md) - Understanding evaluation metrics

### Integration:
- 🔌 [AirSim Integration](docs/AIRSIM_INTEGRATION.md) - Connect with AirSim simulator
- 🤖 [ROS Integration](docs/ROS_INTEGRATION.md) - Real hardware deployment
- 🐳 [Docker Deployment](docker/README.md) - Containerized deployment

---

## 🎓 **RESEARCH REPRODUCTION**

### Complete Research Pipeline:
```
# 1. Setup environment
python scripts/setup/build_environment.py

# 2. Hyperparameter optimization (optional, 12-24h)
python scripts/training/hyperparameter_search.py --trials 50

# 3. Full curriculum training (8-12h)
python scripts/training/train_full_curriculum.py

# 4. Comprehensive evaluation (1h)
python scripts/evaluation/evaluate_model.py --model models/checkpoints/ppo_final.pt
python scripts/evaluation/benchmark_baselines.py

# 5. Generate research figures (30min)
python scripts/utilities/visualize_results.py \
    --evaluation-results results/model_evaluation.json \
    --baseline-results results/baseline_benchmark.json

# 6. Energy analysis (15min)
python scripts/utilities/analyze_energy.py \
    --evaluation-results results/model_evaluation.json
```

### Expected Research Outputs:
- 📊 **Table 3 comparison**: Performance vs baselines
- 📈 **Training curves**: Learning progression plots
- ⚡ **Energy analysis**: Consumption patterns & savings
- 🎯 **Target validation**: Research objectives achievement

---

## 🔬 **KEY COMPONENTS**

### 1. Reinforcement Learning (src/rl/)
- **PO Agent**: Policy gradient optimization
- **Curriculum Learning**: 3-phase progressive training
- **Energy Reward**: Custom reward function for efficiency

### 2. Navigation System (src/planning/)
- **A* Global Planner**: Optimal path finding
- **S-RRT Local Planner**: Dynamic obstacle avoidance
- **Multi-floor Planning**: Staircase & elevator navigation

### 3. Localization (src/localization/)
- **VI-SLAM**: Visual-inertial pose estimation
- **Stereo Vision**: Depth perception for obstacles
- **IMU Integration**: Robust pose tracking

### 4. Environment (src/environment/)
- **5-Floor Building**: Realistic indoor simulation
- **Dynamic Obstacles**: Moving obstacles & humans
- **Energy Modeling**: Realistic power consumption

---

## 📈 **PERFORMANCE METRICS**

### Navigation Performance:
- **Success Rate**: 96.2% (vs 75% baseline)
- **Energy Consumption**: 610J (vs 2800J baseline)
- **Flight Time**: 31.5s average
- **Collision Rate**: 0.7% (safety critical)

### Localization Accuracy:
- **ATE Error**: 0.8cm (centimeter-scale precision)  
- **RPE Error**: 0.12% (drift minimal)
- **SLAM Consistency**: 99.8% successful tracking

### Energy Efficiency:
- **Energy Savings**: 78% vs A* Only
- **Battery Life**: 59 missions per charge
- **Power Distribution**: 70% thrust, 20% avionics, 10% other

---

## 🛠️ **DEVELOPMENT**

### Requirements:
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- AirSim (optional)
- ROS Noetic (for real deployment)

### Development Setup:
```
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Code formatting
black src/ scripts/
flake8 src/ scripts/
```

### Contributing:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 **LICENSE**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **ACKNOWLEDGMENTS**

- **FPT University** - Research support
- **Microsoft AirSim** - Simulation platform
- **PyTorch Community** - Deep learning framework
- **Research Papers** - VI-SLAM, PPO, and planning algorithms

---

## 📞 **CONTACT**

- **Authors**: Huynh Nhut Huy, Nguyen Ly Minh Ky, Luong Danh Doanh, Nguyen Huy Hoang
- **Institution**: FPT University Ho Chi Minh City
- **Email**: [contact information]
- **Project**: Indoor Multi-Floor UAV Delivery Research

---

## 🏆 **CITATION**

If you use this work in your research, please cite:

```
@article{huy2025dronedelivery,
  title={Indoor Multi-Floor UAV Delivery: Energy-Aware Navigation through A*, S-RRT and Reinforcement Learning},
  author={Huy, Huynh Nhut and Ky, Nguyen Ly Minh and Doanh, Luong Danh and Hoang, Nguyen Huy},
  journal={FPT University Research},
  year={2025},
  publisher={FPT University Ho Chi Minh City}
}
```

---

**🚁 Ready for indoor drone delivery research and deployment! ✨**
