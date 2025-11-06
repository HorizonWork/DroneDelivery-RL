# DroneDelivery-RL: Indoor Multi-Floor UAV Delivery System

## Project Overview

This project implements an **Indoor Multi-Floor UAV Delivery System** using energy-aware navigation through A*, S-RRT*, and Reinforcement Learning (PPO). The system matches 100% with the final report specifications while integrating proven AirSim simulation from previous work.

### Key Features
- **5-floor building environment** with 20m×40m×3m per floor
- **Energy-aware PPO agent** with 35-dimensional observation space
- **Multi-layer navigation**: Global A* + Local S-RRT* + PPO control
- **Curriculum learning**: 1→2→5 floor progression (5M timesteps)
- **Target performance**: 96% success, 610±30J energy, 0.7% collision rate
- **AirSim integration** with ORB-SLAM3 visual-inertial localization

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone <repository-url>
cd DroneDelivery-RL

# Setup environment
bash scripts/setup/install_dependencies.sh
bash scripts/setup/setup_conda_env.sh
conda activate drone-delivery-rl

# Setup AirSim
bash scripts/setup/setup_airsim.sh

# Verify installation
python scripts/setup/verify_installation.py
```

### 2. Training
```bash
# Full curriculum training (5M timesteps)
python scripts/training/train_full_curriculum.py

# Monitor progress
python scripts/training/monitor_training.py
```

### 3. Evaluation
```bash
# Evaluate trained model
python scripts/evaluation/evaluate_model.py --model models/final/ppo_curriculum_5M.pt

# Compare against baselines
python scripts/evaluation/benchmark_baselines.py

# Validate Table 3 performance
python scripts/evaluation/validate_performance.py
```

For more details, see the INSTALLATION.md and documentation in the docs/ directory.
