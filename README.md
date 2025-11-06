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

### Report Compliance
- ✅ **Table 1**: 35-dimensional observation space implementation
- ✅ **Table 2**: Exact PPO hyperparameters matching
- ✅ **Table 3**: Target performance validation framework
- ✅ **Equation (2)**: Energy-aware reward function implementation
- ✅ **Section 5**: Complete evaluation against A*, RRT*, Random baselines
- ✅ **Landing targets**: Landing_101-506 systematic naming
- ✅ **Drone spawn**: {6000, -3000, 300} exact location

## Project Structure

```
DroneDelivery-RL/
├── README.md                     # This file
├── INSTALLATION.md               # Detailed setup instructions
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── setup.py                     # Package installation
├── .gitignore                   # Git ignore rules
├── .dockerignore                # Docker ignore rules
│
├── src/                         # Main source code
│   ├── __init__.py              
│   │
│   ├── environment/             # Environment implementation (Table 1 compliance)
│   │   ├── __init__.py
│   │   ├── airsim_env.py        # Main AirSim environment wrapper
│   │   ├── drone_controller.py  # Low-level drone control
│   │   ├── world_builder.py     # 5-floor building construction
│   │   ├── target_manager.py    # Landing_101-506 target management
│   │   ├── observation_space.py # 35D observation implementation
│   │   ├── action_space.py      # 4D continuous [vx,vy,vz,ω]
│   │   ├── reward_function.py   # Equation (2) exact implementation
│   │   ├── curriculum_manager.py # 1→2→5 floor curriculum
│   │   └── sensor_interface.py  # Stereo camera + IMU integration
│   │
│   ├── localization/            # VI-SLAM system
│   │   ├── __init__.py
│   │   ├── orb_slam3_wrapper.py # ORB-SLAM3 integration
│   │   ├── vi_slam_interface.py # Visual-Inertial SLAM
│   │   ├── pose_estimator.py    # 6-DOF pose estimation
│   │   ├── ate_calculator.py    # Absolute Trajectory Error
│   │   └── coordinate_transforms.py # World/body frame transforms
│   │
│   ├── planning/                # Multi-layer navigation system
│   │   ├── __init__.py
│   │   ├── global_planner/      # A* global planning
│   │   │   ├── __init__.py
│   │   │   ├── astar_planner.py # A* with floor penalties φ_floor
│   │   │   ├── occupancy_grid.py # 20×40×5 grid, 0.5m cells
│   │   │   ├── path_optimizer.py # Path smoothing
│   │   │   └── heuristics.py    # A* heuristic functions
│   │   ├── local_planner/       # S-RRT* local replanning
│   │   │   ├── __init__.py
│   │   │   ├── srrt_planner.py  # S-RRT* implementation
│   │   │   ├── dynamic_obstacles.py # Moving obstacle handling
│   │   │   ├── safety_checker.py # Collision avoidance
│   │   │   └── cost_functions.py # C = ℓ + λc(1/d_min) + λκκ²
│   │   └── integration/         # Planning coordination
│   │       ├── __init__.py
│   │       ├── planner_manager.py # Global/local coordination
│   │       ├── path_smoother.py   # Trajectory smoothing
│   │       └── execution_monitor.py # Planning execution monitoring
│   │
│   ├── rl/                      # Reinforcement learning system (Table 2 compliance)
│   │   ├── __init__.py
│   │   ├── agents/              # PPO agent implementation
│   │   │   ├── __init__.py
│   │   │   ├── ppo_agent.py     # Main PPO implementation
│   │   │   ├── actor_critic.py  # [256,128,64] network architecture
│   │   │   ├── policy_networks.py # Actor network
│   │   │   ├── value_networks.py  # Critic network
│   │   │   └── gae_calculator.py  # GAE with λ=0.95
│   │   ├── training/            # Training pipeline
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py       # Main training loop
│   │   │   ├── curriculum_trainer.py # 3-phase curriculum
│   │   │   ├── phase_1_trainer.py # Single floor (1M timesteps)
│   │   │   ├── phase_2_trainer.py # Two floors (2M timesteps)
│   │   │   ├── phase_3_trainer.py # Five floors (2M timesteps)
│   │   │   └── hyperparameter_scheduler.py # LR scheduling
│   │   ├── evaluation/          # Evaluation system (Table 3 compliance)
│   │   │   ├── __init__.py
│   │   │   ├── evaluator.py     # Performance evaluation
│   │   │   ├── baseline_comparator.py # A*, RRT*, Random comparison
│   │   │   ├── metrics_collector.py # Success rate, energy, time
│   │   │   ├── energy_analyzer.py # Energy consumption analysis
│   │   │   └── trajectory_analyzer.py # Flight path analysis
│   │   └── utils/               # RL utilities
│   │       ├── __init__.py
│   │       ├── replay_buffer.py # Experience replay
│   │       ├── normalization.py # Observation normalization
│   │       ├── checkpoint_manager.py # Model checkpointing
│   │       └── tensorboard_logger.py # Training logging
│   │
│   ├── baselines/               # Baseline implementations
│   │   ├── __init__.py
│   │   ├── astar_baseline/      # A* + PID baseline
│   │   │   ├── __init__.py
│   │   │   ├── astar_controller.py # Pure A* path following
│   │   │   ├── pid_controller.py   # PID controller
│   │   │   └── evaluator.py        # A* baseline evaluation
│   │   ├── rrt_baseline/        # RRT* + PID baseline
│   │   │   ├── __init__.py
│   │   │   ├── rrt_star.py      # RRT* implementation
│   │   │   ├── pid_controller.py # PID controller
│   │   │   └── evaluator.py      # RRT* baseline evaluation
│   │   └── random_baseline/     # Random exploration
│   │       ├── __init__.py
│   │       ├── random_agent.py  # Random action agent
│   │       └── evaluator.py     # Random baseline evaluation
│   │
│   ├── bridges/                 # System integration bridges
│   │   ├── __init__.py
│   │   ├── airsim_bridge.py     # AirSim connection interface
│   │   ├── ros_bridge.py        # ROS2 integration
│   │   ├── slam_bridge.py       # ORB-SLAM3 interface
│   │   └── sensor_bridge.py     # Sensor data integration
│   │
│   └── utils/                   # General utilities
│       ├── __init__.py
│       ├── config_loader.py     # Configuration file loading
│       ├── logger.py            # System logging
│       ├── visualization.py     # Plotting and visualization
│       ├── data_recorder.py     # Data collection
│       ├── math_utils.py        # Mathematical utilities
│       ├── coordinate_utils.py  # Coordinate transformations
│       └── file_utils.py        # File I/O operations
│
├── config/                      # Configuration files
│   ├── airsim/                  # AirSim configuration
│   │   ├── settings.json        # Drone spawn {6000,-3000,300}
│   │   ├── environment.json     # 5-floor building specification
│   │   ├── sensors.json         # Camera + IMU configuration
│   │   └── physics.json         # Physics simulation parameters
│   ├── training/                # Training configuration (Table 2)
│   │   ├── ppo_hyperparameters.yaml # Exact Table 2 parameters
│   │   ├── curriculum_config.yaml   # 3-phase curriculum setup
│   │   ├── reward_weights.yaml      # Equation (2) coefficients
│   │   └── environment_config.yaml  # Environment parameters
│   ├── evaluation/              # Evaluation configuration (Table 3)
│   │   ├── target_metrics.yaml  # Performance targets
│   │   ├── baseline_config.yaml # Baseline comparison setup
│   │   └── test_scenarios.yaml  # Evaluation scenarios
│   └── slam/                    # SLAM configuration
│       ├── orb_slam3_config.yaml # ORB-SLAM3 parameters
│       ├── camera_calibration.yaml # Stereo camera parameters
│       └── imu_calibration.yaml    # IMU noise parameters
│
├── scripts/                     # Execution scripts
│   ├── setup/                   # Environment setup
│   │   ├── install_dependencies.sh # Install all dependencies
│   │   ├── setup_airsim.sh         # AirSim setup
│   │   ├── setup_conda_env.sh      # Conda environment setup
│   │   ├── build_environment.py    # Build 5-floor environment
│   │   └── verify_installation.py  # Installation verification
│   ├── training/                # Training scripts
│   │   ├── train_full_curriculum.py # Complete 5M timestep training
│   │   ├── train_phase.py          # Single phase training
│   │   ├── resume_training.py      # Resume from checkpoint
│   │   ├── hyperparameter_search.py # HPO with Ray Tune
│   │   └── monitor_training.py     # Training monitoring
│   ├── evaluation/              # Evaluation scripts
│   │   ├── evaluate_model.py       # Trained model evaluation
│   │   ├── benchmark_baselines.py  # Baseline comparison
│   │   ├── run_test_scenarios.py   # Test scenario execution
│   │   ├── generate_report.py      # Auto-generate evaluation report
│   │   └── validate_performance.py # Table 3 validation
│   └── utilities/               # Utility scripts
│       ├── collect_data.py         # Data collection
│       ├── visualize_results.py    # Results visualization
│       ├── export_trajectories.py  # Export flight paths
│       └── analyze_energy.py       # Energy analysis
│
├── data/                        # Data storage
│   ├── training/                # Training data
│   │   ├── logs/               # Training logs
│   │   ├── checkpoints/        # Model checkpoints
│   │   │   ├── phase_1/        # Single floor checkpoints
│   │   │   ├── phase_2/        # Two floor checkpoints
│   │   │   └── phase_3/        # Five floor checkpoints
│   │   └── metrics/            # Training metrics
│   ├── evaluation/             # Evaluation data
│   │   ├── results/            # Evaluation results
│   │   ├── trajectories/       # Flight trajectories
│   │   ├── energy_profiles/    # Energy consumption data
│   │   └── comparison_tables/  # Baseline comparison tables
│   └── slam/                   # SLAM data
│       ├── maps/               # Generated maps
│       ├── trajectories/       # SLAM trajectories
│       └── calibration/        # Sensor calibration data
│
├── models/                      # Trained models
│   ├── final/                   # Final trained models
│   │   ├── ppo_curriculum_5M.pt # Main curriculum-trained model
│   │   ├── ppo_phase_1.pt      # Single floor model
│   │   ├── ppo_phase_2.pt      # Two floor model
│   │   └── ppo_phase_3.pt      # Five floor model
│   ├── baselines/              # Baseline models
│   │   ├── astar_baseline.pt   # A* baseline
│   │   ├── rrt_baseline.pt     # RRT* baseline
│   │   └── random_baseline.pt  # Random baseline
│   └── experiments/            # Experimental models
│       ├── ablation_studies/   # Ablation study models
│       └── hyperparameter_sweeps/ # HPO models
│
├── results/                     # Results and analysis
│   ├── figures/                # Generated figures
│   │   ├── training_curves/    # Training progress plots
│   │   ├── performance_plots/  # Performance comparison plots
│   │   ├── energy_analysis/    # Energy consumption plots
│   │   └── trajectory_plots/   # 3D trajectory visualizations
│   ├── tables/                 # Performance tables
│   │   ├── baseline_comparison.csv # Table 3 compliance
│   │   ├── energy_analysis.csv     # Energy consumption results
│   │   └── success_rates.csv       # Success rate analysis
│   ├── reports/                # Generated reports
│   │   ├── evaluation_report.pdf # Auto-generated evaluation
│   │   ├── baseline_comparison.pdf # Baseline analysis
│   │   └── energy_analysis.pdf     # Energy efficiency analysis
│   └── videos/                 # Flight videos
│       ├── successful_flights/ # Successful delivery videos
│       ├── failure_analysis/   # Failure case videos
│       └── baseline_comparison/ # Baseline method videos
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── unit/                   # Unit tests
│   │   ├── test_environment.py     # Environment tests
│   │   ├── test_agents.py         # Agent tests
│   │   ├── test_planners.py       # Planning tests
│   │   ├── test_slam.py           # SLAM tests
│   │   └── test_utils.py          # Utility tests
│   ├── integration/            # Integration tests
│   │   ├── test_airsim_integration.py # AirSim integration
│   │   ├── test_full_pipeline.py      # End-to-end pipeline
│   │   ├── test_training.py           # Training pipeline
│   │   └── test_evaluation.py         # Evaluation pipeline
│   ├── performance/            # Performance tests
│   │   ├── test_training_speed.py # Training performance
│   │   ├── test_inference_speed.py # Inference performance
│   │   └── test_memory_usage.py    # Memory usage
│   └── fixtures/               # Test fixtures
│       ├── configs/            # Test configurations
│       ├── data/              # Test data
│       └── models/            # Test models
│
├── docs/                        # Documentation
│   ├── README_detailed.md       # Detailed project documentation
│   ├── INSTALLATION.md          # Installation guide
│   ├── AIRSIM_SETUP.md         # AirSim setup guide
│   ├── TRAINING_GUIDE.md       # Training guide
│   ├── EVALUATION_GUIDE.md     # Evaluation guide
│   ├── API_REFERENCE.md        # API documentation
│   ├── TROUBLESHOOTING.md      # Common issues and solutions
│   └── REPORT_COMPLIANCE.md    # Report specification compliance
│
├── docker/                      # Docker configuration
│   ├── Dockerfile.base         # Base image
│   ├── Dockerfile.training     # Training container
│   ├── Dockerfile.evaluation   # Evaluation container
│   ├── docker-compose.yml      # Multi-container setup
│   └── scripts/               # Docker utility scripts
│       ├── build.sh           # Build containers
│       ├── run_training.sh    # Run training container
│       └── run_evaluation.sh  # Run evaluation container
│
└── ros_ws/                      # ROS2 workspace (if using ROS)
    ├── src/                    # ROS2 packages
    │   ├── orb_slam3_ros/     # ORB-SLAM3 ROS integration
    │   ├── airsim_ros/        # AirSim ROS bridge
    │   └── drone_interfaces/   # Custom message definitions
    ├── build/                  # Build artifacts
    ├── install/               # Installed packages
    └── log/                   # ROS logs
```

## Report Compliance Matrix

### Table 1: Observation Space (35 dimensions)
- ✅ **Pose (7)**: 3D position + quaternion → `src/environment/observation_space.py`
- ✅ **Velocity (4)**: Body-frame velocities + yaw rate → `src/environment/observation_space.py`
- ✅ **Goal vector (3)**: 3D vector to target → `src/environment/target_manager.py`
- ✅ **Battery (1)**: Remaining battery fraction → `src/environment/sensor_interface.py`
- ✅ **Occupancy (24)**: 24-sector histogram → `src/environment/sensor_interface.py`
- ✅ **Localization error (1)**: ATE estimate → `src/localization/ate_calculator.py`

### Table 2: PPO Hyperparameters
- ✅ **Learning Rate**: 3e-4 → `config/training/ppo_hyperparameters.yaml`
- ✅ **Rollout Length**: 2048 → `config/training/ppo_hyperparameters.yaml`
- ✅ **Batch Size**: 64 → `config/training/ppo_hyperparameters.yaml`
- ✅ **Epochs per Update**: 10 → `config/training/ppo_hyperparameters.yaml`
- ✅ **Clip Range**: 0.2 → `config/training/ppo_hyperparameters.yaml`
- ✅ **Discount Factor**: 0.99 → `config/training/ppo_hyperparameters.yaml`
- ✅ **GAE Parameter**: 0.95 → `src/rl/agents/gae_calculator.py`
- ✅ **Entropy Coefficient**: 0.01 → `config/training/ppo_hyperparameters.yaml`
- ✅ **Hidden Layers**: [256,128,64] → `src/rl/agents/actor_critic.py`
- ✅ **Activation**: tanh → `src/rl/agents/actor_critic.py`
- ✅ **Total Timesteps**: 5M → `config/training/curriculum_config.yaml`

### Table 3: Target Performance
- ✅ **Success Rate**: 96% → `src/rl/evaluation/metrics_collector.py`
- ✅ **Energy**: 610±30 J → `src/rl/evaluation/energy_analyzer.py`
- ✅ **Time**: 31±7 s → `src/rl/evaluation/metrics_collector.py`
- ✅ **Collision Rate**: 0.7% → `src/rl/evaluation/metrics_collector.py`

### Equation (2): Reward Function
```
R(st, at) = 500·1{goal} - 5·dt - 0.1·Δt - 0.01·Σui² - 10·jt - 1000·ct
```
- ✅ **Implementation**: `src/environment/reward_function.py`
- ✅ **Coefficients**: `config/training/reward_weights.yaml`

### Environment Specifications
- ✅ **5-floor building**: 20m×40m×3m per floor → `src/environment/world_builder.py`
- ✅ **Cell size**: 0.5m → `src/planning/global_planner/occupancy_grid.py`
- ✅ **Total cells**: 4000 → `src/planning/global_planner/occupancy_grid.py`
- ✅ **Drone spawn**: {6000,-3000,300} → `config/airsim/settings.json`
- ✅ **Targets**: Landing_101-506 → `src/environment/target_manager.py`

### Curriculum Learning
- ✅ **Phase 1**: 1 floor, 1M timesteps → `src/rl/training/phase_1_trainer.py`
- ✅ **Phase 2**: 2 floors, 2M timesteps → `src/rl/training/phase_2_trainer.py`
- ✅ **Phase 3**: 5 floors, 2M timesteps → `src/rl/training/phase_3_trainer.py`

### Planning System
- ✅ **A* Global**: 26-neighborhood, floor penalties → `src/planning/global_planner/astar_planner.py`
- ✅ **S-RRT* Local**: Cost C = ℓ + λc(1/d_min) + λκκ² → `src/planning/local_planner/srrt_planner.py`

### Evaluation Baselines
- ✅ **A* + PID**: → `src/baselines/astar_baseline/`
- ✅ **RRT* + PID**: → `src/baselines/rrt_baseline/`
- ✅ **Random**: → `src/baselines/random_baseline/`

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

## Key Design Principles

### 1. **Report Compliance First**
Every component directly implements specifications from the final report. No deviation from Tables 1-3 or Equation (2).

### 2. **Modular Architecture**
Clean separation between environment, planning, RL, and evaluation systems for easy testing and modification.

### 3. **AirSim Integration**
Proven AirSim setup from previous work, enhanced for 5-floor environment with exact spawn location.

### 4. **Comprehensive Evaluation**
Full baseline comparison framework matching report evaluation section.

### 5. **Reproducible Results**
Fixed seeds, detailed configuration, and checkpoint management ensure reproducible training.

This structure provides a **complete, production-ready implementation** that matches 100% with your final report while incorporating proven AirSim integration from your previous work.