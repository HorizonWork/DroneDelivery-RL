This project implements an Indoor Multi-Floor UAV Delivery System using energy-aware navigation through A, S-RRT, and Reinforcement Learning (PPO). 

- 5-floor building environment with 20m40m3m per floor
- Energy-aware PPO agent with 35-dimensional observation space
- Multi-layer navigation: Global A + Local S-RRT + PPO control
- Curriculum learning: 125 floor progression (5M timesteps)
- Target performance: 96 success, 61030J energy, 0.7 collision rate
- AirSim integration with ORB-SLAM3 visual-inertial localization

-  Table 1: 35-dimensional observation space implementation
-  Table 2: Exact PPO hyperparameters matching
-  Table 3: Target performance validation framework
-  Equation (2): Energy-aware reward function implementation
-  Section 5: Complete evaluation against A, RRT, Random baselines
-  Landing targets: Landing_101-506 systematic naming
-  Drone spawn: {60, -30, 3} exact location

DroneDelivery-RL/
 README.md
 INSTALLATION.md
 requirements.txt
 environment.yml
 setup.py
 .gitignore
 .dockerignore

 src/
    __init__.py

    environment/
       __init__.py
       airsim_env.py
       drone_controller.py
       world_builder.py
       target_manager.py
       observation_space.py
       action_space.py
       reward_function.py
       curriculum_manager.py
       sensor_interface.py

    localization/
       __init__.py
       orb_slam3_wrapper.py
       vi_slam_interface.py
       pose_estimator.py
       ate_calculator.py
       coordinate_transforms.py

    planning/
       __init__.py
       global_planner/
          __init__.py
          astar_planner.py
          occupancy_grid.py
          path_optimizer.py
          heuristics.py
       local_planner/
          __init__.py
          srrt_planner.py
          dynamic_obstacles.py
          safety_checker.py
          cost_functions.py
       integration/
           __init__.py
           planner_manager.py
           path_smoother.py
           execution_monitor.py

    rl/
       __init__.py
       agents/
          __init__.py
          ppo_agent.py
          actor_critic.py
          policy_networks.py
          value_networks.py
          gae_calculator.py
       training/
          __init__.py
          trainer.py
          curriculum_trainer.py
          phase_1_trainer.py
          phase_2_trainer.py
          phase_3_trainer.py
          hyperparameter_scheduler.py
       evaluation/
          __init__.py
          evaluator.py
          baseline_comparator.py
          metrics_collector.py
          energy_analyzer.py
          trajectory_analyzer.py
       utils/
           __init__.py
           replay_buffer.py
           normalization.py
           checkpoint_manager.py
           tensorboard_logger.py

    baselines/
       __init__.py
       astar_baseline/
          __init__.py
          astar_controller.py
          pid_controller.py
          evaluator.py
       rrt_baseline/
          __init__.py
          rrt_star.py
          pid_controller.py
          evaluator.py
       random_baseline/
           __init__.py
           random_agent.py
           evaluator.py

    bridges/
       __init__.py
       airsim_bridge.py
       ros_bridge.py
       slam_bridge.py
       sensor_bridge.py

    utils/
        __init__.py
        config_loader.py
        logger.py
        visualization.py
        data_recorder.py
        math_utils.py
        coordinate_utils.py
        imu_preintegration.py
        file_utils.py

 config/
    airsim/
       settings.json
       environment.json
       sensors.json
       physics.json
    training/
       ppo_hyperparameters.yaml
       curriculum_config.yaml
       reward_weights.yaml
       environment_config.yaml
    evaluation/
       target_metrics.yaml
       baseline_config.yaml
       test_scenarios.yaml
    slam/
        orb_slam3_config.yaml
        camera_calibration.yaml
        imu_calibration.yaml

 scripts/
    setup/
       install_dependencies.sh
       setup_airsim.sh
       setup_conda_env.sh
       build_environment.py
       verify_installation.py
    training/
       train_full_curriculum.py
       train_phase.py
       resume_training.py
       hyperparameter_search.py
       monitor_training.py
    evaluation/
       evaluate_model.py
       benchmark_baselines.py
       run_test_scenarios.py
       generate_report.py
       validate_performance.py
    utilities/
        collect_data.py
        visualize_results.py
        export_trajectories.py
        analyze_energy.py

 data/
    training/
       logs/
       checkpoints/
          phase_1/
          phase_2/
          phase_3/
       metrics/
    evaluation/
       results/
       trajectories/
       energy_profiles/
       comparison_tables/
    slam/
        maps/
        trajectories/
        calibration/

 models/
    final/
       ppo_curriculum_5M.pt
       ppo_phase_1.pt
       ppo_phase_2.pt
       ppo_phase_3.pt
    baselines/
       astar_baseline.pt
       rrt_baseline.pt
       random_baseline.pt
    experiments/
        ablation_studies/
        hyperparameter_sweeps/

 results/
    figures/
       training_curves/
       performance_plots/
       energy_analysis/
       trajectory_plots/
    tables/
       baseline_comparison.csv
       energy_analysis.csv
       success_rates.csv
    reports/
       evaluation_report.pdf
       baseline_comparison.pdf
       energy_analysis.pdf
    videos/
        successful_flights/
        failure_analysis/
        baseline_comparison/

 tests/
    __init__.py
    unit/
       test_environment.py
       test_agents.py
       test_planners.py
       test_slam.py
       test_utils.py
    integration/
       test_airsim_integration.py
       test_full_pipeline.py
       test_training.py
       test_evaluation.py
    performance/
       test_training_speed.py
       test_inference_speed.py
       test_memory_usage.py
    fixtures/
        configs/
        data/
        models/

 docs/
    README_detailed.md
    INSTALLATION.md
    AIRSIM_SETUP.md
    TRAINING_GUIDE.md
    EVALUATION_GUIDE.md
    API_REFERENCE.md
    TROUBLESHOOTING.md
    REPORT_COMPLIANCE.md

 docker/
    Dockerfile.base
    Dockerfile.training
    Dockerfile.evaluation
    docker-compose.yml
    scripts/
        build.sh
        run_training.sh
        run_evaluation.sh

 ros_ws/
     src/
        orb_slam3_ros/
        airsim_ros/
        drone_interfaces/
     build/
     install/
     log/

-  Pose (7): 3D position + quaternion  src/environment/observation_space.py
-  Velocity (4): Body-frame velocities + yaw rate  src/environment/observation_space.py
-  Goal vector (3): 3D vector to target  src/environment/target_manager.py
-  Battery (1): Remaining battery fraction  src/environment/sensor_interface.py
-  Occupancy (24): 24-sector histogram  src/environment/sensor_interface.py
-  Localization error (1): ATE estimate  src/localization/ate_calculator.py

-  Learning Rate: 3e-4  config/training/ppo_hyperparameters.yaml
-  Rollout Length: 2048  config/training/ppo_hyperparameters.yaml
-  Batch Size: 64  config/training/ppo_hyperparameters.yaml
-  Epochs per Update: 10  config/training/ppo_hyperparameters.yaml
-  Clip Range: 0.2  config/training/ppo_hyperparameters.yaml
-  Discount Factor: 0.99  config/training/ppo_hyperparameters.yaml
-  GAE Parameter: 0.95  src/rl/agents/gae_calculator.py
-  Entropy Coefficient: 0.01  config/training/ppo_hyperparameters.yaml
-  Hidden Layers: [256,128,64]  src/rl/agents/actor_critic.py
-  Activation: tanh  src/rl/agents/actor_critic.py
-  Total Timesteps: 5M  config/training/curriculum_config.yaml

-  Success Rate: 96  src/rl/evaluation/metrics_collector.py
-  Energy: 61030 J  src/rl/evaluation/energy_analyzer.py
-  Time: 317 s  src/rl/evaluation/metrics_collector.py
-  Collision Rate: 0.7  src/rl/evaluation/metrics_collector.py

R(st, at) = 5001{goal} - 5dt - 0.1Δt - 0.01Σui² - 10jt - 1000ct

-  Implementation: src/environment/reward_function.py
-  Coefficients: config/training/reward_weights.yaml

-  5-floor building: 20m40m3m per floor  src/environment/world_builder.py
-  Cell size: 0.5m  src/planning/global_planner/occupancy_grid.py
-  Total cells: 4000  src/planning/global_planner/occupancy_grid.py
-  Drone spawn: {60,-30,3}  config/airsim/settings.json
-  Targets: Landing_101-506  src/environment/target_manager.py

-  Phase 1: 1 floor, 1M timesteps  src/rl/training/phase_1_trainer.py
-  Phase 2: 2 floors, 2M timesteps  src/rl/training/phase_2_trainer.py
-  Phase 3: 5 floors, 2M timesteps  src/rl/training/phase_3_trainer.py

-  A Global: 26-neighborhood, floor penalties  src/planning/global_planner/astar_planner.py
-  S-RRT Local: Cost C = ℓ + λc(1/d_min) + λκκ²  src/planning/local_planner/srrt_planner.py

-  A + PID:  src/baselines/astar_baseline/
-  RRT + PID:  src/baselines/rrt_baseline/
-  Random:  src/baselines/random_baseline/

bash
git clone https://github.com/HorizonWork/DroneDelivery-RL
cd DroneDelivery-RL

bash scripts/setup/install_dependencies.sh
bash scripts/setup/setup_conda_env.sh
conda activate drone-delivery-rl

bash scripts/setup/setup_airsim.sh

python scripts/setup/verify_installation.py

bash
python scripts/training/train_full_curriculum.py

python scripts/training/monitor_training.py

bash
python scripts/evaluation/evaluate_model.py --model models/final/ppo_curriculum_5M.pt

python scripts/evaluation/benchmark_baselines.py

python scripts/evaluation/validate_performance.py

Every component directly implements specifications from the final report. No deviation from Tables 1-3 or Equation (2).

Clean separation between environment, planning, RL, and evaluation systems for easy testing and modification.

Proven AirSim setup from previous work, enhanced for 5-floor environment with exact spawn location.

Fixed seeds, detailed configuration, and checkpoint management ensure reproducible training.

