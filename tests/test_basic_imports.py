"""
Basic import tests for DroneDelivery-RL project.
These tests check that all modules can be imported without errors.
"""

import unittest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestBasicImports(unittest.TestCase):
    """Test that all main modules can be imported without errors."""

    def test_src_import(self):
        """Test importing main src package."""
        try:
            import src

            self.assertIsNotNone(src)
        except ImportError as e:
            self.fail(f"Failed to import src: {e}")

    def test_environment_imports(self):
        """Test importing environment modules."""
        modules_to_test = [
            "src.environment",
            "src.environment.airsim_env",
            "src.environment.action_space",
            "src.environment.observation_space",
            "src.environment.reward_function",
            "src.environment.target_manager",
            "src.environment.curriculum_manager",
            "src.environment.sensor_interface",
            "src.environment.drone_controller",
            "src.environment.world_builder",
        ]

        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")

    def test_rl_imports(self):
        """Test importing RL modules."""
        modules_to_test = [
            "src.rl",
            "src.rl.agents",
            "src.rl.agents.ppo_agent",
            "src.rl.agents.actor_critic",
            "src.rl.agents.policy_networks",
            "src.rl.agents.gae_calculator",
            "src.rl.evaluation",
            "src.rl.evaluation.evaluator",
            "src.rl.evaluation.metrics_collector",
            "src.rl.training",
            "src.rl.training.trainer",
            "src.rl.training.curriculum_trainer",
            "src.rl.utils",
            "src.rl.utils.checkpoint_manager",
            "src.rl.utils.replay_buffer",
        ]

        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")

    def test_planning_imports(self):
        """Test importing planning modules."""
        modules_to_test = [
            "src.planning",
            "src.planning.global_planner",
            "src.planning.global_planner.astar_planner",
            "src.planning.global_planner.heuristics",
            "src.planning.global_planner.occupancy_grid",
            "src.planning.global_planner.path_optimizer",
            "src.planning.local_planner",
            "src.planning.local_planner.srrt_planner",
            "src.planning.local_planner.safety_checker",
            "src.planning.integration",
            "src.planning.integration.planner_manager",
            "src.planning.integration.path_smoother",
            "src.planning.integration.execution_monitor",
        ]

        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")

    def test_baselines_imports(self):
        """Test importing baseline modules."""
        modules_to_test = [
            "src.baselines",
            "src.baselines.astar_baseline",
            "src.baselines.astar_baseline.astar_controller",
            "src.baselines.astar_baseline.pid_controller",
            "src.baselines.astar_baseline.evaluator",
            "src.baselines.rrt_baseline",
            "src.baselines.rrt_baseline.rrt_star",
            "src.baselines.rrt_baseline.pid_controller",
            "src.baselines.rrt_baseline.evaluator",
            "src.baselines.random_baseline",
            "src.baselines.random_baseline.random_agent",
            "src.baselines.random_baseline.evaluator",
        ]

        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")

    def test_bridges_imports(self):
        """Test importing bridge modules."""
        modules_to_test = [
            "src.bridges",
            "src.bridges.airsim_bridge",
            "src.bridges.ros_bridge",
            "src.bridges.slam_bridge",
            "src.bridges.sensor_bridge",
        ]

        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")

    def test_localization_imports(self):
        """Test importing localization modules."""
        modules_to_test = [
            "src.localization",
            "src.localization.orb_slam3_wrapper",
            "src.localization.pose_estimator",
            "src.localization.vi_slam_interface",
            "src.localization.ate_calculator",
            "src.localization.coordinate_transforms",
        ]

        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")

    def test_utils_imports(self):
        """Test importing utility modules."""
        modules_to_test = [
            "src.utils",
            "src.utils.config_loader",
            "src.utils.coordinate_utils",
            "src.utils.data_recorder",
            "src.utils.file_utils",
            "src.utils.imu_preintegration",
            "src.utils.logger",
            "src.utils.math_utils",
            "src.utils.visualization",
        ]

        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")


if __name__ == "__main__":
    unittest.main()
