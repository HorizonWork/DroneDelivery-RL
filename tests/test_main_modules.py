import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class TestMainModules(unittest.TestCase):

    def test_environment_basic_functionality(self):

        from src.environment.action_space import ActionSpace
        from src.environment.observation_space import ObservationSpace
        from src.environment.reward_function import RewardFunction
        from src.environment.target_manager import TargetManager

        config = {"action_space": {"low": -1.0, "high": 1.0}}
        action_space = ActionSpace(config)
        self.assertIsNotNone(action_space)

        observation_space = ObservationSpace(config)
        self.assertIsNotNone(observation_space)

        reward_function = RewardFunction(config)
        self.assertIsNotNone(reward_function)

        target_manager = TargetManager(config)
        self.assertIsNotNone(target_manager)

    def test_rl_agent_basic_functionality(self):

        from src.rl.agents.ppo_agent import PPOAgent, PPOConfig
        from src.rl.agents.actor_critic import ActorCriticNetwork
        from src.rl.agents.gae_calculator import GAECalculator

        config = PPOConfig()
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.rollout_length, 2048)
        self.assertEqual(config.batch_size, 64)

        gae_calc = GAECalculator(lam=0.95, gamma=0.99)
        self.assertIsNotNone(gae_calc)

        network_config = {
            "observation_dim": 35,
            "action_dim": 4,
            "hidden_sizes": [256, 128, 64],
            "activation": "tanh",
        }
        actor_critic = ActorCriticNetwork(network_config)
        self.assertIsNotNone(actor_critic)

    def test_planning_basic_functionality(self):

        from src.planning.global_planner.astar_planner import AStarPlanner
        from src.planning.global_planner.heuristics import AStarHeuristics
        from src.planning.global_planner.occupancy_grid import OccupancyGrid3D

        grid_config = {
            "cell_size": 0.5,
            "building_dims": {"length": 20.0, "width": 40.0, "height": 15.0},
        }
        occupancy_grid = OccupancyGrid3D(grid_config)
        self.assertIsNotNone(occupancy_grid)

        heuristics = AStarHeuristics({})
        self.assertIsNotNone(heuristics)

        config = {
            "cell_size": 0.5,
            "building_dims": {"length": 20.0, "width": 40.0, "height": 15.0},
            "floor_transition_penalty": 2.0,
        }
        astar_planner = AStarPlanner(config)
        self.assertIsNotNone(astar_planner)

    def test_baselines_basic_functionality(self):

        from src.baselines.astar_baseline.astar_controller import AStarController

        config = {
            "floors": 5,
            "floor_length": 20.0,
            "floor_width": 40.0,
            "floor_height": 3.0,
            "cell_size": 0.5,
            "floor_penalty": 5.0,
        }
        astar_controller = AStarController(config)
        self.assertIsNotNone(astar_controller)

        start_pos = (0.0, 0.0, 1.5)
        goal_pos = (5.0, 5.0, 1.5)
        path = astar_controller.plan_path(start_pos, goal_pos)
        self.assertIsInstance(path, list)

    def test_utils_basic_functionality(self):

        from src.utils.coordinate_utils import world_to_grid, grid_to_world
        from src.utils.math_utils import normalize_angle

        world_pos = (10.0, 20.0, 3.0)
        grid_pos = world_to_grid(world_pos, cell_size=0.5, floor_height=3.0)
        self.assertIsInstance(grid_pos, tuple)
        self.assertEqual(len(grid_pos), 3)

        converted_back = grid_to_world(grid_pos, cell_size=0.5, floor_height=3.0)
        self.assertIsInstance(converted_back, tuple)
        self.assertEqual(len(converted_back), 3)

        normalized_angle = normalize_angle(
            3.14159  3
        )
        self.assertIsInstance(normalized_angle, float)
        self.assertGreaterEqual(normalized_angle, -3.14159)
        self.assertLessEqual(normalized_angle, 3.14159)

    def test_bridges_basic_structure(self):

        from src.bridges.airsim_bridge import AirSimBridge
        from src.bridges.slam_bridge import SLAMBridge
        from src.bridges.sensor_bridge import SensorBridge

        airsim_config = {"host": "localhost", "port": 41451}
        airsim_bridge = AirSimBridge(airsim_config)
        self.assertIsNotNone(airsim_bridge)

        slam_config = {"slam_method": "orb_slam3"}
        slam_bridge = SLAMBridge(slam_config)
        self.assertIsNotNone(slam_bridge)

        sensor_config = {"stereo_camera": True, "imu": True}
        sensor_bridge = SensorBridge(sensor_config)
        self.assertIsNotNone(sensor_bridge)

    def test_localization_basic_functionality(self):

        from src.localization.coordinate_transforms import transform_pose
        from src.localization.ate_calculator import ATCalculator

        ate_calc = ATCalculator()
        self.assertIsNotNone(ate_calc)

        pose1 = {"position": [0, 0, 0], "orientation": [0, 0, 1]}
        pose2 = {"position": [1, 1, 1], "orientation": [0, 0, 0, 1]}

        self.assertIsInstance(pose1, dict)
        self.assertIsInstance(pose2, dict)
        self.assertIn("position", pose1)
        self.assertIn("orientation", pose1)

    def test_observation_space_functionality(self):

        from src.environment.observation_space import ObservationSpace

        config = {}
        obs_space = ObservationSpace(config)
        self.assertIsNotNone(obs_space)

        self.assertTrue(hasattr(obs_space, "build_observation"))
        self.assertTrue(hasattr(obs_space, "get_observation_info"))

    def test_reward_function_structure(self):

        from src.environment.reward_function import RewardFunction

        config = {}
        reward_func = RewardFunction(config)
        self.assertIsNotNone(reward_func)

        self.assertTrue(hasattr(reward_func, "compute_reward"))
        self.assertTrue(hasattr(reward_func, "get_reward_info"))

if __name__ == "__main__":
    unittest.main()
