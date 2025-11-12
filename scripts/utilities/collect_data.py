#!/usr/bin/env python3
"""
Data Collection Utility
Collects comprehensive dataset from drone delivery system.
Gathers trajectories, energy profiles, performance metrics for analysis.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.environment.airsim_env import AirSimEnvironment as DroneEnvironment
from src.rl.agents.ppo_agent import PPOAgent
from src.rl.initialization import initialize_rl_system
from src.localization.vi_slam_interface import VisualInertialSLAM

from src.utils import setup_logging, load_config


class DataCollector:
    """
    Comprehensive data collection system for research analysis.
    Collects trajectories, energy profiles, SLAM data, and performance metrics.
    """

    def __init__(self, config_path: str):
        self.config = load_config(config_path)

        # Setup logging
        self.logger_system = setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)

        # Initialize systems
        self.environment = DroneEnvironment(self.config.environment)

        # Initialize RL system if model provided
        self.rl_system = None
        self.agent = None

        # Data collection configuration
        self.collection_config = {
            "trajectory_sampling_rate": 20,  # Hz
            "energy_sampling_rate": 20,  # Hz
            "slam_data_rate": 10,  # Hz
            "sensor_data_rate": 100,  # Hz
            "collect_raw_sensor_data": True,
            "collect_intermediate_planning": True,
            "collect_control_commands": True,
        }

        # Data storage
        self.collected_data = {
            "trajectories": [],
            "energy_profiles": [],
            "slam_data": [],
            "sensor_data": [],
            "performance_metrics": [],
            "control_commands": [],
            "planning_data": [],
        }

        self.logger.info("Data Collector initialized")

    def load_trained_model(self, model_path: str):
        """Load trained model for data collection."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Initialize RL system
        self.rl_system = initialize_rl_system(self.config.rl)
        self.agent = self.rl_system["agent"]

        # Load model
        import torch

        checkpoint = torch.load(model_path, map_location=self.agent.device)

        if "agent_state_dict" in checkpoint:
            self.agent.policy.load_state_dict(checkpoint["agent_state_dict"])
        else:
            self.agent.policy.load_state_dict(checkpoint)

        self.agent.policy.eval()

        self.logger.info(f"Model loaded: {model_path}")

    def collect_comprehensive_dataset(
        self, num_episodes: int = 100, scenarios: List[str] = None
    ) -> Dict[str, Any]:
        """
        Collect comprehensive dataset.

        Args:
            num_episodes: Number of episodes to collect
            scenarios: Specific scenarios to collect data from

        Returns:
            Complete dataset
        """
        self.logger.info(
            f"Starting comprehensive data collection: {num_episodes} episodes"
        )
        collection_start = time.time()

        # Default scenarios
        if scenarios is None:
            scenarios = [
                "nominal",
                "high_obstacles",
                "multi_floor",
                "dynamic_environment",
            ]

        # Collect data for each scenario
        for scenario_name in scenarios:
            self.logger.info(f"Collecting data for scenario: {scenario_name}")

            # Setup environment for scenario
            scenario_environment = self._setup_scenario_environment(scenario_name)

            # Collect episodes for scenario
            scenario_episodes = num_episodes // len(scenarios)
            scenario_data = self._collect_scenario_data(
                scenario_environment, scenario_episodes, scenario_name
            )

            # Store scenario data
            self._store_scenario_data(scenario_data, scenario_name)

        collection_time = time.time() - collection_start

        # Finalize dataset
        complete_dataset = self._finalize_dataset()
        complete_dataset["collection_metadata"] = {
            "collection_time_hours": collection_time / 3600,
            "total_episodes": num_episodes,
            "scenarios_collected": scenarios,
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.logger.info(
            f"Data collection completed in {collection_time/60:.1f} minutes"
        )
        return complete_dataset

    def _setup_scenario_environment(self, scenario_name: str) -> DroneEnvironment:
        """Setup environment for specific scenario."""
        env_config = self.config.environment.copy()

        # Scenario-specific configurations
        scenario_configs = {
            "nominal": {
                "floors": 3,
                "obstacle_density": 0.15,
                "dynamic_obstacles": True,
                "dynamic_count": 3,
            },
            "high_obstacles": {
                "floors": 3,
                "obstacle_density": 0.30,
                "dynamic_obstacles": True,
                "dynamic_count": 6,
            },
            "multi_floor": {
                "floors": 5,
                "obstacle_density": 0.20,
                "dynamic_obstacles": True,
                "dynamic_count": 5,
                "complex_floor_layout": True,
            },
            "dynamic_environment": {
                "floors": 3,
                "obstacle_density": 0.15,
                "dynamic_obstacles": True,
                "dynamic_count": 8,
                "human_obstacles": True,
            },
        }

        if scenario_name in scenario_configs:
            scenario_config = scenario_configs[scenario_name]

            # Apply scenario settings
            if "floors" in scenario_config:
                env_config["building"]["floors"] = scenario_config["floors"]

            if "obstacle_density" in scenario_config:
                env_config["obstacles"]["density"] = scenario_config["obstacle_density"]

            if "dynamic_obstacles" in scenario_config:
                env_config["obstacles"]["dynamic_obstacles"] = scenario_config[
                    "dynamic_obstacles"
                ]
                env_config["obstacles"]["dynamic_count"] = scenario_config.get(
                    "dynamic_count", 3
                )

        return DroneEnvironment(env_config)

    def _collect_scenario_data(
        self, environment: DroneEnvironment, num_episodes: int, scenario_name: str
    ) -> Dict[str, Any]:
        """Collect data for specific scenario."""
        scenario_data = {"scenario_name": scenario_name, "episodes": []}

        for episode in range(num_episodes):
            if episode % 10 == 0:
                self.logger.info(
                    f"Scenario {scenario_name}: Episode {episode}/{num_episodes}"
                )

            episode_data = self._collect_episode_data(
                environment, scenario_name, episode
            )
            scenario_data["episodes"].append(episode_data)

        return scenario_data

    def _collect_episode_data(
        self, environment: DroneEnvironment, scenario_name: str, episode_id: int
    ) -> Dict[str, Any]:
        """Collect comprehensive data for single episode."""
        episode_start = time.time()

        # Reset environment
        observation = environment.reset()

        # Initialize data collection
        episode_data = {
            "episode_id": episode_id,
            "scenario": scenario_name,
            "start_time": episode_start,
            "trajectory": [],
            "energy_profile": [],
            "actions": [],
            "observations": [],
            "rewards": [],
            "slam_estimates": [],
            "control_commands": [],
            "sensor_readings": [],
        }

        # Episode execution with data collection
        total_energy = 0.0
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 1000:
            # Collect observation data
            episode_data["observations"].append(observation.tolist())

            # Get action from agent (if available)
            if self.agent:
                action, log_prob, value = self.agent.select_action(
                    observation, deterministic=True
                )

                # Store action and value data
                episode_data["actions"].append(action.tolist())
            else:
                # Random action for data collection without model
                action = np.random.uniform(-1, 1, size=4)
                episode_data["actions"].append(action.tolist())

            # Execute action
            next_observation, reward, done, info = environment.step(action)

            # Collect step data
            position = info.get("position", [0, 0, 0])
            energy_consumption = info.get("energy_consumption", 0.0)

            episode_data["trajectory"].append(position)
            episode_data["energy_profile"].append(energy_consumption)
            episode_data["rewards"].append(reward)

            # Collect SLAM data (if available)
            if "slam_estimate" in info:
                episode_data["slam_estimates"].append(info["slam_estimate"])

            # Collect sensor data
            sensor_data = {
                "timestamp": steps / 20.0,  # 20Hz
                "imu_data": info.get("imu_data", {}),
                "camera_data": info.get("camera_data", {}),
                "lidar_data": info.get("lidar_data", {}),
            }
            episode_data["sensor_readings"].append(sensor_data)

            # Collect control commands
            control_cmd = {
                "timestamp": steps / 20.0,
                "velocity_command": action[:3].tolist(),
                "yaw_rate_command": action[3] if len(action) > 3 else 0.0,
                "thrust_command": info.get("thrust_command", 0.0),
            }
            episode_data["control_commands"].append(control_cmd)

            # Update episode metrics
            total_energy += energy_consumption
            total_reward += reward
            steps += 1

            observation = next_observation

        # Episode completion data
        episode_data.update(
            {
                "end_time": time.time(),
                "episode_duration": time.time() - episode_start,
                "total_steps": steps,
                "total_energy": total_energy,
                "total_reward": total_reward,
                "success": info.get("success", False),
                "collision": info.get("collision", False),
                "final_position": position,
                "goal_position": environment.get_goal_position(),
                "path_length": self._calculate_path_length(episode_data["trajectory"]),
            }
        )

        return episode_data

    def _calculate_path_length(self, trajectory: List[List[float]]) -> float:
        """Calculate total path length from trajectory."""
        if len(trajectory) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(trajectory)):
            segment_length = np.linalg.norm(
                np.array(trajectory[i]) - np.array(trajectory[i - 1])
            )
            total_length += segment_length

        return total_length

    def _store_scenario_data(self, scenario_data: Dict[str, Any], scenario_name: str):
        """Store collected scenario data."""
        # Add to collected data
        self.collected_data["trajectories"].extend(
            [
                (ep["episode_id"], scenario_name, ep["trajectory"])
                for ep in scenario_data["episodes"]
            ]
        )

        self.collected_data["energy_profiles"].extend(
            [
                (
                    ep["episode_id"],
                    scenario_name,
                    ep["energy_profile"],
                    ep["total_energy"],
                )
                for ep in scenario_data["episodes"]
            ]
        )

        self.collected_data["performance_metrics"].extend(
            [
                (
                    ep["episode_id"],
                    scenario_name,
                    ep["success"],
                    ep["collision"],
                    ep["total_reward"],
                    ep["total_energy"],
                    ep["path_length"],
                )
                for ep in scenario_data["episodes"]
            ]
        )

    def _finalize_dataset(self) -> Dict[str, Any]:
        """Finalize collected dataset."""
        # Convert to structured format
        dataset = {
            "metadata": {
                "dataset_version": "1.0",
                "collection_date": time.strftime("%Y-%m-%d"),
                "total_episodes": len(self.collected_data["trajectories"]),
                "data_types": list(self.collected_data.keys()),
            },
            "trajectories": {
                "count": len(self.collected_data["trajectories"]),
                "data": self.collected_data["trajectories"],
            },
            "energy_profiles": {
                "count": len(self.collected_data["energy_profiles"]),
                "data": self.collected_data["energy_profiles"],
            },
            "performance_metrics": {
                "count": len(self.collected_data["performance_metrics"]),
                "data": self.collected_data["performance_metrics"],
            },
        }

        return dataset

    def save_dataset(self, dataset: Dict[str, Any], output_dir: str):
        """Save dataset to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save complete dataset
        dataset_file = output_path / "complete_dataset.json"
        with open(dataset_file, "w") as f:
            json.dump(dataset, f, indent=2, default=str)

        # Save individual components as CSV for analysis
        self._save_trajectories_csv(dataset, output_path)
        self._save_energy_profiles_csv(dataset, output_path)
        self._save_performance_metrics_csv(dataset, output_path)

        self.logger.info(f"Dataset saved to {output_path}")

    def _save_trajectories_csv(self, dataset: Dict[str, Any], output_path: Path):
        """Save trajectories as CSV."""
        trajectory_data = []

        for episode_id, scenario, trajectory in dataset["trajectories"]["data"]:
            for step, position in enumerate(trajectory):
                trajectory_data.append(
                    {
                        "episode_id": episode_id,
                        "scenario": scenario,
                        "step": step,
                        "x": position[0],
                        "y": position[1],
                        "z": position[2],
                        "timestamp": step / 20.0,  # 20Hz
                    }
                )

        df = pd.DataFrame(trajectory_data)
        df.to_csv(output_path / "trajectories.csv", index=False)

    def _save_energy_profiles_csv(self, dataset: Dict[str, Any], output_path: Path):
        """Save energy profiles as CSV."""
        energy_data = []

        for episode_id, scenario, energy_profile, total_energy in dataset[
            "energy_profiles"
        ]["data"]:
            for step, energy in enumerate(energy_profile):
                energy_data.append(
                    {
                        "episode_id": episode_id,
                        "scenario": scenario,
                        "step": step,
                        "instantaneous_energy": energy,
                        "cumulative_energy": sum(energy_profile[: step + 1]),
                        "timestamp": step / 20.0,
                        "total_episode_energy": total_energy,
                    }
                )

        df = pd.DataFrame(energy_data)
        df.to_csv(output_path / "energy_profiles.csv", index=False)

    def _save_performance_metrics_csv(self, dataset: Dict[str, Any], output_path: Path):
        """Save performance metrics as CSV."""
        metrics_data = []

        for (
            episode_id,
            scenario,
            success,
            collision,
            reward,
            energy,
            path_length,
        ) in dataset["performance_metrics"]["data"]:
            metrics_data.append(
                {
                    "episode_id": episode_id,
                    "scenario": scenario,
                    "success": success,
                    "collision": collision,
                    "total_reward": reward,
                    "total_energy": energy,
                    "path_length": path_length,
                    "energy_per_meter": energy / path_length if path_length > 0 else 0,
                }
            )

        df = pd.DataFrame(metrics_data)
        df.to_csv(output_path / "performance_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Collect comprehensive dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/main_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to trained model (optional)"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["nominal", "high_obstacles", "multi_floor"],
        help="Scenarios to collect data from",
    )
    parser.add_argument(
        "--output", type=str, default="data/collected_dataset", help="Output directory"
    )

    args = parser.parse_args()

    # Create collector
    collector = DataCollector(args.config)

    # Load model if provided
    if args.model:
        collector.load_trained_model(args.model)

    # Collect dataset
    dataset = collector.collect_comprehensive_dataset(args.episodes, args.scenarios)

    # Save dataset
    collector.save_dataset(dataset, args.output)

    # Print summary
    print("\n📊 DATA COLLECTION SUMMARY")
    print("=" * 50)
    print(f"Episodes collected: {dataset['metadata']['total_episodes']}")
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(
        f"Collection time: {dataset['collection_metadata']['collection_time_hours']:.1f} hours"
    )
    print(f"Trajectories: {dataset['trajectories']['count']}")
    print(f"Energy profiles: {dataset['energy_profiles']['count']}")
    print(f"Performance records: {dataset['performance_metrics']['count']}")
    print(f"Dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
