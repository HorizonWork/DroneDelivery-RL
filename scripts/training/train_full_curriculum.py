import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import random
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.environment.airsim_env import AirSimEnvironment as DroneEnvironment
from src.rl.agents.ppo_agent import PPOAgent
from src.rl.training.curriculum_trainer import CurriculumManager
from src.rl.initialization import initialize_rl_system
from src.utils import setup_logging, load_config

def set_global_seeds(seed: int) - None:

    logging.info("Setting global random seed to d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FullCurriculumTrainer:

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        rl_seed = 42
        if isinstance(self.config.rl, dict):
            rl_seed = int(self.config.rl.get("seed", rl_seed))
        set_global_seeds(rl_seed)
        self.seed = rl_seed

        self.logger_system = setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)

        self.curriculum_phases = [
            {
                "name": "Phase_1_SingleFloor_Static",
                "description": "Single floor with static obstacles",
                "config": {
                    "floors": 1,
                    "obstacle_density": 0.10,
                    "dynamic_obstacles": False,
                    "complexity_level": "low",
                },
                "success_criteria": {
                    "min_success_rate": 85.0,
                    "min_episodes": 500,
                    "max_timesteps": 1_000_000,
                },
                "timestep_allocation": 1_000_000,
            },
            {
                "name": "Phase_2_TwoFloor_Dynamic",
                "description": "Two floors with dynamic obstacles",
                "config": {
                    "floors": 2,
                    "obstacle_density": 0.15,
                    "dynamic_obstacles": True,
                    "dynamic_count": 3,
                    "complexity_level": "medium",
                },
                "success_criteria": {
                    "min_success_rate": 90.0,
                    "min_episodes": 1000,
                    "max_timesteps": 2_000_000,
                },
                "timestep_allocation": 2_000_000,
            },
            {
                "name": "Phase_3_FiveFloor_Complex",
                "description": "Five floors with full complexity",
                "config": {
                    "floors": 5,
                    "obstacle_density": 0.20,
                    "dynamic_obstacles": True,
                    "dynamic_count": 5,
                    "human_obstacles": True,
                    "complexity_level": "high",
                },
                "success_criteria": {
                    "min_success_rate": 96.0,
                    "min_episodes": 2000,
                    "max_timesteps": 2_000_000,
                },
                "timestep_allocation": 2_000_000,
            },
        ]

        self.rl_system = initialize_rl_system(self.config.rl)
        self.agent = self.rl_system["agent"]

        self.current_phase_index = 0
        self.global_timestep = 0
        self.phase_timestep = 0
        self.episode_count = 0

        self.phase_results = []

        self.logger.info("Full Curriculum Trainer initialized")
        self.logger.info(f"Total phases: {len(self.curriculum_phases)}")
        self.logger.info(f"Target timesteps: 5,000,000")

    def train_full_curriculum(self) - Dict[str, Any]:

        self.logger.info("=== STARTING FULL CURRICULUM TRAINING ===")
        training_start = time.time()

        for phase_index, phase_config in enumerate(self.curriculum_phases):
            self.current_phase_index = phase_index
            phase_results = self._train_phase(phase_config)
            self.phase_results.append(phase_results)

            if not phase_results["success"]:
                self.logger.error(f"Phase {phase_index + 1} failed - stopping training")
                break

        total_training_time = time.time() - training_start

        curriculum_results = {
            "curriculum_completed": True,
            "total_training_time_hours": total_training_time / 3600,
            "total_timesteps": self.global_timestep,
            "total_episodes": self.episode_count,
            "phases_completed": len(self.phase_results),
            "phase_results": self.phase_results,
            "final_performance": self._get_final_performance(),
        }

        self._save_curriculum_results(curriculum_results)

        self.logger.info("=== FULL CURRICULUM TRAINING COMPLETED ===")
        return curriculum_results

    def _train_phase(self, phase_config: Dict[str, Any]) - Dict[str, Any]:

        phase_name = phase_config["name"]
        phase_start_time = time.time()
        self.phase_timestep = 0

        self.logger.info(f"Starting {phase_name}")
        self.logger.info(
            f"Target success rate: {phase_config['success_criteria']['min_success_rate']}"
        )

        environment = self._create_phase_environment(phase_config["config"])

        phase_episodes = 0
        phase_rewards = []
        phase_success_rates = []
        phase_energies = []

        max_timesteps = phase_config["timestep_allocation"]
        evaluation_frequency = 10_000

        while self.phase_timestep  max_timesteps:
            episode_result = self._run_phase_episode(environment)

            phase_episodes += 1
            phase_rewards.append(episode_result["reward"])
            phase_energies.append(episode_result["energy"])

            if len(phase_rewards) = 20:
                recent_successes = sum(1 for r in phase_rewards[-20:] if r  400)
                success_rate = recent_successes / 20  100
                phase_success_rates.append(success_rate)

            if self.phase_timestep  evaluation_frequency == 0:
                current_performance = self._evaluate_phase_performance(environment)
                self.logger.info(
                    f"{phase_name}: {self.phase_timestep:,}/{max_timesteps:,} steps, "
                    f"{current_performance['success_rate']:.1f} success"
                )

            if self._check_phase_completion(phase_config, phase_success_rates):
                self.logger.info(f"{phase_name} completed early - success criteria met")
                break

        phase_training_time = time.time() - phase_start_time

        final_performance = self._evaluate_phase_performance(
            environment, num_episodes=50
        )

        phase_results = {
            "phase_name": phase_name,
            "phase_index": self.current_phase_index,
            "success": final_performance["success_rate"]
            = phase_config["success_criteria"]["min_success_rate"],
            "timesteps_used": self.phase_timestep,
            "episodes_trained": phase_episodes,
            "training_time_hours": phase_training_time / 3600,
            "final_success_rate": final_performance["success_rate"],
            "final_energy_consumption": final_performance["mean_energy"],
            "final_collision_rate": final_performance["collision_rate"],
            "training_history": {
                "rewards": phase_rewards,
                "success_rates": phase_success_rates,
                "energies": phase_energies,
            },
        }

        self.logger.info(f"{phase_name} completed:")
        self.logger.info(f"  Success: {phase_results['success']}")
        self.logger.info(
            f"  Final success rate: {final_performance['success_rate']:.1f}"
        )
        self.logger.info(f"  Training time: {phase_training_time/3600:.1f}h")
        self.logger.info(f"  Timesteps used: {self.phase_timestep:,}")

        return phase_results

    def _create_phase_environment(
        self, phase_config: Dict[str, Any]
    ) - DroneEnvironment:

        env_config = self.config.environment.copy()

        env_config["building"]["floors"] = phase_config["floors"]
        env_config["obstacles"]["density"] = phase_config["obstacle_density"]
        env_config["obstacles"]["dynamic_obstacles"] = phase_config["dynamic_obstacles"]

        if "dynamic_count" in phase_config:
            env_config["obstacles"]["dynamic_count"] = phase_config["dynamic_count"]

        if "human_obstacles" in phase_config:
            env_config["obstacles"]["human_obstacles"] = phase_config["human_obstacles"]

        complexity = phase_config.get("complexity_level", "medium")
        if complexity == "low":
            env_config["obstacles"]["density"] = min(
                env_config["obstacles"]["density"], 0.10
            )
        elif complexity == "high":
            env_config["obstacles"]["density"] = max(
                env_config["obstacles"]["density"], 0.20
            )
            env_config["obstacles"]["dynamic_count"] = max(
                env_config["obstacles"].get("dynamic_count", 0), 5
            )

        return DroneEnvironment(env_config)

    def _run_phase_episode(self, environment: DroneEnvironment) - Dict[str, Any]:

        observation = environment.reset()

        episode_reward = 0.0
        episode_energy = 0.0
        episode_steps = 0
        done = False

        while not done and episode_steps  1000:
            action, _, _ = self.agent.select_action(observation, training=True)

            next_observation, reward, done, info = environment.step(action)

            episode_reward += reward
            episode_energy += info.get("energy_consumption", 0.0)
            episode_steps += 1

            self.phase_timestep += 1
            self.global_timestep += 1

            observation = next_observation

        self.episode_count += 1

        return {
            "reward": episode_reward,
            "energy": episode_energy,
            "steps": episode_steps,
            "success": info.get("success", False),
            "collision": info.get("collision", False),
        }

    def _evaluate_phase_performance(
        self, environment: DroneEnvironment, num_episodes: int = 20
    ) - Dict[str, Any]:

        eval_results = []

        for episode in range(num_episodes):
            observation = environment.reset()

            episode_reward = 0.0
            episode_energy = 0.0
            done = False
            steps = 0

            while not done and steps  1000:
                action, _, _ = self.agent.select_action(observation, deterministic=True)
                observation, reward, done, info = environment.step(action)

                episode_reward += reward
                episode_energy += info.get("energy_consumption", 0.0)
                steps += 1

            eval_results.append(
                {
                    "reward": episode_reward,
                    "energy": episode_energy,
                    "success": info.get("success", False),
                    "collision": info.get("collision", False),
                }
            )

        successes = [r for r in eval_results if r["success"]]

        return {
            "success_rate": len(successes) / len(eval_results)  100,
            "mean_energy": (
                np.mean([r["energy"] for r in successes]) if successes else 0
            ),
            "collision_rate": np.mean([r["collision"] for r in eval_results])  100,
            "mean_reward": np.mean([r["reward"] for r in eval_results]),
        }

    def _check_phase_completion(
        self, phase_config: Dict[str, Any], success_rates: List[float]
    ) - bool:

        if len(success_rates)  10:
            return False

        criteria = phase_config["success_criteria"]

        recent_success_rate = np.mean(success_rates[-10:])
        min_success_rate = criteria["min_success_rate"]

        min_episodes = criteria.get("min_episodes", 100)
        episodes_completed = len(success_rates)

        return (
            recent_success_rate = min_success_rate
            and episodes_completed = min_episodes
        )

    def _get_final_performance(self) - Dict[str, Any]:

        if not self.phase_results:
            return {}

        final_phase = self.phase_results[-1]

        return {
            "final_success_rate": final_phase["final_success_rate"],
            "final_energy_consumption": final_phase["final_energy_consumption"],
            "final_collision_rate": final_phase["final_collision_rate"],
            "target_achieved": final_phase["final_success_rate"] = 96.0,
        }

    def _save_curriculum_results(self, results: Dict[str, Any]):

        output_dir = Path("models/curriculum_training")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / "curriculum_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        summary_file = output_dir / "phase_summary.json"
        phase_summary = {
            "phases": [
                {
                    "name": phase["phase_name"],
                    "success": phase["success"],
                    "final_success_rate": phase["final_success_rate"],
                    "training_time_hours": phase["training_time_hours"],
                    "timesteps_used": phase["timesteps_used"],
                }
                for phase in results["phase_results"]
            ],
            "total_success": all(
                phase["success"] for phase in results["phase_results"]
            ),
            "total_time": results["total_training_time_hours"],
            "final_performance": results["final_performance"],
        }

        with open(summary_file, "w") as f:
            json.dump(phase_summary, f, indent=2)

        self.logger.info(f"Curriculum results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Full curriculum training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/main_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/curriculum_training",
        help="Output directory for results",
    )

    args = parser.parse_args()

    trainer = FullCurriculumTrainer(args.config)

    results = trainer.train_full_curriculum()

    print("\n" + "="  70)
    print(" CURRICULUM TRAINING COMPLETED!")
    print("="  70)

    for i, phase in enumerate(results["phase_results"]):
        status = " SUCCESS" if phase["success"] else " FAILED"
        print(f"Phase {i+1} ({phase['phase_name']}): {status}")
        print(f"  Success Rate: {phase['final_success_rate']:.1f}")
        print(f"  Training Time: {phase['training_time_hours']:.1f}h")
        print(f"  Timesteps: {phase['timesteps_used']:,}")

    final_perf = results["final_performance"]
    print(f"\nFinal Performance:")
    print(f"  Success Rate: {final_perf['final_success_rate']:.1f}")
    print(f"  Energy Consumption: {final_perf['final_energy_consumption']:.0f}J")
    print(
        f"  Target Achieved: {' YES' if final_perf['target_achieved'] else ' NO'}"
    )

    print(f"\nTotal Training Time: {results['total_training_time_hours']:.1f} hours")
    print(f"Total Timesteps: {results['total_timesteps']:,}")
    print("="  70)

if __name__ == "__main__":
    main()
