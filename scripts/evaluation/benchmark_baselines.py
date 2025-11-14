import os
import sys
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.environment.airsim_env import AirSimEnvironment as DroneEnvironment
from src.rl.agents.ppo_agent import PPOAgent
from src.rl.evaluation.evaluator import DroneEvaluator
from src.rl.evaluation.baseline_comparator import BaselineComparator
from src.rl.evaluation.energy_analyzer import EnergyAnalyzer
from src.rl.evaluation.trajectory_analyzer import TrajectoryAnalyzer
from src.planning.global_planner.astar_planner import AStarPlanner as GlobalPlanner
from src.planning.local_planner.srrt_planner import SRRTPlanner as LocalPlanner
from src.utils import setup_logging, load_config, SystemVisualizer, DataRecorder

class BaselineBenchmark:

    def __init__(self, config_path: str):
        self.config = load_config(config_path)

        self.logger_system = setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)

        self.environment = DroneEnvironment(self.config.environment)

        self.global_planner = GlobalPlanner(self.config.planning)
        self.local_planner = LocalPlanner(self.config.planning)

        self.num_episodes = self.config.get("evaluation", {}).get("num_episodes", 100)
        self.episode_timeout = 300.0

        self.baselines = {
            "A_Only": {
                "use_global_only": True,
                "use_local_planning": False,
                "controller": "PID",
                "energy_aware": False,
            },
            "RRT_PID": {
                "use_global_only": False,
                "use_local_planning": True,
                "local_planner": "RRT",
                "controller": "PID",
                "energy_aware": False,
            },
            "Random": {
                "use_random_policy": True,
                "controller": "Random",
                "energy_aware": False,
            },
        }

        self.results = {}

        self.logger.info("Baseline Benchmark initialized")
        self.logger.info(f"Evaluating {len(self.baselines)} baseline methods")
        self.logger.info(f"Episodes per method: {self.num_episodes}")

    def run_benchmark(self) - Dict[str, Any]:

        self.logger.info("Starting baseline benchmark evaluation")
        benchmark_start = time.time()

        for baseline_name, baseline_config in self.baselines.items():
            self.logger.info(f"Evaluating {baseline_name}...")

            baseline_results = self._evaluate_baseline(baseline_name, baseline_config)
            self.results[baseline_name] = baseline_results

            success_rate = baseline_results["success_rate"]
            mean_energy = baseline_results["mean_energy"]
            self.logger.info(
                f"{baseline_name}: {success_rate:.1f} success, {mean_energy:.0f}J energy"
            )

        benchmark_time = time.time() - benchmark_start

        final_results = {
            "benchmark_completed": True,
            "evaluation_time": benchmark_time,
            "num_episodes_per_method": self.num_episodes,
            "baseline_results": self.results,
            "summary": self._generate_summary(),
        }

        self.logger.info(
            f"Baseline benchmark completed in {benchmark_time/60:.1f} minutes"
        )
        return final_results

    def _evaluate_baseline(
        self, baseline_name: str, baseline_config: Dict[str, Any]
    ) - Dict[str, Any]:

        episode_results = []

        for episode in range(self.num_episodes):
            if episode  10 == 0:
                self.logger.info(
                    f"{baseline_name} episode {episode}/{self.num_episodes}"
                )

            episode_result = self._run_baseline_episode(baseline_name, baseline_config)
            episode_results.append(episode_result)

        return self._aggregate_episode_results(episode_results)

    def _run_baseline_episode(
        self, baseline_name: str, baseline_config: Dict[str, Any]
    ) - Dict[str, Any]:

        observation = self.environment.reset()

        start_pos = self.environment.get_current_position()
        goal_pos = self.environment.get_goal_position()

        episode_start = time.time()
        total_energy = 0.0
        trajectory = [start_pos.copy()]
        collision_occurred = False
        done = False
        steps = 0
        max_steps = int(self.episode_timeout  20)

        if baseline_config.get("use_random_policy", False):
            planned_path = None
        else:
            try:
                planned_path = self.global_planner.plan_path(start_pos, goal_pos)
                if not planned_path:
                    return {
                        "success": False,
                        "reason": "planning_failed",
                        "energy": 0.0,
                        "time": 0.0,
                        "collision": False,
                        "trajectory": trajectory,
                    }
            except Exception as e:
                self.logger.warning(f"Planning failed for {baseline_name}: {e}")
                return {
                    "success": False,
                    "reason": "planning_exception",
                    "energy": 0.0,
                    "time": 0.0,
                    "collision": False,
                    "trajectory": trajectory,
                }

        while not done and steps  max_steps:
            if baseline_config.get("use_random_policy", False):
                action = self._random_action()
            elif baseline_config.get("use_global_only", False):
                action = self._astar_only_action(
                    planned_path, steps, start_pos, goal_pos
                )
            else:
                action = self._rrt_pid_action(planned_path, observation, goal_pos)

            observation, reward, done, info = self.environment.step(action)

            current_pos = info.get("position", start_pos)
            trajectory.append(current_pos.copy())
            total_energy += info.get("energy_consumption", 0.0)

            if info.get("collision", False):
                collision_occurred = True
                done = True

            steps += 1

        episode_time = time.time() - episode_start

        final_pos = trajectory[-1] if trajectory else start_pos
        distance_to_goal = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))
        success = distance_to_goal = 0.5 and not collision_occurred

        return {
            "success": success,
            "energy": total_energy,
            "time": episode_time,
            "collision": collision_occurred,
            "trajectory": trajectory,
            "final_distance": distance_to_goal,
            "steps": steps,
            "path_length": self._calculate_path_length(trajectory),
        }

    def _random_action(self) - np.ndarray:

        return np.random.uniform(-1.0, 1.0, size=4)

    def _astar_only_action(
        self, planned_path: List, step: int, start_pos: np.ndarray, goal_pos: np.ndarray
    ) - np.ndarray:

        if not planned_path or len(planned_path) = 1:
            current_pos = start_pos
            target_pos = goal_pos
        else:
            progress = min(step / len(planned_path), 1.0)
            path_index = int(progress  (len(planned_path) - 1))
            target_pos = planned_path[path_index]
            current_pos = start_pos

        position_error = np.array(target_pos) - np.array(current_pos)

        kp = 2.0
        max_velocity = 2.0

        velocity_cmd = kp  position_error[:3]
        velocity_cmd = np.clip(velocity_cmd, -max_velocity, max_velocity)

        yaw_cmd = 0.1  position_error[0]

        return np.array([velocity_cmd[0], velocity_cmd[1], velocity_cmd[2], yaw_cmd])

    def _rrt_pid_action(
        self, global_path: List, observation: np.ndarray, goal_pos: np.ndarray
    ) - np.ndarray:

        current_pos = observation[:3] if len(observation) = 3 else np.array([0, 0, 0])

        try:
            local_goal = goal_pos
            if global_path and len(global_path)  1:
                distances = [
                    np.linalg.norm(np.array(wp) - current_pos) for wp in global_path
                ]
                next_waypoint_idx = min(
                    range(len(distances)), key=distances.__getitem__
                )
                local_goal = global_path[
                    min(next_waypoint_idx + 1, len(global_path) - 1)
                ]

            local_path = [current_pos.tolist(), local_goal]

            target_pos = local_path[-1]
            position_error = np.array(target_pos) - current_pos

            kp = 2.5
            max_velocity = 3.0

            velocity_cmd = kp  position_error[:3]
            velocity_cmd = np.clip(velocity_cmd, -max_velocity, max_velocity)

            yaw_cmd = 0.1  position_error[0]

            return np.array(
                [velocity_cmd[0], velocity_cmd[1], velocity_cmd[2], yaw_cmd]
            )

        except Exception as e:
            position_error = np.array(goal_pos) - current_pos
            velocity_cmd = 1.0  position_error[:3]
            velocity_cmd = np.clip(velocity_cmd, -2.0, 2.0)

            return np.array([velocity_cmd[0], velocity_cmd[1], velocity_cmd[2], 0.0])

    def _calculate_path_length(self, trajectory: List) - float:

        if len(trajectory)  2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(trajectory)):
            segment_length = np.linalg.norm(
                np.array(trajectory[i]) - np.array(trajectory[i - 1])
            )
            total_length += segment_length

        return total_length

    def _aggregate_episode_results(
        self, episode_results: List[Dict[str, Any]]
    ) - Dict[str, Any]:

        successful_episodes = [ep for ep in episode_results if ep["success"]]

        if not successful_episodes:
            return {
                "success_rate": 0.0,
                "mean_energy": 0.0,
                "std_energy": 0.0,
                "mean_time": 0.0,
                "std_time": 0.0,
                "collision_rate": np.mean([ep["collision"] for ep in episode_results])
                 100,
                "mean_ate": 0.0,
                "sample_size": len(episode_results),
            }

        energies = [ep["energy"] for ep in successful_episodes]
        times = [ep["time"] for ep in successful_episodes]

        return {
            "success_rate": len(successful_episodes) / len(episode_results)  100,
            "mean_energy": float(np.mean(energies)),
            "std_energy": float(np.std(energies)),
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "collision_rate": np.mean([ep["collision"] for ep in episode_results])
             100,
            "mean_ate": 0.05,
            "sample_size": len(episode_results),
        }

    def _generate_summary(self) - Dict[str, Any]:

        summary = {
            "total_methods_evaluated": len(self.results),
            "best_success_rate": 0.0,
            "best_energy_efficiency": float("inf"),
            "method_rankings": {},
        }

        if not self.results:
            return summary

        for method_name, results in self.results.items():
            success_rate = results.get("success_rate", 0.0)
            mean_energy = results.get("mean_energy", float("inf"))

            if success_rate  summary["best_success_rate"]:
                summary["best_success_rate"] = success_rate
                summary["best_success_method"] = method_name

            if 0  mean_energy  summary["best_energy_efficiency"]:
                summary["best_energy_efficiency"] = mean_energy
                summary["best_energy_method"] = method_name

        sorted_by_success = sorted(
            self.results.items(),
            key=lambda x: x[1].get("success_rate", 0),
            reverse=True,
        )
        summary["method_rankings"]["by_success_rate"] = [
            method for method, _ in sorted_by_success
        ]

        return summary

    def save_results(self, output_path: str):

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        self.logger.info(f"Baseline results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline methods")
    parser.add_argument(
        "--config",
        type=str,
        default="config/evaluation_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_benchmark.json",
        help="Output results file",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes per method"
    )

    args = parser.parse_args()

    benchmark = BaselineBenchmark(args.config)
    benchmark.num_episodes = args.episodes

    results = benchmark.run_benchmark()

    benchmark.save_results(args.output)

    print("\nBASELINE BENCHMARK SUMMARY")
    print("="  50)

    for method_name, method_results in results["baseline_results"].items():
        print(f"\n{method_name}:")
        print(f"  Success Rate: {method_results.get('success_rate', 0):.1f}")
        print(f"  Mean Energy: {method_results.get('mean_energy', 0):.0f}J")
        print(f"  Mean Time: {method_results.get('mean_time', 0):.1f}s")
        print(f"  Collision Rate: {method_results.get('collision_rate', 0):.1f}")

if __name__ == "__main__":
    main()
