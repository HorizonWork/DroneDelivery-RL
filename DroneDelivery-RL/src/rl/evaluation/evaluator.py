import numpy as np
import logging
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from src.rl.evaluation.metrics_collector import MetricsCollector
from src.rl.evaluation.baseline_comparator import BaselineComparator
from src.rl.evaluation.energy_analyzer import EnergyAnalyzer
from src.rl.evaluation.trajectory_analyzer import TrajectoryAnalyzer

dataclass
class EpisodeResult:

    episode_id: int
    success: bool
    energy_consumption: float
    flight_time: float
    collision_occurred: bool
    ate_error: float
    final_distance_to_goal: float
    path_length: float
    num_waypoints: int
    num_replans: int
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    actions: List[List[float]] = field(default_factory=list)

dataclass
class EvaluationSummary:

    method_name: str
    success_rate: float
    mean_energy: float
    std_energy: float
    mean_time: float
    std_time: float
    collision_rate: float
    mean_ate: float
    std_ate: float

    total_episodes: int
    successful_episodes: int
    energy_efficiency_improvement: float

class DroneEvaluator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.num_evaluation_episodes = config.get('num_episodes', 100)
        self.episode_timeout = config.get('episode_timeout', 300.0)
        self.goal_tolerance = config.get('goal_tolerance', 0.5)
        self.success_distance_threshold = config.get('success_threshold', 0.5)

        self.save_trajectories = config.get('save_trajectories', True)
        self.save_detailed_logs = config.get('save_detailed_logs', True)
        self.output_directory = config.get('output_dir', 'evaluation_results')

        self.metrics_collector = MetricsCollector(config.get('metrics', {}))
        self.baseline_comparator = BaselineComparator(config.get('baselines', {}))
        self.energy_analyzer = EnergyAnalyzer(config.get('energy_analysis', {}))
        self.trajectory_analyzer = TrajectoryAnalyzer(config.get('trajectory_analysis', {}))

        self.episode_results: List[EpisodeResult] = []
        self.evaluation_summaries: Dict[str, EvaluationSummary] = {}

        os.makedirs(self.output_directory, exist_ok=True)

        self.logger.info("Drone Evaluator initialized")
        self.logger.info(f"Evaluation episodes: {self.num_evaluation_episodes}")
        self.logger.info(f"Episode timeout: {self.episode_timeout}s")
        self.logger.info(f"Output directory: {self.output_directory}")

    def evaluate_policy(self, policy_agent, environment,
                       method_name: str = "PPO_Agent") - EvaluationSummary:

        self.logger.info(f"Starting evaluation of {method_name}")
        self.logger.info(f"Running {self.num_evaluation_episodes} episodes")

        evaluation_start = time.time()
        episode_results = []

        for episode_id in range(self.num_evaluation_episodes):
            episode_start = time.time()

            result = self._evaluate_single_episode(
                policy_agent, environment, episode_id
            )

            episode_results.append(result)

            if (episode_id + 1)  10 == 0:
                self.logger.info(f"Completed {episode_id + 1}/{self.num_evaluation_episodes} episodes")
                self._log_interim_statistics(episode_results[-10:])

            if episode_id  20:
                recent_success_rate = np.mean([r.success for r in episode_results[-20:]])
                if recent_success_rate  0.1:
                    self.logger.warning("Low success rate detected - aborting evaluation")
                    break

        self.episode_results = episode_results

        summary = self._generate_evaluation_summary(episode_results, method_name)
        self.evaluation_summaries[method_name] = summary

        self._save_detailed_results(episode_results, method_name)

        evaluation_time = time.time() - evaluation_start

        self.logger.info(f"Evaluation completed in {evaluation_time:.1f}s")
        self.logger.info(f"Results - Success: {summary.success_rate:.1f}, "
                        f"Energy: {summary.mean_energy:.0f}{summary.std_energy:.0f}J, "
                        f"Time: {summary.mean_time:.1f}{summary.std_time:.1f}s")

        return summary

    def _evaluate_single_episode(self, policy_agent, environment,
                                episode_id: int) - EpisodeResult:

        observation = environment.reset()

        episode_reward = 0.0
        step_count = 0
        trajectory = []
        timestamps = []
        rewards = []
        actions = []

        energy_consumption = 0.0
        collision_occurred = False
        start_time = time.time()

        done = False

        while not done and step_count  int(self.episode_timeout  20):
            step_start = time.time()

            action, _ = policy_agent.select_action(observation, deterministic=True)

            next_observation, reward, done, info = environment.step(action)

            current_position = info.get('position', (0, 0, 0))
            trajectory.append(current_position)
            timestamps.append(time.time())
            rewards.append(reward)
            actions.append(action.tolist())

            episode_reward += reward
            energy_consumption += info.get('energy_consumption', 0.0)

            if info.get('collision', False):
                collision_occurred = True

            observation = next_observation
            step_count += 1

            step_duration = time.time() - step_start
            if step_duration  0.05:
                time.sleep(0.05 - step_duration)

        flight_time = time.time() - start_time

        final_position = info.get('position', (0, 0, 0))
        goal_position = info.get('goal_position', (0, 0, 0))
        final_distance = np.linalg.norm(np.array(final_position) - np.array(goal_position))

        success = final_distance = self.success_distance_threshold and not collision_occurred

        ate_error = info.get('ate_error', 0.0)

        path_length = 0.0
        if len(trajectory)  1:
            for i in range(1, len(trajectory)):
                segment_length = np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1]))
                path_length += segment_length

        result = EpisodeResult(
            episode_id=episode_id,
            success=success,
            energy_consumption=energy_consumption,
            flight_time=flight_time,
            collision_occurred=collision_occurred,
            ate_error=ate_error,
            final_distance_to_goal=final_distance,
            path_length=path_length,
            num_waypoints=len(trajectory),
            num_replans=info.get('num_replans', 0),
            trajectory=trajectory,
            timestamps=timestamps,
            rewards=rewards,
            actions=actions
        )

        return result

    def _generate_evaluation_summary(self, results: List[EpisodeResult],
                                   method_name: str) - EvaluationSummary:

        if not results:
            return EvaluationSummary(
                method_name=method_name,
                success_rate=0.0, mean_energy=0.0, std_energy=0.0,
                mean_time=0.0, std_time=0.0, collision_rate=0.0,
                mean_ate=0.0, std_ate=0.0, total_episodes=0,
                successful_episodes=0, energy_efficiency_improvement=0.0
            )

        successes = [r.success for r in results]
        success_rate = np.mean(successes)  100

        successful_results = [r for r in results if r.success]

        if successful_results:
            energies = [r.energy_consumption for r in successful_results]
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)

            times = [r.flight_time for r in successful_results]
            mean_time = np.mean(times)
            std_time = np.std(times)

            ate_errors = [r.ate_error for r in successful_results]
            mean_ate = np.mean(ate_errors)
            std_ate = np.std(ate_errors)
        else:
            mean_energy = std_energy = 0.0
            mean_time = std_time = 0.0
            mean_ate = std_ate = 0.0

        collisions = [r.collision_occurred for r in results]
        collision_rate = np.mean(collisions)  100

        summary = EvaluationSummary(
            method_name=method_name,
            success_rate=success_rate,
            mean_energy=mean_energy,
            std_energy=std_energy,
            mean_time=mean_time,
            std_time=std_time,
            collision_rate=collision_rate,
            mean_ate=mean_ate,
            std_ate=std_ate,
            total_episodes=len(results),
            successful_episodes=len(successful_results),
            energy_efficiency_improvement=0.0
        )

        return summary

    def compare_with_baselines(self, baseline_methods: List[str] = None) - Dict[str, Any]:

        if baseline_methods is None:
            baseline_methods = ['A_Only', 'RRT_PID', 'Random']

        comparison_results = self.baseline_comparator.compare_methods(
            self.evaluation_summaries, baseline_methods
        )

        if 'A_Only' in self.evaluation_summaries and len(self.evaluation_summaries)  1:
            baseline_energy = self.evaluation_summaries['A_Only'].mean_energy

            for method_name, summary in self.evaluation_summaries.items():
                if method_name != 'A_Only' and baseline_energy  0:
                    improvement = (baseline_energy - summary.mean_energy) / baseline_energy  100
                    summary.energy_efficiency_improvement = improvement

        return comparison_results

    def analyze_energy_efficiency(self) - Dict[str, Any]:

        if not self.episode_results:
            return {'error': 'No episode results available'}

        return self.energy_analyzer.analyze_episodes(self.episode_results)

    def analyze_trajectories(self) - Dict[str, Any]:

        if not self.episode_results:
            return {'error': 'No episode results available'}

        return self.trajectory_analyzer.analyze_episodes(self.episode_results)

    def _log_interim_statistics(self, recent_results: List[EpisodeResult]):

        if not recent_results:
            return

        successes = [r.success for r in recent_results]
        success_rate = np.mean(successes)  100

        successful_results = [r for r in recent_results if r.success]

        if successful_results:
            avg_energy = np.mean([r.energy_consumption for r in successful_results])
            avg_time = np.mean([r.flight_time for r in successful_results])

            self.logger.info(f"Recent 10 episodes - Success: {success_rate:.1f}, "
                           f"Energy: {avg_energy:.0f}J, Time: {avg_time:.1f}s")

    def _save_detailed_results(self, results: List[EpisodeResult], method_name: str):

        if not self.save_detailed_logs:
            return

        method_dir = Path(self.output_directory) / method_name
        method_dir.mkdir(exist_ok=True)

        summaries = []
        for result in results:
            summary = {
                'episode_id': result.episode_id,
                'success': result.success,
                'energy_consumption': result.energy_consumption,
                'flight_time': result.flight_time,
                'collision_occurred': result.collision_occurred,
                'ate_error': result.ate_error,
                'final_distance_to_goal': result.final_distance_to_goal,
                'path_length': result.path_length,
                'num_waypoints': result.num_waypoints,
                'num_replans': result.num_replans
            }
            summaries.append(summary)

        with open(method_dir / 'episode_summaries.json', 'w') as f:
            json.dump(summaries, f, indent=2)

        if self.save_trajectories:
            trajectories = {
                'episodes': [
                    {
                        'episode_id': r.episode_id,
                        'trajectory': r.trajectory,
                        'timestamps': r.timestamps,
                        'success': r.success
                    }
                    for r in results
                ]
            }

            with open(method_dir / 'trajectories.json', 'w') as f:
                json.dump(trajectories, f, indent=2)

        self.logger.info(f"Detailed results saved to {method_dir}")

    def generate_evaluation_report(self) - str:

        report_lines = []
        report_lines.append("="  80)
        report_lines.append("INDOOR DRONE DELIVERY EVALUATION REPORT")
        report_lines.append("="  80)
        report_lines.append("")

        if self.evaluation_summaries:
            report_lines.append("PERFORMANCE SUMMARY (Table 3 Format)")
            report_lines.append("-"  50)
            report_lines.append(f"{'Method':15} {'Success':8} {'Energy(J)':12} {'Time(s)':10} {'Collisions':11} {'ATE(m)':8}")
            report_lines.append("-"  50)

            for method_name, summary in self.evaluation_summaries.items():
                report_lines.append(
                    f"{method_name:15} "
                    f"{summary.success_rate:8.1f} "
                    f"{summary.mean_energy:8.0f}{summary.std_energy:3.0f} "
                    f"{summary.mean_time:6.1f}{summary.std_time:3.1f} "
                    f"{summary.collision_rate:11.1f} "
                    f"{summary.mean_ate:4.3f}{summary.std_ate:4.3f}"
                )

            report_lines.append("")

        energy_analysis = self.analyze_energy_efficiency()
        if 'energy_savings_vs_baseline' in energy_analysis:
            report_lines.append("ENERGY EFFICIENCY ANALYSIS")
            report_lines.append("-"  30)
            report_lines.append(f"Energy savings vs A Only: {energy_analysis['energy_savings_vs_baseline']:.1f}")
            report_lines.append(f"Average power consumption: {energy_analysis.get('average_power', 0):.1f}W")
            report_lines.append("")

        trajectory_analysis = self.analyze_trajectories()
        if 'path_efficiency' in trajectory_analysis:
            report_lines.append("TRAJECTORY ANALYSIS")
            report_lines.append("-"  20)
            report_lines.append(f"Average path efficiency: {trajectory_analysis['path_efficiency']:.3f}")
            report_lines.append(f"Average smoothness score: {trajectory_analysis.get('smoothness_score', 0):.3f}")
            report_lines.append("")

        if self.episode_results:
            report_lines.append("DETAILED STATISTICS")
            report_lines.append("-"  20)
            report_lines.append(f"Total episodes evaluated: {len(self.episode_results)}")

            successful_episodes = [r for r in self.episode_results if r.success]
            failed_episodes = [r for r in self.episode_results if not r.success]

            report_lines.append(f"Successful episodes: {len(successful_episodes)}")
            report_lines.append(f"Failed episodes: {len(failed_episodes)}")

            if successful_episodes:
                avg_replans = np.mean([r.num_replans for r in successful_episodes])
                report_lines.append(f"Average replans per successful episode: {avg_replans:.1f}")

            report_lines.append("")

        if any(not r.success for r in self.episode_results):
            report_lines.append("FAILURE ANALYSIS")
            report_lines.append("-"  15)

            failure_reasons = defaultdict(int)
            for result in self.episode_results:
                if not result.success:
                    if result.collision_occurred:
                        failure_reasons['Collision'] += 1
                    elif result.final_distance_to_goal  self.success_distance_threshold:
                        failure_reasons['Goal not reached'] += 1
                    else:
                        failure_reasons['Other'] += 1

            for reason, count in failure_reasons.items():
                percentage = (count / len(self.episode_results))  100
                report_lines.append(f"{reason}: {count} episodes ({percentage:.1f})")

            report_lines.append("")

        report_lines.append("="  80)

        return "\n".join(report_lines)

    def export_results_csv(self, filename: str = "evaluation_results.csv"):

        if not self.episode_results:
            self.logger.warning("No results to export")
            return

        import csv

        filepath = Path(self.output_directory) / filename

        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'episode_id', 'success', 'energy_consumption', 'flight_time',
                'collision_occurred', 'ate_error', 'final_distance_to_goal',
                'path_length', 'num_waypoints', 'num_replans'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.episode_results:
                writer.writerow({
                    'episode_id': result.episode_id,
                    'success': result.success,
                    'energy_consumption': result.energy_consumption,
                    'flight_time': result.flight_time,
                    'collision_occurred': result.collision_occurred,
                    'ate_error': result.ate_error,
                    'final_distance_to_goal': result.final_distance_to_goal,
                    'path_length': result.path_length,
                    'num_waypoints': result.num_waypoints,
                    'num_replans': result.num_replans
                })

        self.logger.info(f"Results exported to {filepath}")

    def get_performance_statistics(self) - Dict[str, Any]:

        if not self.episode_results:
            return {'error': 'No evaluation results available'}

        successes = [r.success for r in self.episode_results]
        energies = [r.energy_consumption for r in self.episode_results if r.success]
        times = [r.flight_time for r in self.episode_results if r.success]
        collisions = [r.collision_occurred for r in self.episode_results]
        ate_errors = [r.ate_error for r in self.episode_results if r.success]

        stats = {
            'episode_statistics': {
                'total_episodes': len(self.episode_results),
                'successful_episodes': len([r for r in self.episode_results if r.success]),
                'failed_episodes': len([r for r in self.episode_results if not r.success]),
                'success_rate': np.mean(successes)  100,
                'collision_rate': np.mean(collisions)  100
            },
            'energy_statistics': {
                'mean_energy': np.mean(energies) if energies else 0.0,
                'std_energy': np.std(energies) if energies else 0.0,
                'min_energy': np.min(energies) if energies else 0.0,
                'max_energy': np.max(energies) if energies else 0.0,
                'median_energy': np.median(energies) if energies else 0.0
            },
            'time_statistics': {
                'mean_time': np.mean(times) if times else 0.0,
                'std_time': np.std(times) if times else 0.0,
                'min_time': np.min(times) if times else 0.0,
                'max_time': np.max(times) if times else 0.0,
                'median_time': np.median(times) if times else 0.0
            },
            'localization_statistics': {
                'mean_ate': np.mean(ate_errors) if ate_errors else 0.0,
                'std_ate': np.std(ate_errors) if ate_errors else 0.0,
                'max_ate': np.max(ate_errors) if ate_errors else 0.0,
                'centimeter_accuracy_achieved': np.mean(ate_errors) = 0.05 if ate_errors else False
            }
        }

        return stats
