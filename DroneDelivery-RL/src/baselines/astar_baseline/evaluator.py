import numpy as np
from typing import Dict, List, Any, Tuple
import time
from dataclasses import dataclass, field

dataclass
class EpisodeResult:

    success: bool = False
    collision: bool = False
    timeout: bool = False
    energy_consumed: float = 0.0
    flight_time: float = 0.0
    path_length: float = 0.0
    ate_error: float = 0.0
    final_distance_to_goal: float = 0.0

dataclass
class EvaluationMetrics:

    success_rate: float = 0.0
    collision_rate: float = 0.0
    timeout_rate: float = 0.0
    mean_energy: float = 0.0
    std_energy: float = 0.0
    mean_time: float = 0.0
    std_time: float = 0.0
    mean_ate: float = 0.0
    std_ate: float = 0.0
    episodes_completed: int = 0

    target_success_rate: float = 96.0
    target_energy_mean: float = 820.0
    target_time_mean: float = 32.0
    target_collision_rate: float = 1.2
    target_ate_mean: float = 0.11

class AStarEvaluator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.episode_results: List[EpisodeResult] = []

        self.num_episodes = config.get('num_episodes', 200)
        self.max_episode_time = config.get('max_episode_time', 300.0)
        self.goal_tolerance = config.get('goal_tolerance', 0.5)

        self.control_dt = config.get('control_dt', 0.05)
        self.drone_mass = config.get('drone_mass', 1.5)

    def evaluate_episode(self, astar_controller, pid_controller, environment) - EpisodeResult:

        result = EpisodeResult()

        obs = environment.reset()
        pid_controller.reset()

        start_time = time.time()
        total_energy = 0.0
        path_length = 0.0
        previous_pos = None

        current_pos = environment.get_drone_position()
        goal_pos = environment.get_goal_position()

        obstacles = environment.get_obstacles()
        astar_controller.update_occupancy_grid(obstacles)
        path = astar_controller.plan_path(current_pos, goal_pos)

        if not path:
            result.success = False
            result.timeout = True
            return result

        astar_controller.set_path(path)

        step = 0
        max_steps = int(self.max_episode_time / self.control_dt)

        while step  max_steps:
            current_pos = environment.get_drone_position()
            current_yaw = environment.get_drone_yaw()

            distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(goal_pos))
            if distance_to_goal  self.goal_tolerance:
                result.success = True
                break

            if environment.check_collision():
                result.collision = True
                break

            waypoint = astar_controller.get_next_waypoint(current_pos)
            if waypoint is None:
                result.success = False
                break

            obstacles = environment.get_obstacles()
            if not astar_controller.is_path_valid(obstacles):
                astar_controller.update_occupancy_grid(obstacles)
                new_path = astar_controller.plan_path(current_pos, goal_pos)

                if new_path:
                    astar_controller.set_path(new_path)
                    waypoint = astar_controller.get_next_waypoint(current_pos)
                else:
                    result.success = False
                    break

            if waypoint:
                vx, vy, vz, yaw_rate = pid_controller.compute_control(
                    current_pos, current_yaw, waypoint
                )
            else:
                vx = vy = vz = yaw_rate = 0.0

            action = np.array([vx, vy, vz, yaw_rate])
            obs, reward, done, info = environment.step(action)

            velocity_magnitude = np.linalg.norm([vx, vy, vz])
            thrust_estimate = self.drone_mass  (velocity_magnitude + 9.81)
            energy_step = (thrust_estimate  2)  self.control_dt
            total_energy += energy_step

            if previous_pos is not None:
                path_length += np.linalg.norm(np.array(current_pos) - np.array(previous_pos))
            previous_pos = current_pos

            step += 1

        result.flight_time = time.time() - start_time
        result.energy_consumed = total_energy
        result.path_length = path_length
        result.final_distance_to_goal = distance_to_goal
        result.ate_error = environment.get_ate_error() if hasattr(environment, 'get_ate_error') else 0.0

        if step = max_steps and not result.success and not result.collision:
            result.timeout = True

        return result

    def evaluate_multiple_episodes(self, astar_controller, pid_controller,
                                 environment, num_episodes: int = None) - EvaluationMetrics:

        if num_episodes is None:
            num_episodes = self.num_episodes

        self.episode_results = []

        print(f"Evaluating A + PID baseline over {num_episodes} episodes...")

        for episode in range(num_episodes):
            if episode  50 == 0:
                print(f"Episode {episode}/{num_episodes}")

            result = self.evaluate_episode(astar_controller, pid_controller, environment)
            self.episode_results.append(result)

        return self._compute_metrics()

    def _compute_metrics(self) - EvaluationMetrics:

        if not self.episode_results:
            return EvaluationMetrics()

        metrics = EvaluationMetrics()
        metrics.episodes_completed = len(self.episode_results)

        successes = [r for r in self.episode_results if r.success]
        collisions = [r for r in self.episode_results if r.collision]
        timeouts = [r for r in self.episode_results if r.timeout]

        metrics.success_rate = len(successes) / len(self.episode_results)  100
        metrics.collision_rate = len(collisions) / len(self.episode_results)  100
        metrics.timeout_rate = len(timeouts) / len(self.episode_results)  100

        if successes:
            energies = [r.energy_consumed for r in successes]
            metrics.mean_energy = np.mean(energies)
            metrics.std_energy = np.std(energies)

        if successes:
            times = [r.flight_time for r in successes]
            metrics.mean_time = np.mean(times)
            metrics.std_time = np.std(times)

        ate_errors = [r.ate_error for r in self.episode_results if r.ate_error  0]
        if ate_errors:
            metrics.mean_ate = np.mean(ate_errors)
            metrics.std_ate = np.std(ate_errors)

        return metrics

    def print_results(self, metrics: EvaluationMetrics):

        print("\n" + "="60)
        print("A + PID BASELINE EVALUATION RESULTS")
        print("="60)

        print(f"Episodes completed: {metrics.episodes_completed}")
        print(f"Success rate: {metrics.success_rate:.1f} (target: {metrics.target_success_rate:.1f})")
        print(f"Collision rate: {metrics.collision_rate:.1f} (target: {metrics.target_collision_rate:.1f})")
        print(f"Timeout rate: {metrics.timeout_rate:.1f}")

        if metrics.mean_energy  0:
            print(f"Energy consumption: {metrics.mean_energy:.0f}{metrics.std_energy:.0f} J "
                  f"(target: {metrics.target_energy_mean:.0f} J)")

        if metrics.mean_time  0:
            print(f"Flight time: {metrics.mean_time:.0f}{metrics.std_time:.0f} s "
                  f"(target: {metrics.target_time_mean:.0f} s)")

        if metrics.mean_ate  0:
            print(f"ATE error: {metrics.mean_ate:.3f}{metrics.std_ate:.3f} m "
                  f"(target: {metrics.target_ate_mean:.3f} m)")

        print("\nPerformance vs Table 3 targets:")
        print(f"  Success rate: {'' if metrics.success_rate = metrics.target_success_rate else ''}")
        print(f"  Collision rate: {'' if metrics.collision_rate = metrics.target_collision_rate else ''}")

        if metrics.mean_energy  0:
            energy_diff = metrics.mean_energy - metrics.target_energy_mean
            print(f"  Energy: {'' if abs(energy_diff) = 50 else ''} "
                  f"({energy_diff:+.0f} J difference)")

    def save_results(self, filepath: str, metrics: EvaluationMetrics):

        import json

        results_dict = {
            'baseline': 'A_PID',
            'episodes_completed': metrics.episodes_completed,
            'success_rate_percent': metrics.success_rate,
            'collision_rate_percent': metrics.collision_rate,
            'timeout_rate_percent': metrics.timeout_rate,
            'energy_mean_joules': metrics.mean_energy,
            'energy_std_joules': metrics.std_energy,
            'time_mean_seconds': metrics.mean_time,
            'time_std_seconds': metrics.std_time,
            'ate_mean_meters': metrics.mean_ate,
            'ate_std_meters': metrics.std_ate,
            'episode_results': [
                {
                    'success': r.success,
                    'collision': r.collision,
                    'timeout': r.timeout,
                    'energy': r.energy_consumed,
                    'time': r.flight_time,
                    'path_length': r.path_length,
                    'ate': r.ate_error,
                    'final_distance': r.final_distance_to_goal
                }
                for r in self.episode_results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {filepath}")
