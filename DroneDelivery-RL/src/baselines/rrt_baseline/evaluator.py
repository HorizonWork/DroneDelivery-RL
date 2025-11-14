import numpy as np
from typing import Dict, List, Any, Tuple
import time
from dataclasses import dataclass
import json

from ..astar_baseline.evaluator import EpisodeResult, EvaluationMetrics

class RRTEvaluator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.episode_results: List[EpisodeResult] = []

        self.num_episodes = config.get('num_episodes', 200)
        self.max_episode_time = config.get('max_episode_time', 300.0)
        self.goal_tolerance = config.get('goal_tolerance', 0.5)

        self.control_dt = config.get('control_dt', 0.05)
        self.drone_mass = config.get('drone_mass', 1.5)

        self.replan_threshold = config.get('replan_threshold', 2.0)
        self.replan_frequency = config.get('replan_frequency', 5.0)

    def evaluate_episode(self, rrt_controller, pid_controller, environment) - EpisodeResult:

        result = EpisodeResult()

        obs = environment.reset()
        pid_controller.reset()

        start_time = time.time()
        last_replan_time = start_time
        total_energy = 0.0
        path_length = 0.0
        previous_pos = None
        planning_time = 0.0

        current_pos = environment.get_drone_position()
        goal_pos = environment.get_goal_position()

        plan_start = time.time()
        obstacles = environment.get_obstacles()
        rrt_controller.update_obstacles(obstacles)
        path = rrt_controller.plan_path(current_pos, goal_pos)
        planning_time += time.time() - plan_start

        if not path:
            result.success = False
            result.timeout = True
            return result

        rrt_controller.set_path(path)

        step = 0
        max_steps = int(self.max_episode_time / self.control_dt)

        while step  max_steps:
            current_time = time.time()

            current_pos = environment.get_drone_position()
            current_yaw = environment.get_drone_yaw()

            distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(goal_pos))
            if distance_to_goal  self.goal_tolerance:
                result.success = True
                break

            if environment.check_collision():
                result.collision = True
                break

            if (current_time - last_replan_time  self.replan_frequency):
                obstacles = environment.get_obstacles()

                needs_replan = False
                for i in range(rrt_controller.path_index, len(rrt_controller.current_path)):
                    waypoint = rrt_controller.current_path[i]
                    for obs in obstacles:
                        if np.linalg.norm(np.array(waypoint) - np.array(obs))  self.replan_threshold:
                            needs_replan = True
                            break
                    if needs_replan:
                        break

                if needs_replan:
                    plan_start = time.time()
                    rrt_controller.update_obstacles(obstacles)
                    new_path = rrt_controller.plan_path(current_pos, goal_pos)
                    planning_time += time.time() - plan_start

                    if new_path:
                        rrt_controller.set_path(new_path)

                last_replan_time = current_time

            waypoint = rrt_controller.get_next_waypoint(current_pos)
            if waypoint is None:
                plan_start = time.time()
                obstacles = environment.get_obstacles()
                rrt_controller.update_obstacles(obstacles)
                emergency_path = rrt_controller.plan_path(current_pos, goal_pos)
                planning_time += time.time() - plan_start

                if emergency_path:
                    rrt_controller.set_path(emergency_path)
                    waypoint = rrt_controller.get_next_waypoint(current_pos)
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

        result.planning_time = planning_time
        result.num_replans = int((result.flight_time) / self.replan_frequency)

        if step = max_steps and not result.success and not result.collision:
            result.timeout = True

        return result

    def evaluate_multiple_episodes(self, rrt_controller, pid_controller,
                                 environment, num_episodes: int = None) - EvaluationMetrics:

        if num_episodes is None:
            num_episodes = self.num_episodes

        self.episode_results = []

        print(f"Evaluating RRT + PID baseline over {num_episodes} episodes...")

        for episode in range(num_episodes):
            if episode  50 == 0:
                print(f"Episode {episode}/{num_episodes}")

            result = self.evaluate_episode(rrt_controller, pid_controller, environment)
            self.episode_results.append(result)

        return self._compute_metrics()

    def _compute_metrics(self) - EvaluationMetrics:

        if not self.episode_results:
            return EvaluationMetrics()

        metrics = EvaluationMetrics()
        metrics.episodes_completed = len(self.episode_results)

        metrics.target_success_rate = 88.0
        metrics.target_energy_mean = 950.0
        metrics.target_time_mean = 45.0
        metrics.target_collision_rate = 2.1
        metrics.target_ate_mean = 0.13

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
        print("RRT + PID BASELINE EVALUATION RESULTS")
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

        print("\nPerformance vs expected RRT targets:")
        print(f"  Success rate: {'' if metrics.success_rate = metrics.target_success_rate else ''}")
        print(f"  Collision rate: {'' if metrics.collision_rate = metrics.target_collision_rate else ''}")

        if metrics.mean_energy  0:
            energy_diff = metrics.mean_energy - metrics.target_energy_mean
            print(f"  Energy: {'' if abs(energy_diff) = 100 else ''} "
                  f"({energy_diff:+.0f} J difference)")
